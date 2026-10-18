"""
Attribute every GPU kernel in an Nsight Systems capture to (phase, op, class) using the nsys SQLite export.

For each kernel: CUPTI launch record (correlationId) -> launching thread -> enclosing NVTX push/pop ranges.
  phase : forward / optimizer / dataloader from base_train.py's NVTX ranges (main thread); kernels launched from
          another thread are the autograd engine, i.e. backward.
  op    : innermost "aten::..." range when the capture used NANOCHAT_EMIT_NVTX=1 (eager), with its input shapes.
  class : one of CLASSES. With op ranges (eager) the class comes from (op, shapes, phase) rules; without them
          (torch.compile) it comes from the kernel name (cuBLAS/FA3 names, Inductor fused-op tokens).

R12 shape coincidence: vocab V = 32768 = B*T, so (32768, 768) is ambiguous between activations and embedding /
lm_head parameters; the phase disambiguates (optimizer phase -> parameter update).

Usage:
  python -m cisc662.profiling.nsys_attrib <capture.sqlite> [--steps 5] [--json out.json] [--show rope,other]
"""
import argparse
import ast
import bisect
import json
import re
import sqlite3
from collections import defaultdict

PHASES = ("forward", "backward", "optimizer", "dataloader")
CLASSES = ("gemm", "attention", "rmsnorm", "rope", "softmax_ce", "relu2_mlp", "weight_cast", "embedding",
           "optimizer", "residual_elementwise", "other")


# --------------------------------------------------------------------------------------------------------------
# classification from (op, shapes) -- eager captures with emit_nvtx ranges

def parse_op_range(text):
    op = text.split(",", 1)[0].strip()
    sizes = []
    m = re.search(r"sizes = (\[.*?\]\])", text)
    if m:
        try:
            sizes = [tuple(s) for s in ast.literal_eval(m.group(1)) if isinstance(s, list)]
        except (ValueError, SyntaxError):
            pass
    return op, sizes


def classify_op(op, sizes, phase, d):
    B, T, C, H, D, F, V = d["B"], d["T"], d["C"], d["H"], d["D"], d["F"], d["V"]
    BT = B * T
    s = set(sizes)
    name = op.replace("aten::", "")
    if phase == "optimizer":
        return "optimizer"
    if "flash" in op or "attention" in name:
        return "attention"
    if name in ("mm", "bmm", "addmm", "matmul", "linear", "baddbmm"):
        return "gemm"
    if "rms_norm" in name:
        return "rmsnorm"
    if name in ("_log_softmax", "_log_softmax_backward_data", "nll_loss_forward", "nll_loss_backward",
                "tanh", "tanh_backward") or (BT, V) in s or (B, T, V) in s:
        return "softmax_ce"
    if (B, T, H, D // 2) in s or (1, T, 1, D // 2) in s:
        return "rope"
    if name.startswith("cub::") or "embedding" in name or name in ("index_select", "index_add_", "gather"):
        return "embedding"
    if name in ("relu", "threshold_backward", "clamp_min") or (name in ("pow", "mul") and (B, T, F) in s):
        return "relu2_mlp"
    # fp32 master weight -> bf16 cast in Linear.forward, and the grad cast back in backward: 2-D weight shapes
    if name in ("copy_", "_to_copy") and sizes and all(len(x) == 2 for x in sizes if x):
        return "weight_cast"
    if name in ("add", "add_", "mul", "div", "sub", "sum", "sigmoid", "sigmoid_backward", "cat", "slice_backward",
                "copy_", "mean", "fill_", "zero_", "neg", "pow", "where"):
        return "residual_elementwise"
    return "other"


# --------------------------------------------------------------------------------------------------------------
# classification from the kernel name -- compiled captures (no op ranges)

LIB_RULES = [
    # project kernels (cisc662 naming convention c662_<kernel_id>_<fwd|bwd>[_variant]) come first
    ("rmsnorm", r"c662_(rmsnorm|mixnorm)"),
    ("rope", r"c662_qkrope"),
    ("softmax_ce", r"c662_ce"),
    ("relu2_mlp", r"c662_mlp"),
    ("gemm", r"c662_gemm"),
    ("attention", r"c662_attention"),
    ("optimizer", r"c662_muon"),
    ("attention", r"flash|fmha|FlashAttn|attention|cudnn.*sdpa"),
    ("gemm", r"nvjet|gemm|xmma|cutlass|cublas|splitKreduce|triton_tem_"),
    ("softmax_ce", r"SoftMax|softmax|nll_loss|tanh"),
    ("rmsnorm", r"layer_norm|rms_norm"),
    ("embedding", r"indexSelect|embedding|index_put|indexFuncLarge|RadixSort|UniqueByKey|DeviceScan|compute_grad_weight"),
    ("rope", r"CatArray"),
    ("weight_cast", r"direct_copy|bfloat16_copy"),
]
# exact op tokens inside Inductor names: triton_{poi,red,per}_fused_<op>_<op>_..._<n>
TOKEN_CLASSES = [
    ("optimizer", {"lerp", "addcdiv", "sqrt", "linalg", "sign", "rsub"}),
    ("softmax_ce", {"softmax", "nll", "tanh", "logsumexp"}),
    ("relu2_mlp", {"relu", "threshold"}),
    ("rmsnorm", {"rms", "rsqrt"}),
    ("rope", {"cat", "neg"}),
    ("embedding", {"embedding"}),
]


def classify_kernel(name, phase):
    if phase == "optimizer":
        return "optimizer"
    if name.startswith("triton_") and "_fused_" in name and not name.startswith("triton_tem_"):
        toks = set(name.split("_fused_", 1)[1].split("_"))
        hits = [cls for cls, keys in TOKEN_CLASSES if toks & keys]
        if hits:
            # Inductor fuses across classes (e.g. RoPE + QK-norm + scale): keep the combination visible
            return "+".join(sorted(set(hits)))
        return "weight_cast" if {"to", "copy"} <= toks and len(toks) <= 5 else "residual_elementwise"
    for cls, pat in LIB_RULES:
        if re.search(pat, name):
            return cls
    return "residual_elementwise" if re.search(r"elementwise|reduce_kernel|Functor", name) else "other"


# --------------------------------------------------------------------------------------------------------------

def load(sqlite_path):
    c = sqlite3.connect(sqlite_path)
    strings = dict(c.execute("select id, value from StringIds"))
    kernels = c.execute("select start, end, correlationId, demangledName, shortName, registersPerThread, "
                        "gridX, gridY, gridZ, blockX, blockY, blockZ, staticSharedMemory, dynamicSharedMemory "
                        "from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    launches = {corr: (start, tid) for start, tid, corr in
                c.execute("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
    ranges = defaultdict(list)
    for start, end, text, text_id, tid in c.execute(
            "select start, end, text, textId, globalTid from NVTX_EVENTS where eventType = 59 and end is not null"):
        ranges[tid].append((start, end, text if text is not None else strings.get(text_id, "")))
    for tid in ranges:
        ranges[tid].sort()
    return strings, kernels, launches, ranges


def stacks_for_launches(ranges_sorted, launch_times):
    """For each launch time on one thread, the NVTX push/pop ranges containing it, outermost first.
    Sweep in time order with a stack; push/pop ranges on one thread are properly nested."""
    order = sorted(range(len(launch_times)), key=lambda i: launch_times[i])
    out = [None] * len(launch_times)
    stack, j = [], 0
    for i in order:
        t = launch_times[i]
        while j < len(ranges_sorted) and ranges_sorted[j][0] <= t:
            r = ranges_sorted[j]
            while stack and stack[-1][1] < r[0]:
                stack.pop()
            stack.append(r)
            j += 1
        while stack and stack[-1][1] < t:
            stack.pop()
        out[i] = [r[2] for r in stack]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--dims", default="B=16,T=2048,C=768,H=6,D=128,F=3072,V=32768")
    ap.add_argument("--json", default=None)
    ap.add_argument("--show", default="", help="comma-separated classes whose top ops/kernels to list")
    args = ap.parse_args()
    d = {k: int(v) for k, v in (kv.split("=") for kv in args.dims.split(","))}

    strings, kernels, launches, ranges = load(args.sqlite)
    main_tids = {tid for tid, rs in ranges.items() if any(r[2].startswith("step_") for r in rs)}

    # NVTX stack of every kernel launch, computed per launching thread
    by_tid = defaultdict(list)
    for ki, k in enumerate(kernels):
        launch = launches.get(k[2])
        if launch and launch[1] in ranges:
            by_tid[launch[1]].append((ki, launch[0]))
    kstack = {}
    for tid, items in by_tid.items():
        for (ki, _), st in zip(items, stacks_for_launches(ranges[tid], [t for _, t in items])):
            kstack[ki] = st

    agg = defaultdict(float)            # (phase, class) -> ns
    detail = defaultdict(lambda: defaultdict(float))
    regs = {}
    total = 0.0
    for ki, (k_start, k_end, corr, dname, sname, nreg, gx, gy, gz, bx, by, bz, ssm, dsm) in enumerate(kernels):
        ns = k_end - k_start
        total += ns
        kname = strings.get(dname) or strings.get(sname, "?")
        launch = launches.get(corr)
        tid = launch[1] if launch else None
        stack = kstack.get(ki, [])
        phase = next((p for p in reversed(stack) if p in PHASES), None)
        if phase is None:
            phase = "backward" if (tid is not None and tid not in main_tids) else "other"
        op_ranges = [r for r in stack if r.startswith(("aten::", "flash_attn", "cub::"))]
        if op_ranges:
            op, sizes = parse_op_range(op_ranges[-1])
            cls = classify_op(op, sizes, phase, d)
            key = f"{op} {sizes}"
        else:
            cls = classify_kernel(kname, phase)
            key = kname[:120]
        agg[(phase, cls)] += ns
        detail[cls][key] += ns
        regs.setdefault(kname[:120], dict(regs_per_thread=nreg, grid=(gx, gy, gz), block=(bx, by, bz),
                                          smem_static=ssm, smem_dynamic=dsm))

    per_class = defaultdict(float)
    per_phase = defaultdict(float)
    for (ph, cls), ns in agg.items():
        per_class[cls] += ns
        per_phase[ph] += ns
    S = args.steps
    print(f"{args.sqlite}\n{len(kernels)} kernels, {total / S / 1e6:.2f} ms GPU kernel time per step")
    print("by phase: " + ", ".join(f"{p} {v / S / 1e6:.2f} ms ({100 * v / total:.1f}%)"
                                   for p, v in sorted(per_phase.items(), key=lambda kv: -kv[1])))
    print(f"{'class':<34}{'ms/step':>9}{'share':>8}   by phase (ms/step)")
    for cls in sorted(per_class, key=lambda c: -per_class[c]):
        phs = ", ".join(f"{p}={agg[(p, cls)] / S / 1e6:.2f}" for p in ("forward", "backward", "optimizer", "other", "dataloader")
                        if agg.get((p, cls)))
        print(f"{cls:<34}{per_class[cls] / S / 1e6:>9.3f}{100 * per_class[cls] / total:>7.1f}%   {phs}")
    for cls in [c for c in args.show.split(",") if c]:
        print(f"\n-- {cls}: top ops/kernels --")
        for key, ns in sorted(detail[cls].items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {ns / S / 1e6:8.3f} ms/step  {key[:150]}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(sqlite=args.sqlite, steps=S, total_ms_per_step=total / S / 1e6,
                           by_phase={p: v / S / 1e6 for p, v in per_phase.items()},
                           by_class={c: dict(ms_per_step=v / S / 1e6, share=v / total) for c, v in per_class.items()},
                           by_phase_class={f"{p}|{c}": v / S / 1e6 for (p, c), v in agg.items()},
                           top={c: sorted(((k, v / S / 1e6) for k, v in dd.items()), key=lambda kv: -kv[1])[:30]
                                for c, dd in detail.items()},
                           kernel_resources=regs), f, indent=1)


if __name__ == "__main__":
    main()
