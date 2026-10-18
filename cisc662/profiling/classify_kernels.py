"""
Classify GPU kernels from an `nsys stats --report cuda_gpu_kern_sum` CSV into the proposal's target classes
(gemm, attention, rmsnorm, rope, softmax_ce) plus the rest of the step (optimizer, cast, embedding,
elementwise, other), and report per-step time per class.

Kernel names differ by mode:
  eager   -> library kernels (cuBLAS nvjet_*, FA3 flash::*, at::native elementwise/reduction functors)
  compile -> Inductor Triton kernels named after the fused aten ops (triton_{poi,red,per}_fused_<ops>_<n>)
A compiled kernel can fuse several classes (e.g. RoPE + QK-norm); such kernels get a combined label like
"rope+rmsnorm" instead of being forced into one class.

Usage:
  python -m cisc662.profiling.classify_kernels <kern_sum.csv> [--steps 5] [--top 25] [--json out.json]
"""
import argparse
import csv
import json
import re

# Order matters: first match wins for library kernels. Project kernels (c662_<kernel_id>_*) come first.
LIB_RULES = [
    ("rmsnorm", r"c662_(rmsnorm|mixnorm)"),
    ("rope", r"c662_qkrope"),
    ("softmax_ce", r"c662_ce"),
    ("relu2_mlp", r"c662_mlp"),
    ("gemm", r"c662_gemm"),
    ("attention", r"c662_attention"),
    ("optimizer", r"c662_muon"),
    ("attention", r"flash|fmha|FlashAttn|flash_fwd|flash_bwd|attention|_sdpa_|cudnn.*sdpa"),
    ("gemm", r"nvjet|gemm|xmma|cutlass|cublas|sm90_|sm80_|Kernel2|splitKreduce|triton_tem_"),
    ("softmax_ce", r"SoftMax|softmax|log_softmax|nll_loss|cross_entropy|logsumexp|tanh"),
    ("rmsnorm", r"rms_norm|RMSNorm|rmsnorm|layer_norm|rsqrt"),
    ("embedding", r"indexSelect|index_select|embedding|Embedding|index_put|indexFuncLarge|scatter_add|_unique"),
    ("rope", r"CatArray|cat_|\bcat\b"),
    ("cast", r"direct_copy|_to_copy|copy_kernel"),
    ("optimizer", r"lerp|_foreach|addcdiv|adam|muon|sqrt"),
]

# Fused-op tokens inside Inductor kernel names -> class contributions.
INDUCTOR_TOKENS = {
    "softmax_ce": ["_log_softmax", "nll_loss", "tanh", "logsumexp", "_softmax", "exp"],
    "rmsnorm": ["rsqrt", "mean", "pow"],
    "rope": ["cat", "neg"],
    "embedding": ["embedding", "index"],
    "optimizer": ["lerp", "addcdiv", "sqrt", "sign"],
}


def classify(name):
    if name.startswith("triton_") and "_fused_" in name and not name.startswith("triton_tem_"):
        ops = name.split("_fused_", 1)[1]
        hits = [cls for cls, toks in INDUCTOR_TOKENS.items() if any(t in ops for t in toks)]
        # rmsnorm tokens (mean/pow/rsqrt) also appear in Muon's norm and in loss means; keep the combined label
        if hits:
            return "+".join(sorted(set(hits)))
        if "_to_copy" in ops:
            return "cast"
        return "elementwise"
    for cls, pat in LIB_RULES:
        if re.search(pat, name):
            return cls
    if re.search(r"elementwise|reduce_kernel|vectorized|unrolled", name):
        return "elementwise"
    return "other"


def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(name=r["Name"], total_ns=float(r["Total Time (ns)"]), instances=int(r["Instances"]),
                             avg_ns=float(r["Avg (ns)"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--steps", type=int, default=5, help="number of training steps captured")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = load(args.csv)
    total = sum(r["total_ns"] for r in rows)
    by_class = {}
    for r in rows:
        r["class"] = classify(r["name"])
        c = by_class.setdefault(r["class"], dict(ns=0.0, kernels=0, launches=0))
        c["ns"] += r["total_ns"]
        c["kernels"] += 1
        c["launches"] += r["instances"]

    print(f"{args.csv}: {len(rows)} distinct kernels, {sum(r['instances'] for r in rows)} launches, "
          f"{total / args.steps / 1e6:.2f} ms GPU kernel time per step")
    print(f"{'class':<28}{'ms/step':>10}{'share':>8}{'kernels':>9}{'launches/step':>15}")
    for cls, c in sorted(by_class.items(), key=lambda kv: -kv[1]["ns"]):
        print(f"{cls:<28}{c['ns'] / args.steps / 1e6:>10.3f}{100 * c['ns'] / total:>7.1f}%{c['kernels']:>9}"
              f"{c['launches'] / args.steps:>15.1f}")
    print(f"\nTop {args.top} kernels:")
    for r in sorted(rows, key=lambda r: -r["total_ns"])[:args.top]:
        print(f"  {100 * r['total_ns'] / total:5.1f}%  {r['total_ns'] / args.steps / 1e6:7.3f} ms/step  "
              f"x{r['instances'] // args.steps:<4} [{r['class']}] {r['name'][:110]}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(csv=args.csv, steps=args.steps, total_ms_per_step=total / args.steps / 1e6,
                           by_class={k: dict(v, ms_per_step=v["ns"] / args.steps / 1e6, share=v["ns"] / total)
                                     for k, v in by_class.items()},
                           kernels=sorted(rows, key=lambda r: -r["total_ns"])), f, indent=1)


if __name__ == "__main__":
    main()
