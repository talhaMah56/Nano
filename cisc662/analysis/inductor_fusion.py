"""
BAS-1 (decision D1): what torch.compile already fuses in R12, and how many DRAM bytes each fused region moves.

Inputs (written by slurm/s2_inductor_dump.sbatch into results/runs/s2_inductor_<jobid>/):
  output_code/graph_NNN_output_code.py   Inductor-generated module per compiled graph (forward, backward, one per
                                         optimizer step variant); or output_code/output_code.log (TORCH_LOGS fallback)
  ncu_step20_raw.csv                     `ncu --import X.ncu-rep --csv --page raw --print-units base` of ONE whole
                                         compiled step (all kernels); fallback ncu_step20_tierA_raw.csv (Triton only)
Every ncu launch is mapped to a region:
  Triton kernels  -> the graph that defines them (forward / backward / AdamW / Muon) + their fused aten ops,
                     source-node names and tensor shapes from the comment block above each kernel in output_code.py
  library kernels -> cisc662.profiling.classify_kernels.classify() + the phase (NVTX range in the ncu CSV when present:
                     kernels launched outside the main thread's ranges are the autograd thread = backward)
and compared with the algorithmic minimum bytes of the region (min_bytes_model(), formulas printed in the summary).

Outputs:
  results/inductor_fusion_table.csv                         (canonical copy; never overwrites another job's table)
  results/runs/s2_inductor_<jobid>/inductor_fusion_table.csv
  results/runs/s2_inductor_<jobid>/inductor_fusion_summary.md   per-region totals + draft vs-compile target lines
  results/runs/s2_inductor_<jobid>/inductor_fusion_regions.json (machine-readable, for roofline / slides)

Usage:
  python -m cisc662.analysis.inductor_fusion cisc662/results/runs/s2_inductor_<jobid> \
      [--attrib cisc662/results/runs/s1_nsys_3400686/attrib_compile_fa3.json] [--steps 1] [--target-gbps 3000]
  python -m cisc662.analysis.inductor_fusion --selftest      # synthetic inputs in the true ncu/Inductor formats
"""
import argparse
import csv
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from cisc662.profiling.classify_kernels import classify as classify_lib_kernel
from cisc662.profiling.classify_kernels import load as load_kern_sum

PKG_DIR = Path(__file__).resolve().parents[1]           # .../nanochat/cisc662
REPO_DIR = PKG_DIR.parent
CANONICAL_TABLE = PKG_DIR / "results" / "inductor_fusion_table.csv"
R12_DIMS = "B=16,T=2048,C=768,H=6,D=128,F=3072,V=32768,L=12"
HBM_GBPS = 3650.0            # measured copy bandwidth (S1 smoke, job 3400653)
LAUNCH_FLOOR_US = 2.0        # smallest useful time per fused-kernel launch at these sizes
XCHECK_TOL = 0.30            # flag |ncu/nsys - 1| above this
XCHECK_MIN_US = 10.0         # ... for kernels with at least this much nsys time per step

TABLE_COLUMNS = ["region", "kernel_name", "launches_per_step", "us_per_step", "dram_bytes_per_step",
                 "min_bytes_per_step", "bytes_ratio", "achieved_GBps", "registers", "achieved_occupancy_pct",
                 "source_ops", "job_id"]
REGION_TOTAL = "__region_total__"

# (region, tier, description). Tier "A" regions get a vs-compile target line.
REGIONS = [
    ("mixnorm_fwd", "A", "residual add + lambda-mix + RMSNorm, forward"),
    ("mixnorm_bwd", "A", "residual add + lambda-mix + RMSNorm, backward (incl. lambda-grad reductions)"),
    ("qkrope_fwd", "A", "QK-norm + RoPE + 1.2 scale, forward"),
    ("qkrope_bwd", "A", "QK-norm + RoPE + 1.2 scale, backward (incl. the RoPE half-slice scatter)"),
    ("relu2_fwd", "A", "relu^2, forward"),
    ("relu2_bwd", "A", "relu^2, backward"),
    ("ce_fwd", "A", "softcap + cross-entropy, forward"),
    ("ce_bwd", "A", "softcap + cross-entropy, backward (dlogits)"),
    ("optim_adamw", "A", "AdamW fused step (lm_head, wte, value embeddings, scalars)"),
    ("optim_muon", "A", "Muon pre/post-NS pointwise (momentum, MuonEq, Muon+, NorMuon, cautious update)"),
    ("optim_muon_ns", "-", "Muon Newton-Schulz pointwise + normalisation (foldable into bmm epilogues)"),
    ("optim_muon_stack", "-", "Muon stack/unstack copies outside the compiled step (avoidable)"),
    ("optim_muon_ns_gemm", "-", "Muon Newton-Schulz batched GEMMs"),
    ("optim_other", "-", "other kernels in the optimizer phase"),
    ("gemm_fwd", "-", "cuBLAS GEMMs, forward"),
    ("gemm_bwd", "-", "cuBLAS GEMMs, backward"),
    ("attention_fwd", "-", "FA3 forward"),
    ("attention_bwd", "-", "FA3 backward (preprocess, main, dQ postprocess)"),
    ("weight_cast_fwd", "-", "fp32 -> bf16 weight casts"),
    ("weight_cast_bwd", "-", "bf16 -> fp32 weight-gradient casts"),
    ("embedding_ve_fwd", "-", "token / value-embedding gathers (+ fused norm or VE gate mix)"),
    ("embedding_ve_bwd", "-", "embedding backward"),
    ("other_elementwise_fwd", "-", "other fused elementwise (smear, ...), forward"),
    ("other_elementwise_bwd", "-", "other fused elementwise (smear, ...), backward"),
    ("other", "-", "unclassified"),
]
REGION_ORDER = {r: i for i, (r, _, _) in enumerate(REGIONS)}
REGION_TIER = {r: t for r, t, _ in REGIONS}
REGION_DESC = {r: d for r, _, d in REGIONS}
PHASES = ("fwd", "bwd", "opt")

# aten ops that only change views (dropped from source_ops)
VIEW_OPS = {"view", "_unsafe_view", "reshape", "unsqueeze", "squeeze", "expand", "permute", "alias"}
CE_OPS = {"tanh", "tanh_backward", "_log_softmax", "_log_softmax_backward_data", "prepare_softmax_online",
          "logsumexp", "_softmax"}
EMB_OPS = {"embedding", "embedding_dense_backward"}
RELU_OPS = {"relu", "threshold_backward"}
RMS_OPS = {"_fused_rms_norm", "_fused_rms_norm_backward", "rsqrt"}
QK_SRC_RE = re.compile(r"^(q|k|cos|sin|x1|x2|y1|y2)(_\d+)?$")    # names in nanochat apply_rotary_emb / QK-norm
MUON_PREPOST_OPS = {"lerp", "clamp_min", "_to_copy", "copy_", "copy", "reciprocal", "sqrt", "rsub", "ge", "mean"}


# ----------------------------------------------------------------------------------------------------------------
# ncu raw CSV

UNIT_SCALE = {  # -> bytes / ns / plain
    "byte": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9, "tbyte": 1e12,
    "kibyte": 1024, "mibyte": 1024 ** 2, "gibyte": 1024 ** 3,
    "ns": 1, "nsecond": 1, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6, "s": 1e9, "second": 1e9,
}
NCU_METRICS = dict(time_ns="gpu__time_duration.sum", dram_read="dram__bytes_read.sum",
                   dram_write="dram__bytes_write.sum", regs="launch__registers_per_thread",
                   occ="sm__warps_active.avg.pct_of_peak_sustained_active", grid="launch__grid_size",
                   block="launch__block_size")


def parse_number(s):
    """ncu numbers: '1,420,832' / '93.89' / 'n/a' / ''. Returns float or None."""
    s = (s or "").strip().replace(",", "")
    if s in ("", "n/a", "N/A", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def nvtx_phase(field):
    """Phase from ncu's 'thread Domain:Push/Pop_Range...' column: '<pid>  "<default domain>:step_20:..."
    "<default domain>:forward:..."'. Returns fwd/bwd/opt; 'none' if the launch had no range (base_train's ranges are
    on the main thread, so that is the autograd thread = backward); 'other' for a range without a phase name."""
    if field is None:
        return None
    if not field.strip():
        return "none"
    for key, ph in ((":forward:", "fwd"), (":backward:", "bwd"), (":optimizer:", "opt")):
        if key in field:
            return ph
    return "other"


NVTX_STEP_RE = re.compile(r":step_(\d+):")


def _nvtx_step(field):
    """Training step of a main-thread launch from its 'step_<n>' NVTX range (None on the autograd thread)."""
    m = NVTX_STEP_RE.search(field or "")
    return int(m.group(1)) if m else None


def load_ncu_raw(path):
    """One dict per kernel launch, sorted by launch ID; byte metrics in bytes, time in ns."""
    with open(path, newline="", errors="replace") as f:
        lines = f.read().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        raise ValueError(f"{path}: no ncu CSV header line starting with \"ID\" (failed import?)")
    rows = list(csv.reader(lines[start:]))
    header, body = rows[0], [r for r in rows[1:] if r]
    col = {c: i for i, c in enumerate(header)}
    for need in ("ID", "Kernel Name", NCU_METRICS["time_ns"]):
        if need not in col:
            raise ValueError(f"{path}: column {need!r} missing; header has {len(header)} columns")
    # a units row (ID field empty) follows the header in ncu --csv --page raw output, with or without base units
    units = {}
    if body and not body[0][col["ID"]].strip():
        units = {c: body[0][i].strip().lower() for c, i in col.items() if i < len(body[0])}
        body = body[1:]
    for key in ("time_ns", "dram_read", "dram_write"):   # an unknown unit would silently be read as bytes / ns
        u = units.get(NCU_METRICS[key], "")
        if u and u not in UNIT_SCALE:
            raise ValueError(f"{path}: unknown unit {u!r} for {NCU_METRICS[key]} (re-import with --print-units base)")
    nvtx_col = next((c for c in header if c.startswith("thread Domain:Push/Pop_Range")), None)
    out = []
    for r in body:
        if len(r) < len(header) or parse_number(r[col["ID"]]) is None:
            continue
        rec = dict(id=int(parse_number(r[col["ID"]])), name=r[col["Kernel Name"]],
                   nvtx=nvtx_phase(r[col[nvtx_col]]) if nvtx_col else None,
                   nvtx_step=_nvtx_step(r[col[nvtx_col]]) if nvtx_col else None)
        for key, metric in NCU_METRICS.items():
            v = parse_number(r[col[metric]]) if metric in col else None
            if v is not None:
                v *= UNIT_SCALE.get(units.get(metric, ""), 1)
            rec[key] = v
        out.append(rec)
    out.sort(key=lambda d: d["id"])
    return out


# ----------------------------------------------------------------------------------------------------------------
# Inductor output_code.py

KERNEL_DEF_RE = re.compile(r"^(triton_\w+) = async_compile\.triton\(", re.M)
SRC_RE = re.compile(r"^# Topologically Sorted Source Nodes: \[(.*?)\], Original ATen: \[(.*?)\]\s*$", re.M)
SHAPE_RE = re.compile(r'Tensor "\w+\[([0-9, ]*)\]')
RUN_RE = re.compile(r"^\s+(triton_\w+)\.run\(", re.M)
AOT_RE = re.compile(r"^# AOT ID: \['(\w+)'\]", re.M)


def _split_list(s):
    return [t.strip() for t in s.split(",") if t.strip()]


def parse_module(text, label):
    """One Inductor module (the text of one output_code.py) -> graph dict."""
    m = AOT_RE.search(text)
    aot = m.group(1) if m else "unknown"
    kind = next((k for k in ("forward", "backward", "inference") if aot.endswith(k)), "unknown")
    kernels = {}
    for chunk in re.split(r"\n# kernel path: ", text)[1:]:
        d = KERNEL_DEF_RE.search(chunk)
        if not d:
            continue
        s = SRC_RE.search(chunk[:d.start()])
        frag = chunk[:d.start()]
        aten = [a.split(".", 1)[1] if a.startswith(("aten.", "prims.", "inductor.")) else a
                for a in _split_list(s.group(2))] if s else []
        shapes = [tuple(int(x) for x in sh.split(",") if x.strip()) for sh in SHAPE_RE.findall(frag)]
        kernels[d.group(1)] = dict(aten=aten, src=_split_list(s.group(1)) if s else [], shapes=shapes)
    for name in KERNEL_DEF_RE.findall(text):          # kernels without a '# kernel path' comment block
        kernels.setdefault(name, dict(aten=[], src=[], shapes=[]))
    runs = Counter(RUN_RE.findall(text))
    n_bmm = len(re.findall(r"extern_kernels\.(?:bmm|baddbmm)\(", text))
    g = dict(label=label, aot=aot, kind=kind, kernels=kernels, runs=dict(runs), n_bmm=n_bmm,
             n_mm=len(re.findall(r"extern_kernels\.(?:mm|addmm)\(", text)))
    g["role"] = graph_role(g)
    return g


def graph_role(g):
    """forward -> fwd, backward -> bwd; inference graphs in a training step are the compiled optimizer steps."""
    if g["kind"] == "forward":
        return "fwd"
    if g["kind"] == "backward":
        return "bwd"
    ops = {a for k in g["kernels"].values() for a in k["aten"]}
    if "linalg_vector_norm" in ops or g["n_bmm"]:
        return "muon"
    if "lerp" in ops:
        return "adamw"
    return "inference"


def load_output_code(run_dir):
    """All graphs under <run_dir>/output_code (graph_NNN_output_code.py, or output_code.log split at '# AOT ID')."""
    d = Path(run_dir) / "output_code"
    files = sorted(d.glob("*output_code.py")) + sorted(d.glob("*.log")) if d.is_dir() else []
    graphs = []
    for f in files:
        text = f.read_text(errors="replace")
        starts = [m.start() for m in AOT_RE.finditer(text)] or [0]
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else len(text)
            g = parse_module(text[s:e], f"{f.name}" + (f"#{i}" if len(starts) > 1 else ""))
            if g["kernels"]:
                graphs.append(g)
    return graphs


def role_phase(role):
    return {"fwd": "fwd", "bwd": "bwd", "adamw": "opt", "muon": "opt"}.get(role)


# ----------------------------------------------------------------------------------------------------------------
# region mapping

def name_tokens(name):
    """Fallback op list from an Inductor kernel name: triton_{poi,red,per}_fused_<op>_<op>_..._<n>."""
    body = name.split("_fused_", 1)[1] if "_fused_" in name else name
    toks = [t for t in body.split("_") if t and not t.isdigit()]
    joined = "_".join(toks)
    ops = set(toks)
    for multi in ("_fused_rms_norm_backward", "_fused_rms_norm", "_log_softmax_backward_data", "_log_softmax",
                  "embedding_dense_backward", "threshold_backward", "tanh_backward", "linalg_vector_norm",
                  "slice_backward", "select_backward", "prepare_softmax_online", "clamp_min", "nll_loss_forward",
                  "nll_loss_backward", "to_copy"):
        if multi.strip("_") in joined:
            ops.add(multi if multi != "to_copy" else "_to_copy")
    return ops


def classify_triton(info, phase, role, dims):
    """Region of one Triton kernel. info: aten ops / source nodes / fragment shapes (empty if not in output_code)."""
    ops = set(info["aten"]) if info and info["aten"] else name_tokens(info["name"])
    src = info.get("src", []) if info else []
    shapes = info.get("shapes", []) if info else []
    if role == "adamw":
        return "optim_adamw"
    if role == "muon" or phase == "opt":
        if role != "muon" and ops & {"sqrt"} and "linalg_vector_norm" not in ops:
            return "optim_adamw"
        return "optim_muon" if ops & MUON_PREPOST_OPS else "optim_muon_ns"
    ph = "bwd" if phase == "bwd" else "fwd"
    if ops & CE_OPS:
        return f"ce_{ph}"
    if ops & EMB_OPS:
        return f"embedding_ve_{ph}"
    half_head = any(len(s) == 4 and s[-1] == dims["D"] // 2 and s[-2] == dims["H"] for s in shapes)
    if any(QK_SRC_RE.match(s) for s in src) or half_head:
        return f"qkrope_{ph}"
    if not src and not shapes and "neg" in ops and ("cat" in ops or "slice" in ops) and ops & RMS_OPS:
        return f"qkrope_{ph}"               # name-only fallback (no output_code for this kernel)
    if ops & RELU_OPS:
        return f"relu2_{ph}"
    if ops & RMS_OPS or {"mean", "pow"} <= ops:
        return f"mixnorm_{ph}"
    if ph == "bwd" and ("sum" in ops or "select_backward" in ops) and ops <= {"mul", "sum", "add", "view",
                                                                            "_unsafe_view", "select_backward",
                                                                            "_to_copy"}:
        return "mixnorm_bwd"                # resid/x0 lambda gradients: sum(g * x), sum(g * x0)
    if "_to_copy" in ops and ops <= {"_to_copy", "t", "view", "_unsafe_view", "clone", "to", "copy"}:
        return f"weight_cast_{ph}"
    return f"other_elementwise_{ph}"


def classify_library(name, phase):
    cls = classify_lib_kernel(name)
    if phase == "opt":
        if cls == "gemm":
            return "optim_muon_ns_gemm"
        if re.search(r"CatArray|multi_tensor_apply|copy", name):
            return "optim_muon_stack"
        return "optim_other"
    ph = "bwd" if phase == "bwd" else "fwd"
    if cls == "attention":
        return f"attention_{ph}"
    if cls == "gemm":
        return f"gemm_{ph}"
    if cls == "embedding":
        return f"embedding_ve_{ph}"
    if cls in ("elementwise", "cast"):
        return f"other_elementwise_{ph}"
    return "other"


def source_ops_str(info):
    if info is None:
        return ""
    return "+".join(sorted({a for a in info["aten"] if a not in VIEW_OPS}))


def library_source(name):
    if re.search(r"flash|FlashAttn", name):
        return "extern:FA3"
    if re.search(r"nvjet|cublas|cutlass|gemm|splitKreduce", name):
        return "extern:cuBLAS"
    if "at::native" in name:
        return "extern:aten"
    return "extern"


def assign_regions(launches, graphs, dims):
    """Adds region / phase / graph / source_ops to every launch (in launch order).

    Phase: NVTX range when the CSV has it (no range on a backward-thread launch -> bwd), else the role of the graph
    that defines the Triton kernel, else a monotonic fwd -> bwd -> opt state machine driven by the launch sequence.
    A kernel name defined in several graphs goes to the graph whose role matches the current phase."""
    index = defaultdict(list)
    for gi, g in enumerate(graphs):
        for name in g["kernels"]:
            index[name].append(gi)
    have_nvtx = any(l.get("nvtx") in PHASES for l in launches)
    state = 0
    for l in launches:
        name = l["name"]
        nv = l.get("nvtx")
        nv_phase = None
        if have_nvtx:
            nv_phase = nv if nv in PHASES else ("bwd" if nv == "none" else None)
        cands = index.get(name, [])
        gi = None
        if cands:
            want = nv_phase or PHASES[state]
            match = [c for c in cands if role_phase(graphs[c]["role"]) == want]
            later = [c for c in cands if PHASES.index(role_phase(graphs[c]["role"]) or "fwd") >= state]
            gi = (match or later or cands)[0]
        role = graphs[gi]["role"] if gi is not None else None
        if nv_phase:
            phase = nv_phase
        elif role_phase(role):
            phase = role_phase(role)
        else:                                   # library kernel or Triton kernel missing from output_code
            phase = PHASES[state]
            if re.search(r"FlashAttnBwd|BwdPreprocess|BwdPostprocess|_backward", name):
                phase = PHASES[max(state, 1)]
            if re.search(r"CatArray|multi_tensor_apply|lerp|linalg_vector_norm", name):
                phase = "opt"
        state = max(state, PHASES.index(phase))
        l["phase"], l["graph"] = phase, graphs[gi]["label"] if gi is not None else ""
        if name.startswith("triton_") and not name.startswith("triton_tem_"):
            info = dict(graphs[gi]["kernels"][name], name=name) if gi is not None else dict(
                aten=[], src=[], shapes=[], name=name)
            l["region"] = classify_triton(info, phase, role, dims)
            l["source_ops"] = source_ops_str(info) if gi is not None else "(not in output_code) " + "+".join(
                sorted(name_tokens(name)))
            l["unmapped"] = gi is None
        else:
            l["region"] = classify_library(name, phase)
            l["source_ops"] = library_source(name)
            l["unmapped"] = False
    return launches


# ----------------------------------------------------------------------------------------------------------------
# algorithmic minimum bytes per region (per step)

def parse_dims(s):
    return {k: int(v) for k, v in (kv.split("=") for kv in s.split(","))}


def param_counts(d):
    """R12 parameter inventory (gpt.py / optim.py): AdamW gets lm_head (fp32), wte + value embeddings (bf16) and the
    scalars; Muon gets every 2-D matrix in the blocks (fp32 master weights)."""
    C, F, V, L, H = d["C"], d["F"], d["V"], d["L"], d["H"]
    n_ve = sum(1 for i in range(L) if i % 2 == (L - 1) % 2)
    adam_fp32 = V * C + (2 * L + 24 + 1 + 1)          # lm_head + resid/x0 lambdas, smear gate/lambda, backout
    adam_bf16 = V * C * (1 + n_ve)                     # wte + value embeddings
    muon = L * (4 * C * C + 2 * C * F) + n_ve * H * 12   # attn q,k,v,proj + mlp fc,proj + ve_gate
    return dict(n_ve=n_ve, adam_fp32=adam_fp32, adam_bf16=adam_bf16, muon=muon, adam_tensors=2 + n_ve + 5,
                muon_groups=4)


def min_bytes_model(d):
    """region -> dict(bytes, launches, formula). Minimum = a fused kernel at the same GEMM/FA3 boundaries as stock
    nanochat reads each distinct input once and writes each output once; activations the backward needs are the
    ones the forward writes anyway (or are recomputed from inputs it reads). A = one (N, C) bf16 activation."""
    N, C, H, D, F, V, L = d["B"] * d["T"], d["C"], d["H"], d["D"], d["F"], d["V"], d["L"]
    A = N * C * 2
    rstd, rstd_qk, cs = N * 4, 2 * N * H * 4, 2 * d["T"] * (D // 2) * 2
    p = param_counts(d)
    out = {}

    def sites(region, rows, extra, extra_txt, launches):
        total_a = sum(n * (r + w) for _, n, r, w in rows)
        txt = "; ".join(f"{desc}: {n} x ({r}R+{w}W)A" for desc, n, r, w in rows)
        out[region] = dict(bytes=total_a * A + extra, launches=launches,
                           formula=f"{total_a}A + {extra_txt} with A = N*C*2 = {A:,} B  [{txt}]")

    sites("mixnorm_fwd", [("layer 0 lambda-mix+norm (read x0; write a, h)", 1, 1, 2),
                          ("layers 1..L-1 MLP-residual add + lambda-mix + norm (read b, mlp_out, x0; write a, h)",
                           L - 1, 3, 2),
                          ("attn-residual add + norm (read a, attn_out; write b, h)", L, 2, 2),
                          ("final add - backout + norm (read b, mlp_out, x_backout; write h)", 1, 3, 1)],
          (2 * L + 1) * rstd, f"(2L+1)*N*4 rstd", 2 * L + 1)
    sites("mixnorm_bwd", [("layer 0 (read dh x3 q/k/v dX GEMMs, db, x0, dx0_acc; write dx0)", 1, 6, 1),
                          ("layers 1..L-1 (read dh x3, db, b, mlp_out, x0, dx0_acc; write dx, dx0_acc)", L - 1, 8, 2),
                          ("attn-residual norm bwd (read dh, dx, b; write db)", L, 3, 1),
                          ("final norm + backout bwd (read dh, h, x_backout; write dx)", 1, 3, 1),
                          ("backout grad added at layer L/2 (read dx)", 1, 1, 0)],
          (2 * L + 1) * rstd, f"(2L+1)*N*4 rstd", 2 * L + 1)
    out["qkrope_fwd"] = dict(bytes=L * (4 * A + rstd_qk + cs), launches=L,
                             formula=f"L*(4A + 2*N*H*4 rstd + cos/sin {cs:,} B): read q,k (GEMM outputs), "
                                     f"write rotated+normed+scaled q,k")
    out["qkrope_bwd"] = dict(bytes=L * (6 * A + rstd_qk + cs), launches=L,
                             formula="L*(6A + rstd + cos/sin): read dq',dk' (FA3 bwd) and q,k; write dq,dk")
    out["relu2_fwd"] = dict(bytes=L * 8 * A, launches=L, formula="L*8A: read h (N x F bf16 = 4A), write relu(h)^2 (4A)")
    out["relu2_bwd"] = dict(bytes=L * 12 * A, launches=L, formula="L*12A: read dy (4A) and h (4A), write dh (4A)")
    logits = N * V * 2
    out["ce_fwd"] = dict(bytes=logits + N * 8 + N * 4, launches=1,
                         formula=f"N*V*2 (read bf16 logits once, {logits:,} B) + N*8 targets + N*4 lse")
    out["ce_bwd"] = dict(bytes=2 * logits + N * 8 + N * 4, launches=1,
                         formula="2*N*V*2 (read logits, write bf16 dlogits) + targets + lse")
    out["optim_adamw"] = dict(bytes=28 * p["adam_fp32"] + 14 * p["adam_bf16"], launches=p["adam_tensors"],
                              formula=f"28 B/param x {p['adam_fp32']:,} fp32 params (lm_head, scalars: read p,g,m,v, "
                                      f"write p,m,v) + 14 B/param x {p['adam_bf16']:,} bf16 params (wte + "
                                      f"{p['n_ve']} value embeddings)")
    out["optim_muon"] = dict(bytes=24 * p["muon"], launches=2 * p["muon_groups"],
                             formula=f"24 B/param x {p['muon']:,} Muon params: pre-NS read g,m (fp32) write m (fp32) "
                                     f"+ X (bf16); post-NS read X (bf16), p write p (fp32); NS pointwise folded into "
                                     f"bmm epilogues and stack/unstack avoided (both 0); extra passes for the global "
                                     f"norms not counted")
    return out


# ----------------------------------------------------------------------------------------------------------------
# aggregation, nsys cross-check, targets

def aggregate(launches, steps):
    """(region, kernel) rows and region totals, per step."""
    per = defaultdict(lambda: dict(n=0, ns=0.0, bytes=0.0, occ_ns=0.0, occ_w=0.0, regs=Counter(), src="",
                                   missing_bytes=0))
    for l in launches:
        a = per[(l["region"], l["name"])]
        t = l["time_ns"] or 0.0
        a["n"] += 1
        a["ns"] += t
        if l["dram_read"] is None and l["dram_write"] is None:
            a["missing_bytes"] += 1
        a["bytes"] += (l["dram_read"] or 0.0) + (l["dram_write"] or 0.0)
        if l["occ"] is not None:
            a["occ_ns"] += l["occ"] * t
            a["occ_w"] += t
        if l["regs"] is not None:
            a["regs"][int(l["regs"])] += 1
        a["src"] = a["src"] or l["source_ops"]
    kern = []
    for (region, name), a in per.items():
        kern.append(dict(region=region, kernel_name=name, launches=a["n"] / steps, us=a["ns"] / steps / 1e3,
                         bytes=a["bytes"] / steps, regs=a["regs"].most_common(1)[0][0] if a["regs"] else None,
                         occ=a["occ_ns"] / a["occ_w"] if a["occ_w"] else None, source_ops=a["src"],
                         missing_bytes=a["missing_bytes"]))
    kern.sort(key=lambda k: (REGION_ORDER.get(k["region"], 99), -k["us"]))
    regions = {}
    for k in kern:
        r = regions.setdefault(k["region"], dict(launches=0.0, us=0.0, bytes=0.0, occ_us=0.0, kernels=0))
        r["launches"] += k["launches"]
        r["us"] += k["us"]
        r["bytes"] += k["bytes"]
        r["occ_us"] += (k["occ"] or 0.0) * k["us"]
        r["kernels"] += 1
    for r in regions.values():
        r["occ"] = r["occ_us"] / r["us"] if r["us"] else None
    return kern, regions


# nsys: <(int)2, (bool)1, (unsigned int)4, (cute::GMMA::Major)0>. One identifier after optional qualifiers, no
# nested quantifiers (a '(?:[\w:]+ ?)+' form backtracks catastrophically on long template names).
CAST_RE = re.compile(r"\((?:unsigned |signed )?[A-Za-z_][\w:]*\)(?=-?\d)")
NS_RE = re.compile(r"\b\w+::")


def nkey(name):
    """Join key between ncu and nsys kernel names. nsys prints template casts and full namespaces
    ('at::native::<unnamed>::CatArray...<(int)2>'), ncu drops both ('at::<unnamed>::CatArray...<2>'): strip
    'void ', casts before numbers, every namespace qualifier and spaces; keep 200 chars."""
    s = name[5:] if name.startswith("void ") else name
    return NS_RE.sub("", CAST_RE.sub("", s)).replace(" ", "")[:200]


def nsys_lookup(per, key):
    """Exact key, else a unique prefix match (the attrib JSON truncates names at 120 chars)."""
    if key in per:
        return per[key]
    hits = [v for k, v in per.items() if len(k) >= 60 and (key.startswith(k) or k.startswith(key))]
    return hits[0] if len(hits) == 1 else None


def load_trace_steps(trace_path, first_kernel):
    """S1 nsys cuda_gpu_trace CSV -> list of per-step {key: [us, launches]}, splitting the kernel sequence at each
    launch of the step's first kernel (the same kernel that opens the ncu step)."""
    steps, cur = [], None
    with open(trace_path, newline="") as f:
        for r in csv.DictReader(f):
            name = r.get("Name", "")
            if not name or name.startswith("[CUDA"):
                continue
            if nkey(name) == first_kernel:
                cur = defaultdict(lambda: [0.0, 0])
                steps.append(cur)
            if cur is None:
                continue
            e = cur[nkey(name)]
            e[0] += float(r["Duration (ns)"]) / 1e3
            e[1] += 1
    return steps


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])) if n else 0.0


def load_nsys(attrib_path, kern_sum_path=None, first_kernel=None):
    """Per-kernel nsys time (us/step) and launches/step from S1. Preference: per-step medians from the
    cuda_gpu_trace CSV next to the attrib JSON (robust to one slow step), else the complete cuda_gpu_kern_sum CSV
    (5-step mean), else the attrib JSON (top 30 kernels per class only)."""
    with open(attrib_path) as f:
        att = json.load(f)
    steps = att.get("steps", 5)
    stem = Path(attrib_path).stem.replace("attrib_", "", 1)
    if kern_sum_path is None:
        guess = Path(attrib_path).with_name(f"{stem}_cuda_gpu_kern_sum.csv")
        kern_sum_path = guess if guess.exists() else None
    trace = Path(attrib_path).with_name(f"{stem}_cuda_gpu_trace.csv")
    step_list = load_trace_steps(trace, nkey(first_kernel)) if (trace.exists() and first_kernel) else []
    per = defaultdict(lambda: dict(us=0.0, launches=None))
    if len(step_list) >= 2:
        for key in {k for st in step_list for k in st}:
            per[key]["us"] = _median([st[key][0] if key in st else 0.0 for st in step_list])
            per[key]["launches"] = _median([st[key][1] if key in st else 0 for st in step_list])
        step_totals = [sum(v[0] for v in st.values()) for st in step_list]
        return dict(per=dict(per), total_us=att.get("total_ms_per_step", 0.0) * 1e3, steps=steps,
                    source=f"{trace} (per-kernel median over {len(step_list)} steps)", attrib=str(attrib_path),
                    step_totals_us=step_totals, median_total_us=_median(step_totals))
    if kern_sum_path:
        for r in load_kern_sum(str(kern_sum_path)):
            e = per[nkey(r["name"])]
            e["us"] += r["total_ns"] / steps / 1e3
            e["launches"] = (e["launches"] or 0) + r["instances"] / steps
        source = str(kern_sum_path)
    else:
        for cls, items in att.get("top", {}).items():
            for name, ms in items:
                per[nkey(name)]["us"] += ms * 1e3
        source = f"{attrib_path} (top-30 per class only)"
    return dict(per=dict(per), total_us=att.get("total_ms_per_step", 0.0) * 1e3, steps=steps, source=source,
                attrib=str(attrib_path), step_totals_us=[], median_total_us=None)


def crosscheck(kern, nsys):
    """Kernel-level ncu vs nsys comparison (names summed over regions)."""
    ncu = defaultdict(lambda: dict(us=0.0, launches=0.0, name=""))
    for k in kern:
        e = ncu[nkey(k["kernel_name"])]
        e["us"] += k["us"]
        e["launches"] += k["launches"]
        e["name"] = e["name"] or k["kernel_name"]
    rows = []
    for key, e in ncu.items():
        s = nsys_lookup(nsys["per"], key)
        if s is None:
            rows.append(dict(name=e["name"], ncu_us=e["us"], nsys_us=None, ratio=None, ncu_launches=e["launches"],
                             nsys_launches=None, flag="not in nsys"))
            continue
        ratio = e["us"] / s["us"] if s["us"] else None
        flag = ""
        if ratio is not None and s["us"] >= XCHECK_MIN_US and abs(ratio - 1) > XCHECK_TOL:
            flag = f"time differs {100 * (ratio - 1):+.0f}%"
        if s["launches"] is not None and round(s["launches"]) != round(e["launches"]):
            flag = (flag + "; " if flag else "") + f"launches {e['launches']:g} vs {s['launches']:g}"
        rows.append(dict(name=e["name"], ncu_us=e["us"], nsys_us=s["us"], ratio=ratio, ncu_launches=e["launches"],
                         nsys_launches=s["launches"], flag=flag))
    matched = {nkey(r["name"]) for r in rows}
    missing = [(k, v["us"]) for k, v in nsys["per"].items() if v["us"] >= XCHECK_MIN_US and k not in matched
               and not any(m.startswith(k) or k.startswith(m) for m in matched if len(k) >= 60)]
    rows.sort(key=lambda r: -abs((r["ncu_us"] or 0) - (r["nsys_us"] or 0)))
    return rows, sorted(missing, key=lambda kv: -kv[1])


def region_nsys_us(kern, nsys):
    """nsys us/step per region; a kernel shared by several regions (e.g. one nvjet GEMM used by the model and by
    Muon) is split in proportion to its ncu time in each region (launch counts would ignore the shape difference)."""
    tot = defaultdict(float)
    for k in kern:
        tot[nkey(k["kernel_name"])] += k["us"]
    out = defaultdict(float)
    for k in kern:
        key = nkey(k["kernel_name"])
        s = nsys_lookup(nsys["per"], key)
        if s and tot[key]:
            out[k["region"]] += s["us"] * k["us"] / tot[key]
    return dict(out)


def target_line(label, us, dram, minb, launches_min, target_gbps):
    """Draft vs-compile target: a fused kernel moving the minimum bytes at 2.7 / target / 3.3 TB/s (never faster than
    LAUNCH_FLOOR_US per launch), against Inductor's measured time."""
    if not us or not minb:
        return f"- **{label}**: no ncu data", None
    bws = sorted({2700.0, target_gbps, 3300.0})
    t = {bw: max(minb / (bw * 1e3), launches_min * LAUNCH_FLOOR_US) for bw in bws}
    sp = {bw: us / t[bw] for bw in bws}
    eff = minb / us / 1e3
    ach = dram / us / 1e3 if dram else 0.0
    ratio = dram / minb if dram else float("nan")
    central = sp[target_gbps]
    if central < 1.15:
        verdict = (f"~parity: Inductor already runs at {eff:,.0f} GB/s effective on the minimum bytes "
                   f"({100 * eff / HBM_GBPS:.0f}% of {HBM_GBPS:.0f}); only removing bytes across the GEMM boundary "
                   f"(epilogue/prologue fusion) can help")
    elif central < 1.3:
        verdict = "marginal (< 1.3x, below the claim threshold)"
    else:
        why = []
        if ratio > 1.2:
            why.append(f"bytes: Inductor moves {ratio:.2f}x the minimum")
        if ach < 0.7 * HBM_GBPS:
            why.append(f"bandwidth: Inductor achieves {ach:.0f} GB/s ({100 * ach / HBM_GBPS:.0f}% of {HBM_GBPS:.0f})")
        verdict = "target >= 1.3x (" + "; ".join(why or ["launch count"]) + ")"
    lo, hi = sp[bws[0]], sp[bws[-1]]
    line = (f"- **{label}**: Inductor {us:,.0f} us/step, {dram / 1e9:.3f} GB DRAM (min {minb / 1e9:.3f} GB, "
            f"ratio {ratio:.2f}), {ach:,.0f} GB/s achieved, {eff:,.0f} GB/s effective on min bytes -> fused kernel "
            f"at {bws[0] / 1e3:.1f}-{bws[-1] / 1e3:.1f} TB/s on min bytes: {t[bws[-1]]:,.0f}-{t[bws[0]]:,.0f} us -> "
            f"**target vs compile {lo:.2f}-{hi:.2f}x** (central {central:.2f}x at {target_gbps / 1e3:.1f} TB/s). "
            f"{verdict}.")
    return line, dict(label=label, inductor_us=us, dram_bytes=dram, min_bytes=minb, bytes_ratio=ratio,
                      achieved_GBps=ach, effective_GBps=eff, target_us={str(int(b)): t[b] for b in bws},
                      speedup={str(int(b)): sp[b] for b in bws}, verdict=verdict)


TARGET_GROUPS = [  # (label, regions) -> target lines; combined lines sum the regions' time, bytes and minima
    ("residual/lambda-mix + RMSNorm fwd", ["mixnorm_fwd"]),
    ("residual/lambda-mix + RMSNorm bwd", ["mixnorm_bwd"]),
    ("residual/lambda-mix + RMSNorm fwd+bwd", ["mixnorm_fwd", "mixnorm_bwd"]),
    ("QK-norm+RoPE+scale fwd", ["qkrope_fwd"]),
    ("QK-norm+RoPE+scale bwd", ["qkrope_bwd"]),
    ("QK-norm+RoPE+scale fwd+bwd", ["qkrope_fwd", "qkrope_bwd"]),
    ("relu^2 fwd", ["relu2_fwd"]),
    ("relu^2 bwd", ["relu2_bwd"]),
    ("softcap+CE fwd", ["ce_fwd"]),
    ("softcap+CE bwd", ["ce_bwd"]),
    ("softcap+CE fwd+bwd (separate fwd and bwd kernels)", ["ce_fwd", "ce_bwd"]),
    ("AdamW", ["optim_adamw"]),
    ("Muon pre/post-NS pointwise", ["optim_muon"]),
    ("Muon all non-GEMM traffic (pre/post + NS pointwise + stack/unstack)",
     ["optim_muon", "optim_muon_ns", "optim_muon_stack"]),
]


GROUP_NOTES = {  # what the bandwidth line cannot show: bytes that only disappear across a GEMM boundary
    "relu^2 fwd": "Beyond the pointwise kernel: the c_fc epilogue (KT-7) avoids re-reading h, -4A/layer = -2.4 GB/step.",
    "relu^2 bwd": "Beyond the pointwise kernel: the c_proj dX epilogue (KT-7) never stores dy, -8A/layer = -4.8 GB/step.",
    "softcap+CE fwd+bwd (separate fwd and bwd kernels)":
        "Beyond this: chunked lm_head+CE (KT-4) never writes or re-reads the 2.15 GB bf16 logits.",
    "AdamW": "Parity expected; any gain must come from fewer launches (13 tensors) or bf16 state, not bandwidth.",
}


def build_targets(regions, minb, target_gbps, dims):
    lines, data = [], []
    for label, regs in TARGET_GROUPS:
        us = sum(regions.get(r, {}).get("us", 0.0) for r in regs)
        dram = sum(regions.get(r, {}).get("bytes", 0.0) for r in regs)
        mb = sum(minb[r]["bytes"] for r in regs if r in minb and r != "optim_muon_ns" and r != "optim_muon_stack")
        nl = sum(minb[r]["launches"] for r in regs if r in minb)
        line, d = target_line(label, us, dram, mb, nl, target_gbps)
        if label in GROUP_NOTES:
            line += " " + GROUP_NOTES[label]
        lines.append(line)
        if d:
            data.append(d)
    # CE with the gradient written in the forward (KT-3): one pass, read logits once + write dlogits once
    us = sum(regions.get(r, {}).get("us", 0.0) for r in ("ce_fwd", "ce_bwd"))
    dram = sum(regions.get(r, {}).get("bytes", 0.0) for r in ("ce_fwd", "ce_bwd"))
    N = dims["B"] * dims["T"]
    line, d = target_line("softcap+CE, gradient written in the forward (KT-3 single pass)", us, dram,
                          2 * N * dims["V"] * 2 + N * 12, 1, target_gbps)
    lines.append(line)
    if d:
        data.append(d)
    return lines, data


# ----------------------------------------------------------------------------------------------------------------
# outputs

def _fmt(v, nd=1):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def table_rows(kern, regions, minb, job_id):
    out = []
    for region in sorted(regions, key=lambda r: REGION_ORDER.get(r, 99)):
        ks = [k for k in kern if k["region"] == region]
        mb = minb.get(region, {}).get("bytes")
        for k in ks:
            sole = len(ks) == 1 and mb
            out.append(dict(region=region, kernel_name=k["kernel_name"], launches_per_step=_fmt(k["launches"], 2),
                            us_per_step=_fmt(k["us"], 2), dram_bytes_per_step=_fmt(k["bytes"], 0),
                            min_bytes_per_step=_fmt(float(mb), 0) if sole else "",
                            bytes_ratio=_fmt(k["bytes"] / mb, 3) if sole else "",
                            achieved_GBps=_fmt(k["bytes"] / k["us"] / 1e3 if k["us"] else None, 1),
                            registers=_fmt(k["regs"]), achieved_occupancy_pct=_fmt(k["occ"], 2),
                            source_ops=k["source_ops"], job_id=job_id))
        r = regions[region]
        out.append(dict(region=region, kernel_name=REGION_TOTAL, launches_per_step=_fmt(r["launches"], 2),
                        us_per_step=_fmt(r["us"], 2), dram_bytes_per_step=_fmt(r["bytes"], 0),
                        min_bytes_per_step=_fmt(float(mb), 0) if mb else "",
                        bytes_ratio=_fmt(r["bytes"] / mb, 3) if mb else "",
                        achieved_GBps=_fmt(r["bytes"] / r["us"] / 1e3 if r["us"] else None, 1), registers="",
                        achieved_occupancy_pct=_fmt(r["occ"], 2), source_ops=f"{r['kernels']} kernels",
                        job_id=job_id))
    return out


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def write_canonical(path, rows, job_id, force):
    """results/inductor_fusion_table.csv is a fixed name: refuse to replace a table from another job unless forced."""
    path = Path(path)
    if path.exists() and not force:
        with open(path, newline="") as f:
            old = next(csv.DictReader(f), {}).get("job_id")
        if old and old != str(job_id):
            print(f"!! {path} holds job {old}; not overwritten (pass --force-canonical to replace it). "
                  f"The per-job copy is in the run directory.")
            return False
    write_csv(path, rows)
    return True


def summary_md(ctx):
    regions, minb, nsys_r = ctx["regions"], ctx["minb"], ctx["nsys_region"]
    tot_us = sum(r["us"] for r in regions.values())
    tot_b = sum(r["bytes"] for r in regions.values())
    L = []
    L.append(f"# Inductor fusion facts (BAS-1, decision D1) — job {ctx['job_id']}\n")
    L.append(f"- ncu: `{ctx['ncu_csv']}` — {ctx['n_launches']} launches over {ctx['steps']} step(s); "
             f"{tot_us / 1e3:.2f} ms/step summed kernel time, {tot_b / 1e9:.2f} GB/step DRAM."
             + (f" NVTX steps in the capture: {ctx['nvtx_steps']}" + (
                 "." if len(ctx["nvtx_steps"]) == ctx["steps"] else
                 f" — **MISMATCH with --steps {ctx['steps']}: per-step numbers are wrong**.")
                if ctx.get("nvtx_steps") else ""))
    L.append(f"- Inductor graphs: {len(ctx['graphs'])} ({', '.join(ctx['graph_roles'])}); "
             f"{ctx['n_unmapped']} Triton launches not found in output_code (classified from the kernel name).")
    L.append(f"- Phase source: {ctx['phase_source']}. Dims: `{ctx['dims_str']}`.")
    if ctx.get("nsys"):
        n = ctx["nsys"]
        L.append(f"- nsys cross-check: `{n['source']}` — S1 {n['total_us'] / 1e3:.2f} ms/step GPU kernel time (5-step "
                 f"mean, `{Path(n['attrib']).name}`) vs ncu {tot_us / 1e3:.2f} ms/step "
                 f"({100 * (tot_us / n['total_us'] - 1):+.1f}%)." + (
                     f" S1 per-step totals: {', '.join(f'{t / 1e3:.2f}' for t in n['step_totals_us'])} ms "
                     f"(median {n['median_total_us'] / 1e3:.2f}; ncu {100 * (tot_us / n['median_total_us'] - 1):+.1f}% "
                     f"vs median)." if n.get("median_total_us") else ""))
    L.append("- Caveats: ncu serialises kernels and by default flushes caches before each kernel (`--cache-control "
             "all`), so bytes are cold-L2 traffic per kernel and small kernels look slower than in nsys; times are at "
             "default (unlocked) clocks (`--clock-control none`). ncu times are for byte/bandwidth facts only, never "
             "headline times (PLAN.md §7.1). achieved GB/s = ncu DRAM bytes / ncu time; effective GB/s = minimum "
             "bytes / ncu time. Writes still dirty in the 60 MiB L2 when a kernel ends are not counted as DRAM "
             "writes, so a streaming kernel can show up to ~50 MB per launch below the minimum (ratio < 1, effective "
             "GB/s above the roof).")
    if ctx["fallback"]:
        L.append("- **Fallback CSV (Tier-A Triton kernels only, 300-launch cap)**: library kernels (GEMM, FA3) are "
                 "absent and the step may be truncated — check the launches column against nsys.")
    L.append("\n## Per-region totals (per step)\n")
    L.append("| region | tier | kernels | launches | ncu µs | nsys µs (S1) | DRAM GB | min GB | ratio | achieved GB/s "
             "| % of 3650 | effective GB/s | occupancy % |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for region in sorted(regions, key=lambda r: REGION_ORDER.get(r, 99)):
        r = regions[region]
        mb = minb.get(region, {}).get("bytes")
        ach = r["bytes"] / r["us"] / 1e3 if r["us"] else 0.0
        L.append(f"| {region} | {REGION_TIER.get(region, '-')} | {r['kernels']} | {r['launches']:g} | {r['us']:,.1f} | "
                 f"{nsys_r.get(region, 0):,.1f} | {r['bytes'] / 1e9:.3f} | {mb / 1e9 if mb else 0:.3f} | "
                 f"{r['bytes'] / mb if mb else float('nan'):.2f} | {ach:,.0f} | {100 * ach / HBM_GBPS:.0f}% | "
                 f"{(mb / r['us'] / 1e3) if (mb and r['us']) else float('nan'):,.0f} | {_fmt(r['occ'], 1)} |")
    L.append(f"| **total** | | | {sum(r['launches'] for r in regions.values()):g} | {tot_us:,.1f} | "
             f"{sum(nsys_r.values()):,.1f} | {tot_b / 1e9:.3f} | | | | | | |")
    L.append("\n## Draft vs-compile targets (Tier-A regions)\n")
    L.append(f"Target logic (BAS-1 step 4): a fused kernel that moves only the minimum bytes at 2.7 / "
             f"{ctx['target_gbps'] / 1e3:.1f} / 3.3 TB/s (74-90% of the measured {HBM_GBPS:.0f} GB/s copy roof; at "
             f"least {LAUNCH_FLOOR_US:g} µs per launch) vs Inductor's measured ncu time. A claim needs >= 1.3x with "
             f"a bootstrap-CI lower bound >= 1.3 in kbench (PLAN.md §7.1) — these lines only set expectations.\n")
    L.extend(ctx["target_lines"])
    L.append("\n## Algorithmic minimum bytes per step (formulas)\n")
    L.append("Model: a fused kernel at the same GEMM / FA3 boundaries as stock nanochat reads each distinct input "
             "once and writes each output once; tensors the backward needs are ones the forward writes anyway (or "
             "recomputes from inputs it already reads). N = B*T tokens.\n")
    for region in sorted(minb, key=lambda r: REGION_ORDER.get(r, 99)):
        L.append(f"- `{region}` = {minb[region]['bytes'] / 1e9:.4f} GB: {minb[region]['formula']}")
    L.append("- Non-Tier-A regions (GEMM, FA3, casts, embeddings, Muon NS/stack) have no minimum here; Muon NS "
             "pointwise and stack/unstack count against the Muon 'all non-GEMM' line with minimum 0.")
    xc = ctx.get("xcheck")
    if xc is not None:
        flagged = [r for r in xc if r["flag"] and r["flag"] != "not in nsys"]
        L.append(f"\n## ncu vs nsys (S1) per kernel: {len(flagged)} flagged (> {100 * XCHECK_TOL:.0f}% time difference "
                 f"for kernels >= {XCHECK_MIN_US:g} µs/step, or different launch counts)\n")
        if flagged:
            L.append("| kernel | ncu µs | nsys µs | ncu/nsys | flag |")
            L.append("|---|---|---|---|---|")
            for r in flagged[:40]:
                L.append(f"| `{r['name'][:90]}` | {r['ncu_us']:,.1f} | {r['nsys_us']:,.1f} | "
                         f"{_fmt(r['ratio'], 2)} | {r['flag']} |")
        notin = [r for r in xc if r["flag"] == "not in nsys"]
        if notin:
            L.append(f"\nIn ncu but not in the S1 nsys capture ({len(notin)}; e.g. different Inductor kernel "
                     f"numbering): " + ", ".join(f"`{r['name'][:70]}` ({r['ncu_us']:.0f} µs)" for r in notin[:15]))
        if ctx.get("xmissing"):
            L.append(f"\nIn S1 nsys (>= {XCHECK_MIN_US:g} µs/step) but not in ncu: " +
                     ", ".join(f"`{k[:70]}` ({us:.0f} µs)" for k, us in ctx["xmissing"][:15]))
    if ctx["unmapped_names"]:
        L.append("\n## Triton kernels not found in output_code\n")
        L.extend(f"- `{n}`" for n in ctx["unmapped_names"])
    L.append("\n## Top kernels per Tier-A region\n")
    for region in [r for r, t, _ in REGIONS if t == "A" and r in regions]:
        ks = [k for k in ctx["kern"] if k["region"] == region][:4]
        L.append(f"- `{region}` ({REGION_DESC[region]}): " + "; ".join(
            f"`{k['kernel_name'][:80]}` x{k['launches']:g} {k['us']:,.0f} µs {k['bytes'] / 1e6:,.0f} MB "
            f"[{k['source_ops'][:80]}]" for k in ks))
    L.append(f"\nGenerated by `python -m cisc662.analysis.inductor_fusion {ctx['run_dir_arg']}`.")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------------------

def resolve(p):
    """Accept paths relative to the cwd, to cisc662/ or to the repo root."""
    if p is None:
        return None
    q = Path(p)
    for cand in (q, PKG_DIR / q, REPO_DIR / q):
        if cand.exists():
            return cand
    return q


def find_ncu_csv(run_dir, override):
    if override:
        return resolve(override), "tierA" in str(override)
    for name, fb in (("ncu_step20_raw.csv", False), ("ncu_step20_tierA_raw.csv", True)):
        p = Path(run_dir) / name
        if p.exists() and p.stat().st_size > 0:
            return p, fb
    raise FileNotFoundError(f"{run_dir}: neither ncu_step20_raw.csv nor ncu_step20_tierA_raw.csv")


def analyze(run_dir, attrib=None, kern_sum=None, ncu_csv=None, steps=1, dims_str=R12_DIMS, target_gbps=3000.0,
            job_id=None, table_out=None, force_canonical=False, quiet=False, run_dir_arg=None):
    run_dir = Path(run_dir)
    if job_id is None:
        m = re.search(r"_(\d+)$", run_dir.name)
        job_id = m.group(1) if m else "na"
    dims = parse_dims(dims_str)
    csv_path, fallback = find_ncu_csv(run_dir, ncu_csv)
    launches = load_ncu_raw(csv_path)
    graphs = load_output_code(run_dir)
    assign_regions(launches, graphs, dims)
    kern, regions = aggregate(launches, steps)
    minb = min_bytes_model(dims)
    target_lines, target_data = build_targets(regions, minb, target_gbps, dims)
    nsys = load_nsys(attrib, kern_sum, launches[0]["name"] if launches else None) if attrib else None
    xcheck, xmissing = crosscheck(kern, nsys) if nsys else (None, None)
    nsys_region = region_nsys_us(kern, nsys) if nsys else {}
    have_nvtx = any(l.get("nvtx") not in (None, "none") for l in launches)
    # every per-step number divides by --steps: it must match the steps the capture actually covers
    nvtx_steps = sorted({l["nvtx_step"] for l in launches if l.get("nvtx_step") is not None})
    if nvtx_steps and len(nvtx_steps) != steps:
        print(f"!! WARNING: the ncu capture covers NVTX steps {nvtx_steps} but --steps={steps}; per-step numbers "
              f"are wrong unless you pass --steps {len(nvtx_steps)}", file=sys.stderr)
    ctx = dict(job_id=job_id, ncu_csv=str(csv_path), n_launches=len(launches), steps=steps, graphs=graphs,
               nvtx_steps=nvtx_steps,
               graph_roles=[f"{g['label'].split('_output')[0]}={g['role']}({len(g['kernels'])})" for g in graphs],
               n_unmapped=sum(1 for l in launches if l["unmapped"]),
               unmapped_names=sorted({l["name"] for l in launches if l["unmapped"]}),
               phase_source="NVTX ranges in the ncu CSV (no range = autograd thread = backward)" if have_nvtx
               else "graph roles + launch-sequence state machine (no NVTX in the CSV)",
               dims_str=dims_str, nsys=nsys, fallback=fallback, regions=regions, minb=minb,
               nsys_region=nsys_region, target_gbps=target_gbps, target_lines=target_lines, xcheck=xcheck,
               xmissing=xmissing, kern=kern, run_dir_arg=run_dir_arg or str(run_dir))
    rows = table_rows(kern, regions, minb, job_id)
    write_csv(run_dir / "inductor_fusion_table.csv", rows)
    canonical_written = write_canonical(table_out or CANONICAL_TABLE, rows, job_id, force_canonical)
    md = summary_md(ctx)
    (run_dir / "inductor_fusion_summary.md").write_text(md)
    with open(run_dir / "inductor_fusion_regions.json", "w") as f:
        json.dump(dict(job_id=job_id, ncu_csv=str(csv_path), steps=steps, dims=dims, hbm_GBps=HBM_GBPS,
                       target_GBps=target_gbps, regions={k: dict(v, tier=REGION_TIER.get(k, "-"),
                                                                 min_bytes=minb.get(k, {}).get("bytes"),
                                                                 min_formula=minb.get(k, {}).get("formula"),
                                                                 nsys_us=nsys_region.get(k))
                                                         for k, v in regions.items()},
                       targets=target_data, graphs=[dict(label=g["label"], aot=g["aot"], role=g["role"],
                                                         kernels=len(g["kernels"]), n_mm=g["n_mm"], n_bmm=g["n_bmm"])
                                                    for g in graphs],
                       crosscheck=xcheck, unmapped=ctx["unmapped_names"]), f, indent=1)
    if not quiet:
        print(md)
        print(f"wrote {run_dir / 'inductor_fusion_table.csv'}, {run_dir / 'inductor_fusion_summary.md'}, "
              f"{run_dir / 'inductor_fusion_regions.json'}" +
              (f", {table_out or CANONICAL_TABLE}" if canonical_written else ""))
    return ctx, rows


# ----------------------------------------------------------------------------------------------------------------
# self-test: synthetic inputs in the true formats (ncu 2025.2 raw CSV incl. units row; torch 2.9.1 output_code.py)

def _fake_kernel(name, src, aten, shapes=()):
    frag = "".join(f'#   %buf{i} : Tensor "bf16[{", ".join(map(str, s))}][1]cuda:0" = PlaceHolder[target=buf{i}]\n'
                   for i, s in enumerate(shapes))
    return (f"# kernel path: /tmp/x/{name[:6]}.py\n"
            f"# Topologically Sorted Source Nodes: [{', '.join(src)}], Original ATen: [{', '.join(aten)}]\n"
            f"# Source node to ATen node mapping:\n# Graph fragment:\n{frag}"
            f"{name} = async_compile.triton('{name}', '''\nimport triton\n@triton.jit\ndef {name}(in_ptr0, xnumel):\n"
            f"    pass\n''', device_str='cuda')\n\n\n")


def _fake_module(aot, kernels, runs, extern=""):
    body = "".join(_fake_kernel(*k) for k in kernels)
    calls = "".join(f"            {n}.run(buf0, buf1, 1024, stream=stream0)\n" for n in runs)
    return (f"# AOT ID: ['{aot}']\nfrom ctypes import c_void_p\nimport torch\n\nasync_compile = AsyncCompile()\n\n\n"
            f"{body}async_compile.wait(globals())\n\nclass Runner:\n    def call(self, args):\n"
            f"        with torch.cuda._DeviceGuard(0):\n{calls}{extern}        return ()\n")


FA3_FWD = ("cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90"
           "<(int)2, cute::tuple<cute::C<(int)1>>, cutlass::bfloat16_t, float, cutlass::arch::Sm90>>>>(T1::Params)")
FA3_BWD = FA3_FWD.replace("FlashAttnFwdSm90", "FlashAttnBwdSm90").replace("MainloopFwd", "MainloopBwd")


def _selftest_inputs(root):
    """Writes a run dir with 4 graphs and a raw CSV; returns (run_dir, expected regions by launch)."""
    run = Path(root) / "s2_inductor_4242"
    (run / "output_code").mkdir(parents=True)
    d = parse_dims(R12_DIMS)
    qk = (d["B"], d["T"], d["H"], d["D"])
    half = (d["B"], d["T"], d["H"], d["D"] // 2)
    fwd = [("triton_per_fused__to_copy_add_embedding_mean_pow_rsqrt_0", ["x", "x_2"],
            ["aten.embedding", "aten._to_copy", "aten.pow", "aten.mean", "aten.add", "aten.rsqrt"]),
           ("triton_poi_fused__to_copy_4", ["to_3", "linear_1"], ["aten._to_copy", "aten.t"]),
           ("triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6",
            ["cos", "sin", "linear_1", "q", "linear_2", "k", "x1", "x2", "y1", "neg", "y2"],
            ["aten.slice", "aten._unsafe_view", "aten.mul", "aten.add", "aten.neg", "aten.cat", "aten._to_copy",
             "aten.pow", "aten.mean", "aten.rsqrt"], [qk]),
           ("triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_7",
            ["getitem_6", "mul_2", "x_4", "y_2", "x_5", "rms_norm_4"],
            ["aten.select", "aten.mul", "aten.add", "aten._to_copy", "aten.pow", "aten.mean", "aten.rsqrt"]),
           ("triton_poi_fused__unsafe_view_pow_relu_9", ["x_6", "relu", "x_7"],
            ["aten._unsafe_view", "aten.relu", "aten.pow"]),
           ("triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view_36",
            ["logits", "tanh", "loss"], ["aten._to_copy", "aten.div", "aten.tanh", "aten.mul",
                                         "prims.prepare_softmax_online", "aten._log_softmax"])]
    bwd = [("triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_tanh_0",
            ["view_55", "loss", "logits", "tanh"], ["aten.nll_loss_backward", "aten._log_softmax",
                                                    "aten._log_softmax_backward_data", "aten.tanh_backward"]),
           ("triton_poi_fused__to_copy_4", [], ["aten._to_copy"]),          # same name as a forward kernel
           ("triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5", ["x_72", "relu_11"],
            ["aten.view", "aten.relu", "aten.pow", "aten.mul", "aten.threshold_backward"]),
           ("triton_poi_fused_add_slice_backward_13", [], ["aten.slice_backward", "aten.add"], [half, half, qk]),
           ("triton_per_fused__fused_rms_norm_backward__to_copy_mul_8", ["k_46"],
            ["aten.mul", "aten._fused_rms_norm_backward", "aten._to_copy"]),
           ("triton_red_fused_mul_sum_18", [], ["aten.mul", "aten.sum"]),
           ("triton_per_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_2", ["x_77"],
            ["aten.view", "aten._fused_rms_norm_backward", "aten._to_copy", "aten.mul", "aten.neg", "aten.sum"])]
    adam = [("triton_poi_fused__to_copy_add_copy_copy__div_lerp_mul_neg_pow_rsub_sqrt_0",
             ["p32", "lerp_", "denom", "sqrt"], ["aten._to_copy", "aten.mul", "aten.rsub", "aten.lerp", "aten.pow",
                                                 "aten.div", "aten.sqrt", "aten.add", "aten.copy_"])]
    muon = [("triton_per_fused__to_copy_lerp_linalg_vector_norm_rsub_0", ["g", "lerp_", "X", "norm"],
             ["aten.lerp", "aten.rsub", "aten._to_copy", "aten.linalg_vector_norm"]),
            ("triton_poi_fused_add_mul_3", ["mul_2", "mul_3", "B"], ["aten.mul", "aten.add"])]
    files = {"graph_001_output_code.py": _fake_module("0_backward", bwd, [k[0] for k in bwd]),
             "graph_002_output_code.py": _fake_module("0_forward", fwd, [k[0] for k in fwd]),
             "graph_003_output_code.py": _fake_module("1_inference", adam, [adam[0][0]]),
             "graph_004_output_code.py": _fake_module("6_inference", muon, [k[0] for k in muon],
                                                      "        extern_kernels.bmm(buf0, buf1, out=buf2)\n")}
    for n, t in files.items():
        (run / "output_code" / n).write_text(t)
    # (name, region, nvtx phase, us, MB read, MB write)
    seq = [(fwd[0][0], "embedding_ve_fwd", "forward", 40.5, 10.5, 23.6),
           (fwd[1][0], "weight_cast_fwd", "forward", 3.0, 2.4, 1.2),
           ("nvjet_tst_128x256_64x4_2x1_v_bz_coopA_TNN", "gemm_fwd", "forward", 70.0, 52.0, 50.3),
           (fwd[2][0], "qkrope_fwd", "forward", 110.0, 101.0, 101.0),
           (FA3_FWD, "attention_fwd", "forward", 215.0, 151.0, 51.0),
           (fwd[3][0], "mixnorm_fwd", "forward", 75.0, 151.0, 101.0),
           (fwd[4][0], "relu2_fwd", "forward", 110.0, 201.0, 201.0),
           (fwd[5][0], "ce_fwd", "forward", 2260.0, 2150.0, 0.3),
           (bwd[0][0], "ce_bwd", "", 2180.0, 2150.0, 2150.0),
           (bwd[1][0], "weight_cast_bwd", "", 2.5, 1.2, 2.4),
           (bwd[2][0], "relu2_bwd", "", 165.0, 403.0, 201.0),
           (bwd[3][0], "qkrope_bwd", "", 28.0, 50.0, 50.0),
           (bwd[4][0], "qkrope_bwd", "", 32.0, 75.0, 50.0),
           (FA3_BWD, "attention_bwd", "", 580.0, 300.0, 250.0),
           ("nvjet_tst_192x192_64x3_1x2_h_bz_coopB_splitK_NTN", "gemm_bwd", "", 240.0, 60.0, 5.0),
           (bwd[5][0], "mixnorm_bwd", "", 3.4, 0.2, 0.0),
           (bwd[6][0], "mixnorm_bwd", "", 79.0, 151.0, 50.0),
           ("triton_poi_fused_add_mul_99", "other_elementwise_bwd", "", 5.0, 3.0, 1.0),   # not in output_code
           (adam[0][0], "optim_adamw", "optimizer", 700.0, 1800.0, 1300.0),
           ("void at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig<at::native::(anonymous "
            "namespace)::OpaqueType<4>, unsigned int, 3, 128, 1>(T1::Params)", "optim_muon_stack", "optimizer",
            399.0, 680.0, 680.0),
           (muon[0][0], "optim_muon", "optimizer", 240.0, 500.0, 200.0),
           ("nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN", "optim_muon_ns_gemm", "optimizer", 120.0, 40.0, 20.0),
           (muon[1][0], "optim_muon_ns", "optimizer", 63.0, 120.0, 60.0),
           ("void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::"
            "TensorListMetadata<2>, at::native::(anonymous namespace)::CopyFunctor<float, float, float, 1>>(T1, T2)",
            "optim_muon_stack", "optimizer", 241.0, 300.0, 300.0)]
    return run, seq


def _write_raw_csv(path, seq, units="base", nvtx=True):
    """The ncu --page raw layout: quoted header, units row, thousands separators, plus unrelated columns."""
    nv_col = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
    header = ["ID", "Process ID", "Process Name", "Host Name"] + ([nv_col] if nvtx else []) + [
        "Kernel Name", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC", "dram__bytes_read.avg",
        "dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum", "launch__block_size",
        "launch__grid_size", "launch__registers_per_thread", "sm__warps_active.avg.pct_of_peak_sustained_active"]
    scaled = units == "scaled"                       # units=None: no units row, base units
    scale_b, scale_t = (1e6, 1e3) if scaled else (1, 1)
    ub, ut = ("Mbyte", "usecond") if scaled else ("byte", "ns")
    unit_row = [""] * (4 + nvtx) + [""] * 7 + [ub, ub, ub, ut, "", "", "register/thread", "%"]

    def num(v, nd=0):
        return f"{v:,.{nd}f}"
    lines = []
    for i, (name, _, ph, us, mr, mw) in enumerate(seq):
        nvf = (f'1234  "<default domain>:step_20:none:none:none:none:none:none"  "<default domain>:{ph}:none:none:'
               f'none:none:none:none" ') if ph else ""
        rb, wb, t = mr * 1e6 / scale_b, mw * 1e6 / scale_b, us * 1e3 / scale_t
        nd_b, nd_t = (3, 3) if scaled else (0, 0)
        lines.append([str(i), "1234", "python3.10", "127.0.0.1"] + ([nvf] if nvtx else []) + [
            name, "1", "7", "(256, 1, 1)", "(32768, 1, 1)", "0", "9.0", num(rb / 132, 2), num(rb, nd_b),
            num(wb, nd_b), num(t, nd_t), "256", "32,768", "40", "86.21"])
    with open(path, "w", newline="") as f:
        f.write("==PROF== Connected to process 1234\n")
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(header)
        if units is not None:
            w.writerow(unit_row)
        w.writerows(lines)


def selftest():
    fails = []

    def check(cond, msg):
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    check(parse_number('"1,420,832"'.strip('"')) == 1420832.0 and parse_number("93.89") == 93.89
          and parse_number("n/a") is None, "parse_number: thousands separators, decimals, n/a")
    toks = name_tokens("triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_16")
    check({"_fused_rms_norm_backward", "_to_copy", "select"} <= toks, f"name_tokens multi-word ops: {sorted(toks)}")
    m = min_bytes_model(parse_dims(R12_DIMS))
    A = 32768 * 768 * 2
    check(m["ce_fwd"]["bytes"] == 32768 * 32768 * 2 + 32768 * 12, "min bytes: CE fwd = bf16 logits once + targets + lse")
    check(abs(m["optim_adamw"]["bytes"] / 1e9 - 3.17) < 0.02, f"min bytes: AdamW {m['optim_adamw']['bytes'] / 1e9:.3f}"
          " GB ~ inventory 3.17 GB")
    check(m["relu2_fwd"]["bytes"] == 12 * 8 * A and m["qkrope_fwd"]["bytes"] > 12 * 4 * A,
          "min bytes: relu2 fwd 8A/layer, qkrope fwd 4A/layer + rstd")
    check(param_counts(parse_dims(R12_DIMS))["adam_tensors"] == 13, "AdamW tensor count = 13 (inventory)")
    pairs = [  # (ncu 2025.2 name, nsys name) as printed for the same kernel in jobs 3406171 / 3400686
        ("void at::<unnamed>::CatArrayBatchedCopy_alignedK_contig<at::<unnamed>::OpaqueType<4>, unsigned int, 2, 128,"
         " 1, 16>(T1 *, at::<unnamed>::CatArrInputTensorMetadata<T1, T2, T4, T5>, int, T2)",
         "void at::native::<unnamed>::CatArrayBatchedCopy_alignedK_contig<at::native::<unnamed>::OpaqueType<(unsigned "
         "int)4>, unsigned int, (int)2, (int)128, (int)1, (int)16>(T1 *, at::native::<unnamed>::CatArrInputTensorMeta"
         "data<T1, T2, T4, T5>, int, T2)"),
        ("void cutlass::device_kernel<flash::FlashAttnBwdPreprocess<cute::tuple<cute::C<64>, cute::C<128>>, "
         "cutlass::bfloat16_t, float, arch::Sm90, 1, 0>>(T1::Params)",
         "void cutlass::device_kernel<flash::FlashAttnBwdPreprocess<cute::tuple<cute::C<(int)64>, cute::C<(int)128>>,"
         " cutlass::bfloat16_t, float, cutlass::arch::Sm90, (bool)1, (bool)0>>(T1::Params)")]
    check(all(nkey(a) == nkey(b) for a, b in pairs), "nkey joins ncu and nsys spellings of CatArray / FA3 names")
    check(nkey("triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_14") !=
          nkey("triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_20"),
          "nkey keeps long Triton names distinct")

    with tempfile.TemporaryDirectory() as td:
        run, seq = _selftest_inputs(td)
        _write_raw_csv(run / "ncu_step20_raw.csv", seq)
        # nsys side: kern_sum CSV (5 steps) with one kernel 50% faster than ncu and one launch-count mismatch
        att = Path(td) / "attrib_compile_fa3.json"
        ks = Path(td) / "compile_fa3_cuda_gpu_kern_sum.csv"
        with open(ks, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Time (%)", "Total Time (ns)", "Instances", "Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)",
                        "StdDev (ns)", "Name"])
            for name, _, _, us, _, _ in seq:
                f_t = 0.5 if "prepare_softmax_online" in name else 1.0
                inst = 10 if "relu_9" in name else 5
                nm = ("void " + name) if name.startswith("cutlass") else name
                w.writerow(["1", us * 1e3 * 5 * f_t, inst, us * 1e3, 0, 0, 0, 0, nm])
        att.write_text(json.dumps(dict(steps=5, total_ms_per_step=sum(s[3] for s in seq) / 1e3, top={})))
        canon = Path(td) / "canonical.csv"
        ctx, rows = analyze(run, attrib=att, table_out=canon, quiet=True)
        got = [l for l in load_ncu_raw(run / "ncu_step20_raw.csv")]
        assign_regions(got, load_output_code(run), parse_dims(R12_DIMS))
        for l, s in zip(got, seq):
            check(l["region"] == s[1], f"region {s[0][:60]:<60} -> {l['region']} (want {s[1]})")
        check(len(ctx["graphs"]) == 4 and [g["role"] for g in ctx["graphs"]] == ["bwd", "fwd", "adamw", "muon"],
              f"graph roles {[g['role'] for g in ctx['graphs']]}")
        check(ctx["unmapped_names"] == ["triton_poi_fused_add_mul_99"], "unmapped Triton kernel reported")
        r = ctx["regions"]
        check(abs(r["qkrope_bwd"]["us"] - 60.0) < 1e-6 and abs(r["qkrope_bwd"]["bytes"] - 225e6) < 1,
              "region totals: qkrope_bwd us and bytes")
        check(abs(sum(x["us"] for x in r.values()) - sum(s[3] for s in seq)) < 1e-6, "total us conserved")
        flagged = {x["name"]: x["flag"] for x in ctx["xcheck"] if x["flag"]}
        ce = next(n for n in flagged if "prepare_softmax_online" in n)
        check("time differs +100%" in flagged[ce], f"xcheck flags CE fwd: {flagged[ce]}")
        check(any("relu_9" in n and "launches" in f for n, f in flagged.items()), "xcheck flags launch mismatch")
        check(not any("FlashAttnFwd" in n for n in flagged), "xcheck matches 'void '-prefixed nsys names")
        check(canon.exists() and (run / "inductor_fusion_summary.md").exists()
              and (run / "inductor_fusion_regions.json").exists(), "outputs written")
        with open(canon, newline="") as f:
            tab = list(csv.DictReader(f))
        check(list(tab[0].keys()) == TABLE_COLUMNS and all(t["job_id"] == "4242" for t in tab), "table columns/job_id")
        tot = {t["region"]: t for t in tab if t["kernel_name"] == REGION_TOTAL}
        check(tot["ce_fwd"]["min_bytes_per_step"] != "" and abs(float(tot["ce_fwd"]["bytes_ratio"]) - 2150.3e6 /
                                                                 m["ce_fwd"]["bytes"]) < 1e-3, "region total ratio")
        # trace CSV next to the attrib JSON: per-kernel medians over steps, robust to one slow step
        with open(Path(td) / "compile_fa3_cuda_gpu_trace.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Start (ns)", "Duration (ns)", "Bytes (MB)", "Name"])
            for step in range(3):
                for name, _, _, us, _, _ in seq:
                    slow = 10.0 if (step == 1 and "relu_9" in name) else 1.0
                    w.writerow([0, us * 1e3 * slow, "", ("void " + name) if name.startswith("cutlass") else name])
                w.writerow([0, 1000, "0.5", "[CUDA memcpy Host-to-Device]"])
        ns = load_nsys(att, first_kernel=seq[0][0])
        relu = nsys_lookup(ns["per"], nkey(seq[6][0]))
        check(len(ns["step_totals_us"]) == 3 and abs(relu["us"] - seq[6][3]) < 1e-6 and relu["launches"] == 1,
              "trace: 3 steps found, per-kernel median ignores the slow step")
        md = (run / "inductor_fusion_summary.md").read_text()
        check("target vs compile" in md and "softcap+CE fwd" in md and "N*V*2" in md, "summary has targets + formulas")
        # canonical guard: another job's table is not overwritten
        run2 = Path(td) / "s2_inductor_5555"
        (run2 / "output_code").mkdir(parents=True)
        for f in (run / "output_code").iterdir():
            (run2 / "output_code" / f.name).write_text(f.read_text())
        _write_raw_csv(run2 / "ncu_step20_raw.csv", seq)
        analyze(run2, table_out=canon, quiet=True)
        with open(canon, newline="") as f:
            check(next(csv.DictReader(f))["job_id"] == "4242", "canonical table of another job kept")
        # units: scaled units (Mbyte/usecond) and no units row / no NVTX column give the same totals
        for variant, kw in (("scaled units", dict(units="scaled")), ("no units row, no NVTX", dict(units=None,
                                                                                                  nvtx=False))):
            p = Path(td) / f"raw_{variant.replace(' ', '_').replace(',', '')}.csv"
            _write_raw_csv(p, seq, **kw)
            ls = load_ncu_raw(p)
            assign_regions(ls, load_output_code(run), parse_dims(R12_DIMS))
            ok_t = abs(sum(l["time_ns"] for l in ls) / 1e3 - sum(s[3] for s in seq)) < 1e-3 * len(seq)
            ok_b = abs(sum(l["dram_read"] + l["dram_write"] for l in ls) - sum((s[4] + s[5]) * 1e6 for s in seq)) < 1e3
            ok_r = [l["region"] for l in ls] == [s[1] for s in seq]
            check(ok_t and ok_b and ok_r, f"{variant}: same time, bytes and regions")
    print(f"\nSELFTEST {'PASS' if not fails else 'FAIL'} ({len(fails)} failures)")
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", nargs="?", help="results/runs/s2_inductor_<jobid>")
    ap.add_argument("--attrib", default=None, help="S1 nsys attribution JSON for the cross-check")
    ap.add_argument("--kern-sum", default=None, help="nsys cuda_gpu_kern_sum CSV (default: next to --attrib)")
    ap.add_argument("--ncu-csv", default=None, help="override the ncu raw CSV (default: ncu_step20_raw.csv, "
                                                    "else ncu_step20_tierA_raw.csv in run_dir)")
    ap.add_argument("--steps", type=int, default=1, help="training steps covered by the ncu capture")
    ap.add_argument("--dims", default=R12_DIMS)
    ap.add_argument("--target-gbps", type=float, default=3000.0, help="central bandwidth of a fused kernel")
    ap.add_argument("--job-id", default=None, help="default: trailing digits of run_dir")
    ap.add_argument("--table-out", default=None, help=f"canonical table path (default {CANONICAL_TABLE})")
    ap.add_argument("--force-canonical", action="store_true", help="replace a canonical table from another job")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.run_dir:
        ap.error("run_dir is required (or --selftest)")
    run_dir = resolve(args.run_dir)
    if not run_dir.is_dir():
        ap.error(f"{args.run_dir}: not a directory")
    analyze(run_dir, attrib=resolve(args.attrib), kern_sum=resolve(args.kern_sum), ncu_csv=args.ncu_csv,
            steps=args.steps, dims_str=args.dims, target_gbps=args.target_gbps, job_id=args.job_id,
            table_out=resolve(args.table_out) if args.table_out else None, force_canonical=args.force_canonical,
            run_dir_arg=args.run_dir)


if __name__ == "__main__":
    main()
