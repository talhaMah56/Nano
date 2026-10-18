"""
Session-1 figures for Presentation 1 (preliminary results). Static PNGs for slides.

Reads the run summaries under cisc662/results/runs/ and writes cisc662/results/figures/s1_*.png:
  s1_kernel_breakdown.png  GPU kernel time per training step by class, eager vs torch.compile (FA3)
  s1_train_throughput.png  seconds per 100 training steps, eager/compile x FA3/SDPA (mean ± std, 3 reps)
  s1_convergence.png       validation bpb of the 2000-step R12 baseline
  s1_cpu_vs_gpu.png        training tokens/s at batch 1: CPU 1 thread, CPU 72 threads, GPU eager, GPU compile
  s1_decode.png            decode tokens/s vs batch size, FA3 vs SDPA (infer_bench)

Style follows the dataviz skill's reference palette: fixed categorical order (blue, orange, aqua),
hairline recessive grid, no dual axes, legend for >= 2 series plus selective direct labels.

Usage: python -m cisc662.analysis.make_figs_s1
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
FIGDIR = os.path.join(ROOT, "figures")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs in the reference palette)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def latest(pattern):
    paths = sorted(glob.glob(os.path.join(ROOT, "runs", pattern)), key=os.path.getmtime)
    return paths[-1] if paths else None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def value_grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------------------------------------------
DISPLAY = [  # (row label, classes from nsys_attrib that map to it)
    ("GEMM (cuBLAS)", ["gemm"]),
    ("Attention (FA3)", ["attention"]),
    ("Softcap + cross-entropy", ["softmax_ce"]),
    ("RMSNorm (incl. QK-norm)", ["rmsnorm"]),
    ("RoPE", ["rope"]),
    ("RoPE + QK-norm, fused", ["rmsnorm+rope"]),
    ("Optimizer (Muon + AdamW)", ["optimizer"]),
    ("relu² MLP activation", ["relu2_mlp"]),
    ("Residual / elementwise", ["residual_elementwise"]),
]


def rows_from_attrib(attrib):
    by = {k: v["ms_per_step"] for k, v in attrib["by_class"].items()}
    used, rows = set(), []
    for label, classes in DISPLAY:
        rows.append((label, sum(by.get(c, 0.0) for c in classes)))
        used.update(classes)
    rows.append(("Other (embedding, casts, mixed)", sum(v for k, v in by.items() if k not in used)))
    return rows, attrib["total_ms_per_step"]


def fig_breakdown():
    eager_p = latest("s1_nsys_*/attrib_eager_fa3_ops.json")
    comp_p = latest("s1_nsys_*/attrib_compile_fa3.json")
    if not (eager_p and comp_p):
        print("skip breakdown: attribution JSON missing")
        return
    eager, e_tot = rows_from_attrib(load_json(eager_p))
    comp, c_tot = rows_from_attrib(load_json(comp_p))
    labels = [r[0] for r in eager][::-1]
    ev = [r[1] for r in eager][::-1]
    cv = [r[1] for r in comp][::-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    y = range(len(labels))
    h = 0.38
    ax.barh([i + h / 2 for i in y], ev, height=h, color=SERIES[0], edgecolor=SURFACE, linewidth=2,
            label=f"PyTorch eager  ({e_tot:.1f} ms/step)")
    ax.barh([i - h / 2 for i in y], cv, height=h, color=SERIES[1], edgecolor=SURFACE, linewidth=2,
            label=f"torch.compile  ({c_tot:.1f} ms/step)")
    for i, (e, c) in enumerate(zip(ev, cv)):  # label only the rows that matter (> 2 ms in either mode), never zeros
        if max(e, c) > 2.0:
            for v, dy in ((e, h / 2), (c, -h / 2)):
                if v >= 0.05:
                    ax.text(v + 0.4, i + dy, f"{v:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(list(y), labels)
    ax.set_xlabel("GPU kernel time per training step (ms)")
    value_grid(ax, "x")
    ax.legend(loc="lower right")
    ax.set_title("Where a training step's GPU time goes — nanochat R12 on one GH200 (FA3 attention)")
    fig.text(0.0, -0.02, "Nsight Systems, 5 steps, per-kernel attribution by phase and aten op. R12: depth 12, "
             "seq 2048, 32k tokens/step, bf16.", fontsize=8, color=MUTED)
    save(fig, "s1_kernel_breakdown.png")


def fig_throughput():
    p = latest("s1_timing_*/summary.json")
    if not p:
        print("skip throughput: timing summary missing")
        return
    tr = load_json(p)["train"]
    order = [("compile_fa3", "torch.compile + FA3 (stock)"), ("compile_sdpa", "torch.compile + SDPA"),
             ("eager_fa3", "eager + FA3"), ("eager_sdpa", "eager + SDPA")]
    order = [(k, l) for k, l in order if k in tr]
    vals = [tr[k]["s_per_100_mean"] for k, _ in order]
    errs = [tr[k]["s_per_100_std"] for k, _ in order]
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    y = list(range(len(order)))[::-1]
    ax.barh(y, vals, xerr=errs, height=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2,
            error_kw=dict(ecolor=INK2, elinewidth=1, capsize=3))
    for yi, (k, _), v in zip(y, order, vals):
        t = tr[k]
        ax.text(v + max(vals) * 0.02, yi, f"{v:.2f} s   ({t['tok_per_sec'] / 1e3:,.0f}k tok/s, MFU {t['mfu_pct']:.0f}%)",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [l for _, l in order])
    ax.set_xlim(0, max(vals) * 1.75)
    ax.set_xlabel("seconds per 100 training steps (lower is better)")
    value_grid(ax, "x")
    ax.set_title("Baseline training speed, R12 on one GH200 (mean ± std of 3 runs)")
    save(fig, "s1_train_throughput.png")


def fig_convergence():
    p = latest("s1_conv_infer_*/summary.json")
    if not p:
        print("skip convergence: conv summary missing")
        return
    conv = load_json(p).get("conv")
    if not conv or not conv["val_bpb"]:
        print("skip convergence: no val bpb")
        return
    xs = [v["step"] for v in conv["val_bpb"]]
    ys = [v["val_bpb"] for v in conv["val_bpb"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(xs, ys, color=SERIES[0], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=INK2)
    ax.set_xlabel("training step (32,768 tokens each)")
    ax.set_ylabel("validation bits per byte")
    value_grid(ax, "y")
    ax.set_title("Loss-curve baseline: R12, torch.compile + FA3, 2000 steps")
    save(fig, "s1_convergence.png")


def fig_cpu_vs_gpu():
    p = latest("s1_cpuref_*/summary.json")
    if not p:
        print("skip cpu_vs_gpu: cpuref summary missing")
        return
    s = load_json(p)
    bars = []
    if "cpu_t1" in s["cpu"]:
        bars.append(("CPU, 1 Grace core (fp32)", s["cpu"]["cpu_t1"]["tok_per_sec"]))
    if "cpu_t72" in s["cpu"]:
        bars.append(("CPU, 72 Grace cores (fp32)", s["cpu"]["cpu_t72"]["tok_per_sec"]))
    for key, label in (("b1-eager_fa3", "GPU eager (bf16)"), ("b1-compile_fa3", "GPU torch.compile (bf16)")):
        if key in s["train"]:
            bars.append((label, s["train"][key]["tok_per_sec"]))
    if len(bars) < 2:
        print("skip cpu_vs_gpu: not enough data")
        return
    # the GPU's real operating point (batch 16) from the timing job, labeled as a different batch
    tp = latest("s1_timing_*/summary.json")
    if tp and "compile_fa3" in load_json(tp)["train"]:
        bars.append(("GPU torch.compile, batch 16 (bf16)", load_json(tp)["train"]["compile_fa3"]["tok_per_sec"]))
    base = bars[0][1]
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    y = list(range(len(bars)))[::-1]
    ax.barh(y, [b[1] for b in bars], height=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.set_xscale("log")
    for yi, (label, v) in zip(y, bars):
        ax.text(v * 1.15, yi, f"{v:,.0f} tok/s  ({v / base:,.0f}× vs 1 core)", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [b[0] for b in bars])
    ticks = [10, 100, 1e3, 1e4, 1e5, 1e6]
    ax.set_xticks(ticks, ["10", "100", "1k", "10k", "100k", "1M"])
    ax.minorticks_off()
    ax.set_xlim(min(b[1] for b in bars) / 2, max(b[1] for b in bars) * 25)
    ax.set_xlabel("training throughput (tokens/s, log scale); batch 1 × 2048 tokens unless noted")
    value_grid(ax, "x")
    ax.set_title("GPU speedup over the serial CPU reference (same R12 model, fp32 on CPU, bf16 on GPU)")
    save(fig, "s1_cpu_vs_gpu.png")


def fig_decode():
    p = latest("s1_conv_infer_*/summary.json")
    if not p:
        print("skip decode: summary missing")
        return
    inf = load_json(p).get("infer", {})
    if not inf:
        print("skip decode: no infer_bench results")
        return
    # Time per output token (TPOT) vs batch, against the HBM-bandwidth floor: each decode step must at least read
    # all weights plus every row's KV cache once, so floor = (weight_bytes + B * kv_read_bytes) / peak_bandwidth.
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ref, all_tpot = None, []
    for i, (key, label) in enumerate((("fa3", "FA3"), ("sdpa", "SDPA"))):
        runs = inf.get(key, [])
        if not runs:
            continue
        ref = ref or runs[0]
        bs = [r["batch_size"] for r in runs[0]["sweep"]]
        tpot = [sum(run["sweep"][j]["tpot_sec"] for run in runs) / len(runs) * 1e3 for j in range(len(bs))]
        all_tpot.append(tpot)
        ax.plot(bs, tpot, color=SERIES[i], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=f"measured, {label}")
    bs = [r["batch_size"] for r in ref["sweep"]]
    floor = [(ref["weight_bytes"] + b * ref["kv_read_bytes_per_step"]) / ref["peak_bandwidth_bytes_per_sec"] * 1e3 for b in bs]
    ratios = [t / f for tp in all_tpot for t, f in zip(tp, floor)]
    ax.plot(bs, floor, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(f"HBM-bandwidth floor at {ref['peak_bandwidth_bytes_per_sec'] / 1e12:.1f} TB/s "
                f"(weights + KV read once per step)", (bs[0], floor[0]), textcoords="offset points", xytext=(0, -13),
                fontsize=8.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(bs, [str(b) for b in bs])
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0.08, 40)
    ax.set_xlabel("decode batch size")
    ax.set_ylabel("time per output token (ms, log scale)")
    value_grid(ax, "y")
    ax.legend(loc="center right")
    ax.set_title(f"Decode is launch-bound: ~{sum(all_tpot[0]) / len(all_tpot[0]):.0f} ms per token at every batch size, "
                 f"{min(ratios):.0f}–{max(ratios):.0f}× above the HBM floor")
    fig.text(0.0, -0.03, "infer_bench, prompt 512, 256 decoded tokens, mean of 2 runs. Nsight Systems: GPU busy 8.4% of "
             "decode time; median kernel 1.9 µs.", fontsize=8, color=MUTED)
    save(fig, "s1_decode.png")


if __name__ == "__main__":
    fig_breakdown()
    fig_throughput()
    fig_convergence()
    fig_cpu_vs_gpu()
    fig_decode()
