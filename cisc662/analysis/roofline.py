"""
ANA-1: GH200 roofline of nanochat R12's kernel classes, eager vs torch.compile.

v0 (S3) uses only data that already exists — no GPU:
  (a) isolated per-op microbenchmarks, results/db/ops_baseline/*.json (kbench, job 3406329; by_job/ copies ignored).
      x = flops / min_bytes (analytic arithmetic intensity: each input read once, each output written once),
      y = flops / median time (CUDA events, cold L2, default clocks).
      For memory-bound ops the "flops" field is an approximate op count, so their x position is approximate too;
      their y-distance below the bandwidth roof is the meaningful quantity.
  (b) in-model compiled regions of one training step, results/runs/s2_inductor_3406171/inductor_fusion_regions.json
      (ncu, job 3406171: DRAM read+write bytes and kernel time per region, --clock-control none, cold L2 per kernel).
      Regions whose FLOPs follow cleanly from the shapes (cuBLAS GEMMs fwd/bwd, FA3 fwd/bwd, Muon Newton-Schulz bmm)
      are roofline points with x = analytic FLOPs / measured DRAM bytes, y = FLOPs / ncu time.
      Memory-bound regions (Inductor-fused elementwise/norm/CE/optimizer) are NOT given invented FLOPs: they are shown
      in a second panel as achieved bandwidth (DRAM bytes / ncu time) against the 3.65 TB/s roof.
  Ceilings (PLAN.md §2, §7.4; job 3406282): HBM 3.65 TB/s measured copy (4.0 datasheet, dashed); bf16 tensor 723 TF/s
  measured burst (990 datasheet, dashed) and 591 TF/s sustained under the module power cap; fp32 48.6 measured
  (67 datasheet, dashed). Ridge = 723 / 3.65 ≈ 198 FLOP/B.

v1 (S4, after BAS-6) will replace the analytic bytes of (a) by ncu bytes.

Outputs:  results/figures/roofline_<version>.png   results/roofline_points_<version>.csv
Usage:    python -m cisc662.analysis.roofline --version v0 [--out cisc662/results/figures/roofline_v0.png]
Style follows make_figs_s1.py (palette, fonts, hairline grid, no dual axes, job-id stamp).
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PKG, "results")
OPS_DIR = os.path.join(RESULTS, "db", "ops_baseline")
REGIONS_JSON = os.path.join(RESULTS, "runs", "s2_inductor_3406171", "inductor_fusion_regions.json")
OPS_JOB, NCU_JOB = "3406329", "3406171"

# ---- ceilings (FLOP/s and B/s) ----
HBM_MEAS, HBM_SHEET = 3.65e12, 4.0e12
BF16_BURST, BF16_SUSTAINED, BF16_SHEET = 723e12, 591e12, 990e12
FP32_MEAS, FP32_SHEET = 48.6e12, 67e12
RIDGE = BF16_BURST / HBM_MEAS

# ---- style (make_figs_s1.py) ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
EAGER, COMPILE = SERIES[0], SERIES[1]
TINT_MEM, TINT_CMP = "#f3f2ee", "#eef3fa"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})

# ---- R12 shapes (research_hotpath-inventory.json / inductor_fusion_regions.json "dims") ----
N, C, F, V, L = 32768, 768, 3072, 32768, 12


def gemm_flops(m, k, n, batch=1):
    return 2.0 * m * k * n * batch


# Forward GEMMs of one step: c_q, c_k, c_v, attn.c_proj (48 x 768x768), mlp.c_fc / mlp.c_proj (24 x 768x3072),
# lm_head (768x32768). ve_gate (K=12) and smear_gate (K=24) GEMMs are < 1e-5 of the total and omitted.
GEMM_FWD = 4 * L * gemm_flops(N, C, C) + 2 * L * gemm_flops(N, C, F) + gemm_flops(N, C, V)
GEMM_BWD = 2 * GEMM_FWD  # dX and dW for every Linear
# Muon Polar-Express NS: 5 iterations x 3 bmm per group. Square group (48,768,768); c_fc (tall) and c_proj (wide)
# groups (12 matrices of 3072x768): X^T X, A@A, X@B. (inventory: 652 + 489 + 489 = 1,631 GFLOP/step)
MUON_NS = 5 * (3 * gemm_flops(C, C, C, 48)
               + 2 * (gemm_flops(C, F, C, 12) + gemm_flops(C, C, C, 12) + gemm_flops(F, C, C, 12)))

# (a) cases drawn on the roofline: (case suffix without form prefix, short label). Everything else (fwd+bwd, SDPA) is
# only in the CSV. GEMMs are drawn as one cluster with one label.
A_PLOT = [
    ("attn_fa3_fwd_16x2048x6x128", "FA3 attention fwd"),
    ("lmhead_ce_fwd_32768x768x32768", "lm_head + softcap/CE fwd\n(bytes: logits never stored)"),
    ("ce_fwd_32768x32768", "softcap + CE fwd"),
    ("qkrope_fwd_16x2048x6x128", "QK-norm + RoPE fwd"),
    ("rmsnorm_fwd_32768x768", "RMSNorm 768 fwd\n(eager = compile)"),
    ("rmsnorm_fwd_196608x128", "QK-norm (128-wide) fwd"),
    ("mixnorm_fwd_32768x768", "λ-mix +\nRMSNorm fwd"),
    ("relu2_fwd_32768x3072", "relu² fwd"),
]
# label placement, tuned by eye for the v0 figure. Memory-side (a) pairs: (text x, text y in FLOP/s, ha, anchor point
# "e"/"c" = eager/compile marker). Compute-side labels (a GEMM cluster, a FA3, all (b) diamonds) sit in one column in
# the empty area above the bandwidth roof, ordered by the target's y so the leaders do not cross.
A_LABEL_POS = {
    "relu2_fwd_32768x3072": (0.32, 4.5e12, "right", "c"),
    "rmsnorm_fwd_32768x768": (0.82, 0.17e12, "right", "e"),
    "mixnorm_fwd_32768x768": (0.62, 12e12, "right", "c"),
    "rmsnorm_fwd_196608x128": (1.35, 0.15e12, "left", "e"),
    "qkrope_fwd_16x2048x6x128": (3.0, 0.26e12, "left", "e"),
    "ce_fwd_32768x32768": (9.0, 0.55e12, "left", "e"),
    "lmhead_ce_fwd_32768x768x32768": (4.5e4, 20e12, "right", "e"),
}
COMPUTE_COL_Y = [1.25e15, 8.3e14, 5.5e14, 3.65e14, 2.45e14, 1.63e14, 1.09e14]

# (b) memory-bound regions shown as bandwidth bars (BAS-1 findings table rows); value = list of region keys summed,
# min-bytes key for the ratio
B_BARS = [
    ("λ-mix + RMSNorm fwd", ["mixnorm_fwd"], "mixnorm_fwd"),
    ("λ-mix + RMSNorm bwd", ["mixnorm_bwd"], "mixnorm_bwd"),
    ("QK-norm + RoPE fwd", ["qkrope_fwd"], "qkrope_fwd"),
    ("QK-norm + RoPE bwd", ["qkrope_bwd"], "qkrope_bwd"),
    ("relu² fwd", ["relu2_fwd"], "relu2_fwd"),
    ("relu² bwd", ["relu2_bwd"], "relu2_bwd"),
    ("softcap + CE fwd", ["ce_fwd"], "ce_fwd"),
    ("softcap + CE bwd", ["ce_bwd"], "ce_bwd"),
    ("AdamW", ["optim_adamw"], "optim_adamw"),
    ("Muon pointwise\n(pre/post, NS, stack)", ["optim_muon", "optim_muon_ns", "optim_muon_stack"], "optim_muon"),
]


def load_ops():
    ops = {}
    for p in sorted(glob.glob(os.path.join(OPS_DIR, "*.json"))):  # top level only: by_job/ is ignored
        with open(p) as f:
            d = json.load(f)
        ops[d["case"]] = d
    return ops


def load_regions():
    with open(REGIONS_JSON) as f:
        return json.load(f)


def b_flops(ops):
    fa3_fwd = ops["eager_attn_fa3_fwd_16x2048x6x128"]["flops"]            # causal-effective, per layer
    fa3_bwd = ops["eager_attn_fa3_fwdbwd_16x2048x6x128"]["flops"] - fa3_fwd  # 2.5x fwd (S recomputed)
    return {
        "gemm_fwd": ("cuBLAS GEMMs fwd (step)", GEMM_FWD),
        "gemm_bwd": ("cuBLAS GEMMs bwd (step)", GEMM_BWD),
        "attention_fwd": ("FA3 fwd (12 layers)", L * fa3_fwd),
        "attention_bwd": ("FA3 bwd (12 layers)", L * fa3_bwd),
        "optim_muon_ns_gemm": ("Muon NS bmm", MUON_NS),
    }


def build_rows(ops, reg):
    rows = []
    for case, d in sorted(ops.items()):
        t = d["median_us"] * 1e-6
        rows.append(dict(name=case.split("_", 1)[1], form=d["form"], source="a_isolated_op", flops=d["flops"],
                         bytes=d["min_bytes"], time_us=d["median_us"], intensity=d["flops"] / d["min_bytes"],
                         achieved_tflops=d["flops"] / t / 1e12, achieved_gbps=d["min_bytes"] / t / 1e9,
                         job_id=d.get("job_id", OPS_JOB), bytes_kind="analytic_min",
                         note=("flops = approximate op count" if d["kernel_id"] not in ("gemm", "attention", "ce_chunked")
                               else "analytic flops")))
    bf = b_flops(ops)
    for key, r in reg["regions"].items():
        t = r["us"] * 1e-6
        fl = bf.get(key, (None, None))[1]
        rows.append(dict(name=key, form="compile", source="b_in_model_ncu", flops=fl if fl else "",
                         bytes=r["bytes"], time_us=r["us"], intensity=fl / r["bytes"] if fl else "",
                         achieved_tflops=fl / t / 1e12 if fl else "", achieved_gbps=r["bytes"] / t / 1e9,
                         job_id=reg.get("job_id", NCU_JOB), bytes_kind="ncu_dram_per_step",
                         note=("analytic flops from R12 shapes" if fl else
                               "memory-bound region: no flops assigned; see bandwidth panel")
                              + (f"; min_bytes {r['min_bytes']:.0f}" if r.get("min_bytes") else "")))
    return rows


def write_csv(rows, path):
    cols = ["name", "form", "source", "flops", "bytes", "time_us", "intensity", "achieved_tflops", "achieved_gbps",
            "job_id", "bytes_kind", "note"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", path)


def draw_ceilings(ax, xlo, xhi):
    import numpy as np
    x = np.logspace(np.log10(xlo), np.log10(xhi), 400)
    # shading: memory-bound left of the measured ridge, compute-bound right of it
    ax.axvspan(xlo, RIDGE, color=TINT_MEM, zorder=0, lw=0)
    ax.axvspan(RIDGE, xhi, color=TINT_CMP, zorder=0, lw=0)
    ax.text(xlo * 1.25, 2.3e15, "memory-bound", fontsize=9, color=INK2, fontweight="bold", va="top")
    ax.text(RIDGE * 1.12, 2.3e15, "compute-bound", fontsize=9, color=INK2, fontweight="bold", va="top")
    ax.text(RIDGE * 1.12, 1.62e15, f"ridge {RIDGE:.0f} FLOP/B = 723 / 3.65", fontsize=7.5, color=INK2, va="top")
    ax.axvline(RIDGE, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=2)
    # datasheet roofs, dashed
    ax.plot(x, np.minimum(HBM_SHEET * x, BF16_SHEET), color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    m = x * HBM_SHEET >= FP32_SHEET
    ax.plot(x[m], [FP32_SHEET] * int(m.sum()), color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    # measured roofs
    ax.plot(x, np.minimum(HBM_MEAS * x, BF16_BURST), color=INK, lw=1.8, zorder=3)
    m = x * HBM_MEAS >= FP32_MEAS
    ax.plot(x[m], [FP32_MEAS] * int(m.sum()), color=INK2, lw=1.1, zorder=3)
    m = x * HBM_MEAS >= BF16_SUSTAINED
    ax.plot(x[m], [BF16_SUSTAINED] * int(m.sum()), color=INK2, lw=0.7, zorder=3)
    # ceiling labels: bf16 at the right edge (clear of the x <= 750 cluster), fp32 at the left end of its line
    xr = 1.3e3
    ax.text(xr, BF16_SHEET * 1.07, "bf16 990 TF/s (datasheet)", fontsize=7.3, color=MUTED, va="bottom")
    ax.text(xr, BF16_BURST * 1.06, "bf16 723 TF/s measured (burst)", fontsize=7.3, color=INK, va="bottom")
    ax.text(xr, BF16_SUSTAINED / 1.07, "591 TF/s sustained (power cap)", fontsize=7.3, color=INK2, va="top")
    x32 = 60.0
    ax.text(x32, FP32_SHEET * 1.07, "fp32 67 TF/s (datasheet)", fontsize=7.3, color=MUTED, va="bottom")
    ax.text(x32, FP32_MEAS / 1.08, "fp32 (non-tensor) 48.6 TF/s measured", fontsize=7.3, color=INK2, va="top")
    # bandwidth labels along the slopes (rotation set after layout)
    ax.text(2.2, HBM_SHEET * 2.2 * 1.25, "HBM 4.0 TB/s (datasheet)", fontsize=7.3, color=MUTED, ha="left",
            va="bottom", rotation_mode="anchor", gid="slope")
    ax.text(2.4, HBM_MEAS * 2.4 / 1.3, "HBM 3.65 TB/s measured (copy)", fontsize=7.5, color=INK, ha="left",
            va="top", rotation_mode="anchor", gid="slope")


def slope_rotation(ax, fig):
    """Screen angle of a slope-1 line on the current log-log axes (degrees)."""
    import numpy as np
    fig.canvas.draw()
    p1 = ax.transData.transform((1.0, 1e12))
    p2 = ax.transData.transform((10.0, 1e13))
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))


def tf(fl, us):
    return fl / (us * 1e-6)


def fig_roofline(ops, reg, out):
    fig = plt.figure(figsize=(15.5, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1], wspace=0.36)
    ax = fig.add_subplot(gs[0, 0])
    xlo, xhi = 0.1, 6e4
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(1e11, 2.5e15)
    draw_ceilings(ax, xlo, xhi)

    def leader(text, xy, xytext, ha, color=INK2, size=7.5):
        ax.annotate(text, xy, xytext=xytext, fontsize=size, color=color, ha=ha, va="center", zorder=7,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=2, shrinkB=3))

    # (a) isolated ops: eager and compile at the same analytic x, joined by a hairline
    gemms = sorted(c for c in ops if c.startswith("eager_gemm"))
    gpts = []
    for c in gemms:
        e, k = ops[c], ops["compile_" + c.split("_", 1)[1]]
        x = e["flops"] / e["min_bytes"]
        gpts.append((x, tf(e["flops"], e["median_us"]), tf(k["flops"], k["median_us"])))
        ax.scatter([x], [gpts[-1][1]], s=24, color=EAGER, edgecolor=SURFACE, linewidth=0.7, zorder=5)
        ax.scatter([x], [gpts[-1][2]], s=24, color=COMPILE, edgecolor=SURFACE, linewidth=0.7, zorder=5)
    a_pts = {}
    for suffix, label in A_PLOT:
        e, k = ops["eager_" + suffix], ops["compile_" + suffix]
        x = e["flops"] / e["min_bytes"]
        ye, yc = tf(e["flops"], e["median_us"]), tf(k["flops"], k["median_us"])
        a_pts[suffix] = (x, ye, yc)
        ax.plot([x, x], [ye, yc], color=AXIS, lw=1.0, zorder=4)
        ax.scatter([x], [ye], s=34, color=EAGER, edgecolor=SURFACE, linewidth=0.8, zorder=5)
        ax.scatter([x], [yc], s=34, color=COMPILE, edgecolor=SURFACE, linewidth=0.8, zorder=5)
        if suffix in A_LABEL_POS:
            tx, ty, ha, which = A_LABEL_POS[suffix]
            leader(label, (x, ye if which == "e" else yc), (tx, ty), ha)

    # (b) in-model compiled regions with analytic FLOPs and ncu DRAM bytes
    bf = b_flops(ops)
    col = []  # (target y, text, xy) for the compute-side label column
    for key, (label, fl) in bf.items():
        r = reg["regions"][key]
        x, y = fl / r["bytes"], tf(fl, r["us"])
        ax.scatter([x], [y], s=48, marker="D", color=COMPILE, edgecolor=INK, linewidth=0.9, zorder=6)
        col.append((y, f"{label} (b): {y / 1e12:.0f} TF/s", (x, y)))
    gx, gtop = min(p[0] for p in gpts), max(max(p[1], p[2]) for p in gpts)
    gx_at = [p for p in gpts if p[0] == gx][0]
    col.append((gtop * 0.97, f"cuBLAS GEMMs (a), {len(gemms)} shapes: {min(min(p[1], p[2]) for p in gpts) / 1e12:.0f}"
                f"–{gtop / 1e12:.0f} TF/s, eager ≈ compile", (gx, max(gx_at[1], gx_at[2]))))
    xa, ye, yc = a_pts["attn_fa3_fwd_16x2048x6x128"]
    col.append((ye * 0.99, f"FA3 attention fwd (a): {ye / 1e12:.0f} TF/s", (xa, ye)))
    col.sort(key=lambda t: -t[0])
    for (yt, text, xy), ty in zip(col, COMPUTE_COL_Y):
        tx = ty / HBM_MEAS / 2.2  # right edge of the text sits left of the bandwidth roof at that height
        leader(text, xy, (tx, ty), "right", color=INK)

    ang = slope_rotation(ax, fig)
    for t in ax.texts:
        if t.get_gid() == "slope":
            t.set_rotation(ang)

    ax.set_xticks([0.1, 1, 10, 100, 1e3, 1e4], ["0.1", "1", "10", "100", "1k", "10k"])
    ax.set_yticks([1e11, 1e12, 1e13, 1e14, 1e15], ["0.1", "1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xlabel("arithmetic intensity (FLOP per DRAM byte, log scale)")
    ax.set_ylabel("achieved TFLOP/s (log scale)")
    ax.set_title("Roofline of nanochat R12 kernel classes on one GH200: eager vs torch.compile")
    handles = [
        Line2D([], [], marker="o", ls="", color=EAGER, markeredgecolor=SURFACE, markersize=7,
               label="eager, isolated op (a)"),
        Line2D([], [], marker="o", ls="", color=COMPILE, markeredgecolor=SURFACE, markersize=7,
               label="torch.compile, isolated op (a)"),
        Line2D([], [], marker="D", ls="", color=COMPILE, markeredgecolor=INK, markersize=6.5,
               label="torch.compile, in-model step region (b)"),
        Line2D([], [], color=INK, lw=1.8, label="measured roof"),
        Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 3)), label="datasheet roof"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7.8)

    # second panel: memory-bound in-model regions as achieved bandwidth (no FLOPs invented)
    bx = fig.add_subplot(gs[0, 1])
    R = reg["regions"]
    labels, vals, notes = [], [], []
    for label, keys, minkey in B_BARS:
        us = sum(R[k]["us"] for k in keys)
        by = sum(R[k]["bytes"] for k in keys)
        mb = R[minkey].get("min_bytes")
        labels.append(label)
        vals.append(by / (us * 1e-6) / 1e12)
        notes.append(f"{by / mb:.2f}× min bytes\n{us / 1e3:.2f} ms/step" if mb else f"{us / 1e3:.2f} ms/step")
    y = list(range(len(labels)))[::-1]
    bx.barh(y, vals, height=0.62, color=COMPILE, edgecolor=SURFACE, linewidth=2)
    for yi, v, n in zip(y, vals, notes):
        bx.text(0.07, yi, f"{v:.2f}", va="center", ha="left", fontsize=7.8, color=SURFACE, fontweight="bold")
        bx.text(4.15, yi, n, va="center", ha="left", fontsize=7.2, color=INK2, linespacing=1.05)
    bx.axvline(HBM_MEAS / 1e12, color=INK, lw=1.4)
    bx.axvline(HBM_SHEET / 1e12, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    bx.text(HBM_MEAS / 1e12 - 0.04, len(labels) - 0.45, "3.65 measured", fontsize=7.3, color=INK, ha="right")
    bx.text(HBM_SHEET / 1e12 + 0.04, len(labels) - 0.45, "4.0", fontsize=7.3, color=MUTED, ha="left")
    bx.set_yticks(y, labels, fontsize=8.3)
    bx.set_xlim(0, 5.4)
    bx.set_ylim(-0.6, len(labels) - 0.1)
    bx.set_xticks([0, 1, 2, 3, 4])
    bx.set_xlabel("achieved DRAM bandwidth, TB/s (ncu bytes / ncu time)")
    bx.grid(axis="x", color=GRID, linewidth=0.6)
    bx.set_axisbelow(True)
    bx.tick_params(length=0)
    bx.set_title("Memory-bound in-model regions (b), torch.compile")

    fig.text(0.0, -0.02,
             "(a) kbench isolated ops at R12 shapes (job 3406329): median of 100 cold-L2 reps × 3 trials, default clocks; "
             "x = analytic FLOPs / minimum bytes (each input read once, each output written once), y = FLOPs / median time. "
             "For memory-bound\nops the FLOP count is an approximate op count, so their x position is approximate; the "
             "vertical gap below the bandwidth roof is what matters. Eager and compile share x (hairline). Forward cases "
             "shown; fwd+bwd and SDPA are in results/roofline_points_v0.csv.\n"
             "(b) ncu of every kernel of one compiled training step (job 3406171; --clock-control none, L2 flushed per "
             "kernel): x = analytic FLOPs from R12 shapes / measured DRAM bytes, y = FLOPs / ncu time; ncu times are "
             "bandwidth facts, not headline times.\n"
             "Ceilings (PLAN §2, §7.4; job 3406282): copy 3.65 TB/s; 8192³ GEMM bf16 723 TF/s burst at 1980 MHz, 591 TF/s "
             "sustained at the 1365 MHz power cap; fp32 48.6 TF/s.",
             fontsize=7.4, color=MUTED, va="top", linespacing=1.35)
    fig.text(1.0, -0.115, "jobs 3406329, 3406171; v0", fontsize=8, color=INK2, ha="right", va="top")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--version", default="v0", choices=["v0"], help="v1 (ncu bytes for (a)) arrives with BAS-6")
    ap.add_argument("--out", default=None, help="PNG path (default results/figures/roofline_<version>.png)")
    ap.add_argument("--csv", default=None, help="CSV path (default results/roofline_points_<version>.csv)")
    a = ap.parse_args()
    out = a.out or os.path.join(RESULTS, "figures", f"roofline_{a.version}.png")
    csv_path = a.csv or os.path.join(RESULTS, f"roofline_points_{a.version}.csv")
    ops, reg = load_ops(), load_regions()
    write_csv(build_rows(ops, reg), csv_path)
    fig_roofline(ops, reg, out)


if __name__ == "__main__":
    main()
