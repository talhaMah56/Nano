"""
ANA-2: figures for Presentation 1 (slides 5 and 6). Reads only results/db (gated rows) -- no GPU.
  p1_kernel_speedups.png   ours vs eager and vs torch.compile, fwd+bwd, per Tier-A kernel (CUDA events, cold L2)
  p1_ce_bounds.png         fused softcap+CE forward: HBM bound vs MUFU bound vs measured (compile, ours)
  p1_ce_memory.png         peak memory above the inputs for softcap+CE fwd+bwd
  p1_bandwidth.png         achieved DRAM bandwidth of our kernels (ncu, warm launch) as % of the 3.65 TB/s measured roof
Usage: python -m cisc662.analysis.make_figs_p1
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB = os.path.join(PKG, "results", "db")
OUT = os.path.join(PKG, "results", "figures")
HBM_TBPS = 3.65
C_EAGER, C_COMPILE, C_OURS, C_BOUND = "#8a8f98", "#d98c2b", "#2a7de1", "#444444"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})


def med(group, case, need_gates=False):
    r = json.load(open(os.path.join(DB, group, case + ".json")))
    if need_gates and not {"G1", "G2"} <= set(r.get("gates_passed") or []):
        raise SystemExit(f"{case}: gates G1/G2 not recorded -- refusing to plot")
    return r["median_us"], r


def fig_speedups():
    rows = [  # label, group, ours case, eager case, compile case
        ("RMSNorm 768-wide\n(32768 rows)", "rmsnorm", "triton_v2_rmsnorm_fwdbwd_32768x768",
         "eager_rmsnorm_fwdbwd_32768x768", "compile_rmsnorm_fwdbwd_32768x768"),
        ("RMSNorm 128-wide\n(QK, 196608 rows)", "rmsnorm", "triton_v2_rmsnorm_fwdbwd_196608x128",
         "eager_rmsnorm_fwdbwd_196608x128", "compile_rmsnorm_fwdbwd_196608x128"),
        ("Residual mix\n+ RMSNorm", "rmsnorm", "triton_v3_mixnorm_fwdbwd_32768x768",
         "eager_mixnorm_fwdbwd_32768x768", "compile_mixnorm_fwdbwd_32768x768"),
        ("QK-norm + RoPE", "qkrope", "ours_qkrope_fwdbwd_16x2048x6x128",
         "eager_qkrope_fwdbwd_16x2048x6x128", "compile_qkrope_fwdbwd_16x2048x6x128"),
        ("Softcap +\ncross-entropy", "ce", "ours_ce_fwdbwd_32768x32768",
         "eager_ce_fwdbwd_32768x32768", "compile_ce_fwdbwd_32768x32768"),
    ]
    labels, ve, vc = [], [], []
    for lab, g, o, e, c in rows:
        mo, _ = med(g, o, True)
        labels.append(lab)
        ve.append(med(g, e)[0] / mo)
        vc.append(med(g, c)[0] / mo)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    n = len(rows)
    ypos = list(range(n))[::-1]
    h = 0.38
    b1 = ax.barh([y + h / 2 for y in ypos], ve, h, color=C_EAGER, label="vs eager PyTorch")
    b2 = ax.barh([y - h / 2 for y in ypos], vc, h, color=C_COMPILE, label="vs torch.compile (Inductor)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_width() * 1.04, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}x",
                    va="center", fontsize=11)
    ax.axvline(1.0, color=C_BOUND, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 40)
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1x", "2x", "5x", "10x", "20x"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("speedup of our Triton kernel, forward + backward (higher is better; 1x = same speed)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Our fused kernels vs the two PyTorch baselines (GH200, R12 shapes)", loc="left", fontsize=14)
    fig.text(0.01, 0.005, "median of 3 process trials, CUDA events, L2 flushed; all plotted kernels passed gates G1/G2",
             fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(OUT, "p1_kernel_speedups.png"))
    plt.close(fig)
    return dict(zip([l.replace("\n", " ") for l in labels], zip(ve, vc)))


def fig_ce_bounds():
    ours, _ = med("ce", "ours_ce_fwd_32768x32768", True)
    comp, _ = med("ce", "compile_ce_fwd_32768x32768")
    items = [("HBM bound\n(2.15 GB at 3.65 TB/s)", 590, C_BOUND), ("MUFU bound\n(~3.3 MUFU/logit)", 950, C_BOUND),
             ("ours (fused)", ours, C_OURS), ("torch.compile", comp, C_COMPILE)]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xs = range(len(items))
    bars = ax.bar(xs, [v for _, v, _ in items], color=[c for _, _, c in items], width=0.6)
    for b, (_, v, _) in zip(bars, items):
        ax.text(b.get_x() + b.get_width() / 2, v + 25, f"{v:.0f} us", ha="center", fontsize=12)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l for l, _, _ in items], fontsize=11)
    ax.set_ylabel("forward time (us)")
    ax.set_ylim(0, max(v for _, v, _ in items) * 1.18)
    ax.set_title("Softcap + cross-entropy forward: limited by the special-function unit, not by memory",
                 loc="left", fontsize=12.5)
    fig.text(0.01, 0.005, "bounds from SASS instruction counts (33 EX2 + 20 RCP per 16 logits) and measured HBM bandwidth",
             fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(OUT, "p1_ce_bounds.png"))
    plt.close(fig)


def fig_ce_memory():
    m = json.load(open(os.path.join(DB, "ce", "peak_memory_3417307.json")))
    vals = [("eager", m["eager"]["peak_above_inputs_GB"], C_EAGER), ("torch.compile", m["compile"]["peak_above_inputs_GB"], C_COMPILE),
            ("ours", m["ours_rec"]["peak_above_inputs_GB"], C_OURS)]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    bars = ax.bar([v[0] for v in vals], [v[1] for v in vals], color=[v[2] for v in vals], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v[1] + 0.3, f"{v[1]:.1f} GB", ha="center", fontsize=12)
    ax.set_ylabel("peak memory above the inputs (GB)")
    ax.set_ylim(0, 20)
    ax.set_title("Softcap + CE forward+backward: 8x less memory than eager", loc="left", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "p1_ce_memory.png"))
    plt.close(fig)


def fig_bandwidth():
    path = sorted(glob.glob(os.path.join(DB, "ncu", "ours_*.json")))[-1]
    recs = json.load(open(path))
    keep = [r for r in recs if "gif" not in r["name"]]
    keep.sort(key=lambda r: r["warm"]["dram_GBps"])
    names = [r["name"].replace("c662_", "") for r in keep]
    pct = [100 * r["warm"]["dram_GBps"] / (HBM_TBPS * 1000) for r in keep]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.barh(names, pct, color=[C_OURS if p > 70 else C_COMPILE for p in pct], height=0.6)
    for b, p, r in zip(bars, pct, keep):
        ax.text(p + 1.5, b.get_y() + b.get_height() / 2,
                f"{p:.0f}%  ({r['warm']['dram_GBps'] / 1000:.2f} TB/s)", va="center", fontsize=11)
    ax.axvline(100, color=C_BOUND, lw=1.2)
    ax.set_xlim(0, 125)
    ax.set_xlabel("achieved DRAM bandwidth, % of the 3.65 TB/s measured copy roof (ncu)")
    ax.set_title("Our kernels at the memory roof, except cross-entropy forward", loc="left", fontsize=12.5)
    fig.text(0.01, 0.005, f"source: {os.path.basename(path)}, warm launch of each kernel", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(OUT, "p1_bandwidth.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(json.dumps(fig_speedups(), indent=1))
    fig_ce_bounds()
    fig_ce_memory()
    fig_bandwidth()
    print("wrote 4 figures to", OUT)
