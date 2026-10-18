"""
Summarize a Session-1 style run directory: base_train timing logs, the convergence log,
infer_bench / decode_bench JSON lines, and the GPU monitor CSV.

Writes <run_dir>/summary.json and <run_dir>/summary.md, prints the markdown.

Usage: python -m cisc662.analysis.summarize_s1 cisc662/results/runs/s1_baseline_<jobid>
"""
import csv
import glob
import json
import os
import re
import statistics
import sys

STEP_RE = re.compile(
    r"step (\d+)/(\d+) .*?\| loss: ([\d.]+) .*?\| dt: ([\d.]+)ms \| tok/sec: ([\d,]+) \| bf16_mfu: ([\d.]+)")
VAL_RE = re.compile(r"Step (\d+) \| Validation bpb: ([\d.]+)")
PEAK_RE = re.compile(r"Peak memory usage: ([\d.]+)MiB")
BATCH_RE = re.compile(r"Total batch size ([\d,]+)")
TIMED_FIRST, TIMED_LAST = 10, 109  # "seconds per 100 training steps" = steps 10..109 inclusive


def parse_train_log(path):
    """Returns (steps, val_bpb list, peak MiB). Step tok/s is recomputed from the logged total batch size and dt,
    because base_train prints tok/sec rounded to an integer (too coarse for slow CPU runs)."""
    steps, vals, peak, batch = [], [], None, None
    with open(path, errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(dict(step=int(m.group(1)), loss=float(m.group(3)), dt_ms=float(m.group(4)),
                                  tok_per_sec=int(m.group(5).replace(",", "")), mfu=float(m.group(6))))
                continue
            m = VAL_RE.search(line)
            if m:
                vals.append(dict(step=int(m.group(1)), val_bpb=float(m.group(2))))
                continue
            m = PEAK_RE.search(line)
            if m:
                peak = float(m.group(1))
                continue
            m = BATCH_RE.search(line)
            if m:
                batch = int(m.group(1).replace(",", ""))
    if batch:
        for st in steps:
            st["tok_per_sec"] = batch / (st["dt_ms"] / 1e3)
    return steps, vals, peak


def timing_stats(steps):
    timed = [s for s in steps if TIMED_FIRST <= s["step"] <= TIMED_LAST]
    if len(timed) < TIMED_LAST - TIMED_FIRST + 1:
        return None
    dts = [s["dt_ms"] for s in timed]
    return dict(
        s_per_100_steps=sum(dts) / 1000.0,
        median_step_ms=statistics.median(dts),
        p10_step_ms=sorted(dts)[len(dts) // 10],
        p90_step_ms=sorted(dts)[9 * len(dts) // 10],
        mean_tok_per_sec=statistics.mean(s["tok_per_sec"] for s in timed),
        mean_mfu_pct=statistics.mean(s["mfu"] for s in timed),
    )


def last_json_line(path):
    with open(path, errors="replace") as f:
        lines = [l.strip() for l in f if l.strip().startswith("{")]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def mean_std(xs):
    return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main(run_dir):
    summary = {"run_dir": run_dir, "train": {}, "conv": None, "infer": {}, "decode": {}, "gpu_monitor": None}

    # training timing runs: train_<mode>_<attn>_r<rep>.log
    groups = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "train_*_r*.log"))):
        name = os.path.basename(path)[len("train_"):-len(".log")]
        key, rep = name.rsplit("_r", 1)
        steps, _, peak = parse_train_log(path)
        st = timing_stats(steps)
        if st is None:
            print(f"warning: {path} incomplete ({len(steps)} steps)")
            continue
        st["peak_mem_MiB"] = peak
        groups.setdefault(key, []).append(st)
    for key, reps in groups.items():
        s100 = mean_std([r["s_per_100_steps"] for r in reps])
        summary["train"][key] = dict(
            reps=len(reps), s_per_100_mean=s100[0], s_per_100_std=s100[1],
            median_step_ms=statistics.mean(r["median_step_ms"] for r in reps),
            tok_per_sec=statistics.mean(r["mean_tok_per_sec"] for r in reps),
            mfu_pct=statistics.mean(r["mean_mfu_pct"] for r in reps),
            peak_mem_MiB=max((r["peak_mem_MiB"] or 0) for r in reps), per_rep=reps)

    # CPU reference runs (cpu_t<threads>.log): only a few steps, so report the median step (excluding step 0)
    # and the extrapolated seconds per 100 steps
    summary["cpu"] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "cpu_*.log"))):
        steps, _, _ = parse_train_log(path)
        timed = [s for s in steps if s["step"] >= 1]
        if not timed:
            print(f"warning: {path} has no timed steps")
            continue
        med = statistics.median(s["dt_ms"] for s in timed)
        summary["cpu"][os.path.basename(path)[:-4]] = dict(
            timed_steps=len(timed), median_step_ms=med, s_per_100_extrapolated=med * 100 / 1000.0,
            tok_per_sec=statistics.median(s["tok_per_sec"] for s in timed))

    # convergence run
    conv = glob.glob(os.path.join(run_dir, "conv_*.log"))
    if conv:
        steps, vals, peak = parse_train_log(conv[0])
        summary["conv"] = dict(log=os.path.basename(conv[0]), steps=len(steps),
                               final_loss=steps[-1]["loss"] if steps else None, val_bpb=vals, peak_mem_MiB=peak,
                               loss_curve=[(s["step"], s["loss"]) for s in steps])

    # inference benchmark JSON (last line of each log)
    for path in sorted(glob.glob(os.path.join(run_dir, "infer_*_r*.log"))):
        name = os.path.basename(path)[len("infer_"):-len(".log")]
        key, rep = name.rsplit("_r", 1)
        js = last_json_line(path)
        if js:
            summary["infer"].setdefault(key, []).append(js)
    for path in sorted(glob.glob(os.path.join(run_dir, "decode_*.log"))):
        js = last_json_line(path)
        if js:
            summary["decode"][os.path.basename(path)[:-4]] = js

    # GPU monitor
    mon = os.path.join(run_dir, "gpu_monitor.csv")
    if os.path.exists(mon):
        with open(mon) as f:
            rows = list(csv.reader(f))
        try:
            body = rows[1:]
            sm = [float(r[1].split()[0]) for r in body if len(r) > 6]
            pw = [float(r[3].split()[0]) for r in body if len(r) > 6]
            ut = [float(r[5].split()[0]) for r in body if len(r) > 6]
            busy = [i for i, u in enumerate(ut) if u >= 50]
            summary["gpu_monitor"] = dict(
                samples=len(sm), sm_clock_MHz_median_when_busy=statistics.median([sm[i] for i in busy]) if busy else None,
                power_W_median_when_busy=statistics.median([pw[i] for i in busy]) if busy else None,
                frac_samples_util_ge_50=len(busy) / max(1, len(ut)))
        except (ValueError, IndexError) as e:
            summary["gpu_monitor"] = f"unparsed: {e}"

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # markdown
    md = ["## Training: seconds per 100 steps (R12, steps 10-109)", "",
          "| mode | attention | reps | s/100 steps | median step (ms) | tok/s | MFU % | peak mem (GiB) |",
          "|---|---|---|---|---|---|---|---|"]
    for key in sorted(summary["train"]):
        t = summary["train"][key]
        mode, attn = key.split("_", 1)
        md.append(f"| {mode} | {attn} | {t['reps']} | {t['s_per_100_mean']:.3f} ± {t['s_per_100_std']:.3f} | "
                  f"{t['median_step_ms']:.2f} | {t['tok_per_sec']:,.0f} | {t['mfu_pct']:.1f} | {t['peak_mem_MiB']/1024:.1f} |")
    if summary["cpu"]:
        md += ["", "## CPU reference (fp32 eager, batch 1 x 2048)", "",
               "| run | timed steps | median step (s) | extrapolated s/100 steps | tok/s |", "|---|---|---|---|---|"]
        for key, c in sorted(summary["cpu"].items()):
            md.append(f"| {key} | {c['timed_steps']} | {c['median_step_ms']/1e3:.2f} | {c['s_per_100_extrapolated']:.1f} | "
                      f"{c['tok_per_sec']:,.0f} |")
    if summary["decode"]:
        md += ["", "## Decode latency (decode_bench, batch 1)", "", "| run | device | threads | TTFT ms | TPOT ms | tok/s |",
               "|---|---|---|---|---|---|"]
        for key, d in sorted(summary["decode"].items()):
            for row in d["sweep"]:
                md.append(f"| {key} | {d['device']} | {d['threads']} | {row['ttft_sec']*1e3:.1f} | "
                          f"{row['tpot_sec']*1e3:.2f} | {row['tok_per_sec']:,.1f} |")
    if summary["conv"]:
        c = summary["conv"]
        md += ["", f"Convergence run: {c['steps']} steps, final smoothed train loss {c['final_loss']}, "
                   f"val bpb {[(v['step'], v['val_bpb']) for v in c['val_bpb']]}"]
    if summary["infer"]:
        md += ["", "## Inference (infer_bench: prompt 512, decode 256)", "",
               "| attention | rep | prefill tok/s | batch | TPOT ms | tok/s | MBU % |", "|---|---|---|---|---|---|---|"]
        for key, reps in sorted(summary["infer"].items()):
            for i, js in enumerate(reps):
                for row in js["sweep"]:
                    md.append(f"| {key} | {i+1} | {js['prefill']['tok_per_sec']:,.0f} | {row['batch_size']} | "
                              f"{row['tpot_sec']*1e3:.2f} | {row['tok_per_sec']:,.0f} | {row['mbu_percent']:.1f} |")
    if summary["gpu_monitor"]:
        md += ["", f"GPU monitor: {summary['gpu_monitor']}"]
    text = "\n".join(md)
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
