"""
BAS-3 step 4: aggregate the decode re-measurement (decode_bench.py, batch 1, 256 decode tokens, 3 reps on the
CPU with 1 thread and 3 reps on the GPU, all in one job on one node) into results/db/decode/s2_remeasure.json:
per-rep tok/s, median and min-max per device, the GPU / CPU-1-thread ratio with its spread, node and clocks.

Input: the log of each decode_bench run (its last line is the JSON summary decode_bench prints; earlier
lines are per-batch dict reprs and nanochat log lines) and, optionally, one nvidia-smi CSV per GPU rep
(`--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu --format=csv` sampled during the rep).
A rep whose log has no JSON summary (crash / timeout) is listed under `failed_reps` and left out of the stats.
Output: --out (the latest result) plus <stem>_<jobid>.json; a latest file from another job is archived first.

Run (login node, CPU only; slurm/s2_harness.sbatch runs it at the end of the job):
  python -m cisc662.analysis.decode_remeasure --job-id 1234 --node gh047 \
      --cpu-logs $OUT/decode/cpu_r*.log --gpu-logs $OUT/decode/gpu_r*.log \
      --gpu-clock-csvs $OUT/decode/gpu_r*.smi.csv --out cisc662/results/db/decode/s2_remeasure.json
"""
import argparse
import csv
import json
import os
import re
import statistics

# S1 values (results/runs/s1_cpuref_3400725/summary.md, s1_conv_infer_3400723): context only, other settings
# (S1 CPU used 16 decode tokens, GPU 32; here 256, so the KV cache is longer and tok/s is not directly comparable).
S1_REFERENCE = {"cpu_1thread_tok_s": 32.8, "gpu_b1_tok_s_range": [68, 107],
                "note": "S1: prompt 512, decode 16 (CPU) / 32 (GPU) tokens; jobs 3400723, 3400725"}


# long header names nvidia-smi prints -> the short names we query with (see results/runs/s1_*/gpu_monitor.csv)
CSV_ALIASES = {"clocks.current.sm": "clocks.sm", "clocks.current.memory": "clocks.mem",
               "clocks.current.graphics": "clocks.gr"}


def parse_decode_log(path):
    """Last JSON line with a 'sweep' key from a decode_bench log, or None."""
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and '"sweep"' in line:
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def num(value):
    m = re.match(r"\s*(-?[0-9]+(?:\.[0-9]+)?)", str(value))
    return float(m.group(1)) if m else None


def clock_summary(path, busy_util=1.0):
    """Median SM clock / power over the samples of one rep where the GPU was busy (utilization >= busy_util)."""
    try:
        with open(path) as f:
            rows = [r for r in csv.reader((l for l in f if l.strip()), skipinitialspace=True)]
    except OSError:
        return None
    if len(rows) < 2:
        return {"csv": path, "n_samples": 0}
    # nvidia-smi writes the long field names in the header ("clocks.current.sm [MHz]" for clocks.sm)
    header = [re.sub(r"\s*\[.*?\]\s*$", "", h.strip()) for h in rows[0]]
    header = [CSV_ALIASES.get(h, h) for h in header]
    recs = [dict(zip(header, r)) for r in rows[1:] if len(r) == len(header) and r[0].strip() != header[0]]
    busy = [r for r in recs if "utilization.gpu" not in r or (num(r["utilization.gpu"]) or 0) >= busy_util]

    def med(key, rs):
        xs = [num(r.get(key)) for r in rs]
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else None
    return {"csv": path, "n_samples": len(recs), "n_busy": len(busy),
            "sm_mhz_median_busy": med("clocks.sm", busy), "mem_mhz_median_busy": med("clocks.mem", busy),
            "power_w_median_busy": med("power.draw", busy), "util_pct_median": med("utilization.gpu", recs)}


def stats(xs):
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs),
            "mean": statistics.fmean(xs), "cv": statistics.pstdev(xs) / statistics.fmean(xs) if len(xs) > 1 else 0.0}


def collect(logs, batch_size):
    """Per-rep rows (batch_size rows of each log's sweep) plus the logs that produced nothing."""
    reps, failed = [], []
    for i, path in enumerate(logs):
        j = parse_decode_log(path)
        rows = [r for r in (j or {}).get("sweep", []) if r.get("batch_size") == batch_size]
        if not rows:
            failed.append(path)
            continue
        for r in rows:
            reps.append({"rep": i + 1, "log": path, "device": j.get("device"), "threads": j.get("threads"),
                         "dtype": j.get("dtype"), "model_tag": j.get("model_tag"), "step": j.get("step"),
                         "prompt_tokens": j.get("prompt_tokens"), "decode_tokens": j.get("decode_tokens"),
                         "decode_steps": r.get("decode_steps"), "tok_per_sec": r.get("tok_per_sec"),
                         "tpot_ms": 1e3 * r["tpot_sec"] if r.get("tpot_sec") is not None else None,
                         "ttft_ms": 1e3 * r["ttft_sec"] if r.get("ttft_sec") is not None else None})
    return reps, failed


def device_block(reps, failed):
    tok = [r["tok_per_sec"] for r in reps if r["tok_per_sec"]]
    return {"reps": reps, "failed_reps": failed, "tok_per_sec": [round(x, 3) for x in tok],
            "tok_per_sec_stats": stats(tok),
            "tpot_ms_stats": stats([r["tpot_ms"] for r in reps if r["tpot_ms"] is not None]),
            "ttft_ms_stats": stats([r["ttft_ms"] for r in reps if r["ttft_ms"] is not None])}


def ratio_block(gpu, cpu):
    """GPU/CPU tok/s ratio: median/median, and the spread as the min and max over all rep pairs."""
    g, c = gpu["tok_per_sec"], cpu["tok_per_sec"]
    if not g or not c:
        return None
    pairs = [x / y for x in g for y in c]
    return {"median_over_median": statistics.median(g) / statistics.median(c),
            "pairwise_min": min(pairs), "pairwise_max": max(pairs), "n_pairs": len(pairs),
            "definition": "GPU tok/s / CPU-1-thread tok/s at batch 1; spread = min/max over all rep pairs"}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cpu-logs", nargs="*", default=[])
    p.add_argument("--gpu-logs", nargs="*", default=[])
    p.add_argument("--gpu-clock-csvs", nargs="*", default=[], help="one nvidia-smi CSV per GPU rep (optional)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--job-id", required=True)
    p.add_argument("--node", default=None)
    p.add_argument("--cpu-bind", default=None, help="how the CPU reps were pinned (e.g. 'numactl --physcpubind=3')")
    p.add_argument("--platform-json", default=None, help="results/db/platform/clocks_<jobid>.json (optional)")
    p.add_argument("--out", default="cisc662/results/db/decode/s2_remeasure.json")
    args = p.parse_args()

    cpu = device_block(*collect(sorted(args.cpu_logs), args.batch_size))
    gpu = device_block(*collect(sorted(args.gpu_logs), args.batch_size))
    gpu["clocks_per_rep"] = [clock_summary(c) for c in sorted(args.gpu_clock_csvs)]
    sm = [c["sm_mhz_median_busy"] for c in gpu["clocks_per_rep"] if c and c.get("sm_mhz_median_busy")]
    gpu["sm_mhz_median_over_reps"] = statistics.median(sm) if sm else None
    warnings = []
    bad_threads = sorted({r["threads"] for r in cpu["reps"] if r["threads"] != 1})
    if bad_threads:
        warnings.append(f"CPU reps ran with torch threads {bad_threads}, expected 1")
    for name, blk in (("cpu", cpu), ("gpu", gpu)):
        if blk["failed_reps"]:
            warnings.append(f"{name}: {len(blk['failed_reps'])} rep(s) without a JSON summary")
    platform_clocks = None
    if args.platform_json and os.path.exists(args.platform_json):
        with open(args.platform_json) as f:
            pj = json.load(f)
        platform_clocks = {k: pj.get(k) for k in ("sustained_sm_mhz", "power_w", "gpu_freq_pinnable")}

    out = {
        "job_id": args.job_id,
        "node": args.node,
        "batch_size": args.batch_size,
        "cpu_bind": args.cpu_bind,
        "cpu_1thread": cpu,
        "gpu": gpu,
        "gpu_over_cpu_1thread": ratio_block(gpu, cpu),
        "platform_clocks": platform_clocks,
        "s1_reference": S1_REFERENCE,
        "warnings": warnings,
    }
    # --out is the "latest" copy; every job also gets <stem>_<jobid>.json, and a latest copy from another job is
    # archived under its own job id before it is replaced, so no job's result is ever overwritten.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    stem = args.out[:-5] if args.out.endswith(".json") else args.out
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                old_job = str(json.load(f).get("job_id"))
        except (OSError, ValueError):
            old_job = "unknown"
        if old_job != str(args.job_id) and not os.path.exists(f"{stem}_{old_job}.json"):
            os.replace(args.out, f"{stem}_{old_job}.json")
    for path in (f"{stem}_{args.job_id}.json", args.out):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, path)
    r = out["gpu_over_cpu_1thread"]
    print(f"decode b={args.batch_size}: CPU-1t {cpu['tok_per_sec']} tok/s, GPU {gpu['tok_per_sec']} tok/s, "
          + (f"ratio {r['median_over_median']:.2f} ({r['pairwise_min']:.2f}-{r['pairwise_max']:.2f})" if r
             else "ratio n/a") + (f"; WARN {warnings}" if warnings else "") + f" -> {args.out}")


if __name__ == "__main__":
    main()
