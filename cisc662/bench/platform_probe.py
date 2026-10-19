"""
BAS-3 platform probe: sustained clocks/power/throttle reasons, pointer-chase memory latency, and the merge of
both with the `srun --gpu-freq=1755` job-step result into results/db/platform/clocks_<jobid>.json.

Subcommands
  clock    bf16 8192^3 GEMM loop for --seconds while a background thread samples
           `nvidia-smi -q -d CLOCK,POWER,PERFORMANCE` (+ one --query-gpu line) every --interval-s seconds.
           Records the sustained SM clock (median over busy samples after a short ramp), power, the active
           clock-event (a.k.a. throttle) reasons, and the GEMM TFLOP/s per chunk.
  latency  single-thread pointer chase (Triton kernel, grid (1,), one warp, serial dependent loads, .cg = L2-only
           caching) over a random single cycle of --bytes (default 1 GiB) plus smaller footprints for contrast.
           ns/access = elapsed / steps after a warm-up. PLAN.md 7.2 uses ~600 ns; sanity range 500-800 ns.
  merge    combines clock + latency JSON + the gpu-freq step (gpufreq_probe.csv sampled by nvidia-smi inside
           the step, and the step's stderr) into results/db/platform/clocks_<jobid>.json. gpu_freq_pinnable is
           true iff every busy sample of the step has its SM clock within --target-mhz +- --tol-mhz. CPU only.

Run (inside a GPU job, see slurm/s2_harness.sbatch):
  python -m cisc662.bench.platform_probe clock --seconds 60 --out $OUT/clock.json
  python -m cisc662.bench.platform_probe latency --bytes 1073741824 --out $OUT/latency.json
  python -m cisc662.bench.platform_probe merge --job-id $SLURM_JOB_ID --clock $OUT/clock.json \
      --latency $OUT/latency.json --gpufreq-csv $OUT/gpufreq_probe.csv --gpufreq-stderr $OUT/gpufreq_step.stderr
"""
import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
import threading
import time

import triton  # light import (no torch, no GPU); torch is imported lazily by the GPU subcommands
import triton.language as tl

# Field names of the active clock-event bitmask: drivers >= 535 call it clocks_event_reasons, older ones
# clocks_throttle_reasons (newer drivers may drop the old alias). We probe which one this nvidia-smi accepts.
REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")
# NVML nvmlClocksEventReason* bits
REASON_BITS = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}
QUERY_BASE = "timestamp,clocks.sm,clocks.gr,clocks.mem,power.draw,temperature.gpu,utilization.gpu"
DEFAULT_PLATFORM_DIR = "cisc662/results/db/platform"


# ----------------------------------------------------------------------------------------------- nvidia-smi
def smi(args, timeout=10):
    """Run nvidia-smi; return stdout or None on any failure (missing binary, no driver, bad field, hang)."""
    try:
        r = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def pick_reason_field(id_args=()):
    """First clock-reason field name this nvidia-smi accepts, or None."""
    for f in REASON_FIELDS:
        if smi(list(id_args) + [f"--query-gpu={f}", "--format=csv,noheader"]) is not None:
            return f
    return None


def decode_reasons(value):
    """'0x0000000000000004' -> ['sw_power_cap']; None if the value is not a hex mask (e.g. '[N/A]')."""
    try:
        mask = int(str(value).strip(), 16)
    except ValueError:
        return None
    return [name for bit, name in REASON_BITS.items() if mask & bit]


def num(value):
    """'1755 MHz' / '524.12 W' / '100 %' -> float; None for N/A or empty."""
    m = re.match(r"\s*(-?[0-9]+(?:\.[0-9]+)?)", str(value))
    return float(m.group(1)) if m else None


# nvidia-smi prints the long form of some query fields in the CSV header (e.g. --query-gpu=clocks.sm gives
# "clocks.current.sm [MHz]", see results/runs/s1_*/gpu_monitor.csv), so headers are mapped to the short names
# we query with. The throttle alias is folded into the clocks_event_reasons name.
FIELD_ALIASES = {
    "clocks.current.sm": "clocks.sm",
    "clocks.current.graphics": "clocks.gr",
    "clocks.current.memory": "clocks.mem",
    "clocks.current.video": "clocks.video",
    "clocks.applications.gr": "clocks.applications.graphics",
    "clocks_throttle_reasons.active": "clocks_event_reasons.active",
}


def canon_field(name):
    return FIELD_ALIASES.get(name, name)


def clean_header(h):
    """'clocks.current.sm [MHz]' -> 'clocks.sm' (unit stripped, long form mapped to the queried name)."""
    return canon_field(re.sub(r"\s*\[.*?\]\s*$", "", h.strip()))


def parse_query_csv(text):
    """nvidia-smi --query-gpu ... --format=csv output (with header) -> list of {field: raw string}."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    rows = list(csv.reader(lines, skipinitialspace=True))
    header = [clean_header(h) for h in rows[0]]
    out = []
    for r in rows[1:]:
        if len(r) != len(header) or r[0].strip().lower() == header[0]:  # skip repeated headers / torn lines
            continue
        out.append({h: v.strip() for h, v in zip(header, r)})
    return out


def parse_smi_q(text):
    """Parse `nvidia-smi -q` text into {'Section/Sub/Key': value} for the FIRST GPU block.

    The format is indentation-structured: section headers are lines without ' : ', leaves are 'Key : Value'.
    'GPU 00000009:01:00.0' block headers contain colons but no ' : ', so they parse as sections.
    """
    flat, stack, gpu_blocks = {}, [], 0
    for raw in text.splitlines():
        if not raw.strip() or raw.startswith("="):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if re.match(r"^GPU [0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9A-Fa-f]+$", line):
            gpu_blocks += 1
            if gpu_blocks > 1:
                break
            stack = []
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        m = re.match(r"^(.*?)\s+:(?:\s(.*))?$", line)
        if m:
            key = "/".join([s[1] for s in stack] + [m.group(1).strip()])
            flat[key] = (m.group(2) or "").strip()
        else:
            stack.append((indent, line))
    return flat


def q_summary(flat):
    """Pick the fields we report from a parsed -q dict (names differ across driver versions)."""
    def first(*keys):
        for k in keys:
            if flat.get(k) not in (None, "", "N/A"):
                return flat[k]
        return None
    active, counters = [], {}
    for k, v in flat.items():
        parts = k.split("/")
        if len(parts) >= 2 and "Reasons" in parts[0]:
            if "Counters" in parts[0]:
                counters[parts[-1]] = v
            elif v.strip().lower() == "active":
                active.append(parts[-1])
    return {
        "sm_mhz": num(first("Clocks/SM")),
        "graphics_mhz": num(first("Clocks/Graphics")),
        "mem_mhz": num(first("Clocks/Memory")),
        "app_graphics_mhz": num(first("Applications Clocks/Graphics")),
        "power_w": num(first("GPU Power Readings/Instantaneous Power Draw", "GPU Power Readings/Power Draw",
                             "GPU Power Readings/Average Power Draw", "Power Readings/Power Draw")),
        "power_limit_w": num(first("GPU Power Readings/Current Power Limit", "GPU Power Readings/Power Limit",
                                   "Power Readings/Power Limit")),
        "perf_state": first("Performance State"),
        "reasons_active": active,
        "reason_counters": counters,
    }


def gpu_id_args():
    """['-i', 'GPU-<uuid>'] for the torch device 0 if nvidia-smi accepts it, else [] (cgroup shows one GPU)."""
    try:
        import torch
        u = str(torch.cuda.get_device_properties(0).uuid)
    except Exception:
        return []
    for cand in (u, "GPU-" + u):
        if smi(["-i", cand, "--query-gpu=uuid", "--format=csv,noheader"]) is not None:
            return ["-i", cand]
    return []


class Sampler(threading.Thread):
    """Samples nvidia-smi every `interval` s until stop(); keeps raw -q text of the first sample only."""

    def __init__(self, interval, id_args, reason_field, t0):
        super().__init__(daemon=True)
        self.interval, self.id_args, self.reason_field, self.t0 = interval, id_args, reason_field, t0
        self.samples, self.first_q_text, self._stop_evt = [], None, threading.Event()

    def sample_once(self):
        t = time.time()
        q = smi(self.id_args + ["-q", "-d", "CLOCK,POWER,PERFORMANCE"])
        fields = QUERY_BASE + ("," + self.reason_field if self.reason_field else "")
        c = smi(self.id_args + [f"--query-gpu={fields}", "--format=csv"])
        s = {"t_s": round(t - self.t0, 3)}
        if q is not None:
            if self.first_q_text is None:
                self.first_q_text = q
            s["q"] = q_summary(parse_smi_q(q))
        rows = parse_query_csv(c) if c else []
        if rows:
            r = rows[0]
            s.update({"sm_mhz": num(r.get("clocks.sm")), "gr_mhz": num(r.get("clocks.gr")),
                      "mem_mhz": num(r.get("clocks.mem")), "power_w": num(r.get("power.draw")),
                      "temp_c": num(r.get("temperature.gpu")), "util_pct": num(r.get("utilization.gpu"))})
            if self.reason_field:
                s["reasons_mask"] = r.get(canon_field(self.reason_field))
                s["reasons"] = decode_reasons(s["reasons_mask"]) if s["reasons_mask"] is not None else None
        if "q" in s:  # csv query failed or a field was missing/[N/A]: fall back to the -q numbers
            for k in ("sm_mhz", "power_w"):
                if s.get(k) is None:
                    s[k] = s["q"][k]
        self.samples.append(s)

    def run(self):
        while not self._stop_evt.is_set():
            t = time.time()
            self.sample_once()
            self._stop_evt.wait(max(0.0, self.interval - (time.time() - t)))

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=30)


def median_or_none(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def env_info():
    info = {"host": platform.node(), "job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID")}
    v = smi(["--query-gpu=name,driver_version,clocks.max.sm,clocks.applications.graphics,power.limit",
             "--format=csv"])
    rows = parse_query_csv(v) if v else []
    if rows:
        info["nvidia_smi"] = rows[0]
    try:
        import torch
        import triton
        info.update({"torch": torch.__version__, "triton": triton.__version__,
                     "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception:
        pass
    return info


# ----------------------------------------------------------------------------------------------- clock
def cmd_clock(args):
    import torch
    assert torch.cuda.is_available(), "clock probe needs a GPU"
    id_args = gpu_id_args()
    reason_field = pick_reason_field(id_args)
    n = args.size
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):  # cuBLAS heuristics / workspace outside the measured loop
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()

    t0 = time.time()
    sampler = Sampler(args.interval_s, id_args, reason_field, t0)
    sampler.sample_once()  # idle (pre-loop) sample at t~0
    sampler.start()
    chunks, loop_t0 = [], time.time()
    flops_per_mm = 2 * n ** 3
    while time.time() - loop_t0 < args.seconds:
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(args.chunk):
            torch.mm(a, b, out=c)
        e1.record()
        e1.synchronize()
        ms = e0.elapsed_time(e1)
        chunks.append({"t_s": round(time.time() - t0, 3), "ms": ms,
                       "tflops": args.chunk * flops_per_mm / (ms * 1e-3) / 1e12})
    loop_t1 = time.time()
    sampler.stop()
    time.sleep(args.interval_s)  # post-loop sample: does the clock drop back when idle?
    sampler.sample_once()

    busy = [s for s in sampler.samples
            if loop_t0 - t0 + args.ramp_s <= s["t_s"] <= loop_t1 - t0]
    reasons_seen = {}
    for s in busy:
        names = s.get("reasons")
        if names is None:
            names = (s.get("q") or {}).get("reasons_active", [])
        for r in names:
            reasons_seen[r] = reasons_seen.get(r, 0) + 1
    tf = [ch["tflops"] for ch in chunks[1:]] or [ch["tflops"] for ch in chunks]
    sm = [s.get("sm_mhz") for s in busy]
    out = {
        "probe": "clock",
        "env": env_info(),
        "gemm": {"size": n, "dtype": "bf16", "chunk_mms": args.chunk, "seconds": round(loop_t1 - loop_t0, 2),
                 "n_chunks": len(chunks), "tflops_median": median_or_none(tf),
                 "tflops_min": min(tf) if tf else None, "tflops_max": max(tf) if tf else None},
        "reason_field": reason_field,
        "nvidia_smi_id_args": id_args,
        "ramp_s": args.ramp_s,
        "n_busy_samples": len(busy),
        "sustained_sm_mhz_median": median_or_none(sm),
        "sustained_sm_mhz_min": min([x for x in sm if x is not None], default=None),
        "sustained_sm_mhz_max": max([x for x in sm if x is not None], default=None),
        "sustained_mem_mhz_median": median_or_none([s.get("mem_mhz") for s in busy]),
        "power_w_median": median_or_none([s.get("power_w") for s in busy]),
        "power_w_max": max([s.get("power_w") for s in busy if s.get("power_w") is not None], default=None),
        "temp_c_max": max([s.get("temp_c") for s in busy if s.get("temp_c") is not None], default=None),
        # reason -> number of busy samples in which it was active (clocks_throttle_reasons.active, decoded)
        "throttle_reasons_active": reasons_seen,
        "idle_pre": sampler.samples[0] if sampler.samples else None,
        "idle_post": sampler.samples[-1] if sampler.samples else None,
        "samples": sampler.samples,
        "chunks": chunks,
        "smi_q_first_sample": sampler.first_q_text,
    }
    write_json(args.out, out)
    print(f"clock: sustained SM {out['sustained_sm_mhz_median']} MHz (n={len(busy)}), power "
          f"{out['power_w_median']} W, GEMM {out['gemm']['tflops_median']} TF/s, reasons {reasons_seen}")


# ----------------------------------------------------------------------------------------------- latency
@triton.jit
def _c662_pointer_chase(nxt_ptr, start_ptr, out_ptr, n_steps):
    # one program, one warp: every lane loads the same address -> one serial chain of dependent loads
    idx = tl.load(start_ptr)
    for _ in range(n_steps):
        idx = tl.load(nxt_ptr + idx, cache_modifier=".cg")  # .cg: cache in L2 only, not L1
    tl.store(out_ptr, idx)


def build_cycle(n, device, seed):
    """Random single cycle over n slots: nxt[perm[i]] = perm[(i+1) % n] (uniform random cyclic permutation)."""
    import torch
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    perm = torch.randperm(n, device=device, generator=g, dtype=torch.int64)
    nxt = torch.empty(n, dtype=torch.int64, device=device)
    nxt[perm] = torch.roll(perm, -1)
    return perm, nxt


def run_chase(nxt, start, n_steps):
    import torch
    out = torch.empty(1, dtype=torch.int64, device=nxt.device)
    _c662_pointer_chase[(1,)](nxt, start, out, n_steps, num_warps=1, num_stages=1)
    return out


def chase_one(nbytes, n_steps, warmup_steps, reps, seed):
    """ns/access for one footprint; verifies the chain ended where the cycle says it must."""
    import torch
    n = max(2, nbytes // 8)
    perm, nxt = build_cycle(n, "cuda", seed)
    res = {"bytes": n * 8, "n_elems": n, "n_steps": n_steps, "warmup_steps": warmup_steps,
           "ns_per_access": [], "verified": True}
    for r in range(reps):
        k = (r * 7919) % n  # different start point per rep
        start = perm[k:k + 1].clone()
        run_chase(nxt, start, warmup_steps)  # JIT (first call) + TLB/L2 warm-up
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        out = run_chase(nxt, start, n_steps)
        e1.record()
        e1.synchronize()
        res["ns_per_access"].append(e0.elapsed_time(e1) * 1e6 / n_steps)
        expect = perm[(k + n_steps) % n].item()
        res["verified"] = res["verified"] and out.item() == expect
    res["ns_median"] = statistics.median(res["ns_per_access"])
    del perm, nxt
    torch.cuda.empty_cache()
    return res


def cmd_latency(args):
    import torch
    assert torch.cuda.is_available(), "latency probe needs a GPU"
    sizes = [args.bytes] + [int(x) for x in args.extra_bytes.split(",") if x.strip()]
    results = []
    for i, nb in enumerate(sizes):
        r = chase_one(nb, args.steps, args.warmup_steps, args.reps, args.seed + i)
        results.append(r)
        print(f"latency: {r['bytes'] / 2**20:.3f} MiB -> {r['ns_median']:.1f} ns/access "
              f"(reps {[round(x, 1) for x in r['ns_per_access']]}, verified={r['verified']})", flush=True)
    main = results[0]
    out = {
        "probe": "latency",
        "env": env_info(),
        "method": "Triton grid (1,), 1 warp, serial dependent int64 loads (.cg), random single cycle "
                  "nxt[perm[i]]=perm[(i+1)%n] built with torch.randperm on the GPU; CUDA events; "
                  "ns/access = elapsed/steps after a warm-up chase",
        "latency_ns": main["ns_median"],
        "latency_bytes": main["bytes"],
        "sane_500_800": 500.0 <= main["ns_median"] <= 800.0,
        "verified": all(r["verified"] for r in results),
        "footprints": results,
    }
    write_json(args.out, out)


# ----------------------------------------------------------------------------------------------- merge
def gpufreq_from_csv(path, target, tol, busy_util, min_busy=5):
    """Evaluate the gpu-freq step: is the SM clock within target +- tol for every busy sample (>= min_busy)?"""
    try:
        with open(path) as f:
            rows = parse_query_csv(f.read())
    except OSError as e:
        return {"error": f"cannot read {path}: {e}", "n_samples": 0}
    samples = []
    for r in rows:
        sm = num(r.get("clocks.sm"))
        if sm is None:
            continue
        util = num(r.get("utilization.gpu")) if "utilization.gpu" in r else None
        reason_val = next((r[f] for f in REASON_FIELDS if f in r), None)
        samples.append({"sm_mhz": sm, "util_pct": util,
                        "reasons": decode_reasons(reason_val) if reason_val is not None else None})
    have_util = any(s["util_pct"] is not None for s in samples)
    # Without a utilization column every sample counts as busy (strict).
    busy = [s for s in samples if not have_util or (s["util_pct"] is not None and s["util_pct"] >= busy_util)]
    inband = [s for s in busy if abs(s["sm_mhz"] - target) <= tol]
    reasons = {}
    for s in busy:
        for x in s["reasons"] or []:
            reasons[x] = reasons.get(x, 0) + 1
    sm = [s["sm_mhz"] for s in busy]
    return {
        "csv": path, "n_samples": len(samples), "n_busy": len(busy), "busy_rule":
            f"utilization.gpu >= {busy_util}%" if have_util else "all samples (no utilization column)",
        "n_in_band": len(inband), "frac_in_band": len(inband) / len(busy) if busy else None,
        "sm_mhz_median_busy": statistics.median(sm) if sm else None,
        "sm_mhz_min_busy": min(sm) if sm else None, "sm_mhz_max_busy": max(sm) if sm else None,
        "reasons_busy": reasons,
        "min_busy": min_busy,
        "pinnable": len(busy) >= min_busy and len(inband) == len(busy),
    }


def read_text(path, limit=20000):
    if not path:
        return None
    try:
        with open(path, errors="replace") as f:
            t = f.read()
    except OSError:
        return None
    return t if len(t) <= limit else t[:limit // 2] + "\n...[truncated]...\n" + t[-limit // 2:]


def read_json(path):
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        return {"error": f"cannot read {path}: {e}"}


def smoke_summary(d):
    """case -> {median_us, gbps, tflops} from a kbench out-dir (if given and present)."""
    if not d or not os.path.isdir(d):
        return None
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            j = read_json(os.path.join(d, fn)) or {}
            if "median_us" in j:
                out[j.get("case", fn[:-5])] = {k: j.get(k) for k in ("median_us", "p10_us", "p90_us", "gbps",
                                                                     "tflops", "trials")}
    return out


def cmd_merge(args):
    clock = read_json(args.clock) or {}
    lat = read_json(args.latency) or {}
    pinned_clock = read_json(args.pinned_clock)
    step = gpufreq_from_csv(args.gpufreq_csv, args.target_mhz, args.tol_mhz, args.busy_util, args.min_busy)
    tgt = f"{args.target_mhz:g}"
    step["target_mhz"], step["tol_mhz"] = args.target_mhz, args.tol_mhz
    step["rc"] = args.gpufreq_rc
    step["stderr"] = read_text(args.gpufreq_stderr)
    if pinned_clock and "error" not in pinned_clock:
        step["gemm_loop_in_step"] = {k: pinned_clock.get(k) for k in (
            "sustained_sm_mhz_median", "sustained_sm_mhz_min", "sustained_sm_mhz_max", "power_w_median",
            "throttle_reasons_active")}
        step["gemm_loop_in_step"]["tflops_median"] = (pinned_clock.get("gemm") or {}).get("tflops_median")
    pinnable = bool(step.get("pinnable")) and args.gpufreq_rc in (None, 0)
    out = {
        "job_id": args.job_id,
        "host": (clock.get("env") or {}).get("host") or (lat.get("env") or {}).get("host"),
        "sustained_sm_mhz": clock.get("sustained_sm_mhz_median"),
        "sustained_sm_mhz_range": [clock.get("sustained_sm_mhz_min"), clock.get("sustained_sm_mhz_max")],
        "sustained_mem_mhz": clock.get("sustained_mem_mhz_median"),
        "power_w": clock.get("power_w_median"),
        "power_w_max": clock.get("power_w_max"),
        "gemm_tflops_sustained": (clock.get("gemm") or {}).get("tflops_median"),
        "throttle_reasons": clock.get("throttle_reasons_active"),
        "reason_field": clock.get("reason_field"),
        "latency_ns": lat.get("latency_ns"),
        "latency_bytes": lat.get("latency_bytes"),
        "latency_sane_500_800": lat.get("sane_500_800"),
        "latency_footprints": [{k: r.get(k) for k in ("bytes", "ns_median", "ns_per_access", "verified")}
                               for r in lat.get("footprints", [])],
        "gpu_freq_pinnable": pinnable,
        "gpu_freq_step": step,
        "clock_policy_recommendation": (
            f"pinned {tgt} MHz via `srun --gpu-freq={tgt}` for the microbenchmark "
            "sensitivity repeat; default clocks (GpuFreqDef=high) + gpu_monitor.csv logging for headline runs"
            if pinnable else
            "default clocks (GpuFreqDef=high) + gpu_monitor.csv logging; --gpu-freq pinning not reliable here"),
        "smoke_default_clocks": smoke_summary(args.smoke_default_dir),
        "smoke_pinned_clocks": smoke_summary(args.smoke_pinned_dir),
        "sources": {"clock": args.clock, "latency": args.latency, "gpufreq_csv": args.gpufreq_csv,
                    "gpufreq_stderr": args.gpufreq_stderr, "pinned_clock": args.pinned_clock},
    }
    for name, j in (("clock", clock), ("latency", lat)):
        if not j or "error" in j:
            out.setdefault("missing", []).append(name)
    path = args.out or os.path.join(DEFAULT_PLATFORM_DIR, f"clocks_{args.job_id}.json")
    write_json(path, out)
    print(f"merge: sustained {out['sustained_sm_mhz']} MHz, {out['power_w']} W, latency {out['latency_ns']} ns, "
          f"gpu_freq_pinnable={pinnable} -> {path}")


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("clock", help="sustained clock/power/throttle reasons under a bf16 GEMM loop")
    c.add_argument("--seconds", type=float, default=60.0)
    c.add_argument("--interval-s", type=float, default=2.0, help="nvidia-smi sampling period")
    c.add_argument("--ramp-s", type=float, default=3.0, help="samples earlier than this into the loop are ignored")
    c.add_argument("--size", type=int, default=8192, help="square GEMM size")
    c.add_argument("--chunk", type=int, default=20, help="GEMMs per timed chunk (one sync per chunk)")
    c.add_argument("--out", required=True)

    l = sub.add_parser("latency", help="single-thread pointer-chase latency")
    l.add_argument("--bytes", type=int, default=1 << 30, help="main footprint (sanity 500-800 ns at 1 GiB)")
    l.add_argument("--extra-bytes", default=str(4 << 20), help="comma list of contrast footprints (4 MiB = L2)")
    l.add_argument("--steps", type=int, default=1 << 20, help="dependent loads per timed chase")
    l.add_argument("--warmup-steps", type=int, default=1 << 14)
    l.add_argument("--reps", type=int, default=3)
    l.add_argument("--seed", type=int, default=0)
    l.add_argument("--out", required=True)

    m = sub.add_parser("merge", help="write results/db/platform/clocks_<jobid>.json")
    m.add_argument("--job-id", required=True)
    m.add_argument("--clock")
    m.add_argument("--latency")
    m.add_argument("--gpufreq-csv", required=True)
    m.add_argument("--gpufreq-stderr")
    m.add_argument("--gpufreq-rc", type=int, default=None, help="exit code of the gpu-freq step (0 = ok)")
    m.add_argument("--pinned-clock", help="clock JSON measured inside the --gpu-freq step (optional)")
    m.add_argument("--smoke-default-dir", help="kbench smoke out-dir at default clocks (optional)")
    m.add_argument("--smoke-pinned-dir", help="kbench smoke out-dir inside the --gpu-freq step (optional)")
    m.add_argument("--target-mhz", type=float, default=1755.0)
    m.add_argument("--tol-mhz", type=float, default=15.0)
    m.add_argument("--busy-util", type=float, default=50.0, help="utilization.gpu %% that counts as busy")
    m.add_argument("--min-busy", type=int, default=5, help="busy samples needed before pinnable can be true")
    m.add_argument("--out", help=f"default {DEFAULT_PLATFORM_DIR}/clocks_<jobid>.json")

    args = p.parse_args()
    {"clock": cmd_clock, "latency": cmd_latency, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
