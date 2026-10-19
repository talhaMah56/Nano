"""
Kernel microbenchmark harness (task INF-1; protocol = PLAN.md §7.1).

- bench(fn, *args, ...): CUDA-event timing of one callable. The first call (JIT / autotune /
  torch.compile) and `warmup` reps are untimed; before every rep a 256 MB buffer is zeroed outside
  the timed region (L2 is 60 MiB) unless flush_l2=False; then a GPU spin (queue pad, 2x the host enqueue
  time) so the event pair times device work only, never host overhead; one synchronize after the reps loop.
- run_trials(suite, trials=3, ...): spawns `trials` separate python processes; each runs the WHOLE
  suite once (`--trial K --raw-out PATH`) and rewrites its raw per-rep times after every case. The
  parent aggregates after every finished trial (median of trial medians, p10-p90 of pooled reps,
  bootstrap CIs vs every named baseline from cisc662/analysis/stats.py) and writes one JSON per case
  to <out_dir>/<case>.json (schema: cisc662/results/db/README.md), so a failed or timed-out later
  trial still leaves the aggregates of the trials that finished.
- Every child records a preamble (git SHA + dirty flag, torch/triton/CUDA versions, device, host,
  SLURM_JOB_ID, HAS_FA3/USE_FA3, nvidia-smi clocks/power at start and end) and aborts with exit
  code 3 if a case that needs FA3 is requested while FA3 is not in use.

Other benchmark modules (ops_at_shapes.py, kernel cards) build a list of Case objects and call
main_cli(); see the 'smoke' suite below for the minimal pattern.

Run (GPU job only):
  python -m cisc662.bench.kbench --suite smoke --trials 3 --out-dir cisc662/results/db/smoke [--job-id ID]
  python -m cisc662.bench.kbench --suite smoke --list      # CPU-only: print the case registry
"""
import argparse
import datetime
import gc
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_ROOT = os.path.join(REPO, "cisc662", "results", "db")
FLUSH_BYTES = 256 * 2**20          # > 4x the 60 MiB L2 of GH200
PAD_MIN_US = 200.0          # queue pad before every start event (see bench())
PAD_FACTOR = 2.0
PAD_CYCLES_PER_US = 1980    # GH200 max SM clock (MHz) = cycles per µs
COPY_GBPS_GH200 = 3650.5           # results/smoke_gh200.json bandwidth.copy_GBps (job 3400653)
FA3_ABORT_RC = 3
SMI_FIELDS = "uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu"
# the throttle field was renamed in newer drivers; try both, then none
SMI_REASON_FIELDS = ("clocks_throttle_reasons.active", "clocks_event_reasons.active", None)


# ----------------------------------------------------------------------------------------------
# Case registry
# ----------------------------------------------------------------------------------------------
@dataclass
class Case:
    """One timed callable. `setup()` runs inside a child process on the GPU and returns
    (fn, args); everything else is static metadata, so the registry builds without CUDA (--list).
    min_bytes / flops are algorithmic minimums (analytic), used for GB/s and TFLOP/s."""
    name: str
    setup: Callable[[], tuple]
    dtype: str
    shape: dict
    min_bytes: Optional[float] = None
    flops: Optional[float] = None
    group: str = ""
    kernel_id: str = ""
    form: str = "eager"            # eager | compile | autotune | <variant name>
    baselines: tuple = ()          # case names this case gets a speedup CI against (if present)
    flush_l2: bool = True
    warmup: int = 25
    reps: int = 100
    requires_fa3: bool = False
    notes: str = ""

    def meta(self):
        return {k: getattr(self, k) for k in ("name", "dtype", "shape", "min_bytes", "flops", "group", "kernel_id",
                                               "form", "flush_l2", "warmup", "reps", "requires_fa3", "notes")} | {
            "baselines": list(self.baselines)}


# ----------------------------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------------------------
_flush_buf = None


def _l2_flush_buffer():
    import torch
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    return _flush_buf


def bench(fn, *args, flush_l2=True, warmup=25, reps=100, label=None, min_bytes=None, flops=None, pad=True):
    """Times fn(*args) with one CUDA event pair per rep; returns a dict of µs statistics.

    Order: one untimed first call (JIT/autotune/compile) + synchronize, `warmup` untimed reps,
    then `reps` timed reps. With flush_l2 the 256 MB buffer is zeroed before every rep (warm-ups
    included) and before the start event, so the zeroing is never inside the timed region.
    Note: the zeroed lines are dirty in L2, so the timed kernel pays their write-back; that bias is
    identical for every variant (same as triton.testing.do_bench).
    Queue pad (pad=True): before every start event a torch.cuda._sleep spin of PAD_FACTOR x the
    host enqueue time of fn (measured in the warm-up; >= PAD_MIN_US) is queued, so the CPU is always
    ahead of the GPU and the event pair measures device time only. Without it, a callable whose
    host-side cost (dynamo guards, AOTAutograd wrappers, autograd.grad, many small eager launches)
    exceeds its GPU time leaves the GPU idle inside the timed region (job 3406282: compiled
    rmsnorm fwd+bwd 349 us vs 122 us eager; warm-L2 RMSNorm slower than cold). host_* fields keep
    the host enqueue time per call (perf_counter around fn, no sync) for the launch-bound analysis."""
    import torch
    flush = _l2_flush_buffer() if flush_l2 else None
    fn(*args)
    torch.cuda.synchronize()
    host_warm = []
    for _ in range(warmup):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()          # empty queue: perf_counter sees the pure host enqueue cost
        h0 = time.perf_counter()
        fn(*args)
        host_warm.append((time.perf_counter() - h0) * 1e6)
    host_warm.sort()
    host_warm_med = host_warm[len(host_warm) // 2] if host_warm else 0.0
    pad_us = max(PAD_MIN_US, PAD_FACTOR * host_warm_med) if pad else 0.0
    pad_cycles = int(pad_us * PAD_CYCLES_PER_US)   # cycles at the max 1980 MHz clock: at lower clocks the pad only gets longer
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    host = []
    torch.cuda.synchronize()
    gc_was_enabled = gc.isenabled()
    gc.disable()          # a GC pause in an early rep could idle the GPU before the backlog builds
    for i in range(reps):
        if flush is not None:
            flush.zero_()
        if pad_cycles:
            torch.cuda._sleep(pad_cycles)
        starts[i].record()
        h0 = time.perf_counter()
        fn(*args)
        host.append((time.perf_counter() - h0) * 1e6)
        ends[i].record()
    if gc_was_enabled:
        gc.enable()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]   # ms -> µs
    hs = sorted(host)
    srt = sorted(times)
    med = srt[len(srt) // 2] if len(srt) % 2 else 0.5 * (srt[len(srt) // 2 - 1] + srt[len(srt) // 2])
    pct = lambda q: srt[min(len(srt) - 1, int(round(q * (len(srt) - 1))))]
    return {
        "label": label,
        "median_us": med,
        "p10_us": pct(0.10),
        "p90_us": pct(0.90),
        "min_us": srt[0],
        "times_us": times,
        "gbps": (min_bytes / (med * 1e-6) / 1e9) if min_bytes else None,
        "tflops": (flops / (med * 1e-6) / 1e12) if flops else None,
        "pad_us": pad_us,
        "host_warmup_median_us": host_warm_med,
        "host_median_us": hs[len(hs) // 2] if hs else None,   # timed loop: can include launch-queue-full stalls
        "host_bound": host_warm_med > med,    # pure enqueue cost (synchronized warm-up) vs device time
    }


# ----------------------------------------------------------------------------------------------
# Environment preamble
# ----------------------------------------------------------------------------------------------
def _run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def nvidia_smi_snapshot(uuid=None):
    """Clocks/power of the GPU this process uses (matched by UUID; all GPUs if no match)."""
    out = query = None
    for reason in SMI_REASON_FIELDS:
        query = SMI_FIELDS + (f",{reason}" if reason else "")
        out = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
        if out is not None:
            break
    if out is None:
        return {"error": "nvidia-smi unavailable"}
    keys = query.split(",")
    rows = [dict(zip(keys, [v.strip() for v in line.split(",")])) for line in out.splitlines() if line.strip()]
    mine = [r for r in rows if uuid and uuid in r.get("uuid", "")]
    snap = mine[0] if len(mine) == 1 else {"all_gpus": rows}
    snap["time"] = datetime.datetime.now().isoformat(timespec="seconds")
    return snap


def env_preamble():
    """Everything a result must carry to be reproducible; called once per child process."""
    import torch
    import triton
    env = {
        "git_sha": _run(["git", "-C", REPO, "rev-parse", "HEAD"]),
        "git_dirty": None,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nanochat_attn_impl": os.environ.get("NANOCHAT_ATTN_IMPL"),
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "l2_flush_bytes": FLUSH_BYTES,
        "clock_mode": "default (GpuFreqDef=high, not pinned)",
    }
    status = _run(["git", "-C", REPO, "status", "--porcelain"])
    if status is not None:
        env["git_dirty"] = bool(status.strip())
        env["git_dirty_paths"] = len(status.splitlines())
    try:
        from nanochat.flash_attention import HAS_FA3, USE_FA3
        env["HAS_FA3"], env["USE_FA3"] = bool(HAS_FA3), bool(USE_FA3)
    except Exception as e:   # recorded, and FA3 cases abort below
        env["HAS_FA3"] = env["USE_FA3"] = False
        env["fa3_import_error"] = repr(e)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env["device"] = p.name
        env["capability"] = f"{p.major}.{p.minor}"
        env["sm_count"] = p.multi_processor_count
        env["gpu_uuid"] = str(getattr(p, "uuid", "")) or None
    return env


# ----------------------------------------------------------------------------------------------
# Child: run the whole suite once
# ----------------------------------------------------------------------------------------------
def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _free_gpu(compiled):
    import torch
    gc.collect()
    if compiled:
        torch._dynamo.reset()      # drop guards/graphs so the next case starts from a clean cache
    torch.cuda.empty_cache()


def run_child(cases, suite, trial, raw_out, deadline=None):
    """Runs every case once in this process, rewriting raw_out after each case. Returns exit code."""
    import torch
    import torch._dynamo
    from torch._dynamo.utils import counters
    # many cases share one python function at different shapes; never let Dynamo fall back to eager
    for knob in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, knob):
            setattr(torch._dynamo.config, knob, 64)
    t0 = time.time()
    env = env_preamble()
    env["smi_start"] = nvidia_smi_snapshot(env.get("gpu_uuid"))
    raw = {"suite": suite, "trial": trial, "pid": os.getpid(), "started": env["smi_start"].get("time"),
           "env": env, "cases": {}, "complete": False}
    print(f"[kbench trial {trial}] host={env['host']} device={env.get('device')} HAS_FA3={env['HAS_FA3']} "
          f"USE_FA3={env['USE_FA3']} git={str(env['git_sha'])[:10]} dirty={env['git_dirty']}", flush=True)

    need_fa3 = [c.name for c in cases if c.requires_fa3]
    if need_fa3 and not (env["HAS_FA3"] and env["USE_FA3"]):
        raw["abort"] = f"FA3 cases requested ({len(need_fa3)}) but HAS_FA3={env['HAS_FA3']} USE_FA3={env['USE_FA3']}"
        _atomic_json(raw_out, raw)
        print(f"ABORT: {raw['abort']}", file=sys.stderr, flush=True)
        return FA3_ABORT_RC
    if not torch.cuda.is_available():
        raw["abort"] = "CUDA not available"
        _atomic_json(raw_out, raw)
        print("ABORT: CUDA not available (kbench runs only inside GPU jobs)", file=sys.stderr, flush=True)
        return 2

    for c in cases:
        rec = {"meta": c.meta(), "status": "ok"}
        if deadline is not None and time.time() > deadline:
            rec["status"] = "skipped_time_budget"
            raw["cases"][c.name] = rec
            _atomic_json(raw_out, raw)
            print(f"[kbench trial {trial}] skip {c.name} (time budget)", flush=True)
            continue
        compiled = c.form != "eager"
        graphs0 = counters["stats"]["unique_graphs"]
        ts = time.time()
        fn = args = res = None
        try:
            torch.manual_seed(0)
            fn, args = c.setup()
            rec["setup_s"] = round(time.time() - ts, 3)
            res = bench(fn, *args, flush_l2=c.flush_l2, warmup=c.warmup, reps=c.reps, label=c.name,
                        min_bytes=c.min_bytes, flops=c.flops)
            rec["times_us"] = [round(t, 4) for t in res["times_us"]]
            rec["median_us"] = res["median_us"]
            for k in ("pad_us", "host_warmup_median_us", "host_median_us", "host_bound"):
                rec[k] = res[k]
            rec["max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            print(f"[kbench trial {trial}] {c.name:<48s} median {res['median_us']:10.2f} us"
                  + (f"  {res['gbps']:7.0f} GB/s" if res["gbps"] else "")
                  + (f"  {res['tflops']:6.1f} TF/s" if res["tflops"] else ""), flush=True)
        except Exception as e:     # OOM or a broken case must not take the rest of the suite down
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
            print(f"[kbench trial {trial}] ERROR {c.name}: {rec['error'][:300]}", flush=True)
        rec["graphs_compiled"] = counters["stats"]["unique_graphs"] - graphs0
        rec["case_s"] = round(time.time() - ts, 3)
        raw["cases"][c.name] = rec
        _atomic_json(raw_out, raw)
        del fn, args, res
        torch.cuda.reset_peak_memory_stats()
        _free_gpu(compiled)

    env["smi_end"] = nvidia_smi_snapshot(env.get("gpu_uuid"))
    raw["complete"] = True
    raw["elapsed_s"] = round(time.time() - t0, 1)
    _atomic_json(raw_out, raw)
    return 0


# ----------------------------------------------------------------------------------------------
# Parent: spawn trials, aggregate, write results/db
# ----------------------------------------------------------------------------------------------
REQUIRED = {"case": str, "dtype": str, "shape": dict, "median_us": float, "p10_us": float, "p90_us": float,
            "gbps": (float, type(None)), "tflops": (float, type(None)), "trials": int, "speedups": dict,
            "gates_passed": list, "env": dict, "job_id": (str, type(None))}


def validate_record(rec):
    """Returns a list of schema problems (empty = valid); schema documented in results/db/README.md."""
    probs = []
    for k, typ in REQUIRED.items():
        if k not in rec:
            probs.append(f"missing {k}")
        elif rec[k] is None and k in ("median_us", "p10_us", "p90_us") and rec.get("trials") == 0:
            continue      # a case with no data in this job (error / time-budget skip) is still a valid record
        elif not isinstance(rec[k], typ) or (typ is float and isinstance(rec[k], bool)):
            if not (typ is float and isinstance(rec[k], int)):
                probs.append(f"{k} has type {type(rec[k]).__name__}")
    for base, s in rec.get("speedups", {}).items():
        if set(s) != {"ratio", "ci_low", "ci_high"} or not all(isinstance(v, float) for v in s.values()):
            probs.append(f"speedups[{base}] malformed")
    return probs


def aggregate(raws, job_id, n_boot=10000):
    """raws: list of child raw dicts (one per trial). Returns {case: record}."""
    from cisc662.analysis.stats import bootstrap_speedup_ci, summarize_trials
    times, meta, status, hostinfo = {}, {}, {}, {}
    for raw in raws:
        for name, rec in raw.get("cases", {}).items():
            meta.setdefault(name, rec["meta"])
            status.setdefault(name, []).append(rec["status"] if rec["status"] != "error"
                                               else f"error: {rec.get('error', '')[:300]}")
            if rec["status"] == "ok" and rec.get("times_us"):
                times.setdefault(name, []).append(rec["times_us"])
                hostinfo.setdefault(name, []).append({"trial": raw.get("trial"), **{k: rec.get(k) for k in
                                                      ("pad_us", "host_warmup_median_us", "host_median_us", "host_bound")}})
    first_env = next((r["env"] for r in raws if r.get("env")), {})
    common_env = {k: v for k, v in first_env.items() if k not in ("smi_start", "smi_end")}
    common_env["trials"] = [{"trial": r.get("trial"), "host": r.get("env", {}).get("host"),
                             "complete": r.get("complete"), "elapsed_s": r.get("elapsed_s"),
                             "smi_start": r.get("env", {}).get("smi_start"),
                             "smi_end": r.get("env", {}).get("smi_end")} for r in raws]
    out = {}
    for name, m in meta.items():
        rec = {"case": name, "dtype": m["dtype"], "shape": m["shape"], "group": m["group"],
               "kernel_id": m["kernel_id"], "form": m["form"], "min_bytes": m["min_bytes"], "flops": m["flops"],
               "flush_l2": m["flush_l2"], "warmup": m["warmup"], "reps": m["reps"], "notes": m["notes"],
               "median_us": None, "p10_us": None, "p90_us": None, "min_us": None, "gbps": None, "tflops": None,
               "trials": len(times.get(name, [])), "trial_medians_us": [], "cv_trials": None,
               "speedups": {}, "gates_passed": [], "env": common_env, "job_id": job_id,
               "status_per_trial": status[name], "trial_times_us": times.get(name, []),
               "host_per_trial": hostinfo.get(name, []),
               "host_bound": any(h.get("host_bound") for h in hostinfo.get(name, []))}
        if name in times:
            s = summarize_trials(times[name])
            rec.update(median_us=s["median"], p10_us=s["p10"], p90_us=s["p90"], min_us=s["min"],
                       trial_medians_us=s["trial_medians"], cv_trials=s["cv_trials"])
            if m["min_bytes"]:
                rec["gbps"] = m["min_bytes"] / (s["median"] * 1e-6) / 1e9
            if m["flops"]:
                rec["tflops"] = m["flops"] / (s["median"] * 1e-6) / 1e12
            for base in m["baselines"]:
                if base in times:
                    r, lo, hi = bootstrap_speedup_ci(times[base], times[name], n_boot=n_boot)
                    rec["speedups"][base] = {"ratio": r, "ci_low": lo, "ci_high": hi}
        out[name] = rec
    return out


def _write_records(records, out_dir, job_id):
    """<out_dir>/<case>.json (latest) + <out_dir>/by_job/<job_id>/<case>.json (never overwritten by
    another job). A latest file from another job is archived under its own job id before replacing."""
    job_dir = os.path.join(out_dir, "by_job", str(job_id))
    os.makedirs(job_dir, exist_ok=True)
    for name, rec in records.items():
        latest = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(latest):
            try:
                with open(latest) as f:
                    old_job = json.load(f).get("job_id")
            except (OSError, ValueError):
                old_job = "unreadable"
            if str(old_job) != str(job_id):
                arch = os.path.join(out_dir, "by_job", str(old_job))
                os.makedirs(arch, exist_ok=True)
                shutil.copy2(latest, os.path.join(arch, f"{name}.json"))   # keeps its gates_passed
        if rec["median_us"] is None and os.path.exists(latest):
            # no data this job (error/skip): record it under the job only, keep the previous latest
            _atomic_json(os.path.join(job_dir, f"{name}.json"), rec)
            continue
        _atomic_json(latest, rec)
        _atomic_json(os.path.join(job_dir, f"{name}.json"), rec)


class _ParentTerminated(Exception):
    """Raised by run_trials' SIGTERM handler while it waits for a trial process."""


def _install_sigterm():
    """SIGTERM (Slurm/`timeout` killing the step) must not lose the trial that is running: while the parent
    waits for a child the handler raises, otherwise it only sets a flag; run_trials then aggregates what the
    trials wrote and stops. Returns (state, previous handler or None if not in the main thread)."""
    state = {"waiting": False, "terminated": False}

    def handler(signum, frame):
        state["terminated"] = True
        if state["waiting"]:
            state["waiting"] = False
            raise _ParentTerminated()
    try:
        return state, signal.signal(signal.SIGTERM, handler)
    except ValueError:          # not the main thread: keep the default behaviour
        return state, None


def run_trials(suite_name, trials=3, out_dir=None, job_id=None, extra_argv=(), module="cisc662.bench.kbench",
               time_budget_s=None, grace_s=180, n_boot=10000):
    """Spawns `trials` child processes (`python -m <module> --suite <suite_name> --trial k --raw-out ...
    <extra_argv>`), aggregates after each one and writes the records. Returns
    {"records": {case: record}, "rc": int, "trials_completed": int, "raw_dir": str}.
    time_budget_s: wall budget for all trials together; children skip cases that would start after it,
    and a child still running grace_s after it is killed (its finished cases are kept)."""
    job_id = str(job_id or os.environ.get("SLURM_JOB_ID") or "nojob-" + time.strftime("%Y%m%d-%H%M%S"))
    out_dir = os.path.abspath(out_dir or os.path.join(DB_ROOT, suite_name))
    raw_dir = os.path.join(out_dir, "by_job", job_id, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    t0 = time.time()
    deadline = t0 + time_budget_s if time_budget_s else None
    raws, rc, records = [], 0, {}
    sig, old_handler = _install_sigterm()
    for k in range(1, trials + 1):
        if sig["terminated"]:
            break
        if deadline is not None and time.time() > deadline:
            print(f"[kbench] time budget exhausted before trial {k}; stopping", flush=True)
            break
        raw_path = os.path.join(raw_dir, f"trial{k}.json")
        cmd = [sys.executable, "-m", module, "--suite", suite_name, "--trial", str(k), "--raw-out", raw_path,
               *extra_argv]
        if deadline is not None:
            cmd += ["--deadline", f"{deadline:.0f}"]
        print(f"[kbench] trial {k}/{trials}: {' '.join(cmd)}", flush=True)
        timeout = (deadline - time.time() + grace_s) if deadline is not None else None
        sig["waiting"] = True
        try:
            child_rc = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                                      timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            child_rc = "timeout"
            print(f"[kbench] trial {k} killed after the time budget + {grace_s}s grace", flush=True)
        except _ParentTerminated:     # subprocess.run has killed and reaped the child
            child_rc = "sigterm"
            print(f"[kbench] SIGTERM during trial {k}: keeping its finished cases and stopping", flush=True)
        finally:
            sig["waiting"] = False
        if os.path.exists(raw_path):
            with open(raw_path) as f:
                raw = json.load(f)
            if raw.get("cases"):
                raws.append(raw)
        print(f"[kbench] trial {k} exit={child_rc}", flush=True)
        if child_rc == FA3_ABORT_RC:
            rc = FA3_ABORT_RC
            break
        if child_rc != 0:
            rc = 1
        if raws:
            records = aggregate(raws, job_id, n_boot=n_boot)
            _write_records(records, out_dir, job_id)
        if sig["terminated"]:
            rc = rc or 1
            break
    if old_handler is not None:
        signal.signal(signal.SIGTERM, old_handler)
    for name, rec in records.items():
        for p in validate_record(rec):
            print(f"[kbench] SCHEMA {name}: {p}", flush=True)
    print(f"[kbench] {len(raws)} trial(s) aggregated, {len(records)} case(s) -> {out_dir} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return {"records": records, "rc": rc, "trials_completed": len(raws), "raw_dir": raw_dir}


# ----------------------------------------------------------------------------------------------
# Smoke suite (INF-1 step 5)
# ----------------------------------------------------------------------------------------------
def _smoke_rmsnorm():
    import torch
    import torch.nn.functional as F
    x = torch.randn(32768, 768, device="cuda", dtype=torch.bfloat16) * 0.5
    return (lambda t: F.rms_norm(t, (t.size(-1),))), (x,)


def _smoke_gemm():
    import torch
    import torch.nn.functional as F
    x = torch.randn(32768, 768, device="cuda", dtype=torch.bfloat16) * 0.5
    s = 3**0.5 * 768**-0.5    # nanochat init_weights: uniform(-s, s) for c_q/c_k/c_v
    w = torch.empty(768, 768, device="cuda", dtype=torch.bfloat16).uniform_(-s, s)
    return F.linear, (x, w)


def _smoke_rmsnorm_8k():
    import torch
    import torch.nn.functional as F
    x = torch.randn(8192, 768, device="cuda", dtype=torch.bfloat16) * 0.5
    return (lambda t: F.rms_norm(t, (t.size(-1),))), (x,)


def smoke_cases(args=None):
    rms = dict(dtype="bfloat16", shape={"x": [32768, 768]}, min_bytes=2 * 32768 * 768 * 2,
               flops=4 * 32768 * 768, group="smoke", kernel_id="rmsnorm")
    rms8 = dict(dtype="bfloat16", shape={"x": [8192, 768]}, min_bytes=2 * 8192 * 768 * 2,
                flops=4 * 8192 * 768, group="smoke", kernel_id="rmsnorm")
    return [
        Case("smoke_rmsnorm_768_cold", _smoke_rmsnorm, flush_l2=True, notes="eager F.rms_norm, cold L2", **rms),
        Case("smoke_rmsnorm_768_warm", _smoke_rmsnorm, flush_l2=False, baselines=("smoke_rmsnorm_768_cold",),
             notes="same as _cold without the L2 flush; 100 MB working set > 60 MiB L2, so little reuse is expected", **rms),
        Case("smoke_rmsnorm_768_8k_cold", _smoke_rmsnorm_8k, flush_l2=True,
             notes="eager F.rms_norm on (8192,768): 25 MB working set fits the 60 MiB L2; cold L2", **rms8),
        Case("smoke_rmsnorm_768_8k_warm", _smoke_rmsnorm_8k, flush_l2=False, baselines=("smoke_rmsnorm_768_8k_cold",),
             notes="L2-resident without the flush: must be faster than _8k_cold (proves the flush works)", **rms8),
        Case("smoke_gemm_32768x768x768", _smoke_gemm, dtype="bfloat16", shape={"x": [32768, 768], "w": [768, 768]},
             min_bytes=(32768 * 768 + 768 * 768 + 32768 * 768) * 2, flops=2 * 32768 * 768 * 768, group="smoke",
             kernel_id="gemm", notes="F.linear bf16 (cuBLAS), M x K x N = 32768 x 768 x 768"),
    ]


def smoke_acceptance(records):
    """Prints the INF-1 acceptance lines; returns True if all pass."""
    cold = records.get("smoke_rmsnorm_768_8k_cold", {})   # flush proof on an L2-resident working set
    warm = records.get("smoke_rmsnorm_768_8k_warm", {})
    cold32 = records.get("smoke_rmsnorm_768_cold", {})
    gemm = records.get("smoke_gemm_32768x768x768", {})
    checks = []
    if cold.get("gbps") and warm.get("gbps"):
        sp = warm.get("speedups", {}).get("smoke_rmsnorm_768_8k_cold", {})
        checks.append(("cold-L2 GB/s < warm-L2 GB/s (8192x768, L2-resident)", cold["gbps"] < warm["gbps"],
                       f"cold {cold['gbps']:.0f} vs warm {warm['gbps']:.0f} GB/s; cold/warm time ratio (warm speedup) "
                       f"{sp.get('ratio', float('nan')):.3f} CI [{sp.get('ci_low', float('nan')):.3f}, "
                       f"{sp.get('ci_high', float('nan')):.3f}]"))
    else:
        checks.append(("cold-L2 GB/s < warm-L2 GB/s (8192x768, L2-resident)", False, "missing rmsnorm records"))
    if gemm.get("tflops"):
        checks.append(("GEMM 32768x768x768 >= 500 TF/s", gemm["tflops"] >= 500,
                       f"{gemm['tflops']:.1f} TF/s ({gemm['tflops'] / 723 * 100:.0f}% of the 723 TF/s 8192^3 ceiling)"))
        cv = gemm.get("cv_trials")
        checks.append(("GEMM trial-to-trial CV < 2%", cv is not None and cv < 0.02,
                       f"cv={cv * 100:.2f}% over {gemm['trials']} trials" if cv is not None
                       else f"needs >= 2 trials (have {gemm.get('trials')})"))
    else:
        checks.append(("GEMM 32768x768x768 >= 500 TF/s", False, "missing gemm record"))
        checks.append(("GEMM trial-to-trial CV < 2%", False, "missing gemm record"))
    if cold32.get("gbps"):
        frac = cold32["gbps"] / COPY_GBPS_GH200
        checks.append(("RMSNorm cold GB/s in 50-100% of 3650", 0.5 <= frac <= 1.0,
                       f"{cold32['gbps']:.0f} GB/s = {frac * 100:.0f}%"))
    else:
        checks.append(("RMSNorm cold GB/s in 50-100% of 3650", False, "missing record"))
    bad = {n: validate_record(r) for n, r in records.items() if validate_record(r)}
    checks.append(("records validate against results/db/README.md schema", not bad and len(records) == 5,
                   f"{len(records)} records" + (f"; problems: {bad}" if bad else "")))
    for what, ok, detail in checks:
        print(f"ACCEPTANCE {'PASS' if ok else 'FAIL'}  {what}: {detail}", flush=True)
    return all(ok for _, ok, _ in checks)


# ----------------------------------------------------------------------------------------------
# CLI shared by every benchmark module
# ----------------------------------------------------------------------------------------------
def print_registry(cases):
    print(f"{'case':<52s} {'form':<9s} {'min_bytes':>12s} {'GFLOP':>10s} {'L2':>5s}  baselines")
    for c in cases:
        mb = f"{c.min_bytes / 1e6:10.1f}MB" if c.min_bytes else "-"
        gf = f"{c.flops / 1e9:10.2f}" if c.flops else "-"
        print(f"{c.name:<52s} {c.form:<9s} {mb:>12s} {gf:>10s} {'cold' if c.flush_l2 else 'warm':>5s}  "
              f"{','.join(c.baselines)}{'  [FA3]' if c.requires_fa3 else ''}")
    print(f"{len(cases)} cases")


def main_cli(suites, module, argv=None, description=None, add_args=None, forward_args=None, on_done=None):
    """suites: {name: build(args) -> [Case]}. add_args(parser) adds module flags; forward_args(args) ->
    list of argv the parent passes to every child; on_done(suite, records) -> bool runs after the trials."""
    p = argparse.ArgumentParser(description=description or __doc__.strip().splitlines()[0])
    p.add_argument("--suite", default=next(iter(suites)), choices=list(suites))
    p.add_argument("--trials", type=int, default=3, help="separate processes, each running the whole suite")
    p.add_argument("--out-dir", default=None, help="default cisc662/results/db/<suite>")
    p.add_argument("--job-id", default=None, help="default $SLURM_JOB_ID")
    p.add_argument("--cases", default=None, help="regex; only cases whose name matches (re.search)")
    p.add_argument("--time-budget-s", type=float, default=None,
                   help="wall budget for all trials; cases not started before it are skipped")
    p.add_argument("--list", action="store_true", help="print the case registry (no CUDA) and exit")
    p.add_argument("--trial", type=int, default=None, help=argparse.SUPPRESS)      # child mode
    p.add_argument("--raw-out", default=None, help=argparse.SUPPRESS)
    p.add_argument("--deadline", type=float, default=None, help=argparse.SUPPRESS)
    if add_args:
        add_args(p)
    args = p.parse_args(argv)

    cases = suites[args.suite](args)
    if args.cases:
        rx = re.compile(args.cases)
        cases = [c for c in cases if rx.search(c.name)]
    names = [c.name for c in cases]
    assert len(names) == len(set(names)), "duplicate case names"
    if args.list:
        print_registry(cases)
        return 0
    if args.trial is not None:          # child: run the whole suite once
        if not args.raw_out:
            p.error("--trial needs --raw-out")
        deadline = args.deadline
        if deadline is None and args.time_budget_s:
            deadline = time.time() + args.time_budget_s
        return run_child(cases, args.suite, args.trial, args.raw_out, deadline)

    extra = list(forward_args(args)) if forward_args else []
    if args.cases:
        extra += ["--cases", args.cases]
    res = run_trials(args.suite, trials=args.trials, out_dir=args.out_dir, job_id=args.job_id, extra_argv=extra,
                     module=module, time_budget_s=args.time_budget_s)
    if on_done:
        on_done(args.suite, res["records"])
    if res["rc"] == FA3_ABORT_RC:
        print("ABORT: an FA3 case was requested but FA3 is not in use (HAS_FA3/USE_FA3 False)", flush=True)
        return FA3_ABORT_RC
    # a failed/timed-out trial is reported above but is not a crash; no trial at all is
    return 0 if res["trials_completed"] > 0 else 1


def _on_done(suite, records):
    if suite == "smoke":
        smoke_acceptance(records)


if __name__ == "__main__":
    sys.exit(main_cli({"smoke": smoke_cases}, module="cisc662.bench.kbench", on_done=_on_done))
