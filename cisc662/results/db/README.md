# results/db — kernel microbenchmark results

Written by `cisc662/bench/kbench.py` (`run_trials`), used by every suite: `kbench --suite smoke` (INF-1),
`ops_at_shapes` (INF-3, BAS-4) and the kernel cards (KT-*, KC-*). Protocol: PLAN.md §7.1. Plotting scripts read
only these files, and refuse rows whose `gates_passed` lacks G1–G2 (G3–G4 for integrated rows), PLAN.md §7.3.

## Layout
```
results/db/<group>/<case>.json                         latest result of each case (the file other scripts read)
results/db/<group>/by_job/<job_id>/<case>.json         the same record, kept per job (never overwritten by another job)
results/db/<group>/by_job/<job_id>/raw/trial<k>.json   raw per-rep times + preamble written by trial process k
```
`<group>` is a kernel_id (`rmsnorm`, `qkrope`, `ce`, `ce_chunked`, `gemm`, `mlp`, `attention`, `muon`) or a suite
group (`smoke`, `ops_baseline`, `platform`, `decode`, `loss`). `<case>` is `<form>_<op>_<shape>`, e.g.
`compile_rmsnorm_fwd_32768x768`; the form is `eager`, `compile` (`torch.compile(fn, dynamic=False)`), `autotune`
(`mode="max-autotune-no-cudagraphs"`) or a kernel variant name (e.g. `triton_v1`).

When a new job writes a case whose latest file came from another job, the old file is first copied to
`by_job/<old job_id>/` (with its `gates_passed`), so no job's results are lost. If a case produced no data in a job
(error or time-budget skip), only `by_job/<job_id>/<case>.json` is written and the previous latest file stays.

## Case record (`<case>.json`)
Fields required by the schema (`kbench.validate_record`) are marked **R**.

| Field | Type | Unit / meaning |
|---|---|---|
| `case` **R** | str | case name (= file name without `.json`) |
| `dtype` **R** | str | main input dtype, e.g. `bfloat16` |
| `shape` **R** | {name: [int]} | input tensor shapes (`[]` = 0-d tensor) |
| `median_us` **R** | float \| null | median over trials of the per-trial median rep time, µs (null only if no trial produced data) |
| `p10_us`, `p90_us` **R** | float \| null | 10th / 90th percentile of all reps pooled over trials, µs |
| `min_us` | float \| null | fastest single rep, µs |
| `gbps` **R** | float \| null | achieved bandwidth = `min_bytes` / `median_us`, GB/s (10^9 B/s) |
| `tflops` **R** | float \| null | `flops` / `median_us`, TFLOP/s (10^12) |
| `trials` **R** | int | number of trial processes that produced data for this case |
| `trial_medians_us` | [float] | the per-trial medians, µs |
| `cv_trials` | float \| null | sample std / mean of `trial_medians_us` (null with one trial); INF-1 requires < 0.02 on the GEMM |
| `trial_times_us` | [[float]] | every timed rep, per trial, µs (input of any later bootstrap) |
| `speedups` **R** | {baseline_case: {ratio, ci_low, ci_high}} | ratio = baseline median / this case's median (> 1 = this case is faster); 95% bootstrap CI, 10,000 resamples stratified by trial (`analysis/stats.py`). One entry per baseline in this job; "≥ 1.3×" may be claimed only if `ci_low` ≥ 1.3 |
| `gates_passed` **R** | [str] | correctness gates passed (`G1`..`G4`), appended by `tests/gates.py record_gate` after the job |
| `env` **R** | object | provenance, see below |
| `job_id` **R** | str \| null | Slurm job id (`nojob-<timestamp>` outside Slurm) |
| `group`, `kernel_id`, `form` | str | as in the layout above |
| `min_bytes` | float \| null | algorithmic minimum DRAM bytes of the op (analytic; ncu `dram__bytes` is reported separately) |
| `flops` | float \| null | analytic FLOP count (GEMM/attention exact; elementwise/reduction ops approximate op counts) |
| `flush_l2` | bool | true = 256 MB buffer zeroed before every rep (cold L2), false = warm L2 |
| `warmup`, `reps` | int | untimed warm-up reps after the untimed first call, timed reps per trial |
| `notes` | str | what exactly is timed and how min_bytes was counted |
| `status_per_trial` | [str] | `ok`, `skipped_time_budget` or `error: <message>` per trial |
| `host_per_trial` | [object] | per trial `{trial, pad_us, host_warmup_median_us, host_median_us, host_bound}`: queue pad used; median host enqueue time of one call measured with an empty queue in the warm-up (clean) and in the timed loop (can include launch-queue-full stalls); host_bound = warm-up host time > device median |
| `host_bound` | bool | true if any trial was host-bound: the op is launch/host-limited in a real loop even though its device time is shown |

`env` (from trial 1, plus a per-trial list): `git_sha`, `git_dirty`, `git_dirty_paths`, `torch`, `triton`, `cuda`,
`cudnn`, `python`, `host`, `slurm_job_id`, `cuda_visible_devices`, `nanochat_attn_impl`, `HAS_FA3`, `USE_FA3`,
`device`, `capability`, `sm_count`, `gpu_uuid`, `bf16_reduced_precision_reduction`, `l2_flush_bytes`, `clock_mode`,
and `trials: [{trial, host, complete, elapsed_s, smi_start, smi_end}]` where `smi_*` are nvidia-smi snapshots
(`clocks.sm` MHz, `clocks.mem` MHz, `power.draw` W, `temperature.gpu` °C, throttle/event reasons bitmask, `time`).

## Raw trial file (`by_job/<job_id>/raw/trial<k>.json`)
`{suite, trial, pid, started, env (incl. smi_start/smi_end), complete, elapsed_s, abort?, cases: {case: {meta,
status, times_us, median_us, pad_us, host_warmup_median_us, host_median_us, host_bound, setup_s, case_s, graphs_compiled, max_mem_gib, error?}}}`. Rewritten atomically after
every case, so a killed trial keeps its finished cases. `graphs_compiled` = Dynamo graphs compiled during the case
(0 for eager; ≥ 1 proves a compile row really ran compiled).

## Timing method (kbench.bench)
One untimed first call (JIT / autotune / torch.compile) → synchronize → `warmup` untimed reps → `reps` timed reps,
each a CUDA event pair around the op only; with `flush_l2` the flush buffer is zeroed before the start event of
every rep; then a **queue pad** (`torch.cuda._sleep`, 2× the host enqueue time measured in the warm-up, ≥ 200 µs)
is queued before the start event so the CPU is always ahead of the GPU and the event pair measures device time
only (added after job 3406282, whose compiled fwd+bwd rows were inflated by host overhead: e.g. compiled RMSNorm
fwd+bwd 349 µs vs 122 µs eager; rows from 3406282 are archived under `by_job/3406282/` and must not be used);
one synchronize after the loop, then the event times are read. Each trial is a separate process that
runs the whole suite once. Known bias: the zeroed flush lines are dirty in L2 and are written back during the timed
op (same for every variant, as in `triton.testing.do_bench`). Clocks are the default (not pinned) unless `clock_mode`
says otherwise.
