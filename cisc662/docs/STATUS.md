# STATUS — read this first when resuming

Last session: **S2 — 2026-10-18 (moved up from Tue 10-20) — Talha (with Claude)**. Plan: [PLAN.md](PLAN.md) · tasks: [TASKS.md](TASKS.md) · rules: [AGENT_GUIDE.md](AGENT_GUIDE.md) · findings: [findings.md](findings.md) · every job: [../results/RUNLOG.md](../results/RUNLOG.md).
SU: S2 **0.263** (cap 1.5) | cumulative **0.597** / planned 25 (hard cap 40; pre-P1 limit 10, re-plan checkpoint 8) | pool balance 6,604 SU (`accounts`, 2026-10-18 13:35). `jobcharge -u tmahmood1 -d 30` shows 0.28 SU more: those are HW2 jobs on 10-12 (`hw2_run`/`hw2_ncu`, ghx4-interactive), not this project.
In-flight jobs: **none**. Queue runner: `slurm/queue.sh` (tmux `cisc662`, windows `queue`/`queue2`; drained and exited).
Storage: ≈ 11 GB on `/work/nvme/bffp/tmahmood1/cisc662` (+64 MB S2 ncu/debug profiles). Git: S1 committed as `a43b52f` on `cisc662` (by Talha, 10-18 12:14); **all S2 work is uncommitted** (needs the user's OK).

## How to resume
```bash
ssh gh-login01.delta.ncsa.illinois.edu            # tmux lives on this node only
tmux attach -t cisc662 || tmux new -s cisc662     # optional: the tmux server was found dead on 10-18; batch jobs never depend on it
source /u/tmahmood1/project/nanochat/cisc662/env/env.sh
accounts | grep bffp; tail -3 $CISC662_WORK/logs/job_ledger.csv; squeue -u $USER
```

## Done in S1
- Environment: uv venv (Python 3.10.22, torch 2.9.1+cu128, triton 3.5.1, kernels 0.11.7) on /work/nvme; FA3 `varunneal/flash-attention-3` aarch64 build cached (sha `de87b9b5af06`), jobs run `HF_HUB_OFFLINE=1`; 12+1 ClimbMix shards; tokenizer trained (vocab 32768, 4.74 chars/token).
- Upstream patches (opt-in, defaults unchanged, uncommitted): `nanochat/flash_attention.py` (`NANOCHAT_ATTN_IMPL=fa3|sdpa`), `scripts/base_train.py` (`NANOCHAT_PROFILE_STEPS=a:b` → cudaProfilerStart/Stop + NVTX step/forward/backward/optimizer ranges; `NANOCHAT_EMIT_NVTX=1`), `nanochat/common.py` (GH200 peaks 989 TF/s, 4.0 TB/s — the name otherwise matched "h200" = 4.8 TB/s).
- Tooling: `slurm/_common.sh` (ledger, `run_step`, `use_local_base_dir`), `bench/smoke.py`, `bench/decode_bench.py` (CPU/GPU decode), `profiling/nsys_attrib.py` (SQLite → phase × class attribution), `profiling/classify_kernels.py`, `analysis/summarize_s1.py`, `analysis/make_figs_s1.py` (5 P1-ready PNGs in `results/figures/`).
- Research (5 reports in `docs/research/`): DeltaAI policy/billing, toolchain feasibility (Triton cross-compile to sm_90/gfx908/gfx90a/gfx942 works on the login node; CCE clang 20 compiles HIP→AMDGCN without ROCm), nanochat hot-path inventory (per-op bytes/FLOPs, 10 fusion opportunities), AMD access routes, prior art with realistic speedup ranges.
- Plan written: PLAN.md, TASKS.md (52 task cards), AGENT_GUIDE.md; reviewed on 10-18 by four critics (proposal fidelity, platform/budget, agent executability, technical correctness) plus a judge; all 5 verified must-fix and all should-fix items applied (ncu profiler window, goal-(i) GPU figure, no-upstream-edit integration, fp32 RMSNorm eps, Inductor CE-tail mechanism, naming/layout conventions, HIP recipe, budget rules).

## Results so far (all R12 unless noted; details in `results/runs/*/summary.md`)
| Job | What | Key numbers |
|---|---|---|
| 3400653 | smoke + ceilings + tokenizer | GH200: 132 SMs, 95 GiB, 60 MiB L2; copy 3.65 TB/s, read 3.40; GEMM bf16 723 / fp16 694 / tf32 382 / fp32 48.6 TF/s; nsys+ncu work for users; compute nodes have internet |
| 3400686 | nsys 5-step kernel breakdown × 5 variants | GPU kernel ms/step: eager+FA3 145.3, compile+FA3 74.3, compile+SDPA 85.9. compile: GEMM 37.2, attention 10.8, optimizer 7.6, RMSNorm 5.3, CE 4.5, relu² 3.3, RMSNorm+RoPE 2.8. eager: GEMM 34.2, CE 29.6, optimizer 19.4, elementwise 17.1, RoPE 14.0, attention 10.1, RMSNorm 9.4, relu² 7.9. `attrib_*.json` |
| 3400724 | s/100 steps, 3 reps each | compile+FA3 **8.046 ± 0.031**; compile+SDPA 9.114 ± 0.015; eager+FA3 15.080 ± 0.018; eager+SDPA 15.983 ± 0.008; peak mem 14.8 / 29.4 GiB; SM clock median 1665 MHz when busy, 524 W |
| 3400723 | 2000-step convergence (`r12_conv`) + infer_bench | val bpb 3.158 → 1.0505; checkpoint `$CISC662_WORK/nanochat_base/base_checkpoints/r12_conv`. Decode (prompt 512, decode 256): b=1 ≈ 83 tok/s (TPOT 11.6 ms), b=8 ≈ 720, b=32 ≈ 2.7k, b=128 ≈ 10.5k; MBU 2–9%; TPOT flat in batch ⇒ launch-bound |
| 3400725 | CPU reference (Grace Neoverse-V2), batch 1 × 2048 | 1 thread 39.86 s/step (3,986 s/100); 72 threads 2.27 s/step; GPU b=1: compile 3.02 s/100, eager 5.57 → GPU/1-core **1,320×**, GPU/72-core 75×. Decode b=1: CPU 1 thr 32.8 tok/s, 72 thr 89.2, GPU 107 (only 3.3×) |

Facts that set vs-compile targets (from traces + the Inductor-generated code in the project cache): eager `F.rms_norm` is already one fused kernel; Inductor fuses RoPE+QK-norm+scale forward into one kernel but at only ≈ 1.8 TB/s; Inductor's CE tail is 2 reduction kernels ≈ 6.4 GB/step (bf16 logits saved, softcap+log-softmax recomputed in the backward from per-row max/lse, dlogits written in place) with the forward at ≈ 0.95 TB/s — the vs-compile headroom is achieved bandwidth, not bytes; weight casts cost 0.66 GB/forward and 73% of decode bytes at b=1. Decode GPU b=1 varied 68–107 tok/s between S1 jobs (BAS-3 re-measures). `F.rms_norm(eps=None)` uses the fp32 epsilon 1.19e-7 in torch 2.9.1 (not the bf16 eps).

## Done in S2 (2026-10-18; jobs 3406171, 3406173, 3406282, 3406329 — details and outputs in `results/RUNLOG.md`)
- **Job queue + provenance:** `slurm/queue.sh` (bounded, ≤ 2 in flight, session-cap and 8-SU checks before every submit, halts on any non-COMPLETED job, archives script + git diff + untracked sources to `$CISC662_WORK/logs/submitted/`) with `slurm/queue_s2.txt`; `results/RUNLOG.md` + `results/runlog.csv` record every job (S1 back-filled).
- **INF-1** `bench/kbench.py` (process-level trials, L2 flush, **queue pad**, host-time fields), `analysis/stats.py` (stratified bootstrap), `results/db/README.md`; smoke acceptance 5/5 PASS (3406329). **INF-3** `bench/ops_at_shapes.py`: 54 cases (27 ops × eager/compile) × 3 trials with CIs in `results/db/ops_baseline/` (3406329; the 3406282 rows are invalid, archived). **INF-2** `tests/` (ref_ops, tolerances, conftest, gates, 109 tests): GPU run 100 passed, 7 xfailed, 1 failed → the failure was test logic, fixed on the login node (CPU replay + login subset green; the GPU re-check rides in the S3 job). Eps fact confirmed on CUDA (bit-exact RMSNorm test passed).
- **BAS-1 / D1** (3406171): `analysis/inductor_fusion.py` → `results/inductor_fusion_table.csv`, `findings.md#inductor-baseline`; KT-1..KT-4 cards now carry measured "Target vs compile" lines.
- **BAS-2** (3406173): `analysis/losscurve.py`; envelope 0.00128, val-bpb Δ 5e-6 ⇒ G4 runs on its floors (0.01 / 0.005); `results/db/loss/noise_envelope.json`, `results/figures/loss_noise_band.png`.
- **BAS-3** (3406282): `bench/platform_probe.py`, `analysis/decode_remeasure.py`, `slurm/s2_harness.sbatch`; `results/db/platform/clocks_3406282.json`, `results/db/decode/s2_remeasure.json`; `findings.md#platform-s2`.
- **ADM-2 partly:** `HF_XET_CHUNK_CACHE_SIZE_BYTES=0` in env.sh; the three deletions (≈ 354 MB) were **blocked by the Claude Code auto-mode classifier** — the user runs them (command in "Next").
- Fixed `run_step` logging rc=0 on failures (`_common.sh`).

## Gates status
No optimized kernels yet. Harness ready: G1/G2 machinery (`tests/tolerances.py`, `gates.py`), G4 machinery (`analysis/losscurve.py` + envelope). Tolerances final under amendment A1 (Decisions).

## Decisions
- **D1 (S2, from BAS-1):** vs-compile targets — softcap+CE fwd 2.4–3.0× (Inductor at 30% of the roof), CE single-pass tail 2.4–2.9×; QK-norm+RoPE bwd 1.7–2.1× (2.09× the minimum bytes), fwd 1.3–1.6×; mixnorm/RMSNorm, relu², AdamW = parity (Inductor at the roof); Muon non-GEMM pointwise moves 4.8× its minimum (KT-8 stronger than planned; revisit at D4).
- **Clock policy (BAS-3):** default clocks + `gpu_monitor.csv`; `--gpu-freq` is not honoured under the power cap; no pinned-clock repeats.
- R12 frozen (PLAN.md §3). Timing metric = sum of dt over steps 10–109.
- Budget: ≤ 10 SU before P1, 25 SU planned, 40 hard cap (PLAN.md §2).
- AMD half: static tier guaranteed; measured tier only if Delta MI100/MI210 access arrives (D2 on 2026-11-03).
- **PLAN §7.3 amendment A1 (decided 2026-10-18, user delegated the call):** bf16 outputs are judged against the stock op where it cannot meet the absolute rule (max-abs ≤ 2× stock, cosine ≥ 0.9999, rtol/atol violations ≤ 2× stock's count — strict whenever stock meets it); attention ≤ 2× FA3 (4e-3 term dropped); dlogits rel-L2 ≤ 1e-2 and ≤ 2× stock. Implemented in `tests/tolerances.py`; login subset 75 passed / 0 xfail; GPU confirmation rides in the S3 job. Rationale: `findings.md#tolerance-a1`.
- Pending: D2 (access, 11-03), D3 (CE ≥ 1.3× vs compile, S5), D4 (scope freeze, S9), D5 (per-kernel go/no-go, S12), D6 (code freeze, 12-08).

## Access status
| Route | Status |
|---|---|
| NCSA Delta GPU (gpuMI100x8: MI100 + MI210; gpuA100x4) via ACCESS exchange | not yet requested (ADM-1) |
| DARWIN (gpu-v100, gpu-mi100) Education allocation | not yet requested |
| Chameleon CHI@TACC gpu_mi100 | not yet requested |
| AMD Developer Cloud MI300X credit | not yet requested |
| Hamza's DeltaAI account | unknown — check `sacctmgr show assoc account=bffp-dtai-gh format=user` |

## Known gotchas
- **kbench timings need the queue pad** (now built in): without it, host overhead of compiled/autograd calls leaks into CUDA-event times (job 3406282 rows invalid). Never compare cold numbers across 3406282 and later jobs.
- **GH200 power cap:** sustained tensor-core work runs at ≈ 1365 MHz / 545 W (`sw_power_cap`), bursts at 1980 MHz; always log clocks, interleave variants.
- `--constraint=nvperf` is a no-op on DeltaAI (counters always on; sbatch prints a warning mentioning perf/xpmem — harmless, billing stays 1000 = 1 SU/h).
- ncu with `--nvtx --target-processes application-only` inside the profiler window profiles one whole compiled step (915 launches) in ≈ 25 s; steps under ncu run at ≈ 138 ms instead of 80 ms (interception), irrelevant for counters.
- **Login-node auto-killer (verified 2026-10-17 by an NCSA policy email):** any process outside Slurm that exceeds 30 CPU-minutes is killed. The victim was a filesystem-wide `bfs … / -name claude_skills` search spawned by the Claude Code harness (36 CPU-min). The tmux server was gone by 01:00 on 10-18. Rules now in AGENT_GUIDE.md §1: no searches from `/`/`$HOME`, split login-node batches, no perpetual loops, restart long Claude sessions. Check the VS Code Claude extension / skills settings for a bare `claude_skills` path that triggers a root search.
- S1 jobs (≤ 45 min) started 8–29 s after becoming eligible; the one 1-h job was cancelled after 8 min pending with a start estimate 8 h out. Keep ≤ 30 min and submit a session's jobs together.
- ncu ignores the `NANOCHAT_PROFILE_STEPS` window unless run with `--profile-from-start off`; a compiled R12 step has ≈ 915 kernel launches.
- `uv sync` must be run as `uv sync --frozen --inexact --extra gpu --group dev` or it rewrites `uv.lock` and removes pandas/ninja/tabulate.
- Login shells export `CC=cc CXX=CC` (Cray); `env.sh` sets gcc/g++. Login-node cgroup: 16 cores, 20 GiB OOM, 1000 tasks.
- `infer_bench.py` asserts CUDA; use `cisc662/bench/decode_bench.py` for CPU decode.
- `test`/`full` partitions are staff-only; `ghx4-interactive` bills 2× and allows one job.
- Scratch to delete (ADM-2): `$CISC662_WORK/scratch/toolchain-feasibility/{fa3_download,hf_home}` (~233 MB).

## Next session (S3, Fri 2026-10-23; cap 1.0 SU) — first actions
1. **User:** ADM-1 access requests (six routes; D2 on 11-03) — still not sent. Decide the PLAN §7.3 tolerance amendment (Decisions). OK to commit S2 work? Run the ADM-2 deletions: `rm -rf $CISC662_WORK/scratch/toolchain-feasibility/{fa3_download,hf_home} $CISC662_WORK/cache/hf/xet` (FA3 itself lives in `cache/hf/hub`, verified; untouched).
2. KT-1 (RMSNorm/mixnorm) + KT-2 (qkrope) into `slurm/s3_kernels.sbatch`; its first step is `pytest cisc662/tests -q -rxs` (re-checks `test_qkrope_g1[r12]` on GPU). Baselines to beat are in `results/db/ops_baseline/` (3406329): e.g. qkrope fwd+bwd compile 317 µs, eager 1900 µs; mixnorm fwd+bwd compile 181 µs; RMSNorm(768) fwd 42.6 µs eager = compile.
3. AMD-1 static pipeline and ANA-1 roofline v0 (login node; split work so no process passes ~20 CPU-min).
4. Submit through the queue: append lines to a new `slurm/queue_s3.txt`, start `bash cisc662/slurm/queue.sh cisc662/slurm/queue_s3.txt 1.0` in tmux, add a **Result** line to RUNLOG.md after each job.
