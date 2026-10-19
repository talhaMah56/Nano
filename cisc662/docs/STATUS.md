# STATUS — read this first when resuming

Last session: **S2 + S3 — 2026-10-18 (both moved up from 10-20 / 10-23) — Talha (with Claude)**. Plan: [PLAN.md](PLAN.md) · tasks: [TASKS.md](TASKS.md) · rules: [AGENT_GUIDE.md](AGENT_GUIDE.md) · findings: [findings.md](findings.md) · every job: [../results/RUNLOG.md](../results/RUNLOG.md) · access drafts: [access_requests.md](access_requests.md).
SU: S2 **0.263** (cap 1.5) · S3 **0.061** (cap 1.0; sacct-based queue estimate 0.064) | cumulative **0.658** (ledger) / planned 25 (hard cap 40; pre-P1 limit 10, re-plan checkpoint 8) | pool balance 6,604 SU (`accounts`, 10-18 22:40). `jobcharge` additionally shows 0.28 SU of HW2 jobs on 10-12 (not this project).
In-flight jobs: **none**. Queue runner: `slurm/queue.sh`; S3 queue drained (tmux `cisc662` on **gh-login03**).
Storage: ≈ 11 GB on `/work/nvme/bffp/tmahmood1/cisc662` (+64 MB S2 ncu/debug profiles). Git: S1 `a43b52f` (Talha); S2 `0216189` + `1144de5` (committed 10-18 with the user's go-ahead; not pushed).

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
- **ADM-2 done:** `HF_XET_CHUNK_CACHE_SIZE_BYTES=0` in env.sh; deleted `scratch/toolchain-feasibility/{fa3_download,hf_home}` and `cache/hf/xet` (≈ 354 MB; the first attempt was blocked by the auto-mode classifier, retried after the user's go-ahead); FA3 blob in `cache/hf/hub` intact; footprint 9.9 GB; `r12_conv` intact.
- Fixed `run_step` logging rc=0 on failures (`_common.sh`).

## Done in S3 (2026-10-18; jobs 3409675, 3409677 + login-node work)
- **KT-1** `kernels/triton/rmsnorm.py` (v1, v2, mixnorm v3 + bwd; `cisc662::rmsnorm`/`mixnorm` triton_ops), `tests/test_rmsnorm.py`, `bench/suite_rmsnorm.py`; **KT-2** `kernels/triton/qkrope.py` (q+k one launch, fwd + single-kernel bwd, fold_scale), `tests/test_qkrope.py`, `bench/suite_qkrope.py`. Pre-GPU adversarial review fixed 2 blockers (`findings.md#kt-review-s3`). **GPU (jobs 3409675 / 3409677): G1+G2 passed** (188 / 94 tests); **qkrope 1.33× vs compile fwd [1.32,1.33], 2.08× fwd+bwd [2.07,2.08]**, 12–15× vs eager; RMSNorm-768 v2 1.31× vs eager and compile (85% of roof), 128-wide parity, mixnorm 1.08–1.10× vs compile. ISA cards `findings.md#rmsnorm`, `#qkrope`. Gotcha: the Triton 3.5.1 interpreter truncates fp32→bf16 (GPU rounds to nearest even), so bf16 gates are meaningless under the interpreter.
- **AMD-1** `analysis/{kernel_registry,xcompile,isa_report}.py` → `results/static_isa.csv` (358 STATIC rows: 16 kernels × 4 targets + 294 tuning pairs, 0 failures), dumps `results/isa/<kid>/triton_<v>/<target>/xc-<config>/`, `findings.md#static-isa`. Rerun after any kernel change: `python -m cisc662.analysis.xcompile --set all --kernels rmsnorm,qkrope --force` then `python -m cisc662.analysis.isa_report` (≈ 30 CPU-s).
- **ANA-1 roofline v0** (`analysis/roofline.py`, `results/figures/roofline_v0.png`, `results/roofline_points_v0.csv`; acceptance met; `findings.md#roofline-v0`).

## Gates status
No optimized kernels yet. Harness ready: G1/G2 machinery (`tests/tolerances.py`, `gates.py`), G4 machinery (`analysis/losscurve.py` + envelope). Tolerances final under amendment A1 (Decisions).

## Decisions
- **D1 (S2, from BAS-1):** vs-compile targets — softcap+CE fwd 2.4–3.0× (Inductor at 30% of the roof), CE single-pass tail 2.4–2.9×; QK-norm+RoPE bwd 1.7–2.1× (2.09× the minimum bytes), fwd 1.3–1.6×; mixnorm/RMSNorm, relu², AdamW = parity (Inductor at the roof); Muon non-GEMM pointwise moves 4.8× its minimum (KT-8 stronger than planned; revisit at D4).
- **KT-2 design (S3):** fused qkrope rotates in fp32 and takes rstd from the rotated row (not the card's un-rotated identity, which bf16 cos/sin break by ~1 ulp); recorded in the KT-2 card.
- **Clock policy (BAS-3):** default clocks + `gpu_monitor.csv`; `--gpu-freq` is not honoured under the power cap; no pinned-clock repeats.
- R12 frozen (PLAN.md §3). Timing metric = sum of dt over steps 10–109.
- Budget: ≤ 10 SU before P1, 25 SU planned, 40 hard cap (PLAN.md §2).
- AMD half: static tier guaranteed; measured tier only if Delta MI100/MI210 access arrives (D2 on 2026-11-03).
- **PLAN §7.3 amendment A1 (decided 2026-10-18, user delegated the call):** bf16 outputs are judged against the stock op where it cannot meet the absolute rule (max-abs ≤ 2× stock, cosine ≥ 0.9999, rtol/atol violations ≤ 2× stock's count — strict whenever stock meets it); attention ≤ 2× FA3 (4e-3 term dropped); dlogits rel-L2 ≤ 1e-2 and ≤ 2× stock. Implemented in `tests/tolerances.py`; login subset 75 passed / 0 xfail; GPU confirmation rides in the S3 job. Rationale: `findings.md#tolerance-a1`.
- Pending: D2 (access, 11-03), D3 (CE ≥ 1.3× vs compile, S5), D4 (scope freeze, S9), D5 (per-kernel go/no-go, S12), D6 (code freeze, 12-08).

## Access status
| Route | Status |
|---|---|
| **All six routes** | drafts ready to send in [access_requests.md](access_requests.md) (2026-10-18); **not sent** — needs Talha / a faculty PI |
| NCSA Delta GPU (gpuMI100x8: MI100 + MI210; gpuA100x4) via ACCESS exchange | not yet requested (ADM-1) |
| DARWIN (gpu-v100, gpu-mi100) Education allocation | not yet requested |
| Chameleon CHI@TACC gpu_mi100 | not yet requested |
| AMD Developer Cloud MI300X credit | not yet requested |
| Hamza's DeltaAI account | **none** in `bffp-dtai-gh` (checked 2026-10-18: 23 users, only `tmahmood1`); asked in the instructor draft |

## Known gotchas
- **Login node is not fixed:** a VS Code/Claude reconnect can land on gh-login01 *or* gh-login03; tmux lives only on the node where it was started and `ssh gh-login0X` needs Duo (non-interactive ssh fails). Check `hostname` first; the S3 queue runs in tmux `cisc662` on **gh-login03** (S2's on gh-login01, exited). Queue state is on the shared filesystem, so a runner may be restarted on any node — just never run two runners on the same queue file.
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

## Next session (S4, planned Tue 10-27; cap 1.9 SU) — first actions
1. **User (only open human item):** send the ADM-1 requests — drafts in [access_requests.md](access_requests.md); log dates in the Access table; follow-up 10-27, D2 on 11-03.
2. **KT-3** fused softcap+CE (`kernels/triton/ce.py`, `tests/test_ce.py`, `bench/suite_ce.py`; gate dlogits with `kind="dlogits"`, A1) — the headline kernel; BAS-1 target 2.4–3.0× fwd, 1.6–2.9× tail vs compile. Pin FMA contraction explicitly in any recompute (`findings.md#kt-review-s3`). Job `slurm/s4_ce.sbatch` (copy the s3 body pattern).
3. **BAS-6** ncu baseline of the top-15 compiled kernels (`--nvtx --target-processes application-only`, see s2_inductor_dump) **plus ncu bytes of `c662_qkrope_*` and `c662_rmsnorm_*`** (KT-2 acceptance still needs ≈ 200 MB/call measured) → ANA-1 roofline v1.
4. **BAS-4** only needs the max-autotune GEMM rows now (eager/compile rows exist from INF-3, job 3406329); **BAS-5** CPU per-op references (CPU-only jobs, ≈ 0.35 SU).
5. Explain the 5% triton_op vs Function gap in qkrope with one nsys capture (fold into BAS-6's job).
6. Login node: AMD-2 can start (HIP RMSNorm via hipify + CCE), AMD-1 static rows auto-refresh with `xcompile --force` after kernel changes.
