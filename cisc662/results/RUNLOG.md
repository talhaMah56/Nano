# RUNLOG — every job this project ran, with its script, outputs and key numbers

Purpose: the provenance trail behind every number in Presentation 1 (~11-17) and Presentation 2 (~12-17).
Machine-readable twin: [`runlog.csv`](runlog.csv). The exact script submitted, the uncommitted git diff and the untracked
`cisc662/` sources at submission time are archived in `/work/nvme/bffp/tmahmood1/cisc662/logs/submitted/<jobid>_<script>*`.
Job stdout: `/work/nvme/bffp/tmahmood1/cisc662/logs/<job-name>-<jobid>.out`. Per-job results: `results/runs/<tag>_<jobid>/`.

- Event lines (`SUBMIT` / `END` / `QUEUE HALTED`) are appended automatically by `slurm/queue.sh`.
- After analysing a job, add a **Result** line under its `END` line: what it measured, the key numbers, the result files,
  and which slide/figure uses it.

## Session 1 (2026-10-17, submitted by hand before the queue existed; 0.33 SU)
| Job | Script | Task | Outputs | Key numbers |
|---|---|---|---|---|
| 3400653 | `slurm/s1_smoke_tok.sbatch` | smoke + tokenizer | `results/smoke_gh200.json` | copy 3.65 TB/s, bf16 GEMM 723 TF/s |
| 3400686 | `slurm/s1_nsys_train.sbatch` | kernel breakdown | `results/runs/s1_nsys_3400686/` | compile+FA3 74.3 ms/step GPU kernel time; eager 145.3 |
| 3400723 | `slurm/s1_conv_infer.sbatch` | convergence + decode | `results/runs/s1_conv_infer_3400723/` | val bpb 1.0505 @2000; decode b=1 ≈ 83 tok/s |
| 3400724 | `slurm/s1_train_timing.sbatch` | s/100 steps | `results/runs/s1_timing_3400724/` | compile+FA3 8.046 ± 0.031 s; eager+FA3 15.080 |
| 3400725 | `slurm/s1_cpu_ref.sbatch` | CPU reference | `results/runs/s1_cpuref_3400725/` | 1 core 39.86 s/step → GPU 1,320× |

Figures from S1: `results/figures/s1_*.png` (made by `analysis/make_figs_s1.py`).

## Session 2 (started 2026-10-18; cap 1.5 SU) — queue `slurm/queue_s2.txt`
- 2026-10-18 12:34 **SUBMIT** `3406171` c662-s2-inductor (BAS-1) -- script `/u/tmahmood1/project/nanochat/cisc662/slurm/s2_inductor_dump.sbatch` (archived `/work/nvme/bffp/tmahmood1/cisc662/logs/submitted/3406171_s2_inductor_dump.sbatch`), git a43b52f + 10 uncommitted paths (diff + untracked sources archived)
- 2026-10-18 12:34 **SUBMIT** `3406173` c662-s2-noise (BAS-2) -- script `/u/tmahmood1/project/nanochat/cisc662/slurm/s2_noise_band.sbatch` (archived `/work/nvme/bffp/tmahmood1/cisc662/logs/submitted/3406173_s2_noise_band.sbatch`), git a43b52f + 10 uncommitted paths (diff + untracked sources archived)
- 2026-10-18 12:38 **END** `3406171` c662-s2-inductor -- **COMPLETED**, 122s, 0.034 SU, FAILED STEPS: none -- log `/work/nvme/bffp/tmahmood1/cisc662/logs/c662-s2-inductor-3406171.out`
- 2026-10-18 12:40 **END** `3406173` c662-s2-noise -- **COMPLETED**, 261s, 0.072 SU, FAILED STEPS: none -- log `/work/nvme/bffp/tmahmood1/cisc662/logs/c662-s2-noise-3406173.out`
  - **Result 3406171 (BAS-1, raw data; analysis pending `analysis/inductor_fusion.py`)**: 11 Inductor graphs dumped (`runs/s2_inductor_3406171/output_code/`, + `ir_post_fusion.txt`, fx graphs); ncu byte survey of the whole compiled step 20 = 915 kernel launches (`runs/s2_inductor_3406171/ncu_step20_raw.csv`; report `/work/nvme/bffp/tmahmood1/cisc662/profiles/s2_inductor_3406171/ncu_step20.ncu-rep`). ncu default `--cache-control all` ⇒ bytes are cold-L2 per-kernel traffic. Note: `--constraint=nvperf` is a no-op on DeltaAI (counters always on; sbatch warns). Uses: D1, P1 "what Inductor already fuses" slide, roofline v1.
  - **Result 3406173 (BAS-2, preliminary; official numbers from `analysis/losscurve.py`)**: two stock R12-L runs, same seed: max |Δloss| over steps > 100 = 0.00128 (step 437); val bpb @1000 1.148793 vs 1.148788 (Δ 5e-6; max Δ over evals 2.1e-4 @500); 79–80 ms/step. ⇒ G4 tolerance is set by the 0.01 floor (2×band = 0.0026). Uses: G4 for every integrated kernel (INT-2), P1 protocol slide.
- 2026-10-18 13:06 **SUBMIT** `3406282` c662-s2-harness (INF-1,INF-2,INF-3,BAS-3) -- script `/u/tmahmood1/project/nanochat/cisc662/slurm/s2_harness.sbatch` (archived `/work/nvme/bffp/tmahmood1/cisc662/logs/submitted/3406282_s2_harness.sbatch`), git a43b52f + 26 uncommitted paths (diff + untracked sources archived)
- 2026-10-18 13:14 **END** `3406282` c662-s2-harness -- **COMPLETED**, 410s, 0.114 SU, FAILED STEPS: pytest_full -- log `/work/nvme/bffp/tmahmood1/cisc662/logs/c662-s2-harness-3406282.out`
- 2026-10-18 13:27 **SUBMIT** `3406329` c662-s2-rerun (INF-1,INF-2,INF-3) -- script `/u/tmahmood1/project/nanochat/cisc662/slurm/s2_rerun.sbatch` (archived `/work/nvme/bffp/tmahmood1/cisc662/logs/submitted/3406329_s2_rerun.sbatch`), git a43b52f + 32 uncommitted paths (diff + untracked sources archived)
- 2026-10-18 13:31 **END** `3406329` c662-s2-rerun -- **COMPLETED**, 165s, 0.046 SU, FAILED STEPS: pytest_full -- log `/work/nvme/bffp/tmahmood1/cisc662/logs/c662-s2-rerun-3406329.out`
  - **Result 3406282 (INF-1/2/3 + BAS-3, harness v1)**: BAS-3 platform facts are valid: sustained 8192³ bf16 GEMM loop **1365 MHz at ≈ 545 W, `sw_power_cap`, 591 TF/s** (bursts 723); `--gpu-freq=1755` not honoured under the power cap ⇒ clock policy = default clocks + `gpu_monitor.csv` (`results/db/platform/clocks_3406282.json`); pointer-chase DRAM latency **344 ns** at 1 GiB (L2-resident 145–165 ns); decode b=1 256 tokens: CPU 1 thread **33.6 tok/s** (cv 0.8%), GPU **104 tok/s** (85–104) ⇒ **3.1×** (2.5–3.2) (`results/db/decode/s2_remeasure.json`); FA3 error vs fp32 SDPA 9.1e-3 (`results/db/tolerances/attention_stock_error.json`). **Its ops_baseline and smoke timings are INVALID** (host overhead inside the event window; archived in `db/*/by_job/3406282/`, superseded by 3406329). pytest: 100 passed, 7 xfailed, 1 failed (`test_qkrope_g1[r12]`: stock bf16 itself misses the §7.3 rtol/atol rule). Uses: P1 platform slide (power cap, latency), goal (i) decode, protocol slide.
  - **Result 3406329 (INF-1/INF-3 re-measured with the kbench queue pad)**: smoke acceptance 5/5 PASS (flush proof on L2-resident 8192×768: warm 1.35× faster than cold, CI [1.336, 1.355]; GEMM 32768×768×768 646 TF/s, cv 0.45%; RMSNorm cold 2354 GB/s = 64% of copy). **ops_baseline: 54 cases × 3 trials, all with CIs** (`results/db/ops_baseline/*.json`); falsification check passed (compiled GEMMs 0.99–1.01× eager; trial cv ≤ 0.5% for 46/54 cases, max 2.0%). Headline eager→compile ratios: CE fwd+bwd 7.8×, lm_head+CE 3.0×, qkrope fwd+bwd 6.0×, mixnorm fwd+bwd 3.9×, RMSNorm(768) 1.00×, FA3 vs SDPA fwd+bwd 2.0×. pytest still 1 failed (`test_qkrope_g1[r12]`, a test-logic bug fixed on the login node after this job; GPU re-check rides in S3). Uses: P1 baseline-per-op table, every KT card's vs-eager/vs-compile baseline.
