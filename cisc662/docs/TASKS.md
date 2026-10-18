# TASKS — executable task cards for the CISC662 nanochat project

How to use: pick a card whose `Depends on` deliverables exist (check STATUS.md), brief an agent with
> "Execute task **<ID>** from `cisc662/docs/TASKS.md`. Follow `cisc662/docs/AGENT_GUIDE.md` exactly. Read `cisc662/docs/STATUS.md` first and update it when done. GPU budget for this task: <cap> SU."

Every card has: Goal · Session · Depends on · GPU budget (cap, in SU = GH200-hours) · Inputs · Steps · Deliverables (exact paths) · Acceptance (what must be true, verified not assumed) · Pitfalls. Where a card has no separate Deliverables line, the paths named in its Steps and Acceptance **are** the deliverables. Workstreams: **ADM** admin, **INF** infrastructure, **BAS** baselines/measurement, **KT** Triton kernels, **KC** CUDA kernels, **AMD** HIP/AMDGCN, **INT** integration and end-to-end, **DEC** decode, **TUNE** tuning, **ANA** analysis/presentations. Platform facts and protocol: PLAN.md §2–§7. Reduced config R12 and workload IDs: PLAN.md §3.

**Conventions every card follows**
- **kernel_id** ∈ {`rmsnorm` (incl. the `mixnorm` fused variant), `qkrope`, `ce`, `ce_chunked`, `gemm`, `mlp` (relu² epilogue), `attention`, `muon`}. Source files: `kernels/{triton,cuda,hip}/<kernel_id>.*`. Kernel functions (what nsys sees): `c662_<kernel_id>_<fwd|bwd>[_<variant>]`. Results: `results/db/<kernel_id>/<case>.json`. ISA dumps: `results/isa/<kernel_id>/<source>/<target>/` with source ∈ {`triton_<variant>`, `inductor`, `cuda_<variant>`, `hip`, `nvcc`, `clang_nvptx`} and target ∈ {`sm_70`, `sm_80`, `sm_90a`, `gfx908`, `gfx90a`, `gfx942`}. Findings anchors: `docs/findings.md#<kernel_id>`.
- **Per-session job scripts**: kernel, decode and tuning cards do not get their own sbatch; they append `run_step` lines to the session's script (`slurm/s3_kernels.sbatch` = KT-1 + KT-2; `s4_ce.sbatch` = KT-3; `s5_kernels.sbatch` = KT-4 + KT-5; `s7_kernels.sbatch` = KC-1 + KC-2 + KT-6; `s10_kernels.sbatch` = KC-3 + KT-7; `s11_decode.sbatch` = DEC-1 + DEC-2; `s13_kernels.sbatch` = KC-4 + TUNE-1 [+ KT-8]). One GPU, ≤ 30 min, ≤ 2 project jobs queued at once.
- **Login-node work** (cross-compiles, analysis, tiny tests) obeys the 30-CPU-minute-per-process rule in AGENT_GUIDE.md §1; real-shape tests and anything with torch.compile run only in jobs.
- **Budget**: CPU-only jobs still draw SU (72 cores bill the GPU rate; 2 cores + 16 GB ≈ 0.144 SU/h). `/sw/user/scripts/jobcharge -u tmahmood1 -d 30` is the authoritative spend; the ledger (`$CISC662_WORK/logs/job_ledger.csv`, SU column from the job's billing weight) is the per-job cross-check; `srun` probes that bypass `_common.sh` are logged by hand in STATUS.md.

Common inputs: `docs/research/research_hotpath-inventory.json` (per-op shapes, bytes, FLOPs, fusion opportunities), `research_prior-art.json` (realistic speedups, design hints), `research_toolchain-feasibility.json` (Triton/CCE/ncu recipes), `research_amd-path.json` (AMDGCN reading, occupancy tables, MFMA facts), `research_deltaai-policy.json` (billing, limits). S1 measured data: `results/runs/s1_*/summary.md`, `results/runs/s1_nsys_3400686/attrib_*.json`, `results/smoke_gh200.json`.

---

## ADM — administration and access

### ADM-1 Send the hardware-access requests (human task, first hour of S2)
- **Goal:** start every route to AMD and V100/A100 hardware in parallel; nothing in the plan waits for them.
- **Session:** S2 · **Depends on:** — · **GPU budget:** 0
- **Steps:**
  1. Email the course instructor: (a) platform change (GH200 primary, DARWIN unreachable), (b) ask whether a static-only AMD half (clearly labelled) is acceptable if no hardware arrives, (c) ask for a DARWIN Education allocation covering `gpu-v100` and `gpu-mi100` (UD IT processing ≥ 2 weeks; PI must be faculty), (d) ask who the PI/allocation manager of `bffp-dtai-gh` is.
  2. Email `help@ncsa.illinois.edu` (cc instructor): can the `bffp` project add **NCSA Delta GPU** SUs (partition `gpuMI100x8`: 8× MI100 gfx908 + 1× MI210 gfx90a, charge factor 0.25; and `gpuA100x4`) by ACCESS exchange from DeltaAI? Request ≈ 20–40 Delta GPU SU; ask that both students be added. Mention the Unix groups `delta_active_users`/`delta_bffp` already exist.
  3. Try `ssh login.delta.ncsa.illinois.edu` and run `accounts` there (free) — record whether login works and which accounts appear.
  4. Ask the advisor/instructor to request Chameleon Cloud PI status (CHI@TACC `gpu_mi100` bare-metal nodes, ROCm image; approval ≈ 1 business day) and add the students.
  5. Join the AMD AI Developer Program (MI300X credit via DigitalOcean, ≈ $100/30 days per one AMD page; credit card probably required). Do **not** redeem the credit until the AMD smoke pack (AMD-6) is ready.
  6. Confirm whether Hamza has his own DeltaAI username in `bffp-dtai-gh` (`sacctmgr show assoc account=bffp-dtai-gh format=user`). Accounts are never shared.
  7. The proposal says "I also hold an NSF ACCESS allocation": log in to allocations.access-ci.org and list every project where Talha (and Hamza) are PI/co-PI/user. If one holds or can exchange credits to **NCSA Delta GPU**, request ≈ 20–40 Delta GPU SU there (same justification as step 2; the DeltaAI→Delta exchange rate is unknown — read it from the exchange calculator and record it). If none, submit an **ACCESS Explore** request with the advisor as co-PI (letter required; "most Explore requests approved within 2 business days" per the how-to page, up to 2 weeks per the policy page) naming NCSA Delta GPU for `gpuMI100x8` and `gpuA100x4`; record the ticket number.
- **Deliverables:** `docs/STATUS.md` "Access status" table with route, date sent, contact, reply, next follow-up date.
- **Acceptance:** all six routes logged with dates; follow-up reminders on 10-27 and 11-03 (D2) noted; the Explore/exchange outcome feeds D2.

### ADM-2 Housekeeping of the work area
- **Session:** S2 · **Depends on:** — · **GPU budget:** 0
- **Steps:** delete `$CISC662_WORK/scratch/toolchain-feasibility/fa3_download` and `.../hf_home` (≈ 233 MB) and `$HF_HOME/xet` chunk cache; set `HF_XET_CHUNK_CACHE_SIZE_BYTES=0` in `env/env.sh`; delete `$CISC662_WORK/nanochat_base/base_checkpoints/{t_*,p_*,c_*}` (throwaway tags) if any exist outside `/tmp`; keep `r12_conv`; run `quota` and `du -sh $CISC662_WORK/*`; record sizes in STATUS.md. Use `du` only on `$CISC662_WORK` (never on `/work` or `$HOME`).
- **Acceptance:** project footprint < 15 GB; `r12_conv` checkpoint intact (`ls $CISC662_WORK/nanochat_base/base_checkpoints/r12_conv`).

### ADM-3 First commit on branch `cisc662` and `docs/DIFF_FROM_UPSTREAM.md` (needs the user's OK to commit)
- **Session:** S2 · **Depends on:** — · **GPU budget:** 0
- **Steps:** write `docs/DIFF_FROM_UPSTREAM.md` listing the three upstream patches (`nanochat/flash_attention.py`: env override `NANOCHAT_ATTN_IMPL` and an optional `softmax_scale=None` passthrough on `flash_attn_func` that defaults to upstream behaviour; `scripts/base_train.py`: profiler window `NANOCHAT_PROFILE_STEPS`/`NANOCHAT_EMIT_NVTX` + NVTX ranges; `nanochat/common.py`: GH200 peak entries) and the statement that the timed code path is otherwise identical to `92d63d4`. Add `cisc662/results/runs/*/gpu_monitor.csv` and `*_cuda_gpu_trace.csv` (large) to `.gitignore` under `cisc662/`. Ask the user before `git add -A cisc662 nanochat/common.py nanochat/flash_attention.py scripts/base_train.py && git commit`. Never commit `uv.lock` changes.
- **Acceptance:** `git status` clean except ignored files; `git log` shows the commit; `git diff 92d63d4 --stat -- nanochat scripts` lists exactly the three files (the integration shim of INT-1 lives in `cisc662/` and patches at run time, so this stays true all semester).

### ADM-4 Record decision D2 (AMD / V100 / A100 access) — 2026-11-03
- **Session:** S6 · **Depends on:** ADM-1 · **GPU budget:** 0
- **Steps:** collect replies; decide per route; if Delta access exists, schedule AMD-6 (env build in S10, runs in S12); if not, set the AMD half to STATIC in PLAN.md §1 and in the P1 deck, and keep accepting late access until 12-01. Same for V100/A100 → AMD-5 static fallback.
- **Deliverables:** STATUS.md "Decisions" entry D2 with outcome; PLAN.md §1 row updated.

### ADM-5 Budget and storage check (every session; 5 minutes)
- `accounts`, `/sw/user/scripts/jobcharge -u tmahmood1 -d 30 --detail | tail` (authoritative), `tail $CISC662_WORK/logs/job_ledger.csv` (cross-check; note hand-logged `srun` probes), `quota`. Write session SU, cumulative SU, pool balance into STATUS.md. Rules: **8 SU before P1 = re-plan checkpoint** (review the remaining S7–S9 caps with the user; GPU work may continue with a written reason); **10 SU before P1 = hard stop**; 25 SU overall planned cap; pool < 1,500 SU → only P2-critical jobs.

### ADM-6 End-of-project cleanup
- **Session:** S17 · archive profiles referenced by the report; delete other `.nsys-rep`/`.ncu-rep`, scratch, and all checkpoints except `r12_conv`; target < 10 GB on `/work/nvme`; final ledger summary in STATUS.md.

---

## INF — infrastructure (harnesses used by every later card)

### INF-1 `cisc662/bench/kbench.py` — kernel microbenchmark harness
- **Goal:** one timing harness that every kernel card uses, implementing PLAN.md §7.1.
- **Session:** S2 · **Depends on:** — · **GPU budget:** 0.3 (its share of the ≈ 25-min `slurm/s2_harness.sbatch`, shared with INF-2, INF-3 and BAS-3)
- **Steps:**
  1. API: `bench(fn, *args, flush_l2=True, warmup=25, reps=100, label=..., min_bytes=None, flops=None) -> dict` using `torch.cuda.Event(enable_timing=True)` pairs per rep; L2 flush = write a 256 MB `torch.empty(..., dtype=uint8).zero_()` buffer before each rep (outside the timed region); returns median/p10/p90/min of reps, achieved GB/s (= `min_bytes`/median) and TFLOP/s.
  2. `run_trials(spec, trials=3)` spawns `trials` separate processes (subprocess with the same argv + `--trial k`) and aggregates: median of trial medians; `cisc662/analysis/stats.py` computes bootstrap 95% CIs of the speedup against **each** named baseline (10k resamples stratified by trial).
  3. Preamble recorded in every result: git SHA + dirty flag, torch/triton/CUDA versions, device name, `nvidia-smi` clocks/power at start and end, `HAS_FA3`/`USE_FA3` (abort if an FA3 case is requested and `HAS_FA3` is False).
  4. Output: one JSON per case into `cisc662/results/db/<group>/<case>.json` with fields `{case, dtype, shape, median_us, p10_us, p90_us, gbps, tflops, trials, speedups: {"<baseline_case>": {"ratio", "ci_low", "ci_high"}}, gates_passed: [], env: {...}, job_id}` (one `speedups` entry per baseline, e.g. `eager_rmsnorm_768` and `compile_rmsnorm_768`); `results/db/README.md` documents the schema.
  5. CLI: `python -m cisc662.bench.kbench --suite smoke` runs eager `F.rms_norm` on (32768,768) bf16 and a 32768×768×768 bf16 GEMM as a self-check: GEMM ≥ 500 TF/s (record the value; 70–90% of the 723 TF/s 8192³ ceiling is expected at this N/K); RMSNorm cold-L2 achieved GB/s between 50% and 100% of `bandwidth.copy_GBps` (3650, measured on a 4 GiB buffer in `results/smoke_gh200.json`; a 50 MB kernel loses launch/ramp time).
- **Deliverables:** `bench/kbench.py`, `analysis/stats.py`, `results/db/README.md`, `results/db/smoke/*.json`, `slurm/s2_harness.sbatch` (runs INF-1 smoke, INF-2 tests, INF-3 suite, BAS-3 probes; ≈ 25 min).
- **Acceptance:** cold-L2 RMSNorm eager GB/s is **lower** than a warm-L2 run of the same case (prove the flush works); trial-to-trial CV < 2% on the GEMM case; JSON validates against the README schema.
- **Pitfalls:** do not time Triton's first call (JIT/autotune); `torch.cuda.synchronize()` only at trial end; keep inputs at realistic activation scale (randn·0.5 for activations, model-init scale for weights).

### INF-2 `cisc662/tests/` — correctness gates G1/G2
- **Goal:** pytest suite with fp64/fp32 references and the tolerance table, so every kernel card can run `pytest cisc662/tests -k <kernel>`.
- **Session:** S2 · **Depends on:** — · **GPU budget:** shares `s2_harness.sbatch`
- **Steps:**
  1. `tests/ref_ops.py`: pure-PyTorch references in a chosen dtype for `rmsnorm(x)` (**eps = `torch.finfo(torch.float32).eps` = 1.1920929e-07 for every input dtype** — that is what `F.rms_norm(eps=None)` does in torch 2.9.1 on the eager fused CUDA kernel, the composite and the Inductor decomposition; add a test that the reference reproduces `F.rms_norm` bit-exactly on a bf16 (64,768) input on CPU and on CUDA), `rope(x, cos, sin)` reproducing `nanochat.gpt.apply_rotary_emb` exactly (sign convention: `y1 = x1*cos + x2*sin; y2 = -x1*sin + x2*cos`, non-interleaved halves), `qkrope(q, k, cos, sin, scale=1.2)` in nanochat's order (RoPE → norm → ×1.2), `softcap_ce(logits, targets, cap=15, ignore_index=-1)` (mean over valid rows, plus a per-row `reduction="none"` form), `relu2`, `residual_mix(x, x0, r, l)`.
  2. `tests/tolerances.py`: the table from PLAN.md §7.3 as code (`check_forward(out, ref, stock_out, dtype)`, `check_backward(...)`), including the "≤ 2× stock error" rule. For attention, measure FA3's max-abs error against the fp32 math-backend SDPA reference once and store it as the stock error (FA3 vs bf16 SDPA agree to 1.95e-3; that is an agreement number, not an error).
  3. `tests/conftest.py`: fixtures for the real R12 shapes and edge shapes (N=1000 rows; T=1; rows with all `ignore_index`; |logits| > 50; positions up to 2047), fixed seeds; pytest markers `real_shape` (R12-sized tensors) and `gpu`, applied to every real-shape/GPU fixture.
  4. `tests/test_ref_ops.py`: the references agree with nanochat's own functions (`nanochat.gpt.norm`, `apply_rotary_emb`, the loss path) to fp32 rounding, and with stock eager bf16 ops within the stock-error tolerance. Gradcheck in fp64 on tiny shapes for each reference. Include a negative test that fails if the RoPE `sin` sign is flipped.
  5. `tests/gates.py`: `record_gate(result_json_path, "G1")` appends to `gates_passed`.
- **Deliverables:** the files above. GPU job: `pytest cisc662/tests -q`. Login node: `pytest cisc662/tests -q -m "not real_shape and not gpu"` (tiny shapes, < 2 GB, < 5 CPU-min).
- **Acceptance:** both commands green (the GPU one inside `s2_harness.sbatch`); the eps bit-exactness test passes on CUDA.

### INF-3 `cisc662/bench/ops_at_shapes.py`
- **Goal:** register every target op at its real shape as a kbench case in three forms: eager, `torch.compile(fn, dynamic=False)`, and (GEMMs only, S4) `torch.compile(mode="max-autotune-no-cudagraphs")`.
- **Session:** S2 · **Depends on:** INF-1 · **GPU budget:** shares `s2_harness.sbatch` (eager and compile rows in S2; the max-autotune GEMM rows only in BAS-4, S4). Do a `--trials 1` dry run first: 25 cases × 2 compiled forms × 3 process trials can exceed 30 min on an Inductor cache miss.
- **Steps:** cases = RMSNorm fwd and fwd+bwd at (32768,768) and (196608,128); QK-norm+RoPE+scale chain on q and k (16,2048,6,128) with cos/sin (1,2048,1,64); softcap+CE forward and forward+backward on (32768,32768) bf16 logits (reference = exactly nanochat's `.float() → 15*tanh(x/15) → F.cross_entropy`); lm_head GEMM + CE region (x (32768,768), W (32768,768)); GEMMs 32768×768×{768, 2304, 3072, 32768} and 32768×3072×768 plus the dW shapes (K=32768); FA3 vs SDPA at (16,2048,6,128) causal (fwd, fwd+bwd); relu² fwd+bwd on (32768,3072); residual-mix+norm chain. Each case declares `min_bytes` and `flops`.
- **Deliverables:** `bench/ops_at_shapes.py`, results in `results/db/ops_baseline/`.
- **Acceptance:** every case has eager and compile rows with CIs.

---

## BAS — baselines and measurement facts

### BAS-1 Inductor fusion facts (decision D1)
- **Goal:** know exactly what torch.compile already fuses in R12 and how many bytes each fused region moves, so vs-compile targets are set from facts.
- **Session:** S2 · **Depends on:** — · **GPU budget:** 0.5
- **Inputs:** `results/runs/s1_nsys_3400686/attrib_compile_fa3.json` (already shows e.g. `triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6` = RoPE+QK-norm fused forward, 1.33 ms/step ≈ 110 µs per call ≈ 1.8 TB/s; CE tail = 2 kernels, 4.45 ms/step: the forward reads bf16 logits once with an online softmax and stores per-row max/lse, the backward re-reads the logits, recomputes softcap + log-softmax and writes bf16 dlogits in place — ≈ 6.4 GB/step, forward at ≈ 0.95 TB/s; from the generated code in the Inductor cache).
- **Steps:**
  1. Job `slurm/s2_inductor_dump.sbatch` (1 GPU, 20 min, `#SBATCH --constraint=nvperf`): run R12 for 3 steps with `TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR=$OUT/compile_debug` (writes `output_code.py` and IR per graph; fall back to `TORCH_LOGS=output_code TORCH_LOGS_OUT=$OUT/output_code.log` if the debug dir is empty). Copy only the `output_code.py` files and the fx graph summaries into `results/runs/s2_inductor_<jobid>/`.
  2. In the same job, ncu on the compiled step **inside the profiler window**: `NANOCHAT_PROFILE_STEPS=20:21 ncu --profile-from-start off -k regex:'triton_|nvjet|flash' --launch-skip 0 --launch-count 900 --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active --clock-control none --csv python -m scripts.base_train <R12 args> --num-iterations=22`. ncu honours cudaProfilerStart/Stop only with `--profile-from-start off`; the S1 trace has ≈ 915 launches per compiled step (772 matching this regex), so the count must span one whole step — with five metrics that is a few replay passes per kernel and fits the 20-min job.
  3. Map each Inductor kernel in the trace to the source ops (names carry `original_aten` tokens; `output_code.py` has the fused op lists) for the regions: residual/λ-mix + RMSNorm, QK-norm+RoPE+scale (fwd and bwd), relu² (fwd and bwd), softcap+CE (fwd and bwd), AdamW/Muon steps.
  4. Write `results/inductor_fusion_table.csv` (region, kernel name, launches/step, µs/step, dram bytes/step, algorithmic minimum bytes, ratio, achieved GB/s, registers, achieved occupancy) and `docs/findings.md#inductor-baseline` with the per-region conclusion and the resulting vs-compile target (e.g. "CE tail: Inductor moves ≈ 6.4 GB in two reduction kernels at ≈ 0.95 / 2 TB/s; a fused kernel at ~3 TB/s → target 2–3× on time").
- **Acceptance:** every Tier-A region has measured bytes vs minimum bytes and achieved GB/s; the "target vs compile" lines of KT-1..KT-4 are updated from this table (edit the cards).
- **Pitfalls:** ncu replays kernels — keep the metric list short; without `--profile-from-start off` the skip/count run from process start (step-0 compile/autotune launches), not from step 20.

### BAS-2 Loss-curve noise envelope + `analysis/losscurve.py`
- **Session:** S2 · **Depends on:** — · **GPU budget:** 0.3
- **Steps:** job `slurm/s2_noise_band.sbatch` (1 GPU, 15 min): two stock R12-L runs (1000 steps, `--eval-every=250 --eval-tokens=2097152`, same seed, `--model-tag=noise_a` / `noise_b`, `use_local_base_dir`). `analysis/losscurve.py`: parse per-step loss (reuse `summarize_s1.parse_train_log`); **the logged `loss:` field is already nanochat's debiased EMA(0.9) (base_train.py `smooth_train_loss`), so use it as-is and do not smooth again**; envelope = max over steps > 100 of |loss_a − loss_b|; val-bpb delta; `compare(stock, variant)` implements the G4 pass rule; overlay + delta plot with the band shaded → PNG.
- **Deliverables:** `results/runs/s2_noise_<jobid>/`, `results/db/loss/noise_envelope.json` (`{band, val_bpb_delta, steps}`), `results/figures/loss_noise_band.png`.
- **Acceptance:** band and val-bpb delta recorded; `python -m cisc662.analysis.losscurve stock.log variant.log --envelope noise_envelope.json` prints PASS/FAIL.

### BAS-3 Clock, power and latency probe (inside `s2_harness.sbatch`)
- **Session:** S2 · **Depends on:** INF-1 · **GPU budget:** 0.2 (shares `s2_harness.sbatch`)
- **Steps:** (1) a 60-s bf16 GEMM loop while sampling `nvidia-smi -q -d CLOCK,POWER,PERFORMANCE` every 2 s; record sustained SM clock, power, and `clocks_throttle_reasons.active`. (2) A job **step** inside the same job: `srun --gpu-freq=1755 --gpus=1 bash -c 'nvidia-smi --query-gpu=clocks.sm,clocks_throttle_reasons.active --format=csv -lms 500 > $OUT/gpufreq_probe.csv & python -m cisc662.bench.kbench --suite smoke; kill %1'`; `gpu_freq_pinnable = true` iff the sampled SM clock sits at 1755 ± 15 MHz for the whole step (keep the step's stderr). (3) Pointer-chase DRAM latency microbenchmark (a 1 GB random cycle, 1 thread) → ns per access, for PLAN.md §7.2. (4) Decode re-measurement: `decode_bench` CPU (1 thread, in the same job on the host) and GPU at b=1, 256 decode tokens, 3 reps each, logging node and clocks — the S1 GPU b=1 value varied 68–107 tok/s between jobs.
- **Deliverables:** `results/db/platform/clocks_<jobid>.json` (sustained clock, power, throttle reasons, latency_ns, gpu_freq_pinnable), `results/db/decode/s2_remeasure.json`; STATUS.md "Clock policy" line (default clocks + logged, or pinned 1755 MHz for microbenchmarks).
- **Acceptance:** policy decided and written; latency value in the 500–800 ns range (sanity); decode ratio GPU/CPU-1-thread reported with its spread.

### BAS-4 Per-op baselines (eager vs compile vs max-autotune) at real shapes
- **Session:** S4 · **Depends on:** INF-1, INF-3 · **GPU budget:** 0.5
- **Steps:** run `ops_at_shapes.py` full suite incl. the max-autotune GEMM rows in `slurm/s4_ops_baseline.sbatch` (1 GPU, 30 min; max-autotune compile is slow — cache on /work/nvme via `TORCHINDUCTOR_CACHE_DIR`). Produce `results/ops_baseline_table.md` (op, shape, eager µs, compile µs, max-autotune µs, GB/s or TFLOP/s, % of 3.65 TB/s or 723 TF/s).
- **Acceptance:** table complete for all cases in INF-3; cuBLAS GEMM at 32768×768×768 reported as % of 723 TF/s; the SDPA backend used on GH200 recorded (cuDNN vs flash vs efficient, from the nsys kernel name).

### BAS-5 CPU per-op references (goal i)
- **Session:** S4 · **Depends on:** INF-2 · **GPU budget:** 0 GPU; ≈ 0.35 SU CPU-only (serial 30 min at 0.144 SU/h ≈ 0.07; 72 cores + 96g = 1.0 SU/h × 15 min = 0.25), counted against the S4 cap
- **Steps:** `bench/cpu_ref.py`: each target op (RMSNorm, RoPE chain, softcap+CE, GEMM 2048×768×768 and 2048×768×3072, SDPA attention at (1,2048,6,128), relu²) in fp32 at reduced N=2048 tokens, 10 reps, median; scale to the R12 shape linearly by bytes (memory-bound) or FLOPs (GEMM/attention) and label SCALED. `bench/cpu_peak.c` (or torch): single-core FMA peak (GFLOP/s) and STREAM-like copy bandwidth, 1 thread and 72 threads. Jobs: `slurm/s4_cpu_ref_serial.sbatch` (`--cpus-per-task=2 --mem=16g`, 30 min, `numactl --physcpubind=0 --membind=0`, `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`) and `slurm/s4_cpu_ref_72.sbatch` (`--cpus-per-task=72 --mem=96g`, 15 min).
- **Deliverables:** `results/db/cpu/*.json`, `results/cpu_reference_table.md` with the S1 full-step numbers folded in — read them from `results/runs/s1_cpuref_3400725/summary.md` rather than from this card: 1 core 39.86 s/step = 3,986 s/100 steps; 72 cores 2.27 s/step; GPU b=1 compile 3.019 s/100 steps = 0.0302 s/step, eager 5.574 s/100 = 0.0557 s/step; decode b=1: CPU 1 thread 32.8 tok/s, 72 threads 89.2, GPU 68–107 across S1 jobs (BAS-3 re-measures it).
- **Acceptance:** table has raw GPU/1-core, GPU/72-core, and peak-normalized and bandwidth-normalized speedups per op; the full-step ratios match PLAN.md §4 (≈ 1,320× and ≈ 75×).

### BAS-6 ncu baseline of the compiled step's top kernels (feeds roofline v1)
- **Session:** S4 · **Depends on:** BAS-1 · **GPU budget:** 0.4
- **Steps:** `slurm/s4_ncu_baseline.sbatch` (nvperf, 25 min): for each of the 15 kernels with the largest time/step in `attrib_compile_fa3.json`, `NANOCHAT_PROFILE_STEPS=20:22 ncu --profile-from-start off -k regex:<name> --launch-skip 1 --launch-count 1 --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section WarpStateStats --clock-control none --csv …` (the second in-window launch = warm steady state). Parse into `results/db/ncu/baseline_compile_<jobid>.json` (per kernel: duration, dram bytes, achieved occupancy, registers, top-2 stall reasons, sectors/request).
- **Acceptance:** 15 kernels covered; `analysis/roofline.py` (ANA-1) can place them with measured bytes.

---

## KT — Triton kernels

### KT-1 Triton RMSNorm (768- and 128-wide) + fused residual/λ-mix + norm (`mixnorm`)
- **Goal:** L2 of the RMSNorm ladder; first full pass through gates → microbench → ISA.
- **Session:** S3 · **Depends on:** INF-1, INF-2 · **GPU budget:** 0.4 (in `slurm/s3_kernels.sbatch`)
- **Inputs:** hot-path inventory ops "rms_norm pre-attention and pre-MLP", "resid/x0 lambda mixing", "QK-norm"; prior art: eager is already `aten::_fused_rms_norm` (4 warps per row, 8-byte loads) — expect ≈ 1.0–1.3× vs eager; Liger RMSNorm loses to torch at H=1024, so benchmark at the real shapes only; Inductor fuses the norm with its neighbours (BAS-1 table).
- **Steps:**
  1. `kernels/triton/rmsnorm.py`: forward `y = x * rsqrt(mean(x²) + eps)` — parameter-free; **eps = `torch.finfo(torch.float32).eps` = 1.1920929e-07 for bf16/fp16/fp32 inputs** (torch 2.9.1 upcasts to fp32 and uses the fp32 epsilon on the eager fused CUDA kernel, the composite and the Inductor decomposition; verified bit-exact on CPU in the plan review, confirm on GPU in the S2 harness job); fp32 accumulation; store `rstd` fp32 (N×4 B). v1: one row per program, BLOCK=1024 masked. v2: `BLOCK_M` rows per program, 768 handled as 3×256 to avoid 25% lane waste; 128-wide variant with 8 rows per program. Divisibility hints on the row stride. Backward: `dx = rstd·(dy − x·rstd²·mean(dy·x))` (no dW), one kernel, reads x and dy, writes dx. Function names `c662_rmsnorm_fwd_v1`, `c662_rmsnorm_bwd`, …
  2. v3 `c662_rmsnorm_fwd_mixnorm`: `x_new = r·x + l·x0` written out **and** `norm(x_new)` in one pass (reads x, x0; writes x_new, h); backward also emits per-row partial sums for dr, dl (finished with a tiny reduction). Shapes (16,2048,768); scalars r, l are 0-d fp32 tensors.
  3. Wrap each as `torch.autograd.Function`; register `rmsnorm` as `torch.library.triton_op("cisc662::rmsnorm", mutates_args=())` with `register_autograd`, so Inductor can see through it (G3 later).
  4. Gates: `tests/test_rmsnorm.py` (G1/G2 at real + edge shapes, both widths, v1/v2/v3).
  5. Microbench with kbench: eager `F.rms_norm`, compile of the same function, ours (v1, v2; v3 against the eager 3-kernel mix+norm chain and its compile).
  6. ISA: save `kernel.asm["ptx"]`, `["sass"]`, `n_regs`, `n_spills` for each variant to `results/isa/rmsnorm/triton_<variant>/sm_90a/`; note load width (`LDG.E.128` count), `SHFL` count, `MUFU.RSQ`.
- **Deliverables:** kernel file, tests, `results/db/rmsnorm/*.json` with `gates_passed:["G1","G2"]`, ISA dumps, ISA card `docs/findings.md#rmsnorm`, `run_step` lines in `slurm/s3_kernels.sbatch`.
- **Acceptance:** G1/G2 pass; result JSONs have CIs vs eager and vs compile; achieved ≥ 80% of 3.65 TB/s for the 768-wide forward (≈ 34 µs or better per (32768,768) call) — or, if not reached, the ncu-backed explanation (DRAM %, achieved occupancy, top-2 stalls, sectors/request) in the ISA card; the card is then complete and the number is reported as measured.
- **Pitfalls:** nanochat applies `norm` to bf16 inputs with `eps=None`; in torch 2.9.1 that is the **fp32** epsilon 1.19e-7, not `torch.finfo(bf16).eps` = 7.8e-3 and not 1e-6 — using 7.8e-3 shifts rstd by 0.4% at mean(x²)≈1 and far more at small activations, so every variant would fail the ≤ 2× stock-error rule against the real stock op. QK-norm rows are 128 wide over the last dim of (B,T,H,128) — contiguous, stride 128.

### KT-2 Fused QK-norm + RoPE + 1.2 scale (`qkrope`, forward and backward)
- **Session:** S3 · **Depends on:** INF-1, INF-2 · **GPU budget:** 0.4 (in `slurm/s3_kernels.sbatch`)
- **Inputs:** inventory op "RoPE apply_rotary_emb" (eager 8 kernels fwd, 11 bwd, 32A traffic per tensor per layer); Inductor already fuses the forward chain into one kernel but only at ≈ 1.8 TB/s (110 µs per call for ≈ 201 MB; 40 regs, 256 threads) — so the forward target vs compile is ≈ 1.3–1.6× if ≥ 2.7 TB/s is reached (pending BAS-1's ncu bytes); identity: RoPE is a rotation ⇒ `norm(rope(x)) == rope(norm(x))` up to the bf16 cos/sin table, so rstd can be computed from the un-rotated row; `apply_rotary_emb` forward map: `y1 = x1·cos + x2·sin; y2 = −x1·sin + x2·cos` on halves of 64; cos/sin are bf16 (1,T,1,64) slices of a 20480-long cache.
- **Steps:**
  1. `kernels/triton/qkrope.py` (`c662_qkrope_fwd`, `c662_qkrope_bwd`): one program per block of (token, head) rows; loads q and k rows (both tensors in the **same** launch), computes `rstd` from the raw 128 values in fp32, applies the rotation with fp32 cos/sin (upcast from the bf16 table nanochat uses — document the rounding), multiplies by `1.2·rstd`, stores bf16. Option `fold_scale=True` skips the 1.2 and returns `softmax_scale = 1.44/sqrt(128)` for attention (training only; decode stores scaled k in the KV cache, so do not fold there). The patched `flash_attn_func` wrapper accepts `softmax_scale=` (documented patch; FA3 takes it natively, SDPA gets `scale=`).
  2. Backward, written explicitly: `g = 1.2·dy` (or the folded factor); `g′ = Rᵀg` with `g′₁ = g₁·cos − g₂·sin`, `g′₂ = g₁·sin + g₂·cos` (transpose of the forward map); `dx = rstd·(g′ − x·rstd²·mean(g′·x))` using the raw x and the saved rstd (valid because R is orthogonal up to the bf16 table). One kernel, no zero-filled slice tensors.
  3. `torch.library.triton_op("cisc662::qkrope")` + autograd; G1/G2 tests including positions up to 2047 and T=1 with an offset (decode path uses `cos[:, T0:T0+T]`).
  4. Microbench vs the eager chain (RoPE+norm+scale on q and k) and vs compile of the same chain; also fwd+bwd.
  5. ISA dumps to `results/isa/qkrope/triton_<variant>/sm_90a/`; note registers and whether loads are 128-bit.
- **Deliverables:** `kernels/triton/qkrope.py`, `tests/test_qkrope.py`, `results/db/qkrope/*.json`, ISA dumps, ISA card `docs/findings.md#qkrope`, `run_step` lines in `slurm/s3_kernels.sbatch`.
- **Acceptance:** G1/G2 pass (bf16 tolerance vs fp64 reference; also ≤ 2× stock error); forward traffic measured by ncu ≈ 2 reads + 2 writes of (16,2048,6,128) bf16 (≈ 200 MB) per call; speedup vs eager ≥ 3× with CI; vs compile reported honestly against the 1.3–1.6× forward target — or, if not reached, the ncu-backed explanation in the ISA card.
- **Pitfalls:** nanochat computes RoPE **before** QK-norm; the fused order is mathematically equal but bf16 rounding differs — tolerance uses the "≤ 2× stock error" rule; the decode path (`kv_cache` branch) must keep the unfused scale convention unless the KV cache is also changed.

### KT-3 Fused softcap + cross-entropy (`ce`), gradient written in the forward
- **Goal:** the biggest vs-eager win (eager chain ≈ 94 GB/step) and a strong vs-compile candidate on **time**, not bytes.
- **Session:** S4 · **Depends on:** INF-1, INF-2, BAS-1 · **GPU budget:** 0.6 (in `slurm/s4_ce.sbatch`)
- **Inputs:** inventory ops "logits slice + .float()", "softcap", "cross_entropy fwd/bwd", "softcap backward"; S1: compile tail = 2 kernels, 4.45 ms/step, ≈ 6.4 GB (bf16 logits saved; softcap + log-softmax recomputed in the backward from per-row max/lse; dlogits written in place), forward at ≈ 0.95 TB/s, backward ≈ 2 TB/s; prior art: Liger CE (gradient in forward, softcap supported), modded-nanogpt PR #251, QuACK CE at V=32K ≈ 3.0 TB/s on H100.
- **Steps:**
  1. `kernels/triton/ce.py` (`c662_ce_fwd`, `c662_ce_bwd`): per row (V=32768 bf16 logits): pass 1 online max/sum-exp over chunks of 4096 with `z = 15·tanh(l/15)` computed in fp32 registers; `lse`; `loss_row = lse − z[target]`; pass 2 (row still in L2) writes `dlogits = (exp(z − lse) − onehot) · (1 − (z/15)²) · (1/n_valid)` in bf16 into a separate buffer. Rows with `target == −1`: loss 0, dlogits 0, excluded from `n_valid`; mean over valid rows (nanochat uses `reduction='mean'`, `ignore_index=-1`); also provide `reduction='none'` (per-row loss, no `n_valid` scaling) because `evaluate_bpb` uses it.
  2. `tanh`: precise (`tl.math.tanh` or an `exp2`-based formula) is the gated variant. `tanh.approx.f32` (`MUFU.TANH`, max abs error ≈ 4.9e-4 → ≈ 7e-3 on z, ≈ 1e-3 relative per-row loss) is **expected to fail the 1e-5 loss gate by ~100×**; evaluate it only as a measured time/instruction-count ablation (plus G4 under an explicitly relaxed gate) and record it in the ISA card.
  3. `grad_out` semantics: `base_train.py` divides the loss by `grad_accum_steps` **before** `backward()`, so the tangent reaching the loss is a 0-d tensor equal to 1/grad_accum_steps (1.0 for R12) and cannot be tested for 1.0 under torch.compile (fake tensors). Implement and measure both forms: (i) gradient-in-forward + a `grad_out` scale pass in the backward (≈ 4.3 GB eager; ≈ 8.6 GB compiled because the scale pass is a separate pointwise kernel); (ii) forward = loss + per-row stats, backward recomputes dlogits from the logits in place (≈ 6.4 GB, the same as Inductor; the win must come from achieved bandwidth and no duplicated transcendental work). KT-4 is the compile-integrated form that avoids the issue (dx/dW are scaled instead, 2 × 50 MB).
  4. `torch.library.triton_op("cisc662::ce")`; G1 (loss rel 1e-5 vs fp64 on the same bf16 logits; dlogits rel-L2 ≤ 1e-2 and ≤ 2× stock) and G2; edge cases: all-ignored rows, |logits| > 50.
  5. kbench: eager tail (`.float → softcap → F.cross_entropy` fwd+bwd), compile tail, ours (both forms); report time, bytes (ncu), and `torch.cuda.max_memory_allocated` deltas (note: Inductor already writes dlogits in place, so in-place is not a memory win vs compile — KT-4's chunking is).
- **Deliverables:** `kernels/triton/ce.py`, `tests/test_ce.py`, `results/db/ce/*.json`, ISA dumps `results/isa/ce/triton_<variant>/sm_90a/`, ISA card `docs/findings.md#ce`, `run_step` lines in `slurm/s4_ce.sbatch`.
- **Acceptance:** G1/G2 pass; measured traffic (ncu) ≈ 4.3 GB per fwd+bwd in eager form (i) (vs ≈ 6.4 GB compile, 94 GB eager) and the compiled-form bytes reported for both (i) and (ii); speedup vs eager ≥ 5× and vs the compile tail ≥ 1.3× **on time** with CI lower bound ≥ 1.3 — if not reached, record the ncu diagnosis (achieved GB/s, spills, row chunking) and continue (decision D3).

### KT-4 Chunked fused lm_head + softcap + CE (`ce_chunked`; never materializes the full logits)
- **Session:** S5 · **Depends on:** KT-3 · **GPU budget:** 0.6 (in `slurm/s5_kernels.sbatch`)
- **Steps:**
  1. `kernels/triton/ce_chunked.py`: `autograd.Function(x_normed (N,768) bf16, W_bf16 (V,768), targets, reduction)`; loop over token chunks of 2048–4096 rows: `logits_c = x_c @ Wᵀ` (cuBLAS bf16, into a reused chunk buffer), KT-3 kernel → `loss_c`, `dlogits_c`; immediately `dx_c = dlogits_c @ W` and `dW += dlogits_cᵀ @ x_c` (fp32 accumulate); backward returns the stored `dx`, `dW` (cast to the weight dtype) scaled by `grad_out`. Variant B (stretch, half a session max): Triton tile-fused version with online LSE across vocab tiles (CCE-style).
  2. Integration hook (used by INT-1): **no edit to `nanochat/gpt.py`.** `cisc662/integrate.py` installs a patched copy of `GPT.forward` on the class (same technique as the `CausalSelfAttention.forward` copy in INT-1) when `CISC662_KERNELS` contains `ce` (KT-3 tail applied to the materialized logits) or `ce_chunked` (this kernel on `x` after the final norm and `self.lm_head.weight`). The copy dispatches exactly like upstream: `targets is None` → stock logits path (decode); `loss_reduction="none"` → stock path or the per-row loss from the same kernel without the mean/`n_valid` scaling, because `evaluate_bpb` (`nanochat/loss_eval.py`) calls `model(x, y, loss_reduction="none")` on the compiled model in every R12-L run; `loss_reduction="mean"` → fused kernel.
  3. Gates: loss, dX, dW vs fp32 reference at full R12 shape (reference needs ≈ 20 GB; use `--mem=96g`).
  4. kbench: eager region (lm_head GEMM + tail, fwd+bwd), compile region, KT-3 (GEMM + fused tail), KT-4; peak memory for each; also report the largest `device-batch-size` that fits with each variant at R12 (document as a different config).
- **Deliverables:** `kernels/triton/ce_chunked.py`, `tests/test_ce_chunked.py`, `results/db/ce_chunked/*.json`, peak-memory table `results/ce_memory_table.md`, ISA card `docs/findings.md#ce_chunked`, `run_step` lines in `slurm/s5_kernels.sbatch`.
- **Acceptance:** G1/G2 pass; region time vs compile with CI; peak-memory saving ≥ 2 GB at R12 (the 2.15 GB logits + saved tensors); the fastest chunk size recorded.
- **Pitfalls:** `dW` accumulation across chunks in fp32 (2× 100 MB extra); Muon still needs `lm_head.weight.grad` with the right shape/dtype (AdamW group for lm_head — check `setup_optimizer`); `x.view(-1, 768)` must be contiguous.

### KT-5 Triton GEMM study at nanochat shapes (`gemm`, Tier B, expected negative result)
- **Session:** S5 · **Depends on:** INF-1, BAS-4 · **GPU budget:** 0.5 (in `slurm/s5_kernels.sbatch`)
- **Steps:** `kernels/triton/gemm.py`: (a) tutorial tiled matmul with grouped L2 ordering, autotune set BM/BN ∈ {64,128,256}, BK ∈ {32,64}, stages 3–4, warps 4/8; (b) persistent kernel with TMA descriptors (Triton 3.5 `tl.make_tensor_descriptor`) — the small-K opening (ThunderKittens beat cuBLAS 1.1–1.57× at K ≤ 512; nanochat's forward K is 768). Shapes: 32768×768×{768,2304,3072,32768}, 32768×3072×768, dW shapes with K=32768 (split-K needed). Compare against cuBLAS (`torch.mm`) and Inductor max-autotune (BAS-4). ISA: HGMMA shape, `LDGSTS`/`UTMALDG`, registers (expect ≈ 120, 0 spills for 128×128×64), shared memory; ncu tensor-pipe utilization and the wave-quantization explanation (tiles per 132 SMs).
- **Deliverables:** `kernels/triton/gemm.py`, `results/db/gemm/*.json`, `results/gemm_table.md` (% of cuBLAS per shape), ISA dumps `results/isa/gemm/triton_<variant>/sm_90a/`, ISA card `docs/findings.md#gemm`, `run_step` lines in `slurm/s5_kernels.sbatch`.
- **Acceptance:** all shapes measured; % of cuBLAS per shape with the counters (tensor-pipe %, stall reasons) that explain the worst shape. No minimum speedup is required — this card documents where and why Triton loses to cuBLAS.

### KT-6 Triton FA2-style attention forward vs FA3 vs SDPA (`attention`, Tier C)
- **Session:** S7 · **Depends on:** INF-1 · **GPU budget:** 0.4 (in `slurm/s7_kernels.sbatch`)
- **Steps:** adapt the Triton fused-attention tutorial (FA2 algorithm, BLOCK_M=128, BLOCK_N=64/128, causal block skipping) to (16,2048,6,128) bf16 causal, in two variants: with TMA descriptors (sm_90) and plain (also used for the AMD cross-compile); G1 vs the fp32 math-backend SDPA reference (max-abs ≤ 4e-3 and ≤ 2× FA3's own error against that reference, from INF-2); kbench vs FA3 (`nanochat.flash_attention._fa3.flash_attn_func`) and SDPA; ncu: tensor-pipe %, registers/spills, stall reasons; backward only as a stretch.
- **Deliverables:** `kernels/triton/attention.py`, `tests/test_attention.py`, `results/db/attention/*.json`, ISA dumps `results/isa/attention/triton_<variant>/sm_90a/`, ISA card `docs/findings.md#attention`, `run_step` lines in `slurm/s7_kernels.sbatch`.
- **Acceptance:** G1 pass; measured % of FA3 (expect 0.5–0.8×) with the instruction-level explanation drafted (feeds AMD-3's FA3 dissection: HGMMA shapes, TMA, warp specialization/`USETMAXREG`).

### KT-7 relu² in the c_fc GEMM epilogue (`mlp`) + derivative in the c_proj dX epilogue
- **Session:** S10 · **Depends on:** KT-5, BAS-1 · **GPU budget:** 0.5 (in `slurm/s10_kernels.sbatch`)
- **Steps:** Triton GEMM from KT-5 with epilogue `h = relu(acc)²` storing bf16 `h` and (for backward) only `relu(acc)`; dX GEMM epilogue applying `2·relu(a)·g`. Compare the MLP block (c_fc + relu² + c_proj fwd+bwd) against cuBLAS+eager and cuBLAS+Inductor (Inductor already fuses the pointwise relu² but cannot fuse into cuBLAS). Gates G1/G2; ISA card.
- **Deliverables:** `kernels/triton/mlp.py`, `tests/test_mlp.py`, `results/db/mlp/*.json`, ISA dumps `results/isa/mlp/triton/sm_90a/`, ISA card `docs/findings.md#mlp`, `run_step` lines in `slurm/s10_kernels.sbatch`.
- **Acceptance:** G1/G2 pass; MLP-block time vs compile with CI (expected gain ≈ 2–3 ms/step if the Triton GEMM is within 5% of cuBLAS; otherwise a documented negative result). Enters the end-to-end tables (INT-6) only after its own G3/G4 in INT-5.

### KT-8 (optional) Muon polar-express (`muon`): baddbmm epilogues + symmetric X·Xᵀ
- **Session:** S13 (only within the S13 cap and if the optimizer is ≥ 5% of the compiled step — S1: 7.6 ms of 74 ms ≈ 10%) · **Depends on:** INF-1 · **GPU budget:** 0.5 (in `slurm/s13_kernels.sbatch`)
- **Steps:** (1) replace `B = b·A + c·A@A; X = a·X + B@X` pointwise ops with `torch.baddbmm(beta, alpha)` epilogues inside a copy of `muon_step_fused` (`cisc662/optim_shim.py`, not upstream); (2) Triton kernel computing only the upper triangle of `X·Xᵀ` for (48,768,768) and (12,768,3072)·(12,3072,768) and mirroring; compare NS-iteration time vs stock compiled `muon_step_fused`; verify updates match stock to bf16 tolerance and run G4.
- **Deliverables:** `kernels/triton/muon.py`, `cisc662/optim_shim.py`, `results/db/muon/*.json`, ISA card `docs/findings.md#muon`.
- **Acceptance:** optimizer-step time before/after with CI; identical loss curve within the envelope (G4).

---

## KC — hand-written CUDA kernels (L3) with SASS analysis

### KC-1 `kernels/cuda/rmsnorm.cu`
- **Session:** S7 · **Depends on:** KT-1, AMD-2 (provides the v0 source) · **GPU budget:** 0.4 (in `slurm/s7_kernels.sbatch`)
- **Steps:**
  0. Start from AMD-2's `kernels/cuda/rmsnorm.cu` v0; after any change, re-run AMD-2's hipify + compile step and overwrite `results/isa/rmsnorm/hip/*` (a KC-1 deliverable), so the static AMD rows always describe the kernel that was timed.
  1. Build with `torch.utils.cpp_extension.load(name="cisc662_rmsnorm", sources=[...], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-lineinfo", "--ptxas-options=-v"])` (no `--use_fast_math`; a fast-math build only as a separately named ablation row), `CC=gcc CXX=g++`, `MAX_JOBS=8`. Design: **one warp per 768-wide row**: each lane issues 3× `LDG.E.128` (uint4 = 8 bf16) before any use (ILP 3), fp32 sum of squares, 5× `__shfl_xor_sync`, `rsqrtf` (→ `MUFU.RSQ`), multiply, pack (`F2FP.BF16.F32.PACK_AB`), 3× `STG.E.128`; 8 rows per 256-thread block; no shared memory; eps = fp32 epsilon (KT-1). Target **0 spills and ≥ 50% theoretical occupancy (≤ 64 regs/thread)**; a `__launch_bounds__`/`maxrregcount=32` build is one row of the ablation table (expected to spill). 128-wide variant: 16 lanes per row, one `LDG.E.128` each, 4-step shuffle. Backward with the same structure; compare recomputing rstd vs loading it. Ablations (each an ncu row): scalar 16-bit loads vs 128-bit; rows per block 1/2/4/8; fp32 vs bf16 accumulation (expected to fail G1 → documented).
  2. Python wrapper `torch.library.custom_op("cisc662::rmsnorm_cuda")` + `register_fake` + `register_autograd`, registered in `cisc662/integrate.py` under `rmsnorm:cuda`.
- **Deliverables:** `.cu`, wrapper, tests, `results/db/rmsnorm/cuda_*.json`, `results/isa/rmsnorm/cuda_<variant>/sm_90a/` (SASS + ptxas `-v` log), refreshed `results/isa/rmsnorm/hip/`, ISA card section comparing eager (`vectorized_layer_norm_kernel`), Inductor, Triton, CUDA: registers, load width, SHFL count, occupancy, sectors/request, DRAM %, top stalls.
- **Acceptance:** G1/G2 pass; ≥ 85% of 3.65 TB/s on the 768-wide forward — or the ncu-backed explanation in the ISA card; SASS shows `LDG.E.128`×3 before the first `FMUL` and no local memory; ablation table with counters.

### KC-2 `kernels/cuda/qkrope.cu` — two layouts
- **Session:** S7 · **Depends on:** KT-2 · **GPU budget:** 0.4 (in `slurm/s7_kernels.sbatch`)
- **Steps:** layout (a): 16 lanes per 128-wide head row, each lane `LDG.E.128` of 8 contiguous bf16; rotate-half partner is lane `i^8` → 4× `__shfl_xor_sync(…, 8)` on packed registers. Layout (b): each lane loads 4 elements from the first half and the matching 4 from the second half with two `LDG.E.64`; no exchange. Both compute rstd via 4-step shuffle, apply rotation+norm+1.2 (or fold), q and k in one launch. ncu decides: instructions, sectors/request (ideal 16 per request at 128-bit), stalls, time. Backward (KT-2 formula) for the winning layout.
- **Deliverables:** `kernels/cuda/qkrope.cu` + wrapper (`cisc662::qkrope_cuda`, registered under `qkrope:cuda`), tests, `results/db/qkrope/cuda_*.json`, `results/isa/qkrope/cuda_{a,b}/sm_90a/` (SASS + ptxas logs), ISA card `docs/findings.md#qkrope`.
- **Acceptance:** G1/G2 pass; ncu table (a) vs (b) vs Triton vs Inductor; ISA card with the explanation.

### KC-3 `kernels/cuda/ce.cu` — register-resident vs shared-memory-staged
- **Session:** S10 · **Depends on:** KT-3 · **GPU budget:** 0.5 (in `slurm/s10_kernels.sbatch`)
- **Steps:** 512 threads per row; register variant: each thread 8× `LDG.E.128` (64 bf16 in 32 packed registers), block max then sum via warp `SHFL` + 16-entry shared exchange (2 `BAR.SYNC`), recompute `z` from registers, write `dlogits` once (≈ 64 registers → 50% occupancy; Little's law: 8 independent loads/thread). Shared variant: stage the 64 KB row in shared memory. Precise `tanhf` is the gated variant; `tanh.approx.f32` (`MUFU.TANH`) is a measured time/instruction-count ablation expected to fail the 1e-5 loss gate (see KT-3). Online-LSE streaming variant if registers spill.
- **Deliverables:** `kernels/cuda/ce.cu` + wrapper (`cisc662::ce_cuda`, registered under `ce:cuda`), tests, `results/db/ce/cuda_*.json`, `results/isa/ce/cuda_{reg,smem}/sm_90a/`, ISA card `docs/findings.md#ce`.
- **Acceptance:** G1/G2 pass for the gated variant; table register vs shared vs Triton vs Inductor with counters; the tanh ablation row with its measured accuracy.

### KC-4 CUDA GEMM ladder (Tier B teaching ladder)
- **Session:** S13 · **Depends on:** KT-5 · **GPU budget:** 0.6 (in `slurm/s13_kernels.sbatch`)
- **Steps:** v1 shared-memory tiled fp32 FFMA (ceiling 48.6 TF/s); v2 bf16 `mma.sync.m16n8k16` + `ldmatrix.x4` + 3-stage `cp.async` (SASS: `HMMA.16816.F32.BF16`, `LDSM`, `LDGSTS`); v3 wgmma + TMA only if v2 is done with ≥ 2 h left (hard timebox). Shapes 32768×768×768 and 32768×768×3072. Report % of cuBLAS, registers, shared memory, tensor-pipe %, stalls; explain the gap (legacy HMMA issue vs warpgroup HGMMA with operands in shared memory; no TMA; wave quantization). Also `hipify-perl` the v1 FFMA kernel and compile it with the AMD-2 recipe for gfx908/gfx90a/gfx942 (static VGPR/LDS/occupancy row; the `mma.sync`/`ldmatrix` v2 does not hipify mechanically — say so).
- **Deliverables:** `kernels/cuda/gemm_v{1,2}.cu` (+ `gemm_v3.cu` if attempted), `kernels/hip/gemm_v1.hip`, tests, `results/db/gemm/cuda_*.json`, `results/isa/gemm/cuda_v*/sm_90a/`, `results/isa/gemm/hip/gfx*/`, ISA card `docs/findings.md#gemm`.
- **Acceptance:** v2 correct (G1 GEMM tolerance) and measured; ISA card "GEMM ladder" with HMMA vs HGMMA evidence from SASS; the HIP v1 row in `static_isa.csv`.

---

## AMD — HIP, AMDGCN and cross-ISA static analysis

### AMD-1 Static cross-ISA pipeline: `analysis/xcompile.py` + `analysis/isa_report.py`
- **Goal:** one command that compiles every Triton kernel for `cuda:90`, `hip:gfx908`, `hip:gfx90a`, `hip:gfx942` on the login node (no GPU) and tabulates registers/occupancy/instruction mix — the guaranteed AMD deliverable.
- **Session:** S3 · **Depends on:** KT-1 (first kernels) · **GPU budget:** 0
- **Inputs:** `$CISC662_WORK/scratch/toolchain-feasibility/scripts/triton_xcompile.py`, `$CISC662_WORK/scratch/amd-rocm-planning/scripts/{xcompile_sweep.py,run_mic.py}`, toolchain JSON (ASTSource API for Triton 3.5.1: `signature` dict with `"constexpr"` entries, `constexprs` by name, `attrs={(i,): [["tt.divisibility",16]]}`; `GPUTarget("hip","gfx90a",64)`), AMD JSON (occupancy tables: gfx908 granule 4 / max 10 waves, gfx90a+ unified 512 regs granule 8 / max 8; LDS 64 KiB per CU; Triton's dynamic LDS is in `metadata["shared"]`, not in the `.amdgcn` occupancy line).
- **Steps:**
  1. Kernel registry `analysis/kernel_registry.py`: for each Triton kernel, its signature, constexprs, divisibility attrs and launch options (num_warps, num_stages, waves_per_eu) **copied from the GH200-tuned launch**.
  2. `xcompile.py`: compiles each (kernel, target) with a per-target `TRITON_CACHE_DIR`/dump dir (so shared `.ttir` files are not overwritten), saves `ttgir/llir/ptx/cubin/sass` or `amdgcn/hsaco` under `results/isa/<kernel_id>/triton_<variant>/<target>/`, runs `nice -n 19`, ≤ 8 threads, batches ≤ 100 compiles and < 20 CPU-minutes per process.
  3. `isa_report.py`: NVIDIA — `cuobjdump -res-usage` (REG/SHARED/LOCAL), SASS histogram (`LDG.E.128/.64/.U16`, `STG.E.128`, `LDGSTS`, `UTMALDG`, `LDSM`, `HGMMA`/`HMMA`, `SHFL.BFLY`, `MUFU.*`, `BAR.SYNC`, `F2FP`); AMD — `; Kernel info` comments + `llvm-readelf --notes` (`.vgpr_count/.agpr_count/.sgpr_count/.vgpr_spill_count/.group_segment_fixed_size`), instruction histogram (`v_mfma_*` by mnemonic, `global_load_dword{,x2,x4}`/`ushort`, `buffer_load_*`, `ds_read/ds_write` by width, DPP/`ds_swizzle`/`ds_bpermute`, `s_waitcnt`, `v_exp_f32`, `v_rsq_f32`, bf16 conversion sequences), **occupancy recomputed** from the tables with Triton's dynamic LDS and workgroup size; static MFMA cycles per tile from the AMD matrix-instruction calculator (`run_mic.py`). Tools: `/opt/cray/pe/cce/20.0.0/cce-clang/aarch64/bin/{llvm-objdump,llvm-readelf}`; `cuobjdump`/`nvdisasm` from the HPC SDK on PATH.
  4. Run on KT-1, KT-2 and the Triton tutorial matmul (128×128×64 bf16) as the first rows; record compiler versions (Triton LLVM 22.0.0git) in every row.
  5. Static tuning pairs (the AMDGCN half of the proposal's "tuning"): per kernel and target, sweep `num_warps` ∈ {4, 8}, `waves_per_eu` ∈ {default, 2, 4} and `num_stages` ∈ {2, 3} (GEMM), and divisibility hints on/off; each pair is a row with VGPR/AGPR, spills, recomputed occupancy, load width and MFMA count, plus a one-line verdict (e.g. "waves_per_eu=4 caps VGPRs at 128 → occupancy 4 but 64 B scratch spills").
- **Deliverables:** scripts; `results/static_isa.csv`; `results/isa/...` dumps; `docs/findings.md#static-isa` with the first side-by-side (RMSNorm sm_90a vs gfx90a: load width with/without divisibility hints, DPP vs `ds_swizzle` vs `shfl.sync.bfly`, bf16 conversion emulation on gfx9).
- **Acceptance:** CSV has ≥ 3 kernels × 4 targets with no empty metric cells; recomputed occupancy differs from the compiler's printed line for the num_stages=3 matmul (documented); every value labelled STATIC.

### AMD-2 HIP RMSNorm via hipify + CCE clang; single-LLVM NVPTX vs AMDGCN control
- **Session:** S5 (static) · **Depends on:** AMD-1 · **GPU budget:** 0
- **Inputs:** `HIPIFY=$CISC662_WORK/scratch/amd-rocm-planning/dl_hipify/hipify-perl`; `FR=$CISC662_WORK/scratch/amd-rocm-planning/hipify_test/fake_rocm` (HIP 6.2 headers from the Triton wheel + `share/hip/version`); `CB=/opt/cray/pe/cce/20.0.0/cce-clang/aarch64/bin`; always `env -u CPATH`. The `-nogpulib -nogpuinc` + `mini_hip.h` form (`.../toolchain-feasibility/hip_clang/`) is only for shim-based kernels; hipified code that includes `<hip/hip_runtime.h>` needs the `--rocm-path=$FR` form.
- **Steps:**
  0. Create `kernels/cuda/rmsnorm.cu` v0 (forward + backward, warp-per-row design exactly as specified in KC-1), compile-checked with `nvcc -c -gencode arch=compute_90a,code=sm_90a --ptxas-options=-v` and with CCE clang — not loaded as a torch extension and not timed (KC-1 does that).
  1. `perl $HIPIFY rmsnorm.cu > kernels/hip/rmsnorm.hip`; fix wave64: 64-bit shuffle masks, 6-step reduction (or DPP via `__builtin_amdgcn_update_dpp`), `warpSize=64` (768/64 = 12 bf16 per lane → expect `global_load_dwordx2`×3 or `dwordx3`), `__launch_bounds__`/`amdgpu_waves_per_eu`; `__builtin_amdgcn_rsqf` → `v_rsq_f32`.
  2. Compile: `env -u CPATH $CB/clang -x hip --offload-arch=gfx90a --cuda-device-only --rocm-path=$FR -DHIP_ENABLE_WARP_SYNC_BUILTINS -O3 -S rmsnorm.hip -o rmsnorm.gfx90a.s` (repeat for gfx908/gfx942) and `… -c --no-gpu-bundle-output -o rmsnorm.gfx90a.hsaco` (no `-nogpulib`/`-nogpuinc`); `$CB/llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx90a`; `$CB/llvm-readelf --notes`. (Verified in the plan review: NumVgprs 14, Occupancy 8 on gfx90a.)
  3. Control: compile the same `.cu` with CCE `clang -x cuda --cuda-gpu-arch=sm_90a --cuda-device-only -nocudalib -nocudainc --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version -O3 -S` and with nvcc, so one table compares nvcc-SASS / clang-NVPTX / clang-AMDGCN with compiler versions.
- **Deliverables:** `kernels/cuda/rmsnorm.cu` (v0), `kernels/hip/{mini_hip.h,rmsnorm.hip}`, `results/isa/rmsnorm/hip/gfx{908,90a,942}/` (`.s`, `.hsaco`, `.dis.s`, notes), `results/isa/rmsnorm/{nvcc,clang_nvptx}/sm_90a/`, rows in `static_isa.csv`, findings entry.
- **Acceptance:** compiles for all three gfx targets; VGPR/SGPR/occupancy/LDS/load width recorded; the wave64 bug class (32-bit mask → compile error with the HIP 6.2 headers) demonstrated in the notes.

### AMD-3 FA3 cubin dissection + Triton attention/GEMM cross-compile
- **Session:** S7 · **Depends on:** AMD-1, KT-6 · **GPU budget:** 0
- **Steps:** `cuobjdump -lelf` then `-sass` on `$HF_HOME/hub/models--varunneal--flash-attention-3/snapshots/*/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/_C.abi3.so` (63 sm_90a cubins; split the work so no process passes 20 CPU-minutes); pick the forward kernel used at hdim 128 causal bf16 (name from the S1 nsys trace: `flash::FlashAttnFwdSm90<...>`); count `HGMMA` shapes, `UTMALDG` (TMA), `USETMAXREG` (warp-specialization register reallocation), `MUFU.EX2`, `SYNCS`/mbarrier ops, `BAR`; registers via `-res-usage`. Compare with KT-6's SASS (TMA? HGMMA? registers? spills?). Cross-compile KT-6 and the KT-5 GEMM to gfx90a/gfx942: MFMA mix, LDS, VGPR+AGPR occupancy (STATIC). S1 only cross-compiled a vector-add and a plain matmul, so if the TMA-descriptor variants do not lower for HIP targets, use the plain FA2-style attention and the non-TMA GEMM for the AMD rows and record that.
- **Deliverables:** `results/isa/attention/fa3/sm_90a/`, `docs/findings.md#attention` ISA card, rows in `static_isa.csv`.
- **Acceptance:** a table "FA3 vs Triton FA2-style" with the five counts above and the measured % of FA3 from KT-6.

### AMD-4 HIP ports of `qkrope` and `ce`; complete AMD static table
- **Session:** S12 · **Depends on:** AMD-2, KC-2, KC-3 · **GPU budget:** 0
- **Steps:** hipify `qkrope.cu` and `ce.cu` with the AMD-2 recipe; CE needs tanh built from `v_exp_f32`/`v_rcp_f32` (the HIP headers' `tanhf` is fine to call, but record what it lowers to) and the **register-resident** variant (a 64 KB bf16 row fills gfx9's 64 KiB LDS — the shared-memory variant does not port: record this architectural finding); DPP for hot reductions; compile for gfx908/gfx90a/gfx942; add all Tier A/B/C kernels (Triton and HIP) to `static_isa.csv`; analytic AMD roofline points (datasheet peaks) labelled STATIC; `analysis/make_figs_amd.py` reads `results/static_isa.csv` and writes `results/figures/amd_static_table.png`.
- **Deliverables:** `kernels/hip/{qkrope,ce}.hip`, `results/isa/{qkrope,ce}/hip/gfx*/`, updated `results/static_isa.csv`, `analysis/make_figs_amd.py`, `docs/findings.md#amd-tuning`.
- **Acceptance:** every kernel class has sm_90a + three gfx rows; ≥ 2 static tuning pairs per HIP kernel (e.g. `amdgpu_waves_per_eu(1)`→`(2)` or `__launch_bounds__` → VGPR count, spills, recomputed occupancy; DPP vs `ds_swizzle`/`ds_bpermute` reduction → instruction and `s_waitcnt` counts; `global_load_dwordx4` vs `dwordx2` via alignment attributes), each a STATIC row with a one-line verdict; `python -m cisc662.analysis.make_figs_amd` renders the table.

### AMD-5 Cross-generation static comparison (V100/A100 fallback)
- **Session:** S12 · **Depends on:** KC-1, KC-2 · **GPU budget:** 0
- **Steps:** `nvcc -gencode arch=compute_70,code=sm_70 / sm_80 / sm_90a` on the CUDA kernels (bf16 is unsupported on sm_70 → fp16 variant of the kernels for the comparison, documented); Triton cross-compile to `cuda:80`; tabulate registers, SASS MMA class (`HMMA.884` / `HMMA.16816` / `HGMMA`), load widths, shared-memory use; analytic roofline for V100 (125 TF fp16, 0.9 TB/s) and A100 (312 TF, 2.0 TB/s) with the kernels' arithmetic intensities. Labelled STATIC/PROJECTED. Replace with measured rows only if DARWIN/Delta access arrives (D2).
- **Deliverables:** `results/crossgen_static.csv`, `results/isa/<kernel_id>/nvcc/sm_{70,80}/`, `results/figures/crossgen_static.png`, `docs/findings.md#cross-generation`.
- **Acceptance:** the CSV has every CUDA kernel × {sm_70 (fp16 variant), sm_80, sm_90a} with registers, MMA class and load width; the figure renders; every value labelled STATIC/PROJECTED.

### AMD-6 Tier-B measured AMD (only if D2 = access granted)
- **Session:** S10 env, S12 runs · **Depends on:** ADM-4, KT-1..KT-3, AMD-2 · **Budget:** ≤ 6 Delta GPU SU (factor 0.25 ⇒ ≤ 24 MI100/MI210 GPU-hours), 0 DeltaAI SU
- **Steps:** on `login.delta.ncsa.illinois.edu`: prefer the pre-staged `/sw/external/MI100/pytorch-2.10.0-amdgpu.sif` (record the version gap vs 2.9.1) or an x86_64 venv with `torch==2.9.1+rocm6.4` on `/work/nvme` (4.5 GB; shared with DeltaAI); smoke pack: `rocminfo`, `torch.version.hip`, `gcnArchName`; gates G1/G2 on the AMD device for every kernel (`NANOCHAT_ATTN_IMPL=sdpa`, `HF_HUB_OFFLINE=1`); kbench microbenchmarks; `rocprofv3 --kernel-trace --stats --output-format csv`; `rocprof-compute profile -n <name> --roof-only` on 2–3 kernels (check MI100 roofline support); the AMD-1/AMD-4 static tuning pairs re-timed; R12-A2 100 steps + D1 on MI210 if time. Jobs `--partition=gpuMI100x8 --gpus=mi210:1` (or `mi100:1`), ≤ 30 min, ≤ 16 cores and ≤ 250 GB per GPU; never the whole node. Results to `results/db/amd_<arch>/`, rows marked MEASURED.
- **Acceptance:** at least one measured kernel time per Tier-A kernel on one AMD arch, with rocprofv3 timings, and the static VGPR/LDS numbers validated against `kernel_trace.csv`.

---

## INT — integration and end-to-end

### INT-1 Integration shim + graph-break gate (G3)
- **Session:** S6 · **Depends on:** KT-1, KT-2, KT-3 (KT-4 when ready) · **GPU budget:** 0.4
- **Steps:**
  1. `cisc662/integrate.py`: reads `CISC662_KERNELS` (comma list of `rmsnorm,mixnorm,qkrope,ce,ce_chunked,mlp`, with per-kernel backend override syntax `rmsnorm:cuda,qkrope:triton`) and `CISC662_BACKEND=triton|cuda` (default `triton`; CUDA wrappers are imported lazily so a Triton-only run never builds the extensions). It monkeypatches `nanochat.gpt.norm`, `nanochat.gpt.apply_rotary_emb` (together with the QK-norm/scale lines: install a patched copy of `CausalSelfAttention.forward` that calls `qkrope` and, if `fold_scale`, passes `softmax_scale=1.44/sqrt(128)` through the patched `flash_attn_func` wrapper), and the loss path (a patched copy of `GPT.forward`, KT-4 step 2, which keeps the `targets is None` and `loss_reduction="none"` paths). **No upstream edit:** patching module globals works because `gpt.py` calls `norm`/`apply_rotary_emb` by name and nothing imports them (`checkpoint_manager.py` and `base_train.py` import only `GPT, GPTConfig, Linear`); the three documented patches remain the only upstream changes and ADM-3's acceptance stays as written. Naming: every project kernel function is `c662_<kernel_id>_<fwd|bwd>[_variant]`.
  2. `bench/train_wrap.py` is the only entry point: import the shim, then `runpy.run_module("scripts.base_train", run_name="__main__")`.
  3. Extend `profiling/nsys_attrib.py` LIB_RULES and `profiling/classify_kernels.py` with project-kernel rules placed before the generic ones: `('rmsnorm', r'c662_(rmsnorm|mixnorm)'), ('rope', r'c662_qkrope'), ('softmax_ce', r'c662_ce'), ('relu2_mlp', r'c662_mlp'), ('gemm', r'c662_gemm'), ('attention', r'c662_attention'), ('optimizer', r'c662_muon')`.
  4. G3 job (`slurm/s6_integration.sbatch`, 1 GPU, 25 min): for each kernel set, run R12 for 30 steps with `TORCH_LOGS=graph_breaks,recompiles` and one `evaluate_bpb` (`--eval-every=25 --eval-tokens=65536`) so the `loss_reduction="none"` path is exercised before INT-2; assert zero new graph breaks vs stock; nsys capture (steps 20–24) proving our kernels run and the replaced stock kernels are gone; record the step time delta and the per-step GPU kernel time.
- **Deliverables:** shim, wrapper, classifier rules, `results/runs/s6_integration_<jobid>/` with `graph_breaks.txt` per variant and `attrib_*.json` via `profiling/nsys_attrib.py`.
- **Acceptance:** G3 pass for every integrated kernel (or the exact graph break documented with the fix plan); `attrib_compile_all.json` lists `c662_*` kernels under their class and the matching stock kernels (`triton_per_fused_*rsqrt*`, `triton_red_fused__log_softmax*`) are absent.
- **Pitfalls:** `torch.compile(dynamic=False)` in base_train — custom ops need `register_fake`; a `custom_op` (opaque) can block Inductor's fusion of neighbours — measure in INT-3 and report both isolated and in-model numbers.

### INT-2 Loss-curve gates (G4) for each integrated variant
- **Session:** S6 · **Depends on:** INT-1, BAS-2 · **GPU budget:** 0.6
- **Steps:** `slurm/s6_loss_gates.sbatch` (1 GPU, 30 min; split if > 4 runs): R12-L (1000 steps) for `qkrope`, `ce` (or `ce_chunked`), `rmsnorm`/`mixnorm`, and `all`; `analysis/losscurve.py` against the stock run and the envelope; overlay figure with the delta panel.
- **Acceptance:** PASS/FAIL per variant recorded in `results/db/loss/<variant>.json` with `gates_passed` updated; a failing variant is excluded from end-to-end tables and gets a findings entry.

### INT-3 End-to-end timing, first pass (compile and eager)
- **Session:** S6 · **Depends on:** INT-1 · **GPU budget:** 0.5
- **Steps:** `slurm/s6_e2e_timing.sbatch` (1 GPU, 30 min): R12 110 steps × {stock, +qkrope, +ce, +rmsnorm, +all} × 3 reps, **interleaved ABAB**, compile mode; eager mode for {stock, +all}; `gpu_monitor.csv`; summarize with `summarize_s1.py` (extend its name parsing) → `results/e2e_table.md` (s/100 steps median, min–max, speedup vs stock with the range rule). Next to the wall-clock numbers report the nsys per-step GPU kernel time and GPU-busy fraction for stock and each variant (the compiled step contains ≈ 6 ms of host time: 80.5 ms wall vs 74.3 ms kernel time, the same size as the per-kernel savings) and one fixed-batch control run (synthetic inputs, no dataloader) to bound the host share.
- **Acceptance:** table complete; peak memory per variant; a one-paragraph interpretation (which kernel moved the step time, whether in-model GPU-time Δ matches the isolated microbench).

### INT-4 P1 data freeze (one job) + before/after nsys
- **Session:** S8 · **Depends on:** INT-3 (INT-2 if a loss curve is shown) · **GPU budget:** 0.3
- **Steps:** one job (`slurm/s8_p1_freeze.sbatch`, 20 min): stock vs the best gated variant, ABAB × 3 reps, then an nsys capture of `+all` (steps 20–24) → `attrib_compile_all.json` for the before/after stacked bars (wall-clock and GPU-time deltas both reported). Freeze every number a P1 slide uses into `results/p1_freeze/` (copies of the JSON/CSV) with job IDs. P1 needs only the *required* list in PLAN.md §10; the rest is included only if already gated (D4). Independent-node repeats belong to the final campaign (INT-6 J14b-rep), not to P1.
- **Acceptance:** `results/p1_freeze/MANIFEST.md` lists every number on a P1 slide and its job ID; the job stayed ≤ 0.3 SU.

### INT-5 Decision D5: per-kernel go/no-go for the final campaign
- **Session:** S12 · **Depends on:** KC-1..KC-3, INT-2 · **GPU budget:** 0.6
- **Steps:** rerun G3 + G4 for **every integrated variant not yet gated individually** (`slurm/s12_gates.sbatch`, split into ≤ 30-min jobs): all CUDA variants (`CISC662_BACKEND=cuda`), `mlp` (KT-7), whichever of `ce`/`ce_chunked` INT-2 did not run, `mixnorm` if INT-2 ran only `rmsnorm`, and any Triton kernel changed since its G4; nsys on `+all` (best backend per kernel) confirming the stock kernels are gone and no new ones appeared; decide per kernel: in-model GPU-time speedup ≥ 1.0 and gates passed → in the final campaign; otherwise excluded and written up.
- **Acceptance:** STATUS.md D5 table (kernel, variant/backend, G1–G4, in-model Δ, decision).

### INT-6 Final measurement campaign (code freeze 12-08)
- **Session:** S14 · **Depends on:** INT-5, TUNE-1 · **GPU budget:** 4.5 (cap; expected ≈ 3.7 SU — the job limits below sum to ≈ 220 GPU-minutes)
- **Steps:** `git tag p2-freeze`; submit at session start (≤ 2 in flight; each ≤ 30 min):
  - J14a loss curves: stock ×2 (fresh envelope), `+all` with `CISC662_BACKEND=triton`, `+all` with `=cuda` (R12-L), plus a per-kernel R12-L for any kernel whose last G4 pass predates its last code change — 30 min (split if > 4 runs)
  - J14b end-to-end compile: stock, +each kernel, +all, ABAB × 5 reps — 30 min; J14b-rep: three independent 15-min jobs (likely different nodes) each running stock vs +all ABAB × 2, so node-to-node variance enters the CI; J14c the same as J14b in eager for stock/+all + SDPA variants — 25 min; all with the nsys GPU-time companion numbers (INT-3)
  - J14d kernel microbenchmarks: all classes × {eager, compile, Triton, CUDA}, 5 trials, L2 flushed; pinned-clock repeat if BAS-3 found `--gpu-freq` works — 30 min
  - J14e-1 ncu (`--profile-from-start off`, window `NANOCHAT_PROFILE_STEPS=20:22`, 1 launch each): `--set full` on each optimized Triton kernel and its Inductor counterpart — 25 min, nvperf; J14e-2: the CUDA kernels + roofline sections on 6 kernels + `--clock-control none` duplicates — 25 min, nvperf
  - J14f nsys: train stock vs +all; decode stock vs optimized (DEC-2) — 20 min
  - J14g CPU reference refresh (CPU-only, ≈ 0.1 SU) if any kernel changed
- **Deliverables:** `results/final/` (every row with `gates_passed`), `results/final/MANIFEST.md`.
- **Acceptance:** `python -m cisc662.analysis.make_figs_p2` regenerates every P2 figure from `results/final/` without manual edits.

---

## DEC — decode path (tokens/sec metric)

### DEC-1 bf16 weight cache + corrected MBU accounting + decode nsys
- **Session:** S11 · **Depends on:** INF-1, BAS-3 · **GPU budget:** 0.4 (in `slurm/s11_decode.sbatch`)
- **Inputs:** inventory "decode Linear weight casts" (660 MB per token of fp32→bf16 casts = 73% of weight traffic at b=1; 581 launches per token; 2 host syncs per token); S1: TPOT flat ≈ 10–13 ms for b=1…128 ⇒ launch-bound; GPU b=1 decode 68–107 tok/s across S1 jobs vs 32.8 tok/s on one CPU thread (2.1–3.3×; BAS-3's re-measurement is the number to quote).
- **Steps:** `bench/decode_opt.py`: (1) cast all `Linear` weights to bf16 once after `load_model` (inference only); (2) recompute MBU with actual bytes (bf16 weights 220 MB + KV read) and 4.0 TB/s; (3) nsys of 64 decode tokens at b=1 and b=128 for stock vs cached: GPU-busy time per token vs idle gaps (`nvtx` ranges per token), launches per token; (4) greedy-token identity check vs stock for 256 tokens (b=1, 8).
- **Deliverables:** `results/db/decode/*.json`, `results/decode_table.md`, `results/figures/decode_timeline_stock.png` (from nsys trace: kernel time vs gap per token).
- **Acceptance:** tokens identical (or ≤ 1 bf16-tie divergence, explained); per-token GPU-busy time and launch count reported; MBU corrected.

### DEC-2 Sync removal + CUDA graphs for the T=1 step
- **Session:** S11 · **Depends on:** DEC-1, KT-2 · **GPU budget:** 0.6 (in `slurm/s11_decode.sbatch`)
- **Steps:** in a cisc662 copy of the engine step (not upstream): device-side position tensor (no `cache_seqlens[0].item()`), cos/sin gathered by a position tensor instead of Python slicing, static KV cache, FA3 `flash_attn_with_kvcache` with a fixed `num_splits`; capture the per-token forward (+ argmax at temperature 0) in a `torch.cuda.CUDAGraph`; alternative: `torch.compile(mode="reduce-overhead")` on the T=1 forward; then plug the fused T=1 kernels (KT-2 path with the unfused scale convention, KT-1 norms). Measure D1 (b=1,8,32,128) for stock / +bf16 cache / +graphs / +kernels; nsys timeline before/after; greedy-token identity.
- **Deliverables:** `cisc662/decode/engine_step.py` (graph-capturable T=1 step), `bench/decode_opt.py` extensions, `results/db/decode/*.json`, `results/decode_table.md`, `results/figures/decode_timeline_opt.png`.
- **Acceptance:** tok/s table with CIs; nsys shows the idle gaps closed (GPU-busy fraction per token); identity check passed; the b=1 result compared with the bandwidth roofline (16.4k tok/s ceiling with bf16-resident weights).

---

## TUNE — instruction-level tuning round

### TUNE-1 Fix the top ncu finding per Tier-A kernel (before/after rows)
- **Session:** S13 · **Depends on:** KC-1..KC-3, BAS-6 · **GPU budget:** 0.8 (in `slurm/s13_kernels.sbatch`)
- **Steps:** for each of RMSNorm, QK-norm+RoPE, softcap-CE (Triton and CUDA): take the dominant stall/limit from ncu (long_scoreboard → more ILP; mio_throttle → fewer shuffles/shared; occupancy-limited → register cap via `__launch_bounds__`/`maxnreg` vs spills; sectors/request → vector width/alignment); change one thing; re-measure with kbench + ncu; keep both rows. Include at least one documented **failure** (e.g. "capping registers raised occupancy but caused spills and was slower").
- **Deliverables:** `docs/findings.md#tuning-log` with before/after counters and SASS diffs; updated `results/db/*`.
- **Acceptance:** ≥ 3 before/after pairs with counters; at least one failure explained at the instruction level.

---

## ANA — analysis, figures, presentations, report

### ANA-1 `analysis/roofline.py` v0 (S3) → v1 with ncu bytes (S4)
- **Session:** S3 (v0), S4 (v1 after BAS-6) · **Depends on:** — / BAS-6 · **GPU budget:** 0
- **Steps:** ceilings per PLAN.md §7.4; points: per kernel class (eager and compile) with analytic FLOPs/bytes from the inventory (v0) and ncu bytes (v1); log-log; memory-bound vs compute-bound shading; style per `make_figs_s1.py` (palette, no dual axes). Output `results/figures/roofline_v0.png`, `roofline_v1.png`; final version in ANA-7 with before→after arrows.
- **Acceptance:** figure renders; GEMM/attention points on the compute side, RMSNorm/RoPE/CE on the memory side; ridge ≈ 198 FLOP/B (measured) marked.

### ANA-2 `analysis/make_figs_p1.py`
- **Session:** S8 · **Depends on:** INT-4, BAS-4, BAS-5, AMD-1 · **GPU budget:** 0
- **Figures/tables (each stamped with job IDs).** *Required:* baseline table; stacked kernel-class bars eager vs compile (+ compile+ours if INT-4 ran); roofline v1 (v0 fallback); goal (i) bars (1 core / 72 cores / GPU, train and decode, with the decode GPU spread); one preliminary kernel row (µs vs eager/compile with CI, GB/s, % of roof, G1/G2 status, labelled preliminary); budget-used chart. *If available by the S8 freeze:* the three-kernel results table; loss-curve overlay for the first integrated variant with the envelope; ISA side-by-side excerpt (RMSNorm sm_90a SASS vs gfx90a AMDGCN) rendered as a table; matmul codegen table from `static_isa.csv`. The script must run with the optional inputs missing.
- **Acceptance:** `python -m cisc662.analysis.make_figs_p1` regenerates all PNGs into `results/figures/p1/` from `results/p1_freeze/`.

### ANA-3 Method and protocol figures (no GPU)
- **Session:** S8 · diagrams: nanochat d12 block with the five kernel classes at their call sites (`gpt.py` line numbers: norm 42, apply_rotary_emb 57, attention 110, MLP 137–141, softcap/CE 511–520); optimization ladder L0–L5 with tools; correctness-gate flow. SVG/PNG into `presentations/p1/figs/`.

### ANA-4 Presentation 1 deck (~12–15 slides)
- **Session:** S9 · **Depends on:** ANA-2, ANA-3 · **GPU budget:** 0.3 (emergency only)
- **Outline:** 1 title; 2 motivation (where time goes in small-LLM training/inference; data-movement framing, "Data Movement Is All You Need"); 3 nanochat overview with the five kernel classes; 4 **platform reality** (proposal vs actual: GPU GH200 instead of V100, CPU host Grace aarch64 instead of EPYC x86, V100/A100/AMD access status, static AMD tier, multi-GPU dropped, MQA→MHA correction, toolchain provenance — own venv for torch/Triton/FA3, system HPC SDK for nvcc/ncu/nsys, CCE clang for HIP); 5 reduced config R12/D1 and why (incl. window L); 6 method ladder; 7 protocol (timing, gates, loss curves, clocks); 8 baselines table; 9 kernel-class breakdown eager vs compile; 10 roofline v1; 11 goal (i) preview (CPU vs GPU; decode only 2–3× at b=1 with its spread — the launch-bound story); 12 first kernel results (preliminary); 13 ISA side-by-side STATIC (sm_90a vs gfx90a) + matmul codegen table, if available; 14 what we expect vs eager and vs compile (calibrated by prior art: QuACK, Liger, CCE, FA3, modded-nanogpt; vs-compile headroom = achieved bandwidth and cuBLAS-boundary bytes), risks, timeline, budget used. Speaker notes per slide. Build with the repo's `paper-slides`/`pptx` tooling or Markdown; every number traceable to `results/p1_freeze/MANIFEST.md`.
- **Acceptance:** deck in `presentations/p1/`; timed rehearsal ≤ the allotted time; a Q&A sheet (`presentations/p1/QA.md`) covering: why GH200 and not DARWIN, AMD status and the static tier, V100/A100 fallback, multi-GPU dropped, MQA→MHA, CPU reference on Grace, toolchain provenance, why window L, budget used.

### ANA-5 ISA cards in `docs/findings.md` (continuous)
- Template per kernel variant: **Source** (eager/Inductor/Triton/CUDA/HIP; compiler + version) · **NVIDIA static** (regs, spills, shared, theoretical occupancy; SASS counts: LDG.E.128/.64/.U16, STG, LDGSTS, UTMALDG, LDSM, HGMMA/HMMA, SHFL, MUFU.*, BAR, F2FP) · **NVIDIA measured** (ncu: duration @ clock mode, dram bytes vs minimum, achieved GB/s, achieved occupancy, sectors/request, top-2 stalls, % of 3.65 TB/s or 723 TF/s) · **AMD static** (VGPR/AGPR/SGPR, spills, LDS, recomputed occupancy for gfx908/gfx90a/gfx942, load width, MFMA mnemonic and count, reduction primitive, s_waitcnt count) · **Verdict** (one paragraph tying the time to specific instructions/counters; failures written with the same rigor). Owner: whoever measures the variant.

### ANA-6 Reconciliation pass (nsys vs CUDA events vs ncu)
- **Session:** S13 · per kernel: nsys in-model time × launches vs kbench isolated time within 10%; explain gaps (L2 state, fusion loss, clocks, host time). Table in `docs/findings.md#reconciliation`.

### ANA-7 `analysis/make_figs_p2.py` + final roofline
- **Session:** S15 · **Depends on:** INT-6 · all P2 figures/tables listed in PLAN.md §10 from `results/final/`; before→after roofline arrows; the AMD STATIC/MEASURED table via `make_figs_amd`; cross-generation table; decode table and timelines; goal (i) table; budget chart.
- **Acceptance:** one command regenerates everything; `MANIFEST.md` maps every figure to job IDs.

### ANA-8 Presentation 2 deck
- **Session:** S15–S16 · results-focused per PLAN.md §10, including 3–4 instruction-level case studies (QK-norm+RoPE; softcap-CE; GEMM vs cuBLAS failure; Triton attention vs FA3), "what did not work and why", the proposal-vs-delivered checklist (PLAN.md §1 rows, including the platform, CPU-host ISA, MQA→MHA, multi-GPU, window-L and toolchain-provenance deviations, and STATIC vs MEASURED labels for AMD/cross-generation), budget used.

### ANA-9 P2 rehearsal and fixes (human; S16; GPU only for a broken figure, ≤ 0.3 SU)

### ANA-10 Final report
- **Session:** S17 · structure: motivation; application and platform (with the proposal deviations); reduced config and workloads; method ladder; protocol; baselines; per-kernel results (goal ii) with ISA cards; roofline; goal (i); AMD (static + measured if any); cross-generation; decode; failures and lessons (goal iii); threats to validity (torch 2.9.1 pinning, reduced-config representativeness incl. window L, compiler-version confounds, toolchains partly from system modules, static AMD, Grace instead of EPYC for goal i, host time inside the wall-clock metric); reproducibility appendix (commands, job IDs, versions, SU used). Source: `docs/findings.md`, `results/final/`.
