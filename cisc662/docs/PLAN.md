# CISC662 project plan — Optimizing nanochat down to the assembly level (NVPTX and AMDGCN)

**Team:** Talha Mahmood (`tmahmood1`, lead), Hamza Mahmood. **Written:** 2026-10-18, end of Session 1.
**Companion documents:** [TASKS.md](TASKS.md) (every task as an executable card), [AGENT_GUIDE.md](AGENT_GUIDE.md) (rules for whoever executes a task — human or agent), [STATUS.md](STATUS.md) (live state; read it first every session).
**Research behind this plan:** `docs/research/*.json` — DeltaAI policy, toolchain feasibility, nanochat hot-path inventory, AMD path, prior art. Cite them instead of re-deriving.

---

## 1. What the proposal commits to, and how each commitment is delivered

| Proposal commitment | How we deliver it | Status |
|---|---|---|
| Reduced configuration; baselines "seconds per 100 training steps" and "decode tokens/sec" with `base_train.py` / `infer_bench.py` | Config **R12** (§3) frozen; baselines measured in S1 (§4) | Done, re-measured in final campaign |
| Baselines: PyTorch eager **and** torch.compile | `TORCH_COMPILE_DISABLE=1` vs stock; both measured | Done |
| Target kernels: GEMM, attention, RMSNorm, RoPE, softmax/cross-entropy | Three tiers (§5): A = RMSNorm, fused QK-norm+RoPE, fused softcap+CE (+chunked lm_head); B = GEMM (Triton + CUDA ladder, expected negative result vs cuBLAS); C = attention (Triton FA2-style vs FA3, FA3 SASS dissection) | Planned |
| Method: eager → compile → Triton → CUDA → HIP → NVPTX/SASS and AMDGCN inspection and tuning | Ladder L0–L5 (§5): **full L0–L5 for Tier A** (RMSNorm, QK-norm+RoPE, softcap+CE); **GEMM** L0–L3 + L5, L4 static only (Triton cross-compile, plus a HIP port of the FFMA v1 kernel); **attention** L0, L2, L4-static, L5 (no hand-written CUDA/HIP attention — FA3 is dissected instead). AMDGCN "tuning" is compile-time/static (VGPR, occupancy, load width, MFMA shape; before/after pairs in AMD-1 and AMD-4) unless D2 grants hardware, in which case AMD-6 adds rocprofv3 timings for the same pairs. Every variant gets an "ISA card" | Planned |
| Per-kernel time from the profiler, not Python wall clock | In-model per-kernel time from nsys (`profiling/nsys_attrib.py`, S1; INT-1/INT-4/INT-6); isolated kernels timed with CUDA events (never Python wall clock); ncu only for counters; ANA-6 reconciles the three | Done S1, continuous |
| Tools: Nsight Systems/Compute; rocprof/Omniperf; cuobjdump/nvdisasm; llvm-objdump | nsys + ncu verified working for users on DeltaAI; cuobjdump/nvdisasm/ptxas on PATH; llvm-objdump + clang with AMDGCN target from Cray CCE 20. rocprofv3 + rocprof-compute (formerly Omniperf) can only be used in AMD-6 if D2 grants AMD hardware; otherwise unused, and said so in P1/P2 | NVIDIA verified S1; AMD conditional |
| Toolchains installed in our own environment rather than system modules | **Partly changed.** Own: torch 2.9.1 + Triton 3.5.1 (with its bundled ptxas 12.8) + FA3 in a uv venv on /work/nvme. System: nvcc 12.9 / ncu 2025.2 / nsys 2025.3 from the HPC SDK 25.5 module (already current on DeltaAI, no reason to duplicate), and Cray CCE 20 clang for HIP→AMDGCN. ROCm cannot be installed on aarch64; HIP headers come from the Triton wheel and `hipify-perl` from a checkout. Every ISA row records its compiler + version (R10) | Stated in P1 |
| Roofline separating memory-bound from compute-bound kernels | Measured GH200 ceilings (3.65 TB/s copy, 723 TF/s bf16) + datasheet (4.0 TB/s, 990 TF/s); per-kernel points from ncu bytes/FLOPs | Planned (S3 v0, S14 final) |
| Goal (i): GPU speedup over the serial CPU reference, interpreted against the parallel hardware | Measured S1: one Grace core vs 72 cores vs GH200, full step and decode; per-op CPU references added later | Partly done |
| Goal (ii): 1.3–2× on selected hot kernels over eager **and** torch.compile | Realistic vs eager for all memory-bound classes; vs compile only for cross-boundary fusions (lm_head+CE, decode path, relu² epilogue). Claims need CI lower bound ≥ 1.3 (§7) | Planned |
| Goal (iii): instruction-level explanation of why each optimization helps or fails | ISA cards + ncu counters per variant; at least two documented negative results (GEMM vs cuBLAS, Triton attention vs FA3) | Planned |
| Numerical agreement + loss-curve comparison before any timing | Gates G1–G4 (§7.3); results DB refuses rows without gates | Planned (S2 harness) |
| Platforms: DARWIN V100 primary; ACCESS A100/H100; AMD MI100/MI250X | **Changed.** DeltaAI GH200 (H100-class) is primary. V100/A100: no access from here → static sm_70/sm_80/sm_90a SASS comparison unless access arrives. AMD: no ROCm on DeltaAI → **Tier A guaranteed = static AMDGCN** (Triton cross-compile + HIP via CCE clang for gfx908/gfx90a/gfx942); **Tier B opportunistic = measured** if NCSA Delta `gpuMI100x8` (MI100 + MI210) access is granted. Presentation 1 states this openly. | Access requests = first task of S2 |
| Serial CPU reference "on the host CPU" (proposal: AMD EPYC x86-64) | **Changed.** Grace Neoverse-V2 (aarch64, SVE2) cores; goal (i) reported raw and normalized by measured single-core FMA/STREAM peaks (BAS-5) | Measured S1, normalization in S4 |
| Application described as "multi-query attention" | **Correction.** `base_train.py:138` sets `n_kv_head = n_head`, so R12 is plain MHA (6 query / 6 KV heads; KV cache 6× larger than MQA). Attention and decode numbers are for MHA; stated on the P1 platform slide | Stated in P1 |
| Multi-GPU "if time permits" | Dropped (budget, scope). On the P1 platform slide and in the Q&A sheet (ANA-4). | — |

## 2. Platform facts that every task relies on (verified 2026-10-17)

- **DeltaAI GH200**: 132 SMs (sm_90), 95 GiB HBM3, 60 MiB L2, max SM clock 1980 MHz, 900 W module cap. **Power-capped under sustained tensor-core load** (BAS-3, job 3406282): a 60-s 8192³ bf16 GEMM loop runs at 1365 MHz, ≈ 545 W, `sw_power_cap`, 591 TF/s (the 723 TF/s ceiling is a short-burst number); `srun --gpu-freq=1755` is not honoured under the cap ⇒ clocks stay default and are logged; comparisons must be interleaved on one GPU. Measured: copy 3.65 TB/s, read 3.40 TB/s; GEMM 8192³ bf16 723, fp16 694, tf32 382, fp32 48.6 TFLOP/s. Grace host: 72 Neoverse-V2 cores per GPU.
- **Billing**: 1 SU = 1 GH200-hour; charge = max(GPU×1000, cores×13.89, GiB×9.02) per hour (MAX_TRES). 1 GPU + ≤72 cores + ≤110 GB = 1 SU/h; CPU-only 2 cores + 16 GB ≈ 0.14 SU/h. `ghx4` = 1×, `ghx4-interactive` = 2× and one job per user; `test`/`full` are staff-only. Never `--exclusive`, `--gpus=4`, `--constraint=perf|xpmem` (whole node, 4 SU/h).
- **Queue**: ~2,000 pending jobs. S1 jobs (≤ 45 min) started 8–29 s after becoming eligible; cluster-wide, ≤ 30-min GPU jobs typically start within minutes but a 30-min wait is common, and longer limits have a much longer tail (the 1-h S1 job was cancelled after 8 min pending with a start estimate 8 h out). Rule: ≤ 30 min walltime; submit all of a session's independent jobs at session start (≤ 2 in flight) instead of chaining them.
- **Pool**: `bffp-dtai-gh` shared by 23 users, 6,604 SU left. **Project budget: ≤ 10 SU before P1, ≤ 25 SU planned total, 40 SU hard cap.** S1 used 0.33 SU.
- **Storage**: code in `/u/tmahmood1/project/nanochat` (git branch `cisc662`); everything heavy in `/work/nvme/bffp/tmahmood1/cisc662` (11 GB now, keep < 50 GB). `/projects/bffp` and `/work/hdd/bffp` are over quota with grace expired — never write there.
- **Login node**: per-user 16 cores, 16 GiB throttle / 20 GiB OOM, 1000 tasks. **NCSA kills any login-node process that exceeds 30 minutes of CPU time** (policy notice received 2026-10-17: a filesystem-wide `bfs … / -name claude_skills` search spawned by the Claude Code harness was killed after 36 CPU-minutes; the tmux server was also found dead on 10-18). Consequences: no torch.compile or benchmarks there; cross-compiles and analysis are fine (niced, ≤ 8 threads, split so no single process passes ~20 CPU-minutes); never search from `/` or `$HOME`; treat tmux as optional — batch jobs never depend on it; a long Claude Code session itself accumulates CPU time, so restart it every few hours.
- **Toolchain**: Python 3.10 venv (uv) with torch 2.9.1+cu128, triton 3.5.1 (AMD backend + bundled ptxas/cuobjdump/nvdisasm), FA3 `varunneal/flash-attention-3` aarch64 build cached (`HF_HUB_OFFLINE=1` in jobs). nsys 2025.3, ncu 2025.2 (counters work without any constraint; `--constraint=nvperf` is free and policy-compliant). Cray CCE 20 clang (`/opt/cray/pe/cce/20.0.0/cce-clang/aarch64/bin`) compiles HIP → AMDGCN and CUDA → NVPTX without ROCm; run it with `env -u CPATH`. Login shells export `CC=cc CXX=CC` (Cray wrappers) — `env.sh` overrides to gcc/g++.

## 3. Frozen workloads

| ID | Definition | Metric |
|---|---|---|
| **R12** (training) | `scripts.base_train --depth=12 --max-seq-len=2048 --device-batch-size=16 --total-batch-size=32768 --window-pattern=L --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --num-iterations=110` — d12: 768 wide, 6×128 heads (MHA), vocab 32768, 32,768 tokens/step, no grad accumulation, bf16, FA3 | sum of `dt` over steps 10–109 = **s/100 steps** |
| Modes | compile+FA3 (stock), compile+SDPA (`NANOCHAT_ATTN_IMPL=sdpa`), eager+FA3 (`TORCH_COMPILE_DISABLE=1`), eager+SDPA | |
| **R12-L** (loss curve / gate) | R12 with `--num-iterations=1000 --eval-every=250 --eval-tokens=2097152`, fixed seed and data order (~2.5 min incl. compile) | the logged train loss (already nanochat's debiased EMA(0.9) — never smooth it again) per step; val bpb |
| **D1** (decode) | `scripts.infer_bench -i base -g r12_conv --prompt-tokens 512 --decode-tokens 256 --batch-sizes 1,8,32,128 -t 0` on checkpoint `r12_conv` (2000-step run, val bpb 1.0505) | TPOT, tok/s, plus nsys GPU-busy time per token |
| **OPS** (per-op microbench) | Exact in-model shapes: RMSNorm (32768,768) and (196608,128); RoPE q/k (16,2048,6,128); softcap+CE (32768×32768); GEMM 32768×768×{768,2304,3072,32768} and 32768×3072×768; attention (16,2048,6,128) causal | CUDA-event time, achieved GB/s or TFLOP/s |
| **CPU** | Same shapes at reduced N (scaled), 1 Grace core and 72 cores; full R12 step at batch 1 already measured | s/step, tok/s |
| Portable **R12-A2** | R12 with `--device-batch-size=8` (grad accum 2) — only if V100/A100/AMD hardware appears | |

Why R12: nanochat's own reference model (hyperparameters tuned at d12); the smallest size where every target kernel is non-trivial (K=768 GEMMs, T=2048 attention, 32768-wide CE, 768³ Muon bmm). Caveats for slides: at 32K tokens/step the optimizer is ~10% of the step and lm_head is 23% of matmul FLOPs — both over-represented vs the speedrun; and R12 uses window pattern L (full causal) instead of nanochat's default SSSL so that the SDPA, CPU and AMD paths are comparable (≈ 4 ms/step more attention than the repo default).

## 4. Baselines measured in Session 1 (jobs 3400653, 3400686, 3400723–3400725; 0.33 SU)

| Metric | compile+FA3 | compile+SDPA | eager+FA3 | eager+SDPA |
|---|---|---|---|---|
| s/100 steps (3 reps, mean ± sd) | **8.046 ± 0.031** | 9.114 ± 0.015 | 15.080 ± 0.018 | 15.983 ± 0.008 |
| tok/s · MFU (nanochat count) | 407k · 36.5% | 360k · 32.2% | 217k · 19.5% | 205k · 18.4% |
| peak memory | 14.8 GiB | 14.8 | 29.4 | 29.4 |

GPU kernel time per step by class (nsys, 5 steps): eager+FA3 145.3 ms — GEMM 34.2, softcap+CE 29.6, optimizer 19.4, residual/elementwise 17.1, RoPE 14.0, attention 10.1, RMSNorm 9.4, relu² 7.9. compile+FA3 74.3 ms — GEMM 37.2, attention 10.8, optimizer 7.6, RMSNorm 5.3, softcap+CE 4.5, relu² 3.3, RMSNorm+RoPE fused 2.8, other < 2. Phase split (compile): forward 30%, backward 59%, optimizer 10%.

Decode D1 (eager, fp32 master weights cast per call): b=1 ≈ 83 tok/s (TPOT 11.6 ms), b=8 ≈ 720, b=32 ≈ 2.7k, b=128 ≈ 10.5k tok/s; MBU 2–9%. **Launch-bound**: TPOT is flat in batch size. CPU reference: 1 Grace core 39.9 s/step (3,986 s/100 steps; GPU compile at batch 1: 3.02 s/100 → **1,320×**); 72 cores 2.27 s/step (GPU 75×); decode at batch 1: CPU 1 thread 32.8 tok/s vs GPU **68–107 tok/s across S1 jobs (2.1–3.3×)** — BAS-3 re-measures both in one job; either way it motivates the decode work.

Facts that set the vs-compile targets (from the S1 traces and the Inductor-generated code in the project cache): eager `F.rms_norm` already runs one fused CUDA kernel; Inductor already fuses RoPE + QK-norm + scale into one forward kernel (`triton_per_fused_..._cat_mean_mul_neg_pow_rsqrt_...`) but it runs at only ≈ 1.8 TB/s (110 µs per call for ≈ 201 MB); Inductor's CE tail is two per-row reduction kernels (4.45 ms/step): the forward reads the bf16 logits once and stores only per-row max and log-sum-exp (online softmax); the backward re-reads the bf16 logits, recomputes softcap + log-softmax from the saved row stats and overwrites the logits buffer in place with bf16 dlogits — traffic ≈ 6.4 GB/step (2.15 + 4.3; to be confirmed by BAS-1 `dram__bytes`), yet the forward kernel reaches only ≈ 0.95 TB/s and the backward ≈ 2 TB/s. **The vs-compile headroom is achieved bandwidth and duplicated transcendental work, not bytes**: a fused tail at ~3 TB/s moving 4.3–6.4 GB is ≈ 1.5–2.2 ms vs 4.45 ms. The Linear fp32→bf16 weight cast costs 0.66 GB per forward and 73% of decode weight traffic at batch 1.

## 5. Kernel strategy: tiers and the optimization ladder

Ladder for every kernel: **L0** eager at real shape → **L1** torch.compile (Inductor code dumped with `TORCH_LOGS=output_code`) → **L2** our Triton kernel (custom autograd, registered with `torch.library`) → **L3** hand-written CUDA (`torch.utils.cpp_extension.load`, sm_90a) → **L4** HIP (hipify-perl + wave64 fixes, compiled with CCE clang to gfx908/gfx90a/gfx942; run on hardware only if Tier-B AMD access exists) → **L5** inspection and tuning from SASS/AMDGCN + ncu counters, fed back into L2–L4.

| Tier | Kernels | Depth | Expected vs eager | Expected vs compile | Why |
|---|---|---|---|---|---|
| **A** | RMSNorm (768 and 128-wide), fused residual/λ-mix + RMSNorm, fused QK-norm+RoPE+scale (fwd+bwd), fused softcap+CE, chunked lm_head+CE | L0–L5 both ISAs | RoPE 3–6×, CE tail 5–10×, RMSNorm 1.0–1.3× | CE/lm_head 1.3–2× on time (headline); QK-norm+RoPE forward 1.3–1.6× if ≥ 2.7 TB/s is reached (Inductor's fused kernel runs at ≈ 1.8 TB/s); RMSNorm ≈ parity (explained at ISA level) | Memory-bound; vs-compile headroom = Inductor kernels well below the bandwidth roof (CE forward ≈ 0.95 TB/s, fused RoPE/QK-norm ≈ 1.8 TB/s) plus bytes at the cuBLAS boundary (lm_head+CE chunking, relu² epilogue) |
| **B** | GEMM: Triton tiled/persistent at nanochat shapes; relu² epilogue in c_fc; CUDA ladder FFMA → mma.sync+ldmatrix+cp.async (wgmma stretch); Muon symmetric X·Xᵀ (optional) | L0–L3, L5; L4 static only (Triton cross-compile; HIP port of the FFMA v1) | parity | 0.8–1.0× of cuBLAS; relu² epilogue may win 2–3 ms/step | Instruction-level study: HGMMA/TMA/warp specialization vs ours |
| **C** | Attention: FA3 vs SDPA vs Triton FA2-style forward; dissection of FA3's sm_90a cubins | L0, L2, L4-static, L5 | — | 0.5–0.8× of FA3 (negative result) | Beating FA3 is out of scope; explaining it is in scope |
| Decode | bf16 weight cache; `.item()` removal + CUDA graphs; fused small-op chains for T=1 | wrapper code | 3–10× tok/s at b=1–32 | n/a (infer_bench is eager) | Measured decode is launch-bound (GPU only 2.1–3.3× one CPU core at b=1 across S1 jobs) |

## 6. Session timeline (≈ 2 sessions/week, 3–5 h each; one STATUS.md update per session)

| Session | Date | Focus | Tasks (see TASKS.md) | GPU cap |
|---|---|---|---|---|
| S1 | Sat 10-17 | Env, data, baselines, nsys, CPU ref | done | 0.33 used |
| **S2** | Tue 10-20 | Access requests; timing+correctness harness; Inductor fusion facts (D1); noise band; clock probe | ADM-1..3, INF-1, INF-2, INF-3, BAS-1, BAS-2, BAS-3 | 1.5 |
| **S3** | Fri 10-23 | Triton RMSNorm + fused QK-norm+RoPE; static ISA pipeline v1; roofline v0 | KT-1, KT-2, AMD-1, ANA-1 | 1.0 |
| **S4** | Tue 10-27 | Fused softcap+CE v1; per-op baselines; CPU per-op references (≈ 0.35 SU CPU-only); ncu baseline → roofline v1 | KT-3, BAS-4, BAS-5, BAS-6, ANA-1 (v1) | 1.9 |
| **S5** | Fri 10-30 | Chunked lm_head+CE v2; Triton GEMM study; HIP RMSNorm via CCE | KT-4, KT-5, AMD-2 | 1.2 |
| **S6** | Tue 11-03 | Integration shim + graph-break check; first loss-curve gates; **D2 access decision** | INT-1, INT-2, INT-3, ADM-4 | 1.5 |
| **S7** | Fri 11-06 | CUDA RMSNorm + QK-norm/RoPE; FA3 cubin dissection; Triton attention fwd | KC-1, KC-2, KT-6, AMD-3 | 1.2 |
| **S8** | Tue 11-10 | **P1 data freeze**: one ABAB timing + nsys job, P1 figures | INT-4, ANA-2, ANA-3 | 0.5 |
| S9 | Fri 11-13 | P1 deck + rehearsal (no new kernels); **D4 scope freeze** | ANA-4 | 0.3 |
| **P1** | ~Tue 11-17 | Presentation 1 | | |
| S10 | Thu 11-19 | CUDA softcap-CE; relu² epilogue GEMM; ISA ablations; AMD-6 environment build if D2 = GO (Delta SU, not DeltaAI) | KC-3, KT-7, ANA-5, AMD-6 (env) | 1.5 |
| S11 | Tue 11-24 | Decode: bf16 cache, sync removal, CUDA graphs (short week) | DEC-1, DEC-2 | 1.0 |
| S12 | Tue 12-01 | HIP ports of RoPE/CE; AMD static table complete; cross-gen static; Delta measured runs if granted; **D5** per-kernel go/no-go | AMD-4, AMD-5, AMD-6, INT-5 | 1.0 (+Delta) |
| S13 | Fri 12-04 | CUDA GEMM ladder; tuning round from ncu findings; reconciliation; Muon symmetric (optional, only within the cap) | KC-4, TUNE-1, ANA-6, KT-8 (optional) | 1.8 |
| **S14** | Tue 12-08 | **Code freeze**; final measurement campaign (all numbers for P2, incl. independent-node repeats) | INT-6 | 4.5 |
| S15 | Fri 12-11 | P2 figures, ISA cards, deck draft | ANA-7, ANA-8 | 0.5 |
| S16 | Mon 12-14 | P2 rehearsal; fixes only | ANA-9 | 0.3 |
| **P2** | ~Thu 12-17 | Presentation 2 | | |
| S17 | 12-18→ | Final report, cleanup | ANA-10, ADM-6 | 0 |

Session caps sum to 19.7 SU (+0.33 used in S1 = 20.0); S1–S9 caps sum to 9.4 SU, inside the ≤ 10 SU pre-P1 limit. Expected use is roughly two-thirds of the caps (≈ 13 SU); project cap 25 SU, hard ceiling 40 SU. CPU-only jobs count too (72 cores bill the GPU rate). Unused cap does not roll forward; draws beyond a session cap are logged with a reason. Rules (same in ADM-5): **8 SU before P1 = re-plan checkpoint** (review the remaining S7–S9 caps with the user; GPU work may continue with a written reason); **10 SU before P1 = hard stop**; any single job > 1.1 SU → investigate before the next submission. Check `accounts`, `jobcharge` (authoritative) and the ledger (per-job cross-check) at every session start.

## 7. Experiment protocol (binding for every reported number)

### 7.1 Timing
- **Kernel microbenchmarks** (`cisc662/bench/kbench.py`): CUDA events around the op; 256 MB L2 flush between reps (L2 is 60 MiB; a (32768,768) bf16 activation is 50 MB); 25 warm-up reps after JIT/autotune; 100 timed reps; a GPU spin (`torch.cuda._sleep`, 2× the host enqueue time, ≥ 200 µs) is queued before every start event so the events measure device time only — without it host overhead (dynamo guards, AOTAutograd, autograd.grad) leaked into compiled fwd+bwd rows (job 3406282 vs 3406329); the host enqueue time is recorded per case (`host_bound`); **3 process-level trials** (5 in the final campaign). Headline = median of trial medians; spread = p10–p90. Speedup CI by bootstrap (10k resamples, stratified by trial). **"≥ 1.3×" is claimed only if the CI lower bound ≥ 1.3.** Achieved GB/s = algorithmic minimum bytes / time; ncu `dram__bytes` reported next to it to expose redundant traffic.
- **End-to-end**: R12 s/100 steps (steps 10–109). Variants interleaved **ABAB** inside one job on one GPU (fresh process per run); 3 reps (5 final). Report median and min–max; a difference counts only if it exceeds the larger variant's range. The wall-clock step contains ≈ 6 ms of host time (compile+FA3: 80.5 ms wall vs 74.3 ms summed kernel time — dataloader, optimizer Python, the per-step `.item()`), the same size as the per-kernel savings; so every end-to-end table also reports the nsys per-step GPU kernel time and GPU-busy fraction per variant — **GPU-time Δ is the attribution metric, s/100 steps the reported one** — plus one fixed-batch control run to bound the host share. Node name, SM clock, power and throttle reasons logged every 500 ms (`gpu_monitor.csv`); a run whose median busy clock deviates > 5% from the others is rerun.
- **Decode**: infer_bench TPOT/tok/s plus nsys GPU-busy time per token (kernel sum vs idle gaps); greedy tokens must match stock.
- **Profilers**: nsys for attribution only (steps 20–24 via `NANOCHAT_PROFILE_STEPS`, `--capture-range=cudaProfilerApi --capture-range-end=stop` as in `s1_nsys_train.sbatch`); ncu for counters, never for headline times (it locks base clocks; use `--clock-control none` when a time is shown). **ncu always runs inside the `NANOCHAT_PROFILE_STEPS` window with `--profile-from-start off`** — without it the cudaProfilerStart/Stop window is ignored and `--launch-skip/--launch-count` count from process start (step-0 compile/autotune launches). Filtered: `-k regex:… --launch-skip N --launch-count 1–3` for sections (the second in-window launch is the warm one); one whole step (`--launch-count ≈ 900`; a compiled step has ≈ 915 launches) only for the five-metric byte survey in BAS-1; `--set full` for ≤ 3 kernels per session; never `--set roofline` on a whole run.
- **Clocks**: default `GpuFreqDef=high`. BAS-3 tests `srun --gpu-freq`; if it works, microbenchmarks get a pinned-clock sensitivity repeat in the final campaign.

### 7.2 Little's-law check (used to explain memory-bound results)
Sustaining 3.65 TB/s at the measured **idle** DRAM latency of ≈ 344 ns (BAS-3, job 3406282: 1-thread pointer chase over 1 GiB; L2-resident 145–165 ns) needs ≈ 1.26 MB in flight ≈ 9.5 KB per SM ≈ 600 outstanding 16-byte loads per SM; loaded latency is higher (the S1 plan assumed ≈ 600 ns ⇒ ≈ 2.2 MB, 17 KB/SM, ≈ 1,050 loads), so read these as a lower bound and a likely value. A kernel at 25% occupancy with one 128-bit load in flight per thread cannot reach peak; it needs several independent loads issued before first use (ILP).

### 7.3 Correctness gates (all must pass before a variant is timed in any table)

| Gate | Test | Tolerance |
|---|---|---|
| **G1 forward** | Real shapes + edge shapes (N not a multiple of the block; rows with `ignore_index=-1`; T=1 decode shape; |logits| > 50 for softcap saturation; positions up to T−1 for RoPE) vs an **fp64 reference** on the same inputs | fp32 outputs: rtol 1e-5, atol 1e-6 (rtol 2e-5 for 32768-long reductions). bf16 outputs (**A1**): max-abs error ≤ 2× the stock bf16 op's error vs the same reference **and** cosine ≥ 0.9999 **and** the number of elements outside rtol 1.6e-2 / atol 1e-2 ≤ 2× the stock op's count (= the original strict rule whenever stock meets it; without a stock op the rtol/atol rule is strict). GEMM: rel-Frobenius ≤ 1e-2 and ≤ 1.25× cuBLAS's error. Attention (**A1**): max-abs ≤ 2× FA3's own error vs the fp32 math-backend SDPA reference (measured 9.1e-3 ⇒ bound 1.8e-2; the original absolute 4e-3 term is reported only). CE loss: rel 1e-5 vs fp64 on the same bf16 logits (fp32 accumulation meets this; `tanh.approx` does not, by ~100×) |
| **G2 backward** | `torch.autograd.gradcheck` in fp64 on tiny shapes + bf16 gradient comparison at real shape | as G1 on dX/dq/dk; dlogits (**A1**): rel-L2 ≤ 1e-2 and ≤ 2× the stock rel-L2; gradient cosine ≥ 0.9999; `ignore_index` rows exactly 0 |
| **G3 compile integration** | kernel registered via `torch.library.triton_op`/`custom_op` + `register_autograd`; R12 (depth 4 dev variant allowed) with `TORCH_LOGS=graph_breaks` | zero new graph breaks; our kernel visible in nsys and the replaced stock kernels gone |
| **G4 loss curve** | R12-L, same seed/data order: stock ×2 (noise envelope) vs each variant and the all-kernels variant | the logged train loss (nanochat's debiased EMA 0.9, used as-is) within max(2× envelope, 0.01) of stock at every step after 100; final val bpb |Δ| ≤ max(0.005, 2× stock–stock Δ); no NaN/Inf |

**Amendment A1 (2026-10-18, decided after GPU jobs 3406282/3406329; `tests/tolerances.py` implements it).** The original bf16 rule "rtol 1.6e-2, atol 1e-2 AND ≤ 2× stock" was violated by the *stock* ops themselves at R12 shapes (QK-norm+RoPE forward 4 / 25.2 M elements, its backward 37 k / 25.2 M, RoPE backward 3 / 25.2 M: bf16 rotation rounding amplified by the RMSNorm rstd on low-norm rows), and FA3's own error (9.1e-3) is above the attention 4e-3 bound, which lies below the bf16 output-rounding floor (7.8e-3 at |y| ≈ 4). A gate that the reference implementation fails cannot separate good kernels from bad ones, so A1 makes the stock op the yardstick where it cannot meet the absolute bound, and keeps the strict rule wherever it can. The fp32, GEMM, CE-loss and G3/G4 rules are unchanged. Rationale and data: `docs/findings.md#tolerance-a1`.

**A1 addendum — scalar parameter gradients of fused kernels (dr, dl of mixnorm; 2026-10-18).** dr and dl are single scalars formed by heavy cancellation over up to 25 M terms (RMSNorm is scale-invariant), so an elementwise rule on |ref| fails correct kernels whenever the sum lands near 0, and a single-scalar "≤ 2× stock" check fails a correct kernel ≈ 7% of the time per draw. The binding rule for such scalars: |ours − ref| ≤ atol + rtol·max(|ref|, ‖summands‖₂) with the fp32 constants, where ref is fp64 autograd evaluated on the stored (bf16) x_new, **and** |ours − exact| ≤ 2 × max(stock's error, the error of the exact function with x_new correctly rounded to bf16). x_new itself is gated by G1, and a dedicated test checks it is rounded once to nearest-even. Simulated false-failure rate 0/560 draws; a real single-ulp recompute mismatch is still caught (1.3–1.5× over the limit). Implemented in `tests/test_rmsnorm.py`.

The results database (`results/db/*.json`) carries `gates_passed`; plotting scripts refuse rows missing G1–G2 (and G3–G4 for integrated/end-to-end rows). A failing variant is still written up — as a goal (iii) failure.

### 7.4 Roofline
Ceilings: HBM 3.65 TB/s measured (4.0 dashed), bf16 tensor 723 TF/s measured (990 dashed), fp32 48.6 (67 dashed). Points: FLOPs analytic from the inventory; bytes from ncu `dram__bytes_read/write`; time from ncu `--clock-control none`. AMD: analytic roofline from datasheets (MI100 92.3 TF bf16 / 1.23 TB/s; MI210 181 / 1.64; MI250X per GCD 191.5 / 1.64; MI300A 980.6 / 5.3), labelled STATIC; rocprof-compute `--roof-only` if hardware arrives.

### 7.5 CPU reference (goal i)
Full-step numbers exist (S1). Per-op: CPU-only ghx4 job (2 cores, 16 GB, ≈ 0.14 SU/h), `numactl --physcpubind=0 --membind=0`, `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`, each op at reduced N scaled by bytes/FLOPs; one 72-core run (≤ 15 min). Report raw speedup, speedup normalized by peak ratios (BF16 tensor peak / one-core fp32 peak measured with an FMA loop; HBM / measured single-core STREAM), and GPU vs 72 cores.

## 8. Decision gates

| Gate | When | Question | GO | NO-GO / fallback |
|---|---|---|---|---|
| D1 | S2 | Which chains does Inductor already fuse, and with how many bytes? | Sets the per-kernel vs-compile targets in TASKS.md | — |
| D2 | S6 (11-03) | NCSA Delta MI100/MI210 (and A100) access granted? DARWIN V100? | Delta env in S10, measured AMD rows in S12 | AMD = STATIC, announced in P1; late access accepted until 12-01 |
| D3 | S5 | Standalone softcap+CE ≥ 1.3× vs compile tail? | Headline kernel = chunked lm_head+CE | Headline shifts to time+memory; failure analysis |
| D4 | S9 | Scope freeze for P1 and the post-P1 kernel list | — | Cut order: Muon symmetric → CUDA GEMM v3 → Triton attention backward → decode CUDA graphs → CUDA QK-norm/RoPE |
| D5 | S12 | Per integrated kernel: G3+G4 passed and in-model speedup ≥ 1.0? | included in end-to-end numbers | excluded, kept as a finding |
| D6 | S14 (12-08) | Code freeze (`git tag p2-freeze`) | final campaign | no features after this |

## 9. Risk register (top items; full mitigations in the research files)

| # | Risk | L/I | Mitigation |
|---|---|---|---|
| R1 | No AMD hardware all semester | High/High | Static AMDGCN tier is built by S3 regardless; requests on 4 routes in S2 (Delta exchange, Chameleon, DARWIN, AMD Developer Cloud); P1 states the scoping |
| R2 | No V100/A100 | High/Med | static sm_70/80/90a comparison; GH200 = H100-class covers "newer tensor cores" |
| R3 | Kernels beat eager but not compile | Med-High/High | targets chosen from D1 facts; parity reported as goal (iii) with ISA evidence; torch 2.9.1 pinned and disclosed |
| R4 | Custom ops break fullgraph / block Inductor fusion | Med/Med | `triton_op` + fake impls; G3; isolated vs in-model numbers reported separately |
| R5 | Numerical drift fails G4 | Med/High | fp64 references, envelope from same-seed reruns, precise-tanh fallback |
| R6 | Queue delays near deadlines; shared pool drained | High/Med | ≤ 30-min jobs; P1 data frozen 11-10, P2 data 12-08; stop rule if pool < 1,500 SU |
| R7 | Silent FA3→SDPA fallback | Low/High | offline cache + `HAS_FA3` assertion in every FA3 job |
| R8 | /work/nvme shared quota filled | Med/High | < 50 GB; summaries copied to git after each job |
| R9 | Hand-written Hopper GEMM/attention becomes a sink | High/Med | Tier B/C scoped as studies; wgmma is a stretch with a one-session timebox |
| R10 | Compiler-version confounds (Triton LLVM 22 vs CCE LLVM 20.1.6 vs nvcc 12.9) | Certain/Low | every ISA row records its compiler; same-source CCE CUDA vs HIP as the single-LLVM control |

## 10. Presentation evidence

**P1 (~11-17, 12–15 slides; emphasis on what/why/how, ~25% results).** *Required (all exist after S1–S4):* platform-reality table (proposal vs actual: GPU, CPU host ISA, V100/A100/AMD status, static AMD tier, multi-GPU dropped, MQA→MHA correction, toolchain provenance); nanochat block diagram with the 5 kernel classes at their call sites; R12/D1 definitions; method ladder figure (eager→compile→Triton→CUDA→HIP→SASS/AMDGCN→nsys/ncu/rocprof) with the correctness gate first; protocol summary; baseline table (§4); stacked kernel-class bars eager vs compile (exists: `results/figures/s1_kernel_breakdown.png`); roofline v1 (ncu bytes from BAS-6; the analytic v0 is the fallback if BAS-6 slips); goal (i) preview (CPU 1-core / 72-core / GPU, train and decode — exists: `s1_cpu_vs_gpu.png`, `s1_decode.png`); **one** preliminary kernel row (microbench vs eager/compile with G1/G2 shown, labelled preliminary); timeline, budget used, risks. *If available by the S8 freeze (D4 decides; P1 does not depend on them):* the three-kernel results table, the loss-curve overlay for the first integrated variant, the ISA side-by-side (RMSNorm sm_90a SASS vs gfx90a AMDGCN: load width, reduction primitive, registers/occupancy, labelled STATIC), the matmul codegen table (HGMMA vs v_mfma_32x32x8bf16_1k vs 32x32x4bf16 with AGPRs).

**P2 (~12-17, results):** end-to-end s/100 steps with CIs (eager / compile / compile+each kernel / compile+all); loss curves with envelope and val bpb; per-kernel speedup bars vs eager and vs compile with 1.3×/2× lines; final hierarchical roofline with before→after arrows; lm_head+CE headline (time, peak memory, batch enabled); decode table (stock / +bf16 cache / +graphs / +kernels) with nsys timeline before/after; goal (i) table (raw, peak-normalized, bandwidth-normalized, vs 72 cores); instruction-level table per variant (regs, spills, occupancy, sectors/request, load width, MMA/MFMA type, top-2 stalls, DRAM %); annotated SASS excerpts; "what did not work and why" (GEMM vs cuBLAS, Triton attention vs FA3, any gate failure); NVPTX vs AMDGCN table labelled STATIC/MEASURED; cross-generation table (static or measured); proposal-vs-delivered checklist; budget used.

## 11. Repository layout
```
cisc662/
  env/        env.sh (source in every shell/job), setup_env.sh
  slurm/      _common.sh (ledger with SU column, offline HF, run_step), sNN_<purpose>.sbatch per session/job type, monitor.sh (bounded)
  bench/      smoke.py, decode_bench.py, kbench.py (timer+L2 flush), ops_at_shapes.py, cpu_ref.py, cpu_peak.c, train_wrap.py, decode_opt.py
  integrate.py  run-time monkeypatch shim (CISC662_KERNELS, CISC662_BACKEND); optim_shim.py (KT-8); decode/engine_step.py (DEC-2)
  kernels/    triton/<kernel_id>.py, cuda/<kernel_id>.cu, hip/<kernel_id>.hip (+ mini_hip.h); kernel_id ∈ rmsnorm, qkrope, ce, ce_chunked, gemm, mlp, attention, muon
  tests/      pytest gates G1–G2 (markers real_shape, gpu); ref_ops.py (fp64/fp32 references); tolerances.py; gates.py
  profiling/  nsys_attrib.py, classify_kernels.py (both know the c662_* kernel names)
  analysis/   summarize_s1.py, make_figs_s1.py, losscurve.py, stats.py, kernel_registry.py, xcompile.py, isa_report.py, roofline.py, make_figs_amd.py, make_figs_p1.py, make_figs_p2.py
  results/    runs/<tag>_<jobid>/ (logs+summary), db/<kernel_id|group>/*.json (schema in db/README.md, gates_passed), isa/<kernel_id>/<source>/<target>/, figures/, p1_freeze/, final/, static_isa.csv, crossgen_static.csv, inductor_fusion_table.csv
  docs/       PLAN.md, TASKS.md, AGENT_GUIDE.md, STATUS.md, DIFF_FROM_UPSTREAM.md, findings.md (ISA cards, tuning log), research/*.json
  presentations/ p1/ (deck, figs/, QA.md), p2/
```
