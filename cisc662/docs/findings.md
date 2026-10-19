# Findings — measured facts, ISA cards, tuning log

Every entry names the job that produced it. Raw data lives under `results/`; this file holds the conclusions.

<a id="inductor-baseline"></a>
## Inductor baseline: what torch.compile already fuses in R12 (BAS-1, decision D1) — job 3406171, 2026-10-18

**Method.** R12, compile+FA3, torch 2.9.1 / Triton 3.5.1. (1) `TORCH_COMPILE_DEBUG=1` with caches disabled → 11 Inductor graphs (1 fwd, 1 bwd, 5 AdamW, 4 Muon), `results/runs/s2_inductor_3406171/output_code/`. (2) ncu inside the profiler window (`NANOCHAT_PROFILE_STEPS=20:21 --profile-from-start off --nvtx`), all 915 launches of step 20, metrics time + DRAM read/write + regs + warps active, `--clock-control none`, default `--cache-control all` (⇒ cold-L2 traffic per kernel). Analysis: `python -m cisc662.analysis.inductor_fusion results/runs/s2_inductor_3406171` → `results/inductor_fusion_table.csv`, `.../inductor_fusion_summary.md` (formulas for every minimum).
**Caveats.** ncu serialises kernels and flushes L2, so its per-step kernel sum (65.8 ms) is 11% below the S1 nsys sum (74.3 ms); ncu times are for bandwidth facts only, never headline times. Minimum bytes = a fused kernel at the stock GEMM/FA3 boundaries reading each input once and writing each output once.

| Region (per step) | Inductor kernels | ncu ms | DRAM GB | min GB | ratio | effective TB/s on min bytes | vs-compile target |
|---|---|---|---|---|---|---|---|
| λ-mix + RMSNorm fwd | 25 | 1.71 | 5.51 | 5.54 | 1.00 | 3.23 (89%) | parity 0.84–1.02× |
| λ-mix + RMSNorm bwd | 28 | 3.16 | 10.14 | 8.56 | 1.18 | 2.71 | 1.0–1.2× |
| QK-norm+RoPE+scale fwd | 1 (×12) | 1.21 | 3.31 | 2.44 | 1.36 | 2.02 | **1.3–1.6×** |
| QK-norm+RoPE+scale bwd | 3 (×12, 60 launches) | 2.31 | 7.62 | 3.65 | **2.09** | 1.58 | **1.7–2.1×** |
| relu² fwd / bwd | 1 / 1 | 1.29 / 1.90 | 4.51 / 6.94 | 4.83 / 7.25 | ≈ 1 | 3.7–3.8 (≈ roof) | parity; only GEMM-epilogue fusion (KT-7) can win, −2.4 / −4.8 GB |
| softcap+CE fwd | 1 (+2 tiny) | 1.94 | 2.15 | 2.15 | 1.00 | **1.11 (30%)** | **2.4–3.0×** |
| softcap+CE bwd | 1 | 1.88 | 4.27 | 4.30 | 0.99 | 2.29 (62%) | 1.2–1.4× |
| softcap+CE single pass (KT-3 form i) | — | 3.82 | 6.42 | 4.30 | 1.49 | 1.12 | **2.4–2.9×** |
| AdamW | 2 (13 launches) | 0.86 | 2.94 | 3.17 | 0.93 | ≈ roof | parity |
| Muon non-GEMM (pre/post + NS pointwise + stack) | 31 | 3.05 | 9.74 | 2.04 | **4.78** | 0.67 | 4–5× (optional KT-8 is stronger than planned) |

**Conclusions (D1).**
1. **Headline kernel confirmed: softcap+CE.** Inductor's CE forward already moves only the minimum bytes but runs at 30% of the bandwidth roof (an online-softmax reduction over 32768-wide rows with `tanh` in the loop). The win is achieved bandwidth plus the single pass, as the plan predicted — target 2.4–3.0× on the forward, 1.6–2.9× on the tail.
2. **QK-norm+RoPE is the second vs-compile candidate**, and the headroom is redundant bytes in the backward (2.09× minimum: separate norm-backward kernels and a zero-filled `slice_backward` scatter for the rotated halves), not bandwidth.
3. **RMSNorm, relu², AdamW: Inductor is at the roof** at the minimum bytes. vs-compile results there are parity explained at the ISA level (goal iii); the only byte savings sit across the cuBLAS boundary (KT-7 epilogues).
4. **Muon's pointwise work moves 4.8× its minimum** (3.05 ms/step, close to the 3.82 ms CE tail). KT-8 stays optional (scope), but D4 should reconsider its rank in the cut order.
5. GEMM regions run at 13–17% achieved occupancy as expected for tensor-core kernels; one small GEMM family uses an **sm_80 WMMA CUTLASS kernel** (`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_32x1`) and cuBLASLt split-K reduce kernels appear (0.88 ms/step) — material for the KT-5 GEMM study.

<a id="loss-noise-envelope"></a>
## Loss-curve noise envelope (BAS-2, gate G4) — job 3406173, 2026-10-18

Two stock R12-L runs (1000 steps, same seed and data order): max |Δ loss| over steps > 100 = **0.00128** (step 437); final val bpb 1.148793 vs 1.148788 (Δ 5e-6). G4 therefore runs on its floors: loss within **0.01**, val bpb within **0.005** (2× the measured noise is far below both). `results/db/loss/noise_envelope.json`, figure `results/figures/loss_noise_band.png`. Gate a variant with `python -m cisc662.analysis.losscurve stock.log variant.log --envelope cisc662/results/db/loss/noise_envelope.json`.

<a id="platform-s2"></a>
## Platform facts measured in S2 (BAS-3) — job 3406282, node gh147, 2026-10-18

- **Power cap, not thermal:** a 60-s back-to-back 8192³ bf16 GEMM loop settles at **1365 MHz (1320–1380), ≈ 545 W, `clocks_event_reasons = sw_power_cap`, 591 TF/s**; short bursts reach 723 TF/s at 1980 MHz. GH200's 900 W budget is shared by the module (Grace + HBM + GPU). Training (S1) ran at a median 1665 MHz / 524 W. ⇒ never compare numbers from different jobs/nodes without the logged clock; interleave variants (ABAB) on one GPU.
- **`srun --gpu-freq=1755` (job step) did not pin the clock**: 0/40 busy samples in 1755 ± 15 MHz (all at the power-capped 1305–1380). Clock policy: default clocks + `gpu_monitor.csv`; no pinned-clock repeats planned.
- **DRAM latency** (1 thread, dependent int64 loads, random single cycle): **344 ns at 1 GiB**, 327 ns at 256 MiB, 165 ns at 32 MiB, 145 ns at 4 MiB / 64 KiB (L2). The card's 500–800 ns "sanity range" was an S1 assumption; PLAN §7.2 now uses 344 ns (idle, lower bound).
- **Decode b=1** (r12_conv, prompt 512, 256 tokens): CPU 1 thread (numactl core 0, fp32) 33.0 / 33.6 / 33.6 tok/s; GPU 85.5 / 104.4 / 104.2 tok/s (first rep slower: warm-up) ⇒ GPU/CPU-1-thread **3.1×** (pairwise 2.5–3.2). Matches S1's 32.8 tok/s CPU and the top of S1's 68–107 GPU range. `results/db/decode/s2_remeasure.json`.
- **FA3 numerical error** vs the fp32 math-backend SDPA reference at (16,2048,6,128) causal: max-abs **9.1e-3** (bf16 output rounding floor 7.8e-3 at |y| ≈ 4); bf16 SDPA has the same max-abs; FA3 vs bf16 SDPA agree to 7.8e-3. `results/db/tolerances/attention_stock_error.json`.

<a id="kbench-host-overhead"></a>
## Measurement pitfall found in S2: host overhead inside CUDA-event timings — jobs 3406282 → 3406329

The first ops baseline (3406282) showed compiled ops slower than eager (RMSNorm fwd+bwd 349 vs 122 µs; GEMM 32768×768×768 109 vs 62 µs; trial cv up to 0.37) and a warm-L2 RMSNorm slower than the cold one. Cause: when the host needs longer to enqueue a call (dynamo guards, AOTAutograd wrappers, `autograd.grad`) than the GPU needs to run it, the GPU idles between the start event and the kernel, so the event pair measures host time; in cold-L2 reps the 256 MB flush kernel happened to hide part of it. Fix (kbench): queue a `torch.cuda._sleep` spin of 2× the measured host enqueue time before every start event. Check after the fix (3406329): compiled GEMMs 0.99–1.01× eager, compiled RMSNorm fwd+bwd = eager (94.4 µs), trial cv ≤ 0.5% for 46/54 cases. The flush proof now uses an L2-resident (8192,768) case (warm 1.35× faster than cold). Many compiled small ops are `host_bound` (host enqueue > device time) — in-model they are hidden only because the training loop keeps the GPU queue full; at decode (T=1) they are not.

<a id="tolerance-a1"></a>
## Decision: correctness-gate amendment A1 (PLAN §7.3) — 2026-10-18, from jobs 3406282 / 3406329

**Observed (GPU, R12 shapes, fp64 references):** the stock bf16 ops miss the original absolute bf16 rule — `qkrope` forward 4 / 25,165,824 elements (max-abs 5.1e-2, cosine ≥ 0.999994), `qkrope` backward 37,285 / 25.2 M (max-abs 0.111), `rope` backward 3 / 25.2 M; small/tail shapes similar. Cause: nanochat rotates in bf16 and then normalizes; rows with a small norm get a large rstd that amplifies the rotation's rounding into the 1e-2 absolute band. FA3 measures max-abs 9.1e-3 against the fp32 SDPA reference (bf16 SDPA identical), above the 4e-3 attention bound, which itself sits below the bf16 output-rounding floor (7.8e-3 where |y| ≈ 4).
**Decision:** the stock op is the yardstick where the absolute bound is unachievable for it, and the strict rule stays wherever stock meets it. bf16: max-abs ≤ 2× stock, cosine ≥ 0.9999, rtol/atol violations ≤ 2× stock's count. Attention: ≤ 2× FA3 (bound 1.8e-2). dlogits: rel-L2 ≤ 1e-2 and ≤ 2× stock (the elementwise atol is vacuous at |dlogits| ~ 1e-6).
**Why this is reasonable:** a gate the reference implementation fails cannot tell good kernels from bad; "no worse than what PyTorch ships" is the standard the proposal implies (numerical agreement with the stock model, then the loss-curve gate G4, which is unchanged and catches any accumulated drift). Our fused kernels compute in fp32 and round once, so they are expected to beat stock on these metrics — the ratio is reported in every result JSON (`stock_ratio`).
**Alternatives rejected:** loosening atol globally (would hide real bugs at the scale of typical activations); fp32 reference for stock-vs-ours (does not address that stock itself is the baseline); keeping the xfails (hides real failures once kernels exist).

<a id="roofline-v0"></a>
## Roofline v0 (ANA-1) — from jobs 3406329 (isolated ops) and 3406171 (in-model ncu), 2026-10-18

`python -m cisc662.analysis.roofline --version v0` → `results/figures/roofline_v0.png`, points in `results/roofline_points_v0.csv` (77 rows). GEMMs (isolated 516–684 TF/s; in-model 665–673 TF/s at ncu's burst clocks, above the 591 TF/s power-capped sustained line), FA3 (fwd 530–543, bwd 432 TF/s) and Muon's Newton-Schulz bmm (611 TF/s) sit on the compute side; RMSNorm, λ-mix+norm, relu², QK-norm+RoPE and softcap+CE on the memory side. Compiled ops reach the bandwidth roof except softcap+CE (31%) and QK-norm+RoPE. Caveats: lm_head+CE is plotted with "logits never stored" bytes (the KT-4 target, not where stock runs); memory-bound FLOPs are approximate op counts, so only the gap below the bandwidth roof is meaningful.
**RMSNorm(768) eager = compile is real, not a harness bug:** different per-rep times, 1 compiled graph per trial, medians equal at the 32-ns event resolution — Inductor lowers a standalone `F.rms_norm` to the same `aten::_fused_rms_norm` CUDA kernel eager uses. vs compile, KT-1's standalone RMSNorm therefore competes with the eager fused kernel; the λ-mix+norm (mixnorm) chain is where Inductor generates its own code.

<a id="static-isa"></a>
## Static cross-ISA pipeline and first side-by-side (AMD-1): login node, no GPU, 2026-10-18 — all values STATIC

**Method.** `python -m cisc662.analysis.xcompile --set base|sweep` (batches of ≤ 100 compiles, `nice -n 19`, 3–32 CPU-s per process) compiles every kernel in `analysis/kernel_registry.py` (rmsnorm: 9 variants from KT-1's own `isa_specs("meta")`/`_signature()`; qkrope: v1/ilp2/ilp4 fwd+bwd from KT-2's `VARIANTS`; the Triton tutorial matmul 128×128×64 bf16 at M=32768, K=768, N=3072, num_warps=8, num_stages=3) for `cuda:90` (sm_90a), `hip:gfx908`, `hip:gfx90a` and `hip:gfx942` with per-target `TRITON_CACHE_DIR`. Dumps go to `results/isa/<kernel_id>/triton_<variant>/<isa>/xc-<config>/`. Then `python -m cisc662.analysis.isa_report` (21 CPU-s) writes `results/static_isa.csv`: 358 rows (64 base rows = 16 kernels × 4 targets, plus 294 tuning-pair rows), no failed compiles and no empty cells. Toolchains (R10, one column per row): Triton 3.5.1 with its bundled LLVM 22.0.0git (7d5de30) for NVPTX and AMDGPU; Triton's bundled ptxas V12.8.93; cuobjdump V12.9.26 from HPC SDK 25.5; llvm-readelf/llvm-objdump from Cray LLVM 20.0.0 (LLVM 20.1.6). Rows carry the kernel-source sha (rmsnorm `d44a5a1d0c02`, qkrope `326ad3badc55`), because both files are still under review.
**AMD occupancy is recomputed** from the LLVM AMDGPU tables. gfx908: min(10, 256 / max(align4 VGPR, align4 AGPR)). gfx90a/942: min(8, 512 / align8(align4 VGPR + AGPR)). The SGPR limit is then applied, and only whole workgroups fit a CU: 4 SIMDs × waves/SIMD ÷ num_warps. Triton's dynamic LDS (`metadata["shared"]`; `.group_segment_fixed_size` is 0 in all 303 AMD rows) is checked against 64 KiB/CU, and there is a 16-workgroup-per-CU limit. Check: the register-only limit equals the compiler's printed `Occupancy:` line in **all 303 AMD rows**. MFMA counts from `llvm-objdump -d` of the HSACO match the `.amdgcn` (96/48/48). NVIDIA occupancy uses the sm_90 limits: 64 K registers per SM in 256-register warp granules, 228 KiB smem per SM with 1 KiB reserved per CTA, 32 CTAs and 64 warps per SM.

**RMSNorm forward, D = 768, sm_90a vs gfx90a (the GH200 launch config; same Triton source)**

| variant · hints | target | regs | occupancy (recomputed) | global loads per thread/lane | stores | reduction primitive | bf16 output conversion |
|---|---|---|---|---|---|---|---|
| v1 (BLOCK_D=1024, 4 warps) · on | sm_90a | 21 | 16/16 warps/SMSP | `LDG.E.128` ×1 (8 bf16) | `STG.E.128` ×1 | `SHFL.BFLY` ×7 + STS/BAR/LDS | `F2FP.BF16.F32.PACK_AB` ×4 (2 elements each) |
| v1 · on | gfx90a | 11 VGPR, 22 SGPR | 8/8 waves/SIMD | `global_load_dwordx2` ×1 (4 bf16) | `global_store_dwordx2` ×1 | DPP ×8: `v_add_f32_dpp` ×5 (row_shr 8/4/2/1, row_bcast:31) + `v_mov_b32_dpp` ×3 (row_bcast:15, 2× quad_perm) + `v_readlane_b32` + LDS/s_barrier | emulated ×4 |
| v1 · off | sm_90a / gfx90a | 18 / 11 | 16 / 8 | `LDG.E.U16` ×6 / `global_load_ushort` ×3 | 16-bit | unchanged | unchanged |
| v2 (BLOCK_M=8, 4 warps) · on | sm_90a | 64 | 8/16 (50%) | `LDG.E.128` ×6 | `STG.E.128` ×6 | `SHFL.BFLY` ×10 | `F2FP` ×24 for 48 elements |
| v2 · on | gfx90a | 38 VGPR, 21 SGPR | 8/8 | `global_load_dwordx4` ×3 (24 bf16 per lane) | `dwordx4` ×3 | DPP ×6 + `ds_swizzle_b32` ×1 | emulated ×24 |
| v2 · off | sm_90a | 72 | 7/16 | `LDG.E.U16` ×48 | `STG.E.U16` ×48 | `SHFL.BFLY` ×42 | `F2FP` ×24 |
| v2 · off | gfx90a | 46 VGPR, 40 SGPR | 8/8 | `global_load_ushort` ×24 | `short` ×24 | DPP ×50 | emulated ×24 |

- **Load width.** With the divisibility hints (16-byte-aligned pointers, row stride a multiple of 16, `ALIGNED=True` → `tl.multiple_of`), both vendors vectorise to 16 B per lane wherever a lane holds ≥ 8 contiguous bf16. Without them, both fall back to scalar 16-bit loads: NVIDIA `LDG.E.U16`, AMD `global_load_ushort`. This holds for every Triton kernel here (`div_hints=off` rows). Wave64 halves the elements per lane at equal `num_warps`. The GH200 configs therefore give AMD 8-byte loads (`dwordx2`) for rmsnorm v1, bwd_768 and mixnorm_fwd/bwd and for qkrope v1. They give 16-byte loads only where a lane still gets ≥ 8 elements: rmsnorm v2_768, v2_128 and bwd_128, and qkrope ilp2/ilp4. An AMD-tuned launch would double BLOCK_M (or halve num_warps), not copy the NVIDIA config.
- **Reduction primitive.** sm_90a uses `shfl.sync.bfly` → `SHFL.BFLY` (5 butterfly steps per warp), then a shared-memory exchange across warps. On gfx90a and gfx942, Triton 3.5.1 lowers the wave64 reduction to DPP, with the add fused into the data movement (`v_add_f32_dpp row_shr:8/4/2/1`, then `row_bcast:15/31`, then `v_readlane_b32 …, 63`). On **gfx908**, the same source gets **no DPP**: `ds_swizzle_b32` ×7 (SWAP 16/8/4/2/1) + `ds_bpermute_b32` ×1. These go through the LDS crossbar and need `lgkmcnt` waits: `s_waitcnt` 16 vs 6 on gfx90a for v1_768. With hints off, the scalar-load layout multiplies the cross-lane work on every target (v2_768: SHFL 10→42, DPP 6→50, gfx908 `ds_swizzle` 5→42).
- **bf16 conversion is emulated on all three gfx9 targets** (no `v_cvt_pk_bf16_f32` before gfx950; 0 in all 303 rows). Each fp32→bf16 element costs `v_bfe_u32` (bit 16) + `v_add3_u32` (+0x7fff + lsb, i.e. round-to-nearest-even) + `v_cmp_o_f32` + `v_cndmask_b32` (NaN), plus one `v_perm_b32` per pair. For v2_768 that is 24+24+24+24+12 = 108 VALU per lane for 24 elements. sm_90a needs 24 `F2FP.BF16.F32.PACK_AB` for 48 elements. The conversion is about 9× more instructions per element on gfx9: 4.5 per element, against 0.5 on sm_90a.
- **Division also differs.** The same `sum / D` is PTX `div.full.f32` (approximate; no `MUFU.RCP` remains in the SASS) but a correctly rounded division on AMD (`v_div_scale_f32` ×2, `v_rcp_f32`, `v_div_fmas_f32`, `v_div_fixup_f32`). This is a numerics difference between backends to keep in mind for AMD-side gates.
- **Registers and occupancy.** No RMSNorm or QK-RoPE base config spills on any target (AMD ≤ 112 VGPR, sm_90a ≤ 167 registers). gfx90a/gfx942 run at full occupancy (8 waves/SIMD) except for mixnorm_bwd (71 VGPR → 7) and qkrope ilp4 bwd (105 → 4). gfx908's 10-wave table loses occupancy earlier: v2_768 has 35 VGPR → 7/10, mixnorm_bwd 71 → 3/10, qkrope ilp4 bwd 112 → 2/10. sm_90a v2_768 uses 64 registers → 8/16 warps per SMSP (50%).

**Matmul num_stages=3: recomputed occupancy differs from the compiler's line (acceptance item)**

| target | VGPR / AGPR | LDS (Triton `shared`) | compiler `Occupancy:` | recomputed waves/SIMD | binding limits | MFMA | MFMA cycles per CU per 128×128×64 step |
|---|---|---|---|---|---|---|---|
| gfx908 | 111 / 32 | 65536 | 2 | 2 | whole-WG VGPR = LDS (1 WG/CU) | `v_mfma_f32_32x32x4bf16` ×96 (+224 `v_accvgpr_read/write`) | 4096 |
| gfx90a | 141 / 0 | 65536 | **3** | **2** | whole-WG VGPR = LDS (1 WG/CU) | `v_mfma_f32_32x32x8bf16_1k` ×48 | 2048 |
| gfx942 | 141 / 0 | 65536 | **3** | **2** | whole-WG VGPR = LDS (1 WG/CU) | `v_mfma_f32_32x32x8_bf16` ×48 | 1024 |
| gfx90a, waves_per_eu=4 | 128 / 0, 64 B scratch | 65536 | **4** | **2** | LDS alone (1 WG/CU) | same | 2048 |
| gfx90a, num_stages=2, waves_per_eu=4 | 128 / 0, 60 B scratch | 32768 | 4 | 4 | VGPR (2 WG/CU) | same | 2048 |
| sm_90a | 126 regs | 98304 + 1 KiB | n/a | 4/16 warps/SMSP | registers = smem (2 CTA/SM) | `HGMMA.64` ×4, `LDGSTS.E.BYPASS.128` ×24 | n/a |

The compiler's occupancy line cannot see Triton's 64 KiB of dynamic LDS. It also cannot see the exact 512-lane workgroup: the kernel only carries `amdgpu-flat-work-group-size=1,512`. Three waves per SIMD give a CU 12 wave slots, which is fewer than the 16 waves of two 8-wave workgroups, so one workgroup per CU is the real limit. On gfx90a and gfx942 the line says 3 where 2 is achievable. With `waves_per_eu=4` the line says 4, but at num_stages=3 LDS still allows only one workgroup: the VGPR cap buys spills (64 B gfx90a, 24 B gfx942, 68 B gfx908) and no occupancy. Only at num_stages=2 does `waves_per_eu=4` reach 4 waves/SIMD, and it still spills (60/48/120 B). That trade only hardware can settle. The static MFMA count (48) includes the pipeline prologue/epilogue copies. The cycle column is analytic: (2·128·128·64 / FLOPs per MFMA) × cycles ÷ 4 SIMDs, with cycles from the AMD matrix-instruction calculator (64 / 64 / 32 cycles). gfx942's in-thread transpose (Triton enables it only for gfx942) replaces gfx90a's 96 `ds_read_u16` with 24 `ds_read_b64`.

**Static tuning pairs (294 rows, one-line `verdict` each in the CSV).**
- `waves_per_eu` 2/4 changes nothing on the memory-bound kernels: 85 of 90 rmsnorm/qkrope rows report "no static change", because their ≤ 108 VGPR already sit under the 128-VGPR cap that `waves_per_eu=4` implies on gfx90a. It matters only for the matmul: 2 fits (cap 256 > 141), 4 spills.
- `num_warps` 4→8 halves the bytes per lane again (rmsnorm v2_768 on gfx90a: `dwordx4`→`dwordx2`, VGPR 38→22, occupancy already 8). Occupancy rises only where registers were the limit (gfx908 v2_768: 7→10; sm_90a v2_768: 8→16 warps/SMSP). For the matmul, 8→4 warps doubles the per-wave accumulator: VGPR 141→170 plus 64 AGPR on gfx90a, occupancy 2→1.
- Hints off on the matmul drops multi-stage pipelining on every target: LDS 64→16 KiB on gfx90a, MFMA copies 48→16, smem 96→32 KiB on sm_90a. Loads become 16-bit (gfx90a: 32 `global_load_ushort`; sm_90a: 64 `LDG.E.U16`), and sm_90a registers double (126→254).
- The only spills outside the matmul are in qkrope ilp4 backward with hints off on sm_90a: 255 registers, 816 B of spill traffic.

**What this does not show.** These are static facts only: no times, no achieved occupancy (grid vs CU count), no cache behaviour. Instruction counts are static code counts: both q and k paths, pipeline copies. The sm_90a rows come from a cross-compile, not from KT-1/KT-2's GPU `--dump-isa`. The `xc-` directory prefix keeps the two apart, and the first GPU dump should be diffed against them. The whole-workgroup occupancy model assumes a workgroup is resident whole on one CU, as on CDNA. Hardware (AMD-6) is needed to rank any tuning pair.

<a id="kt-review-s3"></a>
## KT-1 / KT-2 pre-GPU review (2026-10-18) — defects caught before spending GPU time

1. `tl.multiple_of(stride, 16)` on a scalar argument broke torch's triton_op mutation analysis (it re-traces non-tensor args as constexprs), so every input looked mutated and AOTAutograd's partitioner crashed → hint moved onto the row offset (`tl.multiple_of(rows*stride, 16)`); SASS unchanged (same LDG.E.128 counts and registers).
2. mixnorm's backward recomputed `x_new = r·x + l·x0` with a different FMA contraction than the forward (LLVM chose `fma(l,b,r·a)` for 12/24 elements) → ≈ 2e-4 of recomputed x_new values off by 1 bf16 ulp → dr/dl off by up to 3.5e-3 relative; invisible under the Triton interpreter (no FMA contraction) → explicit `tl.fma(r, a, l*b)` in both kernels. Lesson for every fused fwd/bwd pair that recomputes: pin the contraction explicitly.
3. Gate design for scalar gradients (PLAN §7.3 A1 addendum).

<a id="rmsnorm"></a>
## ISA card: `rmsnorm` (KT-1, Triton) — job 3409675 (gh090), 2026-10-18; gates G1+G2 passed (188 tests incl. test_ref_ops on GPU)

kbench, cold L2, queue pad, 3 trials × 100 reps, default clocks; speedup = median-of-trial-medians ratio, 95% bootstrap CI; torch 2.9.1 / Triton 3.5.1. Results: `results/db/rmsnorm/*.json`; SASS/PTX/meta: `results/isa/rmsnorm/triton_<variant>/sm_90a/`; static AMDGCN rows: `results/static_isa.csv`.

| variant (shape) | µs | TB/s (min bytes) | vs eager [CI] | vs compile [CI] | regs / spills | grid · warps | notes |
|---|---|---|---|---|---|---|---|
| v1 fwd (32768,768) | 34.1 | 2.95 | 1.25 [1.24,1.26] | 1.26 [1.25,1.26] | 21 / 0 | 32768 · 4 | 1 row/program, BLOCK 1024 → 25% lanes masked |
| **v2 fwd (32768,768)** | **32.6** | **3.09 (85%)** | **1.31 [1.30,1.31]** | **1.31 [1.31,1.32]** | 64 / 0 | 4096 · 4 | 8 rows/program, 768 = 3×256 chunks, 6× LDG.E.128 issued before the first SHFL |
| v2 fwd+bwd (32768,768) | 80.1 | 3.14 | 1.18 [1.18,1.19] | 1.18 [1.18,1.19] | bwd 56 / 0 | 8192 · 4 | |
| v1 fwd (196608,128) | 122.6 | 0.82 | 1.02 | **0.27** | 16 / 0 | 196608 · 1 | 1 warp per 128-row: LDG.64, grid of 196k tiny CTAs — launch/scheduling-bound |
| v2 fwd (196608,128) | 33.3 | 3.02 | 3.75 [3.74,3.76] | 0.98 [0.98,0.99] | 36 / 0 | 12288 · 4 | 16 rows/program, LDG.E.128 — parity with Inductor's generated kernel |
| v2 fwd+bwd (196608,128) | 81.7 | 3.08 | 3.15 | 0.96 | bwd 32 / 0 | | |
| v3 mixnorm fwd (16,2048,768) | 60.6 | 3.32 (91%) | 4.68 [4.67,4.69] | 1.08 [1.07,1.08] | 48 / 0 | 8192 · 4 | one pass: writes x_new and h |
| v3 mixnorm fwd+bwd | 166.2 | 3.03 | 4.31 | 1.10 [1.10,1.10] | bwd 127 / 0 + finish 32 | 8192 + 2 | dr/dl via per-row partials + fp64 finish kernel; explicit `tl.fma` |

**Why (goal iii).** (1) 768-wide: eager *and* compile run the same `aten::_fused_rms_norm` kernel (Inductor does not regenerate a standalone rms_norm) at 2.36 TB/s = 64% of the copy roof; v2 reaches 85% by giving each program 8 rows and issuing all six 128-bit loads per thread before the reduction (enough bytes in flight per SM by Little's law, PLAN §7.2), and by avoiding v1's 25% masked lanes (BLOCK 1024 for D=768). (2) 128-wide: v1's one-warp-per-row design is the textbook mistake — 196,608 CTAs of 32 threads with 8-byte loads; v2 fixes it to parity with Inductor, which already generates a multi-row kernel for this shape. (3) mixnorm: Inductor already fuses the λ-mix into the norm (BAS-1: 89% of roof in-model), so a ≈ 1.1× win is the expected ceiling. Our numerical error is 0.38–1.00× stock's on every gated output (stock = the eager bf16 op).
**Vs the plan:** KT-1 acceptance met (G1/G2; CIs vs both baselines; ≥ 80% of 3.65 TB/s for the 768 forward = 32.6 µs ≤ 34 µs). The 1.31× vs compile on the standalone 768 norm beats the BAS-1 "parity" expectation, but only because Inductor does not code-generate standalone RMSNorm; in-model the norm is fused into the λ-mix (≈ 1.1×), so the end-to-end effect will be small. Isolated-kernel numbers; in-model numbers come from INT-1/INT-3.

<a id="qkrope"></a>
## ISA card: `qkrope` (KT-2, Triton fused QK-norm + RoPE + ×1.2, fwd and bwd) — job 3409677 (gh090), 2026-10-18; gates G1+G2 passed (94 GPU tests)

Shape q,k (16,2048,6,128) bf16, cos/sin (1,2048,1,64) bf16; q and k in one launch; fp32 rotation, rstd from the rotated row (KT-2 design decision). Results `results/db/qkrope/*.json`; ISA `results/isa/qkrope/triton_<variant>/sm_90a/`.

| variant | fwd µs | fwd TB/s | fwd vs eager | **fwd vs compile [CI]** | fwd+bwd µs | **fwd+bwd vs compile [CI]** | regs fwd/bwd, spills | LDG.E.128 per thread fwd/bwd |
|---|---|---|---|---|---|---|---|---|
| op (v1, triton_op) | **58.6** | **3.44 (94%)** | 14.7× | **1.33 [1.32,1.33]** | **153.5** | **2.08 [2.07,2.08]** | 32 / 62, 0 | 8 / 12 (+2 × 32-bit rstd) |
| fn v1 | 61.6 | 3.27 | 14.0× | 1.26 | 148.7 | 2.15 [2.14,2.15] | 32 / 62, 0 | 8 / 12 |
| fn ilp2 | 62.4 | 3.23 | 13.8× | 1.24 | 150.3 | 2.12 | 52 / 95, 0 | 16 / 24 |
| fn ilp4 | 62.9 | 3.21 | 13.7× | 1.24 | 151.3 | 2.11 | 94 / 167, 0 | 32 / 48 |

Baselines (same job): eager chain fwd 861.6 µs / fwd+bwd 1898.6 µs; compile fwd 77.7 µs (2.60 TB/s) / fwd+bwd 319.0 µs (1.58 TB/s effective).
**Why (goal iii).** Forward: Inductor's single fused kernel moves 1.36× the minimum bytes (BAS-1) and reaches 2.6 TB/s; ours moves the minimum (one read and one write of q and k, 128-bit vector loads/stores including the cos/sin table) at 94% of the copy roof → 1.33× (CI lower bound 1.32 ≥ 1.3: a valid "≥ 1.3×" claim). Backward: Inductor uses three kernels and five launches per layer, including a zero-filled `slice_backward` scatter of the rotated halves, moving 2.09× the minimum; ours is one kernel that recomputes the rotation from the saved input and the saved fp32 rstd → **2.08× vs compile on fwd+bwd**, above the BAS-1 projection (1.6–1.9×). **ILP does not help**: doubling or quadrupling the loads in flight per thread (ilp2/ilp4) changes nothing — v1's 8 × 16-B loads per thread × 2048 threads/SM ≈ 16k loads in flight per SM, far above the ≈ 600 needed at 344 ns (Little's law) — it only costs registers (167 in the ilp4 backward). Open question: the triton_op path is 5% faster than the autograd.Function path with the same kernel and config (58.6 vs 61.6 µs, tight CIs) — a host/allocation difference to explain with nsys before quoting the fn rows.
**Vs the plan:** KT-2 acceptance: G1/G2 ✓; vs eager ≥ 3× ✓ (14.7× fwd, 12.4× fwd+bwd); vs compile against the 1.3–1.6× forward target ✓ (1.33×); still open: ncu bytes (≈ 200 MB per forward call) — add to BAS-6 (S4).
