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
