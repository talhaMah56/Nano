# Inductor fusion facts (BAS-1, decision D1) — job 3406171

- ncu: `cisc662/results/runs/s2_inductor_3406171/ncu_step20_raw.csv` — 915 launches over 1 step(s); 65.77 ms/step summed kernel time, 116.96 GB/step DRAM. NVTX steps in the capture: [20].
- Inductor graphs: 11 (graph_001=bwd(46), graph_002=fwd(39), graph_003=adamw(1), graph_004=adamw(1), graph_005=adamw(1), graph_006=adamw(1), graph_007=adamw(1), graph_008=muon(15), graph_009=muon(18), graph_010=muon(18), graph_011=muon(18)); 0 Triton launches not found in output_code (classified from the kernel name).
- Phase source: NVTX ranges in the ncu CSV (no range = autograd thread = backward). Dims: `B=16,T=2048,C=768,H=6,D=128,F=3072,V=32768,L=12`.
- nsys cross-check: `cisc662/results/runs/s1_nsys_3400686/compile_fa3_cuda_gpu_trace.csv (per-kernel median over 5 steps)` — S1 74.25 ms/step GPU kernel time (5-step mean, `attrib_compile_fa3.json`) vs ncu 65.77 ms/step (-11.4%). S1 per-step totals: 67.06, 72.81, 76.12, 77.89, 77.40 ms (median 76.12; ncu -13.6% vs median).
- Caveats: ncu serialises kernels and by default flushes caches before each kernel (`--cache-control all`), so bytes are cold-L2 traffic per kernel and small kernels look slower than in nsys; times are at default (unlocked) clocks (`--clock-control none`). ncu times are for byte/bandwidth facts only, never headline times (PLAN.md §7.1). achieved GB/s = ncu DRAM bytes / ncu time; effective GB/s = minimum bytes / ncu time. Writes still dirty in the 60 MiB L2 when a kernel ends are not counted as DRAM writes, so a streaming kernel can show up to ~50 MB per launch below the minimum (ratio < 1, effective GB/s above the roof).

## Per-region totals (per step)

| region | tier | kernels | launches | ncu µs | nsys µs (S1) | DRAM GB | min GB | ratio | achieved GB/s | % of 3650 | effective GB/s | occupancy % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixnorm_fwd | A | 25 | 25 | 1,713.4 | 1,902.5 | 5.514 | 5.540 | 1.00 | 3,218 | 88% | 3,233 | 92.3 |
| mixnorm_bwd | A | 28 | 51 | 3,161.8 | 3,262.3 | 10.137 | 8.560 | 1.18 | 3,206 | 88% | 2,707 | 69.6 |
| qkrope_fwd | A | 1 | 12 | 1,210.7 | 1,339.5 | 3.310 | 2.441 | 1.36 | 2,734 | 75% | 2,016 | 70.7 |
| qkrope_bwd | A | 3 | 60 | 2,313.2 | 2,418.4 | 7.620 | 3.649 | 2.09 | 3,294 | 90% | 1,577 | 69.1 |
| relu2_fwd | A | 1 | 12 | 1,290.8 | 1,319.9 | 4.505 | 4.832 | 0.93 | 3,490 | 96% | 3,743 | 82.9 |
| relu2_bwd | A | 1 | 12 | 1,904.0 | 1,957.9 | 6.936 | 7.248 | 0.96 | 3,643 | 100% | 3,807 | 86.4 |
| ce_fwd | A | 3 | 3 | 1,940.7 | 2,275.3 | 2.150 | 2.148 | 1.00 | 1,108 | 30% | 1,107 | 97.8 |
| ce_bwd | A | 1 | 1 | 1,879.7 | 2,164.9 | 4.266 | 4.295 | 0.99 | 2,270 | 62% | 2,285 | 70.8 |
| optim_adamw | A | 2 | 13 | 863.6 | 908.8 | 2.943 | 3.171 | 0.93 | 3,408 | 93% | 3,672 | 59.8 |
| optim_muon | A | 15 | 32 | 1,310.8 | 1,359.2 | 4.247 | 2.038 | 2.08 | 3,240 | 89% | 1,555 | 89.2 |
| optim_muon_ns | - | 14 | 43 | 1,156.7 | 1,184.4 | 3.754 | 0.000 | nan | 3,245 | 89% | nan | 83.2 |
| optim_muon_stack | - | 2 | 15 | 584.1 | 640.5 | 1.740 | 0.000 | nan | 2,979 | 82% | nan | 69.4 |
| optim_muon_ns_gemm | - | 7 | 60 | 2,666.8 | 3,309.7 | 3.509 | 0.000 | nan | 1,316 | 36% | nan | 14.3 |
| gemm_fwd | - | 5 | 80 | 10,855.4 | 12,761.7 | 12.812 | 0.000 | nan | 1,180 | 32% | nan | 15.0 |
| gemm_bwd | - | 11 | 239 | 21,445.3 | 25,842.6 | 27.646 | 0.000 | nan | 1,289 | 35% | nan | 17.4 |
| attention_fwd | - | 1 | 12 | 2,278.8 | 2,626.1 | 2.396 | 0.000 | nan | 1,051 | 29% | nan | 14.0 |
| attention_bwd | - | 3 | 36 | 7,152.1 | 8,482.5 | 9.410 | 0.000 | nan | 1,316 | 36% | nan | 19.3 |
| weight_cast_fwd | - | 5 | 80 | 368.9 | 321.2 | 0.468 | 0.000 | nan | 1,270 | 35% | nan | 43.6 |
| weight_cast_bwd | - | 5 | 80 | 353.4 | 256.0 | 0.292 | 0.000 | nan | 826 | 23% | nan | 47.1 |
| embedding_ve_fwd | - | 2 | 7 | 314.4 | 337.9 | 0.844 | 0.000 | nan | 2,685 | 74% | nan | 84.7 |
| embedding_ve_bwd | - | 4 | 21 | 918.9 | 1,068.0 | 2.352 | 0.000 | nan | 2,560 | 70% | nan | 85.2 |
| other_elementwise_fwd | - | 4 | 15 | 37.5 | 17.1 | 0.004 | 0.000 | nan | 103 | 3% | nan | 10.2 |
| other_elementwise_bwd | - | 6 | 6 | 49.4 | 46.3 | 0.101 | 0.000 | nan | 2,044 | 56% | nan | 66.0 |
| **total** | | | 915 | 65,770.6 | 75,802.6 | 116.957 | | | | | | |

## Draft vs-compile targets (Tier-A regions)

Target logic (BAS-1 step 4): a fused kernel that moves only the minimum bytes at 2.7 / 3.0 / 3.3 TB/s (74-90% of the measured 3650 GB/s copy roof; at least 2 µs per launch) vs Inductor's measured ncu time. A claim needs >= 1.3x with a bootstrap-CI lower bound >= 1.3 in kbench (PLAN.md §7.1) — these lines only set expectations.

- **residual/lambda-mix + RMSNorm fwd**: Inductor 1,713 us/step, 5.514 GB DRAM (min 5.540 GB, ratio 1.00), 3,218 GB/s achieved, 3,233 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 1,679-2,052 us -> **target vs compile 0.84-1.02x** (central 0.93x at 3.0 TB/s). ~parity: Inductor already runs at 3,233 GB/s effective on the minimum bytes (89% of 3650); only removing bytes across the GEMM boundary (epilogue/prologue fusion) can help.
- **residual/lambda-mix + RMSNorm bwd**: Inductor 3,162 us/step, 10.137 GB DRAM (min 8.560 GB, ratio 1.18), 3,206 GB/s achieved, 2,707 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 2,594-3,170 us -> **target vs compile 1.00-1.22x** (central 1.11x at 3.0 TB/s). ~parity: Inductor already runs at 2,707 GB/s effective on the minimum bytes (74% of 3650); only removing bytes across the GEMM boundary (epilogue/prologue fusion) can help.
- **residual/lambda-mix + RMSNorm fwd+bwd**: Inductor 4,875 us/step, 15.651 GB DRAM (min 14.099 GB, ratio 1.11), 3,210 GB/s achieved, 2,892 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 4,273-5,222 us -> **target vs compile 0.93-1.14x** (central 1.04x at 3.0 TB/s). ~parity: Inductor already runs at 2,892 GB/s effective on the minimum bytes (79% of 3650); only removing bytes across the GEMM boundary (epilogue/prologue fusion) can help.
- **QK-norm+RoPE+scale fwd**: Inductor 1,211 us/step, 3.310 GB DRAM (min 2.441 GB, ratio 1.36), 2,734 GB/s achieved, 2,016 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 740-904 us -> **target vs compile 1.34-1.64x** (central 1.49x at 3.0 TB/s). target >= 1.3x (bytes: Inductor moves 1.36x the minimum).
- **QK-norm+RoPE+scale bwd**: Inductor 2,313 us/step, 7.620 GB DRAM (min 3.649 GB, ratio 2.09), 3,294 GB/s achieved, 1,577 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 1,106-1,351 us -> **target vs compile 1.71-2.09x** (central 1.90x at 3.0 TB/s). target >= 1.3x (bytes: Inductor moves 2.09x the minimum).
- **QK-norm+RoPE+scale fwd+bwd**: Inductor 3,524 us/step, 10.930 GB DRAM (min 6.090 GB, ratio 1.79), 3,102 GB/s achieved, 1,728 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 1,845-2,256 us -> **target vs compile 1.56-1.91x** (central 1.74x at 3.0 TB/s). target >= 1.3x (bytes: Inductor moves 1.79x the minimum).
- **relu^2 fwd**: Inductor 1,291 us/step, 4.505 GB DRAM (min 4.832 GB, ratio 0.93), 3,490 GB/s achieved, 3,743 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 1,464-1,790 us -> **target vs compile 0.72-0.88x** (central 0.80x at 3.0 TB/s). ~parity: Inductor already runs at 3,743 GB/s effective on the minimum bytes (103% of 3650); only removing bytes across the GEMM boundary (epilogue/prologue fusion) can help. Beyond the pointwise kernel: the c_fc epilogue (KT-7) avoids re-reading h, -4A/layer = -2.4 GB/step.
- **relu^2 bwd**: Inductor 1,904 us/step, 6.936 GB DRAM (min 7.248 GB, ratio 0.96), 3,643 GB/s achieved, 3,807 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 2,196-2,684 us -> **target vs compile 0.71-0.87x** (central 0.79x at 3.0 TB/s). ~parity: Inductor already runs at 3,807 GB/s effective on the minimum bytes (104% of 3650); only removing bytes across the GEMM boundary (epilogue/prologue fusion) can help. Beyond the pointwise kernel: the c_proj dX epilogue (KT-7) never stores dy, -8A/layer = -4.8 GB/step.
- **softcap+CE fwd**: Inductor 1,941 us/step, 2.150 GB DRAM (min 2.148 GB, ratio 1.00), 1,108 GB/s achieved, 1,107 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 651-796 us -> **target vs compile 2.44-2.98x** (central 2.71x at 3.0 TB/s). target >= 1.3x (bandwidth: Inductor achieves 1108 GB/s (30% of 3650)).
- **softcap+CE bwd**: Inductor 1,880 us/step, 4.266 GB DRAM (min 4.295 GB, ratio 0.99), 2,270 GB/s achieved, 2,285 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 1,302-1,591 us -> **target vs compile 1.18-1.44x** (central 1.31x at 3.0 TB/s). target >= 1.3x (bandwidth: Inductor achieves 2270 GB/s (62% of 3650)).
- **softcap+CE fwd+bwd (separate fwd and bwd kernels)**: Inductor 3,820 us/step, 6.417 GB DRAM (min 6.443 GB, ratio 1.00), 1,680 GB/s achieved, 1,686 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 1,952-2,386 us -> **target vs compile 1.60-1.96x** (central 1.78x at 3.0 TB/s). target >= 1.3x (bandwidth: Inductor achieves 1680 GB/s (46% of 3650)). Beyond this: chunked lm_head+CE (KT-4) never writes or re-reads the 2.15 GB bf16 logits.
- **AdamW**: Inductor 864 us/step, 2.943 GB DRAM (min 3.171 GB, ratio 0.93), 3,408 GB/s achieved, 3,672 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 961-1,174 us -> **target vs compile 0.74-0.90x** (central 0.82x at 3.0 TB/s). ~parity: Inductor already runs at 3,672 GB/s effective on the minimum bytes (101% of 3650); only removing bytes across the GEMM boundary (epilogue/prologue fusion) can help. Parity expected; any gain must come from fewer launches (13 tensors) or bf16 state, not bandwidth.
- **Muon pre/post-NS pointwise**: Inductor 1,311 us/step, 4.247 GB DRAM (min 2.038 GB, ratio 2.08), 3,240 GB/s achieved, 1,555 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 618-755 us -> **target vs compile 1.74-2.12x** (central 1.93x at 3.0 TB/s). target >= 1.3x (bytes: Inductor moves 2.08x the minimum).
- **Muon all non-GEMM traffic (pre/post + NS pointwise + stack/unstack)**: Inductor 3,052 us/step, 9.741 GB DRAM (min 2.038 GB, ratio 4.78), 3,192 GB/s achieved, 668 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 618-755 us -> **target vs compile 4.04-4.94x** (central 4.49x at 3.0 TB/s). target >= 1.3x (bytes: Inductor moves 4.78x the minimum).
- **softcap+CE, gradient written in the forward (KT-3 single pass)**: Inductor 3,820 us/step, 6.417 GB DRAM (min 4.295 GB, ratio 1.49), 1,680 GB/s achieved, 1,124 GB/s effective on min bytes -> fused kernel at 2.7-3.3 TB/s on min bytes: 1,302-1,591 us -> **target vs compile 2.40-2.94x** (central 2.67x at 3.0 TB/s). target >= 1.3x (bytes: Inductor moves 1.49x the minimum; bandwidth: Inductor achieves 1680 GB/s (46% of 3650)).

## Algorithmic minimum bytes per step (formulas)

Model: a fused kernel at the same GEMM / FA3 boundaries as stock nanochat reads each distinct input once and writes each output once; tensors the backward needs are ones the forward writes anyway (or recomputes from inputs it already reads). N = B*T tokens.

- `mixnorm_fwd` = 5.5398 GB: 110A + (2L+1)*N*4 rstd with A = N*C*2 = 50,331,648 B  [layer 0 lambda-mix+norm (read x0; write a, h): 1 x (1R+2W)A; layers 1..L-1 MLP-residual add + lambda-mix + norm (read b, mlp_out, x0; write a, h): 11 x (3R+2W)A; attn-residual add + norm (read a, attn_out; write b, h): 12 x (2R+2W)A; final add - backout + norm (read b, mlp_out, x_backout; write h): 1 x (3R+1W)A]
- `mixnorm_bwd` = 8.5597 GB: 170A + (2L+1)*N*4 rstd with A = N*C*2 = 50,331,648 B  [layer 0 (read dh x3 q/k/v dX GEMMs, db, x0, dx0_acc; write dx0): 1 x (6R+1W)A; layers 1..L-1 (read dh x3, db, b, mlp_out, x0, dx0_acc; write dx, dx0_acc): 11 x (8R+2W)A; attn-residual norm bwd (read dh, dx, b; write db): 12 x (3R+1W)A; final norm + backout bwd (read dh, h, x_backout; write dx): 1 x (3R+1W)A; backout grad added at layer L/2 (read dx): 1 x (1R+0W)A]
- `qkrope_fwd` = 2.4411 GB: L*(4A + 2*N*H*4 rstd + cos/sin 524,288 B): read q,k (GEMM outputs), write rotated+normed+scaled q,k
- `qkrope_bwd` = 3.6490 GB: L*(6A + rstd + cos/sin): read dq',dk' (FA3 bwd) and q,k; write dq,dk
- `relu2_fwd` = 4.8318 GB: L*8A: read h (N x F bf16 = 4A), write relu(h)^2 (4A)
- `relu2_bwd` = 7.2478 GB: L*12A: read dy (4A) and h (4A), write dh (4A)
- `ce_fwd` = 2.1479 GB: N*V*2 (read bf16 logits once, 2,147,483,648 B) + N*8 targets + N*4 lse
- `ce_bwd` = 4.2954 GB: 2*N*V*2 (read logits, write bf16 dlogits) + targets + lse
- `optim_adamw` = 3.1709 GB: 28 B/param x 25,165,874 fp32 params (lm_head, scalars: read p,g,m,v, write p,m,v) + 14 B/param x 176,160,768 bf16 params (wte + 6 value embeddings)
- `optim_muon` = 2.0384 GB: 24 B/param x 84,935,088 Muon params: pre-NS read g,m (fp32) write m (fp32) + X (bf16); post-NS read X (bf16), p write p (fp32); NS pointwise folded into bmm epilogues and stack/unstack avoided (both 0); extra passes for the global norms not counted
- Non-Tier-A regions (GEMM, FA3, casts, embeddings, Muon NS/stack) have no minimum here; Muon NS pointwise and stack/unstack count against the Muon 'all non-GEMM' line with minimum 0.

## ncu vs nsys (S1) per kernel: 10 flagged (> 30% time difference for kernels >= 10 µs/step, or different launch counts)

| kernel | ncu µs | nsys µs | ncu/nsys | flag |
|---|---|---|---|---|
| `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16` | 883.5 | 622.3 | 1.42 | time differs +42% |
| `triton_poi_fused__to_copy_7` | 151.7 | 98.6 | 1.54 | time differs +54% |
| `triton_poi_fused__to_copy_4` | 144.2 | 108.0 | 1.34 | time differs +34% |
| `triton_red_fused_mul_sum_18` | 108.0 | 79.8 | 1.35 | time differs +35% |
| `void at::vectorized_elementwise_kernel<4, at::FillFunctor<int>, std::array<char *, 1>>(int` | 28.4 | 10.8 | 2.62 | time differs +162% |
| `void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat16, __nv_bfloat16, float, __nv_` | 43.1 | 27.0 | 1.60 | time differs +60% |
| `triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view` | 27.4 | 14.3 | 1.92 | time differs +92% |
| `triton_per_fused__to_copy_lerp_linalg_vector_norm_rsub_1` | 23.0 | 10.7 | 2.16 | time differs +116% |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_32x1_nn_align2>` | 34.1 | 24.9 | 1.37 | time differs +37% |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_32x1_tn_align2>` | 16.8 | 12.6 | 1.33 | time differs +33% |

## Top kernels per Tier-A region

- `mixnorm_fwd` (residual add + lambda-mix + RMSNorm, forward): `triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_pow_rsqrt_select_sigmoid` x1 102 µs 123 MB [_to_copy+add+cat+mean+mul+pow+rsqrt+select+sigmoid+slice]; `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_16` x1 69 µs 227 MB [_to_copy+add+mean+mul+pow+rsqrt+select]; `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_30` x1 68 µs 227 MB [_to_copy+add+mean+mul+pow+rsqrt+select]; `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_18` x1 68 µs 227 MB [_to_copy+add+mean+mul+pow+rsqrt+select]
- `mixnorm_bwd` (residual add + lambda-mix + RMSNorm, backward (incl. lambda-grad reductions)): `triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_neg_select_slice_back` x1 292 µs 981 MB [_fused_rms_norm_backward+_to_copy+add+mul+neg+select+slice_backward]; `triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward` x1 285 µs 881 MB [_fused_rms_norm_backward+_to_copy+add+mul+select+slice_backward]; `triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward` x1 219 µs 628 MB [_fused_rms_norm_backward+_to_copy+add+mul+select+slice_backward]; `triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward` x1 181 µs 629 MB [_fused_rms_norm_backward+_to_copy+add+mul+select+slice_backward]
- `qkrope_fwd` (QK-norm + RoPE + 1.2 scale, forward): `triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view` x12 1,211 µs 3,310 MB [_to_copy+add+cat+mean+mul+neg+pow+rsqrt+slice]
- `qkrope_bwd` (QK-norm + RoPE + 1.2 scale, backward (incl. the RoPE half-slice scatter)): `triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9` x12 955 µs 3,385 MB [_fused_rms_norm_backward+_to_copy+add+mul+neg+slice]; `triton_per_fused__fused_rms_norm_backward__to_copy_mul_8` x24 728 µs 2,446 MB [_fused_rms_norm_backward+_to_copy+mul]; `triton_poi_fused_add_slice_backward_13` x24 630 µs 1,789 MB [add+slice_backward]
- `relu2_fwd` (relu^2, forward): `triton_poi_fused__unsafe_view_pow_relu_9` x12 1,291 µs 4,505 MB [pow+relu]
- `relu2_bwd` (relu^2, backward): `triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5` x12 1,904 µs 6,936 MB [mul+pow+relu+threshold_backward]
- `ce_fwd` (softcap + cross-entropy, forward): `triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_onli` x1 1,910 µs 2,148 MB [_log_softmax+_to_copy+div+mul+prepare_softmax_online+tanh]; `triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub` x1 27 µs 3 MB [_log_softmax+_to_copy+div+mul+nll_loss_forward+sub+tanh]; `triton_per_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub` x1 3 µs 0 MB [_log_softmax+_to_copy+div+mul+nll_loss_forward+sub+tanh]
- `ce_bwd` (softcap + cross-entropy, backward (dlogits)): `triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_d` x1 1,880 µs 4,266 MB [_log_softmax+_log_softmax_backward_data+_to_copy+div+mul+nll_loss_backward+nll_l]
- `optim_adamw` (AdamW fused step (lm_head, wte, value embeddings, scalars)): `triton_poi_fused__to_copy_add_copy_copy__div_lerp_mul_neg_pow_rsub_sqrt_0` x7 660 µs 2,267 MB [_to_copy+add+copy+copy_+div+lerp+mul+neg+pow+rsub+sqrt]; `triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0` x6 204 µs 677 MB [add+copy_+div+lerp+mul+neg+pow+rsub+sqrt]
- `optim_muon` (Muon pre/post-NS pointwise (momentum, MuonEq, Muon+, NorMuon, cautious update)): `triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_linalg_vector_norm_mea` x3 305 µs 1,113 MB [_to_copy+add+clamp_min+copy_+div+ge+lerp+linalg_vector_norm+mean+mul+pow+recipro]; `triton_per_fused__to_copy_clamp_min_copy__div_lerp_linalg_vector_norm_mul_rsub_2` x2 280 µs 962 MB [_to_copy+clamp_min+copy_+div+lerp+linalg_vector_norm+mul+rsub]; `triton_red_fused__to_copy_lerp_linalg_vector_norm_rsub_0` x3 220 µs 680 MB [_to_copy+lerp+linalg_vector_norm+rsub]; `triton_red_fused__to_copy_add_linalg_vector_norm_mul_14` x3 168 µs 510 MB [_to_copy+add+linalg_vector_norm+mul]

Generated by `python -m cisc662.analysis.inductor_fusion cisc662/results/runs/s2_inductor_3406171`.
