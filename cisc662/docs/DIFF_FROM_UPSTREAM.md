# Differences from upstream nanochat

Upstream: karpathy/nanochat at `92d63d4` (2026-07-03). Everything project-specific lives in `cisc662/`.
Only three upstream files are modified, all opt-in (defaults unchanged); the timed code path is otherwise identical to `92d63d4`.
Check with: `git diff 92d63d4 --stat -- nanochat scripts`.

| File | Change | Default behaviour |
|---|---|---|
| `nanochat/flash_attention.py` | `NANOCHAT_ATTN_IMPL=fa3\|sdpa` env override of `_override_impl`; optional `softmax_scale=None` argument on `flash_attn_func` (passed to FA3, or as `scale=` to SDPA) so a fused QK-norm+RoPE kernel can fold the q/k ×1.2 into the softmax scale | unset / `None` → upstream behaviour (verified bit-identical on the SDPA path) |
| `scripts/base_train.py` | `NANOCHAT_PROFILE_STEPS=a:b` → `cudaProfilerStart/Stop` window + NVTX ranges (`step_N`, `forward`, `backward`, `dataloader`, `optimizer`); `NANOCHAT_EMIT_NVTX=1` → per-op NVTX ranges with shapes inside the window | unset → no profiler calls, no NVTX |
| `nanochat/common.py` | `gh200` entries in the peak-FLOPS (989e12) and peak-bandwidth (4.0e12) tables, placed before `h200` (the name "NVIDIA GH200 120GB" otherwise matched H200's 4.8 TB/s) | no effect on non-GH200 GPUs |

Run-time integration of our kernels goes through `cisc662/integrate.py` (monkeypatches `nanochat.gpt` when loaded by `cisc662/bench/train_wrap.py`); it is not an upstream edit.
Eager baselines use `TORCH_COMPILE_DISABLE=1` (no code change).
`uv.lock` is never modified (always `uv sync --frozen --inexact`).
