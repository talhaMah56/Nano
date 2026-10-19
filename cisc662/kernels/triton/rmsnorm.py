"""
Triton RMSNorm (768- and 128-wide rows) and fused residual/lambda-mix + RMSNorm (`mixnorm`), card KT-1.

nanochat semantics (nanochat/gpt.py at the cisc662 branch point):
  norm(x)   = F.rms_norm(x, (x.size(-1),)), parameter-free, eps=None. In torch 2.9.1 eps=None is
              torch.finfo(float32).eps = 1.1920929e-07 for bf16/fp16/fp32 inputs (opmath epsilon),
              NOT finfo(bfloat16).eps. Used on the residual stream (B,T,768) and as QK-norm (B,T,H,128).
  mixnorm   GPT.forward:  x = resid_lambdas[i] * x + x0_lambdas[i] * x0  (fp32 0-d params, bf16
              activations -> bf16 result), then Block.forward: norm(x). We write x_new in the activation
              dtype (what nanochat stores and the residual stream carries on) AND h = norm(x_new) in one
              pass; h is the norm of the *stored* (rounded) x_new, as in nanochat. x_new is computed in fp32
              with one rounding (eager rounds r*x and l*x0 separately; Inductor rounds once, like us).

Kernels (names are what nsys/ncu and profiling/classify_kernels.py match, pattern c662_rmsnorm):
  c662_rmsnorm_fwd_v1        one row per program, BLOCK_D = next_pow2(D) masked (768 -> 1024: 25% idle lanes)
  c662_rmsnorm_fwd_v2        BLOCK_M rows per program; the row is CHUNK-wide pieces held in registers
                             (768 = 3 x 256, 128 = 1 x 128: no masked lanes), all loads issued before use
  c662_rmsnorm_bwd           dx = rstd * (dy - xhat * mean(dy * xhat)), xhat = x * rstd; reads x, dy, rstd
  c662_rmsnorm_fwd_mixnorm   reads x, x0, r, l; writes x_new, h, rstd
  c662_rmsnorm_bwd_mixnorm   reads g_xnew (optional), g_h, x, x0, r, l, rstd (x_new recomputed, not read);
                             writes dx, dx0 and per-row partial sums of dr, dl
  c662_rmsnorm_bwd_mixnorm_finish  2 programs: dr, dl = fp64 sums of those partials (deterministic, no atomics)
All accumulate in fp32 (fp64 inputs compute in fp64, for gradcheck); rstd is stored fp32 (N x 4 B).
Row offsets get tl.multiple_of(rows*stride, 16) when all strides are multiples of 16 (128-bit accesses).

Entry points:
  rmsnorm(x, variant="v2", impl="op")      impl="op": torch.ops.cisc662.rmsnorm (torch.library.triton_op +
                                            register_autograd, traceable by torch.compile); impl="fn":
                                            torch.autograd.Function calling the launchers directly
  mixnorm(x, x0, r, l, impl="op")          -> (x_new, h); torch.ops.cisc662.mixnorm
  reference(x, dtype=f64), reference_mixnorm(x, x0, r, l, dtype=f64, round_xnew_to=None)

Self-test / tools:
  python -m cisc662.kernels.triton.rmsnorm                      # small-shape self-test (CUDA, or CPU with
                                                                # TRITON_INTERPRET=1)
  python -m cisc662.kernels.triton.rmsnorm --dump-isa OUTDIR    # GPU job: compile at the real shapes and write
                                                                # OUTDIR/triton_<variant>/sm_90a/{ptx,sass,ttgir,meta.json}
  python -m cisc662.kernels.triton.rmsnorm --xcompile OUTDIR    # login node: same, cross-compiled for
                                                                # GPUTarget("cuda", 90, 32) without a GPU;
                                                                # registers/spills from ptxas -v / cuobjdump
"""
import json
import os
import re
import subprocess
import sys
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

FP32_EPS = torch.finfo(torch.float32).eps     # 1.1920929e-07: F.rms_norm(eps=None) for bf16/fp16/fp32 (torch 2.9.1)
VARIANTS = ("v1", "v2")
_LOW = (torch.bfloat16, torch.float16)

# Launch configurations per (kernel, D): BLOCK_M rows per program and num_warps. Module-level so a tuning
# job can override them (e.g. CONFIGS[("v2", 768)] = dict(BLOCK_M=4, num_warps=4)). Rationale (no GPU
# tuning yet; see the ISA card): with CHUNK=256 and 128-bit (8 x bf16) accesses one warp covers one
# 256-wide chunk row, so (BLOCK_M, 4 warps) gives BLOCK_M/4 rows per warp and 3*BLOCK_M/4 independent
# LDG.128 per thread per input tensor, reduced with warp shuffles only (no shared memory).
CONFIGS = {
    ("v1", 768): dict(num_warps=4),             # 1024 lanes, 128 threads x 8
    ("v1", 128): dict(num_warps=1),             # 128 lanes, 32 threads x 4
    ("v2", 768): dict(BLOCK_M=8, num_warps=4),  # 2 rows/warp, 6 LDG.128 per thread in flight
    ("v2", 128): dict(BLOCK_M=16, num_warps=4),  # 16 lanes per row, 2 rows/warp, 2 LDG.128 per thread
    ("bwd", 768): dict(BLOCK_M=4, num_warps=4),  # x and dy: 6 LDG.128 per thread
    ("bwd", 128): dict(BLOCK_M=16, num_warps=4),
    ("mix_fwd", 768): dict(BLOCK_M=4, num_warps=4),  # x and x0
    ("mix_bwd", 768): dict(BLOCK_M=4, num_warps=4),  # g_xnew, g_h, x, x0: 12 LDG.128 per thread
}
_DEFAULT_BLOCK_M = {"v2": 4, "bwd": 4, "mix_fwd": 4, "mix_bwd": 4}
FINISH_BLOCK = 2048     # c662_rmsnorm_bwd_mixnorm_finish: 16 iterations over the 32768 R12 rows


def _cfg(kind, D):
    c = CONFIGS.get((kind, D))
    if c is not None:
        return dict(c)
    if kind == "v1":
        return dict(num_warps=4 if triton.next_power_of_2(D) >= 512 else 1)
    if D <= 256:  # narrow rows: several rows per program
        return dict(BLOCK_M=min(16, max(1, 2048 // triton.next_power_of_2(D))), num_warps=4)
    return dict(BLOCK_M=_DEFAULT_BLOCK_M[kind], num_warps=4)


def _chunking(D):
    """(CHUNK, NCHUNK): the row is NCHUNK pieces of CHUNK (a power of two) columns, NCHUNK <= 4."""
    if D <= 256:
        return triton.next_power_of_2(D), 1
    chunk = 256
    while triton.cdiv(D, chunk) > 4:
        chunk *= 2
    return chunk, triton.cdiv(D, chunk)


# ------------------------------------------------------------------------------------------------
# Triton helpers (inlined into the kernels)
# ------------------------------------------------------------------------------------------------
@triton.jit
def _c662_load(base, rmask, cols, c: tl.constexpr, D: tl.constexpr, CHUNK: tl.constexpr, IS_F64: tl.constexpr):
    """Chunk c of BLOCK_M rows (base: (BLOCK_M, CHUNK) pointers of chunk 0), upcast to the compute dtype.
    Only the row mask unless the chunk crosses D (row-uniform masks keep 128-bit vectorisation)."""
    if (c + 1) * CHUNK <= D:
        v = tl.load(base + c * CHUNK, mask=rmask[:, None], other=0.0)
    else:
        v = tl.load(base + c * CHUNK, mask=rmask[:, None] & (cols[None, :] + c * CHUNK < D), other=0.0)
    if not IS_F64:
        v = v.to(tl.float32)
    return v


@triton.jit
def _c662_store(base, v, rmask, cols, c: tl.constexpr, D: tl.constexpr, CHUNK: tl.constexpr):
    if (c + 1) * CHUNK <= D:
        tl.store(base + c * CHUNK, v.to(base.dtype.element_ty), mask=rmask[:, None])
    else:
        tl.store(base + c * CHUNK, v.to(base.dtype.element_ty), mask=rmask[:, None] & (cols[None, :] + c * CHUNK < D))


@triton.jit
def _c662_rows(BLOCK_M: tl.constexpr, N, IDX64: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < N
    if IDX64:
        rows = rows.to(tl.int64)
    return rows, rmask


@triton.jit
def _c662_roff(rows, stride, ALIGNED: tl.constexpr):
    """Row offsets rows*stride, hinted as multiples of 16 elements when every row stride is (128-bit
    accesses). The hint goes on the offsets, not on the scalar stride: torch's triton_op mutation analysis
    (torch/_higher_order_ops/triton_kernel_wrap.py generate_ttir) re-generates the TTIR with every
    non-tensor argument as a constexpr, and tl.multiple_of(constexpr) raises there -- which made
    torch.compile/opcheck treat every input as mutated and fail to partition the joint graph."""
    off = rows * stride
    if ALIGNED:
        off = tl.multiple_of(off, 16)
    return off


@triton.jit
def _c662_rsqrt(ms, eps, IS_F64: tl.constexpr):
    if IS_F64:
        r = 1.0 / tl.sqrt(ms + eps)       # fp64 (gradcheck path); rsqrt.approx is fp32-only
    else:
        r = tl.math.rsqrt(ms + eps)
    return r


# ------------------------------------------------------------------------------------------------
# Kernels
# ------------------------------------------------------------------------------------------------
@triton.jit
def c662_rmsnorm_fwd_v1(X, Y, RSTD, stride_x, stride_y, eps,
                        D: tl.constexpr, BLOCK_D: tl.constexpr, ALIGNED: tl.constexpr, IS_F64: tl.constexpr,
                        IDX64: tl.constexpr):
    row = tl.program_id(0)
    if IDX64:
        row = row.to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(X + _c662_roff(row, stride_x, ALIGNED) + cols, mask=mask, other=0.0)
    if not IS_F64:
        x = x.to(tl.float32)
    rstd = _c662_rsqrt(tl.sum(x * x, axis=0) / D, eps, IS_F64)
    tl.store(Y + _c662_roff(row, stride_y, ALIGNED) + cols, (x * rstd).to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + row, rstd)


@triton.jit
def c662_rmsnorm_fwd_v2(X, Y, RSTD, N, stride_x, stride_y, eps,
                        D: tl.constexpr, BLOCK_M: tl.constexpr, CHUNK: tl.constexpr, NCHUNK: tl.constexpr,
                        ALIGNED: tl.constexpr, IS_F64: tl.constexpr, IDX64: tl.constexpr):
    rows, rmask = _c662_rows(BLOCK_M, N, IDX64)
    cols = tl.arange(0, CHUNK)
    xb = X + _c662_roff(rows, stride_x, ALIGNED)[:, None] + cols[None, :]
    yb = Y + _c662_roff(rows, stride_y, ALIGNED)[:, None] + cols[None, :]
    # all loads first (ILP: NCHUNK independent 128-bit loads per thread per row before the first use)
    a0 = _c662_load(xb, rmask, cols, 0, D, CHUNK, IS_F64)
    if NCHUNK > 1:
        a1 = _c662_load(xb, rmask, cols, 1, D, CHUNK, IS_F64)
    if NCHUNK > 2:
        a2 = _c662_load(xb, rmask, cols, 2, D, CHUNK, IS_F64)
    if NCHUNK > 3:
        a3 = _c662_load(xb, rmask, cols, 3, D, CHUNK, IS_F64)
    sq = a0 * a0                       # chunks combined elementwise: one warp reduction per row
    if NCHUNK > 1:
        sq += a1 * a1
    if NCHUNK > 2:
        sq += a2 * a2
    if NCHUNK > 3:
        sq += a3 * a3
    rstd = _c662_rsqrt(tl.sum(sq, axis=1) / D, eps, IS_F64)
    rs = rstd[:, None]
    _c662_store(yb, a0 * rs, rmask, cols, 0, D, CHUNK)
    if NCHUNK > 1:
        _c662_store(yb, a1 * rs, rmask, cols, 1, D, CHUNK)
    if NCHUNK > 2:
        _c662_store(yb, a2 * rs, rmask, cols, 2, D, CHUNK)
    if NCHUNK > 3:
        _c662_store(yb, a3 * rs, rmask, cols, 3, D, CHUNK)
    tl.store(RSTD + rows, rstd, mask=rmask)


@triton.jit
def c662_rmsnorm_bwd(DY, X, RSTD, DX, N, stride_dy, stride_x, stride_dx,
                     D: tl.constexpr, BLOCK_M: tl.constexpr, CHUNK: tl.constexpr, NCHUNK: tl.constexpr,
                     ALIGNED: tl.constexpr, IS_F64: tl.constexpr, IDX64: tl.constexpr):
    rows, rmask = _c662_rows(BLOCK_M, N, IDX64)
    cols = tl.arange(0, CHUNK)
    xb = X + _c662_roff(rows, stride_x, ALIGNED)[:, None] + cols[None, :]
    gb = DY + _c662_roff(rows, stride_dy, ALIGNED)[:, None] + cols[None, :]
    ob = DX + _c662_roff(rows, stride_dx, ALIGNED)[:, None] + cols[None, :]
    rstd = tl.load(RSTD + rows, mask=rmask, other=0.0)
    x0 = _c662_load(xb, rmask, cols, 0, D, CHUNK, IS_F64)
    g0 = _c662_load(gb, rmask, cols, 0, D, CHUNK, IS_F64)
    if NCHUNK > 1:
        x1 = _c662_load(xb, rmask, cols, 1, D, CHUNK, IS_F64)
        g1 = _c662_load(gb, rmask, cols, 1, D, CHUNK, IS_F64)
    if NCHUNK > 2:
        x2 = _c662_load(xb, rmask, cols, 2, D, CHUNK, IS_F64)
        g2 = _c662_load(gb, rmask, cols, 2, D, CHUNK, IS_F64)
    if NCHUNK > 3:
        x3 = _c662_load(xb, rmask, cols, 3, D, CHUNK, IS_F64)
        g3 = _c662_load(gb, rmask, cols, 3, D, CHUNK, IS_F64)
    rs = rstd[:, None]
    x0 = x0 * rs                                   # xhat
    p = g0 * x0
    if NCHUNK > 1:
        x1 = x1 * rs
        p += g1 * x1
    if NCHUNK > 2:
        x2 = x2 * rs
        p += g2 * x2
    if NCHUNK > 3:
        x3 = x3 * rs
        p += g3 * x3
    c = (tl.sum(p, axis=1) / D)[:, None]
    _c662_store(ob, rs * (g0 - x0 * c), rmask, cols, 0, D, CHUNK)
    if NCHUNK > 1:
        _c662_store(ob, rs * (g1 - x1 * c), rmask, cols, 1, D, CHUNK)
    if NCHUNK > 2:
        _c662_store(ob, rs * (g2 - x2 * c), rmask, cols, 2, D, CHUNK)
    if NCHUNK > 3:
        _c662_store(ob, rs * (g3 - x3 * c), rmask, cols, 3, D, CHUNK)


@triton.jit
def _c662_mix(xb, x0b, r, l, rmask, cols, c: tl.constexpr, D: tl.constexpr, CHUNK: tl.constexpr,
              IS_F64: tl.constexpr):
    """x_new chunk c = r*x + l*x0 in the compute dtype, rounded to the stored activation dtype and back
    (h is the norm of the stored value; x_new has x's dtype). Also returns the upcast x and x0.
    Explicit fma(r, x, l*x0): with `r*a + l*b` LLVM contracts per element, and it picked fma(r, a, l*b)
    in the forward but fma(l, b, r*a) for half the elements of the backward (sm_90 PTX), so ~2e-4 of the
    recomputed x_new differed by one bf16 ulp from the stored one and dr/dl moved by up to ~1e-3 relative.
    One explicit fma makes the forward and the backward recompute bit-identical."""
    a = _c662_load(xb, rmask, cols, c, D, CHUNK, IS_F64)
    b = _c662_load(x0b, rmask, cols, c, D, CHUNK, IS_F64)
    v = tl.fma(tl.broadcast_to(r, a.shape), a, l * b).to(xb.dtype.element_ty)
    if not IS_F64:
        v = v.to(tl.float32)
    return v, a, b


@triton.jit
def c662_rmsnorm_fwd_mixnorm(X, X0, R, L, XN, H, RSTD, N, stride_x, stride_x0, stride_xn, stride_h, eps,
                             D: tl.constexpr, BLOCK_M: tl.constexpr, CHUNK: tl.constexpr, NCHUNK: tl.constexpr,
                             ALIGNED: tl.constexpr, IS_F64: tl.constexpr, IDX64: tl.constexpr):
    rows, rmask = _c662_rows(BLOCK_M, N, IDX64)
    cols = tl.arange(0, CHUNK)
    xb = X + _c662_roff(rows, stride_x, ALIGNED)[:, None] + cols[None, :]
    x0b = X0 + _c662_roff(rows, stride_x0, ALIGNED)[:, None] + cols[None, :]
    nb = XN + _c662_roff(rows, stride_xn, ALIGNED)[:, None] + cols[None, :]
    hb = H + _c662_roff(rows, stride_h, ALIGNED)[:, None] + cols[None, :]
    r = tl.load(R)
    l = tl.load(L)
    if not IS_F64:
        r = r.to(tl.float32)
        l = l.to(tl.float32)
    v0, _a0, _b0 = _c662_mix(xb, x0b, r, l, rmask, cols, 0, D, CHUNK, IS_F64)
    if NCHUNK > 1:
        v1, _a1, _b1 = _c662_mix(xb, x0b, r, l, rmask, cols, 1, D, CHUNK, IS_F64)
    if NCHUNK > 2:
        v2, _a2, _b2 = _c662_mix(xb, x0b, r, l, rmask, cols, 2, D, CHUNK, IS_F64)
    if NCHUNK > 3:
        v3, _a3, _b3 = _c662_mix(xb, x0b, r, l, rmask, cols, 3, D, CHUNK, IS_F64)
    _c662_store(nb, v0, rmask, cols, 0, D, CHUNK)
    sq = v0 * v0
    if NCHUNK > 1:
        _c662_store(nb, v1, rmask, cols, 1, D, CHUNK)
        sq += v1 * v1
    if NCHUNK > 2:
        _c662_store(nb, v2, rmask, cols, 2, D, CHUNK)
        sq += v2 * v2
    if NCHUNK > 3:
        _c662_store(nb, v3, rmask, cols, 3, D, CHUNK)
        sq += v3 * v3
    rstd = _c662_rsqrt(tl.sum(sq, axis=1) / D, eps, IS_F64)
    rs = rstd[:, None]
    _c662_store(hb, v0 * rs, rmask, cols, 0, D, CHUNK)
    if NCHUNK > 1:
        _c662_store(hb, v1 * rs, rmask, cols, 1, D, CHUNK)
    if NCHUNK > 2:
        _c662_store(hb, v2 * rs, rmask, cols, 2, D, CHUNK)
    if NCHUNK > 3:
        _c662_store(hb, v3 * rs, rmask, cols, 3, D, CHUNK)
    tl.store(RSTD + rows, rstd, mask=rmask)


@triton.jit
def _c662_mix_bwd(gxb, ghb, xb, x0b, r, l, rs, c_: tl.constexpr, rmask, cols, D: tl.constexpr,
                  CHUNK: tl.constexpr, IS_F64: tl.constexpr, HAS_GXN: tl.constexpr):
    v, a, b = _c662_mix(xb, x0b, r, l, rmask, cols, c_, D, CHUNK, IS_F64)
    g = _c662_load(ghb, rmask, cols, c_, D, CHUNK, IS_F64)
    if HAS_GXN:
        gx = _c662_load(gxb, rmask, cols, c_, D, CHUNK, IS_F64)
    else:
        gx = g * 0.0
    return v * rs, g, gx, a, b


@triton.jit
def _c662_mix_bwd_out(dxb, dx0b, xh, g, gx, a, b, r, l, rs, cc, rmask, cols, c_: tl.constexpr,
                      D: tl.constexpr, CHUNK: tl.constexpr):
    dxn = rs * (g - xh * cc) + gx
    _c662_store(dxb, r * dxn, rmask, cols, c_, D, CHUNK)
    _c662_store(dx0b, l * dxn, rmask, cols, c_, D, CHUNK)
    return dxn * a, dxn * b            # elementwise; reduced per row once all chunks are added


@triton.jit
def c662_rmsnorm_bwd_mixnorm(GXN, GH, X, X0, R, L, RSTD, DX, DX0, PR, PL, N,
                             stride_gxn, stride_gh, stride_x, stride_x0, stride_dx, stride_dx0,
                             D: tl.constexpr, BLOCK_M: tl.constexpr, CHUNK: tl.constexpr, NCHUNK: tl.constexpr,
                             HAS_GXN: tl.constexpr, ALIGNED: tl.constexpr, IS_F64: tl.constexpr,
                             IDX64: tl.constexpr):
    """dxn = g_xnew + rstd*(g_h - xhat*mean(g_h*xhat)), xhat = x_new*rstd (x_new recomputed bit-identically
    from x, x0, r, l); dx = r*dxn, dx0 = l*dxn; PR[row] = sum(dxn*x), PL[row] = sum(dxn*x0) over the
    row: per-row partials (N floats, summed by the wrapper) keep every reduction inside a warp -- per-program
    scalars needed 6 cross-warp shared-memory reductions (12 LDS/STS, 17 BAR in the sm_90 SASS).
    Masked rows contribute 0."""
    rows, rmask = _c662_rows(BLOCK_M, N, IDX64)
    cols = tl.arange(0, CHUNK)
    gxb = GXN + _c662_roff(rows, stride_gxn, ALIGNED)[:, None] + cols[None, :]
    ghb = GH + _c662_roff(rows, stride_gh, ALIGNED)[:, None] + cols[None, :]
    xb = X + _c662_roff(rows, stride_x, ALIGNED)[:, None] + cols[None, :]
    x0b = X0 + _c662_roff(rows, stride_x0, ALIGNED)[:, None] + cols[None, :]
    dxb = DX + _c662_roff(rows, stride_dx, ALIGNED)[:, None] + cols[None, :]
    dx0b = DX0 + _c662_roff(rows, stride_dx0, ALIGNED)[:, None] + cols[None, :]
    r = tl.load(R)
    l = tl.load(L)
    if not IS_F64:
        r = r.to(tl.float32)
        l = l.to(tl.float32)
    rs = tl.load(RSTD + rows, mask=rmask, other=0.0)[:, None]
    xh0, g0, gx0, a0, b0 = _c662_mix_bwd(gxb, ghb, xb, x0b, r, l, rs, 0, rmask, cols, D, CHUNK, IS_F64,
                                         HAS_GXN)
    if NCHUNK > 1:
        xh1, g1, gx1, a1, b1 = _c662_mix_bwd(gxb, ghb, xb, x0b, r, l, rs, 1, rmask, cols, D, CHUNK, IS_F64,
                                             HAS_GXN)
    if NCHUNK > 2:
        xh2, g2, gx2, a2, b2 = _c662_mix_bwd(gxb, ghb, xb, x0b, r, l, rs, 2, rmask, cols, D, CHUNK, IS_F64,
                                             HAS_GXN)
    if NCHUNK > 3:
        xh3, g3, gx3, a3, b3 = _c662_mix_bwd(gxb, ghb, xb, x0b, r, l, rs, 3, rmask, cols, D, CHUNK, IS_F64,
                                             HAS_GXN)
    p = g0 * xh0
    if NCHUNK > 1:
        p += g1 * xh1
    if NCHUNK > 2:
        p += g2 * xh2
    if NCHUNK > 3:
        p += g3 * xh3
    cc = (tl.sum(p, axis=1) / D)[:, None]
    pr, pl = _c662_mix_bwd_out(dxb, dx0b, xh0, g0, gx0, a0, b0, r, l, rs, cc, rmask, cols, 0, D, CHUNK)
    if NCHUNK > 1:
        p_r, p_l = _c662_mix_bwd_out(dxb, dx0b, xh1, g1, gx1, a1, b1, r, l, rs, cc, rmask, cols, 1, D, CHUNK)
        pr += p_r
        pl += p_l
    if NCHUNK > 2:
        p_r, p_l = _c662_mix_bwd_out(dxb, dx0b, xh2, g2, gx2, a2, b2, r, l, rs, cc, rmask, cols, 2, D, CHUNK)
        pr += p_r
        pl += p_l
    if NCHUNK > 3:
        p_r, p_l = _c662_mix_bwd_out(dxb, dx0b, xh3, g3, gx3, a3, b3, r, l, rs, cc, rmask, cols, 3, D, CHUNK)
        pr += p_r
        pl += p_l
    tl.store(PR + rows, tl.sum(pr, axis=1), mask=rmask)
    tl.store(PL + rows, tl.sum(pl, axis=1), mask=rmask)



@triton.jit
def c662_rmsnorm_bwd_mixnorm_finish(P, DR, DL, N, BLOCK: tl.constexpr):
    """dr = sum(P[0, :N]), dl = sum(P[1, :N]) (per-row partials from c662_rmsnorm_bwd_mixnorm), accumulated in
    fp64, one program each: one tiny deterministic launch instead of torch's cast + sum + cast per scalar."""
    which = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float64)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(P + which * N + offs, mask=offs < N, other=0.0).to(tl.float64)
    s = tl.sum(acc, axis=0)
    if which == 0:
        tl.store(DR, s.to(DR.dtype.element_ty))
    else:
        tl.store(DL, s.to(DL.dtype.element_ty))

# ------------------------------------------------------------------------------------------------
# Launch specs: one place that turns tensors into (kernel, grid, args, constexprs, options). Used by the
# launchers, --dump-isa (real GPU compile) and --xcompile (ASTSource, no GPU; meta tensors work).
# ------------------------------------------------------------------------------------------------
def _acc_dtype(dtype):
    return torch.float64 if dtype == torch.float64 else torch.float32


def _common(N, D, strides, dtype):
    """Constexprs shared by the tiled kernels. Plain Python ints/bools (under fake/symbolic tracing these
    guard on the concrete values, which is what a constexpr specialisation means anyway)."""
    D = int(D)
    chunk, nchunk = _chunking(D)
    return dict(D=D, CHUNK=chunk, NCHUNK=nchunk, ALIGNED=bool(all(s % 16 == 0 for s in strides)),
                IS_F64=dtype == torch.float64, IDX64=bool(N * max(max(strides), D) >= 2**31 - 1))


def spec_fwd(variant, x2, y2, rstd, eps=FP32_EPS):
    N, D = x2.shape[0], int(x2.shape[1])
    if variant == "v1":
        cfg = _cfg("v1", D)
        c = _common(N, D, (x2.stride(0), y2.stride(0)), x2.dtype)
        const = dict(D=c["D"], BLOCK_D=triton.next_power_of_2(c["D"]), ALIGNED=c["ALIGNED"], IS_F64=c["IS_F64"],
                     IDX64=c["IDX64"])
        args = dict(X=x2, Y=y2, RSTD=rstd, stride_x=x2.stride(0), stride_y=y2.stride(0), eps=eps)
        return c662_rmsnorm_fwd_v1, (N,), args, const, dict(num_warps=cfg["num_warps"])
    if variant != "v2":
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    cfg = _cfg("v2", D)
    const = _common(N, D, (x2.stride(0), y2.stride(0)), x2.dtype) | dict(BLOCK_M=cfg["BLOCK_M"])
    args = dict(X=x2, Y=y2, RSTD=rstd, N=N, stride_x=x2.stride(0), stride_y=y2.stride(0), eps=eps)
    return c662_rmsnorm_fwd_v2, (triton.cdiv(N, cfg["BLOCK_M"]),), args, const, dict(num_warps=cfg["num_warps"])


def spec_bwd(dy2, x2, rstd, dx2):
    N, D = x2.shape[0], int(x2.shape[1])
    cfg = _cfg("bwd", D)
    const = _common(N, D, (dy2.stride(0), x2.stride(0), dx2.stride(0)), x2.dtype) | dict(BLOCK_M=cfg["BLOCK_M"])
    args = dict(DY=dy2, X=x2, RSTD=rstd, DX=dx2, N=N, stride_dy=dy2.stride(0), stride_x=x2.stride(0),
                stride_dx=dx2.stride(0))
    return c662_rmsnorm_bwd, (triton.cdiv(N, cfg["BLOCK_M"]),), args, const, dict(num_warps=cfg["num_warps"])


def spec_mix_fwd(x2, x02, r, l, xn2, h2, rstd, eps=FP32_EPS):
    N, D = x2.shape[0], int(x2.shape[1])
    cfg = _cfg("mix_fwd", D)
    const = _common(N, D, (x2.stride(0), x02.stride(0), xn2.stride(0), h2.stride(0)), x2.dtype) | dict(
        BLOCK_M=cfg["BLOCK_M"])
    args = dict(X=x2, X0=x02, R=r, L=l, XN=xn2, H=h2, RSTD=rstd, N=N, stride_x=x2.stride(0),
                stride_x0=x02.stride(0), stride_xn=xn2.stride(0), stride_h=h2.stride(0), eps=eps)
    return (c662_rmsnorm_fwd_mixnorm, (triton.cdiv(N, cfg["BLOCK_M"]),), args, const,
            dict(num_warps=cfg["num_warps"]))


def spec_mix_bwd(gxn2, gh2, x2, x02, r, l, rstd, dx2, dx02, pr, pl):
    """gxn2 may be None (x_new unused downstream): HAS_GXN=False and gh2 is passed as a dummy pointer."""
    N, D = x2.shape[0], int(x2.shape[1])
    cfg = _cfg("mix_bwd", D)
    has_gxn = gxn2 is not None
    g = gxn2 if has_gxn else gh2
    const = _common(N, D, (g.stride(0), gh2.stride(0), x2.stride(0), x02.stride(0), dx2.stride(0), dx02.stride(0)),
                    x2.dtype) | dict(BLOCK_M=cfg["BLOCK_M"], HAS_GXN=has_gxn)
    args = dict(GXN=g, GH=gh2, X=x2, X0=x02, R=r, L=l, RSTD=rstd, DX=dx2, DX0=dx02, PR=pr, PL=pl, N=N,
                stride_gxn=g.stride(0), stride_gh=gh2.stride(0), stride_x=x2.stride(0), stride_x0=x02.stride(0),
                stride_dx=dx2.stride(0), stride_dx0=dx02.stride(0))
    return (c662_rmsnorm_bwd_mixnorm, (triton.cdiv(N, cfg["BLOCK_M"]),), args, const,
            dict(num_warps=cfg["num_warps"]))


def _wrap(kernel):
    """torch.library.wrap_triton for real Triton kernels (traceable inside triton_op); the interpreter's
    InterpretedFunction (TRITON_INTERPRET=1) is not a JITFunction and is launched directly."""
    from triton.runtime.jit import JITFunction
    if isinstance(kernel, JITFunction):
        return torch.library.wrap_triton(kernel)
    return kernel


def _launch(spec):
    kernel, grid, args, const, opts = spec
    return _wrap(kernel)[grid](**args, **const, **opts)


def _rows(x):
    """(N, D) row view of x (contiguous last dim; copies only if needed)."""
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    return x2


def _out_rows(x):
    return torch.empty(x.shape, dtype=x.dtype, device=x.device).view(-1, x.shape[-1])


# ------------------------------------------------------------------------------------------------
# Python launchers (plain tensors in, fresh tensors out; traceable by make_fx via wrap_triton)
# ------------------------------------------------------------------------------------------------
def rmsnorm_fwd(x: torch.Tensor, variant: str = "v2") -> Tuple[torch.Tensor, torch.Tensor]:
    x2 = _rows(x)
    y2 = _out_rows(x)
    rstd = torch.empty(x.shape[:-1], dtype=_acc_dtype(x.dtype), device=x.device)
    if x2.shape[0] > 0:
        _launch(spec_fwd(variant, x2, y2, rstd.view(-1)))
    return y2.view(x.shape), rstd


def rmsnorm_bwd(dy: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor) -> torch.Tensor:
    x2, dy2 = _rows(x), _rows(dy)
    dx2 = _out_rows(x)
    if x2.shape[0] > 0:
        _launch(spec_bwd(dy2, x2, rstd.reshape(-1), dx2))
    return dx2.view(x.shape)


def mixnorm_fwd(x: torch.Tensor, x0: torch.Tensor, r: torch.Tensor, l: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.shape == x0.shape and x.dtype == x0.dtype, "x and x0 must match"
    assert r.numel() == 1 and l.numel() == 1, "r and l are 0-d (scalar) tensors"
    x2, x02 = _rows(x), _rows(x0)
    xn2, h2 = _out_rows(x), _out_rows(x)
    rstd = torch.empty(x.shape[:-1], dtype=_acc_dtype(x.dtype), device=x.device)
    if x2.shape[0] > 0:
        _launch(spec_mix_fwd(x2, x02, r, l, xn2, h2, rstd.view(-1)))
    return xn2.view(x.shape), h2.view(x.shape), rstd


def mixnorm_bwd(g_xnew: Optional[torch.Tensor], g_h: torch.Tensor, x: torch.Tensor, x0: torch.Tensor,
                r: torch.Tensor, l: torch.Tensor, rstd: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x2, x02, gh2 = _rows(x), _rows(x0), _rows(g_h)
    gxn2 = None if g_xnew is None else _rows(g_xnew)
    dx2, dx02 = _out_rows(x), _out_rows(x)
    N, D = x2.shape[0], int(x2.shape[1])
    acc = _acc_dtype(x.dtype)
    dr = torch.empty(r.shape, dtype=r.dtype, device=x.device)
    dl = torch.empty(l.shape, dtype=l.dtype, device=x.device)
    if N == 0:
        return dx2.view(x.shape), dx02.view(x.shape), dr.zero_(), dl.zero_()
    part = torch.empty(2, N, dtype=acc, device=x.device)      # per-row partials of dr (row 0) and dl (row 1)
    _launch(spec_mix_bwd(gxn2, gh2, x2, x02, r, l, rstd.reshape(-1), dx2, dx02, part[0], part[1]))
    # the tiny reduction: 2 x N per-row partials (256 KB at R12) -> dr, dl, fp64 accumulation, no atomics
    _wrap(c662_rmsnorm_bwd_mixnorm_finish)[(2,)](part, dr, dl, N, BLOCK=FINISH_BLOCK, num_warps=8)
    return dx2.view(x.shape), dx02.view(x.shape), dr, dl


# ------------------------------------------------------------------------------------------------
# torch.library ops (G3: torch.compile traces through these; triton_op registers the function itself as
# the fake/meta impl, wrap_triton makes the launches visible to make_fx/Inductor) + autograd
# ------------------------------------------------------------------------------------------------
def _register_ops():
    from torch.library import triton_op

    @triton_op("cisc662::rmsnorm", mutates_args=())
    def rmsnorm_op(x: torch.Tensor, variant: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return rmsnorm_fwd(x, variant)

    @triton_op("cisc662::rmsnorm_bwd", mutates_args=())
    def rmsnorm_bwd_op(dy: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor) -> torch.Tensor:
        return rmsnorm_bwd(dy, x, rstd)

    def rms_setup(ctx, inputs, output):
        x, _ = inputs
        _, rstd = output
        ctx.save_for_backward(x, rstd)
        ctx.mark_non_differentiable(rstd)

    def rms_backward(ctx, gy, _grstd):
        x, rstd = ctx.saved_tensors
        return torch.ops.cisc662.rmsnorm_bwd(gy, x, rstd), None

    rmsnorm_op.register_autograd(rms_backward, setup_context=rms_setup)

    @triton_op("cisc662::mixnorm", mutates_args=())
    def mixnorm_op(x: torch.Tensor, x0: torch.Tensor, r: torch.Tensor, l: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mixnorm_fwd(x, x0, r, l)

    @triton_op("cisc662::mixnorm_bwd", mutates_args=())
    def mixnorm_bwd_op(g_xnew: Optional[torch.Tensor], g_h: torch.Tensor, x: torch.Tensor, x0: torch.Tensor,
                       r: torch.Tensor, l: torch.Tensor, rstd: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return mixnorm_bwd(g_xnew, g_h, x, x0, r, l, rstd)

    def mix_setup(ctx, inputs, output):
        x, x0, r, l = inputs
        rstd = output[2]
        ctx.save_for_backward(x, x0, r, l, rstd)
        ctx.mark_non_differentiable(rstd)

    def mix_backward(ctx, g_xnew, g_h, _grstd):
        x, x0, r, l, rstd = ctx.saved_tensors
        if g_h is None:
            g_h = torch.zeros_like(x)
        dx, dx0, dr, dl = torch.ops.cisc662.mixnorm_bwd(g_xnew, g_h, x, x0, r, l, rstd)
        return dx, dx0, dr, dl

    mixnorm_op.register_autograd(mix_backward, setup_context=mix_setup)


class RMSNormFn(torch.autograd.Function):
    """Eager autograd wrapper over the launchers (no dispatcher hop); impl="fn"."""

    @staticmethod
    def forward(ctx, x, variant="v2"):
        y, rstd = rmsnorm_fwd(x, variant)
        ctx.save_for_backward(x, rstd)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, rstd = ctx.saved_tensors
        return rmsnorm_bwd(gy, x, rstd), None


class MixNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x0, r, l):
        xn, h, rstd = mixnorm_fwd(x, x0, r, l)
        ctx.save_for_backward(x, x0, r, l, rstd)
        return xn, h

    @staticmethod
    def backward(ctx, g_xnew, g_h):
        x, x0, r, l, rstd = ctx.saved_tensors
        if g_h is None:
            g_h = torch.zeros_like(x)
        return mixnorm_bwd(g_xnew, g_h, x, x0, r, l, rstd)


def rmsnorm(x, variant="v2", impl="op"):
    """Drop-in for nanochat.gpt.norm (F.rms_norm(x, (x.size(-1),)), eps = fp32 epsilon)."""
    if impl == "op":
        return torch.ops.cisc662.rmsnorm(x, variant)[0]
    if impl == "fn":
        return RMSNormFn.apply(x, variant)
    raise ValueError(f"impl must be 'op' or 'fn', got {impl!r}")


def mixnorm(x, x0, r, l, impl="op"):
    """(x_new, h) = (r*x + l*x0, norm(x_new)); r, l 0-d tensors (nanochat: fp32 params)."""
    if impl == "op":
        xn, h, _ = torch.ops.cisc662.mixnorm(x, x0, r, l)
        return xn, h
    if impl == "fn":
        return MixNormFn.apply(x, x0, r, l)
    raise ValueError(f"impl must be 'op' or 'fn', got {impl!r}")


# ------------------------------------------------------------------------------------------------
# References (fp32/fp64 PyTorch)
# ------------------------------------------------------------------------------------------------
def reference(x, dtype=torch.float64, eps=FP32_EPS):
    """RMSNorm over the last dim computed entirely in `dtype` (the G1/G2 reference)."""
    xc = x.to(dtype)
    return xc * torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + eps)


def reference_mixnorm(x, x0, r, l, dtype=torch.float64, round_xnew_to=None, eps=FP32_EPS):
    """(x_new, h) in `dtype`. round_xnew_to=torch.bfloat16 rounds x_new to the stored activation dtype before
    the norm with a straight-through gradient (identity), i.e. nanochat's own function evaluated exactly:
    the model stores x_new in bf16 and normalises the stored value. Default: the unrounded fp64 chain."""
    xn = r.to(dtype) * x.to(dtype) + l.to(dtype) * x0.to(dtype)
    xs = xn
    if round_xnew_to is not None:
        xs = xn + (xn.to(round_xnew_to).to(dtype) - xn).detach()
    return xn, reference(xs, dtype=dtype, eps=eps)


# ------------------------------------------------------------------------------------------------
# ISA dumps: real GPU compile (--dump-isa) and GPU-less cross-compile (--xcompile)
# ------------------------------------------------------------------------------------------------
R12_SHAPES = {768: (16 * 2048, 768), 128: (16 * 2048 * 6, 128)}


def isa_specs(device, dtype=torch.bfloat16):
    """{variant_dir_name: spec} at the R12 shapes. device='meta' for the cross-compile."""
    out = {}
    for D, (N, _) in R12_SHAPES.items():
        x = torch.empty(N, D, dtype=dtype, device=device)
        y, dy, dx = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
        rstd = torch.empty(N, dtype=torch.float32, device=device)
        for v in VARIANTS:
            out[f"triton_{v}_{D}"] = spec_fwd(v, x, y, rstd)
        out[f"triton_bwd_{D}"] = spec_bwd(dy, x, rstd, dx)
    N, D = R12_SHAPES[768]
    x, x0, xn, h, gxn, gh, dx, dx0 = (torch.empty(N, D, dtype=dtype, device=device) for _ in range(8))
    r = torch.empty((), dtype=torch.float32, device=device)
    l = torch.empty((), dtype=torch.float32, device=device)
    rstd = torch.empty(N, dtype=torch.float32, device=device)
    out[f"triton_mixnorm_fwd_{D}"] = spec_mix_fwd(x, x0, r, l, xn, h, rstd)
    pr = torch.empty(N, dtype=torch.float32, device=device)
    pl = torch.empty(N, dtype=torch.float32, device=device)
    out[f"triton_mixnorm_bwd_{D}"] = spec_mix_bwd(gxn, gh, x, x0, r, l, rstd, dx, dx0, pr, pl)
    part = torch.empty(2, N, dtype=torch.float32, device=device)
    dr, dl = torch.empty_like(r), torch.empty_like(l)
    out[f"triton_mixnorm_bwd_finish_{D}"] = (c662_rmsnorm_bwd_mixnorm_finish, (2,),
                                             dict(P=part, DR=dr, DL=dl, N=N), dict(BLOCK=FINISH_BLOCK),
                                             dict(num_warps=8))
    return out


def sass_stats(sass):
    """Instruction counts the ISA card reports (load/store width, shuffles, MUFU.RSQ, local memory)."""
    pats = {"LDG.E.128": r"\bLDG\.E\.128\b", "LDG.E.64": r"\bLDG\.E\.64\b", "LDG.E(32)": r"\bLDG\.E(\.CONSTANT)?\s",
            "LDG.E.U16": r"\bLDG\.E\.U16\b", "STG.E.128": r"\bSTG\.E\.128\b", "STG.E.64": r"\bSTG\.E\.64\b",
            "STG.E(32)": r"\bSTG\.E\s", "STG.E.U16": r"\bSTG\.E\.U16\b", "SHFL": r"\bSHFL\.", "MUFU.RSQ": r"\bMUFU\.RSQ\b",
            "LDS": r"\bLDS\b", "STS": r"\bSTS\b", "BAR": r"\bBAR\.", "LDL": r"\bLDL\b", "STL": r"\bSTL\b",
            "F2FP.BF16": r"\bF2FP\.BF16"}
    return {k: len(re.findall(p, sass)) for k, p in pats.items()}


def _write_isa(dirpath, asm, meta):
    os.makedirs(dirpath, exist_ok=True)
    for key, fname in (("ptx", "kernel.ptx"), ("ttgir", "kernel.ttgir"), ("ttir", "kernel.ttir"),
                       ("sass", "kernel.sass")):
        try:
            data = asm[key]
        except Exception as e:  # sass needs cuobjdump/nvdisasm; record why it is missing
            meta.setdefault("missing", {})[key] = f"{type(e).__name__}: {e}"[:300]
            continue
        with open(os.path.join(dirpath, fname), "w") as f:
            f.write(data if isinstance(data, str) else data.decode())
        if key == "sass":
            meta["sass_stats"] = sass_stats(data if isinstance(data, str) else data.decode())
    with open(os.path.join(dirpath, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)


def _ptxas_verbose(ptx_text, arch="sm_90a"):
    """Registers / spills from Triton's bundled ptxas -v (what the driver would load)."""
    import tempfile
    from triton.backends.nvidia.compiler import get_ptxas
    ptxas = get_ptxas()
    exe = ptxas.path if hasattr(ptxas, "path") else ptxas[0]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.ptx")
        with open(p, "w") as f:
            f.write(ptx_text)
        r = subprocess.run([exe, f"-arch={arch}", "-v", p, "-o", os.path.join(d, "k.cubin")],
                           capture_output=True, text=True, timeout=120)
    txt = r.stdout + r.stderr
    m = re.search(r"Used (\d+) registers", txt)
    st = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", txt)
    sm = re.search(r"(\d+) bytes smem", txt)
    return {"ptxas_v": txt.strip()[-1500:], "n_regs": int(m.group(1)) if m else None,
            "spill_store_bytes": int(st.group(1)) if st else None,
            "spill_load_bytes": int(st.group(2)) if st else None,
            "smem_bytes_ptxas": int(sm.group(1)) if sm else None}


def _spec_meta(name, spec):
    kernel, grid, args, const, opts = spec
    return {"variant": name, "kernel": kernel.fn.__name__ if hasattr(kernel, "fn") else str(kernel),
            "grid": list(grid), "constexprs": const, "num_warps": opts.get("num_warps"),
            "shapes": {k: list(v.shape) for k, v in args.items() if torch.is_tensor(v)},
            "dtypes": {k: str(v.dtype).replace("torch.", "") for k, v in args.items() if torch.is_tensor(v)},
            "triton": triton.__version__, "torch": torch.__version__}


def dump_isa(outdir, arch_dir="sm_90a"):
    """GPU job: launch each variant once at the R12 shapes, write ptx/sass/ttgir + meta.json (n_regs,
    n_spills from the loaded module, shared, num_warps, num_stages, grid) per variant."""
    assert torch.cuda.is_available(), "--dump-isa needs a GPU (use --xcompile on the login node)"
    summary = {}
    specs = isa_specs("cuda")
    for name, spec in specs.items():
        # real data so the launch is valid (values irrelevant for the ISA)
        for k, v in spec[2].items():
            if torch.is_tensor(v) and v.is_floating_point():
                v.normal_() if k not in ("RSTD",) else v.fill_(1.0)
        kernel, grid, args, const, opts = spec
        ck = kernel[grid](**args, **const, **opts)
        torch.cuda.synchronize()
        md = ck.metadata._asdict() if hasattr(ck.metadata, "_asdict") else dict(ck.metadata)
        meta = _spec_meta(name, spec) | {
            "name": ck.name, "n_regs": ck.n_regs, "n_spills": ck.n_spills, "shared": md.get("shared"),
            "num_stages": md.get("num_stages"), "num_warps_compiled": md.get("num_warps"), "arch": md.get("arch"),
            "ptx_version": md.get("ptx_version"), "hash": ck.hash, "device": torch.cuda.get_device_name(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "mode": "gpu-compile"}
        try:
            meta |= {k: v for k, v in _ptxas_verbose(ck.asm["ptx"]).items() if k != "n_regs"}
        except Exception as e:
            meta["ptxas_error"] = repr(e)[:300]
        _write_isa(os.path.join(outdir, name, arch_dir), ck.asm, meta)
        summary[name] = {k: meta.get(k) for k in ("name", "n_regs", "n_spills", "shared", "num_warps", "grid",
                                                     "sass_stats")}
        print(name, json.dumps(summary[name], default=str), flush=True)
    with open(os.path.join(outdir, f"summary_{arch_dir}.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    return summary


def _signature(kernel, args, const):
    """ASTSource signature/attrs mirroring Triton's runtime specialisation (16-byte-aligned pointers,
    ints divisible by 16 get tt.divisibility=16)."""
    tmap = {torch.bfloat16: "*bf16", torch.float16: "*fp16", torch.float32: "*fp32", torch.float64: "*fp64"}
    sig, attrs = {}, {}
    for i, name in enumerate(kernel.arg_names):
        if name in const:
            sig[name] = "constexpr"
            continue
        v = args[name]
        if torch.is_tensor(v):
            sig[name] = tmap[v.dtype]
            attrs[(i,)] = [["tt.divisibility", 16]]
        elif isinstance(v, bool):
            sig[name] = "i1"
        elif isinstance(v, int):
            sig[name] = "i32" if abs(v) < 2**31 else "i64"
            if v % 16 == 0:
                attrs[(i,)] = [["tt.divisibility", 16]]
        elif isinstance(v, float):
            sig[name] = "fp32"
        else:
            raise TypeError(f"{name}: {type(v)}")
    return sig, attrs


def xcompile(outdir, arch_dir="sm_90a"):
    """Login node: compile every variant for GPUTarget('cuda', 90, 32) without a GPU and write the same
    files as dump_isa; registers/spills from the bundled ptxas -v. A few CPU-seconds per kernel."""
    from triton.backends.compiler import GPUTarget
    target = GPUTarget("cuda", 90, 32)
    summary = {}
    for name, spec in isa_specs("meta").items():
        kernel, grid, args, const, opts = spec
        sig, attrs = _signature(kernel, args, const)
        src = triton.compiler.ASTSource(fn=kernel, signature=sig, constexprs=dict(const), attrs=attrs)
        ck = triton.compile(src, target=target, options=dict(num_warps=opts["num_warps"]))
        md = ck.metadata._asdict()
        meta = _spec_meta(name, spec) | {"name": ck.name, "shared": md.get("shared"),
                                         "num_stages": md.get("num_stages"),
                                         "num_warps_compiled": md.get("num_warps"), "arch": md.get("arch"),
                                         "ptx_version": md.get("ptx_version"), "hash": ck.hash,
                                         "mode": "xcompile GPUTarget('cuda', 90, 32), no GPU"}
        meta |= _ptxas_verbose(ck.asm["ptx"])
        meta["n_spills"] = None if meta.get("spill_store_bytes") is None else (
            meta["spill_store_bytes"] + meta["spill_load_bytes"])
        _write_isa(os.path.join(outdir, name, arch_dir), ck.asm, meta)
        summary[name] = {k: meta.get(k) for k in ("name", "n_regs", "spill_store_bytes", "spill_load_bytes",
                                                     "shared", "num_warps", "grid", "sass_stats")}
        print(name, json.dumps(summary[name], default=str), flush=True)
    with open(os.path.join(outdir, f"summary_{arch_dir}_xcompile.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    return summary


# ------------------------------------------------------------------------------------------------
# Self-test at a small shape
# ------------------------------------------------------------------------------------------------
def _maxdiff(a, b):
    return float((a.double() - b.double()).abs().max())


def self_test(device=None, rows=(7, 9), dtype=torch.bfloat16):
    """Forward/backward of every variant and of mixnorm vs the fp64 reference at (rows[0], 768) and
    (rows[1], 128); prints max errors and asserts loose bf16-level bounds. Returns True on success."""
    import torch.nn.functional as F
    interp = os.environ.get("TRITON_INTERPRET") == "1"
    device = device or ("cuda" if torch.cuda.is_available() and not interp else "cpu")
    g = torch.Generator(device=device).manual_seed(0)
    ok = True

    def check(what, err, bound):
        nonlocal ok
        good = err <= bound
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} {what:<40s} max|err| {err:.3e} (bound {bound:.1e})")

    tol = 2e-2 if dtype in _LOW else 1e-5
    for n, D in ((rows[0], 768), (rows[1], 128)):
        x = (torch.randn(n, D, generator=g, device=device) * 0.5).to(dtype)
        gy = (torch.randn(n, D, generator=g, device=device) * 0.5).to(dtype)
        ref = reference(x)
        xr = x.double().requires_grad_(True)
        reference(xr).backward(gy.double())
        print(f"rmsnorm ({n},{D}) {dtype} on {device}{' [TRITON_INTERPRET]' if interp else ''}")
        for v in VARIANTS:
            for impl in ("op", "fn"):
                xx = x.clone().requires_grad_(True)
                y = rmsnorm(xx, v, impl=impl)
                y.backward(gy)
                check(f"{v}/{impl} fwd", _maxdiff(y, ref), tol * 4)
                check(f"{v}/{impl} bwd", _maxdiff(xx.grad, xr.grad), tol * 4)
        stock = F.rms_norm(x, (D,))
        check("stock F.rms_norm (sanity)", _maxdiff(stock, ref), tol * 4)
    n, D = rows[0], 768
    x, x0, gxn, gh = ((torch.randn(n, D, generator=g, device=device) * 0.5).to(dtype) for _ in range(4))
    r = torch.tensor(1.15, device=device)
    l = torch.tensor(0.20, device=device)
    leaves64 = [t.double().requires_grad_(True) for t in (x, x0, r, l)]
    xn_ref, h_ref = reference_mixnorm(*leaves64)
    xn_ref.retain_grad()                 # d loss / d x_new = g_xnew + norm backward (fp64)
    (xn_ref * gxn.double()).sum().add((h_ref * gh.double()).sum()).backward()
    # dr = sum(dxn*x), dl = sum(dxn*x0) are sums of random-sign terms that can land near 0: judge them
    # relative to max(|ref|, ||terms||_2) (as test_mixnorm_g2 does), not to |ref| alone
    scales = [float((xn_ref.grad * t.detach()).pow(2).sum().sqrt()) for t in leaves64[:2]]
    print(f"mixnorm ({n},{D}) {dtype}")
    for impl in ("op", "fn"):
        leaves = [t.clone().requires_grad_(True) for t in (x, x0, r, l)]
        xn, h = mixnorm(*leaves, impl=impl)
        torch.autograd.backward([xn, h], [gxn, gh])
        check(f"{impl} x_new", _maxdiff(xn, xn_ref), tol * 4)
        check(f"{impl} h", _maxdiff(h, h_ref), tol * 4)
        for nm, a, b in zip(("dx", "dx0"), leaves[:2], leaves64[:2]):
            check(f"{impl} {nm}", _maxdiff(a.grad, b.grad), tol * 4)
        for nm, a, b, sc in zip(("dr", "dl"), leaves[2:], leaves64[2:], scales):
            rel = abs(float(a.grad) - float(b.grad)) / max(abs(float(b.grad)), sc, 1e-30)
            # bf16: storing x_new in bf16 alone moves these cancelled sums by ~1e-2 vs the unrounded chain
            check(f"{impl} {nm} (relative)", rel, 5e-2 if dtype in _LOW else 1e-4)
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return ok


def _main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="KT-1 Triton RMSNorm/mixnorm: self-test and ISA dumps")
    p.add_argument("--dump-isa", metavar="OUTDIR", help="GPU: compile at the R12 shapes and dump ptx/sass/ttgir/meta")
    p.add_argument("--xcompile", metavar="OUTDIR", help="no GPU: cross-compile for cuda:90 and dump the same files")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="self-test dtype")
    a = p.parse_args(argv)
    if a.dump_isa:
        dump_isa(a.dump_isa)
        return 0
    if a.xcompile:
        xcompile(a.xcompile)
        return 0
    return 0 if self_test(dtype=getattr(torch, a.dtype)) else 1


if __name__ == "__main__":
    # run through the package module so the torch.library ops are registered exactly once per process
    import importlib
    sys.exit(importlib.import_module("cisc662.kernels.triton.rmsnorm")._main())
else:
    _register_ops()
