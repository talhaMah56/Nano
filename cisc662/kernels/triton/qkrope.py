"""
Fused QK-norm + RoPE + 1.2 scale for q and k in one launch (task card KT-2; kernel_id `qkrope`).

nanochat (gpt.py CausalSelfAttention.forward, lines 100-103) does, per q and k of shape (B, T, H, 128):
    x = apply_rotary_emb(x, cos, sin)   # bf16: y1 = x1*cos + x2*sin; y2 = -x1*sin + x2*cos  (halves of 64)
    x = norm(x)                         # F.rms_norm(x, (128,)), eps=None -> fp32 eps 1.19e-7, fp32 inside
    x = x * 1.2                         # bf16
with cos/sin the bf16 (1, T, 1, 64) slice of GPT.cos/GPT.sin (training: [:, 0:T]; decode with a KV cache:
[:, T0:T0+T], T=1 per generated token -- the kernel takes whatever slice it is given).

What this kernel computes (lead decision 2026-10-18, recorded in the KT-2 card; it overrides the card's
"rstd from the un-rotated row" identity): load the bf16 row, upcast to fp32, rotate in fp32 with the
fp32-upcast bf16 cos/sin (exactly nanochat's table values), take rstd = rsqrt(mean(r^2) + eps) of the
ROTATED fp32 row r, multiply by scale*rstd and round once to bf16. Same bytes as any fused form, nanochat's
order of operations, and more accurate than stock (which rounds after the rotation, after the norm and after
the scale). The un-rotated identity norm(rope(x)) == rope(norm(x)) would need cos^2+sin^2 == 1, which the
bf16 table violates by up to ~1e-2 relative (about one bf16 ulp of the output).

Backward (one kernel, recomputes r from the saved input x; saves x and rstd, never the output):
    g  = scale * dy
    r  = R x,  r^ = r * rstd
    dr = rstd * (g - r^ * mean(g * r^))                      # RMSNorm backward
    dx = R^T dr:  dx1 = dr1*cos - dr2*sin,  dx2 = dr1*sin + dr2*cos
R^T is the exact transpose of the forward map (no orthogonality needed), so the gradient is exact for the
function the forward computes (verified by fp64 gradcheck). No gradient for cos/sin (nanochat buffers).

fold_scale=True (training only): the kernel applies scale 1 (pure QK-norm+RoPE) and qkrope() returns
softmax_scale = 1.2**2 / sqrt(head_dim) to pass to the attention call, which is mathematically the same
logits: (1.2 q^).(1.2 k^)/sqrt(D) = q^.k^ * 1.44/sqrt(D). Do NOT fold in decode: nanochat stores the scaled
k in the KV cache, so the stored keys would change convention.

Kernels (names as nsys sees them): c662_qkrope_fwd, c662_qkrope_bwd. Grid = cdiv(B*T*Hq, BLOCK_M) programs
for the q rows followed by cdiv(B*T*Hk, BLOCK_M) programs for the k rows; each program handles BLOCK_M
(token, head) rows of 128 as two (BLOCK_M, 64) half tiles, so every lane issues 16-byte loads (head_dim
contiguous; pointers 16-byte aligned and row offsets multiples of 128 elements are what Triton's
divisibility specialisation sees -- checked in the SASS for LDG.E.128). rstd is stored fp32 (fp64 for fp64
inputs, for gradcheck), N*H*4 bytes per tensor.

Wrappers:
  qkrope(q, k, cos, sin, scale=1.2, fold_scale=False, impl="op")   public entry point
  QKRopeFunction                     torch.autograd.Function over the raw launchers (eager fast path)
  torch.ops.cisc662.qkrope           torch.library.triton_op (Inductor sees the kernel) + register_autograd;
                                     returns (yq, yk, rstd_q, rstd_k); the fake impl is the op body itself
                                     (triton_op registers it, the kernel launch is skipped on fake tensors)
  torch.ops.cisc662.qkrope_bwd       triton_op of the backward kernel
  reference(q, k, cos, sin, ...)     the same math in fp64 PyTorch (= cisc662.tests.ref_ops.qkrope fp64)

Run:
  python -m cisc662.kernels.triton.qkrope                      # self-test (CUDA, else TRITON_INTERPRET=1 on CPU)
  python -m cisc662.kernels.triton.qkrope --dump-isa OUTDIR    # ptx/sass/ttgir + meta.json per variant into
        OUTDIR/triton_<variant>/sm_90a/ at the R12 shape (16,2048,6,128); on a GPU node from real launches
        (n_regs/n_spills from the loaded cubin), without a GPU by cross-compiling for cuda:90 (ptxas -v)
"""
import math
import os
import sys

import torch

if __name__ == "__main__" and "--dump-isa" not in sys.argv and "TRITON_INTERPRET" not in os.environ \
        and not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"     # self-test on a CPU-only node; must precede @triton.jit

import triton
import triton.language as tl

QK_SCALE = 1.2
RMS_EPS = torch.finfo(torch.float32).eps    # what F.rms_norm(eps=None) uses for bf16/fp16/fp32 (torch 2.9.1)

# (BLOCK_M rows per program, num_warps). Elements per lane per half tile = BLOCK_M*64/(32*num_warps):
# v1 = 8 (one 16-byte load per half per lane); ilp2/ilp4 = 2/4 independent 16-byte loads per half per lane.
VARIANTS = {"v1": (16, 4), "ilp2": (32, 4), "ilp4": (64, 4)}
DEFAULT_VARIANT = "v1"


def _interpret():
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


# ----------------------------------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------------------------------
@triton.jit
def _round_bf16_rtne(v):
    """fp32 -> fp32 holding the round-to-nearest-even bf16 value. Interpreter only: TRITON_INTERPRET's
    fp32->bf16 cast truncates (triton 3.5.1 interpreter.cast_impl), the GPU cast is cvt.rn (RTNE)."""
    b = v.to(tl.uint32, bitcast=True)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _to_out(v, ty: tl.constexpr, EMU_RTNE: tl.constexpr):
    if EMU_RTNE:
        if ty == tl.bfloat16:
            v = _round_bf16_rtne(v)
    return v.to(ty)


@triton.jit
def _qkrope_fwd_rows(x_ptr, y_ptr, rstd_ptr, cos_ptr, sin_ptr, blk, n_rows, H, T, cs_stride, scale, eps,
                     D: tl.constexpr, BLOCK_M: tl.constexpr, F64: tl.constexpr, I64: tl.constexpr,
                     EMU_RTNE: tl.constexpr = False):
    HALF: tl.constexpr = D // 2
    if I64:
        rows = blk.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    else:
        rows = blk * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < n_rows
    tok = (rows // H) % T                                     # (b, t, h) row -> position t in the slice
    cols = tl.arange(0, HALF)
    x_off = rows[:, None] * D + cols[None, :]
    c_off = tok[:, None] * cs_stride + cols[None, :]
    m = rmask[:, None]
    x1 = tl.load(x_ptr + x_off, mask=m, other=0.0)
    x2 = tl.load(x_ptr + x_off + HALF, mask=m, other=0.0)
    c = tl.load(cos_ptr + c_off, mask=m, other=0.0)
    s = tl.load(sin_ptr + c_off, mask=m, other=0.0)
    if F64:
        x1, x2, c, s = x1.to(tl.float64), x2.to(tl.float64), c.to(tl.float64), s.to(tl.float64)
    else:
        x1, x2, c, s = x1.to(tl.float32), x2.to(tl.float32), c.to(tl.float32), s.to(tl.float32)
    # nanochat's rotation (rotates by -theta): y1 = x1 cos + x2 sin, y2 = -x1 sin + x2 cos
    r1 = x1 * c + x2 * s
    r2 = x2 * c - x1 * s
    ms = tl.sum(r1 * r1 + r2 * r2, axis=1) / D               # one row reduction over both halves
    if F64:
        rstd = 1.0 / tl.sqrt(ms + eps)          # fp64 + fp32 scalar -> fp64
        f = scale * rstd
    else:
        rstd = tl.math.rsqrt(ms + eps)
        f = scale * rstd
    tl.store(rstd_ptr + rows, rstd, mask=rmask)
    y_ty = y_ptr.dtype.element_ty
    tl.store(y_ptr + x_off, _to_out(r1 * f[:, None], y_ty, EMU_RTNE), mask=m)
    tl.store(y_ptr + x_off + HALF, _to_out(r2 * f[:, None], y_ty, EMU_RTNE), mask=m)


@triton.jit
def c662_qkrope_fwd(q_ptr, k_ptr, yq_ptr, yk_ptr, rq_ptr, rk_ptr, cos_ptr, sin_ptr,
                    n_rows_q, n_rows_k, n_blk_q, Hq, Hk, T, cs_stride, scale, eps,
                    D: tl.constexpr, BLOCK_M: tl.constexpr, F64: tl.constexpr, I64: tl.constexpr,
                     EMU_RTNE: tl.constexpr = False):
    """Programs [0, n_blk_q) normalise+rotate q rows, the rest k rows: q and k in ONE launch."""
    pid = tl.program_id(0)
    if pid < n_blk_q:
        _qkrope_fwd_rows(q_ptr, yq_ptr, rq_ptr, cos_ptr, sin_ptr, pid, n_rows_q, Hq, T, cs_stride, scale, eps,
                         D, BLOCK_M, F64, I64, EMU_RTNE)
    else:
        _qkrope_fwd_rows(k_ptr, yk_ptr, rk_ptr, cos_ptr, sin_ptr, pid - n_blk_q, n_rows_k, Hk, T, cs_stride,
                         scale, eps, D, BLOCK_M, F64, I64, EMU_RTNE)


@triton.jit
def _qkrope_bwd_rows(dy_ptr, x_ptr, rstd_ptr, dx_ptr, cos_ptr, sin_ptr, blk, n_rows, H, T, cs_stride, scale,
                     D: tl.constexpr, BLOCK_M: tl.constexpr, F64: tl.constexpr, I64: tl.constexpr,
                     EMU_RTNE: tl.constexpr = False):
    HALF: tl.constexpr = D // 2
    if I64:
        rows = blk.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    else:
        rows = blk * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < n_rows
    tok = (rows // H) % T
    cols = tl.arange(0, HALF)
    x_off = rows[:, None] * D + cols[None, :]
    c_off = tok[:, None] * cs_stride + cols[None, :]
    m = rmask[:, None]
    x1 = tl.load(x_ptr + x_off, mask=m, other=0.0)
    x2 = tl.load(x_ptr + x_off + HALF, mask=m, other=0.0)
    g1 = tl.load(dy_ptr + x_off, mask=m, other=0.0)
    g2 = tl.load(dy_ptr + x_off + HALF, mask=m, other=0.0)
    c = tl.load(cos_ptr + c_off, mask=m, other=0.0)
    s = tl.load(sin_ptr + c_off, mask=m, other=0.0)
    rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
    if F64:
        x1, x2, c, s = x1.to(tl.float64), x2.to(tl.float64), c.to(tl.float64), s.to(tl.float64)
        g1, g2, rstd = g1.to(tl.float64), g2.to(tl.float64), rstd.to(tl.float64)
        sc = scale
    else:
        x1, x2, c, s = x1.to(tl.float32), x2.to(tl.float32), c.to(tl.float32), s.to(tl.float32)
        g1, g2, rstd = g1.to(tl.float32), g2.to(tl.float32), rstd.to(tl.float32)
        sc = scale
    g1 = g1 * sc
    g2 = g2 * sc
    # recompute the rotated row and its normalised form
    rs = rstd[:, None]
    n1 = (x1 * c + x2 * s) * rs
    n2 = (x2 * c - x1 * s) * rs
    mdot = tl.sum(g1 * n1 + g2 * n2, axis=1) / D
    md = mdot[:, None]
    dr1 = rs * (g1 - n1 * md)
    dr2 = rs * (g2 - n2 * md)
    # transpose of the forward rotation
    dx1 = dr1 * c - dr2 * s
    dx2 = dr1 * s + dr2 * c
    dx_ty = dx_ptr.dtype.element_ty
    tl.store(dx_ptr + x_off, _to_out(dx1, dx_ty, EMU_RTNE), mask=m)
    tl.store(dx_ptr + x_off + HALF, _to_out(dx2, dx_ty, EMU_RTNE), mask=m)


@triton.jit
def c662_qkrope_bwd(dyq_ptr, dyk_ptr, q_ptr, k_ptr, rq_ptr, rk_ptr, dq_ptr, dk_ptr, cos_ptr, sin_ptr,
                    n_rows_q, n_rows_k, n_blk_q, Hq, Hk, T, cs_stride, scale,
                    D: tl.constexpr, BLOCK_M: tl.constexpr, F64: tl.constexpr, I64: tl.constexpr,
                     EMU_RTNE: tl.constexpr = False):
    """Single backward kernel for q and k: reads dy, x (the saved inputs), rstd, cos, sin; writes dx."""
    pid = tl.program_id(0)
    if pid < n_blk_q:
        _qkrope_bwd_rows(dyq_ptr, q_ptr, rq_ptr, dq_ptr, cos_ptr, sin_ptr, pid, n_rows_q, Hq, T, cs_stride, scale,
                         D, BLOCK_M, F64, I64, EMU_RTNE)
    else:
        _qkrope_bwd_rows(dyk_ptr, k_ptr, rk_ptr, dk_ptr, cos_ptr, sin_ptr, pid - n_blk_q, n_rows_k, Hk, T,
                         cs_stride, scale, D, BLOCK_M, F64, I64, EMU_RTNE)


# ----------------------------------------------------------------------------------------------
# Launchers
# ----------------------------------------------------------------------------------------------
_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


def _is_fake(t):
    from torch._subclasses.fake_tensor import FakeTensor
    return isinstance(t, FakeTensor) or t.device.type == "meta"


def _maybe_wrap(kernel, wrap):
    """wrap_triton (needed inside triton_op so make_fx/Inductor see the kernel); the interpreter's
    InterpretedFunction is not a JITFunction, so it is always launched directly."""
    if not wrap or _interpret():
        return kernel
    from torch.library import wrap_triton
    return wrap_triton(kernel)


def _table(t, T, half, name):
    """cos/sin as nanochat passes them: (1, T, 1, D/2) (a slice of the (1, 20480, 1, D/2) buffer), or (T, D/2).
    Returns (tensor with unit last stride, stride along T in elements)."""
    if t.dim() == 4:
        if not (t.shape[0] == 1 and t.shape[1] == T and t.shape[2] == 1 and t.shape[3] == half):
            raise ValueError(f"{name} must be (1, {T}, 1, {half}), got {tuple(t.shape)}")
        if t.stride(3) != 1:
            t = t.contiguous()
        return t, t.stride(1)
    if t.dim() == 2 and tuple(t.shape) == (T, half):
        if t.stride(1) != 1:
            t = t.contiguous()
        return t, t.stride(0)
    raise ValueError(f"{name} must be (1, T, 1, D/2) or (T, D/2), got {tuple(t.shape)}")


def _prep(q, k, cos, sin):
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("q and k must be (B, T, H, D)")
    B, T, Hq, D = q.shape
    if k.shape[0] != B or k.shape[1] != T or k.shape[3] != D:
        raise ValueError(f"k {tuple(k.shape)} does not match q {tuple(q.shape)} in B, T, D")
    if q.dtype != k.dtype or q.dtype not in _DTYPES:
        raise ValueError(f"q/k dtype must match and be one of {_DTYPES}, got {q.dtype}/{k.dtype}")
    if D < 4 or (D // 2) & (D // 2 - 1) or D % 2:
        raise ValueError(f"head_dim must be a power of two >= 4 (two power-of-two halves), got {D}")
    if q.device != k.device or cos.device != q.device or sin.device != q.device:
        raise ValueError("q, k, cos, sin must be on the same device")
    cos, cs_stride = _table(cos, T, D // 2, "cos")
    sin, ss_stride = _table(sin, T, D // 2, "sin")
    if cs_stride != ss_stride:          # one stride serves both tables: give them the same (T, D/2) layout
        cos, sin = cos.reshape(T, D // 2).contiguous(), sin.reshape(T, D // 2).contiguous()
        cs_stride = D // 2
    q, k = q.contiguous(), k.contiguous()
    return q, k, cos, sin, cs_stride, (B, T, Hq, k.shape[2], D)


def _cfg(variant):
    if isinstance(variant, tuple):
        return variant
    return VARIANTS[variant or DEFAULT_VARIANT]


def _launch_fwd(q, k, cos, sin, scale, variant=None, wrap=False):
    """Returns (yq, yk, rstd_q, rstd_k, compiled_kernel_or_None)."""
    q, k, cos, sin, cs_stride, (B, T, Hq, Hk, D) = _prep(q, k, cos, sin)
    block_m, num_warps = _cfg(variant)
    yq, yk = torch.empty_like(q), torch.empty_like(k)
    rdt = torch.float64 if q.dtype == torch.float64 else torch.float32
    rq = torch.empty((B, T, Hq), dtype=rdt, device=q.device)
    rk = torch.empty((B, T, Hk), dtype=rdt, device=q.device)
    nq, nk = B * T * Hq, B * T * Hk
    gq, gk = triton.cdiv(nq, block_m), triton.cdiv(nk, block_m)
    if nq + nk == 0 or (_is_fake(q) and _interpret()):
        return yq, yk, rq, rk, None
    i64 = max(nq, nk) * D + block_m * D >= 2**31
    ck = _maybe_wrap(c662_qkrope_fwd, wrap)[(gq + gk,)](
        q, k, yq, yk, rq, rk, cos, sin, nq, nk, gq, Hq, Hk, T, cs_stride, float(scale), RMS_EPS,
        D=D, BLOCK_M=block_m, F64=q.dtype == torch.float64, I64=i64, EMU_RTNE=_interpret(),
        num_warps=num_warps)
    return yq, yk, rq, rk, ck


def _launch_bwd(dyq, dyk, q, k, cos, sin, rq, rk, scale, variant=None, wrap=False):
    """Returns (dq, dk, compiled_kernel_or_None)."""
    q, k, cos, sin, cs_stride, (B, T, Hq, Hk, D) = _prep(q, k, cos, sin)
    block_m, num_warps = _cfg(variant)
    dyq = torch.zeros_like(q) if dyq is None else dyq.to(q.dtype).contiguous()
    dyk = torch.zeros_like(k) if dyk is None else dyk.to(k.dtype).contiguous()
    rq, rk = rq.contiguous(), rk.contiguous()
    dq, dk = torch.empty_like(q), torch.empty_like(k)
    nq, nk = B * T * Hq, B * T * Hk
    gq, gk = triton.cdiv(nq, block_m), triton.cdiv(nk, block_m)
    if nq + nk == 0 or (_is_fake(q) and _interpret()):
        return dq, dk, None
    i64 = max(nq, nk) * D + block_m * D >= 2**31
    ck = _maybe_wrap(c662_qkrope_bwd, wrap)[(gq + gk,)](
        dyq, dyk, q, k, rq, rk, dq, dk, cos, sin, nq, nk, gq, Hq, Hk, T, cs_stride, float(scale),
        D=D, BLOCK_M=block_m, F64=q.dtype == torch.float64, I64=i64, EMU_RTNE=_interpret(),
        num_warps=num_warps)
    return dq, dk, ck


# ----------------------------------------------------------------------------------------------
# autograd.Function (direct launches; the eager fast path)
# ----------------------------------------------------------------------------------------------
class QKRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, scale=QK_SCALE, variant=None):
        yq, yk, rq, rk, _ = _launch_fwd(q, k, cos, sin, scale, variant)
        ctx.save_for_backward(q, k, cos, sin, rq, rk)
        ctx.scale, ctx.variant = scale, variant
        return yq, yk

    @staticmethod
    def backward(ctx, gyq, gyk):
        q, k, cos, sin, rq, rk = ctx.saved_tensors
        dq, dk, _ = _launch_bwd(gyq, gyk, q, k, cos, sin, rq, rk, ctx.scale, ctx.variant)
        return dq, dk, None, None, None, None


# ----------------------------------------------------------------------------------------------
# torch.library.triton_op registration (compile integration, gate G3)
# ----------------------------------------------------------------------------------------------
def _registered(name):
    try:
        torch._C._dispatch_find_schema_or_throw(f"cisc662::{name}", "")
        return True
    except RuntimeError:
        return False


if not _registered("qkrope"):
    from torch.library import triton_op

    @triton_op("cisc662::qkrope", mutates_args=())
    def _qkrope_op(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        yq, yk, rq, rk, _ = _launch_fwd(q, k, cos, sin, scale, wrap=True)
        return yq, yk, rq, rk

    @triton_op("cisc662::qkrope_bwd", mutates_args=())
    def _qkrope_bwd_op(dyq: torch.Tensor, dyk: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                       cos: torch.Tensor, sin: torch.Tensor, rq: torch.Tensor, rk: torch.Tensor,
                       scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        dq, dk, _ = _launch_bwd(dyq, dyk, q, k, cos, sin, rq, rk, scale, wrap=True)
        return dq, dk

    def _setup_context(ctx, inputs, output):
        q, k, cos, sin, scale = inputs
        _, _, rq, rk = output
        ctx.save_for_backward(q, k, cos, sin, rq, rk)
        ctx.scale = scale
        ctx.mark_non_differentiable(rq, rk)

    def _backward(ctx, gyq, gyk, _grq, _grk):
        q, k, cos, sin, rq, rk = ctx.saved_tensors
        if gyq is None:
            gyq = torch.zeros_like(q)
        if gyk is None:
            gyk = torch.zeros_like(k)
        dq, dk = torch.ops.cisc662.qkrope_bwd(gyq, gyk, q, k, cos, sin, rq, rk, ctx.scale)
        return dq, dk, None, None, None

    _qkrope_op.register_autograd(_backward, setup_context=_setup_context)


# ----------------------------------------------------------------------------------------------
# Public API and reference
# ----------------------------------------------------------------------------------------------
def softmax_scale_folded(head_dim, scale=QK_SCALE):
    """The attention softmax scale that absorbs the q and k scale factors: scale^2 / sqrt(head_dim)."""
    return scale * scale / math.sqrt(head_dim)


def qkrope(q, k, cos, sin, scale=QK_SCALE, fold_scale=False, impl="op", variant=None):
    """nanochat's q/k chain RoPE -> RMSNorm -> *scale, fused (q and k in one launch).

    q, k     (B, T, Hq|Hk, D) bf16 (fp16/fp32/fp64 also accepted), head_dim contiguous
    cos/sin  (1, T, 1, D/2) slices of nanochat's table (decode: cos[:, T0:T0+T])
    impl     "op" = torch.ops.cisc662.qkrope (triton_op; default, compile-visible) or
             "function" = QKRopeFunction (direct launches; `variant` selects the tile config)
    Returns (yq, yk); with fold_scale=True the kernel applies no scale and the result is
    (yq, yk, softmax_scale) with softmax_scale = scale^2/sqrt(D) for the attention call (training only).
    """
    s = 1.0 if fold_scale else float(scale)
    if impl == "op":
        if variant not in (None, DEFAULT_VARIANT):
            raise ValueError("the triton_op uses the default variant; use impl='function' for others")
        yq, yk, _, _ = torch.ops.cisc662.qkrope(q, k, cos, sin, s)
    elif impl == "function":
        yq, yk = QKRopeFunction.apply(q, k, cos, sin, s, variant)
    else:
        raise ValueError(f"impl must be 'op' or 'function', got {impl!r}")
    if fold_scale:
        return yq, yk, softmax_scale_folded(q.shape[-1], scale)
    return yq, yk


def reference(q, k, cos, sin, scale=QK_SCALE, dtype=torch.float64, return_rstd=False):
    """The kernel's math in `dtype` (default fp64) PyTorch: rotate (nanochat's convention) with the upcast
    table, rstd of the rotated row with the fp32 eps, * scale. Output in `dtype`.
    Equal to cisc662.tests.ref_ops.qkrope(..., dtype=torch.float64) for dtype=float64."""
    def one(x):
        x, c, s = x.to(dtype), cos.to(dtype), sin.to(dtype)
        if c.dim() == 2:
            c, s = c[None, :, None, :], s[None, :, None, :]
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        r = torch.cat([x1 * c + x2 * s, x2 * c - x1 * s], -1)
        rstd = torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return r * rstd * scale, rstd.squeeze(-1)
    (yq, rq), (yk, rk) = one(q), one(k)
    return (yq, yk, rq, rk) if return_rstd else (yq, yk)


# ----------------------------------------------------------------------------------------------
# ISA dump
# ----------------------------------------------------------------------------------------------
R12_SHAPE = (16, 2048, 6, 128)


def _sass_counts(sass):
    import re
    keys = {"LDG.E.128": r"LDG\.E\.128\b", "LDG.E.64": r"LDG\.E\.64\b", "LDG.E (32-bit or narrower)":
            r"LDG\.E(\.U16|\.S16|\.U8|\.S8)?\s", "LDG total": r"\bLDG\b", "STG.E.128": r"STG\.E\.128\b",
            "STG.E.64": r"STG\.E\.64\b", "STG total": r"\bSTG\b", "SHFL": r"\bSHFL\b",
            "MUFU.RSQ": r"MUFU\.RSQ\b", "BAR": r"\bBAR\b", "STS": r"\bSTS\b", "LDS": r"\bLDS\b",
            # Triton's disasm prints "<control codes>\t<instruction>;" per line
            "instructions": r"^[-0-9a-fA-F]{2}:\S*\t\S"}
    return {k: len(re.findall(v, sass, flags=re.M)) for k, v in keys.items()}


def _r12_inputs(device, dtype=torch.bfloat16):
    B, T, H, D = R12_SHAPE
    try:
        from nanochat.gpt import GPT
        cos, sin = GPT._precompute_rotary_embeddings(None, 10 * T, D, device=torch.device(device))
    except Exception:  # pragma: no cover - nanochat import problems; same formula
        t = torch.arange(10 * T, dtype=torch.float32, device=device)
        inv = 1.0 / (100000 ** (torch.arange(0, D, 2, dtype=torch.float32, device=device) / D))
        f = torch.outer(t, inv)
        cos, sin = f.cos().to(dtype)[None, :, None, :], f.sin().to(dtype)[None, :, None, :]
    q = torch.randn(R12_SHAPE, device=device).to(dtype)
    k = torch.randn(R12_SHAPE, device=device).to(dtype)
    return q, k, cos[:, :T].to(dtype), sin[:, :T].to(dtype)


def _write_dump(outdir, variant, name, asm, meta):
    import json
    d = os.path.join(outdir, f"triton_{variant}", "sm_90a")
    os.makedirs(d, exist_ok=True)
    for st in ("ttir", "ttgir", "llir", "ptx", "sass"):
        if st in asm and asm[st]:
            with open(os.path.join(d, f"{name}.{st}"), "w") as f:
                f.write(asm[st])
    if "sass" in asm and asm["sass"]:
        meta["sass_counts"] = _sass_counts(asm["sass"])
    path = os.path.join(d, f"{name}.meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return path


def _ptxas_resources(ptx):
    """Registers/spills from Triton's bundled ptxas -v (what the launch would report) for a cross-compile."""
    import re
    import subprocess
    import tempfile
    ptxas = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "k.ptx")
        with open(p, "w") as f:
            f.write(ptx)
        out = subprocess.run([ptxas, "-arch=sm_90a", "-v", p, "-o", os.path.join(td, "k.cubin")],
                             capture_output=True, text=True, timeout=120)
    txt = out.stdout + out.stderr
    regs = re.search(r"Used (\d+) registers", txt)
    st = re.search(r"(\d+) bytes spill stores", txt)
    ld = re.search(r"(\d+) bytes spill loads", txt)
    ver = subprocess.run([ptxas, "--version"], capture_output=True, text=True).stdout.strip().splitlines()
    return {"n_regs": int(regs.group(1)) if regs else None,
            "spill_store_bytes": int(st.group(1)) if st else None,
            "spill_load_bytes": int(ld.group(1)) if ld else None,
            "n_spills": (int(st.group(1)) if st else 0) + (int(ld.group(1)) if ld else 0) if regs else None,
            "ptxas_v": txt.strip(), "ptxas_version": ver[-1] if ver else None}


def _xcompile(kernel, signature, constexprs, attrs, num_warps):
    from triton.backends.compiler import GPUTarget
    src = triton.compiler.ASTSource(fn=kernel, signature=signature, constexprs=constexprs, attrs=attrs)
    return triton.compile(src, target=GPUTarget("cuda", 90, 32), options=dict(num_warps=num_warps))


def dump_isa(outdir, variants=None, mode="auto"):
    """Writes <outdir>/triton_<variant>/sm_90a/{c662_qkrope_fwd,c662_qkrope_bwd}.{ttir,ttgir,llir,ptx,sass,meta.json}
    for the R12 shape. mode: "launch" (GPU; real launches, n_regs/n_spills from the loaded cubin),
    "xcompile" (no GPU: ASTSource + GPUTarget('cuda', 90, 32) with the divisibility attributes a launch at
    this shape specialises on; registers/spills from ptxas -v) or "auto"."""
    import json
    if mode == "auto":
        mode = "launch" if torch.cuda.is_available() else "xcompile"
    variants = variants or list(VARIANTS)
    B, T, H, D = R12_SHAPE
    n = B * T * H
    written = []
    if mode == "launch":
        q, k, cos, sin = _r12_inputs("cuda")
        dy = torch.randn_like(q)
    for v in variants:
        bm, nw = VARIANTS[v]
        gq = triton.cdiv(n, bm)
        grid = [2 * gq, 1, 1]
        for name in ("c662_qkrope_fwd", "c662_qkrope_bwd"):
            meta = {"kernel": name, "variant": v, "BLOCK_M": bm, "num_warps": nw, "grid": grid,
                    "shape": {"q": list(R12_SHAPE), "k": list(R12_SHAPE), "cos": [1, T, 1, D // 2]},
                    "dtype": "bfloat16", "triton": triton.__version__, "torch": torch.__version__,
                    "target": "cuda:90 (sm_90a)", "mode": mode}
            if mode == "launch":
                if name == "c662_qkrope_fwd":
                    yq, yk, rq, rk, ck = _launch_fwd(q, k, cos, sin, QK_SCALE, v)
                else:
                    _, _, ck = _launch_bwd(dy, dy, q, k, cos, sin, rq, rk, QK_SCALE, v)
                torch.cuda.synchronize()
                ck._init_handles()
                meta.update(n_regs=ck.n_regs, n_spills=ck.n_spills, device=torch.cuda.get_device_name(),
                            shared=ck.metadata.shared, num_stages=ck.metadata.num_stages, hash=ck.hash,
                            job_id=os.environ.get("SLURM_JOB_ID"))
            else:
                ptr = "*bf16"
                d16 = [["tt.divisibility", 16]]
                if name == "c662_qkrope_fwd":
                    sig = {a: ptr for a in ("q_ptr", "k_ptr", "yq_ptr", "yk_ptr")}
                    sig.update(rq_ptr="*fp32", rk_ptr="*fp32", cos_ptr=ptr, sin_ptr=ptr)
                    ints = ["n_rows_q", "n_rows_k", "n_blk_q", "Hq", "Hk", "T", "cs_stride"]
                    vals = dict(n_rows_q=n, n_rows_k=n, n_blk_q=gq, Hq=H, Hk=H, T=T, cs_stride=D // 2)
                    sig.update({a: "i32" for a in ints})
                    sig.update(scale="fp32", eps="fp32")
                    kern = c662_qkrope_fwd
                else:
                    sig = {a: ptr for a in ("dyq_ptr", "dyk_ptr", "q_ptr", "k_ptr")}
                    sig.update(rq_ptr="*fp32", rk_ptr="*fp32", dq_ptr=ptr, dk_ptr=ptr, cos_ptr=ptr, sin_ptr=ptr)
                    ints = ["n_rows_q", "n_rows_k", "n_blk_q", "Hq", "Hk", "T", "cs_stride"]
                    vals = dict(n_rows_q=n, n_rows_k=n, n_blk_q=gq, Hq=H, Hk=H, T=T, cs_stride=D // 2)
                    sig.update({a: "i32" for a in ints})
                    sig.update(scale="fp32")
                    kern = c662_qkrope_bwd
                sig.update(D="constexpr", BLOCK_M="constexpr", F64="constexpr", I64="constexpr", EMU_RTNE="constexpr")
                names = list(sig)
                # what a launch specialises on: 16-byte-aligned pointers, ints divisible by 16
                attrs = {(names.index(a),): d16 for a, t in sig.items() if t.startswith("*")}
                attrs.update({(names.index(a),): d16 for a in ints if vals[a] % 16 == 0})
                ck = _xcompile(kern, sig, dict(D=D, BLOCK_M=bm, F64=False, I64=False, EMU_RTNE=False), attrs, nw)
                res = _ptxas_resources(ck.asm["ptx"])
                meta.update(res, shared=ck.metadata.shared, num_stages=ck.metadata.num_stages, hash=ck.hash,
                            divisibility_16_args=[a for a in names if (names.index(a),) in attrs])
            asm = {st: ck.asm[st] for st in ("ttir", "ttgir", "llir", "ptx") if st in ck.asm}
            try:
                asm["sass"] = ck.asm["sass"]
            except Exception as e:     # cuobjdump missing
                meta["sass_error"] = f"{type(e).__name__}: {e}"
            path = _write_dump(outdir, v, name, asm, meta)
            written.append(path)
            sc = meta.get("sass_counts", {})
            print(f"{v:5s} {name:16s} regs={meta.get('n_regs')} spills={meta.get('n_spills')} "
                  f"LDG.E.128={sc.get('LDG.E.128')} LDG={sc.get('LDG total')} STG.E.128={sc.get('STG.E.128')} "
                  f"SHFL={sc.get('SHFL')} MUFU.RSQ={sc.get('MUFU.RSQ')} -> {path}", flush=True)
    with open(os.path.join(outdir, f"summary_{mode}.json"), "w") as f:
        json.dump({"written": written, "mode": mode, "triton": triton.__version__}, f, indent=1)
    return written


# ----------------------------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------------------------
def _self_test():
    dev = "cuda" if torch.cuda.is_available() and not _interpret() else "cpu"
    if dev == "cpu" and not _interpret():
        print("no CUDA: run with TRITON_INTERPRET=1", file=sys.stderr)
        return 2
    torch.manual_seed(0)
    from nanochat.gpt import GPT, apply_rotary_emb, norm
    cos_all, sin_all = GPT._precompute_rotary_embeddings(None, 20480, 128, device=torch.device(dev))
    # nanochat's COMPUTE_DTYPE is bf16 on CUDA (fp32 on CPU); the table under test is the bf16 one
    cos_all, sin_all = cos_all.to(torch.bfloat16), sin_all.to(torch.bfloat16)
    ok = True

    def stock(q, k, cos, sin):
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        return norm(q) * 1.2, norm(k) * 1.2

    for (B, T, Hq, Hk, T0) in ((1, 5, 2, 2, 0), (2, 1, 2, 1, 1234), (1, 3, 3, 3, 2045)):
        cos, sin = cos_all[:, T0:T0 + T], sin_all[:, T0:T0 + T]
        q = (torch.randn(B, T, Hq, 128, device=dev) * 0.5).bfloat16()
        k = (torch.randn(B, T, Hk, 128, device=dev) * 0.5).bfloat16()
        for impl in ("op", "function"):
            yq, yk = qkrope(q, k, cos, sin, impl=impl)
            rq, rk = reference(q, k, cos, sin)
            sq, sk = stock(q, k, cos, sin)
            for name, y, r, s in (("q", yq, rq, sq), ("k", yk, rk, sk)):
                e, es = (y.double() - r).abs().max().item(), (s.double() - r).abs().max().item()
                good = e <= max(2 * es, 1e-2)
                ok &= good
                print(f"fwd B={B} T={T} Hq={Hq} Hk={Hk} T0={T0} {impl:8s} {name}: max|err| {e:.3e} "
                      f"(stock {es:.3e}) {'ok' if good else 'FAIL'}")
            # backward vs fp64 autograd of the reference
            gq, gk = torch.randn_like(q), torch.randn_like(k)
            qa, ka = q.clone().requires_grad_(), k.clone().requires_grad_()
            oq, ok_ = qkrope(qa, ka, cos, sin, impl=impl)
            torch.autograd.backward([oq, ok_], [gq, gk])
            q64, k64 = q.double().requires_grad_(), k.double().requires_grad_()
            r64 = reference(q64, k64, cos, sin)
            torch.autograd.backward(list(r64), [gq.double(), gk.double()])
            for name, g, r in (("dq", qa.grad, q64.grad), ("dk", ka.grad, k64.grad)):
                e = (g.double() - r).abs().max().item()
                rel = e / r.abs().max().item()
                good = rel < 2e-2
                ok &= good
                print(f"bwd B={B} T={T} T0={T0} {impl:8s} {name}: max|err| {e:.3e} rel-to-max {rel:.2e} "
                      f"{'ok' if good else 'FAIL'}")
    # fp64 gradcheck, both paths
    a = torch.rand(1, 3, 1, 4, dtype=torch.float64, device=dev) * 6.28
    c, s = a.cos(), a.sin()
    qd = torch.randn(1, 3, 2, 8, dtype=torch.float64, device=dev, requires_grad=True)
    kd = torch.randn(1, 3, 1, 8, dtype=torch.float64, device=dev, requires_grad=True)
    for impl in ("op", "function"):
        g = torch.autograd.gradcheck(lambda x, y: qkrope(x, y, c, s, impl=impl), (qd, kd))
        ok &= bool(g)
        print(f"gradcheck fp64 {impl}: {g}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _main(argv):
    import argparse
    p = argparse.ArgumentParser(description="KT-2 fused qkrope: self-test or ISA dump")
    p.add_argument("--dump-isa", metavar="OUTDIR", default=None)
    p.add_argument("--mode", default="auto", choices=("auto", "launch", "xcompile"))
    p.add_argument("--variants", default=",".join(VARIANTS))
    a = p.parse_args(argv)
    if a.dump_isa:
        dump_isa(a.dump_isa, [v for v in a.variants.split(",") if v], a.mode)
        return 0
    return _self_test()


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
