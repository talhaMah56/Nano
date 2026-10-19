"""
Fused logit softcap + cross-entropy (task card KT-3; kernel_id `ce`).

nanochat (gpt.py GPT.forward, lines 511-520), on lm_head's bf16 output of shape (B, T, V):
    logits = logits.float()                      # (N, V) fp32 copy, N = B*T
    logits = 15 * torch.tanh(logits / 15)        # softcap
    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1, reduction=...)
Eager runs this as ~13 HBM passes over the 4.3 GB fp32 logits; torch.compile fuses it into two per-row
reduction kernels (online softmax forward, recompute backward, dlogits written in place) that move the
minimum bytes but reach only 30% (fwd) / 62% (bwd) of the HBM roof (BAS-1, job 3406171).

Key fact this kernel exploits: the softcap bounds every logit, |z| <= 15, so exp(z) <= 3.3e6 and a row sum
over V <= 2^20 columns stays far below the fp32 range. **No running max and no online-softmax rescale are
needed**: one pass, one exp per element, shifted by the target's z (loss = log1p(sum_{j!=t} exp(z_j - z_t)),
which stays accurate when the target saturates the softcap and the loss is ~1e-3) (Inductor's online softmax pays an extra
exp per element for the rescale). On GH200 the tail is close to MUFU-bound (tanh = EX2 + RCP, exp = EX2;
MUFU issues 16/clk/SM), so transcendentals per element matter as much as bytes.

Forms (both compute the same function; card KT-3 step 3):
  rec  (form ii, the default and the torch.library op): forward reads the bf16 logits once and stores the
       per-row loss (fp32) and lse2 = log2(sum exp z) (fp32); backward re-reads the logits, recomputes the
       softcap and writes bf16 dlogits = (softmax(z) - onehot(t)) * (1 - tanh^2) * g_row. 6.4 GB per step
       at R12 (same bytes as Inductor), no host sync, compile-safe (gate G3 later).
  gif  (form i, "gradient in forward", eager only): one kernel computes the loss AND writes dlogits scaled
       for the reduction (1/n_valid for mean), 4.3 GB. Autograd still has to apply the incoming grad_out:
       backward returns dlogits as-is when grad_out == 1 (checked with a host sync, as Liger does) and
       scales it otherwise (another 4.3 GB pass). Variants: gif_tile keeps the whole row in registers
       (V <= 32768, 32 warps), gif_2pass streams it twice (pass 2 hits L2 or HBM, recomputes the softcap).
  approx: TANH_APPROX=True swaps libdevice tanh for PTX tanh.approx.f32 (one MUFU.TANH, abs err ~5e-4).
       Benchmarked only as an instruction-count ablation (card KT-3 step 2), never gated or used for results.
       (The card expected it to miss the 1e-5 loss gate by ~100x; job 3417307 measured 8.3e-6 per-row at
       R12-like logits -- the tanh error mostly averages out in the log-sum-exp. See findings.md#ce.)

Ignored rows (target == -1): loss 0, dlogits exactly 0, excluded from n_valid. reduction="mean" with every
row ignored gives NaN (0/0) like F.cross_entropy, with an all-zero gradient.

Kernels (names as nsys sees them): c662_ce_fwd, c662_ce_bwd, c662_ce_gif. One program per row.

Wrappers:
  softcap_ce(logits, targets, reduction="mean", impl="op"|"gif", variant=None, cap=15.0)  public entry point
  torch.ops.cisc662.ce_fwd(logits, targets, cap) -> (loss_rows fp32 (N,), lse2 fp32 (N,))   triton_op
  torch.ops.cisc662.ce_bwd(logits, targets, lse2, grad_rows, cap) -> dlogits                triton_op
  CEGifFunction                     autograd.Function over c662_ce_gif (eager)
  reference(logits, targets, ...)   = cisc662.tests.ref_ops.softcap_ce in fp64

Run:
  python -m cisc662.kernels.triton.ce                      # self-test (CUDA, else TRITON_INTERPRET=1 on CPU)
  python -m cisc662.kernels.triton.ce --dump-isa OUTDIR    # ptx/sass/ttgir + meta.json per variant into
        OUTDIR/triton_<variant>/sm_90a/ at the R12 shape (32768 x 32768 bf16)
"""
import os
import sys

import torch

if __name__ == "__main__" and "--dump-isa" not in sys.argv and "TRITON_INTERPRET" not in os.environ \
        and not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"     # self-test on a CPU-only node; must precede @triton.jit

import triton
import triton.language as tl
from triton.language.extra import libdevice

SOFTCAP = 15.0
IGNORE_INDEX = -1
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453

# variant -> (form, BLOCK_V or None = next_power_of_2(V), num_warps, tanh_approx)
VARIANTS = {
    "rec": ("rec", 4096, 8, False),          # default: form (ii)
    "rec_2k": ("rec", 2048, 4, False),
    "rec_8k": ("rec", 8192, 16, False),
    "gif_tile": ("gif", None, 32, False),    # whole row in registers (V <= 32768)
    "gif_2pass": ("gif", 4096, 8, False),
    "rec_approx": ("rec", 4096, 8, True),    # ablation only (fails G1 by design)
}
DEFAULT_VARIANT = "rec"
MAX_TILE_V = 32768


def _interpret():
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


# TRITON_INTERPRET does not implement libdevice.tanh (returns None): interpreter runs use an exp-based tanh.
# Fixed at import, like the interpreter itself; the GPU always uses libdevice (or tanh.approx for the ablation).
_INTERP_TANH = tl.constexpr(_interpret())


# ----------------------------------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------------------------------
@triton.jit
def _round_bf16_rtne(v):
    """fp32 -> fp32 holding the round-to-nearest-even bf16 value. Interpreter only: TRITON_INTERPRET's
    fp32->bf16 cast truncates (triton 3.5.1), the GPU cast is cvt.rn (RTNE). Same helper as qkrope.py."""
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
def _softcap(l, cap, F64: tl.constexpr, TANH_APPROX: tl.constexpr):
    """th = tanh(l / cap), z = cap * th. The SAME expressions are used by every kernel that recomputes z,
    so forward and backward round identically (lesson from the mixnorm FMA-contraction bug)."""
    if _INTERP_TANH:
        if F64:
            x = l.to(tl.float64) / cap
        else:
            x = l.to(tl.float32) / cap
        a = tl.abs(x)
        th = 1.0 - 2.0 / (tl.exp(2.0 * a) + 1.0)
        th = tl.where(x < 0, -th, th)
    elif F64:
        th = libdevice.tanh(l.to(tl.float64) / cap)
    else:
        x = l.to(tl.float32) / cap
        if TANH_APPROX:
            th = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32,
                                           is_pure=True, pack=1)
        else:
            th = libdevice.tanh(x)
    return th, th * cap


@triton.jit
def _exp(z, F64: tl.constexpr):
    if F64:
        return tl.exp(z)
    return tl.exp2(z * 1.4426950408889634)


@triton.jit
def _log1p(x):
    """Kahan's log1p: accurate for small x without libdevice (the interpreter lacks it)."""
    u = 1.0 + x
    return tl.where(u == 1.0, x, tl.log(u) * (x / (u - 1.0)))


@triton.jit
def _row_sum_exp_excl(base, V, cap, shift, excl, BLOCK_V: tl.constexpr, F64: tl.constexpr,
                      TANH_APPROX: tl.constexpr):
    """S = sum_{j != excl} exp(z_j - shift), streamed in BLOCK_V chunks. shift = z at the target column,
    so z_j - shift is in [-2 cap, 2 cap] (no overflow, no running max) and loss = log1p(S) has no
    cancellation even when the target saturates the softcap (S tiny)."""
    cols0 = tl.arange(0, BLOCK_V)
    if F64:
        acc = tl.zeros([BLOCK_V], dtype=tl.float64)
    else:
        acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    for c in range(0, V, BLOCK_V):
        cols = c + cols0
        m = cols < V
        l = tl.load(base + cols, mask=m, other=0.0)
        _th, z = _softcap(l, cap, F64, TANH_APPROX)
        acc += tl.where(m & (cols != excl), _exp(z - shift, F64), 0.0)
    return tl.sum(acc, axis=0)


@triton.jit
def _target_z(base, t, cap, F64: tl.constexpr, TANH_APPROX: tl.constexpr):
    """z at the target column (t already clamped to a valid index for ignored rows)."""
    lt = tl.load(base + t)
    _th, zt = _softcap(lt, cap, F64, TANH_APPROX)
    return zt


@triton.jit
def c662_ce_fwd(logits_ptr, tgt_ptr, loss_ptr, lse2_ptr, V, stride_row, cap,
                BLOCK_V: tl.constexpr, F64: tl.constexpr, TANH_APPROX: tl.constexpr):
    """Form (ii) forward: per row loss = lse - z_t (0 if ignored) and lse2 = log2(sum exp z)."""
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    t = tl.load(tgt_ptr + row)
    valid = t != -1
    safe_t = tl.where(valid, t, 0)
    zt = _target_z(base, safe_t, cap, F64, TANH_APPROX)
    s = _row_sum_exp_excl(base, V, cap, zt, safe_t, BLOCK_V, F64, TANH_APPROX)
    loss = _log1p(s)                                   # = lse - z_t, without the cancellation
    tl.store(loss_ptr + row, tl.where(valid, loss, 0.0))
    tl.store(lse2_ptr + row, (zt + loss) * 1.4426950408889634)   # log2(sum_j exp z_j), for the backward


@triton.jit
def c662_ce_bwd(logits_ptr, tgt_ptr, lse2_ptr, g_ptr, g_stride, dl_ptr, V, stride_row, dl_stride_row, cap,
                BLOCK_V: tl.constexpr, F64: tl.constexpr, TANH_APPROX: tl.constexpr,
                EMU_RTNE: tl.constexpr = False):
    """Form (ii) backward: dlogits = (exp(z - lse) - onehot(t)) * (1 - tanh^2) * g_row (0 for ignored rows)."""
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    out = dl_ptr + row * dl_stride_row
    t = tl.load(tgt_ptr + row)
    valid = t != -1
    lse2 = tl.load(lse2_ptr + row)
    g = tl.load(g_ptr + row * g_stride)
    if F64:
        g = g.to(tl.float64)
        lse = lse2.to(tl.float64) * 0.6931471805599453
    else:
        g = g.to(tl.float32)
    g = tl.where(valid, g, 0.0)
    cols0 = tl.arange(0, BLOCK_V)
    dl_ty = dl_ptr.dtype.element_ty
    for c in range(0, V, BLOCK_V):
        cols = c + cols0
        m = cols < V
        l = tl.load(base + cols, mask=m, other=0.0)
        th, z = _softcap(l, cap, F64, TANH_APPROX)
        if F64:
            p = tl.exp(z - lse)
        else:
            p = tl.exp2(z * 1.4426950408889634 - lse2)
        p = tl.where(cols == t, p - 1.0, p)
        d = p * (1.0 - th * th) * g
        tl.store(out + cols, _to_out(d, dl_ty, EMU_RTNE), mask=m)


@triton.jit
def c662_ce_gif(logits_ptr, tgt_ptr, loss_ptr, g_ptr, dl_ptr, V, stride_row, dl_stride_row, cap,
                BLOCK_V: tl.constexpr, SINGLE: tl.constexpr, F64: tl.constexpr, TANH_APPROX: tl.constexpr,
                EMU_RTNE: tl.constexpr = False):
    """Form (i): loss AND dlogits (pre-scaled by g = 1/n_valid for mean, 1 otherwise) in one kernel.
    SINGLE: the whole row is one register tile (BLOCK_V >= V): logits read once, transcendentals once.
    Otherwise two streamed passes (sum, then recompute + write)."""
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    out = dl_ptr + row * dl_stride_row
    t = tl.load(tgt_ptr + row)
    valid = t != -1
    safe_t = tl.where(valid, t, 0)
    g = tl.load(g_ptr)
    if F64:
        g = g.to(tl.float64)
    else:
        g = g.to(tl.float32)
    g = tl.where(valid, g, 0.0)
    dl_ty = dl_ptr.dtype.element_ty
    if SINGLE:
        cols = tl.arange(0, BLOCK_V)
        m = cols < V
        l = tl.load(base + cols, mask=m, other=0.0)
        th, z = _softcap(l, cap, F64, TANH_APPROX)
        zt = tl.sum(tl.where(cols == safe_t, z, 0.0), axis=0)
        e = tl.where(m, _exp(z - zt, F64), 0.0)           # target column: exp(0) = 1
        s = tl.sum(tl.where(cols == safe_t, 0.0, e), axis=0)
        p = e * (1.0 / (1.0 + s))
        p = tl.where(cols == t, p - 1.0, p)
        d = p * (1.0 - th * th) * g
        tl.store(out + cols, _to_out(d, dl_ty, EMU_RTNE), mask=m)
    else:
        zt = _target_z(base, safe_t, cap, F64, TANH_APPROX)
        s = _row_sum_exp_excl(base, V, cap, zt, safe_t, BLOCK_V, F64, TANH_APPROX)
        inv_s = 1.0 / (1.0 + s)
        cols0 = tl.arange(0, BLOCK_V)
        for c in range(0, V, BLOCK_V):
            cols = c + cols0
            m = cols < V
            l = tl.load(base + cols, mask=m, other=0.0)
            th, z = _softcap(l, cap, F64, TANH_APPROX)
            p = _exp(z - zt, F64) * inv_s
            p = tl.where(cols == t, p - 1.0, p)
            d = p * (1.0 - th * th) * g
            tl.store(out + cols, _to_out(d, dl_ty, EMU_RTNE), mask=m)
    tl.store(loss_ptr + row, tl.where(valid, _log1p(s), 0.0))


# ----------------------------------------------------------------------------------------------
# Launchers
# ----------------------------------------------------------------------------------------------
_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


def _is_fake(t):
    from torch._subclasses.fake_tensor import FakeTensor
    return isinstance(t, FakeTensor) or t.device.type == "meta"


def _maybe_wrap(kernel, wrap):
    if not wrap or _interpret():
        return kernel
    from torch.library import wrap_triton
    return wrap_triton(kernel)


def _prep(logits, targets):
    if logits.dim() < 1:
        raise ValueError("logits must be (..., V)")
    V = logits.shape[-1]
    if logits.dtype not in _DTYPES:
        raise ValueError(f"logits dtype must be one of {_DTYPES}, got {logits.dtype}")
    lg = logits.reshape(-1, V)
    if lg.stride(-1) != 1:
        lg = lg.contiguous()
    tg = targets.reshape(-1)
    if tg.numel() != lg.shape[0]:
        raise ValueError(f"targets {tuple(targets.shape)} do not match logits rows {lg.shape[0]}")
    if tg.dtype != torch.int64:
        tg = tg.to(torch.int64)
    return lg, tg.contiguous(), V


def _cfg(variant):
    form, block_v, num_warps, approx = VARIANTS[variant or DEFAULT_VARIANT]
    return form, block_v, num_warps, approx


def _block(block_v, V):
    return triton.next_power_of_2(V) if block_v is None else min(block_v, triton.next_power_of_2(V))


def _launch_fwd(logits, targets, cap=SOFTCAP, variant=None, wrap=False):
    """Form (ii) forward. Returns (loss_rows, lse2, compiled_kernel_or_None)."""
    lg, tg, V = _prep(logits, targets)
    form, block_v, num_warps, approx = _cfg(variant)
    if form != "rec":
        raise ValueError(f"variant {variant!r} is not a form (ii) variant")
    n = lg.shape[0]
    rdt = torch.float64 if lg.dtype == torch.float64 else torch.float32
    loss = torch.empty(n, dtype=rdt, device=lg.device)
    lse2 = torch.empty(n, dtype=rdt, device=lg.device)
    if n == 0 or (_is_fake(lg) and _interpret()):
        return loss, lse2, None
    ck = _maybe_wrap(c662_ce_fwd, wrap)[(n,)](
        lg, tg, loss, lse2, V, lg.stride(0), float(cap),
        BLOCK_V=_block(block_v, V), F64=lg.dtype == torch.float64, TANH_APPROX=approx, num_warps=num_warps)
    return loss, lse2, ck


def _launch_bwd(logits, targets, lse2, grad_rows, cap=SOFTCAP, variant=None, wrap=False):
    """Form (ii) backward. grad_rows: (N,) per-row tangent of loss_rows (may be an expanded stride-0 view).
    Returns (dlogits shaped like logits.reshape(-1, V), compiled_kernel_or_None)."""
    lg, tg, V = _prep(logits, targets)
    form, block_v, num_warps, approx = _cfg(variant)
    if form != "rec":
        raise ValueError(f"variant {variant!r} is not a form (ii) variant")
    n = lg.shape[0]
    dl = torch.empty_like(lg)
    if n == 0 or (_is_fake(lg) and _interpret()):
        return dl, None
    g = grad_rows.reshape(-1)
    if g.numel() != n:
        raise ValueError(f"grad_rows has {g.numel()} elements, expected {n}")
    g_stride = g.stride(0) if g.numel() > 1 else 0
    ck = _maybe_wrap(c662_ce_bwd, wrap)[(n,)](
        lg, tg, lse2.contiguous(), g, g_stride, dl, V, lg.stride(0), dl.stride(0), float(cap),
        BLOCK_V=_block(block_v, V), F64=lg.dtype == torch.float64, TANH_APPROX=approx,
        EMU_RTNE=_interpret(), num_warps=num_warps)
    return dl, ck


def _n_valid(tg):
    return (tg != IGNORE_INDEX).sum()


def _launch_gif(logits, targets, reduction="mean", cap=SOFTCAP, variant="gif_tile"):
    """Form (i). Returns (loss_rows, dlogits pre-scaled for `reduction`, n_valid, compiled_kernel_or_None)."""
    lg, tg, V = _prep(logits, targets)
    form, block_v, num_warps, approx = _cfg(variant)
    if form != "gif":
        raise ValueError(f"variant {variant!r} is not a form (i) variant")
    n = lg.shape[0]
    rdt = torch.float64 if lg.dtype == torch.float64 else torch.float32
    loss = torch.empty(n, dtype=rdt, device=lg.device)
    dl = torch.empty_like(lg)
    nv = _n_valid(tg)
    if reduction == "mean":
        g = (1.0 / nv.to(rdt)).reshape(1)       # inf when nothing is valid; every row is then zeroed anyway
    else:
        g = torch.ones(1, dtype=rdt, device=lg.device)
    if n == 0:
        return loss, dl, nv, None
    blk = _block(block_v, V)
    single = blk >= V
    if single and blk > MAX_TILE_V:
        raise ValueError(f"gif_tile keeps the whole row in registers; V={V} > {MAX_TILE_V}, use gif_2pass")
    ck = c662_ce_gif[(n,)](
        lg, tg, loss, g, dl, V, lg.stride(0), dl.stride(0), float(cap),
        BLOCK_V=blk, SINGLE=single, F64=lg.dtype == torch.float64, TANH_APPROX=approx,
        EMU_RTNE=_interpret(), num_warps=num_warps)
    return loss, dl, nv, ck


def _reduce(loss_rows, tg, reduction):
    if reduction == "none":
        return loss_rows
    if reduction == "sum":
        return loss_rows.sum()
    if reduction == "mean":
        return loss_rows.sum() / _n_valid(tg).to(loss_rows.dtype)
    raise ValueError(f"unknown reduction {reduction!r}")


# ----------------------------------------------------------------------------------------------
# torch.library.triton_op registration (form ii; compile integration, gate G3)
# ----------------------------------------------------------------------------------------------
def _registered(name):
    try:
        torch._C._dispatch_find_schema_or_throw(f"cisc662::{name}", "")
        return True
    except RuntimeError:
        return False


if not _registered("ce_fwd"):
    from torch.library import triton_op

    @triton_op("cisc662::ce_fwd", mutates_args=())
    def _ce_fwd_op(logits: torch.Tensor, targets: torch.Tensor, cap: float) -> tuple[torch.Tensor, torch.Tensor]:
        loss, lse2, _ = _launch_fwd(logits, targets, cap, wrap=True)
        return loss, lse2

    @triton_op("cisc662::ce_bwd", mutates_args=())
    def _ce_bwd_op(logits: torch.Tensor, targets: torch.Tensor, lse2: torch.Tensor, grad_rows: torch.Tensor,
                   cap: float) -> torch.Tensor:
        dl, _ = _launch_bwd(logits, targets, lse2, grad_rows, cap, wrap=True)
        return dl.reshape(logits.shape)

    def _setup_context(ctx, inputs, output):
        logits, targets, cap = inputs
        _, lse2 = output
        ctx.save_for_backward(logits, targets, lse2)
        ctx.cap = cap
        ctx.mark_non_differentiable(lse2)

    def _backward(ctx, g_loss, _g_lse2):
        logits, targets, lse2 = ctx.saved_tensors
        if g_loss is None:
            return None, None, None
        dl = torch.ops.cisc662.ce_bwd(logits, targets, lse2, g_loss, ctx.cap)
        return dl, None, None

    _ce_fwd_op.register_autograd(_backward, setup_context=_setup_context)


# ----------------------------------------------------------------------------------------------
# autograd.Function for form (i) (eager only)
# ----------------------------------------------------------------------------------------------
class CEGifFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, reduction="mean", cap=SOFTCAP, variant="gif_tile"):
        loss_rows, dl, nv, _ = _launch_gif(logits, targets, reduction, cap, variant)
        ctx.save_for_backward(dl)
        ctx.reduction, ctx.shape = reduction, logits.shape
        ctx.mark_non_differentiable(nv)
        if reduction == "none":
            return loss_rows.reshape(targets.shape)
        if reduction == "sum":
            return loss_rows.sum()
        return loss_rows.sum() / nv.to(loss_rows.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (dl,) = ctx.saved_tensors
        if ctx.reduction == "none":
            dl = dl * grad_out.reshape(-1, 1).to(dl.dtype)
        elif not bool((grad_out == 1).all()):        # host sync (Liger does the same); skips a 4.3 GB pass
            dl = dl * grad_out.to(dl.dtype)
        return dl.reshape(ctx.shape), None, None, None, None


# ----------------------------------------------------------------------------------------------
# Public API and reference
# ----------------------------------------------------------------------------------------------
def softcap_ce(logits, targets, reduction="mean", impl="op", variant=None, cap=SOFTCAP):
    """nanochat's loss tail .float() -> cap*tanh(x/cap) -> F.cross_entropy(ignore_index=-1), fused.

    logits   (..., V) bf16 (fp16/fp32/fp64 accepted); never upcast in memory
    targets  (...) int64, -1 = ignored
    impl     "op"  = torch.ops.cisc662.ce_fwd/ce_bwd (form ii, triton_op, compile-visible; variant rec*)
             "gif" = CEGifFunction (form i, eager only; variant gif_tile | gif_2pass)
    Returns the loss reduced like F.cross_entropy (fp32; fp64 for fp64 logits).
    """
    if impl == "op":
        if variant not in (None, DEFAULT_VARIANT):
            # non-default form (ii) variants are launched through the same autograd wiring, eagerly
            return _RecFunction.apply(logits, targets, reduction, cap, variant)
        loss_rows, _ = torch.ops.cisc662.ce_fwd(logits, targets, float(cap))
        lg_rows = _reduce(loss_rows, targets.reshape(-1), reduction)
        return lg_rows.reshape(targets.shape) if reduction == "none" else lg_rows
    if impl == "gif":
        return CEGifFunction.apply(logits, targets, reduction, cap, variant or "gif_tile")
    raise ValueError(f"impl must be 'op' or 'gif', got {impl!r}")


class _RecFunction(torch.autograd.Function):
    """Form (ii) with a non-default tile variant (benchmark/ablation path; direct launches)."""
    @staticmethod
    def forward(ctx, logits, targets, reduction, cap, variant):
        loss_rows, lse2, _ = _launch_fwd(logits, targets, cap, variant)
        ctx.save_for_backward(logits, targets, lse2)
        ctx.cap, ctx.variant, ctx.reduction = cap, variant, reduction
        out = _reduce(loss_rows, targets.reshape(-1), reduction)
        return out.reshape(targets.shape) if reduction == "none" else out

    @staticmethod
    def backward(ctx, grad_out):
        logits, targets, lse2 = ctx.saved_tensors
        tg = targets.reshape(-1)
        n = tg.numel()
        if ctx.reduction == "none":
            g = grad_out.reshape(-1)
        elif ctx.reduction == "sum":
            g = grad_out.reshape(1).expand(n)
        else:
            g = (grad_out / _n_valid(tg).to(grad_out.dtype)).reshape(1).expand(n)
        dl, _ = _launch_bwd(logits, targets, lse2, g, ctx.cap, ctx.variant)
        return dl.reshape(logits.shape), None, None, None, None


def reference(logits, targets, reduction="mean", cap=SOFTCAP, dtype=torch.float64):
    """The same math in `dtype` (default fp64) PyTorch = cisc662.tests.ref_ops.softcap_ce."""
    V = logits.shape[-1]
    z = cap * torch.tanh(logits.reshape(-1, V).to(dtype) / cap)
    tg = targets.reshape(-1)
    valid = tg != IGNORE_INDEX
    t = torch.where(valid, tg, torch.zeros_like(tg))
    rows = torch.logsumexp(z, -1) - z.gather(1, t[:, None]).squeeze(1)
    rows = torch.where(valid, rows, torch.zeros_like(rows))
    out = _reduce(rows, tg, reduction)
    return out.reshape(targets.shape) if reduction == "none" else out


# ----------------------------------------------------------------------------------------------
# ISA dump
# ----------------------------------------------------------------------------------------------
R12_N, R12_V = 32768, 32768


def _sass_counts(sass):
    import re
    keys = {"LDG.E.128": r"LDG\.E\.128\b", "LDG.E.64": r"LDG\.E\.64\b", "LDG total": r"\bLDG\b",
            "STG.E.128": r"STG\.E\.128\b", "STG.E.64": r"STG\.E\.64\b", "STG total": r"\bSTG\b",
            "MUFU.EX2": r"MUFU\.EX2\b", "MUFU.RCP": r"MUFU\.RCP\b", "MUFU.TANH": r"MUFU\.TANH\b",
            "MUFU.LG2": r"MUFU\.LG2\b", "MUFU total": r"\bMUFU\b", "SHFL": r"\bSHFL\b", "BAR": r"\bBAR\b",
            "FFMA": r"\bFFMA\b", "FMUL": r"\bFMUL\b", "STL (spill)": r"\bSTL\b", "LDL (spill)": r"\bLDL\b",
            "instructions": r"^[-0-9a-fA-F]{2}:\S*\t\S"}
    return {k: len(re.findall(v, sass, flags=re.M)) for k, v in keys.items()}


def _write_dump(outdir, variant, name, asm, meta):
    import json
    d = os.path.join(outdir, f"triton_{variant}", "sm_90a")
    os.makedirs(d, exist_ok=True)
    for st in ("ttir", "ttgir", "llir", "ptx", "sass"):
        if st in asm and asm[st]:
            with open(os.path.join(d, f"{name}.{st}"), "w") as f:
                f.write(asm[st])
    if asm.get("sass"):
        meta["sass_counts"] = _sass_counts(asm["sass"])
    path = os.path.join(d, f"{name}.meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return path


def dump_isa(outdir, variants=None, n_rows=1024):
    """GPU only: launch each variant on (n_rows, 32768) bf16 (the per-row kernel is identical for any row
    count, so a short launch gives the R12 code) and write ttir/ttgir/llir/ptx/sass + meta.json with the
    loaded cubin's n_regs/n_spills into <outdir>/triton_<variant>/sm_90a/."""
    import json
    if not torch.cuda.is_available():
        raise SystemExit("--dump-isa needs a GPU (launch mode); the static cross-compile lives in "
                         "cisc662/analysis/xcompile.py")
    variants = variants or list(VARIANTS)
    g = torch.Generator(device="cuda").manual_seed(0)
    logits = (torch.randn((n_rows, R12_V), generator=g, device="cuda") * 3).bfloat16()
    targets = torch.randint(0, R12_V, (n_rows,), generator=g, device="cuda")
    written = []
    for v in variants:
        form, block_v, nw, approx = VARIANTS[v]
        launches = []
        if form == "rec":
            loss, lse2, ck_f = _launch_fwd(logits, targets, variant=v)
            gr = torch.full((n_rows,), 1.0 / n_rows, device="cuda")
            _, ck_b = _launch_bwd(logits, targets, lse2, gr, variant=v)
            launches = [("c662_ce_fwd", ck_f), ("c662_ce_bwd", ck_b)]
        else:
            _, _, _, ck = _launch_gif(logits, targets, "mean", variant=v)
            launches = [("c662_ce_gif", ck)]
        torch.cuda.synchronize()
        for name, ck in launches:
            ck._init_handles()
            meta = {"kernel": name, "variant": v, "form": form, "BLOCK_V": _block(block_v, R12_V),
                    "num_warps": nw, "tanh_approx": approx, "shape": {"logits": [R12_N, R12_V]},
                    "dtype": "bfloat16", "triton": triton.__version__, "torch": torch.__version__,
                    "target": "cuda:90 (sm_90a)", "mode": "launch", "n_regs": ck.n_regs,
                    "n_spills": ck.n_spills, "shared": ck.metadata.shared, "num_stages": ck.metadata.num_stages,
                    "hash": ck.hash, "device": torch.cuda.get_device_name(),
                    "job_id": os.environ.get("SLURM_JOB_ID")}
            asm = {st: ck.asm[st] for st in ("ttir", "ttgir", "llir", "ptx") if st in ck.asm}
            try:
                asm["sass"] = ck.asm["sass"]
            except Exception as e:     # cuobjdump missing
                meta["sass_error"] = f"{type(e).__name__}: {e}"
            path = _write_dump(outdir, v, name, asm, meta)
            written.append(path)
            sc = meta.get("sass_counts", {})
            print(f"{v:10s} {name:12s} regs={meta['n_regs']} spills={meta['n_spills']} "
                  f"LDG.E.128={sc.get('LDG.E.128')} STG.E.128={sc.get('STG.E.128')} MUFU={sc.get('MUFU total')} "
                  f"(EX2 {sc.get('MUFU.EX2')}, RCP {sc.get('MUFU.RCP')}, TANH {sc.get('MUFU.TANH')}) "
                  f"STL={sc.get('STL (spill)')} -> {path}", flush=True)
    with open(os.path.join(outdir, "summary_launch.json"), "w") as f:
        json.dump({"written": written, "triton": triton.__version__}, f, indent=1)
    return written


# ----------------------------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------------------------
def _self_test():
    dev = "cuda" if torch.cuda.is_available() and not _interpret() else "cpu"
    if dev == "cpu" and not _interpret():
        print("no CUDA: run with TRITON_INTERPRET=1", file=sys.stderr)
        return 2
    import torch.nn.functional as F
    torch.manual_seed(0)
    ok = True

    def stock(logits, targets, reduction):
        z = SOFTCAP * torch.tanh(logits.float() / SOFTCAP)
        return F.cross_entropy(z.view(-1, z.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction)

    cases = [(64, 512), (5, 1000), (17, 4096 + 3)]
    forms = [("op", None), ("op", "rec_2k"), ("gif", "gif_tile"), ("gif", "gif_2pass")]
    for N, V in cases:
        logits = (torch.randn(N, V, device=dev) * 3).bfloat16()
        logits[0, :8] = 70.0                       # softcap saturation
        targets = torch.randint(0, V, (N,), device=dev)
        targets[1] = -1
        for reduction in ("mean", "none"):
            for impl, var in forms:
                la = logits.clone().requires_grad_(True)
                out = softcap_ce(la, targets, reduction, impl=impl, variant=var)
                ref = reference(logits, targets, reduction)
                rel = ((out.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
                g = torch.randn_like(out) if reduction == "none" else torch.tensor(1.0, device=dev)
                out.backward(g)
                l64 = logits.double().requires_grad_(True)
                reference(l64, targets, reduction).backward(g.double())
                ls = logits.clone().requires_grad_(True)
                stock(ls, targets, reduction).backward(g)
                eg = ((la.grad.double() - l64.grad).norm() / l64.grad.norm()).item()
                es = ((ls.grad.double() - l64.grad).norm() / l64.grad.norm()).item()
                ign0 = bool((la.grad[1] == 0).all())
                good = rel < 1e-5 and eg <= max(2 * es, 1e-2) and ign0
                ok &= good
                print(f"N={N:3d} V={V:5d} {reduction:4s} {impl}/{var or DEFAULT_VARIANT:9s} loss rel {rel:.2e} | "
                      f"dlogits rel-L2 {eg:.2e} (stock {es:.2e}) | ignored row zero {ign0} {'ok' if good else 'FAIL'}")
    # fp64 gradcheck on a tiny problem, both forms
    lg = (torch.randn(4, 16, dtype=torch.float64, device=dev) * 3).requires_grad_(True)
    tg = torch.tensor([3, -1, 15, 0], device=dev)
    for impl, var in (("op", None), ("gif", "gif_tile"), ("gif", "gif_2pass")):
        gc = torch.autograd.gradcheck(lambda x: softcap_ce(x, tg, "mean", impl=impl, variant=var), (lg,))
        gcn = torch.autograd.gradcheck(lambda x: softcap_ce(x, tg, "none", impl=impl, variant=var), (lg,))
        ok &= bool(gc) and bool(gcn)
        print(f"gradcheck fp64 {impl}/{var}: mean {gc} none {gcn}")
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _main(argv):
    import argparse
    p = argparse.ArgumentParser(description="KT-3 fused softcap + cross-entropy: self-test or ISA dump")
    p.add_argument("--dump-isa", metavar="OUTDIR", default=None)
    p.add_argument("--variants", default=",".join(VARIANTS))
    a = p.parse_args(argv)
    if a.dump_isa:
        dump_isa(a.dump_isa, [v for v in a.variants.split(",") if v])
        return 0
    return _self_test()


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
