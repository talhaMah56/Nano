"""
Card KT-1 gates for cisc662/kernels/triton/rmsnorm.py (PLAN.md 7.3 with amendment A1; tolerances.py).

  G1  forward of every variant (v1, v2 = RMSNorm 768/128-wide; v3 = mixnorm) vs the fp64 reference on the
      same bf16 inputs, judged against the stock eager op (F.rms_norm / r*x + l*x0 then F.rms_norm):
      real shapes (32768,768), (16,2048,6,128) and (16,2048,768) [real_shape+gpu]; edge shapes N=1003 rows
      (not a multiple of BLOCK_M), T=1 decode rows, both widths [gpu]; fp32 inputs under the fp32 rule.
  G2  bf16 gradients dx (dx0) vs fp64 autograd of the reference with the stock gradient as yardstick,
      cosine >= 0.9999; mixnorm's scalar gradients dr, dl (fp32 outputs) under the fp32 rule (rtol scaled
      by max(|ref|, ||summands||_2), see test_mixnorm_g2) against fp64 autograd given the forward's stored
      x_new (they are heavily cancelled sums) and
      within 2x max(stock op's error, exactly-rounded-x_new error) vs the unrounded fp64 chain (A1 yardstick);
      torch.autograd.gradcheck in fp64 on tiny shapes (masked and unmasked chunk paths) [gpu].
  G3 prerequisites (G3 itself is INT-1): torch.library.opcheck of the four cisc662 ops [gpu].
  Login-node tests (no marker): references == ref_ops, eps, chunking/launch specs on meta tensors, op
  registration and kernel names, and the kernels' self-test under TRITON_INTERPRET=1 in fp32 (subprocess).

Login node:  python -m pytest cisc662/tests/test_rmsnorm.py -q -m "not real_shape and not gpu"
GPU job:     python -m pytest cisc662/tests/test_rmsnorm.py -q -rxs
"""
import os
import re
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from cisc662.kernels.triton import rmsnorm as K
from cisc662.tests import ref_ops, tolerances as tol
from cisc662.tests.conftest import GPU, R12, REAL, SEED, nanochat_lambdas, randn, require_device

F64 = torch.float64
BF16 = torch.bfloat16
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (name, shape) -- the last dim is the normalised one
RMS_CASES = [
    # 1003 rows: not a multiple of any BLOCK_M in CONFIGS (4, 8, 16) -- 1000 is a multiple of 8 and 4
    pytest.param(("rows1003_768", (1003, R12["C"])), id="rows1003_768", marks=GPU),
    pytest.param(("rows1003_128", (1003, R12["D"])), id="rows1003_128", marks=GPU),
    pytest.param(("t1_768", (1, 1, R12["C"])), id="t1_768", marks=GPU),                      # T=1 decode, 1 row
    pytest.param(("t1_qk", (4, 1, R12["H"], R12["D"])), id="t1_qk", marks=GPU),              # T=1 QK-norm rows
    pytest.param(("r12_resid", (R12["N"], R12["C"])), id="r12_resid", marks=REAL),
    pytest.param(("r12_qk", (R12["B"], R12["T"], R12["H"], R12["D"])), id="r12_qk", marks=REAL),
]
MIX_CASES = [
    pytest.param(("rows1003", (1003, R12["C"])), id="rows1003", marks=GPU),    # partial last block (BLOCK_M 4)
    pytest.param(("t1", (3, 1, R12["C"])), id="t1", marks=GPU),                 # 3 decode rows < BLOCK_M
    pytest.param(("r12", (R12["B"], R12["T"], R12["C"])), id="r12", marks=REAL),
]


def assert_gate(r, extra=""):
    assert r, r.summary() + (" | " + extra if extra else "")


def _cuda_input(shape, seed, dtype=BF16, scale=0.5):
    dev = require_device("cuda")
    return randn(shape, seed, dev, scale, dtype=dtype)


def _lambdas(dev, layer=5):
    r_all, l_all = nanochat_lambdas(device=dev)
    return r_all[layer].detach().clone(), l_all[layer].detach().clone()   # 0-d fp32, as GPT.forward indexes


# --- G1: RMSNorm forward ------------------------------------------------------------------------
@pytest.mark.parametrize("variant", K.VARIANTS)
@pytest.mark.parametrize("case", RMS_CASES)
def test_rmsnorm_g1(case, variant):
    name, shape = case
    x = _cuda_input(shape, SEED)
    D = shape[-1]
    ref = ref_ops.rmsnorm(x, dtype=F64)
    stock = F.rms_norm(x, (D,))
    y, rstd = torch.ops.cisc662.rmsnorm(x, variant)
    assert y.dtype == BF16 and y.shape == x.shape and rstd.shape == x.shape[:-1] and rstd.dtype == torch.float32
    r = tol.check_forward(y, ref, stock_out=stock)
    print(f"[G1 rmsnorm {variant} {name}] {r.summary()}")
    assert_gate(r)
    rstd_ref = torch.rsqrt(x.double().pow(2).mean(-1) + ref_ops.RMS_EPS)
    assert_gate(tol.check_forward(rstd, rstd_ref), "rstd (fp32 output)")


@pytest.mark.parametrize("variant", K.VARIANTS)
@pytest.mark.parametrize("shape", [(1003, 768), (1003, 128), (1, 768)], ids=str)
@pytest.mark.gpu
def test_rmsnorm_g1_fp32(shape, variant):
    x = _cuda_input(shape, SEED + 3, dtype=torch.float32)
    y = K.rmsnorm(x, variant)
    assert_gate(tol.check_forward(y, ref_ops.rmsnorm(x, dtype=F64), reduction_len=shape[-1]))


@pytest.mark.gpu
def test_rmsnorm_impl_fn_matches_op():
    x = _cuda_input((1000, 768), SEED + 4)
    for v in K.VARIANTS:
        assert torch.equal(K.rmsnorm(x, v, impl="fn"), K.rmsnorm(x, v, impl="op"))


@pytest.mark.gpu
def test_rmsnorm_noncontiguous_input():
    """A strided (row-sliced) input is copied to rows, never misread."""
    big = _cuda_input((1000, 2 * 768), SEED + 5)
    x = big[:, :768]
    assert not x.is_contiguous()
    stock = F.rms_norm(x, (768,))
    assert_gate(tol.check_forward(K.rmsnorm(x), ref_ops.rmsnorm(x, dtype=F64), stock_out=stock))


# --- G2: RMSNorm backward -----------------------------------------------------------------------
def _grads(fn, inputs, grad_outs):
    xs = [t.detach().clone().requires_grad_(True) for t in inputs]
    outs = fn(*xs)
    outs = outs if isinstance(outs, tuple) else (outs,)
    pairs = [(o, g) for o, g in zip(outs, grad_outs) if g is not None]
    torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
    return [t.grad for t in xs]


@pytest.mark.parametrize("variant", K.VARIANTS)
@pytest.mark.parametrize("case", RMS_CASES)
def test_rmsnorm_g2(case, variant):
    name, shape = case
    x = _cuda_input(shape, SEED)
    D = shape[-1]
    gy = _cuda_input(shape, SEED + 7)
    (g_ours,) = _grads(lambda t: K.rmsnorm(t, variant), [x], [gy])
    (g_stock,) = _grads(lambda t: F.rms_norm(t, (D,)), [x], [gy])
    (g_ref,) = _grads(lambda t: ref_ops.rmsnorm(t, dtype=F64), [x.double()], [gy.double()])
    assert g_ours.dtype == BF16
    r = tol.check_backward(g_ours, g_ref, stock_grad=g_stock, reduction_len=D)
    print(f"[G2 rmsnorm {variant} {name}] {r.summary()}")
    assert_gate(r, "stock: " + tol.check_backward(g_stock, g_ref, stock_grad=g_stock, reduction_len=D).summary())


@pytest.mark.parametrize("variant", K.VARIANTS)
@pytest.mark.parametrize("shape", [(1003, 768), (1003, 128)], ids=str)
@pytest.mark.gpu
def test_rmsnorm_g2_fp32(shape, variant):
    x = _cuda_input(shape, SEED + 8, dtype=torch.float32)
    gy = _cuda_input(shape, SEED + 9, dtype=torch.float32)
    (g_ours,) = _grads(lambda t: K.rmsnorm(t, variant), [x], [gy])
    (g_ref,) = _grads(lambda t: ref_ops.rmsnorm(t, dtype=F64), [x.double()], [gy.double()])
    assert_gate(tol.check_backward(g_ours, g_ref, reduction_len=shape[-1]))


# --- G1/G2: mixnorm (v3) ------------------------------------------------------------------------
def _stock_mix(x, x0, r, l):
    """GPT.forward's mix followed by Block.forward's norm, verbatim (eager)."""
    xn = r * x + l * x0
    return xn, F.rms_norm(xn, (xn.size(-1),))


@pytest.mark.parametrize("case", MIX_CASES)
def test_mixnorm_g1(case):
    name, shape = case
    x, x0 = _cuda_input(shape, SEED), _cuda_input(shape, SEED + 1)
    r, l = _lambdas(x.device)
    with torch.no_grad():
        xn_ref, h_ref = K.reference_mixnorm(x, x0, r, l)
        xn_s, h_s = _stock_mix(x, x0, r, l)
        xn, h, rstd = torch.ops.cisc662.mixnorm(x, x0, r, l)
    assert xn.dtype == h.dtype == BF16 and xn.shape == h.shape == x.shape and rstd.dtype == torch.float32
    rx = tol.check_forward(xn, xn_ref, stock_out=xn_s)
    rh = tol.check_forward(h, h_ref, stock_out=h_s)
    print(f"[G1 mixnorm {name}] x_new: {rx.summary()}\n[G1 mixnorm {name}] h: {rh.summary()}")
    assert_gate(rx, "x_new")
    assert_gate(rh, "h")
    # h is the norm of the *stored* x_new (nanochat normalises the bf16 residual stream)
    assert_gate(tol.check_forward(h, ref_ops.rmsnorm(xn, dtype=F64), stock_out=F.rms_norm(xn, (shape[-1],))),
                "h vs norm(stored x_new)")


@pytest.mark.parametrize("case", MIX_CASES)
@pytest.mark.parametrize("use_gxnew", [True, False], ids=["g_xnew+g_h", "g_h_only"])
def test_mixnorm_g2(case, use_gxnew):
    name, shape = case
    x, x0 = _cuda_input(shape, SEED), _cuda_input(shape, SEED + 1)
    r, l = _lambdas(x.device)
    gxn = _cuda_input(shape, SEED + 7) if use_gxnew else None
    gh = _cuda_input(shape, SEED + 8)
    g_ours = _grads(lambda *a: K.mixnorm(*a), [x, x0, r, l], [gxn, gh])
    g_stock = _grads(_stock_mix, [x, x0, r, l], [gxn, gh])
    g64 = [None if g is None else g.double() for g in (gxn, gh)]
    ins64 = [t.double() for t in (x, x0, r, l)]
    g_ref = _grads(lambda *a: K.reference_mixnorm(*a), ins64, g64)                       # unrounded fp64 chain
    g_ref_st = _grads(lambda *a: K.reference_mixnorm(*a, round_xnew_to=BF16), ins64, g64)  # x_new rounded exactly
    for nm, go, gr, gs in zip(("dx", "dx0"), g_ours[:2], g_ref[:2], g_stock[:2]):
        assert go.dtype == BF16
        res = tol.check_backward(go, gr, stock_grad=gs, reduction_len=shape[-1])
        print(f"[G2 mixnorm {name} {nm}] {res.summary()}")
        assert_gate(res, nm + " | stock: " + tol.check_backward(gs, gr, stock_grad=gs).summary())
    # dr = sum(dxn * x), dl = sum(dxn * x0) are heavily cancelled sums (RMSNorm is scale invariant: per row
    # sum(dxn_norm * x_new) = 0), so a 1-ulp difference in a few stored x_new elements (fp32 r*x+l*x0 then
    # bf16 vs the correctly rounded value: ~0.1% of elements) moves them by 1e-4..1e-3 relative. The fp32
    # rule is therefore applied to the backward given the forward's stored x_new (G1 gates x_new itself):
    # fp64 autograd of norm(x_new_ours) + g_xnew, contracted with x and x0 in fp64.
    with torch.no_grad():
        xn_ours, _, _ = torch.ops.cisc662.mixnorm(x, x0, r, l)
    xs = xn_ours.double().requires_grad_(True)
    K.reference(xs).backward(gh.double())
    dxn64 = xs.grad + (gxn.double() if gxn is not None else 0.0)
    terms = [dxn64 * x.double(), dxn64 * x0.double()]
    g_given = [t.sum() for t in terms]
    l2 = [float(t.pow(2).sum().sqrt()) for t in terms]
    del terms
    n_red = x.numel()
    rtol = tol.FP32_LONG_RTOL if n_red >= tol.LONG_REDUCTION else tol.FP32_RTOL
    rel = lambda a, b: abs(float(a) - float(b)) / max(abs(float(b)), 1e-30)
    for nm, go, gg, sc, gst, gr, gs in zip(("dr", "dl"), g_ours[2:], g_given, l2, g_ref_st[2:], g_ref[2:],
                                           g_stock[2:]):
        assert go.dtype == torch.float32 and go.shape == ()
        print(f"[G2 mixnorm {name} {nm}] ours {float(go):.8e} | rel err vs fp64-given-our-x_new {rel(go, gg):.2e}, "
              f"vs exactly-rounded-x_new ref {rel(go, gst):.2e}, vs unrounded chain {rel(go, gr):.2e} "
              f"(stock vs unrounded {rel(gs, gr):.2e}) | ||terms||_2 {sc:.3e}")
        # fp32 rule for a scalar that is a sum of n_red random-sign terms: |ours - ref| <= atol + rtol *
        # max(|ref|, ||terms||_2). ||terms||_2 is the typical size of such a sum; with |ref| alone a correct
        # kernel fails whenever the sum happens to land near 0 (CPU emulation with fp32 per-row sums and
        # rsqrt.approx-sized rstd error: 3.8% of draws at (3,1,768), 2.5% at (1003,768); worst 7% of this
        # limit). This limit still catches a 1-ulp x_new recompute mismatch in 2e-4 of the elements (the
        # fwd/bwd FMA-contraction bug fixed in _c662_mix: 1.3-1.5x over it on dr).
        err, lim = abs(float(go) - float(gg)), tol.FP32_ATOL + rtol * max(abs(float(gg)), sc)
        assert err <= lim, f"{nm} (fp32 rule): |ours - fp64| {err:.3e} > {lim:.3e} (rtol {rtol}, ||terms|| {sc:.3e})"
        # A1 yardstick on the unrounded chain: |ours - ref| <= 2 x max(stock's error, the error of the exact
        # function with x_new correctly rounded to bf16). Each error is ONE random scalar (a sum of 25M
        # rounding terms), so "2 x stock" alone fails a correct kernel whenever stock's error happens to land
        # near 0: in a CPU simulation of this check (ours emulated, 560 draws at (4,1,768)/(1000,768)) it
        # failed 37 times (6.6%, ~50% per GPU job over its 12 checks) and never with the max(): the
        # exactly-rounded reference is the best any bf16-storing kernel can do. The tight check of the
        # kernel itself is the fp32 rule above.
        e_ours, e_stock, e_exact = (abs(float(v) - float(gr)) for v in (go, gs, gst))
        bound = tol.STOCK_FACTOR * max(e_stock, e_exact)
        assert e_ours <= bound, (f"{nm}: |ours - ref| {e_ours:.3e} > 2 x max(stock {e_stock:.3e}, "
                                 f"exactly-rounded x_new {e_exact:.3e})")


@pytest.mark.gpu
def test_mixnorm_impl_fn_matches_op():
    x, x0 = _cuda_input((1000, 768), SEED), _cuda_input((1000, 768), SEED + 1)
    r, l = _lambdas(x.device)
    a = K.mixnorm(x, x0, r, l, impl="fn")
    b = K.mixnorm(x, x0, r, l, impl="op")
    assert all(torch.equal(p, q) for p, q in zip(a, b))


@pytest.mark.gpu
def test_mixnorm_xnew_single_rounding():
    """x_new = bf16(r*x + l*x0 in fp32): equals the correctly rounded fp64 value except where the fp32
    intermediate double-rounds (rare, 1 ulp)."""
    x, x0 = _cuda_input((1000, 768), SEED), _cuda_input((1000, 768), SEED + 1)
    r, l = _lambdas(x.device)
    xn, _, _ = torch.ops.cisc662.mixnorm(x, x0, r, l)
    exact = (r.double() * x.double() + l.double() * x0.double()).to(BF16)
    frac = float((xn != exact).float().mean())
    assert frac < 1e-3, f"{frac:.2e} of x_new differ from the correctly rounded value"


# --- G2: fp64 gradcheck on tiny shapes (CUDA: the kernels need a GPU) -----------------------------
GC_SHAPES = [(3, 8), (2, 40), (2, 300), (2, 768)]   # NCHUNK 1 / masked / 2 chunks masked / 3 chunks


def _leaf(shape, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.randn(shape, generator=g, device=dev, dtype=F64).requires_grad_(True)


@pytest.mark.parametrize("impl", ["op", "fn"])
@pytest.mark.parametrize("variant", K.VARIANTS)
@pytest.mark.parametrize("shape", GC_SHAPES, ids=str)
@pytest.mark.gpu
def test_gradcheck_rmsnorm(shape, variant, impl):
    dev = require_device("cuda")
    assert torch.autograd.gradcheck(lambda t: K.rmsnorm(t, variant, impl=impl), (_leaf(shape, 1, dev),))


@pytest.mark.parametrize("impl", ["op", "fn"])
@pytest.mark.parametrize("shape", GC_SHAPES[:3], ids=str)
@pytest.mark.gpu
def test_gradcheck_mixnorm(shape, impl):
    dev = require_device("cuda")
    args = (_leaf(shape, 2, dev), _leaf(shape, 3, dev), _leaf((), 4, dev), _leaf((), 5, dev))
    assert torch.autograd.gradcheck(lambda *a: K.mixnorm(*a, impl=impl), args)
    assert torch.autograd.gradcheck(lambda *a: K.mixnorm(*a, impl=impl)[1], args)   # x_new unused: HAS_GXN=False


# --- G3 prerequisites: torch.library registration (G3 itself = INT-1) -----------------------------
@pytest.mark.gpu
def test_opcheck_rmsnorm():
    x = _cuda_input((64, 768), SEED).requires_grad_(True)
    for v in K.VARIANTS:
        torch.library.opcheck(torch.ops.cisc662.rmsnorm.default, (x, v))
    xd = x.detach()
    _, rstd = torch.ops.cisc662.rmsnorm(xd, "v2")
    torch.library.opcheck(torch.ops.cisc662.rmsnorm_bwd.default, (_cuda_input((64, 768), SEED + 1), xd, rstd))


@pytest.mark.gpu
def test_opcheck_mixnorm():
    x, x0 = (_cuda_input((64, 768), SEED + s).requires_grad_(True) for s in (0, 1))
    r, l = (t.requires_grad_(True) for t in _lambdas(x.device))
    torch.library.opcheck(torch.ops.cisc662.mixnorm.default, (x, x0, r, l))
    _, _, rstd = torch.ops.cisc662.mixnorm(x.detach(), x0.detach(), r.detach(), l.detach())
    g = _cuda_input((64, 768), SEED + 2)
    for gxn in (g, None):
        torch.library.opcheck(torch.ops.cisc662.mixnorm_bwd.default,
                              (gxn, g, x.detach(), x0.detach(), r.detach(), l.detach(), rstd))


# --- login-node tests (no GPU, no kernel launches) -------------------------------------------------
def test_eps_is_fp32_eps():
    assert K.FP32_EPS == ref_ops.RMS_EPS == torch.finfo(torch.float32).eps


def test_reference_is_ref_ops():
    x = randn((64, 768), SEED, "cpu", 0.5)
    assert torch.equal(K.reference(x), ref_ops.rmsnorm(x, dtype=F64))
    x0 = randn((64, 768), SEED + 1, "cpu", 0.5)
    r, l = _lambdas("cpu")
    xn, h = K.reference_mixnorm(x, x0, r, l)
    assert torch.equal(xn, ref_ops.residual_mix(x, x0, r, l, dtype=F64))
    assert torch.equal(h, ref_ops.rmsnorm(xn, dtype=F64))
    xn_b, h_b = K.reference_mixnorm(x, x0, r, l, round_xnew_to=BF16)
    assert torch.equal(xn_b, xn) and torch.equal(h_b, ref_ops.rmsnorm(xn.to(BF16), dtype=F64))


def test_reference_gradcheck_straight_through():
    """round_xnew_to only changes the value, not the gradient path (straight-through)."""
    args = tuple(torch.randn(s, dtype=F64).requires_grad_(True) for s in ((2, 8), (2, 8), (), ()))
    assert torch.autograd.gradcheck(lambda *a: K.reference_mixnorm(*a), args)


def test_chunking_and_specs_on_meta():
    assert K._chunking(768) == (256, 3) and K._chunking(128) == (128, 1)
    specs = K.isa_specs("meta")
    expect = {"triton_v1_768", "triton_v2_768", "triton_bwd_768", "triton_v1_128", "triton_v2_128",
              "triton_bwd_128", "triton_mixnorm_fwd_768", "triton_mixnorm_bwd_768", "triton_mixnorm_bwd_finish_768"}
    assert set(specs) == expect
    for name, (kernel, grid, args, const, opts) in specs.items():
        assert const.get("ALIGNED", True) is True, name          # 128-bit loads need the stride hint
        assert const.get("IDX64", False) is False and const.get("IS_F64", False) is False, name
        if "CHUNK" in const:
            assert const["CHUNK"] * const["NCHUNK"] == const["D"], f"{name}: masked lanes at a real shape"
            assert grid == (triton_cdiv(args["N"], const["BLOCK_M"]),)
    assert specs["triton_v1_768"][3]["BLOCK_D"] == 1024


def triton_cdiv(a, b):
    return -(-a // b)


def test_ops_registered_and_kernel_names():
    for op in ("rmsnorm", "rmsnorm_bwd", "mixnorm", "mixnorm_bwd"):
        assert hasattr(torch.ops.cisc662, op), op
    schema = str(torch.ops.cisc662.mixnorm_bwd.default._schema)
    assert "Tensor? g_xnew" in schema, schema
    names = ["c662_rmsnorm_fwd_v1", "c662_rmsnorm_fwd_v2", "c662_rmsnorm_bwd", "c662_rmsnorm_fwd_mixnorm",
             "c662_rmsnorm_bwd_mixnorm", "c662_rmsnorm_bwd_mixnorm_finish"]
    from cisc662.profiling.classify_kernels import classify
    for n in names:
        fn = getattr(K, n)
        assert getattr(fn, "__name__", None) == n or getattr(getattr(fn, "fn", None), "__name__", None) == n
        assert classify(n) == "rmsnorm", (n, classify(n))   # what nsys attribution will call it


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") == "1", reason="interpreter kernels are not wrap_triton-able")
def test_fake_impls():
    """triton_op registers the launcher itself as the fake impl: shapes/dtypes propagate without a GPU
    (what Dynamo/AOTAutograd need; the HOP mutation analysis itself needs a GPU driver -> opcheck tests)."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    x = randn((64, 768), SEED, "cpu", 0.5)
    r, l = _lambdas("cpu")
    with FakeTensorMode() as m:
        fx, fr, fl = m.from_tensor(x), m.from_tensor(r), m.from_tensor(l)
        for v in K.VARIANTS:
            y, rstd = torch.ops.cisc662.rmsnorm(fx, v)
            assert y.shape == x.shape and y.dtype == BF16 and rstd.shape == (64,) and rstd.dtype == torch.float32
        assert torch.ops.cisc662.rmsnorm_bwd(fx, fx, rstd).shape == x.shape
        xn, h, rstd = torch.ops.cisc662.mixnorm(fx, fx, fr, fl)
        assert xn.dtype == h.dtype == BF16 and rstd.dtype == torch.float32
        for gxn in (fx, None):
            dx, dx0, dr, dl = torch.ops.cisc662.mixnorm_bwd(gxn, fx, fx, fx, fr, fl, rstd)
            assert dx.shape == dx0.shape == x.shape and dr.shape == dl.shape == () and dr.dtype == torch.float32


def test_interpreter_self_test_fp32():
    """The kernels themselves on CPU via TRITON_INTERPRET=1 (fp32: the 3.5.1 interpreter truncates
    fp32->bf16 casts, so bf16 there is not representative of the GPU's round-to-nearest-even)."""
    env = dict(os.environ, TRITON_INTERPRET="1", OMP_NUM_THREADS="2", PYTHONPATH=REPO)
    p = subprocess.run([sys.executable, "-m", "cisc662.kernels.triton.rmsnorm", "--dtype", "float32"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0 and "SELF-TEST PASS" in p.stdout, p.stdout[-3000:] + p.stderr[-3000:]
