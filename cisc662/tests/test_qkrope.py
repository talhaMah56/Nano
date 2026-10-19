"""
Card KT-2: gates G1/G2 (PLAN.md 7.3, amendment A1) for the fused QK-norm + RoPE + 1.2 kernel
cisc662/kernels/triton/qkrope.py.

  reference  fp64: ref_ops.qkrope(q, k, cos, sin, dtype=float64) on the same bf16 inputs and bf16 table
  stock      nanochat's own chain gpt.apply_rotary_emb -> gpt.norm -> *1.2 in bf16 (the lines of
             CausalSelfAttention.forward; proven equal to them in test_ref_ops.py)
  G1         tolerances.check_forward(ours, ref64, stock_out=stock) on yq and yk; rstd vs fp64 (fp32 rule)
  G2         tolerances.check_backward(dq|dk, fp64 grads, stock_grad=stock grads); fp64 gradcheck on tiny shapes
             (G1, G2 and the tiny gradcheck run for the triton_op AND every QKRopeFunction tile variant the
             benchmark times: v1, ilp2, ilp4)
  also       fold_scale == unfolded / 1.2 with softmax_scale = 1.44/sqrt(128); q and k in one launch;
             the module's reference() == ref_ops.qkrope in fp64 (bit-exact); triton_op opcheck

Shapes: R12 (16, 2048, 6, 128) [real_shape+gpu]; positions 0..2047 [gpu]; edge cases on both devices:
B*T*H not a multiple of BLOCK_M, Hk != Hq, positions 2040..2047, T=1 decode slices at offsets 517, 2047
and 9999 (the decode table is 10x the training length).

The CPU ("cpu") variants run the Triton kernels under TRITON_INTERPRET=1 only (the interpreter must be
enabled before the kernel module is imported), so in a normal login-node run they skip and
test_interpreter_subprocess re-runs this file in a child process with TRITON_INTERPRET=1.

Login node:  python -m pytest cisc662/tests -q -m "not real_shape and not gpu"
GPU job:     python -m pytest cisc662/tests/test_qkrope.py -q -rxs
"""
import functools
import math
import os
import subprocess
import sys

import pytest
import torch

import nanochat.gpt as gpt
from cisc662.kernels.triton import qkrope as qk
from cisc662.tests import ref_ops, tolerances as tol
from cisc662.tests.conftest import GPU, R12, REAL, SEED, gen, randn

F64 = torch.float64
BF16 = torch.bfloat16
D = R12["D"]
INTERPRET = os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPLS = ("op", "function")
# every form the benchmark times must pass G1/G2 itself (PLAN 7.3; the S3 job records G1/G2 on every
# non-eager/compile row): the triton_op (default tile v1) and QKRopeFunction with each tile variant
IMPL_VARIANTS = [pytest.param(("op", None), id="op")] + [
    pytest.param(("function", v), id=f"function-{v}") for v in qk.VARIANTS]


# --- helpers ------------------------------------------------------------------------------------
def kernel_device(device):
    d = torch.device(device)
    if d.type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if INTERPRET:
            pytest.skip("TRITON_INTERPRET=1 runs the kernels on CPU tensors only")
    elif not INTERPRET:
        pytest.skip("CPU runs of the Triton kernel need TRITON_INTERPRET=1 (see test_interpreter_subprocess)")
    return d


@functools.lru_cache(maxsize=4)
def _table(device):
    """nanochat's bf16 cos/sin buffer: 10 x the training sequence length, (1, 20480, 1, 64)."""
    return ref_ops.rotary_tables(10 * R12["T"], D, device=device)


def stock_chain(q, k, cos, sin):
    """The exact lines of CausalSelfAttention.forward (= test_ref_ops._nanochat_qk_chain)."""
    q, k = gpt.apply_rotary_emb(q, cos, sin), gpt.apply_rotary_emb(k, cos, sin)
    q, k = gpt.norm(q), gpt.norm(k)
    return q * 1.2, k * 1.2


def run_grads(fn, a, b, ga, gb):
    a, b = a.detach().clone().requires_grad_(True), b.detach().clone().requires_grad_(True)
    oa, ob = fn(a, b)[:2]
    torch.autograd.backward([oa, ob], [ga, gb])
    return a.grad, b.grad


def assert_gate(r, ctx=""):
    assert r, (ctx + " " if ctx else "") + r.summary()


# --- cases --------------------------------------------------------------------------------------
# (name, B, T, Hq, Hk, first position)
EDGE = [
    ("tiny", 2, 16, 6, 6, 0),
    ("rows_odd", 3, 7, 5, 5, 100),          # B*T*H = 105 rows: not a multiple of any BLOCK_M
    ("gqa", 2, 9, 6, 2, 300),               # n_kv_head != n_head (nanochat allows it; R12 has 6/6)
    ("tail_pos", 1, 8, 6, 6, 2040),         # positions 2040..2047
    ("decode_t1", 4, 1, 6, 6, 2047),        # T=1 decode slice cos[:, T0:T0+1]
    ("decode_t1_off", 2, 1, 6, 6, 517),
    ("decode_t1_far", 1, 1, 6, 6, 9999),    # beyond the training length (decode table is 20480 long)
]
CASES = [pytest.param((*c, "cpu"), id=f"{c[0]}-cpu") for c in EDGE]
CASES += [pytest.param((*c, "cuda"), id=f"{c[0]}-cuda", marks=GPU) for c in EDGE]
CASES += [
    pytest.param(("pos2047", 2, 2048, 6, 6, 0, "cuda"), id="pos2047-cuda", marks=GPU),     # positions 0..2047
    pytest.param(("r12", R12["B"], R12["T"], R12["H"], R12["H"], 0, "cuda"), id="r12", marks=REAL),
]


@pytest.fixture(params=CASES)
def case(request):
    name, B, T, Hq, Hk, pos0, dev = request.param
    dev = kernel_device(dev)
    cos, sin = _table(str(dev))
    return dict(name=name, q=randn((B, T, Hq, D), SEED, dev, 0.5), k=randn((B, T, Hk, D), SEED + 1, dev, 0.5),
                cos=cos[:, pos0:pos0 + T], sin=sin[:, pos0:pos0 + T], pos0=pos0, device=dev)


def _tiny_device_params():
    return [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="cuda", marks=GPU)]


# --- reference (no Triton involved) -------------------------------------------------------------
def test_module_reference_is_ref_ops_fp64():
    """qkrope.reference (what the kernel computes, in fp64) is ref_ops.qkrope(dtype=fp64), bit for bit."""
    cos, sin = _table("cpu")
    q, k = randn((2, 5, 3, D), SEED, "cpu", 0.5), randn((2, 5, 2, D), SEED + 1, "cpu", 0.5)
    c, s = cos[:, 40:45], sin[:, 40:45]
    a = qk.reference(q, k, c, s)
    b = ref_ops.qkrope(q, k, c, s, dtype=F64)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_softmax_scale_folded_value():
    assert qk.softmax_scale_folded(128) == pytest.approx(1.44 / math.sqrt(128), rel=1e-15)


# --- G1 -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("impl_variant", IMPL_VARIANTS)
def test_qkrope_g1(case, impl_variant):
    impl, variant = impl_variant
    q, k, cos, sin = (case[n] for n in ("q", "k", "cos", "sin"))
    yq, yk = qk.qkrope(q, k, cos, sin, impl=impl, variant=variant)
    assert yq.dtype == BF16 and yk.dtype == BF16 and yq.shape == q.shape and yk.shape == k.shape
    q64, k64 = ref_ops.qkrope(q, k, cos, sin, dtype=F64)
    qs, ks = stock_chain(q, k, cos, sin)
    for name, y, r, s in (("q", yq, q64, qs), ("k", yk, k64, ks)):
        assert_gate(tol.check_forward(y, r, stock_out=s), f"[{case['name']} {impl}/{variant} {name}]")


def test_qkrope_rstd(case):
    """The saved rstd (fp32) is rsqrt(mean(r^2) + eps) of the ROTATED row (fp32 rule vs fp64)."""
    q, k, cos, sin = (case[n] for n in ("q", "k", "cos", "sin"))
    _, _, rq, rk = torch.ops.cisc662.qkrope(q, k, cos, sin, 1.2)
    assert rq.dtype == torch.float32 and rq.shape == q.shape[:-1] and rk.shape == k.shape[:-1]
    _, _, rq64, rk64 = qk.reference(q, k, cos, sin, return_rstd=True)
    for r, r64 in ((rq, rq64), (rk, rk64)):
        assert_gate(tol.check_forward(r, r64, reduction_len=D))


# --- G2 -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("impl_variant", IMPL_VARIANTS)
def test_qkrope_g2(case, impl_variant):
    impl, variant = impl_variant
    q, k, cos, sin = (case[n] for n in ("q", "k", "cos", "sin"))
    gq, gk = randn(q.shape, SEED + 7, q.device), randn(k.shape, SEED + 8, k.device)
    o = run_grads(lambda a, b: qk.qkrope(a, b, cos, sin, impl=impl, variant=variant), q, k, gq, gk)
    s = run_grads(lambda a, b: stock_chain(a, b, cos, sin), q, k, gq, gk)
    r = run_grads(lambda a, b: ref_ops.qkrope(a, b, cos, sin, dtype=F64), q.double(), k.double(),
                  gq.double(), gk.double())
    for name, g_o, g_r, g_s in zip(("dq", "dk"), o, r, s):
        assert g_o.dtype == BF16
        assert_gate(tol.check_backward(g_o, g_r, stock_grad=g_s, reduction_len=D),
                    f"[{case['name']} {impl}/{variant} {name}]")


def _angles(T, half, seed, device):
    a = torch.rand((1, T, 1, half), generator=gen(seed, "cpu"), dtype=F64) * 6.28
    return a.cos().to(device), a.sin().to(device)


def _leaf(shape, seed, device):
    return torch.randn(shape, generator=gen(seed, "cpu"), dtype=F64).to(device).requires_grad_(True)


@pytest.mark.parametrize("impl_variant", IMPL_VARIANTS)
@pytest.mark.parametrize("device", _tiny_device_params())
def test_gradcheck_fp64_tiny(device, impl_variant):
    """fp64 gradcheck, head_dim 8, random angles (cos^2+sin^2 = 1 to fp64), Hq != Hk; every tile variant."""
    impl, variant = impl_variant
    dev = kernel_device(device)
    cos, sin = _angles(3, 4, 4, dev)
    fn = lambda q, k: qk.qkrope(q, k, cos, sin, impl=impl, variant=variant)
    assert torch.autograd.gradcheck(fn, (_leaf((1, 3, 2, 8), 5, dev), _leaf((1, 3, 1, 8), 6, dev)))


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("device", _tiny_device_params())
def test_gradcheck_fp64_nanochat_table(device, impl):
    """fp64 gradcheck at head_dim 128 with nanochat's bf16 table at positions 2046..2047 (cos^2+sin^2 != 1:
    the backward is the exact transpose of the rotation, so it holds without orthogonality)."""
    dev = kernel_device(device)
    cos, sin = _table(str(dev))
    c, s = cos[:, 2046:2048], sin[:, 2046:2048]
    fn = lambda q, k: qk.qkrope(q, k, c, s, impl=impl)
    assert torch.autograd.gradcheck(fn, (_leaf((1, 2, 2, D), 7, dev), _leaf((1, 2, 1, D), 8, dev)),
                                    fast_mode=True)


# --- fold_scale ---------------------------------------------------------------------------------
@pytest.mark.parametrize("device", _tiny_device_params())
def test_fold_scale(device):
    dev = kernel_device(device)
    cos, sin = _table(str(dev))
    B, T, H = 2, 12, 6
    c, s = cos[:, 100:100 + T], sin[:, 100:100 + T]
    q, k = randn((B, T, H, D), SEED, dev, 0.5), randn((B, T, H, D), SEED + 1, dev, 0.5)
    yq, yk = qk.qkrope(q, k, c, s)
    fq, fk, sm = qk.qkrope(q, k, c, s, fold_scale=True)
    assert sm == pytest.approx(1.2 ** 2 / math.sqrt(D), rel=1e-15)
    # folded == unfolded / 1.2 up to one bf16 rounding of each (<= 2^-8 relative each)
    for f, u in ((fq, yq), (fk, yk)):
        d = (f.float() - u.float() / 1.2).abs()
        assert bool((d <= 2.0 ** -7 * f.float().abs() * 1.01 + 1e-6).all()), float(d.max())
    # same attention logits: (fq . fk) * 1.44/sqrt(D) == (yq . yk) / sqrt(D)
    lf = torch.einsum("bthd,bshd->bhts", fq.double(), fk.double()) * sm
    lu = torch.einsum("bthd,bshd->bhts", yq.double(), yk.double()) / math.sqrt(D)
    q64, k64 = ref_ops.qkrope(q, k, c, s, dtype=F64)
    lr = torch.einsum("bthd,bshd->bhts", q64, k64) / math.sqrt(D)
    err_f, err_u = float((lf - lr).abs().max()), float((lu - lr).abs().max())
    assert err_f <= 2 * err_u + 1e-3 * float(lr.abs().max()), (err_f, err_u)
    # backward in fp64: fold with upstream grad 1.2*dy == unfolded with dy
    # (1e-6: the unfolded kernel applies 1.2 as an fp32 scalar, 1.2000000477)
    q64l, k64l = q.double(), k.double()
    gq, gk = randn(q.shape, SEED + 7, dev, dtype=F64), randn(k.shape, SEED + 8, dev, dtype=F64)
    gu = run_grads(lambda a, b: qk.qkrope(a, b, c, s), q64l, k64l, gq, gk)
    gf = run_grads(lambda a, b: qk.qkrope(a, b, c, s, fold_scale=True), q64l, k64l, 1.2 * gq, 1.2 * gk)
    for a, b in zip(gf, gu):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-9)


# --- one launch for q and k ---------------------------------------------------------------------
class _CountingKernel:
    def __init__(self, kernel):
        self.kernel, self.grids = kernel, []

    def __getitem__(self, grid):
        self.grids.append(grid)
        return self.kernel[grid]


@pytest.mark.parametrize("device", _tiny_device_params())
def test_q_and_k_in_one_launch(device, monkeypatch):
    """Forward and backward are one kernel launch each for q AND k (grid = q blocks + k blocks)."""
    dev = kernel_device(device)
    cos, sin = _table(str(dev))
    B, T, Hq, Hk = 2, 7, 5, 3
    c, s = cos[:, 50:50 + T], sin[:, 50:50 + T]
    q, k = randn((B, T, Hq, D), SEED, dev, 0.5), randn((B, T, Hk, D), SEED + 1, dev, 0.5)
    fwd, bwd = _CountingKernel(qk.c662_qkrope_fwd), _CountingKernel(qk.c662_qkrope_bwd)
    monkeypatch.setattr(qk, "c662_qkrope_fwd", fwd)
    monkeypatch.setattr(qk, "c662_qkrope_bwd", bwd)
    bm = qk.VARIANTS[qk.DEFAULT_VARIANT][0]
    expect = (-(-B * T * Hq // bm) - (-B * T * Hk // bm),)
    gq, gk = randn(q.shape, SEED + 7, dev), randn(k.shape, SEED + 8, dev)
    dq, dk = run_grads(lambda a, b: qk.qkrope(a, b, c, s, impl="function"), q, k, gq, gk)
    assert fwd.grids == [expect] and bwd.grids == [expect], (fwd.grids, bwd.grids, expect)
    # and both tensors really came out of that launch
    yq, yk = qk.qkrope(q, k, c, s, impl="function")
    q64, k64 = ref_ops.qkrope(q, k, c, s, dtype=F64)
    assert_gate(tol.check_forward(yq, q64, stock_out=stock_chain(q, k, c, s)[0]))
    assert_gate(tol.check_forward(yk, k64, stock_out=stock_chain(q, k, c, s)[1]))
    assert bool(dq.abs().sum() > 0) and bool(dk.abs().sum() > 0)


@pytest.mark.gpu
@pytest.mark.parametrize("impl", IMPLS)
def test_one_cuda_kernel_per_direction(impl):
    """On the GPU: the forward is exactly one CUDA kernel (c662_qkrope_fwd), the backward contains exactly
    one c662_qkrope_bwd."""
    dev = kernel_device("cuda")
    cos, sin = _table("cuda")
    q = randn((2, 64, 6, D), SEED, dev, 0.5).requires_grad_(True)
    k = randn((2, 64, 6, D), SEED + 1, dev, 0.5).requires_grad_(True)
    c, s = cos[:, :64], sin[:, :64]
    gq, gk = torch.randn_like(q), torch.randn_like(k)
    qk.qkrope(q, k, c, s, impl=impl)   # JIT outside the profiler
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as pf:
        yq, yk = qk.qkrope(q, k, c, s, impl=impl)
        torch.cuda.synchronize()
    kern = [e.name for e in pf.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert kern.count("c662_qkrope_fwd") == 1 and len(kern) == 1, kern
    with profile(activities=[ProfilerActivity.CUDA]) as pb:
        torch.autograd.grad([yq, yk], [q, k], [gq, gk])
        torch.cuda.synchronize()
    kern = [e.name for e in pb.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert kern.count("c662_qkrope_bwd") == 1, kern


# --- torch.library registration -----------------------------------------------------------------
@pytest.mark.parametrize("device", _tiny_device_params())
def test_triton_op_opcheck(device):
    dev = kernel_device(device)
    cos, sin = _table(str(dev))
    q, k = randn((1, 4, 2, D), SEED, dev, 0.5), randn((1, 4, 2, D), SEED + 1, dev, 0.5)
    c, s = cos[:, 10:14], sin[:, 10:14]
    utils = ["test_schema", "test_autograd_registration", "test_faketensor"]
    if dev.type == "cuda":
        utils.append("test_aot_dispatch_static")
    torch.library.opcheck(torch.ops.cisc662.qkrope.default, (q.requires_grad_(True), k, c, s, 1.2),
                          test_utils=utils)


# --- login node: run the CPU variants above under the Triton interpreter ------------------------
@pytest.mark.skipif(INTERPRET, reason="already running under TRITON_INTERPRET=1")
@pytest.mark.skipif(torch.cuda.is_available(), reason="GPU node: the cuda variants test the real kernels")
def test_interpreter_subprocess():
    env = dict(os.environ, TRITON_INTERPRET="1", OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.abspath(__file__),
           "-m", "not real_shape and not gpu", "-k", "not test_interpreter_subprocess"]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1200)
    tail = r.stdout[-6000:] + "\n" + r.stderr[-3000:]
    assert r.returncode == 0, tail
    assert " passed" in r.stdout and " skipped" not in r.stdout.splitlines()[-1], tail
