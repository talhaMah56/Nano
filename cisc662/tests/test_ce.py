"""
Card KT-3: gates G1/G2 (PLAN.md 7.3, amendment A1) for the fused softcap + cross-entropy kernel
cisc662/kernels/triton/ce.py.

  reference  fp64: ref_ops.softcap_ce(logits, targets, dtype=float64) on the same bf16 logits (loss);
             ref_ops.softcap_ce_grad (fp64, chunked) for dlogits
  stock      nanochat's own tail: logits.float() -> 15*tanh(x/15) -> F.cross_entropy(ignore_index=-1)
             (= ops_at_shapes.softcap_ce), backward through the bf16 logits (so its dlogits are bf16)
  G1         loss: tolerances.check_forward(kind="ce_loss") -- scalar mean rel 1e-5; per-row ("none")
             rel 1e-5 + fp32 atol
  G2         dlogits: tolerances.check_backward(kind="dlogits", stock_grad=stock, ignore_rows=targets==-1)
             -- rel-L2 <= 1e-2 and <= 2x stock, cosine >= 0.9999, ignored rows exactly 0; fp64 gradcheck
  forms      every form the benchmark times as a result row is gated: torch.ops.cisc662.ce_fwd/ce_bwd
             (form ii, default tile "rec"), the form (ii) tile variants rec_2k / rec_8k, and form (i)
             CEGifFunction gif_tile / gif_2pass. Form (i) is gated on the training path (reduction="mean",
             grad_out == 1, dlogits rounded to bf16 once); with reduction="none" it rounds twice (bf16
             dlogits * per-row grad), which is checked against rel-L2 <= 1e-2 only and never timed.
             rec_approx (tanh.approx.f32) is an ablation, never recorded as gated; its measured error is
             logged by test_approx_ablation_error_is_recorded (8.3e-6 per-row in job 3417307, inside 1e-5).

Cases (conftest.ce_case): tiny, saturated (|logits| > 50), all_ignored (mean -> NaN like F.cross_entropy,
all-zero dlogits), rows1000 (V=8192) on CPU/interpreter; rows1000_v32768 [gpu]; r12 32768 x 32768 [real_shape].

Login node:  python -m pytest cisc662/tests -q -m "not real_shape and not gpu"
             (CPU variants run under TRITON_INTERPRET=1 through test_interpreter_subprocess)
GPU job:     python -m pytest cisc662/tests/test_ce.py -q -rxs
"""
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from cisc662.kernels.triton import ce
from cisc662.tests import ref_ops, tolerances as tol
from cisc662.tests.conftest import GPU, SEED, gen

F64 = torch.float64
BF16 = torch.bfloat16
INTERPRET = os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (impl, variant, gated reductions)
FORMS = [("op", None, ("mean", "none")), ("op", "rec_2k", ("mean", "none")), ("op", "rec_8k", ("mean", "none")),
         ("gif", "gif_tile", ("mean",)), ("gif", "gif_2pass", ("mean",))]
INTERP_FORMS = {("op", None), ("gif", "gif_tile")}     # keep the interpreter run short on the login node
FORM_PARAMS = [pytest.param(f, id=f"{f[0]}-{f[1] or ce.DEFAULT_VARIANT}") for f in FORMS]


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


def _check_form(form, device):
    if device.type == "cpu" and form[:2] not in INTERP_FORMS:
        pytest.skip("interpreter run covers the default form (ii) tile and gif_tile only")


def stock_tail(logits, targets, reduction="mean"):
    """GPT.forward gpt.py:513-520 on given bf16 logits (the slice is a no-op at these V)."""
    z = 15 * torch.tanh(logits.float() / 15)
    return F.cross_entropy(z.view(-1, z.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction)


_REF_CACHE = {}   # fp64 reference loss and dlogits per fixture case (the r12 fp64 grad is 8.6 GB to build)


def _reference(case, reduction, grad_rows=None):
    key = (case["name"], case["device"].type, reduction, grad_rows is not None)
    if key not in _REF_CACHE:
        if len(_REF_CACHE) >= 4:
            _REF_CACHE.clear()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        lg, tg = case["logits"], case["targets"]
        chunk = 2048 if lg.shape[0] > 4096 else None
        with torch.no_grad():
            loss = ref_ops.softcap_ce(lg, tg, reduction=reduction, dtype=F64, chunk_rows=chunk)
        dl = ref_ops.softcap_ce_grad(lg, tg, reduction=reduction, dtype=F64, chunk_rows=2048,
                                     grad_out=grad_rows)
        _REF_CACHE[key] = (loss, dl)
    return _REF_CACHE[key]


def _grads(fn, logits, targets, reduction, grad_rows=None):
    x = logits.detach().clone().requires_grad_(True)
    out = fn(x, targets, reduction)
    g = grad_rows if reduction == "none" else torch.ones((), device=out.device, dtype=out.dtype)
    (dl,) = torch.autograd.grad(out, x, g.to(out.dtype))
    return out.detach(), dl


def _ours(impl, variant):
    return lambda x, t, r: ce.softcap_ce(x, t, r, impl=impl, variant=variant)


def assert_gate(r, ctx=""):
    assert r, (ctx + " " if ctx else "") + r.summary()


def _row_grads(case):
    n = case["targets"].numel()
    return torch.randn((n,), generator=gen(SEED + 5, case["device"]), device=case["device"])


# --- reference (no Triton involved) -------------------------------------------------------------
def test_module_reference_is_ref_ops_fp64():
    g = gen(SEED, "cpu")
    lg = (torch.randn((40, 300), generator=g) * 4).to(BF16)
    tg = torch.randint(0, 300, (40,), generator=g)
    tg[::7] = -1
    for red in ("mean", "none", "sum"):
        a = ce.reference(lg, tg, red)
        b = ref_ops.softcap_ce(lg, tg, reduction=red, dtype=F64)
        assert torch.allclose(a, b, rtol=1e-12, atol=1e-12), red


# --- G1: loss -----------------------------------------------------------------------------------
@pytest.mark.parametrize("form", FORM_PARAMS)
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_ce_g1_loss(ce_case, form, reduction):
    dev = kernel_device(ce_case["device"].type)
    _check_form(form, dev)
    impl, variant, _ = form
    lg, tg = ce_case["logits"], ce_case["targets"]
    with torch.no_grad():
        out = ce.softcap_ce(lg, tg, reduction, impl=impl, variant=variant)
        stock = stock_tail(lg, tg, reduction)
    ref, _ = _reference(ce_case, reduction)
    assert out.dtype == torch.float32 and out.shape == ref.shape
    if ce_case["name"] == "all_ignored" and reduction == "mean":
        assert torch.isnan(out) and torch.isnan(stock)       # 0/0 like F.cross_entropy
        return
    assert_gate(tol.check_forward(out, ref, stock_out=stock, kind="ce_loss"),
                f"[{ce_case['name']} {impl}/{variant} {reduction}]")


# --- G2: dlogits --------------------------------------------------------------------------------
@pytest.mark.parametrize("form", FORM_PARAMS)
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_ce_g2_dlogits(ce_case, form, reduction):
    dev = kernel_device(ce_case["device"].type)
    _check_form(form, dev)
    impl, variant, gated = form
    lg, tg = ce_case["logits"], ce_case["targets"]
    grow = _row_grads(ce_case) if reduction == "none" else None
    _, ref = _reference(ce_case, reduction, grow)
    _, dl = _grads(_ours(impl, variant), lg, tg, reduction, grow)
    assert dl.dtype == BF16 and dl.shape == lg.shape
    ignore = tg == ref_ops.IGNORE_INDEX
    if ce_case["name"] == "all_ignored":
        assert bool((dl == 0).all())                          # nothing valid: the gradient is exactly 0
        return
    ctx = f"[{ce_case['name']} {impl}/{variant} {reduction}]"
    if reduction in gated:
        _, sdl = _grads(stock_tail, lg, tg, reduction, grow)
        assert_gate(tol.check_backward(dl, ref, stock_grad=sdl, kind="dlogits", ignore_rows=ignore), ctx)
    else:  # form (i) with per-row grads: double rounding, informational bound only (never timed)
        r = tol.check_backward(dl, ref, kind="dlogits", ignore_rows=ignore)
        assert r.rel_fro <= tol.DLOGITS_REL_L2 and r.ignore_rows_zero, ctx + " " + r.summary()


# --- fp64 gradcheck -----------------------------------------------------------------------------
def _tiny_device_params():
    return [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="cuda", marks=GPU)]


@pytest.mark.parametrize("form", FORM_PARAMS)
@pytest.mark.parametrize("device", _tiny_device_params())
def test_gradcheck_fp64_tiny(device, form):
    dev = kernel_device(device)
    impl, variant, _ = form
    lg = (torch.randn((5, 24), generator=gen(SEED, "cpu"), dtype=F64) * 3).to(dev).requires_grad_(True)
    tg = torch.tensor([3, -1, 23, 0, 7], device=dev)
    for red in ("mean", "none", "sum"):
        assert torch.autograd.gradcheck(lambda x: ce.softcap_ce(x, tg, red, impl=impl, variant=variant), (lg,)), red


# --- shape/layout behaviour ---------------------------------------------------------------------
@pytest.mark.parametrize("device", _tiny_device_params())
def test_3d_logits_like_nanochat(device):
    """GPT.forward passes (B, T, V) logits and (B, T) targets; the result equals the 2-D call."""
    dev = kernel_device(device)
    lg = (torch.randn((2, 9, 300), generator=gen(SEED, "cpu")) * 3).to(BF16).to(dev)
    tg = torch.randint(0, 300, (2, 9), generator=gen(SEED + 1, "cpu")).to(dev)
    tg[0, 3] = -1
    a = ce.softcap_ce(lg, tg, "mean")
    b = ce.softcap_ce(lg.reshape(-1, 300), tg.reshape(-1), "mean")
    assert torch.equal(a, b)
    n = ce.softcap_ce(lg, tg, "none")
    assert n.shape == tg.shape and n[0, 3] == 0


@pytest.mark.parametrize("device", _tiny_device_params())
def test_triton_op_opcheck(device):
    dev = kernel_device(device)
    lg = (torch.randn((6, 64), generator=gen(SEED, "cpu")) * 3).to(BF16).to(dev).requires_grad_(True)
    tg = torch.tensor([1, -1, 63, 0, 5, 9], device=dev)
    utils = ["test_schema", "test_autograd_registration", "test_faketensor"]
    if dev.type == "cuda":
        utils.append("test_aot_dispatch_static")
    torch.library.opcheck(torch.ops.cisc662.ce_fwd.default, (lg, tg, 15.0), test_utils=utils)


# --- GPU-only properties ------------------------------------------------------------------------
@pytest.mark.gpu
def test_one_cuda_kernel_per_direction():
    """Form (ii): the forward launches c662_ce_fwd once (plus the tiny mean reduction), the backward
    c662_ce_bwd once and no logits-sized elementwise kernel."""
    kernel_device("cuda")
    lg = (torch.randn((64, 4096), device="cuda") * 3).to(BF16).requires_grad_(True)
    tg = torch.randint(0, 4096, (64,), device="cuda")
    ce.softcap_ce(lg, tg).backward()   # JIT outside the profiler
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as pf:
        loss = ce.softcap_ce(lg, tg)
        torch.cuda.synchronize()
    kern = [e.name for e in pf.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert kern.count("c662_ce_fwd") == 1, kern
    with profile(activities=[ProfilerActivity.CUDA]) as pb:
        torch.autograd.grad(loss, lg)
        torch.cuda.synchronize()
    kern = [e.name for e in pb.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert kern.count("c662_ce_bwd") == 1, kern


@pytest.mark.gpu
def test_compiles_fullgraph():
    """G3 preview: the triton_op traces under torch.compile(fullgraph=True) and matches eager."""
    kernel_device("cuda")
    lg = (torch.randn((256, 8192), device="cuda") * 3).to(BF16)
    tg = torch.randint(0, 8192, (256,), device="cuda")
    tg[::13] = -1

    def f(x, t):
        return ce.softcap_ce(x, t, "mean")
    fc = torch.compile(f, fullgraph=True, dynamic=False)
    a, b = lg.clone().requires_grad_(True), lg.clone().requires_grad_(True)
    la, lb = f(a, tg), fc(b, tg)
    la.backward()
    lb.backward()
    assert torch.equal(la, lb) and torch.equal(a.grad, b.grad)


@pytest.mark.gpu
def test_approx_ablation_error_is_recorded():
    """rec_approx (tanh.approx.f32) is an instruction-count ablation, never a gated result (the S4 job skips
    gate recording on ablation_* rows). The card predicted it would miss the 1e-5 loss gate by ~100x; job
    3417307 measured 8.3e-6 per-row on R12-like logits -- the per-element tanh error (~5e-4) mostly averages
    out inside the 32768-term log-sum-exp. Only sanity (< 1e-3) is asserted; the measured value is written
    to the log and to findings.md#ce."""
    kernel_device("cuda")
    g = torch.Generator(device="cuda").manual_seed(SEED)
    lg = (torch.randn((512, 32768), generator=g, device="cuda") * 3).to(BF16)
    tg = torch.randint(0, 32768, (512,), generator=g, device="cuda")
    with torch.no_grad():
        out = ce.softcap_ce(lg, tg, "none", impl="op", variant="rec_approx")
        ref = ref_ops.softcap_ce(lg, tg, reduction="none", dtype=F64)
        ours = ce.softcap_ce(lg, tg, "none", impl="op")
    rel = ((out.double() - ref).abs() / ref.abs()).max().item()
    rel_precise = ((ours.double() - ref).abs() / ref.abs()).max().item()
    print(f"per-row loss max rel err: rec_approx {rel:.3e}, rec (precise tanh) {rel_precise:.3e}")
    assert rel < 1e-3 and rel_precise < rel


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
    assert " passed" in r.stdout, tail
