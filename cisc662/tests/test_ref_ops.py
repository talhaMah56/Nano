"""
Card INF-2: the references in ref_ops.py are nanochat's functions, the tolerance code implements
PLAN.md 7.3, and record_gate keeps result rows intact.

  1. references == nanochat (bit-exact in bf16 where nanochat's eager op is the same computation,
     fp32-rounding agreement otherwise), incl. the F.rms_norm eps and the RoPE sign convention;
  2. the stock eager bf16 ops pass the G1/G2 gates against the fp64 reference (so the gates are
     achievable), and known-wrong variants (bf16 eps, flipped RoPE sin) fail them;
  3. torch.autograd.gradcheck in fp64 on tiny shapes for every reference;
  4. tolerance and gate-recording unit tests;
  5. GPU job only: real shapes, cuBLAS GEMM error, and FA3's error vs the fp32 math SDPA reference
     (stored to results/db/tolerances/attention_stock_error[_<jobid>].json for the attention gate).

Login node:  python -m pytest cisc662/tests -q -m "not real_shape and not gpu"
GPU job:     python -m pytest cisc662/tests -q
"""
import datetime
import json
import math
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import nanochat.gpt as gpt
from cisc662.tests import gates, ref_ops, tolerances as tol
from cisc662.tests.conftest import CE_CASES, R12, SEED, gen, job_id, randn

F64 = torch.float64
BF16 = torch.bfloat16


# --- helpers ------------------------------------------------------------------------------------
def mismatch_report(a, b):
    """Why two tensors are not bit-identical: count, max |diff|, max diff in units of b's ulp."""
    d = (a.float() - b.float()).abs()
    n = int((a != b).sum())
    if a.dtype == b.dtype == torch.bfloat16:
        ulps = bf16_ulps(a, b)
    else:
        ulps = d / (torch.finfo(b.dtype).eps * b.float().abs().clamp_min(torch.finfo(b.dtype).tiny))
    return f"{n}/{a.numel()} elements differ, max |diff| {float(d.max()):.3e}, max {float(ulps.max()):.1f} ulp"


def assert_bitexact(a, b):
    """Bit-identical, with NaN == NaN (the mean CE of an all-ignored batch is NaN in both)."""
    assert a.dtype == b.dtype and a.shape == b.shape
    same = (a == b) | (torch.isnan(a) & torch.isnan(b))
    assert bool(same.all()), mismatch_report(a, b)


def bf16_ulps(a, b):
    """|a - b| in units of the bf16 spacing at max(|a|, |b|) (exact: 2^(floor(log2|v|) - 7))."""
    af, bf = a.float(), b.float()
    _, e = torch.frexp(torch.maximum(af.abs(), bf.abs()))  # |v| = m * 2^e, m in [0.5, 1)
    return (af - bf).abs() / torch.ldexp(torch.ones_like(af), e - 8)


def cuda_ulp_mismatch(a, b, device, max_ulp):
    """"" if a and b are bit-identical (always required on CPU, where F.rms_norm is the composite =
    our reference). On CUDA, F.rms_norm dispatches to the fused aten::_fused_rms_norm kernel, whose
    fp32 row reduction runs in a different order, so a few outputs may round to the neighbouring
    bf16 value: that returns a report (the caller xfails, visible in the job summary for the lead),
    but only if every difference is <= max_ulp bf16 ulps on <= 0.1% of the elements; else it fails."""
    same = (a == b) | (torch.isnan(a) & torch.isnan(b))
    if bool(same.all()):
        return ""
    rep = mismatch_report(a, b)
    assert torch.device(device).type == "cuda", rep
    u = bf16_ulps(a, b)[~same]
    frac = float((~same).float().mean())
    assert float(u.max()) <= max_ulp and frac <= 1e-3, f"{rep}; max {float(u.max()):.1f} exact ulp"
    return f"{rep} (<= {max_ulp} ulp)"


def xfail_if_cuda_ulp(reports, why):
    reports = [r for r in reports if r]
    if reports:
        pytest.xfail("CUDA fused rms_norm not bit-exact: " + " | ".join(reports) + f" -- {why}")


def grads(fn, inputs, gy):
    """Gradients of <fn(*inputs), gy> w.r.t. every input (fresh leaves, same dtype as given)."""
    xs = [x.detach().clone().requires_grad_(True) for x in inputs]
    out = fn(*xs)
    out.backward(gy)
    return [x.grad for x in xs]


def assert_gate(r):
    assert r, r.summary()


def assert_gate_vs_stock(r, r_stock):
    """Under PLAN 7.3 amendment A1 the bf16 gate already judges ours against the stock op (violations
    <= 2x stock's count, max-abs <= 2x, cosine), so a failure is a real failure. (Before A1 this xfailed
    when stock itself missed the absolute rtol/atol rule: jobs 3406282/3406329.) r_stock is kept in the
    assertion message as context."""
    assert r, r.summary() + " | stock: " + r_stock.summary()


def stub_attention(monkeypatch, captured=None):
    """Replace nanochat's attention call by a stub that records q, k and returns v, so nanochat's
    module code runs on any device without FA3 (the trunk output is not what we test)."""
    def fake(q, k, v, causal=False, window_size=(-1, -1), softmax_scale=None):
        if captured is not None:
            captured.update(q=q, k=k, causal=causal, window_size=window_size)
        return v
    monkeypatch.setattr(gpt.flash_attn, "flash_attn_func", fake)


# --- RMSNorm ------------------------------------------------------------------------------------
def test_rms_eps_is_fp32_eps():
    assert ref_ops.RMS_EPS == torch.finfo(torch.float32).eps
    assert abs(ref_ops.RMS_EPS - 1.1920929e-07) < 1e-14


def test_rmsnorm_bitexact_vs_F_rms_norm(device):
    """INF-2 acceptance: the reference reproduces F.rms_norm(eps=None) bit-exactly on bf16 (64, 768)."""
    x = randn((64, 768), SEED, device, 0.5)
    stock = F.rms_norm(x, (768,))
    assert_bitexact(gpt.norm(x), stock)
    xfail_if_cuda_ulp([cuda_ulp_mismatch(ref_ops.rmsnorm(x), stock, device, 1)],
                      "eps itself is settled by test_rmsnorm_eps_is_not_bf16_eps[cuda]")


def test_rmsnorm_eps_is_not_bf16_eps(device):
    """On small-norm rows (mean(x^2) ~ 1e-6) eps matters: the fp32-eps reference tracks F.rms_norm
    to <= 1 bf16 ulp (exactly, on CPU); eps = finfo(bf16).eps, 1e-6 and 0 are each off by >= 5%
    (>> 1 ulp). Robust to reduction order, so this settles the eps on CUDA too."""
    x = randn((64, 768), SEED, device, 1e-3)
    stock = F.rms_norm(x, (768,))
    good = ref_ops.rmsnorm(x)
    assert float(bf16_ulps(good, stock).max()) <= 1.0, mismatch_report(good, stock)
    if device.type == "cpu":
        assert_bitexact(good, stock)
    bad = ref_ops.rmsnorm(x, eps=torch.finfo(BF16).eps)
    assert float((bad.float() - stock.float()).abs().max()) > 0.5
    for alt in (1e-6, 0.0):
        assert float(bf16_ulps(ref_ops.rmsnorm(x, eps=alt), stock).max()) > 4.0, f"eps={alt} not excluded"


def test_rmsnorm_fp64_ref_vs_nanochat_fp32(rmsnorm_case):
    x = rmsnorm_case["x"].float()
    D = x.shape[-1]
    assert_gate(tol.check_forward(gpt.norm(x), ref_ops.rmsnorm(x, dtype=F64), reduction_len=D))
    assert_gate(tol.check_forward(ref_ops.rmsnorm(x, dtype=torch.float32), ref_ops.rmsnorm(x, dtype=F64),
                                  reduction_len=D))


def test_rmsnorm_g1_bf16(rmsnorm_case):
    x = rmsnorm_case["x"]
    ref = ref_ops.rmsnorm(x, dtype=F64)
    stock = F.rms_norm(x, (x.shape[-1],))
    assert_gate(tol.check_forward(ref_ops.rmsnorm(x), ref, stock_out=stock))


def test_rmsnorm_bf16_eps_bug_fails_g1():
    """The gate catches the eps mistake the review found (a kernel using finfo(bf16).eps)."""
    x = randn((64, 768), SEED, "cpu", 0.05)
    ref = ref_ops.rmsnorm(x, dtype=F64)
    wrong = F.rms_norm(x, (768,), eps=torch.finfo(BF16).eps)
    assert not tol.check_forward(wrong, ref, stock_out=F.rms_norm(x, (768,)))


def test_rmsnorm_g2_bf16(rmsnorm_case):
    x = rmsnorm_case["x"]
    D = x.shape[-1]
    gy = randn(x.shape, SEED + 7, x.device)
    (g_stock,) = grads(lambda t: F.rms_norm(t, (D,)), [x], gy)
    (g_ours,) = grads(ref_ops.rmsnorm, [x], gy)
    (g_ref,) = grads(lambda t: ref_ops.rmsnorm(t, dtype=F64), [x.double()], gy.double())
    assert_gate_vs_stock(tol.check_backward(g_ours, g_ref, stock_grad=g_stock, reduction_len=D),
                         tol.check_backward(g_stock, g_ref, stock_grad=g_stock, reduction_len=D))


# --- RoPE ---------------------------------------------------------------------------------------
def test_rotary_tables_match_nanochat():
    cos_n, sin_n = gpt.GPT._precompute_rotary_embeddings(None, R12["T"], R12["D"], device="cpu")
    cos, sin = ref_ops.rotary_tables(R12["T"], R12["D"], dtype=cos_n.dtype)
    assert_bitexact(cos, cos_n)
    assert_bitexact(sin, sin_n)


def test_rope_bitexact_vs_nanochat(rope_case):
    q, cos, sin = rope_case["q"], rope_case["cos"], rope_case["sin"]
    assert_bitexact(ref_ops.rope(q, cos, sin), gpt.apply_rotary_emb(q, cos, sin))


def test_rope_fp64_ref_vs_nanochat_fp32(rope_case):
    q, cos, sin = (rope_case[k].float() for k in ("q", "cos", "sin"))
    assert_gate(tol.check_forward(gpt.apply_rotary_emb(q, cos, sin), ref_ops.rope(q, cos, sin, dtype=F64)))


def test_rope_g1_bf16(rope_case):
    q, cos, sin = rope_case["q"], rope_case["cos"], rope_case["sin"]
    stock = gpt.apply_rotary_emb(q, cos, sin)
    assert_gate(tol.check_forward(ref_ops.rope(q, cos, sin), ref_ops.rope(q, cos, sin, dtype=F64),
                                  stock_out=stock))


@pytest.mark.parametrize("pos0,T", [(1, 16), (2040, 8)])
def test_rope_flipped_sin_fails(pos0, T):
    """Negative test: the textbook (+theta) rotation must NOT pass for nanochat's (-theta) one.
    (Positions > 0 only: at position 0 sin = 0 and both conventions coincide.)"""
    cos, sin = ref_ops.rotary_tables(R12["T"], 128)
    cos, sin = cos[:, pos0:pos0 + T], sin[:, pos0:pos0 + T]
    q = randn((2, T, 6, 128), SEED, "cpu", 0.5)
    stock = gpt.apply_rotary_emb(q, cos, sin)
    ref = ref_ops.rope(q, cos, sin, dtype=F64)
    flipped = ref_ops.rope(q, cos, sin, flip_sin=True)
    assert not torch.equal(flipped, stock)
    r = tol.check_forward(flipped, ref, stock_out=stock)
    assert not r, r.summary()
    assert not tol.check_forward(ref_ops.rope(q.float(), cos, sin, dtype=torch.float32, flip_sin=True),
                                 ref)
    assert tol.check_forward(ref_ops.rope(q, cos, sin), ref, stock_out=stock)  # control


def test_rope_g2_bf16(rope_case):
    q, cos, sin = rope_case["q"], rope_case["cos"], rope_case["sin"]
    gy = randn(q.shape, SEED + 7, q.device)
    (g_stock,) = grads(lambda t: gpt.apply_rotary_emb(t, cos, sin), [q], gy)
    (g_ours,) = grads(lambda t: ref_ops.rope(t, cos, sin), [q], gy)
    (g_ref,) = grads(lambda t: ref_ops.rope(t, cos, sin, dtype=F64), [q.double()], gy.double())
    assert_gate_vs_stock(tol.check_backward(g_ours, g_ref, stock_grad=g_stock),
                         tol.check_backward(g_stock, g_ref, stock_grad=g_stock))


# --- QK-norm + RoPE + scale chain ---------------------------------------------------------------
def test_qkrope_bitexact_vs_causal_self_attention(device, monkeypatch):
    """Run nanochat's CausalSelfAttention.forward and capture the q, k it hands to attention:
    they must equal qkrope(c_q(x), c_k(x)) bit-exactly (order RoPE -> norm -> x1.2)."""
    cfg = gpt.GPTConfig()
    attn = gpt.CausalSelfAttention(cfg, layer_idx=0).to(device)  # layer 0 has no value embedding
    assert attn.ve_gate is None
    B, T, H, D = 2, 16, cfg.n_head, cfg.n_embd // cfg.n_head
    x = randn((B, T, cfg.n_embd), SEED, device, 0.5)
    cos, sin = ref_ops.rotary_tables(R12["T"], D, device=device)
    cos, sin = cos[:, 100:100 + T], sin[:, 100:100 + T]
    captured = {}
    stub_attention(monkeypatch, captured)
    with torch.no_grad():
        attn(x, None, (cos, sin), (R12["T"], 0), None)
        q_lin = attn.c_q(x).view(B, T, H, D)
        k_lin = attn.c_k(x).view(B, T, H, D)
        q_ref, k_ref = ref_ops.qkrope(q_lin, k_lin, cos, sin)
        q_chain, k_chain = _nanochat_qk_chain(q_lin, k_lin, cos, sin)
    # the x1.2 comes after the norm: per-head RMS of q is 1.2, not 1
    rms = captured["q"].float().pow(2).mean(-1).sqrt()
    assert abs(float(rms.mean()) - ref_ops.QK_SCALE) < 1e-2
    assert_bitexact(q_chain, captured["q"])  # the helper below is the module's code path
    assert_bitexact(k_chain, captured["k"])
    xfail_if_cuda_ulp([cuda_ulp_mismatch(q_ref, captured["q"], device, 2),
                       cuda_ulp_mismatch(k_ref, captured["k"], device, 2)],
                      "the order RoPE -> norm -> x1.2 is proven bit-exactly by the [cpu] case")


def _nanochat_qk_chain(q, k, cos, sin):
    """The exact lines of CausalSelfAttention.forward (proven equal to it by the test above)."""
    q, k = gpt.apply_rotary_emb(q, cos, sin), gpt.apply_rotary_emb(k, cos, sin)
    q, k = gpt.norm(q), gpt.norm(k)
    return q * 1.2, k * 1.2


def test_qkrope_g1(rope_case):
    q, k, cos, sin = (rope_case[n] for n in ("q", "k", "cos", "sin"))
    q64, k64 = ref_ops.qkrope(q, k, cos, sin, dtype=F64)
    qs, ks = _nanochat_qk_chain(q, k, cos, sin)
    qo, ko = ref_ops.qkrope(q, k, cos, sin)
    # fp32 checks first, so an xfail below (stock bf16 itself misses rtol/atol: job 3406282 saw
    # 4/25.2M q elements at max-abs 5.1e-2 -- bf16 RoPE rounding amplified by rstd) never hides them
    qf, kf = _nanochat_qk_chain(q.float(), k.float(), cos.float(), sin.float())
    assert_gate(tol.check_forward(qf, q64, reduction_len=q.shape[-1]))
    assert_gate(tol.check_forward(kf, k64, reduction_len=k.shape[-1]))
    # the stock-side result needs stock_out too, else it also fails "stock error required" and never xfails
    assert_gate_vs_stock(tol.check_forward(qo, q64, stock_out=qs), tol.check_forward(qs, q64, stock_out=qs))
    assert_gate_vs_stock(tol.check_forward(ko, k64, stock_out=ks), tol.check_forward(ks, k64, stock_out=ks))


def test_qkrope_g2_bf16(rope_case):
    q, k, cos, sin = (rope_case[n] for n in ("q", "k", "cos", "sin"))
    gq, gk = randn(q.shape, SEED + 7, q.device), randn(k.shape, SEED + 8, k.device)

    def run(fn, a, b, ga, gb):
        a, b = a.detach().clone().requires_grad_(True), b.detach().clone().requires_grad_(True)
        oa, ob = fn(a, b)
        torch.autograd.backward([oa, ob], [ga, gb])
        return a.grad, b.grad

    s = run(lambda a, b: _nanochat_qk_chain(a, b, cos, sin), q, k, gq, gk)
    o = run(lambda a, b: ref_ops.qkrope(a, b, cos, sin), q, k, gq, gk)
    r = run(lambda a, b: ref_ops.qkrope(a, b, cos, sin, dtype=F64), q.double(), k.double(),
            gq.double(), gk.double())
    for g_o, g_r, g_s in zip(o, r, s):
        D = q.shape[-1]
        assert_gate_vs_stock(tol.check_backward(g_o, g_r, stock_grad=g_s, reduction_len=D),
                             tol.check_backward(g_s, g_r, stock_grad=g_s, reduction_len=D))


# --- softcap + cross-entropy --------------------------------------------------------------------
class _ConstHead(nn.Module):
    """Stands in for GPT.lm_head: returns fixed logits so GPT.forward's own loss lines run on them."""
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, x):
        return self.logits


def nanochat_loss_path(logits, targets, reduction, monkeypatch):
    """Run the real GPT.forward of a 1-layer toy model (CPU) whose lm_head returns `logits`."""
    N, V = logits.shape
    cfg = gpt.GPTConfig(sequence_len=N, vocab_size=V, n_layer=1, n_head=1, n_kv_head=1, n_embd=32,
                        window_pattern="L")
    model = gpt.GPT(cfg)  # default nn init on CPU; the trunk output is discarded by _ConstHead
    model.lm_head = _ConstHead(logits.cpu().view(1, N, V))
    stub_attention(monkeypatch)
    with torch.no_grad():
        return model(torch.zeros((1, N), dtype=torch.long), targets.cpu().view(1, N),
                     loss_reduction=reduction)


def stock_ce(logits, targets, reduction="mean"):
    """GPT.forward's loss lines verbatim (equality with GPT.forward is tested on CPU)."""
    lg = logits.float()
    lg = 15 * torch.tanh(lg / 15)
    return F.cross_entropy(lg.view(-1, lg.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction)


CPU_CE_CASES = [p for p in CE_CASES if p.values[0][-1] == "cpu"]


@pytest.mark.parametrize("ce_case", CPU_CE_CASES, indirect=True)
def test_softcap_ce_vs_nanochat_loss_path(ce_case, monkeypatch):
    """CPU only (the toy GPT runs on CPU); GPU/real-shape cases compare against stock_ce, which this
    test proves is GPT.forward's loss path."""
    lg, tg = ce_case["logits"], ce_case["targets"]
    for red in ("mean", "none"):
        nano = nanochat_loss_path(lg, tg, red, monkeypatch)
        assert_bitexact(stock_ce(lg, tg, red), nano)  # stock_ce is the loss path
        ours32 = ref_ops.softcap_ce(lg, tg, reduction=red)
        ours64 = ref_ops.softcap_ce(lg, tg, reduction=red, dtype=F64)
        assert_gate(tol.check_forward(ours32, nano.double(), kind="ce_loss"))
        assert_gate(tol.check_forward(nano, ours64, kind="ce_loss"))  # fp32 path meets rel 1e-5
    if ce_case["name"] == "all_ignored":
        assert math.isnan(float(ref_ops.softcap_ce(lg, tg))) and math.isnan(float(stock_ce(lg, tg)))
        assert bool((ref_ops.softcap_ce(lg, tg, reduction="none") == 0).all())


def test_softcap_ce_g1(ce_case):
    lg, tg = ce_case["logits"], ce_case["targets"]
    with torch.no_grad():
        for red in ("mean", "none"):
            ref = ref_ops.softcap_ce(lg, tg, reduction=red, dtype=F64, chunk_rows=2048)
            assert_gate(tol.check_forward(stock_ce(lg, tg, red), ref, kind="ce_loss"))
            assert_gate(tol.check_forward(ref_ops.softcap_ce(lg, tg, reduction=red, chunk_rows=2048), ref,
                                          kind="ce_loss"))
            if red == "none":
                assert bool((ref[ce_case["ignore"]] == 0).all())


def test_softcap_saturation_bounded():
    lg = torch.tensor([[80.0, -75.0, 60.0, 0.0], [55.0, 51.0, -52.0, 1.0]], dtype=BF16)
    tg = torch.tensor([0, 2])
    z = ref_ops._softcap(lg, ref_ops.SOFTCAP, F64)
    assert bool((z.abs() < ref_ops.SOFTCAP).all()) and float(z.abs().max()) > 14.9
    loss = ref_ops.softcap_ce(lg, tg, reduction="none", dtype=F64)
    assert bool(torch.isfinite(loss).all()) and float(loss[1]) > 25.0  # target at -15: large loss


def test_softcap_ce_g2(ce_case):
    lg, tg, ign, V = ce_case["logits"], ce_case["targets"], ce_case["ignore"], ce_case["V"]
    if bool(ign.all()):
        pytest.skip("mean over zero valid rows is NaN; no gradient to compare")
    leaf = lg.detach().clone().requires_grad_(True)
    stock_ce(leaf, tg).backward()
    g_stock = leaf.grad
    del leaf
    g_ref = ref_ops.softcap_ce_grad(lg, tg, chunk_rows=2048)
    r = tol.check_backward(g_stock, g_ref, stock_grad=g_stock, kind="dlogits", ignore_rows=ign)
    assert_gate(r)
    assert r.ignore_rows_zero is True


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_softcap_ce_grad_analytic_matches_autograd(reduction):
    g = gen(SEED, "cpu")
    lg = torch.randn((9, 13), generator=g, dtype=F64) * 20
    tg = torch.randint(0, 13, (9,), generator=g)
    tg[[2, 5]] = -1
    go = torch.randn((9,), generator=g, dtype=F64) if reduction == "none" else torch.tensor(1.7, dtype=F64)
    (g_auto,) = grads(lambda t: ref_ops.softcap_ce(t, tg, reduction=reduction, dtype=F64), [lg], go)
    g_ana = ref_ops.softcap_ce_grad(lg, tg, reduction=reduction, chunk_rows=4, grad_out=go)
    torch.testing.assert_close(g_ana, g_auto, rtol=1e-12, atol=1e-14)
    assert bool((g_ana[[2, 5]] == 0).all())


# --- relu^2 and residual mix --------------------------------------------------------------------
def test_relu2_bitexact_vs_nanochat_mlp(device):
    mlp = gpt.MLP(gpt.GPTConfig()).to(device)
    mlp.c_fc, mlp.c_proj = nn.Identity(), nn.Identity()  # leaves exactly F.relu(x).square()
    x = randn((64, R12["F"]), SEED, device, 0.5)
    with torch.no_grad():
        assert_bitexact(ref_ops.relu2(x), mlp(x))


def test_relu2_g1_g2(relu2_case):
    x = relu2_case["x"]
    stock_fn = lambda t: F.relu(t).square()
    assert_gate(tol.check_forward(ref_ops.relu2(x), ref_ops.relu2(x, dtype=F64), stock_out=stock_fn(x)))
    gy = randn(x.shape, SEED + 7, x.device)
    (g_stock,) = grads(stock_fn, [x], gy)
    (g_ours,) = grads(ref_ops.relu2, [x], gy)
    (g_ref,) = grads(lambda t: ref_ops.relu2(t, dtype=F64), [x.double()], gy.double())
    assert_gate_vs_stock(tol.check_backward(g_ours, g_ref, stock_grad=g_stock),
                         tol.check_backward(g_stock, g_ref, stock_grad=g_stock))


def test_residual_mix_bitexact_vs_nanochat(resid_case):
    x, x0, i = resid_case["x"], resid_case["x0"], resid_case["layer"]
    r_all, l_all = resid_case["r_all"], resid_case["l_all"]
    with torch.no_grad():
        nano = r_all[i] * x + l_all[i] * x0  # GPT.forward, verbatim
        assert_bitexact(ref_ops.residual_mix(x, x0, r_all[i], l_all[i]), nano)
        assert_bitexact(ref_ops.residual_mix(x, x0, float(r_all[i]), float(l_all[i])), nano)


def test_residual_mix_g1_g2(resid_case):
    x, x0, i = resid_case["x"], resid_case["x0"], resid_case["layer"]
    r, l = resid_case["r_all"][i].detach(), resid_case["l_all"][i].detach()
    with torch.no_grad():
        stock = r * x + l * x0
        ref = ref_ops.residual_mix(x, x0, r, l, dtype=F64)
        assert_gate(tol.check_forward(ref_ops.residual_mix(x, x0, r, l), ref, stock_out=stock))
    gy = randn(x.shape, SEED + 7, x.device)
    g_stock = grads(lambda a, b: r * a + l * b, [x, x0], gy)
    g_ours = grads(lambda a, b: ref_ops.residual_mix(a, b, r, l), [x, x0], gy)
    g_ref = grads(lambda a, b: ref_ops.residual_mix(a, b, r, l, dtype=F64), [x.double(), x0.double()],
                  gy.double())
    for g_o, g_r, g_s in zip(g_ours, g_ref, g_stock):
        assert_gate_vs_stock(tol.check_backward(g_o, g_r, stock_grad=g_s),
                             tol.check_backward(g_s, g_r, stock_grad=g_s))


# --- fp64 gradcheck on tiny shapes (G2 first half) ----------------------------------------------
def _leaf(shape, seed, scale=1.0):
    return (torch.randn(shape, generator=gen(seed, "cpu"), dtype=F64) * scale).requires_grad_(True)


def _angles(T, half, seed):
    a = torch.rand((1, T, 1, half), generator=gen(seed, "cpu"), dtype=F64) * 6.28
    return a.cos(), a.sin()


def test_gradcheck_rmsnorm():
    assert torch.autograd.gradcheck(lambda x: ref_ops.rmsnorm(x, dtype=F64), (_leaf((3, 8), 1),))


def test_gradcheck_rope():
    cos, sin = _angles(3, 4, 2)
    assert torch.autograd.gradcheck(lambda x: ref_ops.rope(x, cos, sin, dtype=F64), (_leaf((1, 3, 2, 8), 3),))


def test_gradcheck_qkrope():
    cos, sin = _angles(3, 4, 4)
    fn = lambda q, k: ref_ops.qkrope(q, k, cos, sin, dtype=F64)
    assert torch.autograd.gradcheck(fn, (_leaf((1, 3, 2, 8), 5), _leaf((1, 3, 2, 8), 6)))


@pytest.mark.parametrize("reduction,chunk", [("mean", None), ("none", None), ("mean", 2), ("sum", 3)])
def test_gradcheck_softcap_ce(reduction, chunk):
    tg = torch.tensor([3, -1, 0, 6, 2])
    fn = lambda z: ref_ops.softcap_ce(z, tg, reduction=reduction, dtype=F64, chunk_rows=chunk)
    assert torch.autograd.gradcheck(fn, (_leaf((5, 7), 7, scale=20.0),))


def test_gradcheck_relu2():
    assert torch.autograd.gradcheck(lambda x: ref_ops.relu2(x, dtype=F64), (_leaf((4, 6), 8),))


def test_gradcheck_residual_mix():
    fn = lambda x, x0, r, l: ref_ops.residual_mix(x, x0, r, l, dtype=F64)
    args = (_leaf((2, 3, 4), 9), _leaf((2, 3, 4), 10), _leaf((), 11), _leaf((), 12))
    assert torch.autograd.gradcheck(fn, args)


def test_gradcheck_attention_ref():
    fn = lambda q, k, v: ref_ops.attention(q, k, v, causal=True, dtype=F64)
    assert torch.autograd.gradcheck(fn, tuple(_leaf((1, 4, 2, 8), s) for s in (13, 14, 15)))


def test_attention_ref_vs_explicit_softmax():
    q, k, v = (torch.randn((2, 5, 3, 8), generator=gen(s, "cpu"), dtype=F64) for s in (16, 17, 18))
    y = ref_ops.attention(q, k, v, causal=True, dtype=F64)
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    s = qt @ kt.transpose(-1, -2) / math.sqrt(8)
    s = s.masked_fill(torch.ones(5, 5, dtype=torch.bool).triu(1), float("-inf"))
    torch.testing.assert_close(y, (s.softmax(-1) @ vt).transpose(1, 2), rtol=1e-12, atol=1e-12)


# --- tolerance code unit tests ------------------------------------------------------------------
def test_tol_fp32_rules():
    ref = torch.randn(1000, generator=gen(1, "cpu"), dtype=F64) + 3.0
    assert tol.check_forward((ref * (1 + 5e-6)).float(), ref)
    assert not tol.check_forward((ref * (1 + 5e-5)).float(), ref)
    near = (ref * (1 + 1.5e-5)).float()  # between the 1e-5 and 2e-5 rtol
    assert not tol.check_forward(near, ref, reduction_len=768)
    assert tol.check_forward(near, ref, reduction_len=tol.LONG_REDUCTION)


def test_tol_bf16_stock_rule():
    ref = torch.randn(4096, generator=gen(2, "cpu"), dtype=F64)
    stock = ref.to(BF16)                     # error = bf16 rounding
    e_stock = float((stock.double() - ref).abs().max())
    # 4 half-ulps of relative error: inside rtol 1.6e-2 everywhere, but ~4x the stock op's max error
    sign = torch.sign(torch.randn(4096, generator=gen(3, "cpu"), dtype=F64))
    worse = (ref * (1 + 4 * 2.0 ** -9 * sign)).to(BF16)
    assert tol.check_forward(stock, ref, stock_out=stock)
    r = tol.check_forward(worse, ref, stock_out=stock)
    assert not r and r.stock_ratio > 2.0 and r.n_violations == 0  # inside rtol/atol, caught by 2x stock
    r = tol.check_forward(stock, ref)          # no stock -> cannot pass the bf16 rule
    assert not r and "stock error required" in r.reasons[0]
    assert tol.check_forward(stock, ref, stock_error=e_stock)


def test_tol_gemm():
    a = randn((64, 96), 4, "cpu", 0.5)
    w = randn((48, 96), 5, "cpu", 1 / math.sqrt(96))
    ref = a.double() @ w.double().T
    stock = F.linear(a, w)
    r = tol.check_forward((a.float() @ w.float().T).to(BF16), ref, stock_out=stock, kind="gemm")
    assert r and r.rel_fro < 1e-2, r.summary()
    noisy = (ref * (1 + 0.02 * torch.randn(ref.shape, generator=gen(6, "cpu"), dtype=F64))).to(BF16)
    assert not tol.check_forward(noisy, ref, stock_out=stock, kind="gemm")


def test_tol_attention():
    ref = torch.randn((2, 8, 3, 16), generator=gen(7, "cpu"))
    noise = torch.rand(ref.shape, generator=gen(8, "cpu")) * 2 - 1
    assert tol.check_forward(ref + 1.5e-3 * noise, ref, kind="attention", stock_error=1e-3)
    assert not tol.check_forward(ref + 2.5e-3 * noise, ref, kind="attention", stock_error=1e-3)  # > 2x
    # A1: the absolute 4e-3 term is gone; only "<= 2x FA3's own error" binds
    assert tol.check_forward(ref + 5e-3 * noise, ref, kind="attention", stock_error=4e-3)
    assert not tol.check_forward(ref + 9e-3 * noise, ref, kind="attention", stock_error=4e-3)    # > 2x


def test_tol_bf16_amendment_a1():
    """PLAN 7.3 A1 for bf16 outputs judged against a stock output: rtol/atol violations may be at most
    2x stock's count (strict when stock has none), max-abs <= 2x stock, cosine >= 0.9999."""
    ref = torch.randn(4096, 128, generator=gen(11, "cpu"), dtype=F64)
    stock = ref.to(BF16)
    clean = ref.to(BF16)
    assert tol.check_forward(clean, ref, stock_out=stock)                 # stock meets rtol/atol
    one_bad = clean.clone(); one_bad[0, 0] += 0.05
    r = tol.check_forward(one_bad, ref, stock_out=stock)
    assert not r and r.stock_violations == 0, r.summary()                 # strict when stock is clean
    stock_bad = clean.clone(); stock_bad[:2, 0] += 0.05                   # stock: 2 violations
    ours_ok = clean.clone(); ours_ok[2:6, 0] += 0.05                      # 4 <= 2x2, same max-abs
    assert tol.check_forward(ours_ok, ref, stock_out=stock_bad)
    ours_many = clean.clone(); ours_many[2:7, 0] += 0.05                  # 5 > 2x2
    assert not tol.check_forward(ours_many, ref, stock_out=stock_bad)
    ours_big = clean.clone(); ours_big[2, 0] += 0.2                       # max-abs > 2x stock's
    assert not tol.check_forward(ours_big, ref, stock_out=stock_bad)
    # without a stock output the rtol/atol rule is strict (and the stock rule cannot be applied)
    assert not tol.check_forward(one_bad, ref)


def test_tol_dlogits_a1():
    ref = torch.randn(64, 512, generator=gen(12, "cpu"), dtype=F64) * 1e-6
    noise = torch.randn(ref.shape, generator=gen(13, "cpu"), dtype=F64)
    stock = (ref * (1 + 4e-3 * noise)).to(BF16)
    assert tol.check_backward((ref * (1 + 5e-3 * noise)).to(BF16), ref, stock_grad=stock, kind="dlogits")
    assert not tol.check_backward((ref * (1 + 3e-2 * noise)).to(BF16), ref, stock_grad=stock, kind="dlogits")


def test_tol_ce_and_nan():
    ref = torch.tensor(2.5, dtype=F64)
    assert tol.check_forward(torch.tensor(2.5 * (1 + 5e-6)), ref, kind="ce_loss")
    assert not tol.check_forward(torch.tensor(2.5 * (1 + 5e-5)), ref, kind="ce_loss")
    nan = torch.tensor(float("nan"))
    assert tol.check_forward(nan, nan.double(), kind="ce_loss")       # all-ignored batch
    assert not tol.check_forward(nan, ref, kind="ce_loss")
    with pytest.raises(ValueError):
        tol.check_forward(torch.zeros(3), torch.zeros(4, dtype=F64))


def test_tol_backward_cosine_and_ignore_rows():
    ref = torch.randn((6, 10), generator=gen(9, "cpu"), dtype=F64)
    ref[[1, 4]] = 0
    ign = torch.zeros(6, dtype=torch.bool)
    ign[[1, 4]] = True
    r = tol.check_backward(ref.float(), ref, ignore_rows=ign)
    assert r and r.cosine > 0.999999 and r.ignore_rows_zero
    leak = ref.float().clone()
    leak[1, 3] = 1e-9                         # tiny, inside atol, but not exactly zero
    r = tol.check_backward(leak, ref, ignore_rows=ign)
    assert not r and r.ignore_rows_zero is False
    assert not tol.check_backward((-ref).float(), ref)  # cosine -1
    assert tol.cosine(torch.zeros(3), torch.zeros(3, dtype=F64)) == 1.0


# --- gates.record_gate --------------------------------------------------------------------------
def test_record_gate(tmp_path):
    p = tmp_path / "case.json"
    row = {"case": "x", "median_us": 1.5, "gates_passed": [], "env": {"a": 1}, "job_id": "123"}
    p.write_text(json.dumps(row))
    assert gates.record_gate(str(p), "G1") == ["G1"]
    assert gates.record_gate(str(p), "G1") == ["G1"]  # idempotent
    assert gates.record_gate(p, "G2") == ["G1", "G2"]
    got = json.loads(p.read_text())
    assert got == {**row, "gates_passed": ["G1", "G2"]}
    assert gates.has_gates(str(p)) and not gates.has_gates(got, gates.REQUIRED_FOR_END_TO_END)
    assert list(tmp_path.iterdir()) == [p]           # no temp files left behind
    q = tmp_path / "nogates.json"
    q.write_text(json.dumps({"case": "y"}))
    assert gates.record_gate(str(q), "G3") == ["G3"]
    with pytest.raises(ValueError):
        gates.record_gate(str(p), "G9")
    with pytest.raises(FileNotFoundError):
        gates.record_gate(str(tmp_path / "missing.json"), "G1")


# --- GPU job only: real-shape GEMM and the attention stock error --------------------------------
@pytest.mark.real_shape
@pytest.mark.gpu
@pytest.mark.parametrize("n_out", [768, 3072])
def test_gemm_cublas_within_gate(cuda, n_out):
    a = randn((R12["N"], R12["C"]), SEED, cuda, 0.5)
    w = randn((n_out, R12["C"]), SEED + 1, cuda, 1 / math.sqrt(R12["C"]))
    stock = F.linear(a, w)
    ref = a.double() @ w.double().T
    r = tol.check_forward(stock, ref, stock_out=stock, kind="gemm")
    print(f"cuBLAS 32768x768x{n_out}: {r.summary()}")
    assert_gate(r)


ATTN_SHAPE = (R12["B"], R12["T"], R12["H"], R12["D"])


def _write_attention_stock_error(payload):
    """results/db/tolerances/attention_stock_error_<jobid>.json always; the canonical
    attention_stock_error.json only if absent (the first GPU measurement is the frozen stock error;
    later jobs never overwrite it)."""
    d = os.path.dirname(tol.ATTN_STOCK_ERROR_PATH)
    os.makedirs(d, exist_ok=True)
    jid = payload["job_id"] or "nojob_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(d, f"attention_stock_error_{jid}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    if not os.path.exists(tol.ATTN_STOCK_ERROR_PATH):
        with open(tol.ATTN_STOCK_ERROR_PATH, "w") as f:
            json.dump(payload, f, indent=2)


@pytest.mark.real_shape
@pytest.mark.gpu
def test_attention_fa3_stock_error(cuda):
    """Measure FA3's max-abs error vs the fp32 math-backend SDPA reference at (16, 2048, 6, 128)
    causal (the attention gate's "<= 2x FA3's own error") and store it. Fails if FA3 is far above the
    bf16 output-rounding floor. Under PLAN 7.3 A1 the 4e-3 absolute term is reported only (job 3406282:
    FA3 9.1e-3, rounding floor 7.8e-3); the binding attention rule is <= 2x this stored error."""
    from nanochat import flash_attention as fa
    if not fa.HAS_FA3:
        pytest.skip("FA3 not available (needs the cached varunneal/flash-attention-3 kernel, HF_HUB_OFFLINE=1)")
    B, T, H, D = ATTN_SHAPE
    per_seed = []
    with torch.no_grad():
        for s in range(3):
            # inputs as the model sees them: q, k after QK-norm and x1.2, v at activation scale
            q = ref_ops.rmsnorm(randn(ATTN_SHAPE, SEED + 10 * s, cuda)) * ref_ops.QK_SCALE
            k = ref_ops.rmsnorm(randn(ATTN_SHAPE, SEED + 10 * s + 1, cuda)) * ref_ops.QK_SCALE
            v = randn(ATTN_SHAPE, SEED + 10 * s + 2, cuda, 1.0)
            ref = ref_ops.attention(q, k, v, causal=True, dtype=torch.float32)
            outs = {
                "fa3_causal": fa._fa3.flash_attn_func(q, k, v, causal=True),
                "fa3_nanochat_call": fa._fa3.flash_attn_func(q, k, v, causal=True, window_size=(T, 0)),
                "sdpa_bf16": F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)),
                                                            is_causal=True).transpose(1, 2),
            }
            outs = {n: (o[0] if isinstance(o, tuple) else o) for n, o in outs.items()}
            # floor: rounding the exact (fp32-reference) output to bf16 -- no bf16 kernel can beat it
            row = {"seed": SEED + 10 * s, "ref_max_abs_value": float(ref.abs().max()),
                   "bf16_output_rounding_floor": tol.measure(ref.to(BF16), ref)["max_abs"]}
            for n, o in outs.items():
                st = tol.measure(o, ref)
                row[f"{n}_max_abs"] = st["max_abs"]
                row[f"{n}_rmse"] = math.sqrt(st["diff2"] / st["numel"])
            row["fa3_vs_sdpa_bf16_max_abs"] = tol.measure(outs["fa3_causal"], outs["sdpa_bf16"])["max_abs"]
            per_seed.append(row)
            del ref, outs
            torch.cuda.empty_cache()
    fa3_err = max(max(r["fa3_causal_max_abs"], r["fa3_nanochat_call_max_abs"]) for r in per_seed)
    payload = {
        "case": "attention_fa3_stock_error", "shape_BTHD": list(ATTN_SHAPE), "causal": True,
        "dtype": "bfloat16", "reference": "SDPA math backend, fp32, same bf16 inputs",
        "fa3_max_abs_err": fa3_err, "per_seed": per_seed,
        "sdpa_bf16_max_abs_err": max(r["sdpa_bf16_max_abs"] for r in per_seed),
        "fa3_vs_sdpa_bf16_max_abs": max(r["fa3_vs_sdpa_bf16_max_abs"] for r in per_seed),
        "bf16_output_rounding_floor": max(r["bf16_output_rounding_floor"] for r in per_seed),
        "fa3_meets_abs_bound": fa3_err <= tol.ATTN_MAX_ABS,
        "gate": {"amendment": tol.AMENDMENT, "stock_factor": tol.ATTN_STOCK_FACTOR,
                 "original_max_abs_reported_only": tol.ATTN_MAX_ABS},
        "job_id": job_id(), "device": torch.cuda.get_device_name(),
        "torch": torch.__version__, "utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }
    _write_attention_stock_error(payload)
    print(json.dumps({k: payload[k] for k in ("fa3_max_abs_err", "sdpa_bf16_max_abs_err",
                                              "fa3_vs_sdpa_bf16_max_abs", "bf16_output_rounding_floor",
                                              "job_id")}))
    floor = payload["bf16_output_rounding_floor"]
    assert fa3_err <= 4 * max(floor, tol.ATTN_MAX_ABS), (
        f"FA3's error {fa3_err:.3e} is far above the bf16 rounding floor {floor:.3e}: wrong call/inputs?")
