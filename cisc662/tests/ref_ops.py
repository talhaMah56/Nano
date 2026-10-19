"""
Pure-PyTorch reference implementations of the nanochat ops we replace with custom kernels.

Every function takes an optional `dtype`:
  - dtype=None   reproduce nanochat's own eager computation exactly (same dtype, same op order,
                 same rounding points) -- used to prove the reference *is* nanochat's function;
  - dtype=torch.float64 (or float32) cast all inputs once and compute everything in that dtype --
                 the G1/G2 reference of PLAN.md section 7.3 ("fp64 reference on the same inputs").
All functions are differentiable (torch.autograd.gradcheck runs on them in fp64), except the
chunked analytic softcap_ce_grad, which exists so the 32768x32768 fp64 reference fits in memory.

Sources (nanochat/gpt.py at the cisc662 branch point):
  rmsnorm      norm(x) = F.rms_norm(x, (x.size(-1),)) with eps=None. In torch 2.9.1 eps=None means
               torch.finfo(opmath dtype).eps = finfo(float32).eps = 1.1920929e-07 for fp32/bf16/fp16
               inputs (eager CPU/CUDA kernels and the Inductor path) -- NOT finfo(bfloat16).eps.
  rope         apply_rotary_emb: non-interleaved halves, y1 = x1*cos + x2*sin, y2 = x1*(-sin) + x2*cos
  qkrope       CausalSelfAttention.forward: rope(q), rope(k) -> norm(q), norm(k) -> q*1.2, k*1.2
  softcap_ce   GPT.forward: logits.float(); 15*tanh(logits/15); F.cross_entropy(ignore_index=-1)
  relu2        MLP.forward: F.relu(x).square()
  residual_mix GPT.forward: resid_lambdas[i] * x + x0_lambdas[i] * x0 (lambdas are fp32 params)
  attention    fp32 math-backend SDPA (the attention gate's reference; not a nanochat function)

Usage:  from cisc662.tests import ref_ops;  y64 = ref_ops.rmsnorm(x, dtype=torch.float64)
"""
import torch
import torch.nn.functional as F

# F.rms_norm(eps=None) in torch 2.9.1 uses the epsilon of the *opmath* type (float for bf16/fp16).
RMS_EPS = torch.finfo(torch.float32).eps
SOFTCAP = 15.0
QK_SCALE = 1.2
IGNORE_INDEX = -1

_LOW_PRECISION = (torch.bfloat16, torch.float16)


def _opmath(dtype):
    """Accumulation dtype PyTorch's eager kernels use for `dtype` (float for bf16/fp16)."""
    return torch.float32 if dtype in _LOW_PRECISION else dtype


def _cast(t, dtype):
    return t if dtype is None or not torch.is_tensor(t) else t.to(dtype)


def rmsnorm(x, eps=None, dtype=None):
    """RMSNorm over the last dim, no weight. dtype=None returns x.dtype, bit-matching F.rms_norm.
    eps=None: RMS_EPS (fp32 epsilon, what F.rms_norm uses for the bf16/fp16/fp32 inputs we gate),
    except for an fp64 input in dtype=None mode, where F.rms_norm uses finfo(float64).eps."""
    if dtype is None:
        xc, out_dtype = x.to(_opmath(x.dtype)), x.dtype
        if eps is None and x.dtype == torch.float64:
            eps = torch.finfo(torch.float64).eps
    else:
        xc, out_dtype = x.to(dtype), dtype
    eps = RMS_EPS if eps is None else eps
    rstd = torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + eps)
    return (xc * rstd).to(out_dtype)


def rope(x, cos, sin, dtype=None, flip_sin=False):
    """nanochat.gpt.apply_rotary_emb. x: (B, T, H, D); cos/sin: (1, T, 1, D/2).

    flip_sin=True gives the *textbook* (+theta) rotation; it exists only for the negative test that
    proves the gates can tell the two sign conventions apart.
    """
    assert x.ndim == 4, "rope expects (B, T, H, D)"
    x, cos, sin = _cast(x, dtype), _cast(cos, dtype), _cast(sin, dtype)
    if flip_sin:
        sin = -sin
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def qkrope(q, k, cos, sin, scale=QK_SCALE, dtype=None):
    """The QK chain of CausalSelfAttention.forward: RoPE -> RMSNorm -> x scale (in that order)."""
    q, k = rope(q, cos, sin, dtype=dtype), rope(k, cos, sin, dtype=dtype)
    q, k = rmsnorm(q, dtype=dtype), rmsnorm(k, dtype=dtype)
    return q * scale, k * scale


def _softcap(logits, cap, dtype):
    z = logits.to(torch.float32 if dtype is None else dtype)  # nanochat: logits.float()
    return cap * torch.tanh(z / cap)


def softcap_ce(logits, targets, cap=SOFTCAP, ignore_index=IGNORE_INDEX, reduction="mean",
               dtype=None, chunk_rows=None):
    """Softcapped cross-entropy, nanochat's loss path, written out explicitly (logsumexp - gather)
    so that it is an independent check of F.cross_entropy.

    logits: (..., V), targets: (...). dtype=None computes in fp32 as nanochat does.
    reduction="mean": mean over rows whose target != ignore_index (NaN if there are none, like
    F.cross_entropy). reduction="none": per-row loss, exactly 0 on ignored rows. reduction="sum".
    chunk_rows: evaluate in row chunks (no autograd graph kept across chunks if called under
    torch.no_grad) so an fp64 reference of the (32768, 32768) case needs O(chunk) extra memory.
    """
    V = logits.shape[-1]
    lg, tg = logits.reshape(-1, V), targets.reshape(-1)
    n = lg.shape[0]
    step = n if chunk_rows is None else chunk_rows
    rows = []
    for s in range(0, n, step):
        z = _softcap(lg[s:s + step], cap, dtype)
        t = tg[s:s + step]
        valid = t != ignore_index
        safe_t = torch.where(valid, t, torch.zeros_like(t))
        nll = torch.logsumexp(z, dim=-1) - z.gather(1, safe_t[:, None]).squeeze(1)
        rows.append(torch.where(valid, nll, torch.zeros_like(nll)))
    per_row = torch.cat(rows) if len(rows) > 1 else rows[0]
    if reduction == "none":
        return per_row.reshape(targets.shape)
    if reduction == "sum":
        return per_row.sum()
    if reduction == "mean":
        return per_row.sum() / (tg != ignore_index).sum().to(per_row.dtype)
    raise ValueError(f"unknown reduction {reduction!r}")


@torch.no_grad()
def softcap_ce_grad(logits, targets, cap=SOFTCAP, ignore_index=IGNORE_INDEX, reduction="mean",
                    dtype=torch.float64, chunk_rows=4096, grad_out=None):
    """Analytic d(loss)/d(logits) of softcap_ce, computed in row chunks (G2 reference at real shape).

    d/dx of cap*tanh(x/cap) is 1 - tanh(x/cap)^2, so dlogits = (softmax(z) - onehot(t)) * (1 - th^2)
    * g, with g = grad_out / n_valid for "mean", grad_out for "sum", grad_out[row] for "none".
    Ignored rows are exactly 0. Returns a tensor of logits' shape in `dtype`.
    """
    V = logits.shape[-1]
    lg, tg = logits.reshape(-1, V), targets.reshape(-1)
    n = lg.shape[0]
    out = torch.empty(lg.shape, dtype=dtype, device=logits.device)
    n_valid = (tg != ignore_index).sum().to(dtype)
    if reduction == "none":
        g_rows = (torch.ones(n, dtype=dtype, device=logits.device) if grad_out is None
                  else grad_out.reshape(-1).to(dtype))
    else:
        g = 1.0 if grad_out is None else grad_out.to(dtype)
        g_rows = torch.full((n,), 1.0, dtype=dtype, device=logits.device) * g
        if reduction == "mean":
            g_rows = g_rows / n_valid
    for s in range(0, n, chunk_rows):
        th = torch.tanh(lg[s:s + chunk_rows].to(dtype) / cap)
        z = cap * th
        t = tg[s:s + chunk_rows]
        valid = t != ignore_index
        p = torch.softmax(z, dim=-1)
        rows = torch.arange(p.shape[0], device=p.device)
        p[rows[valid], t[valid]] -= 1.0
        p *= (1.0 - th * th)
        p *= (g_rows[s:s + chunk_rows] * valid.to(dtype))[:, None]
        out[s:s + chunk_rows] = p
    return out.reshape(logits.shape)


def relu2(x, dtype=None):
    """MLP activation: relu(x)^2."""
    return F.relu(_cast(x, dtype)).square()


def residual_mix(x, x0, r, l, dtype=None):
    """r * x + l * x0 (GPT.forward, start of every layer).

    dtype=None reproduces nanochat exactly: r and l are fp32 parameters (0-dim tensors after
    indexing) multiplying bf16 activations, so the result stays in x.dtype. Python floats are
    accepted and turned into 0-dim fp32 tensors so the rounding matches the model.
    """
    if dtype is None:
        def as_param(v):
            return v if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32, device=x.device)
        return as_param(r) * x + as_param(l) * x0
    x, x0 = x.to(dtype), x0.to(dtype)
    r = r.to(dtype) if torch.is_tensor(r) else r
    l = l.to(dtype) if torch.is_tensor(l) else l
    return r * x + l * x0


def attention(q, k, v, causal=True, softmax_scale=None, dtype=torch.float32):
    """Attention gate reference: SDPA with the math backend in `dtype` (fp32 per PLAN.md 7.3).
    q, k, v: (B, T, H, D) like FA3. Returns (B, T, H, D) in `dtype`."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    qt, kt, vt = (t.to(dtype).transpose(1, 2) for t in (q, k, v))
    # The math backend runs bmm: force true fp32 (no TF32) even if something in the process called
    # torch.set_float32_matmul_precision("high") (nanochat's compute_init does), else the
    # "fp32 reference" carries ~1e-3 TF32 error -- the size of the FA3 error it is meant to measure.
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal, scale=softmax_scale)
    finally:
        torch.set_float32_matmul_precision(prev)
    return y.transpose(1, 2)


def rotary_tables(seq_len, head_dim, base=100000, device="cpu", dtype=torch.bfloat16):
    """cos/sin exactly as GPT._precompute_rotary_embeddings (fp32 math, cast to the compute dtype),
    shaped (1, T, 1, D/2). Kept here so tests do not need a GPT instance; test_ref_ops checks it
    against nanochat's own method."""
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
    return cos[None, :, None, :], sin[None, :, None, :]
