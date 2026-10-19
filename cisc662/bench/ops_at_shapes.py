"""
Every target op of nanochat R12 at its exact in-model shape, as kbench cases (task INF-3; PLAN.md §3 OPS).

Forms: eager (the python function as nanochat runs it), compile (torch.compile(fn, dynamic=False) of the
same function; Dynamo/Inductor compile happens in kbench's untimed first call) and autotune
(torch.compile(mode="max-autotune-no-cudagraphs"), GEMM ops only, used from S4/BAS-4 on, not in S2).
fwd+bwd cases compile the forward function only and call torch.autograd.grad on its outputs, so the
backward is the AOTAutograd/Inductor backward graph, as in training. Speedups: compile rows vs the eager
row, autotune rows vs eager and compile, FA3 rows vs the SDPA row of the same form.

Inputs follow nanochat: activations randn*0.5 in bf16, q/k at their post-QK-norm scale (unit RMS * 1.2),
weights at init_weights scale, cos/sin from GPT._precompute_rotary_embeddings, fp32 master weights for
lm_head (cast inside, as Linear.forward does). min_bytes = algorithmic minimum DRAM traffic and flops =
analytic count (approximate op counts for elementwise/reduction ops); see CASE NOTES in --list output.

Run (GPU job only; do a --trials 1 dry run first, an Inductor cache miss costs minutes):
  python -m cisc662.bench.ops_at_shapes --trials 3 --forms eager,compile --out-dir cisc662/results/db/ops_baseline \
      [--cases REGEX] [--time-budget-s 900] [--job-id ID]
  python -m cisc662.bench.ops_at_shapes --list [--forms eager,compile,autotune]     # CPU-only registry check
"""
import sys
from functools import partial

import torch
import torch.nn.functional as F

from cisc662.bench.kbench import Case, main_cli

# R12 shapes (PLAN.md §3)
B, T, H, D = 16, 2048, 6, 128
N = B * T                     # 32768 tokens per step
C = 768                       # n_embd
V = 32768                     # vocab (no padding at 32768)
FF = 4 * C                    # 3072
BF = 2                        # bytes per bf16 element
SOFTCAP = 15
QK_SCALE = 1.2
ROPE_SEQ = 10 * T             # nanochat precomputes rotary for 10x the sequence length and slices
FORMS = ("eager", "compile", "autotune")


# ----------------------------------------------------------------------------------------------
# Input builders (run on the GPU inside a kbench child)
# ----------------------------------------------------------------------------------------------
def _t(*shape, scale=0.5, grad=False, dtype=None):
    x = (torch.randn(*shape, device="cuda", dtype=torch.float32) * scale).to(dtype or torch.bfloat16)
    return x.requires_grad_(grad)


def _uniform_w(n_out, n_in, gain=1.0, dtype=None):
    """nanochat init_weights: uniform(-s, s), s = sqrt(3)/sqrt(n_embd) (c_fc uses gain 0.4).
    c_proj weights are zero-initialised in nanochat; zeros would lower power and inflate clocks, so the
    benchmark uses the same uniform law with the fan-in of the layer instead."""
    s = gain * 3**0.5 * n_in**-0.5
    return torch.empty(n_out, n_in, device="cuda", dtype=dtype or torch.bfloat16).uniform_(-s, s)


def _cos_sin():
    from nanochat.gpt import GPT
    # self is unused when device is given; this is nanochat's exact table (base 1e5, bf16)
    cos, sin = GPT._precompute_rotary_embeddings(None, ROPE_SEQ, D, device=torch.device("cuda"))
    return cos[:, :T], sin[:, :T]          # (1, 2048, 1, 64), as GPT.forward slices it


def _grad_like(out):
    outs = out if isinstance(out, tuple) else (out,)
    return tuple(None if o.dim() == 0 else torch.randn_like(o) * 0.5 for o in outs)


# ----------------------------------------------------------------------------------------------
# The op functions (python functions so torch.compile has a frame to trace; builtins would run eager).
# No imports inside traced functions; nanochat functions are bound in closures at setup time.
# ----------------------------------------------------------------------------------------------
def gemm_fwd(x, w):
    return F.linear(x, w)                   # (M,K) @ (N,K)^T, as Linear.forward with a bf16 weight


def gemm_dw(dy, x):
    return torch.mm(dy.t(), x)              # autograd's grad_weight of F.linear: dY^T @ X, K = 32768


def relu2(x):
    return F.relu(x).square()               # MLP.forward, gpt.py:139


def softcap_ce(logits, targets):
    """GPT.forward gpt.py:511-520 exactly: slice (no-op at V=32768) -> .float() -> 15*tanh(x/15) -> CE."""
    softcap = SOFTCAP
    logits = logits[..., :V]
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


def lm_head_ce(x, w, targets):
    """Linear.forward of lm_head (fp32 master weight cast to bf16) followed by the softcap+CE tail."""
    logits = F.linear(x, w.to(dtype=x.dtype))
    return softcap_ce(logits, targets)


def _make_qk_chain():
    """CausalSelfAttention.forward gpt.py:100-104 order: RoPE -> QK-norm -> *1.2 (verified in the code)."""
    from nanochat.gpt import apply_rotary_emb, norm

    def qk_rope_norm_scale(q, k, cos, sin):
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * QK_SCALE
        k = k * QK_SCALE
        return q, k
    return qk_rope_norm_scale


def _make_mix_norm():
    """GPT.forward gpt.py:500 (x = resid_lambdas[i]*x + x0_lambdas[i]*x0) then Block.forward's norm(x)."""
    from nanochat.gpt import norm

    def resid_mix_norm(x, x0, r, l):
        x = r * x + l * x0
        return x, norm(x)
    return resid_mix_norm


def _make_attn_fa3():
    from nanochat.flash_attention import flash_attn_func

    def attn_fa3(q, k, v):
        # nanochat's API with R12's window (2048, 0) (pattern L); kbench aborts unless HAS_FA3 and USE_FA3
        y = flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        return y[0] if isinstance(y, tuple) else y      # some FA3 builds also return the LSE
    return attn_fa3


def _attn_sdpa(q, k, v):
    # torch SDPA in (B,H,T,D), what nanochat's fallback calls for a full window (is_causal=True)
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                          is_causal=True).transpose(1, 2)


# ----------------------------------------------------------------------------------------------
# Per-op setups: return (forward fn, inputs, indices of inputs to differentiate or None)
# ----------------------------------------------------------------------------------------------
def _setup_rmsnorm(rows, cols, bwd):
    from nanochat.gpt import norm
    return norm, (_t(rows, cols, grad=bwd),), (0,) if bwd else None


def _setup_qkrope(bwd):
    from nanochat.gpt import norm
    with torch.no_grad():   # q/k as they leave c_q/c_k: unit-RMS rows (the chain re-normalises anyway)
        q = norm(_t(B, T, H, D, scale=1.0))
        k = norm(_t(B, T, H, D, scale=1.0))
    q.requires_grad_(bwd)
    k.requires_grad_(bwd)
    cos, sin = _cos_sin()
    return _make_qk_chain(), (q, k, cos, sin), (0, 1) if bwd else None


def _targets():
    return torch.randint(0, V, (B, T), device="cuda", dtype=torch.int64)


def _setup_ce(bwd):
    return softcap_ce, (_t(B, T, V, grad=bwd), _targets()), (0,) if bwd else None


def _setup_lm_head_ce(bwd):
    x = _t(B, T, C, grad=bwd)
    w = (torch.randn(V, C, device="cuda", dtype=torch.float32) * 0.001).requires_grad_(bwd)   # std 0.001
    return lm_head_ce, (x, w, _targets()), (0, 1) if bwd else None


def _setup_gemm(m, k, n, gain):
    return gemm_fwd, (_t(m, k), _uniform_w(n, k, gain)), None


def _setup_gemm_dw(n_out, k_in):
    return gemm_dw, (_t(N, n_out), _t(N, k_in)), None


def _setup_attn(impl, bwd):
    q, k = _t(B, T, H, D, scale=QK_SCALE, grad=bwd), _t(B, T, H, D, scale=QK_SCALE, grad=bwd)
    v = _t(B, T, H, D, grad=bwd)
    return (_make_attn_fa3() if impl == "fa3" else _attn_sdpa), (q, k, v), (0, 1, 2) if bwd else None


def _setup_relu2(bwd):
    return relu2, (_t(N, FF, grad=bwd),), (0,) if bwd else None


def _setup_mixnorm(bwd):
    r = torch.tensor(1.15, device="cuda", dtype=torch.float32).requires_grad_(bwd)   # layer-0 init values
    l = torch.tensor(0.20, device="cuda", dtype=torch.float32).requires_grad_(bwd)
    return _make_mix_norm(), (_t(B, T, C, grad=bwd), _t(B, T, C, grad=bwd), r, l), (0, 1, 2, 3) if bwd else None


def _timed(op_setup, form):
    """Builds the timed callable for one form: compile wraps the forward function; fwd+bwd adds
    torch.autograd.grad (no .grad accumulation, so every rep does the same work)."""
    f, inputs, wrt = op_setup()
    if form == "compile":
        f = torch.compile(f, dynamic=False)
    elif form == "autotune":
        f = torch.compile(f, dynamic=False, mode="max-autotune-no-cudagraphs")
    if wrt is None:
        return f, inputs
    diff = [inputs[i] for i in wrt]
    probe = f(*inputs)                     # compiles (untimed) and gives output shapes for grad_outputs
    grads = _grad_like(probe)
    del probe

    def fwd_bwd(*args):
        out = f(*args)
        return torch.autograd.grad(out if isinstance(out, tuple) else (out,), diff, grads)
    return fwd_bwd, inputs


# ----------------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------------
def _ops():
    """(name, kernel_id, setup, dtype, shape, min_bytes, flops, gemm?, fa3?, baseline-op, notes) per op."""
    A = N * C * BF                         # one (32768,768) bf16 activation, 50.3 MB
    E = B * T * H * D * BF                 # one (16,2048,6,128) bf16 tensor, 50.3 MB
    CS = 2 * T * (D // 2) * BF             # cos + sin, 0.5 MB
    L = N * V * BF                         # bf16 logits, 2.15 GB
    TG = N * 8                             # int64 targets
    WL = V * C * 4                         # fp32 lm_head master weight, 100.7 MB
    R = N * FF * BF                        # (32768,3072) bf16, 201 MB
    attn_f = 2 * B * H * T * T * D         # causal-effective forward FLOPs, 103.1 GFLOP
    ops = []

    for rows, cols in ((N, C), (B * T * H, D)):
        el = rows * cols
        for bwd in (False, True):
            ops.append((f"rmsnorm_{'fwdbwd' if bwd else 'fwd'}_{rows}x{cols}", "rmsnorm",
                        partial(_setup_rmsnorm, rows, cols, bwd), "bfloat16", {"x": [rows, cols]},
                        (5 if bwd else 2) * el * BF, (10 if bwd else 4) * el, False, False, None,
                        "nanochat.gpt.norm = F.rms_norm(eps=None); fwdbwd min = fwd (read x, write y) + bwd "
                        "(read dy, x; write dx)"))
    nel = B * T * H * D
    for bwd in (False, True):
        ops.append((f"qkrope_{'fwdbwd' if bwd else 'fwd'}_{B}x{T}x{H}x{D}", "qkrope", partial(_setup_qkrope, bwd),
                    "bfloat16", {"q": [B, T, H, D], "k": [B, T, H, D], "cos": [1, T, 1, D // 2],
                                 "sin": [1, T, 1, D // 2]},
                    (10 * E + 2 * CS) if bwd else (4 * E + CS), (48 if bwd else 16) * nel, False, False, None,
                    "RoPE -> rms_norm -> *1.2 on q and k (gpt.py:100-104); fwdbwd min = fwd + bwd "
                    "(read dq', dk', q, k, cos, sin; write dq, dk)"))
    for bwd in (False, True):
        ops.append((f"ce_{'fwdbwd' if bwd else 'fwd'}_{N}x{V}", "ce", partial(_setup_ce, bwd), "bfloat16",
                    {"logits": [B, T, V], "targets": [B, T]}, (2 * L if bwd else L) + TG,
                    (20 if bwd else 12) * N * V, False, False, None,
                    "bf16 logits -> .float() -> 15*tanh(x/15) -> F.cross_entropy(ignore_index=-1), mean; "
                    "fwdbwd min = one fused pass (read logits once, write dlogits once)"))
    for bwd in (False, True):
        ops.append((f"lmhead_ce_{'fwdbwd' if bwd else 'fwd'}_{N}x{C}x{V}", "ce_chunked", partial(_setup_lm_head_ce, bwd),
                    "bfloat16", {"x": [B, T, C], "w_fp32": [V, C], "targets": [B, T]},
                    (2 * A + 2 * WL + TG) if bwd else (A + WL + TG),
                    (3 if bwd else 1) * 2 * N * V * C + (20 if bwd else 12) * N * V, False, False, None,
                    "F.linear(x, W_fp32.to(bf16)) + softcap+CE; min = never materialising logits "
                    "(read x, W; + write dx, dW fp32 for fwdbwd)"))
    # forward GEMMs: M x K x N (x (M,K) @ W(N,K)^T); weight gain as in init_weights
    for k, n, gain, who in ((C, C, 1.0, "c_q/c_k/c_v/c_proj"), (C, 3 * C, 1.0, "fused qkv (3x768)"),
                            (C, FF, 0.4, "mlp.c_fc"), (C, V, None, "lm_head"), (FF, C, 1.0, "mlp.c_proj")):
        setup = partial(_setup_gemm, N, k, n, gain) if gain is not None else partial(_setup_gemm_lmhead, k, n)
        ops.append((f"gemm_{N}x{k}x{n}", "gemm", setup, "bfloat16", {"x": [N, k], "w": [n, k]},
                    (N * k + k * n + N * n) * BF, 2 * N * k * n, True, False, None, f"forward {who}, F.linear bf16"))
    # weight-gradient GEMMs: dW (n_out, k_in) = dY^T (n_out, 32768) @ X (32768, k_in)
    for n_out, k_in, who in ((C, C, "c_q/c_k/c_v/c_proj"), (FF, C, "mlp.c_fc"), (C, FF, "mlp.c_proj"),
                             (V, C, "lm_head")):
        ops.append((f"gemm_dw_{n_out}x{N}x{k_in}", "gemm", partial(_setup_gemm_dw, n_out, k_in), "bfloat16",
                    {"dy": [N, n_out], "x": [N, k_in]}, (N * n_out + N * k_in + n_out * k_in) * BF,
                    2 * N * n_out * k_in, True, False, None, f"dW of {who}: torch.mm(dy.t(), x), K = 32768"))
    for impl in ("sdpa", "fa3"):
        for bwd in (False, True):
            mode = "fwdbwd" if bwd else "fwd"
            ops.append((f"attn_{impl}_{mode}_{B}x{T}x{H}x{D}", "attention", partial(_setup_attn, impl, bwd), "bfloat16",
                        {"q": [B, T, H, D], "k": [B, T, H, D], "v": [B, T, H, D]}, (12 if bwd else 4) * E,
                        (3.5 if bwd else 1) * attn_f, False, impl == "fa3",
                        f"attn_sdpa_{mode}_{B}x{T}x{H}x{D}" if impl == "fa3" else None,
                        ("nanochat.flash_attention.flash_attn_func(causal, window (2048,0)) -> FA3" if impl == "fa3"
                         else "F.scaled_dot_product_attention(is_causal=True) on (B,H,T,D) views")
                        + "; causal-effective FLOPs, bwd = 2.5x fwd (5 matmuls incl. the S recompute, FA2/FA3 "
                        "benchmark convention)"))
    for bwd in (False, True):
        ops.append((f"relu2_{'fwdbwd' if bwd else 'fwd'}_{N}x{FF}", "mlp", partial(_setup_relu2, bwd), "bfloat16",
                    {"x": [N, FF]}, (5 if bwd else 2) * R, (5 if bwd else 2) * N * FF, False, False, None,
                    "F.relu(x).square() (gpt.py:139); fwdbwd min = fwd + bwd (read dy, x; write dx)"))
    for bwd in (False, True):
        ops.append((f"mixnorm_{'fwdbwd' if bwd else 'fwd'}_{N}x{C}", "rmsnorm", partial(_setup_mixnorm, bwd),
                    "bfloat16", {"x": [B, T, C], "x0": [B, T, C], "resid_lambda": [], "x0_lambda": []},
                    (10 if bwd else 4) * A, (21 if bwd else 7) * N * C, False, False, None,
                    "x = r*x + l*x0 (fp32 0-d lambdas) then norm(x); returns (x, norm(x)); fwdbwd min = fwd "
                    "(read x, x0; write x, y) + bwd (read dy, dx, x, x0; write dx, dx0)"))
    return ops


def _setup_gemm_lmhead(k, n):
    w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.001).to(torch.bfloat16)   # lm_head std 0.001
    return gemm_fwd, (_t(N, k), w), None


def build_cases(args):
    forms = [f.strip() for f in getattr(args, "forms", "eager,compile").split(",") if f.strip()]
    bad = [f for f in forms if f not in FORMS]
    if bad:
        raise SystemExit(f"unknown --forms {bad}; choose from {FORMS}")
    cases = []
    for (name, kid, setup, dtype, shape, min_bytes, flops, is_gemm, fa3, base_op, notes) in _ops():
        for form in forms:          # forms of one op are adjacent: same thermal state, ABAB-like
            if form == "autotune" and not is_gemm:
                continue
            baselines = []
            if form != "eager":
                baselines.append(f"eager_{name}")
            if form == "autotune":
                baselines.append(f"compile_{name}")
            if base_op:
                baselines.append(f"{form}_{base_op}")
            cases.append(Case(f"{form}_{name}", partial(_timed, setup, form), dtype=dtype, shape=shape,
                              min_bytes=float(min_bytes), flops=float(flops), group="ops_baseline", kernel_id=kid,
                              form=form, baselines=tuple(baselines), requires_fa3=fa3, notes=notes))
    return cases


def _add_args(p):
    p.add_argument("--forms", default="eager,compile", help="comma list of eager,compile,autotune (autotune = "
                   "max-autotune-no-cudagraphs, GEMM ops only; S4/BAS-4)")


def main(argv=None):
    return main_cli({"ops_baseline": build_cases}, module="cisc662.bench.ops_at_shapes", argv=argv,
                    description="nanochat R12 ops at real shapes (INF-3)", add_args=_add_args,
                    forward_args=lambda a: ["--forms", a.forms])


if __name__ == "__main__":
    sys.exit(main())
