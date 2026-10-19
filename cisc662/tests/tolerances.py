"""
Correctness-gate tolerances of PLAN.md section 7.3 (G1 forward, G2 backward) as code,
including amendment A1 (2026-10-18, after GPU job 3406282 showed stock bf16 ops and FA3 failing
the original absolute bounds; see PLAN.md 7.3 and docs/findings.md#tolerance-a1).

  fp32 outputs        rtol 1e-5, atol 1e-6 (rtol 2e-5 for reductions >= 32768 long)
  bf16/fp16 outputs   max-abs error <= 2x the stock op's max-abs error against the same reference
                      AND cosine >= 0.9999 AND #elements outside rtol 1.6e-2 / atol 1e-2 <= 2x the
                      stock op's count (A1: = the original strict rule whenever stock meets it).
                      Without a stock output the rtol/atol rule is strict and the stock rule fails.
  GEMM                rel-Frobenius error <= 1e-2 AND <= 1.25x cuBLAS's rel-Frobenius error
  attention           max-abs <= 2x FA3's own max-abs error vs the fp32 math-backend SDPA reference
                      (stored in results/db/tolerances/attention_stock_error.json). A1 dropped the
                      absolute 4e-3 term: it is below the bf16 output-rounding floor (7.8e-3 at |y|~4)
                      and FA3 itself measures 9.1e-3.
  dlogits             rel-L2 <= 1e-2 AND <= 2x the stock rel-L2 (A1; the elementwise atol is vacuous at
                      |dlogits| ~ 1e-9..1e-5), plus the G2 cosine and ignore-row rules
  CE loss             rel 1e-5 vs fp64 on the same bf16 logits (per-row form: rtol 1e-5, atol 1e-6)
  G2 gradients        as G1 on dX/dq/dk (dlogits: kind="dlogits"), plus cosine >= 0.9999 and
                      ignore_index rows == 0

check_forward / check_backward return a GateResult (truthy iff passed) carrying the measured errors,
so tests can `assert r, r.summary()` and harnesses can store `r.as_dict()`.
Errors are computed in fp64 in chunks, so (32768, 32768) tensors do not need a second full copy.

Usage:
  from cisc662.tests.tolerances import check_forward, check_backward
  r = check_forward(my_bf16_out, ref_fp64, stock_out=F.rms_norm(x, (768,)))
  assert r, r.summary()
"""
import json
import math
import os
from dataclasses import asdict, dataclass, field

import torch

# --- the table (PLAN.md 7.3) --------------------------------------------------------------------
FP32_RTOL, FP32_ATOL = 1e-5, 1e-6
FP32_LONG_RTOL = 2e-5            # reductions of length >= LONG_REDUCTION (the 32768-wide CE rows)
LONG_REDUCTION = 32768
BF16_RTOL, BF16_ATOL = 1.6e-2, 1e-2
STOCK_FACTOR = 2.0               # bf16 rule and attention rule: <= 2x the stock op's error
GEMM_REL_FRO, GEMM_STOCK_FACTOR = 1e-2, 1.25
ATTN_MAX_ABS, ATTN_STOCK_FACTOR = 4e-3, 2.0    # ATTN_MAX_ABS: original bound, reported only (A1)
DLOGITS_REL_L2, DLOGITS_STOCK_FACTOR = 1e-2, 2.0
CE_REL = 1e-5
GRAD_COSINE = 0.9999
BF16_COSINE = 0.9999             # A1: bf16 forward outputs judged relative to stock also need this
AMENDMENT = "A1"

KINDS = ("elementwise", "gemm", "attention", "ce_loss", "dlogits")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ATTN_STOCK_ERROR_PATH = os.path.join(REPO, "cisc662", "results", "db", "tolerances",
                                     "attention_stock_error.json")

_CHUNK = 1 << 24  # elements per fp64 error chunk (128 MB)


@dataclass
class GateResult:
    passed: bool
    kind: str
    dtype: str
    max_abs: float = float("nan")
    max_rel: float = float("nan")       # max |out-ref| / |ref| over elements with ref != 0
    n_violations: int = 0                # elements outside atol + rtol*|ref| (elementwise kinds)
    numel: int = 0
    rel_fro: float = float("nan")        # GEMM only
    stock_error: float = None            # the stock op's error in the same metric, if used
    stock_ratio: float = None            # our error / stock error
    cosine: float = None                 # backward; bf16 forward judged vs stock (A1)
    stock_violations: int = None         # the stock op's rtol/atol violation count (A1)
    ignore_rows_zero: bool = None        # backward only, when ignore rows were given
    limits: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    def __bool__(self):
        return self.passed

    def as_dict(self):
        return asdict(self)

    def summary(self):
        status = "PASS" if self.passed else "FAIL"
        bits = [f"{status} {self.kind}/{self.dtype}", f"max_abs={self.max_abs:.3e}",
                f"max_rel={self.max_rel:.3e}"]
        if self.kind == "gemm":
            bits.append(f"rel_fro={self.rel_fro:.3e}")
        if self.n_violations:
            bits.append(f"violations={self.n_violations}/{self.numel}")
        if self.stock_error is not None:
            bits.append(f"stock_err={self.stock_error:.3e} ratio={self.stock_ratio:.2f}")
        if self.cosine is not None:
            bits.append(f"cosine={self.cosine:.8f}")
        if self.ignore_rows_zero is not None:
            bits.append(f"ignore_rows_zero={self.ignore_rows_zero}")
        if self.reasons:
            bits.append("reasons: " + "; ".join(self.reasons))
        return " ".join(bits)


# --- error measurement --------------------------------------------------------------------------
def _flat_pairs(out, ref):
    if out.shape != ref.shape:
        raise ValueError(f"shape mismatch: out {tuple(out.shape)} vs ref {tuple(ref.shape)}")
    o, r = out.detach().reshape(-1), ref.detach().reshape(-1)
    if o.device != r.device:
        o = o.to(r.device)
    for s in range(0, o.numel(), _CHUNK):
        yield o[s:s + _CHUNK].double(), r[s:s + _CHUNK].double()


def measure(out, ref, rtol=0.0, atol=0.0):
    """fp64 error statistics of out vs ref: max_abs, max_rel, violations of |d| <= atol+rtol|ref|,
    squared Frobenius norms (for GEMM), dot products (for cosine) and NaN/Inf mismatches.
    Positions where both are NaN (e.g. the mean CE of an all-ignored batch) count as equal."""
    st = dict(max_abs=0.0, max_rel=0.0, viol=0, numel=0, diff2=0.0, ref2=0.0, out2=0.0, dot=0.0,
              nonfinite_mismatch=0)
    for o, r in _flat_pairs(out, ref):
        both_nan = torch.isnan(o) & torch.isnan(r)
        same_inf = torch.isinf(o) & torch.isinf(r) & (o == r)
        finite = torch.isfinite(o) & torch.isfinite(r)
        st["nonfinite_mismatch"] += int((~(finite | both_nan | same_inf)).sum())
        o, r = o[finite], r[finite]
        d = (o - r).abs()
        st["numel"] += int(both_nan.numel())
        if d.numel():
            st["max_abs"] = max(st["max_abs"], float(d.max()))
            nz = r != 0
            if bool(nz.any()):
                st["max_rel"] = max(st["max_rel"], float((d[nz] / r[nz].abs()).max()))
            st["viol"] += int((d > atol + rtol * r.abs()).sum())
            st["diff2"] += float((d * d).sum())
            st["ref2"] += float((r * r).sum())
            st["out2"] += float((o * o).sum())
            st["dot"] += float((o * r).sum())
    return st


def _is_low_precision(dtype):
    return dtype in (torch.bfloat16, torch.float16)


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def load_attention_stock_error(path=ATTN_STOCK_ERROR_PATH):
    """FA3's max-abs error vs the fp32 math SDPA reference, as measured on the GPU node."""
    with open(path) as f:
        return float(json.load(f)["fa3_max_abs_err"])


# --- G1 -----------------------------------------------------------------------------------------
def check_forward(out, ref, stock_out=None, dtype=None, kind="elementwise", reduction_len=None,
                  stock_error=None):
    """G1: judge `out` against `ref` (the fp64 reference on the same inputs).

    dtype          the precision class of `out` (default out.dtype): fp32 or bf16/fp16 rules.
    kind           "elementwise" (norm, rope, relu2, residual mix, ...), "gemm", "attention",
                   "ce_loss" (a scalar loss or the per-row reduction="none" form).
    stock_out      the stock op's output on the same inputs (eager bf16 op, cuBLAS, FA3); its error
                   vs `ref` sets the "<= k x stock" bound. Alternatively pass stock_error directly
                   (attention: defaults to the stored FA3 measurement).
    reduction_len  length of the reduction producing each output; >= 32768 relaxes fp32 rtol to 2e-5.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    dtype = out.dtype if dtype is None else dtype
    res = GateResult(passed=True, kind=kind, dtype=_dtype_name(dtype))

    def fail(msg):
        res.passed = False
        res.reasons.append(msg)

    def stock_metric(metric):
        """Error of stock_out vs ref in the given metric ('max_abs' or 'rel_fro')."""
        if stock_out is not None:
            s = measure(stock_out, ref)
            return math.sqrt(s["diff2"] / s["ref2"]) if metric == "rel_fro" else s["max_abs"]
        return stock_error

    def apply_stock_rule(ours, factor, metric):
        stock = stock_metric(metric)
        if stock is None:
            fail(f"stock error required for the <= {factor}x stock rule")
            return
        res.stock_error = float(stock)
        res.stock_ratio = float(ours / stock) if stock > 0 else (0.0 if ours == 0 else math.inf)
        res.limits["stock_factor"] = factor
        if ours > factor * stock:
            fail(f"{metric} {ours:.3e} > {factor} x stock {stock:.3e}")

    if kind == "gemm":
        st = measure(out, ref)
        res.rel_fro = math.sqrt(st["diff2"] / st["ref2"]) if st["ref2"] > 0 else math.sqrt(st["diff2"])
        res.limits["rel_fro"] = GEMM_REL_FRO
        if res.rel_fro > GEMM_REL_FRO:
            fail(f"rel_fro {res.rel_fro:.3e} > {GEMM_REL_FRO}")
        apply_stock_rule(res.rel_fro, GEMM_STOCK_FACTOR, "rel_fro")
    elif kind == "attention":
        st = measure(out, ref)
        res.limits["original_max_abs_reported_only"] = ATTN_MAX_ABS   # A1
        if stock_out is None and stock_error is None and os.path.exists(ATTN_STOCK_ERROR_PATH):
            stock_error = load_attention_stock_error()
        apply_stock_rule(st["max_abs"], ATTN_STOCK_FACTOR, "max_abs")
    elif kind == "ce_loss":
        if out.numel() == 1:  # the scalar (mean/sum) loss: plain relative error
            rtol, atol = CE_REL, 0.0
        else:                 # per-row losses: rel 1e-5 with the fp32 atol for near-zero rows
            rtol, atol = CE_REL, FP32_ATOL
        st = measure(out, ref, rtol, atol)
        res.limits.update(rtol=rtol, atol=atol)
        if st["viol"]:
            fail(f"{st['viol']} loss values outside rtol={rtol} atol={atol}")
    elif kind == "dlogits":
        st = measure(out, ref)
        rel = math.sqrt(st["diff2"] / st["ref2"]) if st["ref2"] > 0 else math.sqrt(st["diff2"])
        res.rel_fro = rel
        res.limits["rel_l2"] = DLOGITS_REL_L2
        if rel > DLOGITS_REL_L2:
            fail(f"rel_l2 {rel:.3e} > {DLOGITS_REL_L2}")
        apply_stock_rule(rel, DLOGITS_STOCK_FACTOR, "rel_fro")
    else:  # elementwise
        low = _is_low_precision(dtype)
        if low:
            rtol, atol = BF16_RTOL, BF16_ATOL
        else:
            long_red = reduction_len is not None and reduction_len >= LONG_REDUCTION
            rtol, atol = (FP32_LONG_RTOL if long_red else FP32_RTOL), FP32_ATOL
        st = measure(out, ref, rtol, atol)
        res.limits.update(rtol=rtol, atol=atol)
        stock_st = measure(stock_out, ref, rtol, atol) if (low and stock_out is not None) else None
        if stock_st is None:
            if st["viol"]:
                fail(f"{st['viol']} elements outside rtol={rtol} atol={atol}")
        else:  # A1: violations allowed up to 2x the stock op's own count (0 if stock meets the rule)
            res.stock_violations = stock_st["viol"]
            res.limits["violations_vs_stock_factor"] = STOCK_FACTOR
            if st["viol"] > STOCK_FACTOR * stock_st["viol"]:
                fail(f"{st['viol']} elements outside rtol={rtol} atol={atol} > {STOCK_FACTOR} x stock's "
                     f"{stock_st['viol']}")
            res.cosine = (st["dot"] / math.sqrt(st["out2"] * st["ref2"])) if st["out2"] and st["ref2"] else \
                (1.0 if st["out2"] == st["ref2"] == 0 else 0.0)
            res.limits["cosine"] = BF16_COSINE
            if not res.cosine >= BF16_COSINE:
                fail(f"cosine {res.cosine:.8f} < {BF16_COSINE}")
        if low:
            apply_stock_rule(st["max_abs"], STOCK_FACTOR, "max_abs")

    res.max_abs, res.max_rel = st["max_abs"], st["max_rel"]
    res.n_violations, res.numel = st["viol"], st["numel"]
    if kind != "gemm" and st["ref2"] > 0:  # informational for every kind (e.g. dlogits rel-L2, KT-3)
        res.rel_fro = math.sqrt(st["diff2"] / st["ref2"])
    if st["nonfinite_mismatch"]:
        fail(f"{st['nonfinite_mismatch']} NaN/Inf mismatches")
    return res


# --- G2 -----------------------------------------------------------------------------------------
def cosine(a, b):
    """Cosine similarity of two tensors, accumulated in fp64 (1.0 if both are all-zero)."""
    st = measure(a, b)
    if st["out2"] == 0 and st["ref2"] == 0:
        return 1.0
    if st["out2"] == 0 or st["ref2"] == 0:
        return 0.0
    return st["dot"] / math.sqrt(st["out2"] * st["ref2"])


def check_backward(grad, ref_grad, stock_grad=None, dtype=None, kind="elementwise",
                   reduction_len=None, stock_error=None, ignore_rows=None):
    """G2: the G1 rules applied to a gradient, plus cosine(grad, ref_grad) >= 0.9999 and, if
    `ignore_rows` (bool mask over the leading dims, e.g. targets == -1) is given, those rows of
    `grad` exactly 0. For dlogits pass kind="dlogits" (rel-L2 rule, A1)."""
    if kind == "ce_loss":
        raise ValueError("dlogits are an elementwise gradient; use kind='elementwise'")
    res = check_forward(grad, ref_grad, stock_out=stock_grad, dtype=dtype, kind=kind,
                        reduction_len=reduction_len, stock_error=stock_error)
    res.cosine = cosine(grad, ref_grad)
    res.limits["cosine"] = GRAD_COSINE
    if not res.cosine >= GRAD_COSINE:  # also catches NaN
        res.passed = False
        res.reasons.append(f"cosine {res.cosine:.8f} < {GRAD_COSINE}")
    if ignore_rows is not None:
        g = grad.detach().reshape(ignore_rows.numel(), -1)
        mask = ignore_rows.reshape(-1).to(g.device)
        res.ignore_rows_zero = bool((g[mask] == 0).all()) if bool(mask.any()) else True
        if not res.ignore_rows_zero:
            res.passed = False
            res.reasons.append("ignore_index rows have non-zero gradient")
    return res
