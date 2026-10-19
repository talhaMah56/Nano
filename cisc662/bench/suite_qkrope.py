"""
KT-2 microbenchmark: fused QK-norm + RoPE + 1.2 (cisc662/kernels/triton/qkrope.py) vs nanochat's chain
(task card KT-2 step 4; protocol PLAN.md 7.1 via cisc662/bench/kbench.py).

Cases at the R12 shape q, k (16, 2048, 6, 128) bf16, cos/sin the bf16 (1, 2048, 1, 64) slice of nanochat's
(1, 20480, 1, 64) table, inputs exactly as cisc662/bench/ops_at_shapes.py builds them (_setup_qkrope: unit-RMS
q/k rows, GPT._precompute_rotary_embeddings), forward and fwd+bwd (torch.autograd.grad with fixed upstream
grads, as ops_at_shapes._timed):
  eager_*            the eager chain apply_rotary_emb -> norm -> *1.2 (gpt.py:100-103)
  compile_*          torch.compile(chain, dynamic=False) (Inductor: 1 fused fwd kernel, 3 bwd kernels)
  ours_*             torch.ops.cisc662.qkrope (triton_op, variant v1), baselines eager_* and compile_*
  ours_fn_<v>_*      QKRopeFunction (direct launches) with tile variant v in qkrope.VARIANTS
min_bytes / flops as ops_at_shapes (fwd: read q, k, cos, sin + write yq, yk; fwd+bwd adds read dy, q, k,
cos, sin + write dq, dk); our kernels also write/read rstd (2 x 196608 x 4 B = 1.6 MB, 0.8% of the forward
minimum), not counted.

Run (GPU job only; --list works anywhere):
  python -m cisc662.bench.suite_qkrope --trials 3 --out-dir cisc662/results/db/qkrope --job-id ID --time-budget-s 600
  python -m cisc662.bench.suite_qkrope --list
Then record gates (only after test_qkrope.py passed in the same job):
  python -m cisc662.tests.gates cisc662/results/db/qkrope/<ours case>.json G1 G2
"""
import sys
from functools import partial

from cisc662.bench.kbench import Case, main_cli
from cisc662.bench.ops_at_shapes import B, BF, D, H, T, _setup_qkrope, _timed

GROUP = KERNEL_ID = "qkrope"
OURS_VARIANTS = ("v1", "ilp2", "ilp4")      # must be keys of qkrope.VARIANTS


def _setup_ours(bwd, impl, variant=None):
    """Same inputs as ops_at_shapes._setup_qkrope; forward fn = our kernel."""
    from cisc662.kernels.triton import qkrope as qk
    _, inputs, wrt = _setup_qkrope(bwd)
    if impl == "op":
        def f(q, k, cos, sin):
            return qk.qkrope(q, k, cos, sin, impl="op")
    else:
        def f(q, k, cos, sin):
            return qk.qkrope(q, k, cos, sin, impl="function", variant=variant)
    return f, inputs, wrt


def build_cases(args=None):
    E = B * T * H * D * BF                 # one (16,2048,6,128) bf16 tensor, 50.3 MB
    CS = 2 * T * (D // 2) * BF             # cos + sin, 0.5 MB
    nel = B * T * H * D
    shape = {"q": [B, T, H, D], "k": [B, T, H, D], "cos": [1, T, 1, D // 2], "sin": [1, T, 1, D // 2]}
    cases = []
    for bwd in (False, True):
        op = f"qkrope_{'fwdbwd' if bwd else 'fwd'}_{B}x{T}x{H}x{D}"
        common = dict(dtype="bfloat16", shape=shape, min_bytes=float((10 * E + 2 * CS) if bwd else (4 * E + CS)),
                      flops=float((48 if bwd else 16) * nel), group=GROUP, kernel_id=KERNEL_ID)
        eager, comp = f"eager_{op}", f"compile_{op}"
        chain = "RoPE -> rms_norm -> *1.2 on q and k (gpt.py:100-103)"
        cases.append(Case(eager, partial(_timed, partial(_setup_qkrope, bwd), "eager"), form="eager",
                          notes=f"eager chain: {chain}", **common))
        cases.append(Case(comp, partial(_timed, partial(_setup_qkrope, bwd), "compile"), form="compile",
                          baselines=(eager,), notes=f"torch.compile(dynamic=False) of the chain: {chain}", **common))
        cases.append(Case(f"ours_{op}", partial(_timed, partial(_setup_ours, bwd, "op"), "eager"),
                          form="triton_op_v1", baselines=(eager, comp),
                          notes="torch.ops.cisc662.qkrope (triton_op, variant v1): c662_qkrope_fwd"
                                + (" + c662_qkrope_bwd" if bwd else "") + "; q and k in one launch", **common))
        for v in OURS_VARIANTS:
            cases.append(Case(f"ours_fn_{v}_{op}", partial(_timed, partial(_setup_ours, bwd, "function", v), "eager"),
                              form=f"triton_{v}", baselines=(eager, comp, f"ours_{op}"),
                              notes=f"QKRopeFunction (direct launches), tile variant {v}", **common))
    return cases


def _summary(suite, records):
    """One line per ours case: median, GB/s on min bytes, speedup vs eager / compile with CI."""
    for name, r in records.items():
        if not name.startswith("ours") or r.get("median_us") is None:
            continue
        bits = [f"{name:<44s} {r['median_us']:9.2f} us {r['gbps'] or 0:7.0f} GB/s"]
        for base, sp in r.get("speedups", {}).items():
            bits.append(f"vs {base.split('_')[0]}{'(ours)' if base.startswith('ours') else ''} "
                        f"{sp['ratio']:.2f}x [{sp['ci_low']:.2f}, {sp['ci_high']:.2f}]")
        print("SUMMARY " + "  ".join(bits), flush=True)


def main(argv=None):
    return main_cli({"qkrope": build_cases}, module="cisc662.bench.suite_qkrope", argv=argv,
                    description="KT-2 fused qkrope vs nanochat's chain at R12 shape", on_done=_summary)


if __name__ == "__main__":
    sys.exit(main())
