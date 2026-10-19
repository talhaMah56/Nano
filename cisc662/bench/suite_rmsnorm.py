"""
KT-1 microbenchmark suite: Triton RMSNorm (v1, v2) and fused mix+norm (v3) vs eager and torch.compile at the
R12 shapes (PLAN.md 7.1 protocol via kbench: CUDA events, 256 MB L2 flush, queue pad, 25 warm-up + 100 timed
reps per trial, process-level trials, bootstrap speedup CIs).

Cases (kernel_id "rmsnorm", group "rmsnorm"), each for fwd and fwd+bwd:
  rmsnorm 32768x768 (residual-stream norm) and 196608x128 (QK-norm rows of (16,2048,6,128)):
      eager    nanochat.gpt.norm = F.rms_norm(x, (D,)) (eps=None -> fp32 eps; one aten::_fused_rms_norm kernel)
      compile  torch.compile(norm, dynamic=False)
      triton_v1, triton_v2   cisc662.kernels.triton.rmsnorm.rmsnorm(x, "v1"|"v2") via torch.ops.cisc662.rmsnorm
  mixnorm 16x2048x768 (x = r*x + l*x0 with fp32 0-d lambdas, then norm(x); returns (x, norm(x))):
      eager    the 3-op eager chain (mul, mul+add, rms_norm), compile of the same function
      triton_v3  torch.ops.cisc662.mixnorm (one kernel forward; one kernel + a tiny fp64 sum backward)
Inputs, eager/compile forms and fwd+bwd timing (torch.autograd.grad, no .grad accumulation) are exactly
cisc662/bench/ops_at_shapes.py's, so the eager/compile rows are comparable with results/db/ops_baseline/.
Every "ours" case names BOTH eager_<op> and compile_<op> as baselines (speedup CI vs each); compile rows name
eager_<op>. min_bytes/flops = ops_at_shapes' algorithmic minimums (rmsnorm fwd: read x + write y; the fp32
rstd our forward also stores is 0.13% (768) / 0.78% (128) on top and not counted, so all forms of one op
share one denominator).

Run (GPU job only):
  python -m cisc662.bench.suite_rmsnorm --trials 3 --out-dir cisc662/results/db/rmsnorm --job-id ID \
      --time-budget-s 900 [--record-gates G1,G2] [--cases REGEX]
  python -m cisc662.bench.suite_rmsnorm --list          # CPU-only registry check
--record-gates: after the trials, add those gates to every ours (triton_v1/v2/v3) row written by THIS job
(cisc662/tests/gates.py). Pass it only when `pytest cisc662/tests/test_rmsnorm.py` passed in the same job.
"""
import argparse
import os
import sys
from functools import partial

import torch

from cisc662.bench import ops_at_shapes as OPS
from cisc662.bench.kbench import DB_ROOT, Case, main_cli

B, T, H, D, N, C = OPS.B, OPS.T, OPS.H, OPS.D, OPS.N, OPS.C
BF = OPS.BF
OURS_FORMS = ("triton_v1", "triton_v2", "triton_v3")
GROUP = "rmsnorm"


# ----------------------------------------------------------------------------------------------
# Our setups, same (fn, inputs, wrt) contract as ops_at_shapes._setup_*; OPS._timed adds autograd.grad
# ----------------------------------------------------------------------------------------------
def _setup_ours_rmsnorm(rows, cols, bwd, variant):
    from cisc662.kernels.triton import rmsnorm as K

    def ours_rmsnorm(x):
        return K.rmsnorm(x, variant)
    return ours_rmsnorm, (OPS._t(rows, cols, grad=bwd),), (0,) if bwd else None


def _setup_ours_mixnorm(bwd):
    from cisc662.kernels.triton import rmsnorm as K
    _, inputs, wrt = OPS._setup_mixnorm(bwd)     # identical inputs: x, x0 (B,T,C) bf16, r=1.15, l=0.20 fp32 0-d

    def ours_mixnorm(x, x0, r, l):
        return K.mixnorm(x, x0, r, l)
    return ours_mixnorm, inputs, wrt


def _ops():
    """(op name, eager/compile setup, {ours form: setup}, dtype, shape, min_bytes, flops, notes)."""
    out = []
    for rows, cols in ((N, C), (B * T * H, D)):
        el = rows * cols
        for bwd in (False, True):
            mode = "fwdbwd" if bwd else "fwd"
            out.append((f"rmsnorm_{mode}_{rows}x{cols}", partial(OPS._setup_rmsnorm, rows, cols, bwd),
                        {f"triton_{v}": partial(_setup_ours_rmsnorm, rows, cols, bwd, v) for v in ("v1", "v2")},
                        "bfloat16", {"x": [rows, cols]}, (5 if bwd else 2) * el * BF, (10 if bwd else 4) * el,
                        "nanochat.gpt.norm = F.rms_norm(eps=None, fp32 eps); ours v1 = one row/program (BLOCK_D="
                        "next_pow2(D), masked), v2 = BLOCK_M rows/program with 256/128-wide register chunks; ours "
                        "fwd also writes fp32 rstd (not in min_bytes); fwdbwd min = fwd + bwd (read dy, x; write dx)"))
    A = N * C * BF
    for bwd in (False, True):
        mode = "fwdbwd" if bwd else "fwd"
        out.append((f"mixnorm_{mode}_{N}x{C}", partial(OPS._setup_mixnorm, bwd),
                    {"triton_v3": partial(_setup_ours_mixnorm, bwd)}, "bfloat16",
                    {"x": [B, T, C], "x0": [B, T, C], "resid_lambda": [], "x0_lambda": []},
                    (10 if bwd else 4) * A, (21 if bwd else 7) * N * C,
                    "x = r*x + l*x0 (fp32 0-d lambdas) then norm(x); returns (x, norm(x)); ours v3 = "
                    "c662_rmsnorm_fwd_mixnorm (+ c662_rmsnorm_bwd_mixnorm and an N-element fp64 sum for dr, dl); "
                    "fwdbwd min = fwd (read x, x0; write x, y) + bwd (read dy, dx, x, x0; write dx, dx0)"))
    return out


def build_cases(args=None):
    forms = None
    if args is not None and getattr(args, "forms", None):
        forms = {f.strip() for f in args.forms.split(",") if f.strip()}
    cases = []
    for name, base_setup, ours, dtype, shape, min_bytes, flops, notes in _ops():
        common = dict(dtype=dtype, shape=shape, min_bytes=float(min_bytes), flops=float(flops), group=GROUP,
                      kernel_id="rmsnorm", notes=notes)
        # forms of one op are adjacent (same thermal state, ABAB-like across trials)
        for form in ("eager", "compile"):
            if forms is None or form in forms:
                cases.append(Case(f"{form}_{name}", partial(OPS._timed, base_setup, form), form=form,
                                  baselines=() if form == "eager" else (f"eager_{name}",), **common))
        for form, setup in ours.items():
            if forms is None or form in forms:
                cases.append(Case(f"{form}_{name}", partial(OPS._timed, setup, form), form=form,
                                  baselines=(f"eager_{name}", f"compile_{name}"), **common))
    return cases


def _add_args(p):
    p.add_argument("--forms", default=None, help="comma subset of eager,compile,triton_v1,triton_v2,triton_v3 (default all)")
    p.add_argument("--record-gates", default=None,
                   help="comma list of gates (e.g. G1,G2) to record on this job's ours rows after the trials")


def _forward_args(a):
    return ["--forms", a.forms] if a.forms else []


def _record_gates(suite, records):
    """on_done hook: kbench does not hand the parsed args over, so re-read the two flags we need."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--record-gates", default=None)
    p.add_argument("--out-dir", default=None)
    a, _ = p.parse_known_args()
    if not a.record_gates:
        return
    from cisc662.tests.gates import record_gate
    gates = [g.strip() for g in a.record_gates.split(",") if g.strip()]
    out_dir = os.path.abspath(a.out_dir or os.path.join(DB_ROOT, suite))
    for name, rec in records.items():
        if rec.get("form") not in OURS_FORMS or rec.get("median_us") is None:
            continue
        for path in (os.path.join(out_dir, f"{name}.json"),
                     os.path.join(out_dir, "by_job", str(rec.get("job_id")), f"{name}.json")):
            if os.path.exists(path):
                for g in gates:
                    record_gate(path, g)
        print(f"[suite_rmsnorm] {name}: gates_passed += {gates}", flush=True)


def main(argv=None):
    return main_cli({"rmsnorm": build_cases}, module="cisc662.bench.suite_rmsnorm", argv=argv,
                    description="KT-1 Triton RMSNorm/mixnorm vs eager and torch.compile at the R12 shapes",
                    add_args=_add_args, forward_args=_forward_args, on_done=_record_gates)


if __name__ == "__main__":
    sys.exit(main())
