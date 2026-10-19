"""
KT-3 microbenchmark: fused softcap + cross-entropy (cisc662/kernels/triton/ce.py) vs nanochat's loss tail
(task card KT-3 step 5; protocol PLAN.md 7.1 via cisc662/bench/kbench.py).

Cases at the R12 shape: bf16 logits (16, 2048, 32768) = 2.15 GB, int64 targets (16, 2048), reduction "mean",
inputs exactly as cisc662/bench/ops_at_shapes.py builds them (_setup_ce), forward and fwd+bwd
(torch.autograd.grad; the scalar loss gets grad_out = 1, R12's grad_accum = 1 training path):
  eager_*            logits.float() -> 15*tanh(x/15) -> F.cross_entropy (gpt.py:513-520)
  compile_*          torch.compile of the same (Inductor: online-softmax fwd kernel + recompute bwd kernel)
  ours_*             torch.ops.cisc662.ce_fwd/ce_bwd (form ii, tile "rec"), baselines eager_* and compile_*
  ours_<variant>_*   rec_2k / rec_8k (form ii tiles), gif_tile / gif_2pass (form i, gradient in forward)
  ablation_approx_*  rec with tanh.approx.f32: NOT a result (fails G1 by design); instruction-count ablation
min_bytes as ops_at_shapes: fwd = read logits + targets; fwd+bwd = one fused pass (read logits once, write
dlogits once). Form (ii) moves 1.5x that minimum by design (re-reads the logits in the backward).

Also --peak-mem: torch.cuda.max_memory_allocated above the inputs for one fwd+bwd of each form (card KT-3
step 5), written to <out-dir>/peak_memory_<job>.json.

Run (GPU job only; --list works anywhere):
  python -m cisc662.bench.suite_ce --trials 3 --out-dir cisc662/results/db/ce --job-id ID --time-budget-s 600
  python -m cisc662.bench.suite_ce --peak-mem --out-dir cisc662/results/db/ce --job-id ID
"""
import json
import os
import sys
from functools import partial

from cisc662.bench.kbench import Case, main_cli
from cisc662.bench.ops_at_shapes import B, BF, T, V, _setup_ce, _timed

GROUP = KERNEL_ID = "ce"
N = B * T
GATED_VARIANTS = ("rec_2k", "rec_8k", "gif_tile", "gif_2pass")   # gated in test_ce.py; timed as results


def _ours_fn(impl, variant):
    from cisc662.kernels.triton import ce

    def f(logits, targets):
        return ce.softcap_ce(logits, targets, "mean", impl=impl, variant=variant)
    return f


def _setup_ours(bwd, impl, variant=None):
    _, inputs, wrt = _setup_ce(bwd)
    return _ours_fn(impl, variant), inputs, wrt


def build_cases(args=None):
    L = N * V * BF                      # bf16 logits, 2.15 GB
    TG = N * 8                          # int64 targets
    shape = {"logits": [B, T, V], "targets": [B, T]}
    cases = []
    for bwd in (False, True):
        op = f"ce_{'fwdbwd' if bwd else 'fwd'}_{N}x{V}"
        common = dict(dtype="bfloat16", shape=shape, min_bytes=float((2 * L if bwd else L) + TG),
                      flops=float((20 if bwd else 12) * N * V), group=GROUP, kernel_id=KERNEL_ID)
        eager, comp = f"eager_{op}", f"compile_{op}"
        tail = "logits.float() -> 15*tanh(x/15) -> F.cross_entropy(ignore_index=-1), mean"
        cases.append(Case(eager, partial(_timed, partial(_setup_ce, bwd), "eager"), form="eager",
                          notes=f"eager: {tail}", **common))
        cases.append(Case(comp, partial(_timed, partial(_setup_ce, bwd), "compile"), form="compile",
                          baselines=(eager,), notes=f"torch.compile(dynamic=False): {tail}", **common))
        cases.append(Case(f"ours_{op}", partial(_timed, partial(_setup_ours, bwd, "op"), "eager"),
                          form="triton_op_rec", baselines=(eager, comp),
                          notes="torch.ops.cisc662.ce_fwd" + (" + ce_bwd" if bwd else "")
                                + " (form ii: loss + lse2 forward, recompute backward), tile rec", **common))
        for v in GATED_VARIANTS:
            impl = "gif" if v.startswith("gif") else "op"
            cases.append(Case(f"ours_{v}_{op}", partial(_timed, partial(_setup_ours, bwd, impl, v), "eager"),
                              form=f"triton_{v}", baselines=(eager, comp, f"ours_{op}"),
                              notes=("form (i) CEGifFunction: loss and dlogits in one kernel"
                                     if impl == "gif" else "form (ii) via direct launches") + f", variant {v}",
                              **common))
        cases.append(Case(f"ablation_approx_{op}", partial(_timed, partial(_setup_ours, bwd, "op", "rec_approx"),
                                                           "eager"),
                          form="ablation_tanh_approx", baselines=(comp, f"ours_{op}"),
                          notes="ABLATION ONLY (fails G1 by design): rec with tanh.approx.f32 (MUFU.TANH)", **common))
    return cases


def _summary(suite, records):
    for name, r in records.items():
        if not (name.startswith("ours") or name.startswith("ablation")) or r.get("median_us") is None:
            continue
        bits = [f"{name:<44s} {r['median_us']:9.1f} us {r['gbps'] or 0:7.0f} GB/s(min)"]
        for base, sp in r.get("speedups", {}).items():
            bits.append(f"vs {base.split('_')[0]} {sp['ratio']:.2f}x [{sp['ci_low']:.2f}, {sp['ci_high']:.2f}]")
        print("SUMMARY " + "  ".join(bits), flush=True)


def peak_memory(out_dir, job_id):
    """Peak allocated bytes above the live inputs for one fwd+bwd of each form at R12 (KT-3 step 5)."""
    import torch
    res = {"job_id": job_id, "shape": [B, T, V], "note": "max_memory_allocated - memory_allocated before the "
           "call, inputs (2.15 GB logits) already resident; fwd+bwd via torch.autograd.grad"}
    forms = [("eager", "eager", None), ("compile", "compile", None), ("ours_rec", "op", None)] + \
            [(f"ours_{v}", "gif" if v.startswith("gif") else "op", v) for v in GATED_VARIANTS]
    for name, kind, var in forms:
        if kind in ("eager", "compile"):
            fn, inputs = _timed(partial(_setup_ce, True), kind)
        else:
            fn, inputs = _timed(partial(_setup_ours, True, kind, var), "eager")
        fn(*inputs)                      # warm (compile / JIT)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = fn(*inputs)
        torch.cuda.synchronize()
        res[name] = {"peak_above_inputs_GB": (torch.cuda.max_memory_allocated() - base) / 1e9}
        print(f"PEAKMEM {name:16s} {res[name]['peak_above_inputs_GB']:.2f} GB", flush=True)
        del out, fn, inputs
        torch.cuda.empty_cache()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"peak_memory_{job_id}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--peak-mem" in argv:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--peak-mem", action="store_true")
        p.add_argument("--out-dir", required=True)
        p.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "local"))
        a = p.parse_args(argv)
        peak_memory(a.out_dir, a.job_id)
        return 0
    return main_cli({"ce": build_cases}, module="cisc662.bench.suite_ce", argv=argv,
                    description="KT-3 fused softcap+CE vs nanochat's loss tail at R12 shape", on_done=_summary)


if __name__ == "__main__":
    sys.exit(main())
