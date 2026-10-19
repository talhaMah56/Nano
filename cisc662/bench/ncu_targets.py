"""
Launch every project Triton kernel at the R12 shape a few times so Nsight Compute can measure warm instances
(task card BAS-6: DRAM bytes, occupancy, stalls of our kernels next to the Inductor/cuBLAS/FA3 baseline).

Each kernel is launched ITERS times (default 3); profile with `-k regex:^c662_` and take the last invocation
per kernel name as the warm one (cisc662/profiling/ncu_summary.py does that). Inputs as the benchmark suites
build them (cisc662/bench/ops_at_shapes.py).

  python -m cisc662.bench.ncu_targets [--iters 3] [--only rmsnorm,qkrope,ce]
"""
import argparse
import sys

import torch


def _rmsnorm(iters):
    from cisc662.kernels.triton import rmsnorm as rn
    from cisc662.bench.ops_at_shapes import B, C, T
    x = (torch.randn(B, T, C, device="cuda") * 0.5).bfloat16()
    dy = torch.randn_like(x)
    for _ in range(iters):
        _, rstd = rn.rmsnorm_fwd(x, "v2")          # c662_rmsnorm_fwd_v2 (the P1 preliminary kernel row)
        rn.rmsnorm_bwd(dy, x, rstd)                # c662_rmsnorm_bwd


def _qkrope(iters):
    from cisc662.bench.ops_at_shapes import _setup_qkrope
    from cisc662.kernels.triton import qkrope as qk
    _, (q, k, cos, sin), _ = _setup_qkrope(True)
    gq, gk = torch.randn_like(q), torch.randn_like(k)
    for _ in range(iters):
        yq, yk = qk.qkrope(q, k, cos, sin, impl="op")
        torch.autograd.grad([yq, yk], [q, k], [gq, gk])


def _ce(iters):
    from cisc662.bench.ops_at_shapes import B, T, V
    from cisc662.kernels.triton import ce
    logits = (torch.randn(B, T, V, device="cuda") * 0.5).bfloat16().requires_grad_(True)
    targets = torch.randint(0, V, (B, T), device="cuda")
    for _ in range(iters):
        for impl, v in (("op", None), ("gif", "gif_tile"), ("gif", "gif_2pass")):
            loss = ce.softcap_ce(logits, targets, "mean", impl=impl, variant=v)
            torch.autograd.grad(loss, logits)


TARGETS = {"rmsnorm": _rmsnorm, "qkrope": _qkrope, "ce": _ce}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--only", default=",".join(TARGETS))
    a = p.parse_args(argv)
    for name in [t for t in a.only.split(",") if t]:
        TARGETS[name](a.iters)
        torch.cuda.synchronize()
        print(f"ncu_targets: {name} x{a.iters} done", flush=True)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
