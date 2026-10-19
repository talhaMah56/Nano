"""
Per-op serial / parallel CPU references for goal (i) (task card BAS-5), on the Grace (Neoverse-V2) host.

Each target op runs in fp32 eager PyTorch on CPU at ONE R12 sequence (N = 2048 tokens; R12 has B = 16
sequences of 2048) -- 3 warm-up + `--reps` timed calls, median. Every op is per-sequence independent at
R12 shapes (attention is per sequence; everything else is per row), so the R12 time is scaled x16 and
labelled SCALED. Also measures the host's achievable ceilings with the same thread count:
  fp32 GEMM GFLOP/s (torch.mm 2048x2048x2048) and copy GB/s (1 GiB fp32 tensor copy).

Run inside a CPU-only Slurm job (slurm/s4_cpu_ref.sbatch): threads come from --threads (sets
torch.set_num_threads and must match OMP_NUM_THREADS); pin the serial run with numactl.
  python -m cisc662.bench.cpu_ref --threads 1 --out cisc662/results/db/cpu/cpu_ref_t1_<job>.json
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

import torch
import torch.nn.functional as F

N_SEQ = 2048           # tokens per sequence (R12 T)
SCALE_TO_R12 = 16      # R12 B
C, H, D, FF, V = 768, 6, 128, 3072, 32768


def _time(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def _ops():
    g = torch.Generator().manual_seed(0)
    r = lambda *s, sc=0.5: torch.randn(*s, generator=g) * sc
    from nanochat.gpt import apply_rotary_emb, norm
    x768 = r(N_SEQ, C)
    q, k = r(1, N_SEQ, H, D), r(1, N_SEQ, H, D)
    t = torch.arange(N_SEQ, dtype=torch.float32)
    inv = 1.0 / (100000 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    f = torch.outer(t, inv)
    cos, sin = f.cos()[None, :, None, :], f.sin()[None, :, None, :]
    logits = r(N_SEQ, V, sc=3.0)
    tg = torch.randint(0, V, (N_SEQ,), generator=g)
    w768, w3072 = r(C, C, sc=0.036), r(FF, C, sc=0.014)
    h3072 = r(N_SEQ, FF)
    qa, ka, va = r(1, H, N_SEQ, D), r(1, H, N_SEQ, D), r(1, H, N_SEQ, D)

    def ce_fwd():
        z = 15 * torch.tanh(logits / 15)
        return F.cross_entropy(z, tg, ignore_index=-1)

    lg_req = logits.clone().requires_grad_(True)

    def ce_fwdbwd():
        z = 15 * torch.tanh(lg_req / 15)
        torch.autograd.grad(F.cross_entropy(z, tg, ignore_index=-1), lg_req)

    def qk_chain():
        a, b = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        return norm(a) * 1.2, norm(b) * 1.2

    E = N_SEQ * C * 4
    # name -> (fn, analytic min bytes at one sequence, flops at one sequence, R12 kernel_id)
    return {
        "rmsnorm_fwd_2048x768": (lambda: norm(x768), 2 * E, 4 * N_SEQ * C, "rmsnorm"),
        "qkrope_fwd_1x2048x6x128": (qk_chain, 4 * N_SEQ * H * D * 4, 16 * N_SEQ * H * D, "qkrope"),
        "ce_fwd_2048x32768": (ce_fwd, N_SEQ * V * 4, 12 * N_SEQ * V, "ce"),
        "ce_fwdbwd_2048x32768": (ce_fwdbwd, 2 * N_SEQ * V * 4, 20 * N_SEQ * V, "ce"),
        "gemm_2048x768x768": (lambda: F.linear(x768, w768), 4 * (N_SEQ * C * 2 + C * C), 2 * N_SEQ * C * C, "gemm"),
        "gemm_2048x768x3072": (lambda: F.linear(x768, w3072), 4 * (N_SEQ * (C + FF) + C * FF), 2 * N_SEQ * C * FF,
                               "gemm"),
        "attn_sdpa_fwd_1x6x2048x128": (lambda: F.scaled_dot_product_attention(qa, ka, va, is_causal=True),
                                       4 * 4 * H * N_SEQ * D, 2 * 2 * H * N_SEQ * N_SEQ * D / 2, "attention"),
        "relu2_fwd_2048x3072": (lambda: F.relu(h3072).square(), 2 * N_SEQ * FF * 4, 2 * N_SEQ * FF, "mlp"),
    }


def ceilings(reps):
    a, b = torch.randn(2048, 2048), torch.randn(2048, 2048)
    tm, _, _ = _time(lambda: torch.mm(a, b), max(3, reps // 2))
    src = torch.empty(2**28, dtype=torch.float32).uniform_()          # 1 GiB
    dst = torch.empty_like(src)
    tc, _, _ = _time(lambda: dst.copy_(src), max(3, reps // 2), warm=1)
    return {"fp32_gemm_GFLOPs": 2 * 2048**3 / tm / 1e9, "copy_GBps": 2 * src.numel() * 4 / tc / 1e9}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--only", default=None, help="comma list of op names")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    if os.environ.get("OMP_NUM_THREADS") not in (None, str(a.threads)):
        print(f"warning: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} != --threads {a.threads}", flush=True)
    res = {"threads": torch.get_num_threads(), "host": platform.node(), "machine": platform.machine(),
           "torch": torch.__version__, "job_id": os.environ.get("SLURM_JOB_ID"), "dtype": "float32",
           "n_tokens_measured": N_SEQ, "scale_to_r12": SCALE_TO_R12, "ops": {}}
    res["ceilings"] = ceilings(a.reps)
    print(f"ceilings ({a.threads} threads): {res['ceilings']}", flush=True)
    ops = _ops()
    for name, (fn, nbytes, flops, kid) in ops.items():
        if a.only and name not in a.only.split(","):
            continue
        reps = a.reps if not (a.threads == 1 and name.startswith(("ce_", "attn"))) else max(3, a.reps // 3)
        med, lo, hi = _time(fn, reps)
        res["ops"][name] = {"kernel_id": kid, "median_s": med, "min_s": lo, "max_s": hi, "reps": reps,
                            "min_bytes": nbytes, "flops": flops, "GBps": nbytes / med / 1e9,
                            "GFLOPs": flops / med / 1e9, "r12_scaled_s": med * SCALE_TO_R12, "label": "SCALED"}
        print(f"{name:32s} {med * 1e3:10.2f} ms  ({nbytes / med / 1e9:6.1f} GB/s, {flops / med / 1e9:7.1f} GFLOP/s)"
              f"  -> R12 x{SCALE_TO_R12}: {med * SCALE_TO_R12:8.3f} s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
