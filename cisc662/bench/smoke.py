"""
Session-1 smoke test and machine characterization for one GPU.

Checks the toolchain end to end and measures the empirical roofline ceilings that
all later kernel results are plotted against:
  - device properties (SMs, memory, clocks)
  - FA3 availability through nanochat's loader (HF kernels hub) vs SDPA fallback
  - achieved HBM bandwidth (copy and read-only reduction)
  - achieved dense GEMM throughput for bf16 / fp16 / tf32 / fp32
  - a Triton kernel compiled and run, with its PTX/SASS stats (regs, spills)

Run:  python -m cisc662.bench.smoke --out cisc662/results/smoke.json
"""
import argparse
import json
import os
import platform
import subprocess

import torch
import triton
import triton.language as tl


def cuda_time_ms(fn, iters=20, warmup=5):
    """Median time of fn() in ms using CUDA events (one event pair per iteration)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def measure_bandwidth(nbytes=4 * 2**30):
    n = nbytes // 2
    x = torch.empty(n, dtype=torch.bfloat16, device="cuda").uniform_()
    y = torch.empty_like(x)
    copy_ms = cuda_time_ms(lambda: y.copy_(x))
    sum_ms = cuda_time_ms(lambda: x.sum(dtype=torch.float32))
    out = {
        "buffer_GiB": nbytes / 2**30,
        "copy_GBps": 2 * nbytes / (copy_ms * 1e-3) / 1e9,  # read + write
        "read_sum_GBps": nbytes / (sum_ms * 1e-3) / 1e9,   # read only
    }
    del x, y
    torch.cuda.empty_cache()
    return out


def measure_gemm(size=8192):
    out = {}
    cases = [
        ("bf16", torch.bfloat16, False),
        ("fp16", torch.float16, False),
        ("tf32", torch.float32, True),
        ("fp32", torch.float32, False),
    ]
    for name, dtype, allow_tf32 in cases:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        a = torch.randn(size, size, dtype=dtype, device="cuda")
        b = torch.randn(size, size, dtype=dtype, device="cuda")
        ms = cuda_time_ms(lambda: torch.mm(a, b), iters=10, warmup=3)
        out[f"{name}_TFLOPs"] = 2 * size**3 / (ms * 1e-3) / 1e12
        del a, b
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    return out


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


def triton_check(dump_dir):
    n = 1 << 24
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    k = _add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()
    ok = torch.allclose(out, x + y)
    os.makedirs(dump_dir, exist_ok=True)
    for key in ("ttgir", "ptx", "sass"):
        with open(os.path.join(dump_dir, f"add_kernel.{key}"), "w") as f:
            f.write(k.asm[key])
    sass = k.asm["sass"]
    return {
        "correct": bool(ok),
        "n_regs": k.n_regs,
        "n_spills": k.n_spills,
        "ptx_ld_global_v4": k.asm["ptx"].count("ld.global.v4"),
        "sass_LDG_E_128": sass.count("LDG.E.128"),
        "dump_dir": dump_dir,
    }


def fa3_check():
    from nanochat.flash_attention import HAS_FA3, USE_FA3
    info = {"HAS_FA3": HAS_FA3, "USE_FA3": USE_FA3}
    if HAS_FA3:
        # numerical agreement of FA3 vs SDPA on one causal case at the reduced-config head_dim
        from nanochat import flash_attention as fa
        B, T, H, D = 2, 1024, 6, 128
        q, k, v = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        y_fa3 = fa._fa3.flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
        if isinstance(y_fa3, tuple):
            y_fa3 = y_fa3[0]
        y_ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
        info["fa3_vs_sdpa_max_abs_err"] = (y_fa3.float() - y_ref.float()).abs().max().item()
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dump-dir", default=os.path.join(os.environ.get("CISC662_WORK", "."), "scratch", "triton_dump"))
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    report = {
        "host": platform.node(),
        "machine": platform.machine(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "device": {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "total_memory_GiB": props.total_memory / 2**30,
            "l2_cache_MiB": getattr(props, "L2_cache_size", 0) / 2**20,
        },
    }
    try:
        q = "clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.draw,power.limit,temperature.gpu"
        report["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        pass

    print(json.dumps(report, indent=2))
    report["fa3"] = fa3_check()
    print("FA3:", report["fa3"])
    report["triton_add"] = triton_check(args.dump_dir)
    print("Triton:", report["triton_add"])
    report["bandwidth"] = measure_bandwidth()
    print("Bandwidth:", report["bandwidth"])
    report["gemm_8192"] = measure_gemm()
    print("GEMM:", report["gemm_8192"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
