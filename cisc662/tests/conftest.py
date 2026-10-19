"""
Shared pytest setup for the correctness gates (card INF-2, PLAN.md 7.3).

  markers   real_shape  R12-sized tensors (GB-scale, e.g. the fp64 (32768, 32768) CE reference);
                        always paired with gpu, so they only run in the GPU job
            gpu         needs CUDA; the fixtures skip cleanly when CUDA is absent
  fixtures  device                 "cpu" and (gpu-marked) "cuda"
            rmsnorm_case, rope_case, ce_case, relu2_case, resid_case
                                   R12 real shapes (real_shape+gpu) and edge shapes (N=1000 rows,
                                   T=1 decode, rows all ignore_index, |logits| > 50, positions up
                                   to 2047), generated from fixed seeds on the case's device

Login node:  python -m pytest cisc662/tests -q -m "not real_shape and not gpu"   (tiny shapes)
GPU job:     python -m pytest cisc662/tests -q
"""
import os

import pytest
import torch

from cisc662.tests import ref_ops

SEED = 1234
# R12 (PLAN.md section 3): d12, 6 x 128 heads, 16 x 2048 tokens, vocab 32768, MLP 4x.
R12 = dict(B=16, T=2048, H=6, D=128, C=768, V=32768, N=32768, F=3072)

GPU = [pytest.mark.gpu]
REAL = [pytest.mark.real_shape, pytest.mark.gpu]


def pytest_configure(config):
    config.addinivalue_line("markers", "real_shape: R12-sized tensors (GPU job only, paired with gpu)")
    config.addinivalue_line("markers", "gpu: needs CUDA; skips cleanly when CUDA is unavailable")


def require_device(device):
    """Skip the calling test/fixture if `device` is CUDA and CUDA is unavailable."""
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(device)


def gen(seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def randn(shape, seed, device, scale=1.0, dtype=torch.bfloat16):
    """randn * scale generated in fp32 from a fixed seed, then cast (realistic activation scale)."""
    return (torch.randn(shape, generator=gen(seed, device), device=device) * scale).to(dtype)


def job_id():
    return os.environ.get("SLURM_JOB_ID")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(SEED)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=GPU)])
def device(request):
    return require_device(request.param)


@pytest.fixture
def cuda():
    return require_device("cuda")


# --- RMSNorm: (rows, D) bf16 activations ----------------------------------------------------------
RMSNORM_CASES = [
    pytest.param(("tiny", (64, 768), "cpu"), id="tiny"),
    pytest.param(("rows1000", (1000, 768), "cpu"), id="rows1000"),            # N not a block multiple
    pytest.param(("rows1000_d128", (1000, 128), "cpu"), id="rows1000_d128"),
    pytest.param(("rows1000_cuda", (1000, 768), "cuda"), id="rows1000_cuda", marks=GPU),
    pytest.param(("r12_resid", (R12["N"], R12["C"]), "cuda"), id="r12_resid", marks=REAL),
    pytest.param(("r12_qk", (R12["N"] * R12["H"], R12["D"]), "cuda"), id="r12_qk", marks=REAL),
]


@pytest.fixture(params=RMSNORM_CASES)
def rmsnorm_case(request):
    name, shape, dev = request.param
    dev = require_device(dev)
    return dict(name=name, x=randn(shape, SEED, dev, 0.5), device=dev)


# --- RoPE / QK chain: q, k (B, T, H, D) with cos/sin (1, T, 1, D/2) at positions pos0.. ------------
ROPE_CASES = [
    pytest.param(("tiny", 2, 16, 6, 128, 0, "cpu"), id="tiny"),
    pytest.param(("tail_pos", 1, 8, 6, 128, 2040, "cpu"), id="tail_pos"),     # positions 2040..2047
    pytest.param(("decode_t1", 4, 1, 6, 128, 2047, "cpu"), id="decode_t1"),   # T=1 decode shape
    pytest.param(("tiny_cuda", 2, 16, 6, 128, 2032, "cuda"), id="tiny_cuda", marks=GPU),
    pytest.param(("r12", R12["B"], R12["T"], R12["H"], R12["D"], 0, "cuda"), id="r12", marks=REAL),
]


@pytest.fixture(params=ROPE_CASES)
def rope_case(request):
    name, B, T, H, D, pos0, dev = request.param
    dev = require_device(dev)
    cos, sin = ref_ops.rotary_tables(R12["T"], D, device=dev)  # positions 0..2047
    return dict(name=name, q=randn((B, T, H, D), SEED, dev, 0.5), k=randn((B, T, H, D), SEED + 1, dev, 0.5),
                cos=cos[:, pos0:pos0 + T], sin=sin[:, pos0:pos0 + T], pos0=pos0, device=dev)


# --- softcap + CE: bf16 logits (N, V), int64 targets with ignore_index rows -------------------------
# (name, N, V, logit scale, fraction of ignored rows, saturate, device)
CE_CASES = [
    pytest.param(("tiny", 64, 512, 3.0, 0.1, False, "cpu"), id="tiny"),
    pytest.param(("saturated", 32, 512, 3.0, 0.1, True, "cpu"), id="saturated"),     # |logits| > 50
    pytest.param(("all_ignored", 8, 512, 3.0, 1.0, False, "cpu"), id="all_ignored"),
    pytest.param(("rows1000", 1000, 8192, 3.0, 0.05, False, "cpu"), id="rows1000"),
    pytest.param(("rows1000_v32768", 1000, R12["V"], 3.0, 0.05, True, "cuda"), id="rows1000_v32768",
                 marks=GPU),
    pytest.param(("r12", R12["N"], R12["V"], 3.0, 0.01, True, "cuda"), id="r12", marks=REAL),
]


@pytest.fixture(params=CE_CASES)
def ce_case(request):
    name, N, V, scale, ign_frac, saturate, dev = request.param
    dev = require_device(dev)
    g = gen(SEED, dev)
    logits = torch.randn((N, V), generator=g, device=dev) * scale
    targets = torch.randint(0, V, (N,), generator=g, device=dev)
    if saturate:  # a block of large logits, some exactly at the target column (softcap saturation)
        rows = torch.arange(0, N, 3, device=dev)
        logits[rows, targets[rows]] = 80.0
        logits[rows, (targets[rows] + 1) % V] = -75.0
        logits[: min(N, 4), : min(V, 64)] = 60.0
    ignore = torch.rand((N,), generator=g, device=dev) < ign_frac
    if ign_frac >= 1.0:
        ignore[:] = True
    targets = torch.where(ignore, torch.full_like(targets, ref_ops.IGNORE_INDEX), targets)
    return dict(name=name, logits=logits.to(torch.bfloat16), targets=targets, ignore=ignore,
                V=V, device=dev)


# --- relu^2: (rows, 4C) bf16 --------------------------------------------------------------------
RELU2_CASES = [
    pytest.param(("tiny", (64, R12["F"]), "cpu"), id="tiny"),
    pytest.param(("rows1000", (1000, R12["F"]), "cpu"), id="rows1000"),
    pytest.param(("r12", (R12["N"], R12["F"]), "cuda"), id="r12", marks=REAL),
]


@pytest.fixture(params=RELU2_CASES)
def relu2_case(request):
    name, shape, dev = request.param
    dev = require_device(dev)
    return dict(name=name, x=randn(shape, SEED, dev, 0.5), device=dev)


# --- residual / x0 mix: x, x0 (B, T, C) bf16, lambdas as nanochat's init_weights sets them ------
RESID_CASES = [
    pytest.param(("tiny", (2, 16, R12["C"]), "cpu"), id="tiny"),
    pytest.param(("r12", (R12["B"], R12["T"], R12["C"]), "cuda"), id="r12", marks=REAL),
]


def nanochat_lambdas(n_layer=12, device="cpu"):
    """resid_lambdas / x0_lambdas as GPT.init_weights sets them (fp32 parameters)."""
    # same Python-float formula as init_weights, then stored into fp32 parameters
    r = torch.tensor([1.15 - (0.10 * i / max(n_layer - 1, 1)) for i in range(n_layer)], dtype=torch.float32)
    l = torch.tensor([0.20 - (0.15 * i / max(n_layer - 1, 1)) for i in range(n_layer)], dtype=torch.float32)
    return torch.nn.Parameter(r.to(device)), torch.nn.Parameter(l.to(device))


@pytest.fixture(params=RESID_CASES)
def resid_case(request):
    name, shape, dev = request.param
    dev = require_device(dev)
    r, l = nanochat_lambdas(device=dev)
    return dict(name=name, x=randn(shape, SEED, dev, 0.5), x0=randn(shape, SEED + 1, dev, 0.5),
                r_all=r, l_all=l, layer=5, device=dev)
