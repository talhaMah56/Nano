"""
Kernel registry for the static cross-ISA pipeline (task card AMD-1).

One `KernelSpec` per compiled Triton kernel: the JIT function, the ASTSource signature / constexprs /
divisibility attributes that a real launch at the R12 shape specialises on, and the GH200 launch options.
Nothing here launches anything; `cisc662.analysis.xcompile` feeds these specs to
`triton.compile(ASTSource(...), target=GPUTarget(...))` without a GPU.

Where the specs come from (copied, not re-derived, so they follow the kernel files):
  rmsnorm   cisc662/kernels/triton/rmsnorm.py  -> isa_specs("meta") (meta tensors at the R12 shapes, CONFIGS
            for BLOCK_M/num_warps) and _signature() (Triton's runtime specialisation: 16-B aligned pointers,
            ints % 16 == 0 get tt.divisibility=16). Hints off = no attrs and ALIGNED=False.
  qkrope    cisc662/kernels/triton/qkrope.py   -> VARIANTS {name: (BLOCK_M, num_warps)}, R12_SHAPE; signature
            built exactly as its `--dump-isa --mode xcompile` path builds it. Hints off = no attrs.
  matmul    the Triton tutorial matmul (03-matrix-multiplication.py, grouped launch order, ACTIVATION="")
            defined below, bf16 in / fp32 accumulate / bf16 out, tile 128x128x64, GROUP_SIZE_M=8, at the
            nanochat c_fc shape M=32768 (B*T), K=768, N=3072; options num_warps=8, num_stages=3 (the S1
            cross-compile config; this tile has not been tuned on GH200). Unit strides are constexpr 1 as a
            launch specialises them. Hints off = no attrs.

The kernel modules are imported read-only; if their signatures change, this file follows automatically for
rmsnorm (it calls the module's own spec builders) and for qkrope (arg names are read from the JIT function).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import triton
import triton.language as tl

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D16 = [["tt.divisibility", 16]]


@dataclass
class KernelSpec:
    kernel_id: str               # results/isa/<kernel_id>/...
    variant: str                 # results/isa/<kernel_id>/triton_<variant>/...
    fn: object                   # triton JITFunction
    signature: Dict[str, str]
    constexprs: Dict[str, object]
    attrs: Dict[tuple, list]
    options: Dict[str, int]      # GH200 launch options (num_warps, optionally num_stages)
    source: str                  # repo-relative path of the kernel source
    shape: str                   # human-readable launch shape
    sweep: str = "ofat"          # "ofat" (one factor at a time from base) | "full" | "none"
    has_loop: bool = False       # num_stages only matters for kernels with a pipelined loop
    hints_off_constexprs: Dict[str, object] = field(default_factory=dict)
    notes: str = ""

    @property
    def kname(self) -> str:
        return self.fn.__name__ if hasattr(self.fn, "__name__") else self.fn.fn.__name__

    def build(self, hints: bool = True):
        """(signature, constexprs, attrs) for ASTSource, with or without divisibility hints."""
        if hints:
            return dict(self.signature), dict(self.constexprs), {k: list(v) for k, v in self.attrs.items()}
        const = dict(self.constexprs)
        const.update(self.hints_off_constexprs)
        return dict(self.signature), const, {}

    def source_sha(self) -> str:
        p = os.path.join(REPO, self.source)
        try:
            with open(p, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:12]
        except OSError:
            return "n/a"


# ----------------------------------------------------------------------------------------------
# Triton tutorial matmul (python/tutorials/03-matrix-multiplication.py, triton 3.5.x), autotune removed.
# Lives here on purpose: it is the GEMM reference row of the static table, not a project kernel.
# ----------------------------------------------------------------------------------------------
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr, ACTIVATION: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.bfloat16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


MATMUL_SHAPE = dict(M=16 * 2048, N=3072, K=768)     # nanochat d12 c_fc at B=16, T=2048
MATMUL_TILE = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8)


def _matmul_specs() -> List[KernelSpec]:
    M, N, K = MATMUL_SHAPE["M"], MATMUL_SHAPE["N"], MATMUL_SHAPE["K"]
    # row-major A (M,K), B (K,N), C (M,N); unit strides are specialised to constexpr 1 by a launch
    vals = dict(M=M, N=N, K=K, stride_am=K, stride_ak=1, stride_bk=N, stride_bn=1, stride_cm=N, stride_cn=1)
    sig, const, attrs = {}, {}, {}
    for i, name in enumerate(matmul_kernel.arg_names):
        if name in ("a_ptr", "b_ptr", "c_ptr"):
            sig[name] = "*bf16"
            attrs[(i,)] = D16
        elif name in vals:
            if vals[name] == 1:
                sig[name] = "constexpr"
                const[name] = 1
            else:
                sig[name] = "i32"
                if vals[name] % 16 == 0:
                    attrs[(i,)] = D16
        else:
            sig[name] = "constexpr"
            const[name] = MATMUL_TILE.get(name, "")      # ACTIVATION=""
    return [KernelSpec(
        kernel_id="matmul", variant="tutorial_128x128x64", fn=matmul_kernel, signature=sig, constexprs=const,
        attrs=attrs, options=dict(num_warps=8, num_stages=3), source="cisc662/analysis/kernel_registry.py",
        shape=f"A[{M},{K}] @ B[{K},{N}] bf16 -> C bf16 (fp32 acc)", sweep="full", has_loop=True,
        notes="Triton tutorial matmul, 128x128x64 tile, GROUP_SIZE_M=8; options = S1 cross-compile config")]


# ----------------------------------------------------------------------------------------------
# RMSNorm family (KT-1): reuse the module's own meta-tensor specs and signature builder
# ----------------------------------------------------------------------------------------------
def _rmsnorm_specs() -> List[KernelSpec]:
    from cisc662.kernels.triton import rmsnorm as R
    out = []
    for name, spec in R.isa_specs("meta").items():
        kernel, grid, args, const, opts = spec
        sig, attrs = R._signature(kernel, args, const)
        variant = name[len("triton_"):] if name.startswith("triton_") else name
        shapes = {k: list(v.shape) for k, v in args.items() if hasattr(v, "shape") and len(v.shape) == 2}
        shape = ", ".join(f"{k}{v}" for k, v in list(shapes.items())[:2]) or "see source"
        hoff = {"ALIGNED": False} if "ALIGNED" in const else {}
        out.append(KernelSpec(
            kernel_id="rmsnorm", variant=variant, fn=kernel, signature=sig, constexprs=dict(const), attrs=attrs,
            options={k: v for k, v in opts.items() if k in ("num_warps", "num_stages")},
            source="cisc662/kernels/triton/rmsnorm.py", shape=f"{shape} bf16, grid={list(grid)}",
            sweep="ofat", has_loop="finish" in variant, hints_off_constexprs=hoff,
            notes=f"rmsnorm.CONFIGS launch (num_warps={opts.get('num_warps')}); constexprs {dict(const)}"))
    return out


# ----------------------------------------------------------------------------------------------
# Fused QK-norm + RoPE (KT-2)
# ----------------------------------------------------------------------------------------------
def _qkrope_specs() -> List[KernelSpec]:
    from cisc662.kernels.triton import qkrope as Q
    B, T, H, D = Q.R12_SHAPE
    n = B * T * H
    out = []
    for v, (bm, nw) in Q.VARIANTS.items():
        gq = triton.cdiv(n, bm)
        vals = dict(n_rows_q=n, n_rows_k=n, n_blk_q=gq, Hq=H, Hk=H, T=T, cs_stride=D // 2)
        for kern in (Q.c662_qkrope_fwd, Q.c662_qkrope_bwd):
            const = dict(D=D, BLOCK_M=bm, F64=False, I64=False, EMU_RTNE=False)
            sig, attrs = {}, {}
            for i, a in enumerate(kern.arg_names):
                if a in const:
                    sig[a] = "constexpr"
                elif a.endswith("_ptr"):
                    sig[a] = "*fp32" if a in ("rq_ptr", "rk_ptr") else "*bf16"
                    attrs[(i,)] = D16
                elif a in vals:
                    sig[a] = "i32"
                    if vals[a] % 16 == 0:
                        attrs[(i,)] = D16
                elif a in ("scale", "eps"):
                    sig[a] = "fp32"
                else:
                    raise KeyError(f"qkrope: unknown argument {a!r} of {kern.__name__}; update kernel_registry")
            out.append(KernelSpec(
                kernel_id="qkrope", variant=v, fn=kern, signature=sig, constexprs=const, attrs=attrs,
                options=dict(num_warps=nw), source="cisc662/kernels/triton/qkrope.py",
                shape=f"q,k[{B},{T},{H},{D}] bf16, cos/sin[1,{T},1,{D // 2}], grid=[{2 * gq}]", sweep="ofat",
                notes=f"qkrope.VARIANTS[{v!r}] = (BLOCK_M={bm}, num_warps={nw})"))
    return out


_BUILDERS: Dict[str, Callable[[], List[KernelSpec]]] = {
    "rmsnorm": _rmsnorm_specs, "qkrope": _qkrope_specs, "matmul": _matmul_specs}


def kernel_ids() -> List[str]:
    return list(_BUILDERS)


def specs(kernel_ids_: Optional[List[str]] = None) -> List[KernelSpec]:
    out = []
    for k in (kernel_ids_ or kernel_ids()):
        out.extend(_BUILDERS[k]())
    return out


if __name__ == "__main__":
    for s in specs():
        sig, const, attrs = s.build()
        print(f"{s.kernel_id:8s} {s.variant:22s} {s.kname:34s} opts={s.options} attrs={len(attrs)} "
              f"const={const}")
