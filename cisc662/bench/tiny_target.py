"""Tiny GPU workload used only to check that nsys / ncu work for non-admin users on this system."""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


if __name__ == "__main__":
    n = 1 << 24
    x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        _add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
        torch.mm(a, a)
    torch.cuda.synchronize()
    print("tiny_target ok")
