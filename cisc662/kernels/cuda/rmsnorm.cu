// RMSNorm v0 (AMD-2 source; KC-1 will load, test and time it). Parameter-free, as nanochat's norm():
//   y = x * rsqrt(mean(x^2) + eps), eps = fp32 epsilon, bf16 in/out, fp32 accumulation.
// Design: one 32-lane group per row. Each lane loads VEC=8 bf16 (one 16-byte uint4) per piece and issues ALL of its
// loads before using any (ILP = D / (32*8) pieces: 3 for D=768, 1 for D=128 with 16 lanes). 128-bit loads/stores,
// fp32 sum of squares, XOR-shuffle reduction inside the 32-lane group, rsqrtf, multiply, pack to bf16, store.
// No shared memory. ROWS_PER_BLOCK groups per 256-thread block.
// Backward: dx = rstd * (dy - xhat * mean(dy * xhat)), xhat = x * rstd, rstd is recomputed from x (not loaded).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define FP32_EPS 1.1920929e-07f

template <int D>
struct Cfg {                    // lanes per row, 16-byte pieces per lane
    static constexpr int LANES = (D >= 256) ? 32 : D / 8;
    static constexpr int PIECES = D / (LANES * 8);
};

__device__ __forceinline__ float group_sum(float v, int lanes) {
    for (int m = lanes / 2; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    return v;
}

__device__ __forceinline__ void unpack(uint4 u, float* f) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) { float2 t = __bfloat1622float2(p[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}

__device__ __forceinline__ uint4 pack(const float* f) {
    uint4 u; __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return u;
}

template <int D>
__device__ __forceinline__ void rms_fwd_row(const uint4* __restrict__ x, uint4* __restrict__ y, int lane) {
    constexpr int L = Cfg<D>::LANES, P = Cfg<D>::PIECES;
    uint4 v[P];
#pragma unroll
    for (int p = 0; p < P; ++p) v[p] = x[p * L + lane];           // all loads first (ILP = P)
    float f[P][8], ss = 0.f;
#pragma unroll
    for (int p = 0; p < P; ++p) {
        unpack(v[p], f[p]);
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = fmaf(f[p][i], f[p][i], ss);
    }
    ss = group_sum(ss, L);
    const float rstd = rsqrtf(ss * (1.0f / D) + FP32_EPS);
#pragma unroll
    for (int p = 0; p < P; ++p) {
#pragma unroll
        for (int i = 0; i < 8; ++i) f[p][i] *= rstd;
        y[p * L + lane] = pack(f[p]);
    }
}

template <int D>
__global__ void __launch_bounds__(256) c662_rmsnorm_fwd(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, int rows) {
    constexpr int L = Cfg<D>::LANES, G = 32 / L;                  // row groups per warp
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (blockIdx.x * 8 + warp) * G + lane / L;
    if (row >= rows) return;
    rms_fwd_row<D>(reinterpret_cast<const uint4*>(x) + (size_t)row * (D / 8), reinterpret_cast<uint4*>(y) + (size_t)row * (D / 8), lane % L);
}

template <int D>
__global__ void __launch_bounds__(256) c662_rmsnorm_bwd(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ dy,
                                                         __nv_bfloat16* __restrict__ dx, int rows) {
    constexpr int L = Cfg<D>::LANES, P = Cfg<D>::PIECES, G = 32 / L;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (blockIdx.x * 8 + warp) * G + lane / L, l = lane % L;
    if (row >= rows) return;
    const uint4* xr = reinterpret_cast<const uint4*>(x) + (size_t)row * (D / 8);
    const uint4* gr = reinterpret_cast<const uint4*>(dy) + (size_t)row * (D / 8);
    uint4 vx[P], vg[P];
#pragma unroll
    for (int p = 0; p < P; ++p) { vx[p] = xr[p * L + l]; vg[p] = gr[p * L + l]; }
    float fx[P][8], fg[P][8], ss = 0.f;
#pragma unroll
    for (int p = 0; p < P; ++p) {
        unpack(vx[p], fx[p]); unpack(vg[p], fg[p]);
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = fmaf(fx[p][i], fx[p][i], ss);
    }
    ss = group_sum(ss, L);
    const float rstd = rsqrtf(ss * (1.0f / D) + FP32_EPS);
    float dot = 0.f;
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
        for (int i = 0; i < 8; ++i) dot = fmaf(fg[p][i], fx[p][i] * rstd, dot);
    dot = group_sum(dot, L) * (1.0f / D);
    uint4* dr = reinterpret_cast<uint4*>(dx) + (size_t)row * (D / 8);
#pragma unroll
    for (int p = 0; p < P; ++p) {
#pragma unroll
        for (int i = 0; i < 8; ++i) fx[p][i] = rstd * (fg[p][i] - fx[p][i] * rstd * dot);
        dr[p * L + l] = pack(fx[p]);
    }
}

// explicit instantiations (compile-check and ISA dumps; KC-1 adds the torch extension wrapper)
template __global__ void c662_rmsnorm_fwd<768>(const __nv_bfloat16*, __nv_bfloat16*, int);
template __global__ void c662_rmsnorm_fwd<128>(const __nv_bfloat16*, __nv_bfloat16*, int);
template __global__ void c662_rmsnorm_bwd<768>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int);
template __global__ void c662_rmsnorm_bwd<128>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int);
