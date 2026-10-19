# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/jg/cjg3kd5j3hptxoqlltzf2kboxkh6sqj6ryh6eg5fzx4jdsrdiojn.py
# Topologically Sorted Source Nodes: [x, x_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#   x => embedding
#   x_2 => add, convert_element_type, mean, pow_1, rsqrt
# Graph fragment:
#   %primals_1 : Tensor "i64[16, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %primals_4 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %embedding : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=embedding]
#   %buf1 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf1]
#   %embedding : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_4, %primals_1), kwargs = {})
#   %convert_element_type : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   return %embedding,%buf1,%rsqrt
triton_per_fused__to_copy_add_embedding_mean_pow_rsqrt_0 = async_compile.triton('triton_per_fused__to_copy_add_embedding_mean_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mean_pow_rsqrt_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mean_pow_rsqrt_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, R0_BLOCK], 32768, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32768), "index out of bounds: 0 <= tmp4 < 32768")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 768*tmp4), r0_mask, other=0.0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp13 = 768.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1.1920928955078125e-07
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp6, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/zw/czwk5r43e2lhcct3nidcdawk3alxp5hmed2vtd2arfdirptd4tgf.py
# Topologically Sorted Source Nodes: [to_2, linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => permute
#   to_2 => convert_element_type_3
# Graph fragment:
#   %primals_6 : Tensor "f32[1, 24][24, 1]cuda:0" = PlaceHolder[target=primals_6]
#   %convert_element_type_3 : Tensor "bf16[1, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_6, torch.bfloat16), kwargs = {})
#   %permute : Tensor "bf16[24, 1][1, 24]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_3, [1, 0]), kwargs = {})
#   return %permute
triton_poi_fused__to_copy_t_1 = async_compile.triton('triton_poi_fused__to_copy_t_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/dw/cdwqnpcmeim5imwbqibij54yrgxzstmdszd2r2q55ozsy4n3u4mh.py
# Topologically Sorted Source Nodes: [x_2, getitem_2, linear], Original ATen: [aten._to_copy, aten.mul, aten.slice, aten.clone]
# Source node to ATen node mapping:
#   getitem_2 => slice_3, slice_4
#   linear => clone
#   x_2 => convert_element_type, convert_element_type_1, mul
# Graph fragment:
#   %embedding : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %convert_element_type : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %slice_3 : Tensor "bf16[16, 2047, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_1, 1, 1, 9223372036854775807), kwargs = {})
#   %slice_4 : Tensor "bf16[16, 2047, 24][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_3, 2, 0, 24), kwargs = {})
#   %clone : Tensor "bf16[16, 2047, 24][49128, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused__to_copy_clone_mul_slice_2 = async_compile.triton('triton_poi_fused__to_copy_clone_mul_slice_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_mul_slice_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4716288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_mul_slice_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 24)
    x1 = ((xindex // 24) % 2047)
    x2 = xindex // 49128
    x3 = (xindex % 49128)
    tmp0 = tl.load(in_ptr0 + (768 + x0 + 768*x1 + 1572864*x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (1 + x1 + 2048*x2), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 49152*x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/vx/cvxws2fz6vz4yxz5xfnh5zvtjwtau6iuag3lpnln7poqnv5u7vmd.py
# Topologically Sorted Source Nodes: [x_2, getitem_2, linear], Original ATen: [aten._to_copy, aten.mul, aten.slice, aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
#   getitem_2 => slice_3, slice_4
#   linear => clone, view
#   x_2 => convert_element_type, convert_element_type_1, mul
# Graph fragment:
#   %clone : Tensor "bf16[16, 2047, 24][49152, 24, 1]cuda:0" = PlaceHolder[target=clone]
#   %convert_element_type : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %slice_3 : Tensor "bf16[16, 2047, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_1, 1, 1, 9223372036854775807), kwargs = {})
#   %slice_4 : Tensor "bf16[16, 2047, 24][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_3, 2, 0, 24), kwargs = {})
#   %clone : Tensor "bf16[16, 2047, 24][49128, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view : Tensor "bf16[32752, 24][24, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [32752, 24]), kwargs = {})
#   return %view
triton_poi_fused__to_copy__unsafe_view_clone_mul_slice_3 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_clone_mul_slice_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_clone_mul_slice_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4716288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_clone_mul_slice_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 24)
    x1 = xindex // 24
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 24*((x1 % 2047)) + 49152*(x1 // 2047)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/zc/czcjafq6hgszirwplrxkdtwhk6j477jhmbuag4ydy3m64kqmhgv6.py
# Topologically Sorted Source Nodes: [x_2, to_1, getitem_2, linear, sigmoid, gate, getitem_3, getitem_5, mul_1, add, x_3, getitem_6, mul_2, getitem_7, mul_3, x_4, rms_norm_1], Original ATen: [aten._to_copy, aten.mul, aten.slice, aten._unsafe_view, aten.sigmoid, aten.add, aten.cat, aten.select, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   add => add_1
#   gate => mul_1
#   getitem_2 => slice_3
#   getitem_3 => slice_5
#   getitem_5 => slice_7
#   getitem_6 => select
#   getitem_7 => select_1
#   linear => view_1
#   mul_1 => mul_2
#   mul_2 => mul_3
#   mul_3 => mul_4
#   rms_norm_1 => add_3, convert_element_type_6, convert_element_type_7, mean_1, mul_5, pow_2, rsqrt_1
#   sigmoid => sigmoid
#   to_1 => convert_element_type_2
#   x_2 => convert_element_type, convert_element_type_1, mul
#   x_3 => cat
#   x_4 => add_2
# Graph fragment:
#   %embedding : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %primals_5 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_5]
#   %mm : Tensor "bf16[32752, 1][1, 1]cuda:0" = PlaceHolder[target=mm]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %buf8 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf8]
#   %rsqrt_1 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %convert_element_type : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.bfloat16), kwargs = {})
#   %slice_3 : Tensor "bf16[16, 2047, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_1, 1, 1, 9223372036854775807), kwargs = {})
#   %view_1 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [16, 2047, 1]), kwargs = {})
#   %sigmoid : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_1,), kwargs = {})
#   %mul_1 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %sigmoid), kwargs = {})
#   %slice_5 : Tensor "bf16[16, 1, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_1, 1, 0, 1), kwargs = {})
#   %slice_7 : Tensor "bf16[16, 2047, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_1, 1, 0, -1), kwargs = {})
#   %mul_2 : Tensor "bf16[16, 2047, 768][1572096, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %slice_7), kwargs = {})
#   %add_1 : Tensor "bf16[16, 2047, 768][1572096, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_3, %mul_2), kwargs = {})
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=14] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_5, %add_1], 1), kwargs = {})
#   %select : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 0), kwargs = {})
#   %mul_3 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %cat), kwargs = {})
#   %select_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 0), kwargs = {})
#   %mul_4 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %cat), kwargs = {})
#   %add_2 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_4), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_6, 2), kwargs = {})
#   %mean_1 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_3 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_1 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_5 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %rsqrt_1), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.bfloat16), kwargs = {})
#   return %cat,%buf8,%rsqrt_1,%convert_element_type_7
triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_pow_rsqrt_select_sigmoid_slice_4 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_pow_rsqrt_select_sigmoid_slice_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_pow_rsqrt_select_sigmoid_slice_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 10, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 589824, 'r0_': 301989888}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_pow_rsqrt_select_sigmoid_slice_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    x0 = (xindex % 2048)
    r0_2 = r0_index
    x1 = xindex // 2048
    x3 = xindex
    tmp37 = tl.load(in_ptr4 + (0))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, R0_BLOCK])
    tmp41 = tl.load(in_ptr5 + (0))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, R0_BLOCK])
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r0_2 + 768*(x0) + 1572864*x1), r0_mask & tmp4, other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (tl.broadcast_to(2048*x1 + (x0), [XBLOCK, R0_BLOCK])), r0_mask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 2048, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr0 + (768 + r0_2 + 768*((-1) + x0) + 1572864*x1), r0_mask & tmp12, other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.load(in_ptr1 + (tl.broadcast_to(1 + 2048*x1 + ((-1) + x0), [XBLOCK, R0_BLOCK])), r0_mask & tmp12, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.load(in_ptr2 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(tmp12, tmp21, 0.0)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.load(in_ptr3 + (tl.broadcast_to(2047*x1 + ((-1) + x0), [XBLOCK, R0_BLOCK])), r0_mask & tmp12, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp23 * tmp25
    tmp27 = tl.load(in_ptr0 + (r0_2 + 768*((-1) + x0) + 1572864*x1), r0_mask & tmp12, other=0.0).to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.load(in_ptr1 + (tl.broadcast_to(2048*x1 + ((-1) + x0), [XBLOCK, R0_BLOCK])), r0_mask & tmp12, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp26 * tmp31
    tmp33 = tmp19 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp12, tmp33, tmp34)
    tmp36 = tl.where(tmp4, tmp11, tmp35)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp39 * tmp36
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp43 * tmp36
    tmp45 = tmp40 + tmp44
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46 * tmp46
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, R0_BLOCK])
    tmp50 = tl.where(r0_mask, tmp48, 0)
    tmp51 = tl.sum(tmp50, 1)[:, None].to(tl.float32)
    tmp52 = 768.0
    tmp53 = (tmp51 / tmp52)
    tmp54 = 1.1920928955078125e-07
    tmp55 = tmp53 + tmp54
    tmp56 = libdevice.rsqrt(tmp55)
    tmp57 = tmp46 * tmp56
    tmp58 = tmp57.to(tl.float32)
    tl.store(out_ptr0 + (r0_2 + 768*x3), tmp36, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp56, None)
    tl.store(out_ptr1 + (r0_2 + 768*x3), tmp58, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/og/cog5mdujmlxplml6c5ogvyo2k7gx56yhdcojymppyabim5x2qknh.py
# Topologically Sorted Source Nodes: [to_3, linear_1], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_1 => permute_1
#   to_3 => convert_element_type_8
# Graph fragment:
#   %primals_9 : Tensor "f32[768, 768][768, 1]cuda:0" = PlaceHolder[target=primals_9]
#   %convert_element_type_8 : Tensor "bf16[768, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_9, torch.bfloat16), kwargs = {})
#   %permute_1 : Tensor "bf16[768, 768][1, 768]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_8, [1, 0]), kwargs = {})
#   return %permute_1
triton_poi_fused__to_copy_t_5 = async_compile.triton('triton_poi_fused__to_copy_t_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/vm/cvmpnq3swcnennyypkz64qxr4hgn2ngfjv7lc3poppk6uzvf3ogx.py
# Topologically Sorted Source Nodes: [cos, sin, linear_1, q, linear_2, k, x1, x2, mul_4, mul_5, y1, neg, mul_6, mul_7, y2, q_1, x1_1, x2_1, mul_8, mul_9, y1_1, mul_10, mul_11, y2_1, k_1, q_2, k_2, q_3, k_3], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   cos => slice_1
#   k => view_7
#   k_1 => cat_2
#   k_2 => add_9, convert_element_type_19, convert_element_type_20, mean_3, mul_15, pow_4, rsqrt_3
#   k_3 => mul_17
#   linear_1 => view_3
#   linear_2 => view_6
#   mul_10 => mul_12
#   mul_11 => mul_13
#   mul_4 => mul_6
#   mul_5 => mul_7
#   mul_6 => mul_8
#   mul_7 => mul_9
#   mul_8 => mul_10
#   mul_9 => mul_11
#   neg => neg
#   q => view_4
#   q_1 => cat_1
#   q_2 => add_8, convert_element_type_17, convert_element_type_18, mean_2, mul_14, pow_3, rsqrt_2
#   q_3 => mul_16
#   sin => slice_2
#   x1 => slice_8
#   x1_1 => slice_10
#   x2 => slice_9
#   x2_1 => slice_11
#   y1 => add_4
#   y1_1 => add_6
#   y2 => add_5
#   y2_1 => add_7
# Graph fragment:
#   %mm_1 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %primals_2 : Tensor "bf16[1, 20480, 1, 64][1310720, 64, 64, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "bf16[1, 20480, 1, 64][1310720, 64, 64, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %mm_2 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %cat_1 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %cat_2 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0" = PlaceHolder[target=cat_2]
#   %buf19 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 196608]cuda:0" = PlaceHolder[target=buf19]
#   %buf21 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 196608]cuda:0" = PlaceHolder[target=buf21]
#   %rsqrt_2 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_2]
#   %rsqrt_3 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %slice_1 : Tensor "bf16[1, 2048, 1, 64][1310720, 64, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_2, 1, 0, 2048), kwargs = {})
#   %slice_2 : Tensor "bf16[1, 2048, 1, 64][1310720, 64, 64, 1]cuda:0"[num_users=25] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_3, 1, 0, 2048), kwargs = {})
#   %view_3 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [16, 2048, 768]), kwargs = {})
#   %view_4 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [16, 2048, 6, 128]), kwargs = {})
#   %view_6 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [16, 2048, 768]), kwargs = {})
#   %view_7 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_6, [16, 2048, 6, 128]), kwargs = {})
#   %slice_8 : Tensor "bf16[16, 2048, 6, 64][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_4, 3, 0, 64), kwargs = {})
#   %slice_9 : Tensor "bf16[16, 2048, 6, 64][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_4, 3, 64, 9223372036854775807), kwargs = {})
#   %mul_6 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_8, %slice_1), kwargs = {})
#   %mul_7 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_9, %slice_2), kwargs = {})
#   %add_4 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %neg : Tensor "bf16[1, 2048, 1, 64][131072, 64, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %mul_8 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_8, %neg), kwargs = {})
#   %mul_9 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_9, %slice_1), kwargs = {})
#   %add_5 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %cat_1 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_4, %add_5], 3), kwargs = {})
#   %slice_10 : Tensor "bf16[16, 2048, 6, 64][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_7, 3, 0, 64), kwargs = {})
#   %slice_11 : Tensor "bf16[16, 2048, 6, 64][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%view_7, 3, 64, 9223372036854775807), kwargs = {})
#   %mul_10 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_10, %slice_1), kwargs = {})
#   %mul_11 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_11, %slice_2), kwargs = {})
#   %add_6 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %mul_11), kwargs = {})
#   %mul_12 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_10, %neg), kwargs = {})
#   %mul_13 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_11, %slice_1), kwargs = {})
#   %add_7 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_13), kwargs = {})
#   %cat_2 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_6, %add_7], 3), kwargs = {})
#   %convert_element_type_17 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_17, 2), kwargs = {})
#   %mean_2 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [3], True), kwargs = {})
#   %add_8 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_2, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_2 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_14 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, %rsqrt_2), kwargs = {})
#   %convert_element_type_18 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14, torch.bfloat16), kwargs = {})
#   %convert_element_type_19 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_2, torch.float32), kwargs = {})
#   %pow_4 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_19, 2), kwargs = {})
#   %mean_3 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [3], True), kwargs = {})
#   %add_9 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_3, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_3 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_15 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %rsqrt_3), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
#   %mul_16 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, 1.2), kwargs = {})
#   %mul_17 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, 1.2), kwargs = {})
#   return %cat_1,%cat_2,%buf19,%buf21,%rsqrt_2,%rsqrt_3,%mul_16,%mul_17
triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 12, 'num_reduction': 2, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 738197504}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 196608
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x4 = xindex
    x1 = ((xindex // 6) % 2048)
    tmp0 = r0_3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x4 + (r0_3)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (64*x1 + (r0_3)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.load(in_ptr0 + (64 + 128*x4 + (r0_3)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (64*x1 + (r0_3)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 128, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr0 + (128*x4 + ((-64) + r0_3)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (64*x1 + ((-64) + r0_3)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = -tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tl.load(in_ptr0 + (64 + 128*x4 + ((-64) + r0_3)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr1 + (64*x1 + ((-64) + r0_3)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp14, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp13, tmp26)
    tmp28 = tl.load(in_ptr3 + (128*x4 + (r0_3)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tmp28 * tmp6
    tmp30 = tl.load(in_ptr3 + (64 + 128*x4 + (r0_3)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp30 * tmp9
    tmp32 = tmp29 + tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp4, tmp32, tmp33)
    tmp35 = tl.load(in_ptr3 + (128*x4 + ((-64) + r0_3)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp35 * tmp19
    tmp37 = tl.load(in_ptr3 + (64 + 128*x4 + ((-64) + r0_3)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp38 = tmp37 * tmp22
    tmp39 = tmp36 + tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp14, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp34, tmp41)
    tmp43 = tmp27.to(tl.float32)
    tmp44 = tmp43 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
    tmp47 = tl.sum(tmp45, 1)[:, None].to(tl.float32)
    tmp48 = tmp42.to(tl.float32)
    tmp49 = tmp48 * tmp48
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK, R0_BLOCK])
    tmp52 = tl.sum(tmp50, 1)[:, None].to(tl.float32)
    tmp53 = 128.0
    tmp54 = (tmp47 / tmp53)
    tmp55 = 1.1920928955078125e-07
    tmp56 = tmp54 + tmp55
    tmp57 = libdevice.rsqrt(tmp56)
    tmp58 = (tmp52 / tmp53)
    tmp59 = tmp58 + tmp55
    tmp60 = libdevice.rsqrt(tmp59)
    tmp61 = tmp43 * tmp57
    tmp62 = tmp61.to(tl.float32)
    tmp63 = 1.2
    tmp64 = tmp62 * tmp63
    tmp65 = tmp48 * tmp60
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tmp66 * tmp63
    tl.store(out_ptr0 + (r0_3 + 128*x4), tmp27, None)
    tl.store(out_ptr1 + (r0_3 + 128*x4), tmp42, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp57, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp60, None)
    tl.store(out_ptr2 + (r0_3 + 128*x4), tmp64, None)
    tl.store(out_ptr3 + (r0_3 + 128*x4), tmp67, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/3z/c3z7aft5l4aqp6fsoqarnh7ac2d3hu5mfdptyykarthzjmixip3l.py
# Topologically Sorted Source Nodes: [getitem_6, mul_2, getitem_7, mul_3, x_4, y_2, x_5, rms_norm_4], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_6 => select
#   getitem_7 => select_1
#   mul_2 => mul_3
#   mul_3 => mul_4
#   rms_norm_4 => add_11, convert_element_type_24, convert_element_type_25, mean_4, mul_18, pow_5, rsqrt_4
#   x_4 => add_2
#   x_5 => add_10
#   y_2 => view_13
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %mm_4 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %add_10 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_10]
#   %buf33 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf33]
#   %rsqrt_4 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_4]
#   %select : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 0), kwargs = {})
#   %mul_3 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %cat), kwargs = {})
#   %select_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 0), kwargs = {})
#   %mul_4 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %cat), kwargs = {})
#   %add_2 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_4), kwargs = {})
#   %view_13 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [16, 2048, 768]), kwargs = {})
#   %add_10 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_13), kwargs = {})
#   %convert_element_type_24 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.float32), kwargs = {})
#   %pow_5 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_24, 2), kwargs = {})
#   %mean_4 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [2], True), kwargs = {})
#   %add_11 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_4, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_4 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_18 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %rsqrt_4), kwargs = {})
#   %convert_element_type_25 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_18, torch.bfloat16), kwargs = {})
#   return %add_10,%buf33,%rsqrt_4,%convert_element_type_25
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_7 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 301989888}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp4 + tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = 768.0
    tmp19 = (tmp17 / tmp18)
    tmp20 = 1.1920928955078125e-07
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp12 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp11, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp22, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp24, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/qr/cqrxfap7du52jgxlmb2xtllqwdaixzuvap4esvughgh437yu3ugx.py
# Topologically Sorted Source Nodes: [to_7, x_6], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   to_7 => convert_element_type_26
#   x_6 => permute_5
# Graph fragment:
#   %primals_13 : Tensor "f32[3072, 768][768, 1]cuda:0" = PlaceHolder[target=primals_13]
#   %convert_element_type_26 : Tensor "bf16[3072, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_13, torch.bfloat16), kwargs = {})
#   %permute_5 : Tensor "bf16[768, 3072][1, 768]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_26, [1, 0]), kwargs = {})
#   return %permute_5
triton_poi_fused__to_copy_t_8 = async_compile.triton('triton_poi_fused__to_copy_t_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/h5/ch5xbxuy47ycesyhpnacwukr2d7sfizjvqswbnkd6gs25w2yvrlp.py
# Topologically Sorted Source Nodes: [x_6, relu, x_7], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
# Source node to ATen node mapping:
#   relu => relu
#   x_6 => view_15
#   x_7 => pow_6
# Graph fragment:
#   %mm_5 : Tensor "bf16[32768, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %view_15 : Tensor "bf16[16, 2048, 3072][6291456, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [16, 2048, 3072]), kwargs = {})
#   %relu : Tensor "bf16[16, 2048, 3072][6291456, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_15,), kwargs = {})
#   %pow_6 : Tensor "bf16[16, 2048, 3072][6291456, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu, 2), kwargs = {})
#   return %pow_6
triton_poi_fused__unsafe_view_pow_relu_9 = async_compile.triton('triton_poi_fused__unsafe_view_pow_relu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_pow_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 603979776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_pow_relu_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100663296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = tmp2 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/ki/cki4upiizw5vcot6sonbi7pwsv37bjutxb4ks5nicqhqxpwkujiq.py
# Topologically Sorted Source Nodes: [x_8, x_9, getitem_12, mul_14, getitem_13, mul_15, x_10, rms_norm_5], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_12 => select_2
#   getitem_13 => select_3
#   mul_14 => mul_19
#   mul_15 => mul_20
#   rms_norm_5 => add_14, convert_element_type_32, convert_element_type_33, mean_5, mul_21, pow_7, rsqrt_5
#   x_10 => add_13
#   x_8 => view_17
#   x_9 => add_12
# Graph fragment:
#   %add_10 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_10]
#   %mm_6 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_6]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_12 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_12]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %buf43 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf43]
#   %rsqrt_5 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_5]
#   %view_17 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_6, [16, 2048, 768]), kwargs = {})
#   %add_12 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_17), kwargs = {})
#   %select_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 1), kwargs = {})
#   %mul_19 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %add_12), kwargs = {})
#   %select_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 1), kwargs = {})
#   %mul_20 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %cat), kwargs = {})
#   %add_13 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %mul_20), kwargs = {})
#   %convert_element_type_32 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_13, torch.float32), kwargs = {})
#   %pow_7 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_32, 2), kwargs = {})
#   %mean_5 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [2], True), kwargs = {})
#   %add_14 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_5, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_5 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_21 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_32, %rsqrt_5), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_21, torch.bfloat16), kwargs = {})
#   return %add_12,%buf43,%rsqrt_5,%convert_element_type_33
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_10 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_10(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (1))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (1))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/cc/cccvainxridc25fjg5xea2guwbidmoa4mxwziertgw7uufcdaqc3.py
# Topologically Sorted Source Nodes: [to_13, linear_10], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_10 => permute_10
#   to_13 => convert_element_type_43
# Graph fragment:
#   %primals_19 : Tensor "f32[6, 12][12, 1]cuda:0" = PlaceHolder[target=primals_19]
#   %convert_element_type_43 : Tensor "bf16[6, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_19, torch.bfloat16), kwargs = {})
#   %permute_10 : Tensor "bf16[12, 6][1, 12]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_43, [1, 0]), kwargs = {})
#   return %permute_10
triton_poi_fused__to_copy_t_11 = async_compile.triton('triton_poi_fused__to_copy_t_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/7v/c7v5urowy2vdvfxpljyew4fs7mpmu6eofi4ajktlsfoumhjnomsi.py
# Topologically Sorted Source Nodes: [embedding_1, linear_9, v_1, ve_1, linear_10, sigmoid_1, gate_1, unsqueeze, mul_17, v_2], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
# Source node to ATen node mapping:
#   embedding_1 => embedding_1
#   gate_1 => mul_22
#   linear_10 => view_29
#   linear_9 => view_25
#   mul_17 => mul_23
#   sigmoid_1 => sigmoid_1
#   unsqueeze => unsqueeze
#   v_1 => view_26
#   v_2 => add_15
#   ve_1 => view_27
# Graph fragment:
#   %primals_1 : Tensor "i64[16, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %primals_15 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=primals_15]
#   %mm_9 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_9]
#   %mm_10 : Tensor "bf16[32768, 6][6, 1]cuda:0" = PlaceHolder[target=mm_10]
#   %embedding_1 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=embedding_1]
#   %embedding_1 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_15, %primals_1), kwargs = {})
#   %view_25 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [16, 2048, 768]), kwargs = {})
#   %view_26 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_25, [16, 2048, 6, 128]), kwargs = {})
#   %view_27 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%embedding_1, [16, 2048, 6, 128]), kwargs = {})
#   %view_29 : Tensor "bf16[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [16, 2048, 6]), kwargs = {})
#   %sigmoid_1 : Tensor "bf16[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_29,), kwargs = {})
#   %mul_22 : Tensor "bf16[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, 3), kwargs = {})
#   %unsqueeze : Tensor "bf16[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_22, -1), kwargs = {})
#   %mul_23 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %view_27), kwargs = {})
#   %add_15 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_26, %mul_23), kwargs = {})
#   return %embedding_1,%add_15
triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12 = async_compile.triton('triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 768
    x0 = (xindex % 768)
    x4 = xindex
    x3 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x4), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.full([XBLOCK], 32768, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32768), "index out of bounds: 0 <= tmp4 < 32768")
    tmp6 = tl.load(in_ptr1 + (x0 + 768*tmp4), None).to(tl.float32)
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = 3.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11 * tmp6
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x4), tmp6, None)
    tl.store(in_out_ptr0 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/3n/c3njdcgrzzd3jmmx4ujexhtjobacjueefhz3zmhcovvuhyvyi5jj.py
# Topologically Sorted Source Nodes: [getitem_12, mul_14, getitem_13, mul_15, x_10, y_5, x_11, rms_norm_8], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_12 => select_2
#   getitem_13 => select_3
#   mul_14 => mul_19
#   mul_15 => mul_20
#   rms_norm_8 => add_23, convert_element_type_53, convert_element_type_54, mean_8, mul_36, pow_10, rsqrt_8
#   x_10 => add_13
#   x_11 => add_22
#   y_5 => view_32
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_12 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_12]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %mm_11 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_11]
#   %add_22 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_22]
#   %buf71 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf71]
#   %rsqrt_8 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_8]
#   %select_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 1), kwargs = {})
#   %mul_19 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %add_12), kwargs = {})
#   %select_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 1), kwargs = {})
#   %mul_20 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %cat), kwargs = {})
#   %add_13 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %mul_20), kwargs = {})
#   %view_32 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_11, [16, 2048, 768]), kwargs = {})
#   %add_22 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_32), kwargs = {})
#   %convert_element_type_53 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.float32), kwargs = {})
#   %pow_10 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_53, 2), kwargs = {})
#   %mean_8 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_10, [2], True), kwargs = {})
#   %add_23 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_8, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_8 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_23,), kwargs = {})
#   %mul_36 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_53, %rsqrt_8), kwargs = {})
#   %convert_element_type_54 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_36, torch.bfloat16), kwargs = {})
#   return %add_22,%buf71,%rsqrt_8,%convert_element_type_54
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_13 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_13', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_13(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/wd/cwdyxg6ejeunrant3chj5x3jd7zpgzts3ujandj4mclgnvdrofoe.py
# Topologically Sorted Source Nodes: [x_14, x_15, getitem_19, mul_28, getitem_20, mul_29, x_16, rms_norm_9], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_19 => select_4
#   getitem_20 => select_5
#   mul_28 => mul_37
#   mul_29 => mul_38
#   rms_norm_9 => add_26, convert_element_type_61, convert_element_type_62, mean_9, mul_39, pow_12, rsqrt_9
#   x_14 => view_36
#   x_15 => add_24
#   x_16 => add_25
# Graph fragment:
#   %add_22 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_22]
#   %mm_13 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_13]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_24 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_24]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %buf80 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf80]
#   %rsqrt_9 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_9]
#   %view_36 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [16, 2048, 768]), kwargs = {})
#   %add_24 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %view_36), kwargs = {})
#   %select_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 2), kwargs = {})
#   %mul_37 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, %add_24), kwargs = {})
#   %select_5 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 2), kwargs = {})
#   %mul_38 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %cat), kwargs = {})
#   %add_25 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %mul_38), kwargs = {})
#   %convert_element_type_61 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_25, torch.float32), kwargs = {})
#   %pow_12 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_61, 2), kwargs = {})
#   %mean_9 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_12, [2], True), kwargs = {})
#   %add_26 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_9, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_9 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_26,), kwargs = {})
#   %mul_39 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_61, %rsqrt_9), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_39, torch.bfloat16), kwargs = {})
#   return %add_24,%buf80,%rsqrt_9,%convert_element_type_62
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_14 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_14(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (2))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (2))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/wx/cwxd3p47tcqsfcgpfpts5jkhjrfje7ewq2sgn3ysuj377r6pgfuk.py
# Topologically Sorted Source Nodes: [getitem_19, mul_28, getitem_20, mul_29, x_16, y_8, x_17, rms_norm_12], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_19 => select_4
#   getitem_20 => select_5
#   mul_28 => mul_37
#   mul_29 => mul_38
#   rms_norm_12 => add_34, convert_element_type_79, convert_element_type_80, mean_12, mul_52, pow_15, rsqrt_12
#   x_16 => add_25
#   x_17 => add_33
#   y_8 => view_48
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_24 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_24]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %mm_17 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_17]
#   %add_33 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_33]
#   %buf105 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf105]
#   %rsqrt_12 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_12]
#   %select_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 2), kwargs = {})
#   %mul_37 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, %add_24), kwargs = {})
#   %select_5 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 2), kwargs = {})
#   %mul_38 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %cat), kwargs = {})
#   %add_25 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %mul_38), kwargs = {})
#   %view_48 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_17, [16, 2048, 768]), kwargs = {})
#   %add_33 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_25, %view_48), kwargs = {})
#   %convert_element_type_79 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.float32), kwargs = {})
#   %pow_15 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_79, 2), kwargs = {})
#   %mean_12 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_15, [2], True), kwargs = {})
#   %add_34 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_12, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_12 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_34,), kwargs = {})
#   %mul_52 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_79, %rsqrt_12), kwargs = {})
#   %convert_element_type_80 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_52, torch.bfloat16), kwargs = {})
#   return %add_33,%buf105,%rsqrt_12,%convert_element_type_80
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_15 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_15(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (2))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/lc/clc6sebzn4hk4o47j7tyzfo2s2vu7t7kx5x5ziewgo3cdcceulzy.py
# Topologically Sorted Source Nodes: [x_20, x_21, getitem_25, mul_40, getitem_26, mul_41, x_22, rms_norm_13], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_25 => select_6
#   getitem_26 => select_7
#   mul_40 => mul_53
#   mul_41 => mul_54
#   rms_norm_13 => add_37, convert_element_type_87, convert_element_type_88, mean_13, mul_55, pow_17, rsqrt_13
#   x_20 => view_52
#   x_21 => add_35
#   x_22 => add_36
# Graph fragment:
#   %add_33 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_33]
#   %mm_19 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_19]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_35 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_35]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %buf115 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf115]
#   %rsqrt_13 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_13]
#   %view_52 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_19, [16, 2048, 768]), kwargs = {})
#   %add_35 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %view_52), kwargs = {})
#   %select_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 3), kwargs = {})
#   %mul_53 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_6, %add_35), kwargs = {})
#   %select_7 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 3), kwargs = {})
#   %mul_54 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_7, %cat), kwargs = {})
#   %add_36 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %mul_54), kwargs = {})
#   %convert_element_type_87 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_36, torch.float32), kwargs = {})
#   %pow_17 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_87, 2), kwargs = {})
#   %mean_13 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_17, [2], True), kwargs = {})
#   %add_37 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_13, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_13 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_37,), kwargs = {})
#   %mul_55 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_87, %rsqrt_13), kwargs = {})
#   %convert_element_type_88 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_55, torch.bfloat16), kwargs = {})
#   return %add_35,%buf115,%rsqrt_13,%convert_element_type_88
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_16 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_16', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_16(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (3))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (3))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/b2/cb2xqef5eczbpm6igzqvxplv7eo6n7lkw3befc7kvebrb623koaw.py
# Topologically Sorted Source Nodes: [getitem_25, mul_40, getitem_26, mul_41, x_22, y_11, x_23, rms_norm_16], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_25 => select_6
#   getitem_26 => select_7
#   mul_40 => mul_53
#   mul_41 => mul_54
#   rms_norm_16 => add_46, convert_element_type_108, convert_element_type_109, mean_16, mul_70, pow_20, rsqrt_16
#   x_22 => add_36
#   x_23 => add_45
#   y_11 => view_67
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_35 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_35]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %mm_24 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_24]
#   %add_45 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_45]
#   %buf143 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf143]
#   %rsqrt_16 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_16]
#   %select_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 3), kwargs = {})
#   %mul_53 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_6, %add_35), kwargs = {})
#   %select_7 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 3), kwargs = {})
#   %mul_54 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_7, %cat), kwargs = {})
#   %add_36 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %mul_54), kwargs = {})
#   %view_67 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_24, [16, 2048, 768]), kwargs = {})
#   %add_45 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_36, %view_67), kwargs = {})
#   %convert_element_type_108 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_45, torch.float32), kwargs = {})
#   %pow_20 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_108, 2), kwargs = {})
#   %mean_16 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_20, [2], True), kwargs = {})
#   %add_46 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_16, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_16 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_46,), kwargs = {})
#   %mul_70 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_108, %rsqrt_16), kwargs = {})
#   %convert_element_type_109 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_70, torch.bfloat16), kwargs = {})
#   return %add_45,%buf143,%rsqrt_16,%convert_element_type_109
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_17 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_17(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (3))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/eb/ceb4x6w2dfojodxvpfctovwjnrpzd6ortbtycvqvyf4a3zzjwzx4.py
# Topologically Sorted Source Nodes: [x_26, x_27, getitem_32, mul_54, getitem_33, mul_55, x_28, rms_norm_17], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_32 => select_8
#   getitem_33 => select_9
#   mul_54 => mul_71
#   mul_55 => mul_72
#   rms_norm_17 => add_49, convert_element_type_116, convert_element_type_117, mean_17, mul_73, pow_22, rsqrt_17
#   x_26 => view_71
#   x_27 => add_47
#   x_28 => add_48
# Graph fragment:
#   %add_45 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_45]
#   %mm_26 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_26]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_47 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_47]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %buf152 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf152]
#   %rsqrt_17 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_17]
#   %view_71 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_26, [16, 2048, 768]), kwargs = {})
#   %add_47 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %view_71), kwargs = {})
#   %select_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 4), kwargs = {})
#   %mul_71 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %add_47), kwargs = {})
#   %select_9 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 4), kwargs = {})
#   %mul_72 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, %cat), kwargs = {})
#   %add_48 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %mul_72), kwargs = {})
#   %convert_element_type_116 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_48, torch.float32), kwargs = {})
#   %pow_22 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_116, 2), kwargs = {})
#   %mean_17 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_22, [2], True), kwargs = {})
#   %add_49 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_17, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_17 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_49,), kwargs = {})
#   %mul_73 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_116, %rsqrt_17), kwargs = {})
#   %convert_element_type_117 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_73, torch.bfloat16), kwargs = {})
#   return %add_47,%buf152,%rsqrt_17,%convert_element_type_117
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_18 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_18', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_18(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (4))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (4))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/6w/c6wsfisvpejoeojuthoidjbd7h54w7dwmpey7bzs2cszbm33rlnj.py
# Topologically Sorted Source Nodes: [getitem_32, mul_54, getitem_33, mul_55, x_28, y_14, x_29, rms_norm_20], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_32 => select_8
#   getitem_33 => select_9
#   mul_54 => mul_71
#   mul_55 => mul_72
#   rms_norm_20 => add_57, convert_element_type_134, convert_element_type_135, mean_20, mul_86, pow_25, rsqrt_20
#   x_28 => add_48
#   x_29 => add_56
#   y_14 => view_83
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_47 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_47]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %mm_30 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_30]
#   %add_56 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_56]
#   %buf177 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf177]
#   %rsqrt_20 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_20]
#   %select_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 4), kwargs = {})
#   %mul_71 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %add_47), kwargs = {})
#   %select_9 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 4), kwargs = {})
#   %mul_72 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, %cat), kwargs = {})
#   %add_48 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %mul_72), kwargs = {})
#   %view_83 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_30, [16, 2048, 768]), kwargs = {})
#   %add_56 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_48, %view_83), kwargs = {})
#   %convert_element_type_134 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_56, torch.float32), kwargs = {})
#   %pow_25 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_134, 2), kwargs = {})
#   %mean_20 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_25, [2], True), kwargs = {})
#   %add_57 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_20, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_20 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_57,), kwargs = {})
#   %mul_86 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_134, %rsqrt_20), kwargs = {})
#   %convert_element_type_135 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_86, torch.bfloat16), kwargs = {})
#   return %add_56,%buf177,%rsqrt_20,%convert_element_type_135
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_19 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_19(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (4))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/x2/cx22vl7lf5wy7s53yycqyufbuiqd3su4gx3n3dq5mw4yisgg5jcj.py
# Topologically Sorted Source Nodes: [x_32, x_33, getitem_38, mul_66, getitem_39, mul_67, x_34, rms_norm_21], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_38 => select_10
#   getitem_39 => select_11
#   mul_66 => mul_87
#   mul_67 => mul_88
#   rms_norm_21 => add_60, convert_element_type_142, convert_element_type_143, mean_21, mul_89, pow_27, rsqrt_21
#   x_32 => view_87
#   x_33 => add_58
#   x_34 => add_59
# Graph fragment:
#   %add_56 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_56]
#   %mm_32 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_32]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_58 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_58]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %buf187 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf187]
#   %rsqrt_21 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_21]
#   %view_87 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_32, [16, 2048, 768]), kwargs = {})
#   %add_58 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_56, %view_87), kwargs = {})
#   %select_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 5), kwargs = {})
#   %mul_87 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_10, %add_58), kwargs = {})
#   %select_11 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 5), kwargs = {})
#   %mul_88 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_11, %cat), kwargs = {})
#   %add_59 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %mul_88), kwargs = {})
#   %convert_element_type_142 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_59, torch.float32), kwargs = {})
#   %pow_27 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_142, 2), kwargs = {})
#   %mean_21 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_27, [2], True), kwargs = {})
#   %add_60 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_21, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_21 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_60,), kwargs = {})
#   %mul_89 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_142, %rsqrt_21), kwargs = {})
#   %convert_element_type_143 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_89, torch.bfloat16), kwargs = {})
#   return %add_58,%buf187,%rsqrt_21,%convert_element_type_143
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_20 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_20', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_20(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (5))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (5))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/v5/cv5xuglri6weyq2jvjginjm5lzmt7qbgkqw2l2rbhzpwr54en5ef.py
# Topologically Sorted Source Nodes: [getitem_38, mul_66, getitem_39, mul_67, x_34, y_17, x_35, rms_norm_24], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_38 => select_10
#   getitem_39 => select_11
#   mul_66 => mul_87
#   mul_67 => mul_88
#   rms_norm_24 => add_69, convert_element_type_163, convert_element_type_164, mean_24, mul_104, pow_30, rsqrt_24
#   x_34 => add_59
#   x_35 => add_68
#   y_17 => view_102
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_58 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_58]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %mm_37 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_37]
#   %add_68 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_68]
#   %buf215 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf215]
#   %rsqrt_24 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_24]
#   %select_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 5), kwargs = {})
#   %mul_87 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_10, %add_58), kwargs = {})
#   %select_11 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 5), kwargs = {})
#   %mul_88 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_11, %cat), kwargs = {})
#   %add_59 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %mul_88), kwargs = {})
#   %view_102 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_37, [16, 2048, 768]), kwargs = {})
#   %add_68 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_59, %view_102), kwargs = {})
#   %convert_element_type_163 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_68, torch.float32), kwargs = {})
#   %pow_30 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_163, 2), kwargs = {})
#   %mean_24 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_30, [2], True), kwargs = {})
#   %add_69 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_24, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_24 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_69,), kwargs = {})
#   %mul_104 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_163, %rsqrt_24), kwargs = {})
#   %convert_element_type_164 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_104, torch.bfloat16), kwargs = {})
#   return %add_68,%buf215,%rsqrt_24,%convert_element_type_164
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_21 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_21', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_21(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (5))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/rk/crkdekvbsdp6vc4bkht546mgadr2dgaynggh2hbzi5deb26x2ao2.py
# Topologically Sorted Source Nodes: [x_38, x_39, getitem_45, mul_80, getitem_46, mul_81, x_40, rms_norm_25], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_45 => select_12
#   getitem_46 => select_13
#   mul_80 => mul_105
#   mul_81 => mul_106
#   rms_norm_25 => add_72, convert_element_type_171, convert_element_type_172, mean_25, mul_107, pow_32, rsqrt_25
#   x_38 => view_106
#   x_39 => add_70
#   x_40 => add_71
# Graph fragment:
#   %add_68 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_68]
#   %mm_39 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_39]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_70 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_70]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %buf224 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf224]
#   %rsqrt_25 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_25]
#   %view_106 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_39, [16, 2048, 768]), kwargs = {})
#   %add_70 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_68, %view_106), kwargs = {})
#   %select_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 6), kwargs = {})
#   %mul_105 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_12, %add_70), kwargs = {})
#   %select_13 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 6), kwargs = {})
#   %mul_106 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_13, %cat), kwargs = {})
#   %add_71 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_105, %mul_106), kwargs = {})
#   %convert_element_type_171 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_71, torch.float32), kwargs = {})
#   %pow_32 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_171, 2), kwargs = {})
#   %mean_25 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_32, [2], True), kwargs = {})
#   %add_72 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_25, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_25 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_72,), kwargs = {})
#   %mul_107 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_171, %rsqrt_25), kwargs = {})
#   %convert_element_type_172 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_107, torch.bfloat16), kwargs = {})
#   return %add_70,%buf224,%rsqrt_25,%convert_element_type_172
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_22 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_22(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (6))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (6))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/zi/czinoeuylkycmebtwnf7lqdxli6skfuj4z4tlgr32oaouylb44wb.py
# Topologically Sorted Source Nodes: [getitem_45, mul_80, getitem_46, mul_81, x_40, y_20, x_41, rms_norm_28], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_45 => select_12
#   getitem_46 => select_13
#   mul_80 => mul_105
#   mul_81 => mul_106
#   rms_norm_28 => add_80, convert_element_type_189, convert_element_type_190, mean_28, mul_120, pow_35, rsqrt_28
#   x_40 => add_71
#   x_41 => add_79
#   y_20 => view_118
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_70 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_70]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %mm_43 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_43]
#   %add_79 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_79]
#   %buf249 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf249]
#   %rsqrt_28 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_28]
#   %select_12 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 6), kwargs = {})
#   %mul_105 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_12, %add_70), kwargs = {})
#   %select_13 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 6), kwargs = {})
#   %mul_106 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_13, %cat), kwargs = {})
#   %add_71 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_105, %mul_106), kwargs = {})
#   %view_118 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_43, [16, 2048, 768]), kwargs = {})
#   %add_79 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_71, %view_118), kwargs = {})
#   %convert_element_type_189 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_79, torch.float32), kwargs = {})
#   %pow_35 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_189, 2), kwargs = {})
#   %mean_28 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_35, [2], True), kwargs = {})
#   %add_80 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_28, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_28 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_80,), kwargs = {})
#   %mul_120 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_189, %rsqrt_28), kwargs = {})
#   %convert_element_type_190 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_120, torch.bfloat16), kwargs = {})
#   return %add_79,%buf249,%rsqrt_28,%convert_element_type_190
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_23 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_23', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_23(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (6))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (6))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/b2/cb2e7nbkgvhp3hoeyg37fzempsqei5wcf4rfqwixo5eruts76rwd.py
# Topologically Sorted Source Nodes: [x_44, x_45, getitem_51, mul_92, getitem_52, mul_93, x_46, rms_norm_29], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_51 => select_14
#   getitem_52 => select_15
#   mul_92 => mul_121
#   mul_93 => mul_122
#   rms_norm_29 => add_83, convert_element_type_197, convert_element_type_198, mean_29, mul_123, pow_37, rsqrt_29
#   x_44 => view_122
#   x_45 => add_81
#   x_46 => add_82
# Graph fragment:
#   %add_79 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_79]
#   %mm_45 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_45]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_81 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_81]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %buf259 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf259]
#   %rsqrt_29 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_29]
#   %view_122 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_45, [16, 2048, 768]), kwargs = {})
#   %add_81 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_79, %view_122), kwargs = {})
#   %select_14 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 7), kwargs = {})
#   %mul_121 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_14, %add_81), kwargs = {})
#   %select_15 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 7), kwargs = {})
#   %mul_122 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_15, %cat), kwargs = {})
#   %add_82 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_121, %mul_122), kwargs = {})
#   %convert_element_type_197 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_82, torch.float32), kwargs = {})
#   %pow_37 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_197, 2), kwargs = {})
#   %mean_29 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_37, [2], True), kwargs = {})
#   %add_83 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_29, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_29 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_83,), kwargs = {})
#   %mul_123 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_197, %rsqrt_29), kwargs = {})
#   %convert_element_type_198 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_123, torch.bfloat16), kwargs = {})
#   return %add_81,%buf259,%rsqrt_29,%convert_element_type_198
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_24 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_24(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (7))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (7))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/3v/c3v7ttfhaehigi7rbdtbvhwqtxx4h3hdh36jaedijatl3jlyblyl.py
# Topologically Sorted Source Nodes: [getitem_51, mul_92, getitem_52, mul_93, x_46, y_23, x_47, rms_norm_32], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_51 => select_14
#   getitem_52 => select_15
#   mul_92 => mul_121
#   mul_93 => mul_122
#   rms_norm_32 => add_92, convert_element_type_218, convert_element_type_219, mean_32, mul_138, pow_40, rsqrt_32
#   x_46 => add_82
#   x_47 => add_91
#   y_23 => view_137
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_81 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_81]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %mm_50 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_50]
#   %add_91 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_91]
#   %buf287 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf287]
#   %rsqrt_32 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_32]
#   %select_14 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 7), kwargs = {})
#   %mul_121 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_14, %add_81), kwargs = {})
#   %select_15 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 7), kwargs = {})
#   %mul_122 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_15, %cat), kwargs = {})
#   %add_82 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_121, %mul_122), kwargs = {})
#   %view_137 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_50, [16, 2048, 768]), kwargs = {})
#   %add_91 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_82, %view_137), kwargs = {})
#   %convert_element_type_218 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_91, torch.float32), kwargs = {})
#   %pow_40 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_218, 2), kwargs = {})
#   %mean_32 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_40, [2], True), kwargs = {})
#   %add_92 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_32, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_32 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_92,), kwargs = {})
#   %mul_138 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_218, %rsqrt_32), kwargs = {})
#   %convert_element_type_219 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_138, torch.bfloat16), kwargs = {})
#   return %add_91,%buf287,%rsqrt_32,%convert_element_type_219
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_25 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_25', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_25(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (7))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/sh/cshl26upyd7vwimo2c77nl6pqhpgodnb4oe6zvwecqiephbrojzw.py
# Topologically Sorted Source Nodes: [x_50, x_51, getitem_58, mul_106, getitem_59, mul_107, x_52, rms_norm_33], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_58 => select_16
#   getitem_59 => select_17
#   mul_106 => mul_139
#   mul_107 => mul_140
#   rms_norm_33 => add_95, convert_element_type_226, convert_element_type_227, mean_33, mul_141, pow_42, rsqrt_33
#   x_50 => view_141
#   x_51 => add_93
#   x_52 => add_94
# Graph fragment:
#   %add_91 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_91]
#   %mm_52 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_52]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_93 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_93]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %buf296 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf296]
#   %rsqrt_33 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_33]
#   %view_141 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_52, [16, 2048, 768]), kwargs = {})
#   %add_93 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, %view_141), kwargs = {})
#   %select_16 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 8), kwargs = {})
#   %mul_139 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_16, %add_93), kwargs = {})
#   %select_17 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 8), kwargs = {})
#   %mul_140 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_17, %cat), kwargs = {})
#   %add_94 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_139, %mul_140), kwargs = {})
#   %convert_element_type_226 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_94, torch.float32), kwargs = {})
#   %pow_42 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_226, 2), kwargs = {})
#   %mean_33 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_42, [2], True), kwargs = {})
#   %add_95 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_33, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_33 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_95,), kwargs = {})
#   %mul_141 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_226, %rsqrt_33), kwargs = {})
#   %convert_element_type_227 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_141, torch.bfloat16), kwargs = {})
#   return %add_93,%buf296,%rsqrt_33,%convert_element_type_227
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_26 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_26', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_26(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (8))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (8))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/24/c243lat5hk5p57lb3innachlkraswczynp3ql6mw6ddbegofj7b2.py
# Topologically Sorted Source Nodes: [getitem_58, mul_106, getitem_59, mul_107, x_52, y_26, x_53, rms_norm_36], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_58 => select_16
#   getitem_59 => select_17
#   mul_106 => mul_139
#   mul_107 => mul_140
#   rms_norm_36 => add_103, convert_element_type_244, convert_element_type_245, mean_36, mul_154, pow_45, rsqrt_36
#   x_52 => add_94
#   x_53 => add_102
#   y_26 => view_153
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_93 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_93]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %mm_56 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_56]
#   %add_102 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_102]
#   %buf321 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf321]
#   %rsqrt_36 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_36]
#   %select_16 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 8), kwargs = {})
#   %mul_139 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_16, %add_93), kwargs = {})
#   %select_17 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 8), kwargs = {})
#   %mul_140 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_17, %cat), kwargs = {})
#   %add_94 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_139, %mul_140), kwargs = {})
#   %view_153 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_56, [16, 2048, 768]), kwargs = {})
#   %add_102 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, %view_153), kwargs = {})
#   %convert_element_type_244 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_102, torch.float32), kwargs = {})
#   %pow_45 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_244, 2), kwargs = {})
#   %mean_36 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_45, [2], True), kwargs = {})
#   %add_103 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_36, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_36 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_103,), kwargs = {})
#   %mul_154 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_244, %rsqrt_36), kwargs = {})
#   %convert_element_type_245 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_154, torch.bfloat16), kwargs = {})
#   return %add_102,%buf321,%rsqrt_36,%convert_element_type_245
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_27 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_27', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_27(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (8))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (8))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/7u/c7ujnyf2cxthtomdaojjpwyfe5moixezaadp5ncn6to6uykjytep.py
# Topologically Sorted Source Nodes: [x_56, x_57, getitem_64, mul_118, getitem_65, mul_119, x_58, rms_norm_37], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_64 => select_18
#   getitem_65 => select_19
#   mul_118 => mul_155
#   mul_119 => mul_156
#   rms_norm_37 => add_106, convert_element_type_252, convert_element_type_253, mean_37, mul_157, pow_47, rsqrt_37
#   x_56 => view_157
#   x_57 => add_104
#   x_58 => add_105
# Graph fragment:
#   %add_102 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_102]
#   %mm_58 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_58]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_104 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_104]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %buf331 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf331]
#   %rsqrt_37 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_37]
#   %view_157 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_58, [16, 2048, 768]), kwargs = {})
#   %add_104 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_102, %view_157), kwargs = {})
#   %select_18 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 9), kwargs = {})
#   %mul_155 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_18, %add_104), kwargs = {})
#   %select_19 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 9), kwargs = {})
#   %mul_156 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_19, %cat), kwargs = {})
#   %add_105 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %mul_156), kwargs = {})
#   %convert_element_type_252 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_105, torch.float32), kwargs = {})
#   %pow_47 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_252, 2), kwargs = {})
#   %mean_37 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_47, [2], True), kwargs = {})
#   %add_106 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_37, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_37 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_106,), kwargs = {})
#   %mul_157 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_252, %rsqrt_37), kwargs = {})
#   %convert_element_type_253 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_157, torch.bfloat16), kwargs = {})
#   return %add_104,%buf331,%rsqrt_37,%convert_element_type_253
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_28 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_28', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_28(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (9))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (9))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/nz/cnzydw5ca72y6uoanskrnuiiplhkkeu2342cu2srgprpsfzxukag.py
# Topologically Sorted Source Nodes: [getitem_64, mul_118, getitem_65, mul_119, x_58, y_29, x_59, rms_norm_40], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_64 => select_18
#   getitem_65 => select_19
#   mul_118 => mul_155
#   mul_119 => mul_156
#   rms_norm_40 => add_115, convert_element_type_273, convert_element_type_274, mean_40, mul_172, pow_50, rsqrt_40
#   x_58 => add_105
#   x_59 => add_114
#   y_29 => view_172
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_104 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_104]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %mm_63 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_63]
#   %add_114 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_114]
#   %buf359 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf359]
#   %rsqrt_40 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_40]
#   %select_18 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 9), kwargs = {})
#   %mul_155 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_18, %add_104), kwargs = {})
#   %select_19 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 9), kwargs = {})
#   %mul_156 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_19, %cat), kwargs = {})
#   %add_105 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %mul_156), kwargs = {})
#   %view_172 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_63, [16, 2048, 768]), kwargs = {})
#   %add_114 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_105, %view_172), kwargs = {})
#   %convert_element_type_273 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_114, torch.float32), kwargs = {})
#   %pow_50 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_273, 2), kwargs = {})
#   %mean_40 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_50, [2], True), kwargs = {})
#   %add_115 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_40, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_40 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_115,), kwargs = {})
#   %mul_172 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_273, %rsqrt_40), kwargs = {})
#   %convert_element_type_274 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_172, torch.bfloat16), kwargs = {})
#   return %add_114,%buf359,%rsqrt_40,%convert_element_type_274
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_29 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_29', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_29(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (9))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (9))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/m3/cm3n5grjfiu6cvf7is6bvogxau5unoxzyddto3cwfzgosnkcsurs.py
# Topologically Sorted Source Nodes: [x_62, x_63, getitem_71, mul_132, getitem_72, mul_133, x_64, rms_norm_41], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_71 => select_20
#   getitem_72 => select_21
#   mul_132 => mul_173
#   mul_133 => mul_174
#   rms_norm_41 => add_118, convert_element_type_281, convert_element_type_282, mean_41, mul_175, pow_52, rsqrt_41
#   x_62 => view_176
#   x_63 => add_116
#   x_64 => add_117
# Graph fragment:
#   %add_114 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_114]
#   %mm_65 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_65]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_116 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_116]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %buf368 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf368]
#   %rsqrt_41 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_41]
#   %view_176 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_65, [16, 2048, 768]), kwargs = {})
#   %add_116 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_114, %view_176), kwargs = {})
#   %select_20 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 10), kwargs = {})
#   %mul_173 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_20, %add_116), kwargs = {})
#   %select_21 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 10), kwargs = {})
#   %mul_174 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_21, %cat), kwargs = {})
#   %add_117 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %mul_174), kwargs = {})
#   %convert_element_type_281 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_117, torch.float32), kwargs = {})
#   %pow_52 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_281, 2), kwargs = {})
#   %mean_41 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_52, [2], True), kwargs = {})
#   %add_118 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_41, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_41 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_118,), kwargs = {})
#   %mul_175 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_281, %rsqrt_41), kwargs = {})
#   %convert_element_type_282 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_175, torch.bfloat16), kwargs = {})
#   return %add_116,%buf368,%rsqrt_41,%convert_element_type_282
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_30 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_30', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_30(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (10))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (10))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/7z/c7znbwjnbqxbnly4idy6gpmvdpa7255rehc7upvwt2bsg6uqhulp.py
# Topologically Sorted Source Nodes: [getitem_71, mul_132, getitem_72, mul_133, x_64, y_32, x_65, rms_norm_44], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_71 => select_20
#   getitem_72 => select_21
#   mul_132 => mul_173
#   mul_133 => mul_174
#   rms_norm_44 => add_126, convert_element_type_299, convert_element_type_300, mean_44, mul_188, pow_55, rsqrt_44
#   x_64 => add_117
#   x_65 => add_125
#   y_32 => view_188
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_116 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_116]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %mm_69 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_69]
#   %add_125 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_125]
#   %buf393 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf393]
#   %rsqrt_44 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_44]
#   %select_20 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 10), kwargs = {})
#   %mul_173 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_20, %add_116), kwargs = {})
#   %select_21 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 10), kwargs = {})
#   %mul_174 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_21, %cat), kwargs = {})
#   %add_117 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %mul_174), kwargs = {})
#   %view_188 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_69, [16, 2048, 768]), kwargs = {})
#   %add_125 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_117, %view_188), kwargs = {})
#   %convert_element_type_299 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_125, torch.float32), kwargs = {})
#   %pow_55 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_299, 2), kwargs = {})
#   %mean_44 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_55, [2], True), kwargs = {})
#   %add_126 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_44, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_44 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_126,), kwargs = {})
#   %mul_188 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_299, %rsqrt_44), kwargs = {})
#   %convert_element_type_300 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_188, torch.bfloat16), kwargs = {})
#   return %add_125,%buf393,%rsqrt_44,%convert_element_type_300
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_31 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_31(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (10))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (10))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/3u/c3u6gog3yo3uiz7e2jbkbged7qgzpgmbgkwoupdomftnjqxctmtb.py
# Topologically Sorted Source Nodes: [x_68, x_69, getitem_77, mul_144, getitem_78, mul_145, x_70, rms_norm_45], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_77 => select_22
#   getitem_78 => select_23
#   mul_144 => mul_189
#   mul_145 => mul_190
#   rms_norm_45 => add_129, convert_element_type_307, convert_element_type_308, mean_45, mul_191, pow_57, rsqrt_45
#   x_68 => view_192
#   x_69 => add_127
#   x_70 => add_128
# Graph fragment:
#   %add_125 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_125]
#   %mm_71 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_71]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_127 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_127]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %buf403 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf403]
#   %rsqrt_45 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_45]
#   %view_192 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_71, [16, 2048, 768]), kwargs = {})
#   %add_127 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_125, %view_192), kwargs = {})
#   %select_22 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 11), kwargs = {})
#   %mul_189 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_22, %add_127), kwargs = {})
#   %select_23 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 11), kwargs = {})
#   %mul_190 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_23, %cat), kwargs = {})
#   %add_128 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_189, %mul_190), kwargs = {})
#   %convert_element_type_307 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_128, torch.float32), kwargs = {})
#   %pow_57 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_307, 2), kwargs = {})
#   %mean_45 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_57, [2], True), kwargs = {})
#   %add_129 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_45, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_45 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_129,), kwargs = {})
#   %mul_191 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_307, %rsqrt_45), kwargs = {})
#   %convert_element_type_308 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_191, torch.bfloat16), kwargs = {})
#   return %add_127,%buf403,%rsqrt_45,%convert_element_type_308
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_32 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_32', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_32(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (11))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.load(in_ptr2 + (11))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp2, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/f4/cf4mklyzpp7jluvh4k4k2hnuyipasoolo4c3o7vmjwkqnrs7bh7i.py
# Topologically Sorted Source Nodes: [getitem_77, mul_144, getitem_78, mul_145, x_70, y_35, x_71, rms_norm_48], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   getitem_77 => select_22
#   getitem_78 => select_23
#   mul_144 => mul_189
#   mul_145 => mul_190
#   rms_norm_48 => add_138, convert_element_type_328, convert_element_type_329, mean_48, mul_206, pow_60, rsqrt_48
#   x_70 => add_128
#   x_71 => add_137
#   y_35 => view_207
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_127 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_127]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %mm_76 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_76]
#   %add_137 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_137]
#   %buf431 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf431]
#   %rsqrt_48 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_48]
#   %select_22 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 11), kwargs = {})
#   %mul_189 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_22, %add_127), kwargs = {})
#   %select_23 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 11), kwargs = {})
#   %mul_190 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_23, %cat), kwargs = {})
#   %add_128 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_189, %mul_190), kwargs = {})
#   %view_207 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_76, [16, 2048, 768]), kwargs = {})
#   %add_137 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_128, %view_207), kwargs = {})
#   %convert_element_type_328 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_137, torch.float32), kwargs = {})
#   %pow_60 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_328, 2), kwargs = {})
#   %mean_48 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_60, [2], True), kwargs = {})
#   %add_138 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_48, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_48 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_138,), kwargs = {})
#   %mul_206 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_328, %rsqrt_48), kwargs = {})
#   %convert_element_type_329 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_206, torch.bfloat16), kwargs = {})
#   return %add_137,%buf431,%rsqrt_48,%convert_element_type_329
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_33 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_33', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_33(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (11))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (11))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 768.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1.1920928955078125e-07
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/pg/cpgiogia6m4w3tobrzm3dtfcglq6cpt2ugf7k2ep22mxei7fp7ms.py
# Topologically Sorted Source Nodes: [x_74, x_75, to_87, mul_158, x_76, x_77], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.mul, aten.sub, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   mul_158 => mul_207
#   to_87 => convert_element_type_336
#   x_74 => view_211
#   x_75 => add_139
#   x_76 => sub
#   x_77 => add_140, convert_element_type_337, convert_element_type_338, mean_49, mul_208, pow_62, rsqrt_49
# Graph fragment:
#   %add_137 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_137]
#   %mm_78 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_78]
#   %primals_93 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_93]
#   %add_81 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_81]
#   %sub : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=sub]
#   %buf440 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf440]
#   %rsqrt_49 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_49]
#   %view_211 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_78, [16, 2048, 768]), kwargs = {})
#   %add_139 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_137, %view_211), kwargs = {})
#   %convert_element_type_336 : Tensor "bf16[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_93, torch.bfloat16), kwargs = {})
#   %mul_207 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_336, %add_81), kwargs = {})
#   %sub : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_139, %mul_207), kwargs = {})
#   %convert_element_type_337 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.float32), kwargs = {})
#   %pow_62 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_337, 2), kwargs = {})
#   %mean_49 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_62, [2], True), kwargs = {})
#   %add_140 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_49, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_49 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_140,), kwargs = {})
#   %mul_208 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_337, %rsqrt_49), kwargs = {})
#   %convert_element_type_338 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_208, torch.bfloat16), kwargs = {})
#   return %sub,%buf440,%rsqrt_49,%convert_element_type_338
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_sub_34 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_sub_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_sub_34', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 352321536}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_sub_34(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp15 = 768.0
    tmp16 = (tmp14 / tmp15)
    tmp17 = 1.1920928955078125e-07
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp9 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp8, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp19, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp21, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/jo/cjoh2gp3ig5fp2t52yuymrkxbc4tw6qhxgir4qwfmjzxxfp76orr.py
# Topologically Sorted Source Nodes: [to_88, logits], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   logits => permute_79
#   to_88 => convert_element_type_339
# Graph fragment:
#   %primals_94 : Tensor "f32[32768, 768][768, 1]cuda:0" = PlaceHolder[target=primals_94]
#   %convert_element_type_339 : Tensor "bf16[32768, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_94, torch.bfloat16), kwargs = {})
#   %permute_79 : Tensor "bf16[768, 32768][1, 768]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_339, [1, 0]), kwargs = {})
#   return %permute_79
triton_poi_fused__to_copy_t_35 = async_compile.triton('triton_poi_fused__to_copy_t_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201326592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/s2/cs2nruia55wjqgvuqoylwc6i2sqsio6m6dx2qszwnqgn46rr656a.py
# Topologically Sorted Source Nodes: [logits, logits_2, truediv, tanh, logits_3, view_54, , loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.view, prims.prepare_softmax_online, aten._log_softmax]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default
#   logits => view_213
#   logits_2 => convert_element_type_342
#   logits_3 => mul_209
#   loss => log
#   tanh => tanh
#   truediv => div
#   view_54 => view_214
# Graph fragment:
#   %mm_79 : Tensor "bf16[32768, 32768][32768, 1]cuda:0" = PlaceHolder[target=mm_79]
#   %getitem_47 : Tensor "f32[32768, 1][1, 32768]cuda:0" = PlaceHolder[target=getitem_47]
#   %view_213 : Tensor "bf16[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_79, [16, 2048, 32768]), kwargs = {})
#   %convert_element_type_342 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_213, torch.float32), kwargs = {})
#   %div : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_342, 15), kwargs = {})
#   %tanh : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_209 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 15), kwargs = {})
#   %view_214 : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_209, [-1, 32768]), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%view_214, 1), kwargs = {})
#   %log : Tensor "f32[32768, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%getitem_47,), kwargs = {})
#   return %getitem_46,%getitem_47,%log
triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view_36 = async_compile.triton('triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288, 'r0_': 2147483648}}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view_36(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 32768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp8_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 32768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = 0.06666666666666667
        tmp3 = tmp1 * tmp2
        tmp4 = libdevice.tanh(tmp3)
        tmp5 = 15.0
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])

        _tmp8_max_next, _tmp8_sum_next = triton_helpers.online_softmax_combine(
            _tmp8_max, _tmp8_sum, tmp7, False
        )

        _tmp8_max = tl.where(r0_mask, _tmp8_max_next, _tmp8_max)
        _tmp8_sum = tl.where(r0_mask, _tmp8_sum_next, _tmp8_sum)

    tmp8, tmp9 = triton_helpers.online_softmax_reduce(
        _tmp8_max, _tmp8_sum, 1, False)
    tmp8 = tmp8[:, None]
    tmp9 = tmp9[:, None]
    tl.store(out_ptr0 + (x0), tmp8, None)
    tmp10 = tl_math.log(tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/fg/cfg2jno4q2ax4xcanz3opupxd7gace7sxtwv767xnk3jktralcex.py
# Topologically Sorted Source Nodes: [logits, logits_2, truediv, tanh, logits_3, view_54, view_55, , loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.view, aten.sub, aten._log_softmax, aten.nll_loss_forward]
# Source node to ATen node mapping:
#    => sub_tensor
#   logits => view_213
#   logits_2 => convert_element_type_342
#   logits_3 => mul_209
#   loss => full_default, full_default_1, gather, ne, neg_24, squeeze, sub_2, sum_2, sum_3, unsqueeze_6, where, where_1
#   tanh => tanh
#   truediv => div
#   view_54 => view_214
#   view_55 => view_215
# Graph fragment:
#   %primals_95 : Tensor "i64[16, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_95]
#   %mm_79 : Tensor "bf16[32768, 32768][32768, 1]cuda:0" = PlaceHolder[target=mm_79]
#   %getitem_46 : Tensor "f32[32768, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_46]
#   %log : Tensor "f32[32768, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %view_213 : Tensor "bf16[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_79, [16, 2048, 32768]), kwargs = {})
#   %convert_element_type_342 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_213, torch.float32), kwargs = {})
#   %div : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_342, 15), kwargs = {})
#   %tanh : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_209 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 15), kwargs = {})
#   %view_214 : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_209, [-1, 32768]), kwargs = {})
#   %view_215 : Tensor "i64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_95, [-1]), kwargs = {})
#   %sub_tensor : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_214, %getitem_46), kwargs = {})
#   %sub_2 : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log), kwargs = {})
#   %ne : Tensor "b8[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_215, -1), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "i64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %view_215, %full_default), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 1), kwargs = {})
#   %gather : Tensor "f32[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_2, 1, %unsqueeze_6), kwargs = {})
#   %squeeze : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg_24 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg_24, %full_default_1), kwargs = {})
#   %sum_2 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne,), kwargs = {})
#   %sum_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   return %buf449,%buf453
triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_37 = async_compile.triton('triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    _tmp29 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr2 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr3 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], -1, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tmp2.to(tl.int64)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
        tmp7 = tl.full([1, 1], 0, tl.int64)
        tmp8 = tl.where(tmp2, tmp0, tmp7)
        tmp9 = tl.full([XBLOCK, R0_BLOCK], 32768, tl.int32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp8 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp8)
        tl.device_assert(((0 <= tmp12) & (tmp12 < 32768)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp12 < 32768")
        tmp14 = tl.load(in_ptr1 + (tmp12 + 32768*r0_1 + 268435456*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = 0.06666666666666667
        tmp17 = tmp15 * tmp16
        tmp18 = libdevice.tanh(tmp17)
        tmp19 = 15.0
        tmp20 = tmp18 * tmp19
        tmp22 = tmp20 - tmp21
        tmp24 = tmp22 - tmp23
        tmp25 = -tmp24
        tmp26 = 0.0
        tmp27 = tl.where(tmp2, tmp25, tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(r0_mask & xmask, tmp30, _tmp29)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp29, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmp0as0rz9w/hb/chbmisk4skc7gsjkrffm7l53kjrnyej2225qqvhqs7vcjtqxohzw.py
# Topologically Sorted Source Nodes: [logits, logits_2, truediv, tanh, logits_3, view_54, view_55, , loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.view, aten.sub, aten._log_softmax, aten.nll_loss_forward]
# Source node to ATen node mapping:
#    => sub_tensor
#   logits => view_213
#   logits_2 => convert_element_type_342
#   logits_3 => mul_209
#   loss => convert_element_type_343, div_1, full_default, full_default_1, gather, ne, neg_24, squeeze, sub_2, sum_2, sum_3, unsqueeze_6, where, where_1
#   tanh => tanh
#   truediv => div
#   view_54 => view_214
#   view_55 => view_215
# Graph fragment:
#   %buf453 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=buf453]
#   %buf449 : Tensor "i64[4][1]cuda:0" = PlaceHolder[target=buf449]
#   %sum_2 : Tensor "i64[][]cuda:0" = PlaceHolder[target=sum_2]
#   %sum_3 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_3]
#   %convert_element_type_343 : Tensor "f32[][]cuda:0" = PlaceHolder[target=convert_element_type_343]
#   %view_213 : Tensor "bf16[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_79, [16, 2048, 32768]), kwargs = {})
#   %convert_element_type_342 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_213, torch.float32), kwargs = {})
#   %div : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_342, 15), kwargs = {})
#   %tanh : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_209 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 15), kwargs = {})
#   %view_214 : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_209, [-1, 32768]), kwargs = {})
#   %view_215 : Tensor "i64[32768][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_95, [-1]), kwargs = {})
#   %sub_tensor : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_214, %getitem_46), kwargs = {})
#   %sub_2 : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log), kwargs = {})
#   %ne : Tensor "b8[32768][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_215, -1), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "i64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %view_215, %full_default), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 1), kwargs = {})
#   %gather : Tensor "f32[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_2, 1, %unsqueeze_6), kwargs = {})
#   %squeeze : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg_24 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f32[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg_24, %full_default_1), kwargs = {})
#   %sum_2 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne,), kwargs = {})
#   %convert_element_type_343 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.float32), kwargs = {})
#   %sum_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %div_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %convert_element_type_343), kwargs = {})
#   return %sum_3,%sum_2,%convert_element_type_343,%div_1
triton_per_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_38 = async_compile.triton('triton_per_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 24}}
)
@triton.jit
def triton_per_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_38(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp4 = tl.load(in_ptr1 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None].to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.int64)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = (tmp3 / tmp8)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp9, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95 = args
        args.clear()
        assert_size_stride(primals_1, (16, 2048), (2048, 1))
        assert_size_stride(primals_2, (1, 20480, 1, 64), (1310720, 64, 64, 1))
        assert_size_stride(primals_3, (1, 20480, 1, 64), (1310720, 64, 64, 1))
        assert_size_stride(primals_4, (32768, 768), (768, 1))
        assert_size_stride(primals_5, (1, ), (1, ))
        assert_size_stride(primals_6, (1, 24), (24, 1))
        assert_size_stride(primals_7, (12, ), (1, ))
        assert_size_stride(primals_8, (12, ), (1, ))
        assert_size_stride(primals_9, (768, 768), (768, 1))
        assert_size_stride(primals_10, (768, 768), (768, 1))
        assert_size_stride(primals_11, (768, 768), (768, 1))
        assert_size_stride(primals_12, (768, 768), (768, 1))
        assert_size_stride(primals_13, (3072, 768), (768, 1))
        assert_size_stride(primals_14, (768, 3072), (3072, 1))
        assert_size_stride(primals_15, (32768, 768), (768, 1))
        assert_size_stride(primals_16, (768, 768), (768, 1))
        assert_size_stride(primals_17, (768, 768), (768, 1))
        assert_size_stride(primals_18, (768, 768), (768, 1))
        assert_size_stride(primals_19, (6, 12), (12, 1))
        assert_size_stride(primals_20, (768, 768), (768, 1))
        assert_size_stride(primals_21, (3072, 768), (768, 1))
        assert_size_stride(primals_22, (768, 3072), (3072, 1))
        assert_size_stride(primals_23, (768, 768), (768, 1))
        assert_size_stride(primals_24, (768, 768), (768, 1))
        assert_size_stride(primals_25, (768, 768), (768, 1))
        assert_size_stride(primals_26, (768, 768), (768, 1))
        assert_size_stride(primals_27, (3072, 768), (768, 1))
        assert_size_stride(primals_28, (768, 3072), (3072, 1))
        assert_size_stride(primals_29, (32768, 768), (768, 1))
        assert_size_stride(primals_30, (768, 768), (768, 1))
        assert_size_stride(primals_31, (768, 768), (768, 1))
        assert_size_stride(primals_32, (768, 768), (768, 1))
        assert_size_stride(primals_33, (6, 12), (12, 1))
        assert_size_stride(primals_34, (768, 768), (768, 1))
        assert_size_stride(primals_35, (3072, 768), (768, 1))
        assert_size_stride(primals_36, (768, 3072), (3072, 1))
        assert_size_stride(primals_37, (768, 768), (768, 1))
        assert_size_stride(primals_38, (768, 768), (768, 1))
        assert_size_stride(primals_39, (768, 768), (768, 1))
        assert_size_stride(primals_40, (768, 768), (768, 1))
        assert_size_stride(primals_41, (3072, 768), (768, 1))
        assert_size_stride(primals_42, (768, 3072), (3072, 1))
        assert_size_stride(primals_43, (32768, 768), (768, 1))
        assert_size_stride(primals_44, (768, 768), (768, 1))
        assert_size_stride(primals_45, (768, 768), (768, 1))
        assert_size_stride(primals_46, (768, 768), (768, 1))
        assert_size_stride(primals_47, (6, 12), (12, 1))
        assert_size_stride(primals_48, (768, 768), (768, 1))
        assert_size_stride(primals_49, (3072, 768), (768, 1))
        assert_size_stride(primals_50, (768, 3072), (3072, 1))
        assert_size_stride(primals_51, (768, 768), (768, 1))
        assert_size_stride(primals_52, (768, 768), (768, 1))
        assert_size_stride(primals_53, (768, 768), (768, 1))
        assert_size_stride(primals_54, (768, 768), (768, 1))
        assert_size_stride(primals_55, (3072, 768), (768, 1))
        assert_size_stride(primals_56, (768, 3072), (3072, 1))
        assert_size_stride(primals_57, (32768, 768), (768, 1))
        assert_size_stride(primals_58, (768, 768), (768, 1))
        assert_size_stride(primals_59, (768, 768), (768, 1))
        assert_size_stride(primals_60, (768, 768), (768, 1))
        assert_size_stride(primals_61, (6, 12), (12, 1))
        assert_size_stride(primals_62, (768, 768), (768, 1))
        assert_size_stride(primals_63, (3072, 768), (768, 1))
        assert_size_stride(primals_64, (768, 3072), (3072, 1))
        assert_size_stride(primals_65, (768, 768), (768, 1))
        assert_size_stride(primals_66, (768, 768), (768, 1))
        assert_size_stride(primals_67, (768, 768), (768, 1))
        assert_size_stride(primals_68, (768, 768), (768, 1))
        assert_size_stride(primals_69, (3072, 768), (768, 1))
        assert_size_stride(primals_70, (768, 3072), (3072, 1))
        assert_size_stride(primals_71, (32768, 768), (768, 1))
        assert_size_stride(primals_72, (768, 768), (768, 1))
        assert_size_stride(primals_73, (768, 768), (768, 1))
        assert_size_stride(primals_74, (768, 768), (768, 1))
        assert_size_stride(primals_75, (6, 12), (12, 1))
        assert_size_stride(primals_76, (768, 768), (768, 1))
        assert_size_stride(primals_77, (3072, 768), (768, 1))
        assert_size_stride(primals_78, (768, 3072), (3072, 1))
        assert_size_stride(primals_79, (768, 768), (768, 1))
        assert_size_stride(primals_80, (768, 768), (768, 1))
        assert_size_stride(primals_81, (768, 768), (768, 1))
        assert_size_stride(primals_82, (768, 768), (768, 1))
        assert_size_stride(primals_83, (3072, 768), (768, 1))
        assert_size_stride(primals_84, (768, 3072), (3072, 1))
        assert_size_stride(primals_85, (32768, 768), (768, 1))
        assert_size_stride(primals_86, (768, 768), (768, 1))
        assert_size_stride(primals_87, (768, 768), (768, 1))
        assert_size_stride(primals_88, (768, 768), (768, 1))
        assert_size_stride(primals_89, (6, 12), (12, 1))
        assert_size_stride(primals_90, (768, 768), (768, 1))
        assert_size_stride(primals_91, (3072, 768), (768, 1))
        assert_size_stride(primals_92, (768, 3072), (3072, 1))
        assert_size_stride(primals_93, (1, ), (1, ))
        assert_size_stride(primals_94, (32768, 768), (768, 1))
        assert_size_stride(primals_95, (16, 2048), (2048, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            buf1 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf2 = reinterpret_tensor(buf1, (16, 2048, 1), (2048, 1, 1), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [x, x_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy_add_embedding_mean_pow_rsqrt_0:1
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_embedding_mean_pow_rsqrt_0.run(buf2, primals_1, primals_4, buf0, 32768, 768, stream=stream0)
            del primals_4
            buf3 = empty_strided_cuda((24, 1), (1, 24), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_2, linear], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_1:2
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_1.run(primals_6, buf3, 24, stream=stream0)
            del primals_6
            buf4 = empty_strided_cuda((16, 2047, 24), (49152, 24, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_2, getitem_2, linear], Original ATen: [aten._to_copy, aten.mul, aten.slice, aten.clone]
            # [Provenance debug handles] triton_poi_fused__to_copy_clone_mul_slice_2:3
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clone_mul_slice_2.run(buf0, buf2, buf4, 786048, stream=stream0)
            buf5 = empty_strided_cuda((32752, 24), (24, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_2, getitem_2, linear], Original ATen: [aten._to_copy, aten.mul, aten.slice, aten.clone, aten._unsafe_view]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_clone_mul_slice_3:4
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_clone_mul_slice_3.run(buf4, buf5, 786048, stream=stream0)
            del buf4
            buf6 = empty_strided_cuda((32752, 1), (1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:142
            extern_kernels.mm(buf5, buf3, out=buf6)
            buf7 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            buf8 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf9 = reinterpret_tensor(buf8, (16, 2048, 1), (2048, 1, 1), 0); del buf8  # reuse
            buf11 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_2, to_1, getitem_2, linear, sigmoid, gate, getitem_3, getitem_5, mul_1, add, x_3, getitem_6, mul_2, getitem_7, mul_3, x_4, rms_norm_1], Original ATen: [aten._to_copy, aten.mul, aten.slice, aten._unsafe_view, aten.sigmoid, aten.add, aten.cat, aten.select, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_pow_rsqrt_select_sigmoid_slice_4:5
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_pow_rsqrt_select_sigmoid_slice_4.run(buf9, buf0, buf2, primals_5, buf6, primals_7, primals_8, buf7, buf11, 32768, 768, stream=stream0)
            buf10 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_3, linear_1], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:6
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_9, buf10, 589824, stream=stream0)
            del primals_9
            buf12 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_6, mul_2, getitem_7, mul_3, x_4, rms_norm_1, linear_1], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:143
            extern_kernels.mm(reinterpret_tensor(buf11, (32768, 768), (768, 1), 0), buf10, out=buf12)
            buf13 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_4, linear_2], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:7
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_10, buf13, 589824, stream=stream0)
            del primals_10
            buf14 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_6, mul_2, getitem_7, mul_3, x_4, rms_norm_1, linear_1, linear_2], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:144
            extern_kernels.mm(reinterpret_tensor(buf11, (32768, 768), (768, 1), 0), buf13, out=buf14)
            buf15 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_5, linear_3], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:8
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_11, buf15, 589824, stream=stream0)
            del primals_11
            buf16 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_6, mul_2, getitem_7, mul_3, x_4, rms_norm_1, linear_1, linear_3], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:145
            extern_kernels.mm(reinterpret_tensor(buf11, (32768, 768), (768, 1), 0), buf15, out=buf16)
            buf17 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf18 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf19 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf21 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf20 = reinterpret_tensor(buf19, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf19  # reuse
            buf22 = reinterpret_tensor(buf21, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf21  # reuse
            buf23 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf24 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, linear_1, q, linear_2, k, x1, x2, mul_4, mul_5, y1, neg, mul_6, mul_7, y2, q_1, x1_1, x2_1, mul_8, mul_9, y1_1, mul_10, mul_11, y2_1, k_1, q_2, k_2, q_3, k_3], Original ATen: [aten.slice, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.neg, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6:9
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6.run(buf20, buf22, buf12, primals_2, primals_3, buf14, buf17, buf18, buf23, buf24, 196608, 128, stream=stream0)
            del buf12
            del buf14
            # Topologically Sorted Source Nodes: [linear_3, v, y], Original ATen: [aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf25 = torch.ops.flash_attn_3._flash_attn_forward.default(buf23, buf24, reinterpret_tensor(buf16, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf26 = buf25[0]
            assert_size_stride(buf26, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf26, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf27 = buf25[1]
            assert_size_stride(buf27, (16, 6, 2048), (12288, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf27, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf25
            buf30 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_6, y_2], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:10
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_12, buf30, 589824, stream=stream0)
            del primals_12
            buf31 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_1, y_2], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:146
            extern_kernels.mm(reinterpret_tensor(buf26, (32768, 768), (768, 1), 0), buf30, out=buf31)
            buf32 = reinterpret_tensor(buf31, (16, 2048, 768), (1572864, 768, 1), 0); del buf31  # reuse
            buf33 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf34 = reinterpret_tensor(buf33, (16, 2048, 1), (2048, 1, 1), 0); del buf33  # reuse
            buf36 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_6, mul_2, getitem_7, mul_3, x_4, y_2, x_5, rms_norm_4], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_7:11
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_7.run(buf32, buf34, primals_7, buf7, primals_8, buf36, 32768, 768, stream=stream0)
            buf35 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_7, x_6], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:12
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_13, buf35, 2359296, stream=stream0)
            del primals_13
            buf37 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_4, x_6], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:147
            extern_kernels.mm(reinterpret_tensor(buf36, (32768, 768), (768, 1), 0), buf35, out=buf37)
            buf38 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_8, x_8], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:13
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_14, buf38, 2359296, stream=stream0)
            del primals_14
            buf39 = empty_strided_cuda((16, 2048, 3072), (6291456, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_6, relu, x_7], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_pow_relu_9:14
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_pow_relu_9.run(buf37, buf39, 100663296, stream=stream0)
            buf40 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_6, relu, x_7, x_8], Original ATen: [aten._unsafe_view, aten.relu, aten.pow, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:148
            extern_kernels.mm(reinterpret_tensor(buf39, (32768, 3072), (3072, 1), 0), buf38, out=buf40)
            buf41 = reinterpret_tensor(buf40, (16, 2048, 768), (1572864, 768, 1), 0); del buf40  # reuse
            buf43 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf44 = reinterpret_tensor(buf43, (16, 2048, 1), (2048, 1, 1), 0); del buf43  # reuse
            buf45 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_8, x_9, getitem_12, mul_14, getitem_13, mul_15, x_10, rms_norm_5], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_10:15
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_10.run(buf41, buf44, buf32, primals_7, primals_8, buf7, buf45, 32768, 768, stream=stream0)
            buf50 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_12, linear_9], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:16
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_18, buf50, 589824, stream=stream0)
            del primals_18
            buf51 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_7, linear_9], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:149
            extern_kernels.mm(reinterpret_tensor(buf45, (32768, 768), (768, 1), 0), buf50, out=buf51)
            buf52 = empty_strided_cuda((12, 6), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_13, linear_10], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_11:17
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_11.run(primals_19, buf52, 72, stream=stream0)
            del primals_19
            buf53 = empty_strided_cuda((32768, 6), (6, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_14, linear_10], Original ATen: [aten.slice, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:150
            extern_kernels.mm(reinterpret_tensor(buf45, (32768, 12), (768, 1), 0), buf52, out=buf53)
            buf42 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            buf54 = reinterpret_tensor(buf51, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [embedding_1, linear_9, v_1, ve_1, linear_10, sigmoid_1, gate_1, unsqueeze, mul_17, v_2], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12:18
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12.run(buf54, primals_1, primals_15, buf53, buf42, 25165824, stream=stream0)
            del primals_15
            buf46 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_10, linear_7], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:19
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_16, buf46, 589824, stream=stream0)
            del primals_16
            buf47 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:151
            extern_kernels.mm(reinterpret_tensor(buf45, (32768, 768), (768, 1), 0), buf46, out=buf47)
            buf48 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_11, linear_8], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:20
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_17, buf48, 589824, stream=stream0)
            del primals_17
            buf49 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_7, linear_8], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:152
            extern_kernels.mm(reinterpret_tensor(buf45, (32768, 768), (768, 1), 0), buf48, out=buf49)
            buf55 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf56 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf57 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf59 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf58 = reinterpret_tensor(buf57, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf57  # reuse
            buf60 = reinterpret_tensor(buf59, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf59  # reuse
            buf61 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf62 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_7, q_4, linear_8, k_4, x1_2, x2_2, mul_18, mul_19, y1_2, mul_20, mul_21, y2_2, q_5, x1_3, x2_3, mul_22, mul_23, y1_3, mul_24, mul_25, y2_3, k_5, q_6, k_6, q_7, k_7], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6:21
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6.run(buf58, buf60, buf47, primals_2, primals_3, buf49, buf55, buf56, buf61, buf62, 196608, 128, stream=stream0)
            del buf47
            del buf49
            # Topologically Sorted Source Nodes: [y_3], Original ATen: [flash_attn_3._flash_attn_forward]
            buf63 = torch.ops.flash_attn_3._flash_attn_forward.default(buf61, buf62, buf54, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf64 = buf63[0]
            assert_size_stride(buf64, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf64, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf65 = buf63[1]
            assert_size_stride(buf65, (16, 6, 2048), (12288, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf65, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf63
            buf68 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_14, y_5], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:22
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_20, buf68, 589824, stream=stream0)
            del primals_20
            buf69 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_4, y_5], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:153
            extern_kernels.mm(reinterpret_tensor(buf64, (32768, 768), (768, 1), 0), buf68, out=buf69)
            buf70 = reinterpret_tensor(buf69, (16, 2048, 768), (1572864, 768, 1), 0); del buf69  # reuse
            buf71 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf72 = reinterpret_tensor(buf71, (16, 2048, 1), (2048, 1, 1), 0); del buf71  # reuse
            buf74 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_12, mul_14, getitem_13, mul_15, x_10, y_5, x_11, rms_norm_8], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_13:23
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_13.run(buf70, buf72, primals_7, buf41, primals_8, buf7, buf74, 32768, 768, stream=stream0)
            buf73 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_15, x_12], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:24
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_21, buf73, 2359296, stream=stream0)
            del primals_21
            buf75 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_8, x_12], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:154
            extern_kernels.mm(reinterpret_tensor(buf74, (32768, 768), (768, 1), 0), buf73, out=buf75)
            buf76 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_16, x_14], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:25
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_22, buf76, 2359296, stream=stream0)
            del primals_22
            buf77 = empty_strided_cuda((16, 2048, 3072), (6291456, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_12, relu_1, x_13], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_pow_relu_9:26
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_pow_relu_9.run(buf75, buf77, 100663296, stream=stream0)
            buf78 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_12, relu_1, x_13, x_14], Original ATen: [aten._unsafe_view, aten.relu, aten.pow, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:155
            extern_kernels.mm(reinterpret_tensor(buf77, (32768, 3072), (3072, 1), 0), buf76, out=buf78)
            buf79 = reinterpret_tensor(buf78, (16, 2048, 768), (1572864, 768, 1), 0); del buf78  # reuse
            buf80 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf81 = reinterpret_tensor(buf80, (16, 2048, 1), (2048, 1, 1), 0); del buf80  # reuse
            buf83 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_14, x_15, getitem_19, mul_28, getitem_20, mul_29, x_16, rms_norm_9], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_14:27
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_14.run(buf79, buf81, buf70, primals_7, primals_8, buf7, buf83, 32768, 768, stream=stream0)
            buf82 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_17, linear_14], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:28
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_23, buf82, 589824, stream=stream0)
            del primals_23
            buf84 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_19, mul_28, getitem_20, mul_29, x_16, rms_norm_9, linear_14], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:156
            extern_kernels.mm(reinterpret_tensor(buf83, (32768, 768), (768, 1), 0), buf82, out=buf84)
            buf85 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_18, linear_15], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:29
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_24, buf85, 589824, stream=stream0)
            del primals_24
            buf86 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_19, mul_28, getitem_20, mul_29, x_16, rms_norm_9, linear_14, linear_15], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:157
            extern_kernels.mm(reinterpret_tensor(buf83, (32768, 768), (768, 1), 0), buf85, out=buf86)
            buf87 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_19, linear_16], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:30
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_25, buf87, 589824, stream=stream0)
            del primals_25
            buf88 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_19, mul_28, getitem_20, mul_29, x_16, rms_norm_9, linear_14, linear_16], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:158
            extern_kernels.mm(reinterpret_tensor(buf83, (32768, 768), (768, 1), 0), buf87, out=buf88)
            buf89 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf90 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf91 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf93 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf92 = reinterpret_tensor(buf91, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf91  # reuse
            buf94 = reinterpret_tensor(buf93, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf93  # reuse
            buf95 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf96 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_14, q_8, linear_15, k_8, x1_4, x2_4, mul_30, mul_31, y1_4, mul_32, mul_33, y2_4, q_9, x1_5, x2_5, mul_34, mul_35, y1_5, mul_36, mul_37, y2_5, k_9, q_10, k_10, q_11, k_11], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6:31
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6.run(buf92, buf94, buf84, primals_2, primals_3, buf86, buf89, buf90, buf95, buf96, 196608, 128, stream=stream0)
            del buf84
            del buf86
            # Topologically Sorted Source Nodes: [linear_16, v_3, y_6], Original ATen: [aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf97 = torch.ops.flash_attn_3._flash_attn_forward.default(buf95, buf96, reinterpret_tensor(buf88, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf98 = buf97[0]
            assert_size_stride(buf98, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf98, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf99 = buf97[1]
            assert_size_stride(buf99, (16, 6, 2048), (12288, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf99, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf97
            buf102 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_20, y_8], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:32
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_26, buf102, 589824, stream=stream0)
            del primals_26
            buf103 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_7, y_8], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:159
            extern_kernels.mm(reinterpret_tensor(buf98, (32768, 768), (768, 1), 0), buf102, out=buf103)
            buf104 = reinterpret_tensor(buf103, (16, 2048, 768), (1572864, 768, 1), 0); del buf103  # reuse
            buf105 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf106 = reinterpret_tensor(buf105, (16, 2048, 1), (2048, 1, 1), 0); del buf105  # reuse
            buf108 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_19, mul_28, getitem_20, mul_29, x_16, y_8, x_17, rms_norm_12], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_15:33
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_15.run(buf104, buf106, primals_7, buf79, primals_8, buf7, buf108, 32768, 768, stream=stream0)
            buf107 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_21, x_18], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:34
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_27, buf107, 2359296, stream=stream0)
            del primals_27
            buf109 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_12, x_18], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:160
            extern_kernels.mm(reinterpret_tensor(buf108, (32768, 768), (768, 1), 0), buf107, out=buf109)
            buf110 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_22, x_20], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:35
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_28, buf110, 2359296, stream=stream0)
            del primals_28
            buf111 = empty_strided_cuda((16, 2048, 3072), (6291456, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_18, relu_2, x_19], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_pow_relu_9:36
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_pow_relu_9.run(buf109, buf111, 100663296, stream=stream0)
            buf112 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_18, relu_2, x_19, x_20], Original ATen: [aten._unsafe_view, aten.relu, aten.pow, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:161
            extern_kernels.mm(reinterpret_tensor(buf111, (32768, 3072), (3072, 1), 0), buf110, out=buf112)
            buf113 = reinterpret_tensor(buf112, (16, 2048, 768), (1572864, 768, 1), 0); del buf112  # reuse
            buf115 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf116 = reinterpret_tensor(buf115, (16, 2048, 1), (2048, 1, 1), 0); del buf115  # reuse
            buf117 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_20, x_21, getitem_25, mul_40, getitem_26, mul_41, x_22, rms_norm_13], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_16:37
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_16.run(buf113, buf116, buf104, primals_7, primals_8, buf7, buf117, 32768, 768, stream=stream0)
            buf122 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_26, linear_22], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:38
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_32, buf122, 589824, stream=stream0)
            del primals_32
            buf123 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_20, linear_22], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:162
            extern_kernels.mm(reinterpret_tensor(buf117, (32768, 768), (768, 1), 0), buf122, out=buf123)
            buf124 = empty_strided_cuda((12, 6), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_27, linear_23], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_11:39
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_11.run(primals_33, buf124, 72, stream=stream0)
            del primals_33
            buf125 = empty_strided_cuda((32768, 6), (6, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_27, linear_23], Original ATen: [aten.slice, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:163
            extern_kernels.mm(reinterpret_tensor(buf117, (32768, 12), (768, 1), 0), buf124, out=buf125)
            buf114 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            buf126 = reinterpret_tensor(buf123, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [embedding_2, linear_22, v_4, ve_3, linear_23, sigmoid_2, gate_2, unsqueeze_1, mul_43, v_5], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12:40
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12.run(buf126, primals_1, primals_29, buf125, buf114, 25165824, stream=stream0)
            del primals_29
            buf118 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_24, linear_20], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:41
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_30, buf118, 589824, stream=stream0)
            del primals_30
            buf119 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:164
            extern_kernels.mm(reinterpret_tensor(buf117, (32768, 768), (768, 1), 0), buf118, out=buf119)
            buf120 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_25, linear_21], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:42
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_31, buf120, 589824, stream=stream0)
            del primals_31
            buf121 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_20, linear_21], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:165
            extern_kernels.mm(reinterpret_tensor(buf117, (32768, 768), (768, 1), 0), buf120, out=buf121)
            buf127 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf128 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf129 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf131 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf130 = reinterpret_tensor(buf129, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf129  # reuse
            buf132 = reinterpret_tensor(buf131, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf131  # reuse
            buf133 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf134 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_20, q_12, linear_21, k_12, x1_6, x2_6, mul_44, mul_45, y1_6, mul_46, mul_47, y2_6, q_13, x1_7, x2_7, mul_48, mul_49, y1_7, mul_50, mul_51, y2_7, k_13, q_14, k_14, q_15, k_15], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6:43
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6.run(buf130, buf132, buf119, primals_2, primals_3, buf121, buf127, buf128, buf133, buf134, 196608, 128, stream=stream0)
            del buf119
            del buf121
            # Topologically Sorted Source Nodes: [y_9], Original ATen: [flash_attn_3._flash_attn_forward]
            buf135 = torch.ops.flash_attn_3._flash_attn_forward.default(buf133, buf134, buf126, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf136 = buf135[0]
            assert_size_stride(buf136, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf136, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf137 = buf135[1]
            assert_size_stride(buf137, (16, 6, 2048), (12288, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf137, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf135
            buf140 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_28, y_11], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:44
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_34, buf140, 589824, stream=stream0)
            del primals_34
            buf141 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_10, y_11], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:166
            extern_kernels.mm(reinterpret_tensor(buf136, (32768, 768), (768, 1), 0), buf140, out=buf141)
            buf142 = reinterpret_tensor(buf141, (16, 2048, 768), (1572864, 768, 1), 0); del buf141  # reuse
            buf143 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf144 = reinterpret_tensor(buf143, (16, 2048, 1), (2048, 1, 1), 0); del buf143  # reuse
            buf146 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_25, mul_40, getitem_26, mul_41, x_22, y_11, x_23, rms_norm_16], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_17:45
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_17.run(buf142, buf144, primals_7, buf113, primals_8, buf7, buf146, 32768, 768, stream=stream0)
            buf145 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_29, x_24], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:46
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_35, buf145, 2359296, stream=stream0)
            del primals_35
            buf147 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_16, x_24], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:167
            extern_kernels.mm(reinterpret_tensor(buf146, (32768, 768), (768, 1), 0), buf145, out=buf147)
            buf148 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_30, x_26], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:47
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_36, buf148, 2359296, stream=stream0)
            del primals_36
            buf149 = empty_strided_cuda((16, 2048, 3072), (6291456, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_24, relu_3, x_25], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_pow_relu_9:48
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_pow_relu_9.run(buf147, buf149, 100663296, stream=stream0)
            buf150 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_24, relu_3, x_25, x_26], Original ATen: [aten._unsafe_view, aten.relu, aten.pow, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:168
            extern_kernels.mm(reinterpret_tensor(buf149, (32768, 3072), (3072, 1), 0), buf148, out=buf150)
            buf151 = reinterpret_tensor(buf150, (16, 2048, 768), (1572864, 768, 1), 0); del buf150  # reuse
            buf152 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf153 = reinterpret_tensor(buf152, (16, 2048, 1), (2048, 1, 1), 0); del buf152  # reuse
            buf155 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_26, x_27, getitem_32, mul_54, getitem_33, mul_55, x_28, rms_norm_17], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_18:49
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_18.run(buf151, buf153, buf142, primals_7, primals_8, buf7, buf155, 32768, 768, stream=stream0)
            buf154 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_31, linear_27], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:50
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_37, buf154, 589824, stream=stream0)
            del primals_37
            buf156 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_32, mul_54, getitem_33, mul_55, x_28, rms_norm_17, linear_27], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:169
            extern_kernels.mm(reinterpret_tensor(buf155, (32768, 768), (768, 1), 0), buf154, out=buf156)
            buf157 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_32, linear_28], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:51
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_38, buf157, 589824, stream=stream0)
            del primals_38
            buf158 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_32, mul_54, getitem_33, mul_55, x_28, rms_norm_17, linear_27, linear_28], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:170
            extern_kernels.mm(reinterpret_tensor(buf155, (32768, 768), (768, 1), 0), buf157, out=buf158)
            buf159 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_33, linear_29], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:52
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_39, buf159, 589824, stream=stream0)
            del primals_39
            buf160 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_32, mul_54, getitem_33, mul_55, x_28, rms_norm_17, linear_27, linear_29], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:171
            extern_kernels.mm(reinterpret_tensor(buf155, (32768, 768), (768, 1), 0), buf159, out=buf160)
            buf161 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf162 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf163 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf165 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf164 = reinterpret_tensor(buf163, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf163  # reuse
            buf166 = reinterpret_tensor(buf165, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf165  # reuse
            buf167 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf168 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_27, q_16, linear_28, k_16, x1_8, x2_8, mul_56, mul_57, y1_8, mul_58, mul_59, y2_8, q_17, x1_9, x2_9, mul_60, mul_61, y1_9, mul_62, mul_63, y2_9, k_17, q_18, k_18, q_19, k_19], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6:53
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6.run(buf164, buf166, buf156, primals_2, primals_3, buf158, buf161, buf162, buf167, buf168, 196608, 128, stream=stream0)
            del buf156
            del buf158
            # Topologically Sorted Source Nodes: [linear_29, v_6, y_12], Original ATen: [aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf169 = torch.ops.flash_attn_3._flash_attn_forward.default(buf167, buf168, reinterpret_tensor(buf160, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf170 = buf169[0]
            assert_size_stride(buf170, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf170, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf171 = buf169[1]
            assert_size_stride(buf171, (16, 6, 2048), (12288, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf171, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf169
            buf174 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_34, y_14], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:54
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_40, buf174, 589824, stream=stream0)
            del primals_40
            buf175 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_13, y_14], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:172
            extern_kernels.mm(reinterpret_tensor(buf170, (32768, 768), (768, 1), 0), buf174, out=buf175)
            buf176 = reinterpret_tensor(buf175, (16, 2048, 768), (1572864, 768, 1), 0); del buf175  # reuse
            buf177 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf178 = reinterpret_tensor(buf177, (16, 2048, 1), (2048, 1, 1), 0); del buf177  # reuse
            buf180 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_32, mul_54, getitem_33, mul_55, x_28, y_14, x_29, rms_norm_20], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_19:55
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_19.run(buf176, buf178, primals_7, buf151, primals_8, buf7, buf180, 32768, 768, stream=stream0)
            buf179 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_35, x_30], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:56
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_41, buf179, 2359296, stream=stream0)
            del primals_41
            buf181 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_20, x_30], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:173
            extern_kernels.mm(reinterpret_tensor(buf180, (32768, 768), (768, 1), 0), buf179, out=buf181)
            buf182 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_36, x_32], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:57
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_42, buf182, 2359296, stream=stream0)
            del primals_42
            buf183 = empty_strided_cuda((16, 2048, 3072), (6291456, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_30, relu_4, x_31], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_pow_relu_9:58
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_pow_relu_9.run(buf181, buf183, 100663296, stream=stream0)
            buf184 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_30, relu_4, x_31, x_32], Original ATen: [aten._unsafe_view, aten.relu, aten.pow, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:174
            extern_kernels.mm(reinterpret_tensor(buf183, (32768, 3072), (3072, 1), 0), buf182, out=buf184)
            buf185 = reinterpret_tensor(buf184, (16, 2048, 768), (1572864, 768, 1), 0); del buf184  # reuse
            buf187 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf188 = reinterpret_tensor(buf187, (16, 2048, 1), (2048, 1, 1), 0); del buf187  # reuse
            buf189 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_32, x_33, getitem_38, mul_66, getitem_39, mul_67, x_34, rms_norm_21], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_20:59
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_20.run(buf185, buf188, buf176, primals_7, primals_8, buf7, buf189, 32768, 768, stream=stream0)
            buf194 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_40, linear_35], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:60
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_46, buf194, 589824, stream=stream0)
            del primals_46
            buf195 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_33, linear_35], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:175
            extern_kernels.mm(reinterpret_tensor(buf189, (32768, 768), (768, 1), 0), buf194, out=buf195)
            buf196 = empty_strided_cuda((12, 6), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_41, linear_36], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_11:61
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_11.run(primals_47, buf196, 72, stream=stream0)
            del primals_47
            buf197 = empty_strided_cuda((32768, 6), (6, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_40, linear_36], Original ATen: [aten.slice, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:176
            extern_kernels.mm(reinterpret_tensor(buf189, (32768, 12), (768, 1), 0), buf196, out=buf197)
            buf186 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            buf198 = reinterpret_tensor(buf195, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0); del buf195  # reuse
            # Topologically Sorted Source Nodes: [embedding_3, linear_35, v_7, ve_5, linear_36, sigmoid_3, gate_3, unsqueeze_2, mul_69, v_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12:62
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12.run(buf198, primals_1, primals_43, buf197, buf186, 25165824, stream=stream0)
            del primals_43
            buf190 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_38, linear_33], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:63
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_44, buf190, 589824, stream=stream0)
            del primals_44
            buf191 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:177
            extern_kernels.mm(reinterpret_tensor(buf189, (32768, 768), (768, 1), 0), buf190, out=buf191)
            buf192 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_39, linear_34], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:64
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_45, buf192, 589824, stream=stream0)
            del primals_45
            buf193 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_33, linear_34], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:178
            extern_kernels.mm(reinterpret_tensor(buf189, (32768, 768), (768, 1), 0), buf192, out=buf193)
            buf199 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf200 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf201 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf203 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf202 = reinterpret_tensor(buf201, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf201  # reuse
            buf204 = reinterpret_tensor(buf203, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf203  # reuse
            buf205 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf206 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_33, q_20, linear_34, k_20, x1_10, x2_10, mul_70, mul_71, y1_10, mul_72, mul_73, y2_10, q_21, x1_11, x2_11, mul_74, mul_75, y1_11, mul_76, mul_77, y2_11, k_21, q_22, k_22, q_23, k_23], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6:65
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6.run(buf202, buf204, buf191, primals_2, primals_3, buf193, buf199, buf200, buf205, buf206, 196608, 128, stream=stream0)
            del buf191
            del buf193
            # Topologically Sorted Source Nodes: [y_15], Original ATen: [flash_attn_3._flash_attn_forward]
            buf207 = torch.ops.flash_attn_3._flash_attn_forward.default(buf205, buf206, buf198, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf208 = buf207[0]
            assert_size_stride(buf208, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf208, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf209 = buf207[1]
            assert_size_stride(buf209, (16, 6, 2048), (12288, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf209, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf207
            buf212 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_42, y_17], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:66
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_48, buf212, 589824, stream=stream0)
            del primals_48
            buf213 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_16, y_17], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:179
            extern_kernels.mm(reinterpret_tensor(buf208, (32768, 768), (768, 1), 0), buf212, out=buf213)
            buf214 = reinterpret_tensor(buf213, (16, 2048, 768), (1572864, 768, 1), 0); del buf213  # reuse
            buf215 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf216 = reinterpret_tensor(buf215, (16, 2048, 1), (2048, 1, 1), 0); del buf215  # reuse
            buf218 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_38, mul_66, getitem_39, mul_67, x_34, y_17, x_35, rms_norm_24], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_21:67
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_21.run(buf214, buf216, primals_7, buf185, primals_8, buf7, buf218, 32768, 768, stream=stream0)
            buf217 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_43, x_36], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:68
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_49, buf217, 2359296, stream=stream0)
            del primals_49
            buf219 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_24, x_36], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:180
            extern_kernels.mm(reinterpret_tensor(buf218, (32768, 768), (768, 1), 0), buf217, out=buf219)
            buf220 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_44, x_38], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:69
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_50, buf220, 2359296, stream=stream0)
            del primals_50
            buf221 = empty_strided_cuda((16, 2048, 3072), (6291456, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_36, relu_5, x_37], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_pow_relu_9:70
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_pow_relu_9.run(buf219, buf221, 100663296, stream=stream0)
            buf222 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_36, relu_5, x_37, x_38], Original ATen: [aten._unsafe_view, aten.relu, aten.pow, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:181
            extern_kernels.mm(reinterpret_tensor(buf221, (32768, 3072), (3072, 1), 0), buf220, out=buf222)
            buf223 = reinterpret_tensor(buf222, (16, 2048, 768), (1572864, 768, 1), 0); del buf222  # reuse
            buf224 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf225 = reinterpret_tensor(buf224, (16, 2048, 1), (2048, 1, 1), 0); del buf224  # reuse
            buf227 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_38, x_39, getitem_45, mul_80, getitem_46, mul_81, x_40, rms_norm_25], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_22:71
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_22.run(buf223, buf225, buf214, primals_7, primals_8, buf7, buf227, 32768, 768, stream=stream0)
            buf226 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_45, linear_40], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:72
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_51, buf226, 589824, stream=stream0)
            del primals_51
            buf228 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_45, mul_80, getitem_46, mul_81, x_40, rms_norm_25, linear_40], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:182
            extern_kernels.mm(reinterpret_tensor(buf227, (32768, 768), (768, 1), 0), buf226, out=buf228)
            buf229 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_46, linear_41], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:73
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_52, buf229, 589824, stream=stream0)
            del primals_52
            buf230 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_45, mul_80, getitem_46, mul_81, x_40, rms_norm_25, linear_40, linear_41], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:183
            extern_kernels.mm(reinterpret_tensor(buf227, (32768, 768), (768, 1), 0), buf229, out=buf230)
            buf231 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_47, linear_42], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:74
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_53, buf231, 589824, stream=stream0)
            del primals_53
            buf232 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_45, mul_80, getitem_46, mul_81, x_40, rms_norm_25, linear_40, linear_42], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:184
            extern_kernels.mm(reinterpret_tensor(buf227, (32768, 768), (768, 1), 0), buf231, out=buf232)
            buf233 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf234 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf235 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf237 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf236 = reinterpret_tensor(buf235, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf235  # reuse
            buf238 = reinterpret_tensor(buf237, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf237  # reuse
            buf239 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf240 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_40, q_24, linear_41, k_24, x1_12, x2_12, mul_82, mul_83, y1_12, mul_84, mul_85, y2_12, q_25, x1_13, x2_13, mul_86, mul_87, y1_13, mul_88, mul_89, y2_13, k_25, q_26, k_26, q_27, k_27], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6:75
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6.run(buf236, buf238, buf228, primals_2, primals_3, buf230, buf233, buf234, buf239, buf240, 196608, 128, stream=stream0)
            del buf228
            del buf230
            # Topologically Sorted Source Nodes: [linear_42, v_9, y_18], Original ATen: [aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf241 = torch.ops.flash_attn_3._flash_attn_forward.default(buf239, buf240, reinterpret_tensor(buf232, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf242 = buf241[0]
            assert_size_stride(buf242, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf242, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf243 = buf241[1]
            assert_size_stride(buf243, (16, 6, 2048), (12288, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf243, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf241
            buf246 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_48, y_20], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:76
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_54, buf246, 589824, stream=stream0)
            del primals_54
            buf247 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_19, y_20], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:185
            extern_kernels.mm(reinterpret_tensor(buf242, (32768, 768), (768, 1), 0), buf246, out=buf247)
            buf248 = reinterpret_tensor(buf247, (16, 2048, 768), (1572864, 768, 1), 0); del buf247  # reuse
            buf249 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf250 = reinterpret_tensor(buf249, (16, 2048, 1), (2048, 1, 1), 0); del buf249  # reuse
            buf252 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_45, mul_80, getitem_46, mul_81, x_40, y_20, x_41, rms_norm_28], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_23:77
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_23.run(buf248, buf250, primals_7, buf223, primals_8, buf7, buf252, 32768, 768, stream=stream0)
            buf251 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_49, x_42], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:78
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_55, buf251, 2359296, stream=stream0)
            del primals_55
            buf253 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_28, x_42], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:186
            extern_kernels.mm(reinterpret_tensor(buf252, (32768, 768), (768, 1), 0), buf251, out=buf253)
            buf254 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_50, x_44], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:79
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_56, buf254, 2359296, stream=stream0)
            del primals_56
            buf255 = empty_strided_cuda((16, 2048, 3072), (6291456, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_42, relu_6, x_43], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_pow_relu_9:80
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_pow_relu_9.run(buf253, buf255, 100663296, stream=stream0)
            buf256 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_42, relu_6, x_43, x_44], Original ATen: [aten._unsafe_view, aten.relu, aten.pow, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:187
            extern_kernels.mm(reinterpret_tensor(buf255, (32768, 3072), (3072, 1), 0), buf254, out=buf256)
            buf257 = reinterpret_tensor(buf256, (16, 2048, 768), (1572864, 768, 1), 0); del buf256  # reuse
            buf259 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf260 = reinterpret_tensor(buf259, (16, 2048, 1), (2048, 1, 1), 0); del buf259  # reuse
            buf261 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_44, x_45, getitem_51, mul_92, getitem_52, mul_93, x_46, rms_norm_29], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_24:81
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_24.run(buf257, buf260, buf248, primals_7, primals_8, buf7, buf261, 32768, 768, stream=stream0)
            buf266 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_54, linear_48], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:82
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_60, buf266, 589824, stream=stream0)
            del primals_60
            buf267 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_46, linear_48], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:188
            extern_kernels.mm(reinterpret_tensor(buf261, (32768, 768), (768, 1), 0), buf266, out=buf267)
            buf268 = empty_strided_cuda((12, 6), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_55, linear_49], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_11:83
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_11.run(primals_61, buf268, 72, stream=stream0)
            del primals_61
            buf269 = empty_strided_cuda((32768, 6), (6, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_53, linear_49], Original ATen: [aten.slice, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:189
            extern_kernels.mm(reinterpret_tensor(buf261, (32768, 12), (768, 1), 0), buf268, out=buf269)
            buf258 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            buf270 = reinterpret_tensor(buf267, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0); del buf267  # reuse
            # Topologically Sorted Source Nodes: [embedding_4, linear_48, v_10, ve_7, linear_49, sigmoid_4, gate_4, unsqueeze_3, mul_95, v_11], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12:84
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12.run(buf270, primals_1, primals_57, buf269, buf258, 25165824, stream=stream0)
            del primals_57
            buf262 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_52, linear_46], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:85
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_58, buf262, 589824, stream=stream0)
            del primals_58
            buf263 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:190
            extern_kernels.mm(reinterpret_tensor(buf261, (32768, 768), (768, 1), 0), buf262, out=buf263)
            buf264 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_53, linear_47], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:86
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_59, buf264, 589824, stream=stream0)
            del primals_59
            buf265 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_46, linear_47], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:191
            extern_kernels.mm(reinterpret_tensor(buf261, (32768, 768), (768, 1), 0), buf264, out=buf265)
            buf271 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf272 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf273 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf275 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf274 = reinterpret_tensor(buf273, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf273  # reuse
            buf276 = reinterpret_tensor(buf275, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf275  # reuse
            buf277 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf278 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_46, q_28, linear_47, k_28, x1_14, x2_14, mul_96, mul_97, y1_14, mul_98, mul_99, y2_14, q_29, x1_15, x2_15, mul_100, mul_101, y1_15, mul_102, mul_103, y2_15, k_29, q_30, k_30, q_31, k_31], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6:87
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6.run(buf274, buf276, buf263, primals_2, primals_3, buf265, buf271, buf272, buf277, buf278, 196608, 128, stream=stream0)
            del buf263
            del buf265
            # Topologically Sorted Source Nodes: [y_21], Original ATen: [flash_attn_3._flash_attn_forward]
            buf279 = torch.ops.flash_attn_3._flash_attn_forward.default(buf277, buf278, buf270, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf280 = buf279[0]
            assert_size_stride(buf280, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf280, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf281 = buf279[1]
            assert_size_stride(buf281, (16, 6, 2048), (12288, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf281, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf279
            buf284 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_56, y_23], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:88
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_62, buf284, 589824, stream=stream0)
            del primals_62
            buf285 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_22, y_23], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:192
            extern_kernels.mm(reinterpret_tensor(buf280, (32768, 768), (768, 1), 0), buf284, out=buf285)
            buf286 = reinterpret_tensor(buf285, (16, 2048, 768), (1572864, 768, 1), 0); del buf285  # reuse
            buf287 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf288 = reinterpret_tensor(buf287, (16, 2048, 1), (2048, 1, 1), 0); del buf287  # reuse
            buf290 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_51, mul_92, getitem_52, mul_93, x_46, y_23, x_47, rms_norm_32], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_25:89
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_25.run(buf286, buf288, primals_7, buf257, primals_8, buf7, buf290, 32768, 768, stream=stream0)
            buf289 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_57, x_48], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:90
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_63, buf289, 2359296, stream=stream0)
            del primals_63
            buf291 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_32, x_48], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:193
            extern_kernels.mm(reinterpret_tensor(buf290, (32768, 768), (768, 1), 0), buf289, out=buf291)
            buf292 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_58, x_50], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:91
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_64, buf292, 2359296, stream=stream0)
            del primals_64
            buf293 = empty_strided_cuda((16, 2048, 3072), (6291456, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_48, relu_7, x_49], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_pow_relu_9:92
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_pow_relu_9.run(buf291, buf293, 100663296, stream=stream0)
            buf294 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_48, relu_7, x_49, x_50], Original ATen: [aten._unsafe_view, aten.relu, aten.pow, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:194
            extern_kernels.mm(reinterpret_tensor(buf293, (32768, 3072), (3072, 1), 0), buf292, out=buf294)
            buf295 = reinterpret_tensor(buf294, (16, 2048, 768), (1572864, 768, 1), 0); del buf294  # reuse
            buf296 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf297 = reinterpret_tensor(buf296, (16, 2048, 1), (2048, 1, 1), 0); del buf296  # reuse
            buf299 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_50, x_51, getitem_58, mul_106, getitem_59, mul_107, x_52, rms_norm_33], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_26:93
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_26.run(buf295, buf297, buf286, primals_7, primals_8, buf7, buf299, 32768, 768, stream=stream0)
            buf298 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_59, linear_53], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:94
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_65, buf298, 589824, stream=stream0)
            del primals_65
            buf300 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_58, mul_106, getitem_59, mul_107, x_52, rms_norm_33, linear_53], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:195
            extern_kernels.mm(reinterpret_tensor(buf299, (32768, 768), (768, 1), 0), buf298, out=buf300)
            buf301 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_60, linear_54], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:95
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_66, buf301, 589824, stream=stream0)
            del primals_66
            buf302 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_58, mul_106, getitem_59, mul_107, x_52, rms_norm_33, linear_53, linear_54], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:196
            extern_kernels.mm(reinterpret_tensor(buf299, (32768, 768), (768, 1), 0), buf301, out=buf302)
            buf303 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_61, linear_55], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:96
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_67, buf303, 589824, stream=stream0)
            del primals_67
            buf304 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_58, mul_106, getitem_59, mul_107, x_52, rms_norm_33, linear_53, linear_55], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:197
            extern_kernels.mm(reinterpret_tensor(buf299, (32768, 768), (768, 1), 0), buf303, out=buf304)
            buf305 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf306 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf307 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf309 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf308 = reinterpret_tensor(buf307, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf307  # reuse
            buf310 = reinterpret_tensor(buf309, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf309  # reuse
            buf311 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf312 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_53, q_32, linear_54, k_32, x1_16, x2_16, mul_108, mul_109, y1_16, mul_110, mul_111, y2_16, q_33, x1_17, x2_17, mul_112, mul_113, y1_17, mul_114, mul_115, y2_17, k_33, q_34, k_34, q_35, k_35], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6:97
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6.run(buf308, buf310, buf300, primals_2, primals_3, buf302, buf305, buf306, buf311, buf312, 196608, 128, stream=stream0)
            del buf300
            del buf302
            # Topologically Sorted Source Nodes: [linear_55, v_12, y_24], Original ATen: [aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf313 = torch.ops.flash_attn_3._flash_attn_forward.default(buf311, buf312, reinterpret_tensor(buf304, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf314 = buf313[0]
            assert_size_stride(buf314, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf314, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf315 = buf313[1]
            assert_size_stride(buf315, (16, 6, 2048), (12288, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf315, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf313
            buf318 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_62, y_26], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:98
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_68, buf318, 589824, stream=stream0)
            del primals_68
            buf319 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_25, y_26], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:198
            extern_kernels.mm(reinterpret_tensor(buf314, (32768, 768), (768, 1), 0), buf318, out=buf319)
            buf320 = reinterpret_tensor(buf319, (16, 2048, 768), (1572864, 768, 1), 0); del buf319  # reuse
            buf321 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf322 = reinterpret_tensor(buf321, (16, 2048, 1), (2048, 1, 1), 0); del buf321  # reuse
            buf324 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_58, mul_106, getitem_59, mul_107, x_52, y_26, x_53, rms_norm_36], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_27:99
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_27.run(buf320, buf322, primals_7, buf295, primals_8, buf7, buf324, 32768, 768, stream=stream0)
            buf323 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_63, x_54], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:100
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_69, buf323, 2359296, stream=stream0)
            del primals_69
            buf325 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_36, x_54], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:199
            extern_kernels.mm(reinterpret_tensor(buf324, (32768, 768), (768, 1), 0), buf323, out=buf325)
            buf326 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_64, x_56], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:101
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_70, buf326, 2359296, stream=stream0)
            del primals_70
            buf327 = empty_strided_cuda((16, 2048, 3072), (6291456, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_54, relu_8, x_55], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_pow_relu_9:102
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_pow_relu_9.run(buf325, buf327, 100663296, stream=stream0)
            buf328 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_54, relu_8, x_55, x_56], Original ATen: [aten._unsafe_view, aten.relu, aten.pow, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:200
            extern_kernels.mm(reinterpret_tensor(buf327, (32768, 3072), (3072, 1), 0), buf326, out=buf328)
            buf329 = reinterpret_tensor(buf328, (16, 2048, 768), (1572864, 768, 1), 0); del buf328  # reuse
            buf331 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf332 = reinterpret_tensor(buf331, (16, 2048, 1), (2048, 1, 1), 0); del buf331  # reuse
            buf333 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_56, x_57, getitem_64, mul_118, getitem_65, mul_119, x_58, rms_norm_37], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_28:103
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_28.run(buf329, buf332, buf320, primals_7, primals_8, buf7, buf333, 32768, 768, stream=stream0)
            buf338 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_68, linear_61], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:104
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_74, buf338, 589824, stream=stream0)
            del primals_74
            buf339 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_59, linear_61], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:201
            extern_kernels.mm(reinterpret_tensor(buf333, (32768, 768), (768, 1), 0), buf338, out=buf339)
            buf340 = empty_strided_cuda((12, 6), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_69, linear_62], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_11:105
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_11.run(primals_75, buf340, 72, stream=stream0)
            del primals_75
            buf341 = empty_strided_cuda((32768, 6), (6, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_66, linear_62], Original ATen: [aten.slice, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:202
            extern_kernels.mm(reinterpret_tensor(buf333, (32768, 12), (768, 1), 0), buf340, out=buf341)
            buf330 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            buf342 = reinterpret_tensor(buf339, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0); del buf339  # reuse
            # Topologically Sorted Source Nodes: [embedding_5, linear_61, v_13, ve_9, linear_62, sigmoid_5, gate_5, unsqueeze_4, mul_121, v_14], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12:106
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12.run(buf342, primals_1, primals_71, buf341, buf330, 25165824, stream=stream0)
            del primals_71
            buf334 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_66, linear_59], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:107
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_72, buf334, 589824, stream=stream0)
            del primals_72
            buf335 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_59], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:203
            extern_kernels.mm(reinterpret_tensor(buf333, (32768, 768), (768, 1), 0), buf334, out=buf335)
            buf336 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_67, linear_60], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:108
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_73, buf336, 589824, stream=stream0)
            del primals_73
            buf337 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_59, linear_60], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:204
            extern_kernels.mm(reinterpret_tensor(buf333, (32768, 768), (768, 1), 0), buf336, out=buf337)
            buf343 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf344 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf345 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf347 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf346 = reinterpret_tensor(buf345, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf345  # reuse
            buf348 = reinterpret_tensor(buf347, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf347  # reuse
            buf349 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf350 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_59, q_36, linear_60, k_36, x1_18, x2_18, mul_122, mul_123, y1_18, mul_124, mul_125, y2_18, q_37, x1_19, x2_19, mul_126, mul_127, y1_19, mul_128, mul_129, y2_19, k_37, q_38, k_38, q_39, k_39], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6:109
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6.run(buf346, buf348, buf335, primals_2, primals_3, buf337, buf343, buf344, buf349, buf350, 196608, 128, stream=stream0)
            del buf335
            del buf337
            # Topologically Sorted Source Nodes: [y_27], Original ATen: [flash_attn_3._flash_attn_forward]
            buf351 = torch.ops.flash_attn_3._flash_attn_forward.default(buf349, buf350, buf342, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf352 = buf351[0]
            assert_size_stride(buf352, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf352, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf353 = buf351[1]
            assert_size_stride(buf353, (16, 6, 2048), (12288, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf353, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf351
            buf356 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_70, y_29], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:110
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_76, buf356, 589824, stream=stream0)
            del primals_76
            buf357 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_28, y_29], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:205
            extern_kernels.mm(reinterpret_tensor(buf352, (32768, 768), (768, 1), 0), buf356, out=buf357)
            buf358 = reinterpret_tensor(buf357, (16, 2048, 768), (1572864, 768, 1), 0); del buf357  # reuse
            buf359 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf360 = reinterpret_tensor(buf359, (16, 2048, 1), (2048, 1, 1), 0); del buf359  # reuse
            buf362 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_64, mul_118, getitem_65, mul_119, x_58, y_29, x_59, rms_norm_40], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_29:111
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_29.run(buf358, buf360, primals_7, buf329, primals_8, buf7, buf362, 32768, 768, stream=stream0)
            buf361 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_71, x_60], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:112
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_77, buf361, 2359296, stream=stream0)
            del primals_77
            buf363 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_40, x_60], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:206
            extern_kernels.mm(reinterpret_tensor(buf362, (32768, 768), (768, 1), 0), buf361, out=buf363)
            buf364 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_72, x_62], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:113
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_78, buf364, 2359296, stream=stream0)
            del primals_78
            buf365 = empty_strided_cuda((16, 2048, 3072), (6291456, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_60, relu_9, x_61], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_pow_relu_9:114
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_pow_relu_9.run(buf363, buf365, 100663296, stream=stream0)
            buf366 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_60, relu_9, x_61, x_62], Original ATen: [aten._unsafe_view, aten.relu, aten.pow, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:207
            extern_kernels.mm(reinterpret_tensor(buf365, (32768, 3072), (3072, 1), 0), buf364, out=buf366)
            buf367 = reinterpret_tensor(buf366, (16, 2048, 768), (1572864, 768, 1), 0); del buf366  # reuse
            buf368 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf369 = reinterpret_tensor(buf368, (16, 2048, 1), (2048, 1, 1), 0); del buf368  # reuse
            buf371 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_62, x_63, getitem_71, mul_132, getitem_72, mul_133, x_64, rms_norm_41], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_30:115
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_30.run(buf367, buf369, buf358, primals_7, primals_8, buf7, buf371, 32768, 768, stream=stream0)
            buf370 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_73, linear_66], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:116
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_79, buf370, 589824, stream=stream0)
            del primals_79
            buf372 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_71, mul_132, getitem_72, mul_133, x_64, rms_norm_41, linear_66], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:208
            extern_kernels.mm(reinterpret_tensor(buf371, (32768, 768), (768, 1), 0), buf370, out=buf372)
            buf373 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_74, linear_67], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:117
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_80, buf373, 589824, stream=stream0)
            del primals_80
            buf374 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_71, mul_132, getitem_72, mul_133, x_64, rms_norm_41, linear_66, linear_67], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:209
            extern_kernels.mm(reinterpret_tensor(buf371, (32768, 768), (768, 1), 0), buf373, out=buf374)
            buf375 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_75, linear_68], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:118
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_81, buf375, 589824, stream=stream0)
            del primals_81
            buf376 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_71, mul_132, getitem_72, mul_133, x_64, rms_norm_41, linear_66, linear_68], Original ATen: [aten.select, aten.mul, aten.add, aten._to_copy, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:210
            extern_kernels.mm(reinterpret_tensor(buf371, (32768, 768), (768, 1), 0), buf375, out=buf376)
            buf377 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf378 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf379 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf381 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf380 = reinterpret_tensor(buf379, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf379  # reuse
            buf382 = reinterpret_tensor(buf381, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf381  # reuse
            buf383 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf384 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_66, q_40, linear_67, k_40, x1_20, x2_20, mul_134, mul_135, y1_20, mul_136, mul_137, y2_20, q_41, x1_21, x2_21, mul_138, mul_139, y1_21, mul_140, mul_141, y2_21, k_41, q_42, k_42, q_43, k_43], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6:119
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6.run(buf380, buf382, buf372, primals_2, primals_3, buf374, buf377, buf378, buf383, buf384, 196608, 128, stream=stream0)
            del buf372
            del buf374
            # Topologically Sorted Source Nodes: [linear_68, v_15, y_30], Original ATen: [aten._unsafe_view, aten.view, flash_attn_3._flash_attn_forward]
            buf385 = torch.ops.flash_attn_3._flash_attn_forward.default(buf383, buf384, reinterpret_tensor(buf376, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf386 = buf385[0]
            assert_size_stride(buf386, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf386, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf387 = buf385[1]
            assert_size_stride(buf387, (16, 6, 2048), (12288, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf387, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf385
            buf390 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_76, y_32], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:120
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_82, buf390, 589824, stream=stream0)
            del primals_82
            buf391 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_31, y_32], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:211
            extern_kernels.mm(reinterpret_tensor(buf386, (32768, 768), (768, 1), 0), buf390, out=buf391)
            buf392 = reinterpret_tensor(buf391, (16, 2048, 768), (1572864, 768, 1), 0); del buf391  # reuse
            buf393 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf394 = reinterpret_tensor(buf393, (16, 2048, 1), (2048, 1, 1), 0); del buf393  # reuse
            buf396 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_71, mul_132, getitem_72, mul_133, x_64, y_32, x_65, rms_norm_44], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_31:121
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_31.run(buf392, buf394, primals_7, buf367, primals_8, buf7, buf396, 32768, 768, stream=stream0)
            buf395 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_77, x_66], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:122
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_83, buf395, 2359296, stream=stream0)
            del primals_83
            buf397 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_44, x_66], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:212
            extern_kernels.mm(reinterpret_tensor(buf396, (32768, 768), (768, 1), 0), buf395, out=buf397)
            buf398 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_78, x_68], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:123
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_84, buf398, 2359296, stream=stream0)
            del primals_84
            buf399 = empty_strided_cuda((16, 2048, 3072), (6291456, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_66, relu_10, x_67], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_pow_relu_9:124
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_pow_relu_9.run(buf397, buf399, 100663296, stream=stream0)
            buf400 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_66, relu_10, x_67, x_68], Original ATen: [aten._unsafe_view, aten.relu, aten.pow, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:213
            extern_kernels.mm(reinterpret_tensor(buf399, (32768, 3072), (3072, 1), 0), buf398, out=buf400)
            buf401 = reinterpret_tensor(buf400, (16, 2048, 768), (1572864, 768, 1), 0); del buf400  # reuse
            buf403 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf404 = reinterpret_tensor(buf403, (16, 2048, 1), (2048, 1, 1), 0); del buf403  # reuse
            buf405 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_68, x_69, getitem_77, mul_144, getitem_78, mul_145, x_70, rms_norm_45], Original ATen: [aten._unsafe_view, aten.add, aten.select, aten.mul, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_32:125
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_32.run(buf401, buf404, buf392, primals_7, primals_8, buf7, buf405, 32768, 768, stream=stream0)
            buf410 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_82, linear_74], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:126
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_88, buf410, 589824, stream=stream0)
            del primals_88
            buf411 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_72, linear_74], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:214
            extern_kernels.mm(reinterpret_tensor(buf405, (32768, 768), (768, 1), 0), buf410, out=buf411)
            buf412 = empty_strided_cuda((12, 6), (1, 12), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_83, linear_75], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_11:127
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_11.run(primals_89, buf412, 72, stream=stream0)
            del primals_89
            buf413 = empty_strided_cuda((32768, 6), (6, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_79, linear_75], Original ATen: [aten.slice, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:215
            extern_kernels.mm(reinterpret_tensor(buf405, (32768, 12), (768, 1), 0), buf412, out=buf413)
            buf402 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            buf414 = reinterpret_tensor(buf411, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0); del buf411  # reuse
            # Topologically Sorted Source Nodes: [embedding_6, linear_74, v_16, ve_11, linear_75, sigmoid_6, gate_6, unsqueeze_5, mul_147, v_17], Original ATen: [aten.embedding, aten._unsafe_view, aten.view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.add]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12:128
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_embedding_mul_sigmoid_unsqueeze_view_12.run(buf414, primals_1, primals_85, buf413, buf402, 25165824, stream=stream0)
            del primals_85
            buf406 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_80, linear_72], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:129
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_86, buf406, 589824, stream=stream0)
            del primals_86
            buf407 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:216
            extern_kernels.mm(reinterpret_tensor(buf405, (32768, 768), (768, 1), 0), buf406, out=buf407)
            buf408 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_81, linear_73], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:130
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_87, buf408, 589824, stream=stream0)
            del primals_87
            buf409 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_72, linear_73], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:217
            extern_kernels.mm(reinterpret_tensor(buf405, (32768, 768), (768, 1), 0), buf408, out=buf409)
            buf415 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf416 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf417 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf419 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            buf418 = reinterpret_tensor(buf417, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf417  # reuse
            buf420 = reinterpret_tensor(buf419, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf419  # reuse
            buf421 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            buf422 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cos, sin, neg, linear_72, q_44, linear_73, k_44, x1_22, x2_22, mul_148, mul_149, y1_22, mul_150, mul_151, y2_22, q_45, x1_23, x2_23, mul_152, mul_153, y1_23, mul_154, mul_155, y2_23, k_45, q_46, k_46, q_47, k_47], Original ATen: [aten.slice, aten.neg, aten._unsafe_view, aten.view, aten.mul, aten.add, aten.cat, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6:131
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_cat_mean_mul_neg_pow_rsqrt_slice_view_6.run(buf418, buf420, buf407, primals_2, primals_3, buf409, buf415, buf416, buf421, buf422, 196608, 128, stream=stream0)
            del buf407
            del buf409
            # Topologically Sorted Source Nodes: [y_33], Original ATen: [flash_attn_3._flash_attn_forward]
            buf423 = torch.ops.flash_attn_3._flash_attn_forward.default(buf421, buf422, buf414, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, attention_chunk=0, softcap=0.0, rotary_interleaved=True, scheduler_metadata=None, num_splits=1, pack_gqa=None, sm_margin=0)
            buf424 = buf423[0]
            assert_size_stride(buf424, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf424, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            buf425 = buf423[1]
            assert_size_stride(buf425, (16, 6, 2048), (12288, 2048, 1), 'torch.ops.flash_attn_3._flash_attn_forward.default')
            assert_alignment(buf425, 16, 'torch.ops.flash_attn_3._flash_attn_forward.default')
            del buf423
            buf428 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_84, y_35], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_5:132
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_5.run(primals_90, buf428, 589824, stream=stream0)
            del primals_90
            buf429 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_34, y_35], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:218
            extern_kernels.mm(reinterpret_tensor(buf424, (32768, 768), (768, 1), 0), buf428, out=buf429)
            buf430 = reinterpret_tensor(buf429, (16, 2048, 768), (1572864, 768, 1), 0); del buf429  # reuse
            buf431 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf432 = reinterpret_tensor(buf431, (16, 2048, 1), (2048, 1, 1), 0); del buf431  # reuse
            buf434 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_77, mul_144, getitem_78, mul_145, x_70, y_35, x_71, rms_norm_48], Original ATen: [aten.select, aten.mul, aten.add, aten._unsafe_view, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_33:133
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_select_33.run(buf430, buf432, primals_7, buf401, primals_8, buf7, buf434, 32768, 768, stream=stream0)
            buf433 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_85, x_72], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:134
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_91, buf433, 2359296, stream=stream0)
            del primals_91
            buf435 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_48, x_72], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:219
            extern_kernels.mm(reinterpret_tensor(buf434, (32768, 768), (768, 1), 0), buf433, out=buf435)
            buf436 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_86, x_74], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_8:135
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_8.run(primals_92, buf436, 2359296, stream=stream0)
            del primals_92
            buf437 = empty_strided_cuda((16, 2048, 3072), (6291456, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_72, relu_11, x_73], Original ATen: [aten._unsafe_view, aten.relu, aten.pow]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_pow_relu_9:136
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_pow_relu_9.run(buf435, buf437, 100663296, stream=stream0)
            buf438 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_72, relu_11, x_73, x_74], Original ATen: [aten._unsafe_view, aten.relu, aten.pow, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:220
            extern_kernels.mm(reinterpret_tensor(buf437, (32768, 3072), (3072, 1), 0), buf436, out=buf438)
            buf439 = reinterpret_tensor(buf438, (16, 2048, 768), (1572864, 768, 1), 0); del buf438  # reuse
            buf440 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf441 = reinterpret_tensor(buf440, (16, 2048, 1), (2048, 1, 1), 0); del buf440  # reuse
            buf443 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_74, x_75, to_87, mul_158, x_76, x_77], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.mul, aten.sub, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_sub_34:137
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_sub_34.run(buf439, buf441, buf430, primals_93, buf257, buf443, 32768, 768, stream=stream0)
            buf442 = empty_strided_cuda((768, 32768), (1, 768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_88, logits], Original ATen: [aten._to_copy, aten.t]
            # [Provenance debug handles] triton_poi_fused__to_copy_t_35:138
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_35.run(primals_94, buf442, 25165824, stream=stream0)
            del primals_94
            buf444 = empty_strided_cuda((32768, 32768), (32768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_77, logits], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:221
            extern_kernels.mm(reinterpret_tensor(buf443, (32768, 768), (768, 1), 0), buf442, out=buf444)
            buf445 = empty_strided_cuda((32768, 1), (1, 1), torch.float32)
            buf446 = empty_strided_cuda((32768, 1), (1, 32768), torch.float32)
            buf447 = reinterpret_tensor(buf446, (32768, 1), (1, 1), 0); del buf446  # reuse
            # Topologically Sorted Source Nodes: [logits, logits_2, truediv, tanh, logits_3, view_54, , loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.view, prims.prepare_softmax_online, aten._log_softmax]
            # [Provenance debug handles] triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view_36:139
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_prepare_softmax_online_tanh_view_36.run(buf447, buf444, buf445, 32768, 32768, stream=stream0)
            buf449 = empty_strided_cuda((4, ), (1, ), torch.int64)
            buf453 = empty_strided_cuda((4, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [logits, logits_2, truediv, tanh, logits_3, view_54, view_55, , loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.view, aten.sub, aten._log_softmax, aten.nll_loss_forward]
            # [Provenance debug handles] triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_37:140
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_37.run(primals_95, buf444, buf445, buf447, buf449, buf453, 4, 8192, stream=stream0)
            buf454 = empty_strided_cuda((), (), torch.float32)
            buf451 = empty_strided_cuda((), (), torch.float32)
            buf455 = buf454; del buf454  # reuse
            # Topologically Sorted Source Nodes: [logits, logits_2, truediv, tanh, logits_3, view_54, view_55, , loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten.view, aten.sub, aten._log_softmax, aten.nll_loss_forward]
            # [Provenance debug handles] triton_per_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_38:141
            stream0 = get_raw_stream(0)
            triton_per_fused__log_softmax__to_copy__unsafe_view_div_mul_nll_loss_forward_sub_tanh_view_38.run(buf455, buf453, buf449, buf451, 1, 4, stream=stream0)
            del buf449
            del buf453
        return (buf455, primals_1, primals_2, primals_3, primals_5, primals_7, primals_8, primals_93, primals_95, buf0, buf2, buf5, buf6, buf7, buf9, reinterpret_tensor(buf11, (32768, 768), (768, 1), 0), reinterpret_tensor(buf16, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), buf17, buf18, buf20, buf22, buf23, buf24, buf26, buf27, buf32, buf34, reinterpret_tensor(buf36, (32768, 768), (768, 1), 0), buf37, reinterpret_tensor(buf39, (32768, 3072), (3072, 1), 0), buf41, buf42, buf44, reinterpret_tensor(buf45, (32768, 768), (768, 1), 0), reinterpret_tensor(buf45, (32768, 12), (768, 1), 0), buf53, buf54, buf55, buf56, buf58, buf60, buf61, buf62, buf64, buf65, buf70, buf72, reinterpret_tensor(buf74, (32768, 768), (768, 1), 0), buf75, reinterpret_tensor(buf77, (32768, 3072), (3072, 1), 0), buf79, buf81, reinterpret_tensor(buf83, (32768, 768), (768, 1), 0), reinterpret_tensor(buf88, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), buf89, buf90, buf92, buf94, buf95, buf96, buf98, buf99, buf104, buf106, reinterpret_tensor(buf108, (32768, 768), (768, 1), 0), buf109, reinterpret_tensor(buf111, (32768, 3072), (3072, 1), 0), buf113, buf114, buf116, reinterpret_tensor(buf117, (32768, 768), (768, 1), 0), reinterpret_tensor(buf117, (32768, 12), (768, 1), 0), buf125, buf126, buf127, buf128, buf130, buf132, buf133, buf134, buf136, buf137, buf142, buf144, reinterpret_tensor(buf146, (32768, 768), (768, 1), 0), buf147, reinterpret_tensor(buf149, (32768, 3072), (3072, 1), 0), buf151, buf153, reinterpret_tensor(buf155, (32768, 768), (768, 1), 0), reinterpret_tensor(buf160, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), buf161, buf162, buf164, buf166, buf167, buf168, buf170, buf171, buf176, buf178, reinterpret_tensor(buf180, (32768, 768), (768, 1), 0), buf181, reinterpret_tensor(buf183, (32768, 3072), (3072, 1), 0), buf185, buf186, buf188, reinterpret_tensor(buf189, (32768, 768), (768, 1), 0), reinterpret_tensor(buf189, (32768, 12), (768, 1), 0), buf197, buf198, buf199, buf200, buf202, buf204, buf205, buf206, buf208, buf209, buf214, buf216, reinterpret_tensor(buf218, (32768, 768), (768, 1), 0), buf219, reinterpret_tensor(buf221, (32768, 3072), (3072, 1), 0), buf223, buf225, reinterpret_tensor(buf227, (32768, 768), (768, 1), 0), reinterpret_tensor(buf232, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), buf233, buf234, buf236, buf238, buf239, buf240, buf242, buf243, buf248, buf250, reinterpret_tensor(buf252, (32768, 768), (768, 1), 0), buf253, reinterpret_tensor(buf255, (32768, 3072), (3072, 1), 0), buf257, buf258, buf260, reinterpret_tensor(buf261, (32768, 768), (768, 1), 0), reinterpret_tensor(buf261, (32768, 12), (768, 1), 0), buf269, buf270, buf271, buf272, buf274, buf276, buf277, buf278, buf280, buf281, buf286, buf288, reinterpret_tensor(buf290, (32768, 768), (768, 1), 0), buf291, reinterpret_tensor(buf293, (32768, 3072), (3072, 1), 0), buf295, buf297, reinterpret_tensor(buf299, (32768, 768), (768, 1), 0), reinterpret_tensor(buf304, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), buf305, buf306, buf308, buf310, buf311, buf312, buf314, buf315, buf320, buf322, reinterpret_tensor(buf324, (32768, 768), (768, 1), 0), buf325, reinterpret_tensor(buf327, (32768, 3072), (3072, 1), 0), buf329, buf330, buf332, reinterpret_tensor(buf333, (32768, 768), (768, 1), 0), reinterpret_tensor(buf333, (32768, 12), (768, 1), 0), buf341, buf342, buf343, buf344, buf346, buf348, buf349, buf350, buf352, buf353, buf358, buf360, reinterpret_tensor(buf362, (32768, 768), (768, 1), 0), buf363, reinterpret_tensor(buf365, (32768, 3072), (3072, 1), 0), buf367, buf369, reinterpret_tensor(buf371, (32768, 768), (768, 1), 0), reinterpret_tensor(buf376, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), buf377, buf378, buf380, buf382, buf383, buf384, buf386, buf387, buf392, buf394, reinterpret_tensor(buf396, (32768, 768), (768, 1), 0), buf397, reinterpret_tensor(buf399, (32768, 3072), (3072, 1), 0), buf401, buf402, buf404, reinterpret_tensor(buf405, (32768, 768), (768, 1), 0), reinterpret_tensor(buf405, (32768, 12), (768, 1), 0), buf413, buf414, buf415, buf416, buf418, buf420, buf421, buf422, buf424, buf425, buf430, buf432, reinterpret_tensor(buf434, (32768, 768), (768, 1), 0), buf435, reinterpret_tensor(buf437, (32768, 3072), (3072, 1), 0), buf439, buf441, reinterpret_tensor(buf443, (32768, 768), (768, 1), 0), buf444, buf445, buf447, buf451, reinterpret_tensor(buf442, (32768, 768), (768, 1), 0), reinterpret_tensor(buf436, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf433, (3072, 768), (768, 1), 0), reinterpret_tensor(buf428, (768, 768), (768, 1), 0), reinterpret_tensor(buf412, (6, 12), (12, 1), 0), reinterpret_tensor(buf410, (768, 768), (768, 1), 0), reinterpret_tensor(buf408, (768, 768), (768, 1), 0), reinterpret_tensor(buf406, (768, 768), (768, 1), 0), reinterpret_tensor(buf398, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf395, (3072, 768), (768, 1), 0), reinterpret_tensor(buf390, (768, 768), (768, 1), 0), reinterpret_tensor(buf375, (768, 768), (768, 1), 0), reinterpret_tensor(buf373, (768, 768), (768, 1), 0), reinterpret_tensor(buf370, (768, 768), (768, 1), 0), reinterpret_tensor(buf364, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf361, (3072, 768), (768, 1), 0), reinterpret_tensor(buf356, (768, 768), (768, 1), 0), reinterpret_tensor(buf340, (6, 12), (12, 1), 0), reinterpret_tensor(buf338, (768, 768), (768, 1), 0), reinterpret_tensor(buf336, (768, 768), (768, 1), 0), reinterpret_tensor(buf334, (768, 768), (768, 1), 0), reinterpret_tensor(buf326, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf323, (3072, 768), (768, 1), 0), reinterpret_tensor(buf318, (768, 768), (768, 1), 0), reinterpret_tensor(buf303, (768, 768), (768, 1), 0), reinterpret_tensor(buf301, (768, 768), (768, 1), 0), reinterpret_tensor(buf298, (768, 768), (768, 1), 0), reinterpret_tensor(buf292, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf289, (3072, 768), (768, 1), 0), reinterpret_tensor(buf284, (768, 768), (768, 1), 0), reinterpret_tensor(buf268, (6, 12), (12, 1), 0), reinterpret_tensor(buf266, (768, 768), (768, 1), 0), reinterpret_tensor(buf264, (768, 768), (768, 1), 0), reinterpret_tensor(buf262, (768, 768), (768, 1), 0), reinterpret_tensor(buf254, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf251, (3072, 768), (768, 1), 0), reinterpret_tensor(buf246, (768, 768), (768, 1), 0), reinterpret_tensor(buf231, (768, 768), (768, 1), 0), reinterpret_tensor(buf229, (768, 768), (768, 1), 0), reinterpret_tensor(buf226, (768, 768), (768, 1), 0), reinterpret_tensor(buf220, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf217, (3072, 768), (768, 1), 0), reinterpret_tensor(buf212, (768, 768), (768, 1), 0), reinterpret_tensor(buf196, (6, 12), (12, 1), 0), reinterpret_tensor(buf194, (768, 768), (768, 1), 0), reinterpret_tensor(buf192, (768, 768), (768, 1), 0), reinterpret_tensor(buf190, (768, 768), (768, 1), 0), reinterpret_tensor(buf182, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf179, (3072, 768), (768, 1), 0), reinterpret_tensor(buf174, (768, 768), (768, 1), 0), reinterpret_tensor(buf159, (768, 768), (768, 1), 0), reinterpret_tensor(buf157, (768, 768), (768, 1), 0), reinterpret_tensor(buf154, (768, 768), (768, 1), 0), reinterpret_tensor(buf148, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf145, (3072, 768), (768, 1), 0), reinterpret_tensor(buf140, (768, 768), (768, 1), 0), reinterpret_tensor(buf124, (6, 12), (12, 1), 0), reinterpret_tensor(buf122, (768, 768), (768, 1), 0), reinterpret_tensor(buf120, (768, 768), (768, 1), 0), reinterpret_tensor(buf118, (768, 768), (768, 1), 0), reinterpret_tensor(buf110, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf107, (3072, 768), (768, 1), 0), reinterpret_tensor(buf102, (768, 768), (768, 1), 0), reinterpret_tensor(buf87, (768, 768), (768, 1), 0), reinterpret_tensor(buf85, (768, 768), (768, 1), 0), reinterpret_tensor(buf82, (768, 768), (768, 1), 0), reinterpret_tensor(buf76, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf73, (3072, 768), (768, 1), 0), reinterpret_tensor(buf68, (768, 768), (768, 1), 0), reinterpret_tensor(buf52, (6, 12), (12, 1), 0), reinterpret_tensor(buf50, (768, 768), (768, 1), 0), reinterpret_tensor(buf48, (768, 768), (768, 1), 0), reinterpret_tensor(buf46, (768, 768), (768, 1), 0), reinterpret_tensor(buf38, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf35, (3072, 768), (768, 1), 0), reinterpret_tensor(buf30, (768, 768), (768, 1), 0), reinterpret_tensor(buf15, (768, 768), (768, 1), 0), reinterpret_tensor(buf13, (768, 768), (768, 1), 0), reinterpret_tensor(buf10, (768, 768), (768, 1), 0), reinterpret_tensor(buf3, (1, 24), (24, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((1, 20480, 1, 64), (1310720, 64, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((1, 20480, 1, 64), (1310720, 64, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_16 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((6, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_30 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((6, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_44 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((6, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_58 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((6, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_72 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((6, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_86 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((6, 12), (12, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((16, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
