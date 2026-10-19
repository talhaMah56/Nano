# AOT ID: ['6_inference']
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


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/jl/cjlvgcbw7emci6hvcj26otkjrwze3jwshrsnrjf2s74w633gfmxd.py
# Topologically Sorted Source Nodes: [g, sub, lerp_, X, float_1, norm], Original ATen: [aten.lerp, aten.rsub, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X => convert_element_type
#   float_1 => convert_element_type_1
#   g => abs_2, add_1, ge_1, mul_1, sub_3, sub_4, where_2, where_3
#   lerp_ => abs_1, add, ge, mul, sub_1, sub_2, where, where_1
#   norm => pow_1, sum_1
#   sub => sub
# Graph fragment:
#   %arg0_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg0_1]
#   %copy_ : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=copy_]
#   %copy__1 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=copy__1]
#   %abs_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg0_1,), kwargs = {})
#   %ge_1 : Tensor "b8[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_2, 0.5), kwargs = {})
#   %sub_3 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, 1), kwargs = {})
#   %where_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %sub_3, %arg0_1), kwargs = {})
#   %sub : Tensor "f32[][]cpu"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg0_1), kwargs = {})
#   %abs_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %ge : Tensor "b8[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %sub_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, 1), kwargs = {})
#   %where : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %sub_1, %sub), kwargs = {})
#   %sub_2 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %mul : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %sub_2), kwargs = {})
#   %where_1 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %arg1_1, %arg2_1), kwargs = {})
#   %add : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %where_1), kwargs = {})
#   %sub_4 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %arg1_1), kwargs = {})
#   %mul_1 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_2, %sub_4), kwargs = {})
#   %where_3 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %add, %arg1_1), kwargs = {})
#   %add_1 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %where_3), kwargs = {})
#   %convert_element_type : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1, 2), kwargs = {})
#   %sum_1 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [-2, -1], True), kwargs = {})
#   return %sum_1
triton_per_fused__to_copy_lerp_linalg_vector_norm_rsub_0 = async_compile.triton('triton_per_fused__to_copy_lerp_linalg_vector_norm_rsub_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': 'fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_lerp_linalg_vector_norm_rsub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36, 'r0_': 3456}}
)
@triton.jit
def triton_per_fused__to_copy_lerp_linalg_vector_norm_rsub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 6
    r0_numel = 72
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = in_ptr0
    tmp12 = tl.load(in_ptr1 + (r0_1 + 72*x0), r0_mask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr2 + (r0_1 + 72*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 0.5
    tmp3 = tmp1 >= tmp2
    tmp4 = 1.0
    tmp5 = tmp0 - tmp4
    tmp6 = tl.where(tmp3, tmp5, tmp0)
    tmp7 = tmp4 - tmp0
    tmp8 = tl_math.abs(tmp7)
    tmp9 = tmp8 >= tmp2
    tmp10 = tmp7 - tmp4
    tmp11 = tl.where(tmp9, tmp10, tmp7)
    tmp14 = tmp12 - tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp9, tmp12, tmp13)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17 - tmp12
    tmp19 = tmp6 * tmp18
    tmp20 = tl.where(tmp3, tmp17, tmp12)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(r0_mask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp28, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/bp/cbpmgfo4wvd7uq4rbylsdpdiv637us3xhvlsanafnar6tjwug42g.py
# Topologically Sorted Source Nodes: [g, sub, lerp_, X, norm, target, float_2, norm_1, row_norm, truediv_1, to_1, X_1], Original ATen: [aten.lerp, aten.rsub, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.clamp_min, aten.mul, aten.copy_]
# Source node to ATen node mapping:
#   X => convert_element_type
#   X_1 => mul_2
#   float_2 => convert_element_type_2
#   g => abs_2, add_1, ge_1, mul_1, sub_3, sub_4, where_2, where_3
#   lerp_ => abs_1, add, ge, mul, sub_1, sub_2, where, where_1
#   norm => pow_2
#   norm_1 => pow_3, pow_4, sum_2
#   row_norm => clamp_min
#   sub => sub
#   target => div
#   to_1 => convert_element_type_3
#   truediv_1 => div_1
# Graph fragment:
#   %arg0_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg0_1]
#   %copy_ : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=copy_]
#   %copy__1 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=copy__1]
#   %sum_1 : Tensor "f32[6, 1, 1][1, 6, 6]cuda:0" = PlaceHolder[target=sum_1]
#   %sum_2 : Tensor "f32[6, 6, 1][6, 1, 36]cuda:0" = PlaceHolder[target=sum_2]
#   %add_1 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=add_1]
#   %mul_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=mul_2]
#   %add : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=add]
#   %abs_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%arg0_1,), kwargs = {})
#   %ge_1 : Tensor "b8[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_2, 0.5), kwargs = {})
#   %sub_3 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, 1), kwargs = {})
#   %where_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %sub_3, %arg0_1), kwargs = {})
#   %sub : Tensor "f32[][]cpu"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg0_1), kwargs = {})
#   %abs_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %ge : Tensor "b8[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %sub_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, 1), kwargs = {})
#   %where : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %sub_1, %sub), kwargs = {})
#   %sub_2 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %mul : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %sub_2), kwargs = {})
#   %where_1 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %arg1_1, %arg2_1), kwargs = {})
#   %add : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %where_1), kwargs = {})
#   %sub_4 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %arg1_1), kwargs = {})
#   %mul_1 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_2, %sub_4), kwargs = {})
#   %where_3 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %add, %arg1_1), kwargs = {})
#   %add_1 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %where_3), kwargs = {})
#   %convert_element_type : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %pow_2 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %div : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_2, 2.449489742783178), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_2, 2), kwargs = {})
#   %sum_2 : Tensor "f32[6, 6, 1][6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [-1], True), kwargs = {})
#   %pow_4 : Tensor "f32[6, 6, 1][6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %clamp_min : Tensor "f32[6, 6, 1][6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_4, 1e-06), kwargs = {})
#   %div_1 : Tensor "f32[6, 6, 1][6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, %clamp_min), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[6, 6, 1][6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.bfloat16), kwargs = {})
#   %mul_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %convert_element_type_3), kwargs = {})
#   %copy_ : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg1_1, %add_1), kwargs = {})
#   %copy__1 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg2_1, %add), kwargs = {})
#   return %sum_2,%mul_2,%add,%add_1,%buf48,%buf49
triton_per_fused__to_copy_clamp_min_copy__div_lerp_linalg_vector_norm_mul_rsub_1 = async_compile.triton('triton_per_fused__to_copy_clamp_min_copy__div_lerp_linalg_vector_norm_mul_rsub_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': 'fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*bf16', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_clamp_min_copy__div_lerp_linalg_vector_norm_mul_rsub_1', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr5'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 15552}}
)
@triton.jit
def triton_per_fused__to_copy_clamp_min_copy__div_lerp_linalg_vector_norm_mul_rsub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 36
    r0_numel = 12
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x3 = xindex // 6
    tmp0 = in_ptr0
    tmp12 = tl.load(in_ptr1 + (r0_1 + 12*x0), r0_mask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr2 + (r0_1 + 12*x0), r0_mask & xmask, other=0.0)
    tmp29 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl_math.abs(tmp0)
    tmp2 = 0.5
    tmp3 = tmp1 >= tmp2
    tmp4 = 1.0
    tmp5 = tmp0 - tmp4
    tmp6 = tl.where(tmp3, tmp5, tmp0)
    tmp7 = tmp4 - tmp0
    tmp8 = tl_math.abs(tmp7)
    tmp9 = tmp8 >= tmp2
    tmp10 = tmp7 - tmp4
    tmp11 = tl.where(tmp9, tmp10, tmp7)
    tmp14 = tmp12 - tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp9, tmp12, tmp13)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17 - tmp12
    tmp19 = tmp6 * tmp18
    tmp20 = tl.where(tmp3, tmp17, tmp12)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(r0_mask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp30 = libdevice.sqrt(tmp29)
    tmp31 = 0.4082482904638631
    tmp32 = tmp30 * tmp31
    tmp33 = libdevice.sqrt(tmp28)
    tmp34 = 1e-06
    tmp35 = triton_helpers.maximum(tmp33, tmp34)
    tmp36 = (tmp32 / tmp35)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp22 * tmp37
    tl.store(out_ptr1 + (r0_1 + 12*x0), tmp38, r0_mask & xmask)
    tl.store(out_ptr4 + (r0_1 + 12*x0), tmp21, r0_mask & xmask)
    tl.store(out_ptr5 + (r0_1 + 12*x0), tmp17, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/zv/czvidtludvzeyyvejzdc3n37ylhwtihdxdpftlhzlwsnwfb7kpbn.py
# Topologically Sorted Source Nodes: [norm_2, mul_1, add, X_2], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div]
# Source node to ATen node mapping:
#   X_2 => div_2
#   add => add_2
#   mul_1 => mul_3
#   norm_2 => convert_element_type_4, convert_element_type_5, pow_5, pow_6, sum_3
# Graph fragment:
#   %mul_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=mul_2]
#   %sum_3 : Tensor "f32[6, 1, 1][1, 6, 6]cuda:0" = PlaceHolder[target=sum_3]
#   %convert_element_type_4 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.float32), kwargs = {})
#   %pow_5 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_4, 2), kwargs = {})
#   %sum_3 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [-2, -1], True), kwargs = {})
#   %pow_6 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_6, torch.bfloat16), kwargs = {})
#   %mul_3 : Tensor "bf16[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, 1.01), kwargs = {})
#   %add_2 : Tensor "bf16[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 1e-06), kwargs = {})
#   %div_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, %add_2), kwargs = {})
#   return %sum_3,%expand_5
triton_per_fused_add_div_linalg_vector_norm_mul_2 = async_compile.triton('triton_per_fused_add_div_linalg_vector_norm_mul_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_linalg_vector_norm_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36, 'r0_': 2592}}
)
@triton.jit
def triton_per_fused_add_div_linalg_vector_norm_mul_2(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 6
    r0_numel = 72
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1.01
    tmp10 = tmp8 * tmp9
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = (tmp0 / tmp12)
    tl.store(out_ptr1 + (r0_1 + 72*x0), tmp13, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/7g/c7gg4o4bs36ykugtrll6ipp4bazxmythiul6rld3liprfjdyiakk.py
# Topologically Sorted Source Nodes: [mul_2, mul_3, B], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B => add_3
#   mul_2 => mul_4
#   mul_3 => mul_5
# Graph fragment:
#   %expand_3 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0" = PlaceHolder[target=expand_3]
#   %bmm_1 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %mul_4 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm, -22.48329292557795), kwargs = {})
#   %mul_5 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_1, 15.878769915207462), kwargs = {})
#   %add_3 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   return %expand_4
triton_poi_fused_add_mul_3 = async_compile.triton('triton_poi_fused_add_mul_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = -22.48329292557795
    tmp2 = tmp0 * tmp1
    tmp4 = 15.878769915207462
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/jd/cjdmbqbtydg6fk32j4cbdb545haooehf3zcebo2oqhneivenscns.py
# Topologically Sorted Source Nodes: [norm_2, mul_1, add, X_2, mul_4, X_3], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div]
# Source node to ATen node mapping:
#   X_2 => div_2
#   X_3 => add_4
#   add => add_2
#   mul_1 => mul_3
#   mul_4 => mul_6
#   norm_2 => convert_element_type_5, pow_6
# Graph fragment:
#   %mul_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=mul_2]
#   %sum_3 : Tensor "f32[6, 1, 1][1, 6, 6]cuda:0" = PlaceHolder[target=sum_3]
#   %bmm_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %pow_6 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_6, torch.bfloat16), kwargs = {})
#   %mul_3 : Tensor "bf16[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, 1.01), kwargs = {})
#   %add_2 : Tensor "bf16[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 1e-06), kwargs = {})
#   %div_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, %add_2), kwargs = {})
#   %mul_6 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, 8.156554524902461), kwargs = {})
#   %add_4 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %bmm_2), kwargs = {})
#   return %expand_11
triton_poi_fused_add_div_linalg_vector_norm_mul_4 = async_compile.triton('triton_poi_fused_add_div_linalg_vector_norm_mul_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_linalg_vector_norm_mul_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_linalg_vector_norm_mul_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 72
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2), xmask).to(tl.float32)
    tmp2 = libdevice.sqrt(tmp1)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.01
    tmp5 = tmp3 * tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = (tmp0 / tmp7)
    tmp9 = 8.156554524902461
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + (x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/fr/cfrdwgfupds5m6ripgy2hokhqrldrzpsyvtxofhu2vyqs2uoz6v5.py
# Topologically Sorted Source Nodes: [mul_5, mul_6, B_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_1 => add_5
#   mul_5 => mul_7
#   mul_6 => mul_8
# Graph fragment:
#   %expand_9 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0" = PlaceHolder[target=expand_9]
#   %bmm_4 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %mul_7 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, -2.808917465908714), kwargs = {})
#   %mul_8 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_4, 0.5000178451051316), kwargs = {})
#   %add_5 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %mul_8), kwargs = {})
#   return %expand_10
triton_poi_fused_add_mul_5 = async_compile.triton('triton_poi_fused_add_mul_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = -2.808917465908714
    tmp2 = tmp0 * tmp1
    tmp4 = 0.5000178451051316
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/43/c43adrv6orjidfzujn5wvwtmhqrheic4azsyp6wz4fl625urjp22.py
# Topologically Sorted Source Nodes: [norm_2, mul_1, add, X_2, mul_4, X_3, mul_7, X_4], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div]
# Source node to ATen node mapping:
#   X_2 => div_2
#   X_3 => add_4
#   X_4 => add_6
#   add => add_2
#   mul_1 => mul_3
#   mul_4 => mul_6
#   mul_7 => mul_9
#   norm_2 => convert_element_type_5, pow_6
# Graph fragment:
#   %mul_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=mul_2]
#   %sum_3 : Tensor "f32[6, 1, 1][1, 6, 6]cuda:0" = PlaceHolder[target=sum_3]
#   %bmm_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %bmm_5 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %pow_6 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_6, torch.bfloat16), kwargs = {})
#   %mul_3 : Tensor "bf16[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, 1.01), kwargs = {})
#   %add_2 : Tensor "bf16[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 1e-06), kwargs = {})
#   %div_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, %add_2), kwargs = {})
#   %mul_6 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, 8.156554524902461), kwargs = {})
#   %add_4 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %bmm_2), kwargs = {})
#   %mul_9 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 4.042929935166739), kwargs = {})
#   %add_6 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %bmm_5), kwargs = {})
#   return %expand_17
triton_poi_fused_add_div_linalg_vector_norm_mul_6 = async_compile.triton('triton_poi_fused_add_div_linalg_vector_norm_mul_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_linalg_vector_norm_mul_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_linalg_vector_norm_mul_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 72
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2), xmask).to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (x2), xmask).to(tl.float32)
    tmp2 = libdevice.sqrt(tmp1)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.01
    tmp5 = tmp3 * tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = (tmp0 / tmp7)
    tmp9 = 8.156554524902461
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = 4.042929935166739
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/on/coniirvc3jtixqj7vzeqj2quqn3k7pptzho232jmwrmqsahcehqy.py
# Topologically Sorted Source Nodes: [mul_8, mul_9, B_2], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_2 => add_7
#   mul_8 => mul_10
#   mul_9 => mul_11
# Graph fragment:
#   %expand_15 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0" = PlaceHolder[target=expand_15]
#   %bmm_7 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %mul_10 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_6, -2.772484153217685), kwargs = {})
#   %mul_11 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_7, 0.5060648178503393), kwargs = {})
#   %add_7 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %mul_11), kwargs = {})
#   return %expand_16
triton_poi_fused_add_mul_7 = async_compile.triton('triton_poi_fused_add_mul_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = -2.772484153217685
    tmp2 = tmp0 * tmp1
    tmp4 = 0.5060648178503393
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/vu/cvun76gqrrk5irn5uixxvyohcq6krdfggtjmupag3ast5vyp64s2.py
# Topologically Sorted Source Nodes: [norm_2, mul_1, add, X_2, mul_4, X_3, mul_7, X_4, mul_10, X_5], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div]
# Source node to ATen node mapping:
#   X_2 => div_2
#   X_3 => add_4
#   X_4 => add_6
#   X_5 => add_8
#   add => add_2
#   mul_1 => mul_3
#   mul_10 => mul_12
#   mul_4 => mul_6
#   mul_7 => mul_9
#   norm_2 => convert_element_type_5, pow_6
# Graph fragment:
#   %mul_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=mul_2]
#   %sum_3 : Tensor "f32[6, 1, 1][1, 6, 6]cuda:0" = PlaceHolder[target=sum_3]
#   %bmm_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %bmm_5 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_5]
#   %bmm_8 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %pow_6 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_6, torch.bfloat16), kwargs = {})
#   %mul_3 : Tensor "bf16[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, 1.01), kwargs = {})
#   %add_2 : Tensor "bf16[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 1e-06), kwargs = {})
#   %div_2 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, %add_2), kwargs = {})
#   %mul_6 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, 8.156554524902461), kwargs = {})
#   %add_4 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %bmm_2), kwargs = {})
#   %mul_9 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, 4.042929935166739), kwargs = {})
#   %add_6 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %bmm_5), kwargs = {})
#   %mul_12 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 3.8916678022926607), kwargs = {})
#   %add_8 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %bmm_8), kwargs = {})
#   return %expand_23
triton_poi_fused_add_div_linalg_vector_norm_mul_8 = async_compile.triton('triton_poi_fused_add_div_linalg_vector_norm_mul_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_linalg_vector_norm_mul_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_linalg_vector_norm_mul_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 72
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (x2), xmask).to(tl.float32)
    tmp19 = tl.load(in_ptr3 + (x2), xmask).to(tl.float32)
    tmp2 = libdevice.sqrt(tmp1)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.01
    tmp5 = tmp3 * tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = (tmp0 / tmp7)
    tmp9 = 8.156554524902461
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = 4.042929935166739
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 3.8916678022926607
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tl.store(in_out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/ox/coxnmax6kznmcypddw3qcmnt5egbrmblvvuh2utz6ub4myw4ujqf.py
# Topologically Sorted Source Nodes: [mul_11, mul_12, B_3], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_3 => add_9
#   mul_11 => mul_13
#   mul_12 => mul_14
# Graph fragment:
#   %expand_21 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0" = PlaceHolder[target=expand_21]
#   %bmm_10 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0" = PlaceHolder[target=bmm_10]
#   %mul_13 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, -2.3681294933425376), kwargs = {})
#   %mul_14 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_10, 0.46449024233003106), kwargs = {})
#   %add_9 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_14), kwargs = {})
#   return %expand_22
triton_poi_fused_add_mul_9 = async_compile.triton('triton_poi_fused_add_mul_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = -2.3681294933425376
    tmp2 = tmp0 * tmp1
    tmp4 = 0.46449024233003106
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/27/c27lm2tedq6d5y45vm7mbq4cqurme47vyumdzmak6xzbimtj6hcl.py
# Topologically Sorted Source Nodes: [mul_13, X_6], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   X_6 => add_10
#   mul_13 => mul_15
# Graph fragment:
#   %expand_23 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=expand_23]
#   %bmm_11 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %mul_15 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 3.285753657755655), kwargs = {})
#   %add_10 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %bmm_11), kwargs = {})
#   return %expand_29
triton_poi_fused_add_mul_10 = async_compile.triton('triton_poi_fused_add_mul_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp1 = 3.285753657755655
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/c3/cc3jbjzt6zjwxrj5da5bjlfasp5ghh52wvphl4soovk37hwlhp62.py
# Topologically Sorted Source Nodes: [mul_14, mul_15, B_4], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   B_4 => add_11
#   mul_14 => mul_16
#   mul_15 => mul_17
# Graph fragment:
#   %expand_27 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0" = PlaceHolder[target=expand_27]
#   %bmm_13 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0" = PlaceHolder[target=bmm_13]
#   %mul_16 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_12, -1.7097828382687081), kwargs = {})
#   %mul_17 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_13, 0.42323551169305323), kwargs = {})
#   %add_11 : Tensor "bf16[6, 6, 6][36, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %mul_17), kwargs = {})
#   return %expand_28
triton_poi_fused_add_mul_11 = async_compile.triton('triton_poi_fused_add_mul_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = -1.7097828382687081
    tmp2 = tmp0 * tmp1
    tmp4 = 0.42323551169305323
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/vx/cvxycjmklvqzcfummng6sypyfdfsdpz65ctqn4tortho2b6w42er.py
# Topologically Sorted Source Nodes: [mul_13, X_6, mul_16, X_7, g_1, norm_3], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X_6 => add_10
#   X_7 => add_12
#   g_1 => convert_element_type_36
#   mul_13 => mul_15
#   mul_16 => mul_18
#   norm_3 => pow_7, sum_4
# Graph fragment:
#   %expand_23 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=expand_23]
#   %bmm_11 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %bmm_14 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_14]
#   %mul_15 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 3.285753657755655), kwargs = {})
#   %add_10 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %bmm_11), kwargs = {})
#   %mul_18 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 2.3465413258596377), kwargs = {})
#   %add_12 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %bmm_14), kwargs = {})
#   %convert_element_type_36 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %pow_7 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_36, 2), kwargs = {})
#   %sum_4 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [-2, -1], True), kwargs = {})
#   return %sum_4
triton_per_fused__to_copy_add_linalg_vector_norm_mul_12 = async_compile.triton('triton_per_fused__to_copy_add_linalg_vector_norm_mul_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_linalg_vector_norm_mul_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36, 'r0_': 2592}}
)
@triton.jit
def triton_per_fused__to_copy_add_linalg_vector_norm_mul_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 6
    r0_numel = 72
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = 3.285753657755655
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 2.3465413258596377
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/tg/ctg4ek3swnymapxxixhb4m5u6surarlnh5stu7setb5aokigngx3.py
# Topologically Sorted Source Nodes: [mul_13, X_6, mul_16, X_7, g_1, norm_3, current_norm, truediv_3, g_2, square, v_mean, sum_1, v_norm_sq, v_norm, mul_19, sub_1, lerp__2, clamp_min_2, step_size, square_1, scaled_sq_sum, sum_2, v_norm_new, clamp_min_3, truediv_4, final_scale], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.reciprocal, aten.pow, aten.mean, aten.sum, aten.sqrt, aten.rsub, aten.lerp, aten.rsqrt, aten.div, aten.copy_]
# Source node to ATen node mapping:
#   X_6 => add_10
#   X_7 => add_12
#   clamp_min_2 => clamp_min_2
#   clamp_min_3 => clamp_min_3
#   current_norm => clamp_min_1
#   final_scale => mul_25
#   g_1 => convert_element_type_36
#   g_2 => mul_20
#   lerp__2 => abs_3, add_13, ge_2, mul_22, sub_6, sub_7, where_4, where_5
#   mul_13 => mul_15
#   mul_16 => mul_18
#   mul_19 => mul_23
#   norm_3 => pow_8
#   scaled_sq_sum => mul_24
#   square => pow_9
#   square_1 => pow_10
#   step_size => rsqrt
#   sub_1 => sub_5
#   sum_1 => sum_5
#   sum_2 => sum_6
#   truediv_3 => mul_19, reciprocal
#   truediv_4 => div_3
#   v_mean => mean
#   v_norm => sqrt
#   v_norm_new => sqrt_1
#   v_norm_sq => mul_21
# Graph fragment:
#   %expand_23 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=expand_23]
#   %bmm_11 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %bmm_14 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_14]
#   %sum_4 : Tensor "f32[6, 1, 1][1, 6, 6]cuda:0" = PlaceHolder[target=sum_4]
#   %mean : Tensor "f32[6, 1, 12][12, 72, 1]cuda:0" = PlaceHolder[target=mean]
#   %arg4_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg4_1]
#   %copy__3 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0" = PlaceHolder[target=copy__3]
#   %sum_5 : Tensor "f32[6, 1, 1][1, 6, 6]cuda:0" = PlaceHolder[target=sum_5]
#   %sum_6 : Tensor "f32[6, 1, 1][1, 6, 6]cuda:0" = PlaceHolder[target=sum_6]
#   %add_13 : Tensor "f32[6, 1, 12][12, 72, 1]cuda:0" = PlaceHolder[target=add_13]
#   %mul_25 : Tensor "f32[6, 1, 12][12, 72, 1]cuda:0" = PlaceHolder[target=mul_25]
#   %mul_15 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 3.285753657755655), kwargs = {})
#   %add_10 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %bmm_11), kwargs = {})
#   %mul_18 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 2.3465413258596377), kwargs = {})
#   %add_12 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %bmm_14), kwargs = {})
#   %convert_element_type_36 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %pow_8 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %clamp_min_1 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_8, 1e-06), kwargs = {})
#   %reciprocal : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_1,), kwargs = {})
#   %mul_19 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 2.449489742783178), kwargs = {})
#   %mul_20 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_36, %mul_19), kwargs = {})
#   %pow_9 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_20, 2), kwargs = {})
#   %mean : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [-2], True), kwargs = {})
#   %sum_5 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mean, [-2, -1], True), kwargs = {})
#   %mul_21 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_5, 6), kwargs = {})
#   %sqrt : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mul_21,), kwargs = {})
#   %mul_23 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 6), kwargs = {})
#   %sub_5 : Tensor "f32[][]cpu"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg4_1), kwargs = {})
#   %abs_3 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_5,), kwargs = {})
#   %ge_2 : Tensor "b8[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_3, 0.5), kwargs = {})
#   %sub_6 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_5, 1), kwargs = {})
#   %where_4 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_2, %sub_6, %sub_5), kwargs = {})
#   %sub_7 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean, %arg5_1), kwargs = {})
#   %mul_22 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_4, %sub_7), kwargs = {})
#   %where_5 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_2, %mean, %arg5_1), kwargs = {})
#   %add_13 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %where_5), kwargs = {})
#   %clamp_min_2 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_13, 1e-10), kwargs = {})
#   %rsqrt : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%clamp_min_2,), kwargs = {})
#   %pow_10 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%rsqrt, 2), kwargs = {})
#   %mul_24 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %pow_10), kwargs = {})
#   %sum_6 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_24, [-2, -1], True), kwargs = {})
#   %sqrt_1 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_6,), kwargs = {})
#   %clamp_min_3 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sqrt_1, 1e-10), kwargs = {})
#   %div_3 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sqrt, %clamp_min_3), kwargs = {})
#   %mul_25 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%rsqrt, %div_3), kwargs = {})
#   %copy__3 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg5_1, %add_13), kwargs = {})
#   return %mean,%sum_5,%sum_6,%mul_25,%add_13,%buf65
triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_linalg_vector_norm_mean_mul_pow_reciprocal_rsqrt_rsub_sqrt_sum_13 = async_compile.triton('triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_linalg_vector_norm_mean_mul_pow_reciprocal_rsqrt_rsub_sqrt_sum_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': 'fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_linalg_vector_norm_mean_mul_pow_reciprocal_rsqrt_rsub_sqrt_sum_13', 'mutated_arg_names': ['in_ptr5', 'out_ptr5'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 21, 'num_reduction': 2, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18, 'r0_': 4032}}
)
@triton.jit
def triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_linalg_vector_norm_mean_mul_pow_reciprocal_rsqrt_rsub_sqrt_sum_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 6
    r0_numel = 12
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (12 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr1 + (12 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr2 + (12 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr0 + (24 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr1 + (24 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr2 + (24 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp42 = tl.load(in_ptr0 + (36 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp44 = tl.load(in_ptr1 + (36 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp47 = tl.load(in_ptr2 + (36 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp53 = tl.load(in_ptr0 + (48 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp55 = tl.load(in_ptr1 + (48 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp58 = tl.load(in_ptr2 + (48 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp64 = tl.load(in_ptr0 + (60 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp66 = tl.load(in_ptr1 + (60 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp69 = tl.load(in_ptr2 + (60 + r0_1 + 72*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp82 = in_ptr4
    tmp90 = tl.load(in_ptr5 + (r0_1 + 12*x0), r0_mask & xmask, other=0.0)
    tmp1 = 3.285753657755655
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 2.3465413258596377
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = 1e-06
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = tl.full([1, 1], 1, tl.int32)
    tmp15 = (tmp14 / tmp13)
    tmp16 = 2.449489742783178
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 * tmp17
    tmp19 = tmp18 * tmp18
    tmp21 = tmp20 * tmp1
    tmp23 = tmp21 + tmp22
    tmp24 = tmp23 * tmp5
    tmp26 = tmp24 + tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp17
    tmp29 = tmp28 * tmp28
    tmp30 = tmp19 + tmp29
    tmp32 = tmp31 * tmp1
    tmp34 = tmp32 + tmp33
    tmp35 = tmp34 * tmp5
    tmp37 = tmp35 + tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp38 * tmp17
    tmp40 = tmp39 * tmp39
    tmp41 = tmp30 + tmp40
    tmp43 = tmp42 * tmp1
    tmp45 = tmp43 + tmp44
    tmp46 = tmp45 * tmp5
    tmp48 = tmp46 + tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp49 * tmp17
    tmp51 = tmp50 * tmp50
    tmp52 = tmp41 + tmp51
    tmp54 = tmp53 * tmp1
    tmp56 = tmp54 + tmp55
    tmp57 = tmp56 * tmp5
    tmp59 = tmp57 + tmp58
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp60 * tmp17
    tmp62 = tmp61 * tmp61
    tmp63 = tmp52 + tmp62
    tmp65 = tmp64 * tmp1
    tmp67 = tmp65 + tmp66
    tmp68 = tmp67 * tmp5
    tmp70 = tmp68 + tmp69
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tmp71 * tmp17
    tmp73 = tmp72 * tmp72
    tmp74 = tmp63 + tmp73
    tmp75 = 6.0
    tmp76 = (tmp74 / tmp75)
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, R0_BLOCK])
    tmp79 = tl.where(r0_mask & xmask, tmp77, 0)
    tmp80 = tl.sum(tmp79, 1)[:, None].to(tl.float32)
    tmp81 = tmp76 * tmp75
    tmp83 = 1.0
    tmp84 = tmp83 - tmp82
    tmp85 = tl_math.abs(tmp84)
    tmp86 = 0.5
    tmp87 = tmp85 >= tmp86
    tmp88 = tmp84 - tmp83
    tmp89 = tl.where(tmp87, tmp88, tmp84)
    tmp91 = tmp76 - tmp90
    tmp92 = tmp89 * tmp91
    tmp93 = tl.where(tmp87, tmp76, tmp90)
    tmp94 = tmp92 + tmp93
    tmp95 = 1e-10
    tmp96 = triton_helpers.maximum(tmp94, tmp95)
    tmp97 = libdevice.rsqrt(tmp96)
    tmp98 = tmp97 * tmp97
    tmp99 = tmp81 * tmp98
    tmp100 = tl.broadcast_to(tmp99, [XBLOCK, R0_BLOCK])
    tmp102 = tl.where(r0_mask & xmask, tmp100, 0)
    tmp103 = tl.sum(tmp102, 1)[:, None].to(tl.float32)
    tmp104 = tmp80 * tmp75
    tmp105 = libdevice.sqrt(tmp104)
    tmp106 = libdevice.sqrt(tmp103)
    tmp107 = triton_helpers.maximum(tmp106, tmp95)
    tmp108 = (tmp105 / tmp107)
    tmp109 = tmp97 * tmp108
    tl.store(out_ptr3 + (r0_1 + 12*x0), tmp109, r0_mask & xmask)
    tl.store(out_ptr5 + (r0_1 + 12*x0), tmp94, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpd6v5fk0o/6d/c6diwx7eiqdw3fmohn5v4tsygtr2h3pdhhrnaoszfvggv4nfuvxo.py
# Topologically Sorted Source Nodes: [mul_13, X_6, mul_16, X_7, g_1, norm_3, current_norm, truediv_3, g_2, v_norm_sq, v_norm, sub_1, lerp__2, clamp_min_2, step_size, v_norm_new, clamp_min_3, truediv_4, final_scale, g_3, mul_24, mul_25, mul_26, mul_23, mask, mul_27, add_11, sub_], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.reciprocal, aten.sqrt, aten.rsub, aten.lerp, aten.rsqrt, aten.div, aten.ge, aten.sub, aten.copy_]
# Source node to ATen node mapping:
#   X_6 => add_10
#   X_7 => add_12
#   add_11 => add_14
#   clamp_min_2 => clamp_min_2
#   clamp_min_3 => clamp_min_3
#   current_norm => clamp_min_1
#   final_scale => mul_25
#   g_1 => convert_element_type_36
#   g_2 => mul_20
#   g_3 => mul_26
#   lerp__2 => abs_3, add_13, ge_2, mul_22, sub_6, sub_7, where_4, where_5
#   mask => ge_3
#   mul_13 => mul_15
#   mul_16 => mul_18
#   mul_23 => mul_27
#   mul_24 => mul_28
#   mul_25 => mul_29
#   mul_26 => mul_30
#   mul_27 => mul_31
#   norm_3 => pow_8
#   step_size => rsqrt
#   sub_ => sub_8
#   sub_1 => sub_5
#   truediv_3 => mul_19, reciprocal
#   truediv_4 => div_3
#   v_norm => sqrt
#   v_norm_new => sqrt_1
#   v_norm_sq => mul_21
# Graph fragment:
#   %expand_23 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=expand_23]
#   %bmm_11 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_11]
#   %bmm_14 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=bmm_14]
#   %sum_4 : Tensor "f32[6, 1, 1][1, 6, 6]cuda:0" = PlaceHolder[target=sum_4]
#   %mul_25 : Tensor "f32[6, 1, 12][12, 72, 1]cuda:0" = PlaceHolder[target=mul_25]
#   %copy__2 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=copy__2]
#   %arg6_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg6_1]
#   %mul_26 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=mul_26]
#   %arg7_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg7_1]
#   %sub_8 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0" = PlaceHolder[target=sub_8]
#   %mul_15 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 3.285753657755655), kwargs = {})
#   %add_10 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %bmm_11), kwargs = {})
#   %mul_18 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 2.3465413258596377), kwargs = {})
#   %add_12 : Tensor "bf16[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %bmm_14), kwargs = {})
#   %convert_element_type_36 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %pow_8 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %clamp_min_1 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_8, 1e-06), kwargs = {})
#   %reciprocal : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min_1,), kwargs = {})
#   %mul_19 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 2.449489742783178), kwargs = {})
#   %mul_20 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_36, %mul_19), kwargs = {})
#   %mul_21 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_5, 6), kwargs = {})
#   %sqrt : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%mul_21,), kwargs = {})
#   %sub_5 : Tensor "f32[][]cpu"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg4_1), kwargs = {})
#   %abs_3 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_5,), kwargs = {})
#   %ge_2 : Tensor "b8[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_3, 0.5), kwargs = {})
#   %sub_6 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_5, 1), kwargs = {})
#   %where_4 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_2, %sub_6, %sub_5), kwargs = {})
#   %sub_7 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean, %arg5_1), kwargs = {})
#   %mul_22 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_4, %sub_7), kwargs = {})
#   %where_5 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_2, %mean, %arg5_1), kwargs = {})
#   %add_13 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %where_5), kwargs = {})
#   %clamp_min_2 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_13, 1e-10), kwargs = {})
#   %rsqrt : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%clamp_min_2,), kwargs = {})
#   %sqrt_1 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%sum_6,), kwargs = {})
#   %clamp_min_3 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sqrt_1, 1e-10), kwargs = {})
#   %div_3 : Tensor "f32[6, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sqrt, %clamp_min_3), kwargs = {})
#   %mul_25 : Tensor "f32[6, 1, 12][12, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%rsqrt, %div_3), kwargs = {})
#   %mul_26 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %mul_25), kwargs = {})
#   %mul_28 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg6_1, %mul_26), kwargs = {})
#   %mul_29 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg6_1, %arg7_1), kwargs = {})
#   %mul_30 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %arg3_1), kwargs = {})
#   %mul_27 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %arg3_1), kwargs = {})
#   %ge_3 : Tensor "b8[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%mul_27, 0), kwargs = {})
#   %mul_31 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %ge_3), kwargs = {})
#   %add_14 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %mul_31), kwargs = {})
#   %sub_8 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg3_1, %add_14), kwargs = {})
#   %copy__2 : Tensor "f32[6, 6, 12][72, 12, 1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg3_1, %sub_8), kwargs = {})
#   return %mul_26,%sub_8,%buf56
triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_linalg_vector_norm_mul_reciprocal_rsqrt_rsub_sqrt_sub_14 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_linalg_vector_norm_mul_reciprocal_rsqrt_rsub_sqrt_sub_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': 'fp32', 'in_ptr7': 'fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_linalg_vector_norm_mul_reciprocal_rsqrt_rsub_sqrt_sub_14', 'mutated_arg_names': ['in_ptr5', 'out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9504}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_linalg_vector_norm_mul_reciprocal_rsqrt_rsub_sqrt_sub_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = xindex // 72
    x0 = (xindex % 12)
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0 + 12*x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x3), xmask)
    tmp22 = in_ptr6
    tmp24 = in_ptr7
    tmp1 = 3.285753657755655
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 2.3465413258596377
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = 1e-06
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = (tmp14 / tmp13)
    tmp16 = 2.449489742783178
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 * tmp17
    tmp20 = tmp18 * tmp19
    tmp23 = tmp22 * tmp20
    tmp25 = tmp22 * tmp24
    tmp26 = tmp25 * tmp21
    tmp27 = tmp20 * tmp21
    tmp28 = 0.0
    tmp29 = tmp27 >= tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp26 * tmp30
    tmp32 = tmp23 + tmp31
    tmp33 = tmp21 - tmp32
    tl.store(out_ptr0 + (x3), tmp33, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
        args.clear()
        assert_size_stride(arg0_1, (), ())
        assert_size_stride(arg1_1, (6, 6, 12), (72, 12, 1))
        assert_size_stride(arg2_1, (6, 6, 12), (72, 12, 1))
        assert_size_stride(arg3_1, (6, 6, 12), (72, 12, 1))
        assert_size_stride(arg4_1, (), ())
        assert_size_stride(arg5_1, (6, 1, 12), (12, 12, 1))
        assert_size_stride(arg6_1, (), ())
        assert_size_stride(arg7_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((6, 1, 1), (1, 6, 6), torch.float32)
            # Topologically Sorted Source Nodes: [g, sub, lerp_, X, float_1, norm], Original ATen: [aten.lerp, aten.rsub, aten._to_copy, aten.linalg_vector_norm]
            # [Provenance debug handles] triton_per_fused__to_copy_lerp_linalg_vector_norm_rsub_0:616
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_lerp_linalg_vector_norm_rsub_0.run(arg0_1.item(), arg1_1, arg2_1, buf0, 6, 72, stream=stream0)
            buf2 = empty_strided_cuda((6, 6, 12), (72, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [g, sub, lerp_, X, norm, target, float_2, norm_1, row_norm, truediv_1, to_1, X_1], Original ATen: [aten.lerp, aten.rsub, aten._to_copy, aten.linalg_vector_norm, aten.div, aten.clamp_min, aten.mul, aten.copy_]
            # [Provenance debug handles] triton_per_fused__to_copy_clamp_min_copy__div_lerp_linalg_vector_norm_mul_rsub_1:617
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_clamp_min_copy__div_lerp_linalg_vector_norm_mul_rsub_1.run(arg0_1.item(), arg1_1, arg2_1, buf0, buf2, arg1_1, arg2_1, 36, 12, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            buf3 = buf0; del buf0  # reuse
            buf4 = empty_strided_cuda((6, 6, 12), (72, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [norm_2, mul_1, add, X_2], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div]
            # [Provenance debug handles] triton_per_fused_add_div_linalg_vector_norm_mul_2:618
            stream0 = get_raw_stream(0)
            triton_per_fused_add_div_linalg_vector_norm_mul_2.run(buf2, buf3, buf4, 6, 72, stream=stream0)
            buf5 = empty_strided_cuda((6, 6, 6), (36, 6, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [norm_2, mul_1, add, X_2, getattr_1, A], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:631
            extern_kernels.bmm(buf4, reinterpret_tensor(buf4, (6, 12, 6), (72, 1, 12), 0), out=buf5)
            buf6 = empty_strided_cuda((6, 6, 6), (36, 6, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:632
            extern_kernels.bmm(buf5, buf5, out=buf6)
            buf7 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [mul_2, mul_3, B], Original ATen: [aten.mul, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_mul_3:619
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_3.run(buf7, buf6, 216, stream=stream0)
            buf8 = empty_strided_cuda((6, 6, 12), (72, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_2, mul_3, B, matmul_2], Original ATen: [aten.mul, aten.add, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:633
            extern_kernels.bmm(buf7, buf4, out=buf8)
            buf9 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [norm_2, mul_1, add, X_2, mul_4, X_3], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div]
            # [Provenance debug handles] triton_poi_fused_add_div_linalg_vector_norm_mul_4:620
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_linalg_vector_norm_mul_4.run(buf2, buf3, buf8, buf9, 432, stream=stream0)
            buf10 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [norm_2, mul_1, add, X_2, mul_4, X_3, getattr_2, A_1], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:634
            extern_kernels.bmm(buf9, reinterpret_tensor(buf9, (6, 12, 6), (72, 1, 12), 0), out=buf10)
            buf11 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:635
            extern_kernels.bmm(buf10, buf10, out=buf11)
            buf12 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [mul_5, mul_6, B_1], Original ATen: [aten.mul, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_mul_5:621
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_5.run(buf12, buf11, 216, stream=stream0)
            buf13 = empty_strided_cuda((6, 6, 12), (72, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_5, mul_6, B_1, matmul_5], Original ATen: [aten.mul, aten.add, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:636
            extern_kernels.bmm(buf12, buf9, out=buf13)
            buf14 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [norm_2, mul_1, add, X_2, mul_4, X_3, mul_7, X_4], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div]
            # [Provenance debug handles] triton_poi_fused_add_div_linalg_vector_norm_mul_6:622
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_linalg_vector_norm_mul_6.run(buf2, buf3, buf8, buf13, buf14, 432, stream=stream0)
            buf15 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [norm_2, mul_1, add, X_2, mul_4, X_3, mul_7, X_4, getattr_3, A_2], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:637
            extern_kernels.bmm(buf14, reinterpret_tensor(buf14, (6, 12, 6), (72, 1, 12), 0), out=buf15)
            buf16 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:638
            extern_kernels.bmm(buf15, buf15, out=buf16)
            buf17 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [mul_8, mul_9, B_2], Original ATen: [aten.mul, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_mul_7:623
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_7.run(buf17, buf16, 216, stream=stream0)
            buf18 = empty_strided_cuda((6, 6, 12), (72, 12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul_8, mul_9, B_2, matmul_8], Original ATen: [aten.mul, aten.add, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:639
            extern_kernels.bmm(buf17, buf14, out=buf18)
            del buf14
            buf19 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [norm_2, mul_1, add, X_2, mul_4, X_3, mul_7, X_4, mul_10, X_5], Original ATen: [aten.linalg_vector_norm, aten.mul, aten.add, aten.div]
            # [Provenance debug handles] triton_poi_fused_add_div_linalg_vector_norm_mul_8:624
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_linalg_vector_norm_mul_8.run(buf19, buf3, buf8, buf13, buf18, 432, stream=stream0)
            buf20 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [getattr_4, A_3], Original ATen: [aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:640
            extern_kernels.bmm(buf19, reinterpret_tensor(buf19, (6, 12, 6), (72, 1, 12), 0), out=buf20)
            buf21 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:641
            extern_kernels.bmm(buf20, buf20, out=buf21)
            buf22 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [mul_11, mul_12, B_3], Original ATen: [aten.mul, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_mul_9:625
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_9.run(buf22, buf21, 216, stream=stream0)
            buf23 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [mul_11, mul_12, B_3, matmul_11], Original ATen: [aten.mul, aten.add, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:642
            extern_kernels.bmm(buf22, buf19, out=buf23)
            buf24 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [mul_13, X_6], Original ATen: [aten.mul, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_mul_10:626
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_10.run(buf19, buf23, buf24, 432, stream=stream0)
            buf25 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [mul_13, X_6, getattr_5, A_4], Original ATen: [aten.mul, aten.add, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:643
            extern_kernels.bmm(buf24, reinterpret_tensor(buf24, (6, 12, 6), (72, 1, 12), 0), out=buf25)
            buf26 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:644
            extern_kernels.bmm(buf25, buf25, out=buf26)
            buf27 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [mul_14, mul_15, B_4], Original ATen: [aten.mul, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_mul_11:627
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_11.run(buf27, buf26, 216, stream=stream0)
            del buf26
            buf28 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [mul_14, mul_15, B_4, matmul_14], Original ATen: [aten.mul, aten.add, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:645
            extern_kernels.bmm(buf27, buf24, out=buf28)
            del buf24
            del buf27
            buf29 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [mul_13, X_6, mul_16, X_7, g_1, norm_3], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm]
            # [Provenance debug handles] triton_per_fused__to_copy_add_linalg_vector_norm_mul_12:628
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_linalg_vector_norm_mul_12.run(buf19, buf23, buf28, buf29, 6, 72, stream=stream0)
            buf33 = empty_strided_cuda((6, 1, 12), (12, 72, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_13, X_6, mul_16, X_7, g_1, norm_3, current_norm, truediv_3, g_2, square, v_mean, sum_1, v_norm_sq, v_norm, mul_19, sub_1, lerp__2, clamp_min_2, step_size, square_1, scaled_sq_sum, sum_2, v_norm_new, clamp_min_3, truediv_4, final_scale], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.reciprocal, aten.pow, aten.mean, aten.sum, aten.sqrt, aten.rsub, aten.lerp, aten.rsqrt, aten.div, aten.copy_]
            # [Provenance debug handles] triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_linalg_vector_norm_mean_mul_pow_reciprocal_rsqrt_rsub_sqrt_sum_13:629
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_clamp_min_copy__div_lerp_linalg_vector_norm_mean_mul_pow_reciprocal_rsqrt_rsub_sqrt_sum_13.run(buf19, buf23, buf28, buf29, arg4_1.item(), arg5_1, buf33, arg5_1, 6, 12, stream=stream0)
            del arg4_1
            del arg5_1
            # Topologically Sorted Source Nodes: [mul_13, X_6, mul_16, X_7, g_1, norm_3, current_norm, truediv_3, g_2, v_norm_sq, v_norm, sub_1, lerp__2, clamp_min_2, step_size, v_norm_new, clamp_min_3, truediv_4, final_scale, g_3, mul_24, mul_25, mul_26, mul_23, mask, mul_27, add_11, sub_], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.linalg_vector_norm, aten.clamp_min, aten.reciprocal, aten.sqrt, aten.rsub, aten.lerp, aten.rsqrt, aten.div, aten.ge, aten.sub, aten.copy_]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_linalg_vector_norm_mul_reciprocal_rsqrt_rsub_sqrt_sub_14:630
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clamp_min_copy__div_ge_lerp_linalg_vector_norm_mul_reciprocal_rsqrt_rsub_sqrt_sub_14.run(buf19, buf23, buf28, buf29, buf33, arg3_1, arg6_1.item(), arg7_1.item(), arg3_1, 432, stream=stream0)
            del arg3_1
            del arg6_1
            del arg7_1
            del buf19
            del buf23
            del buf28
            del buf29
            del buf33
        return ()

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((6, 6, 12), (72, 12, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((6, 6, 12), (72, 12, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((6, 6, 12), (72, 12, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((6, 1, 12), (12, 12, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
