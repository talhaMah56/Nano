# AOT ID: ['0_backward']
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


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/bj/cbji2jgrscdxb6xyurzhwivzmf2wrolv4tlnlriw2zo2ah27n2zk.py
# Topologically Sorted Source Nodes: [view_55, loss, , logits, logits_2, truediv, tanh, logits_3, view_54], Original ATen: [aten.nll_loss_backward, aten.view, aten.nll_loss_forward, aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten._log_softmax, aten._log_softmax_backward_data, aten.tanh_backward]
# Source node to ATen node mapping:
#    => scatter_upon_const_tensor
#   logits => view_213
#   logits_2 => convert_element_type_342
#   logits_3 => mul_209
#   loss => full_default, full_default_1, sub_1, sub_2
#   tanh => tanh
#   truediv => div
#   view_54 => view_214
#   view_55 => view_215
# Graph fragment:
#   %primals_95 : Tensor "i64[16, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_95]
#   %tangents_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_1]
#   %convert_element_type_343 : Tensor "f32[][]cuda:0" = PlaceHolder[target=convert_element_type_343]
#   %mm_79 : Tensor "bf16[32768, 32768][32768, 1]cuda:0" = PlaceHolder[target=mm_79]
#   %amax : Tensor "f32[32768, 1][1, 1]cuda:0" = PlaceHolder[target=amax]
#   %log : Tensor "f32[32768, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %sum_4 : Tensor "f32[32768, 1][1, 32768]cuda:0" = PlaceHolder[target=sum_4]
#   %sub_3 : Tensor "f32[32768, 32768][32768, 1]cuda:0" = PlaceHolder[target=sub_3]
#   %div_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_343), kwargs = {})
#   %view_215 : Tensor "i64[32768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_95, [-1]), kwargs = {})
#   %unsqueeze_7 : Tensor "i64[32768, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_215, 1), kwargs = {})
#   %ne_3 : Tensor "b8[32768, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_7, -1), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "i64[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze_7, %full_default), kwargs = {})
#   %scatter_upon_const_tensor : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [32768, 32768], background_val: 0, dtype: torch.float32, dim: 1, selector: %where_2, val: -1.0})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : Tensor "f32[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %div_2, %full_default_1), kwargs = {})
#   %mul_210 : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_upon_const_tensor, %where_3), kwargs = {})
#   %view_213 : Tensor "bf16[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_79, [16, 2048, 32768]), kwargs = {})
#   %convert_element_type_342 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_213, torch.float32), kwargs = {})
#   %div : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_342, 15), kwargs = {})
#   %tanh : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul_209 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, 15), kwargs = {})
#   %view_214 : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_209, [-1, 32768]), kwargs = {})
#   %sub_1 : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_214, %amax), kwargs = {})
#   %sub_2 : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, %log), kwargs = {})
#   %exp_1 : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_4 : Tensor "f32[32768, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_210, [1], True), kwargs = {})
#   %mul_211 : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %sum_4), kwargs = {})
#   %sub_3 : Tensor "f32[32768, 32768][32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_210, %mul_211), kwargs = {})
#   %view_216 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_3, [16, 2048, 32768]), kwargs = {})
#   %mul_212 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_216, 15), kwargs = {})
#   %mul_213 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %tanh), kwargs = {})
#   %sub_4 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_213), kwargs = {})
#   %mul_214 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_212, %sub_4), kwargs = {})
#   %div_3 : Tensor "f32[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_214, 15), kwargs = {})
#   %convert_element_type_344 : Tensor "bf16[16, 2048, 32768][67108864, 32768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_3, torch.bfloat16), kwargs = {})
#   return %sum_4,%sub_3,%convert_element_type_344
triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_view_0 = async_compile.triton('triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_view_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_view_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288, 'r0_': 6442450944}}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_view_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tl.full([1, 1], -1, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = r0_1
        tmp6 = tmp4 == tmp5
        tmp7 = -1.0
        tmp8 = 0.0
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp14 = (tmp11 / tmp13)
        tmp15 = tl.where(tmp2, tmp14, tmp8)
        tmp16 = tmp9 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp29 = tl.load(in_ptr1 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp31 = tl.load(in_ptr2 + (0))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp43 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp36 = tl.load(in_out_ptr0 + (r0_1 + 32768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.full([1, 1], -1, tl.int64)
        tmp21 = tmp0 != tmp20
        tmp22 = tl.full([1, 1], 0, tl.int64)
        tmp23 = tl.where(tmp21, tmp0, tmp22)
        tmp24 = r0_1
        tmp25 = tmp23 == tmp24
        tmp26 = -1.0
        tmp27 = 0.0
        tmp28 = tl.where(tmp25, tmp26, tmp27)
        tmp33 = (tmp30 / tmp32)
        tmp34 = tl.where(tmp21, tmp33, tmp27)
        tmp35 = tmp28 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = 0.06666666666666667
        tmp39 = tmp37 * tmp38
        tmp40 = libdevice.tanh(tmp39)
        tmp41 = 15.0
        tmp42 = tmp40 * tmp41
        tmp44 = tmp42 - tmp43
        tmp46 = tmp44 - tmp45
        tmp47 = libdevice.exp(tmp46)
        tmp48 = tmp47 * tmp18
        tmp49 = tmp35 - tmp48
        tmp50 = tmp49 * tmp41
        tmp51 = tmp40 * tmp40
        tmp52 = 1.0
        tmp53 = tmp52 - tmp51
        tmp54 = tmp50 * tmp53
        tmp55 = tmp54 * tmp38
        tmp56 = tmp55.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 32768*x0), tmp56, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/ni/cniij5bavssl7ds4eex2onvyl2oqtuvakg4eykosife6u4ebkgln.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_80 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_80]
#   %convert_element_type_349 : Tensor "f32[32768, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_80, torch.float32), kwargs = {})
#   return %convert_element_type_349
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 251658240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/o7/co7mqdwn5zl5ra4lhbeoskm6v2rc5blomp7cb6ljnaxw6ovlhbnw.py
# Topologically Sorted Source Nodes: [clone_1, x_77], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.neg, aten.sum]
# Source node to ATen node mapping:
#   clone_1 => convert_element_type_350
#   x_77 => convert_element_type_337, mul_208
# Graph fragment:
#   %sub : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=sub]
#   %rsqrt_49 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_49]
#   %mm_81 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_81]
#   %sum_5 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_5]
#   %add_81 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_81]
#   %view_218 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_81, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_350 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_218, torch.float32), kwargs = {})
#   %convert_element_type_337 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.float32), kwargs = {})
#   %mul_208 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_337, %rsqrt_49), kwargs = {})
#   %mul_216 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_208, %convert_element_type_350), kwargs = {})
#   %sum_5 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_216, [2], True), kwargs = {})
#   %div_4 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_208, 768), kwargs = {})
#   %mul_217 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sum_5), kwargs = {})
#   %sub_5 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_350, %mul_217), kwargs = {})
#   %mul_218 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_49), kwargs = {})
#   %convert_element_type_352 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_218, torch.bfloat16), kwargs = {})
#   %neg_25 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_352,), kwargs = {})
#   %mul_220 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_25, %add_81), kwargs = {})
#   %sum_6 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_220, [0, 1, 2], True), kwargs = {})
#   return %sum_5,%buf7,%convert_element_type_352
triton_per_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_2 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 655360, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = 0.0013020833333333333
    tmp12 = tmp3 * tmp11
    tmp13 = tmp12 * tmp10
    tmp14 = tmp5 - tmp13
    tmp15 = tmp14 * tmp2
    tmp16 = tmp15.to(tl.float32)
    tmp17 = -tmp16
    tmp19 = tmp17 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp16, r0_mask)
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp23, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/km/ckmf2xps7yipjnwlqix4bz36s4m5bfbyzkqrkflanlue7fkp7eqw.py
# Topologically Sorted Source Nodes: [clone_1, x_77], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.neg, aten.sum]
# Source node to ATen node mapping:
#   clone_1 => convert_element_type_350
#   x_77 => convert_element_type_337, mul_208
# Graph fragment:
#   %buf7 : Tensor "f32[1, 1, 1, 16, 2048, 1][32768, 32768, 32768, 2048, 1, 32768]cuda:0" = PlaceHolder[target=buf7]
#   %sum_6 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=sum_6]
#   %view_218 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_81, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_350 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_218, torch.float32), kwargs = {})
#   %convert_element_type_337 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.float32), kwargs = {})
#   %mul_208 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_337, %rsqrt_49), kwargs = {})
#   %div_4 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_208, 768), kwargs = {})
#   %mul_217 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sum_5), kwargs = {})
#   %sub_5 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_350, %mul_217), kwargs = {})
#   %mul_218 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_49), kwargs = {})
#   %convert_element_type_352 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_218, torch.bfloat16), kwargs = {})
#   %neg_25 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_352,), kwargs = {})
#   %mul_220 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_25, %add_81), kwargs = {})
#   %sum_6 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_220, [0, 1, 2], True), kwargs = {})
#   %view_219 : Tensor "bf16[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_6, [1]), kwargs = {})
#   %convert_element_type_353 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_219, torch.float32), kwargs = {})
#   return %sum_6,%convert_element_type_353
triton_red_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_3 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 131072}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_3(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 32768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/2b/c2bbkishr2optwcyzro36c2wntd4fizhmhu6xbtkukbnqedihfvz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_82 : Tensor "bf16[768, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_82]
#   %convert_element_type_358 : Tensor "f32[768, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_82, torch.float32), kwargs = {})
#   return %convert_element_type_358
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/26/c26swijaoznbhs3n6bhld22ytlzhdpg6nvfi632oorkjd6vs7ysg.py
# Topologically Sorted Source Nodes: [x_72, relu_11, scalar_tensor_4], Original ATen: [aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.threshold_backward]
# Source node to ATen node mapping:
#   relu_11 => relu_11
#   scalar_tensor_4 => full_default_5
#   x_72 => view_209
# Graph fragment:
#   %mm_77 : Tensor "bf16[32768, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_77]
#   %mm_83 : Tensor "bf16[32768, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_83]
#   %view_221 : Tensor "bf16[16, 2048, 3072][6291456, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_83, [16, 2048, 3072]), kwargs = {})
#   %view_209 : Tensor "bf16[16, 2048, 3072][6291456, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_77, [16, 2048, 3072]), kwargs = {})
#   %relu_11 : Tensor "bf16[16, 2048, 3072][6291456, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_209,), kwargs = {})
#   %pow_63 : Tensor "bf16[16, 2048, 3072][6291456, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu_11, 1.0), kwargs = {})
#   %mul_221 : Tensor "bf16[16, 2048, 3072][6291456, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_63, 2.0), kwargs = {})
#   %mul_222 : Tensor "bf16[16, 2048, 3072][6291456, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_221, %mul_221), kwargs = {})
#   %le : Tensor "b8[16, 2048, 3072][6291456, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_11, 0), kwargs = {})
#   %full_default_5 : Tensor "bf16[][]cuda:0"[num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "bf16[16, 2048, 3072][6291456, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default_5, %mul_222), kwargs = {})
#   return %where_4
triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5 = async_compile.triton('triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 805306368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100663296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp6 = 2.0
    tmp7 = tmp2 * tmp6
    tmp8 = tmp5 * tmp7
    tmp9 = tl.where(tmp4, tmp3, tmp8)
    tl.store(in_out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/f5/cf5jvihagee4xr4rljviiz5snsxn6ypx5dv46lohzdkqnnjg4cej.py
# Topologically Sorted Source Nodes: [clone_3, rms_norm_48], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
# Source node to ATen node mapping:
#   clone_3 => convert_element_type_364
#   rms_norm_48 => convert_element_type_328, mul_206
# Graph fragment:
#   %add_137 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_137]
#   %rsqrt_48 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_48]
#   %mm_85 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_85]
#   %convert_element_type_352 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=convert_element_type_352]
#   %sum_7 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_7]
#   %view_223 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_85, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_364 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_223, torch.float32), kwargs = {})
#   %convert_element_type_328 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_137, torch.float32), kwargs = {})
#   %mul_206 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_328, %rsqrt_48), kwargs = {})
#   %mul_224 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_206, %convert_element_type_364), kwargs = {})
#   %sum_7 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_224, [2], True), kwargs = {})
#   %div_5 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_206, 768), kwargs = {})
#   %mul_225 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_5, %sum_7), kwargs = {})
#   %sub_6 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_364, %mul_225), kwargs = {})
#   %mul_226 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_48), kwargs = {})
#   %convert_element_type_366 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_226, torch.bfloat16), kwargs = {})
#   %add_141 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_352, %convert_element_type_366), kwargs = {})
#   return %sum_7,%add_141
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_6 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp12 = 0.0013020833333333333
    tmp13 = tmp3 * tmp12
    tmp14 = tmp13 * tmp10
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp2
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp11 + tmp17
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp18, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/vc/cvch25752va3gzesbbq5zsqxemwgydmyheipcojhhmkf2ppg5sg4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_86 : Tensor "bf16[768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_86]
#   %convert_element_type_371 : Tensor "f32[768, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_86, torch.float32), kwargs = {})
#   return %convert_element_type_371
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5898240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/w6/cw65kimqpoub7qkrd4gie5xv7xucib72pg6xlobuixuvdd7v5oj4.py
# Topologically Sorted Source Nodes: [clone_5, k_46], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
# Source node to ATen node mapping:
#   clone_5 => convert_element_type_372
#   k_46 => convert_element_type_323, mul_203
# Graph fragment:
#   %cat_24 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0" = PlaceHolder[target=cat_24]
#   %rsqrt_47 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_47]
#   %getitem_49 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0" = PlaceHolder[target=getitem_49]
#   %mul_227 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_49, 1.2), kwargs = {})
#   %convert_element_type_372 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_227, torch.float32), kwargs = {})
#   %convert_element_type_323 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_24, torch.float32), kwargs = {})
#   %mul_203 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_323, %rsqrt_47), kwargs = {})
#   %mul_230 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_203, %convert_element_type_372), kwargs = {})
#   %sum_8 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_230, [3], True), kwargs = {})
#   return %sum_8
triton_per_fused__fused_rms_norm_backward__to_copy_mul_8 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_mul_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_mul_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296, 'r0_': 100663296}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_mul_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = 1.2
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp3 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/uq/cuqrvlspmliw7dr7wxxbsyveoygl5vygpgav3lr52vj53pee2wi3.py
# Topologically Sorted Source Nodes: [clone_5, k_46, clone_7, q_46, cos, sin, neg], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.slice, aten.neg, aten.add]
# Source node to ATen node mapping:
#   clone_5 => convert_element_type_372
#   clone_7 => convert_element_type_375
#   cos => slice_1
#   k_46 => convert_element_type_323, mul_203
#   neg => neg
#   q_46 => convert_element_type_321, mul_202
#   sin => slice_2
# Graph fragment:
#   %getitem_49 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0" = PlaceHolder[target=getitem_49]
#   %cat_24 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0" = PlaceHolder[target=cat_24]
#   %rsqrt_47 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_47]
#   %sum_8 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 196608]cuda:0" = PlaceHolder[target=sum_8]
#   %primals_2 : Tensor "bf16[1, 20480, 1, 64][1310720, 64, 64, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "bf16[1, 20480, 1, 64][1310720, 64, 64, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %getitem_48 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0" = PlaceHolder[target=getitem_48]
#   %cat_23 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0" = PlaceHolder[target=cat_23]
#   %rsqrt_46 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_46]
#   %sum_9 : Tensor "f32[16, 2048, 6, 1][12288, 6, 1, 196608]cuda:0" = PlaceHolder[target=sum_9]
#   %mul_227 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_49, 1.2), kwargs = {})
#   %mul_228 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_48, 1.2), kwargs = {})
#   %convert_element_type_372 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_227, torch.float32), kwargs = {})
#   %convert_element_type_323 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_24, torch.float32), kwargs = {})
#   %mul_203 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_323, %rsqrt_47), kwargs = {})
#   %div_6 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_203, 128), kwargs = {})
#   %mul_231 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, %sum_8), kwargs = {})
#   %sub_7 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_372, %mul_231), kwargs = {})
#   %mul_232 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_47), kwargs = {})
#   %convert_element_type_374 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_232, torch.bfloat16), kwargs = {})
#   %convert_element_type_375 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_228, torch.float32), kwargs = {})
#   %convert_element_type_321 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_23, torch.float32), kwargs = {})
#   %mul_202 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_321, %rsqrt_46), kwargs = {})
#   %div_7 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_202, 128), kwargs = {})
#   %mul_235 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_7, %sum_9), kwargs = {})
#   %sub_8 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_375, %mul_235), kwargs = {})
#   %mul_236 : Tensor "f32[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_46), kwargs = {})
#   %convert_element_type_377 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_236, torch.bfloat16), kwargs = {})
#   %slice_62 : Tensor "bf16[16, 2048, 6, 64][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_374, 3, 0, 64), kwargs = {})
#   %slice_63 : Tensor "bf16[16, 2048, 6, 64][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_374, 3, 64, 128), kwargs = {})
#   %slice_1 : Tensor "bf16[1, 2048, 1, 64][1310720, 64, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_2, 1, 0, 2048), kwargs = {})
#   %mul_237 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_63, %slice_1), kwargs = {})
#   %slice_2 : Tensor "bf16[1, 2048, 1, 64][1310720, 64, 64, 1]cuda:0"[num_users=25] = call_function[target=torch.ops.aten.slice.Tensor](args = (%primals_3, 1, 0, 2048), kwargs = {})
#   %neg : Tensor "bf16[1, 2048, 1, 64][131072, 64, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %mul_238 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_63, %neg), kwargs = {})
#   %mul_239 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_62, %slice_2), kwargs = {})
#   %add_142 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_237, %mul_239), kwargs = {})
#   %mul_240 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_62, %slice_1), kwargs = {})
#   %add_143 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_238, %mul_240), kwargs = {})
#   %slice_64 : Tensor "bf16[16, 2048, 6, 64][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_377, 3, 0, 64), kwargs = {})
#   %slice_65 : Tensor "bf16[16, 2048, 6, 64][1572864, 768, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_377, 3, 64, 128), kwargs = {})
#   %mul_241 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_65, %slice_1), kwargs = {})
#   %mul_242 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_65, %neg), kwargs = {})
#   %mul_243 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_64, %slice_2), kwargs = {})
#   %add_145 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_241, %mul_243), kwargs = {})
#   %mul_244 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_64, %slice_1), kwargs = {})
#   %add_146 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_242, %mul_244), kwargs = {})
#   return %add_142,%add_143,%add_145,%add_146
triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9 = async_compile.triton('triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3145728, 'x': 403177472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196608
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y1 = ((yindex // 6) % 2048)
    tmp0 = tl.load(in_ptr0 + (64 + x3 + 128*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (64 + x3 + 128*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (y4), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y4), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3 + 64*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (x3 + 128*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr1 + (x3 + 128*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp28 = tl.load(in_ptr5 + (x3 + 64*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp35 = tl.load(in_ptr6 + (64 + x3 + 128*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp38 = tl.load(in_ptr7 + (64 + x3 + 128*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp40 = tl.load(in_ptr8 + (y4), ymask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr9 + (y4), ymask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr6 + (x3 + 128*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp52 = tl.load(in_ptr7 + (x3 + 128*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 1.2
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 0.0078125
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 - tmp11
    tmp13 = tmp12 * tmp6
    tmp14 = tmp13.to(tl.float32)
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp1
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21 * tmp6
    tmp23 = tmp22 * tmp8
    tmp24 = tmp23 * tmp10
    tmp25 = tmp19 - tmp24
    tmp26 = tmp25 * tmp6
    tmp27 = tmp26.to(tl.float32)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp16 + tmp29
    tmp31 = -tmp28
    tmp32 = tmp14 * tmp31
    tmp33 = tmp27 * tmp15
    tmp34 = tmp32 + tmp33
    tmp36 = tmp35 * tmp1
    tmp37 = tmp36.to(tl.float32)
    tmp39 = tmp38.to(tl.float32)
    tmp41 = tmp39 * tmp40
    tmp42 = tmp41 * tmp8
    tmp44 = tmp42 * tmp43
    tmp45 = tmp37 - tmp44
    tmp46 = tmp45 * tmp40
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp47 * tmp15
    tmp50 = tmp49 * tmp1
    tmp51 = tmp50.to(tl.float32)
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp53 * tmp40
    tmp55 = tmp54 * tmp8
    tmp56 = tmp55 * tmp43
    tmp57 = tmp51 - tmp56
    tmp58 = tmp57 * tmp40
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp59 * tmp28
    tmp61 = tmp48 + tmp60
    tmp62 = tmp47 * tmp31
    tmp63 = tmp59 * tmp15
    tmp64 = tmp62 + tmp63
    tl.store(out_ptr0 + (x3 + 64*y4), tmp30, xmask & ymask)
    tl.store(out_ptr1 + (x3 + 64*y4), tmp34, xmask & ymask)
    tl.store(out_ptr2 + (x3 + 64*y4), tmp61, xmask & ymask)
    tl.store(out_ptr3 + (x3 + 64*y4), tmp64, xmask & ymask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/jf/cjfru5fecjeroab7cqggbnf4plquif3tkzx6rufw4ad7l7h4mmcj.py
# Topologically Sorted Source Nodes: [loss, linear_75, sigmoid_6, gate_6, unsqueeze_5, full_6], Original ATen: [aten.nll_loss_forward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   full_6 => full_default_12
#   gate_6 => mul_192
#   linear_75 => view_204
#   loss => full_default_1
#   sigmoid_6 => sigmoid_6
#   unsqueeze_5 => unsqueeze_5
# Graph fragment:
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_204 : Tensor "bf16[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_75, [16, 2048, 6]), kwargs = {})
#   %sigmoid_6 : Tensor "bf16[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_204,), kwargs = {})
#   %mul_192 : Tensor "bf16[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_6, 3), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_192, -1), kwargs = {})
#   %mul_245 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_50, %unsqueeze_5), kwargs = {})
#   %view_229 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_245, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_404 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_229, torch.float32), kwargs = {})
#   %eq : Tensor "b8[16, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, -1), kwargs = {})
#   %unsqueeze_8 : Tensor "b8[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %where_5 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_8, %full_default_1, %convert_element_type_404), kwargs = {})
#   %full_default_12 : Tensor "f32[32768, 768][768, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([32768, 768], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : Tensor "f32[32768, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_12, [%primals_1], %where_5, True), kwargs = {})
#   return %index_put
triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10 = async_compile.triton('triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201326592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/7y/c7y43d63mtwrqsvrezk3xmuksffsn5dhk3nxk7khkgpn5ulgamvl.py
# Topologically Sorted Source Nodes: [loss, linear_75, sigmoid_6, gate_6, unsqueeze_5, ve_11, full_6], Original ATen: [aten.nll_loss_forward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten.squeeze, aten.sigmoid_backward, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   full_6 => full_default_12
#   gate_6 => mul_192
#   linear_75 => view_204
#   loss => full_default_1
#   sigmoid_6 => sigmoid_6
#   unsqueeze_5 => unsqueeze_5
#   ve_11 => view_202
# Graph fragment:
#   %getitem_50 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0" = PlaceHolder[target=getitem_50]
#   %embedding_6 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=embedding_6]
#   %primals_1 : Tensor "i64[16, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %mm_75 : Tensor "bf16[32768, 6][6, 1]cuda:0" = PlaceHolder[target=mm_75]
#   %index_put : Tensor "f32[32768, 768][768, 1]cuda:0" = PlaceHolder[target=index_put]
#   %sum_10 : Tensor "bf16[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0" = PlaceHolder[target=sum_10]
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_204 : Tensor "bf16[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_75, [16, 2048, 6]), kwargs = {})
#   %sigmoid_6 : Tensor "bf16[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_204,), kwargs = {})
#   %mul_192 : Tensor "bf16[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_6, 3), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_192, -1), kwargs = {})
#   %mul_245 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_50, %unsqueeze_5), kwargs = {})
#   %view_202 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%embedding_6, [16, 2048, 6, 128]), kwargs = {})
#   %mul_246 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_50, %view_202), kwargs = {})
#   %sum_10 : Tensor "bf16[16, 2048, 6, 1][12288, 6, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_246, [3], True), kwargs = {})
#   %squeeze_1 : Tensor "bf16[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%sum_10, -1), kwargs = {})
#   %mul_247 : Tensor "bf16[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, 3), kwargs = {})
#   %convert_element_type_378 : Tensor "f32[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_247, torch.float32), kwargs = {})
#   %convert_element_type_379 : Tensor "f32[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid_6, torch.float32), kwargs = {})
#   %sub_9 : Tensor "f32[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_379), kwargs = {})
#   %mul_248 : Tensor "f32[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_379, %sub_9), kwargs = {})
#   %mul_249 : Tensor "f32[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_378, %mul_248), kwargs = {})
#   %convert_element_type_380 : Tensor "bf16[16, 2048, 6][12288, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_249, torch.bfloat16), kwargs = {})
#   %view_229 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_245, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_404 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_229, torch.float32), kwargs = {})
#   %eq : Tensor "b8[16, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, -1), kwargs = {})
#   %unsqueeze_8 : Tensor "b8[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %where_5 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_8, %full_default_1, %convert_element_type_404), kwargs = {})
#   %full_default_12 : Tensor "f32[32768, 768][768, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([32768, 768], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : Tensor "f32[32768, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_12, [%primals_1], %where_5, True), kwargs = {})
#   return %sum_10,%buf54,%convert_element_type_380
triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11 = async_compile.triton('triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_1 = r0_index
    x0 = xindex
    x3 = xindex // 6
    x2 = (xindex % 6)
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp24 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp7 = tl.full([XBLOCK, R0_BLOCK], 32768, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 32768), "index out of bounds: 0 <= tmp10 < 32768")
    tmp12 = tl.full([1, 1], -1, tl.int64)
    tmp13 = tmp6 == tmp12
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = 3.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp0 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = 0.0
    tmp21 = tl.where(tmp13, tmp20, tmp19)
    tmp22 = tmp5 * tmp16
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = 1.0
    tmp28 = tmp27 - tmp26
    tmp29 = tmp26 * tmp28
    tmp30 = tmp23 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.atomic_add(out_ptr0 + (r0_1 + 128*x2 + 768*tmp10), tmp21, None, sem='relaxed')
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp31, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/3m/c3mmk5g2a6io3mjeoblg6ogkzgg2tv3qkxm3ti7ysul4t6s5bvie.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_88 : Tensor "bf16[6, 12][12, 1]cuda:0" = PlaceHolder[target=mm_88]
#   %convert_element_type_385 : Tensor "f32[6, 12][12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_88, torch.float32), kwargs = {})
#   return %convert_element_type_385
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/zz/czz4d5unkj3raujkt2uvppjgrojsb66k5jjwozin4vpcmdsqsulg.py
# Topologically Sorted Source Nodes: [full_1, _generalized_scatter, _generalized_scatter_1], Original ATen: [aten.slice_backward, aten.add]
# Source node to ATen node mapping:
#   _generalized_scatter => slice_scatter_default
#   _generalized_scatter_1 => slice_scatter_default_1
#   full_1 => full_default_6
# Graph fragment:
#   %add_142 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0" = PlaceHolder[target=add_142]
#   %add_143 : Tensor "bf16[16, 2048, 6, 64][786432, 384, 64, 1]cuda:0" = PlaceHolder[target=add_143]
#   %full_default_6 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.full.default](args = ([16, 2048, 6, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_6, %add_142, 3, 64, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_6, %add_143, 3, 0, 64), kwargs = {})
#   %add_144 : Tensor "bf16[16, 2048, 6, 128][1572864, 768, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %slice_scatter_default_1), kwargs = {})
#   return %add_144
triton_poi_fused_add_slice_backward_13 = async_compile.triton('triton_poi_fused_add_slice_backward_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201326592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_slice_backward_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-64) + x0 + 64*x1), tmp2, other=0.0).to(tl.float32)
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tmp0 < tmp1
    tmp7 = tl.load(in_ptr1 + (x0 + 64*x1), tmp6, other=0.0).to(tl.float32)
    tmp8 = tl.where(tmp6, tmp7, tmp4)
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/46/c46abxj7fyqla3vq76m2l5kbpeep3wctwhkj4kfqc23a67d7epku.py
# Topologically Sorted Source Nodes: [full_5, _generalized_scatter_4, clone_9, getitem_77, mul_144, getitem_78, mul_145, x_70, rms_norm_45, getitem_71, mul_132, getitem_72, mul_133, x_64, rms_norm_41], Original ATen: [aten.view, aten.slice_backward, aten.add, aten._fused_rms_norm_backward, aten.select, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   _generalized_scatter_4 => slice_scatter_default_4
#   clone_9 => convert_element_type_401
#   full_5 => full_default_10
#   getitem_71 => select_20
#   getitem_72 => select_21
#   getitem_77 => select_22
#   getitem_78 => select_23
#   mul_132 => mul_173
#   mul_133 => mul_174
#   mul_144 => mul_189
#   mul_145 => mul_190
#   rms_norm_41 => convert_element_type_281, mul_175
#   rms_norm_45 => convert_element_type_307, mul_191
#   x_64 => add_117
#   x_70 => add_128
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_127 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_127]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %rsqrt_45 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_45]
#   %add_116 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_116]
#   %rsqrt_41 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_41]
#   %mul_191 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=mul_191]
#   %mm_89 : Tensor "bf16[32768, 12][12, 1]cuda:0" = PlaceHolder[target=mm_89]
#   %mm_91 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_91]
#   %mm_93 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_93]
#   %mm_95 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_95]
#   %add_141 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_141]
#   %sum_11 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_11]
#   %add_151 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_151]
#   %view_228 : Tensor "bf16[16, 2048, 12][24576, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_89, [16, 2048, 12]), kwargs = {})
#   %full_default_10 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([16, 2048, 768], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_4 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_228, 2, 0, 12), kwargs = {})
#   %view_232 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_91, [16, 2048, 768]), kwargs = {})
#   %add_148 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_4, %view_232), kwargs = {})
#   %view_235 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_93, [16, 2048, 768]), kwargs = {})
#   %add_149 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_148, %view_235), kwargs = {})
#   %view_238 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_95, [16, 2048, 768]), kwargs = {})
#   %add_150 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_149, %view_238), kwargs = {})
#   %convert_element_type_401 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_150, torch.float32), kwargs = {})
#   %select_22 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 11), kwargs = {})
#   %mul_189 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_22, %add_127), kwargs = {})
#   %select_23 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 11), kwargs = {})
#   %mul_190 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_23, %cat), kwargs = {})
#   %add_128 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_189, %mul_190), kwargs = {})
#   %convert_element_type_307 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_128, torch.float32), kwargs = {})
#   %mul_191 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_307, %rsqrt_45), kwargs = {})
#   %mul_251 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_191, %convert_element_type_401), kwargs = {})
#   %sum_11 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_251, [2], True), kwargs = {})
#   %div_8 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_191, 768), kwargs = {})
#   %mul_252 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_8, %sum_11), kwargs = {})
#   %sub_10 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_401, %mul_252), kwargs = {})
#   %mul_253 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %rsqrt_45), kwargs = {})
#   %convert_element_type_403 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_253, torch.bfloat16), kwargs = {})
#   %add_151 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_141, %convert_element_type_403), kwargs = {})
#   %mul_256 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_151, %select_22), kwargs = {})
#   %view_239 : Tensor "bf16[32768, 768][768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_256, [32768, 768]), kwargs = {})
#   %select_20 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 10), kwargs = {})
#   %mul_173 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_20, %add_116), kwargs = {})
#   %select_21 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 10), kwargs = {})
#   %mul_174 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_21, %cat), kwargs = {})
#   %add_117 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %mul_174), kwargs = {})
#   %convert_element_type_281 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_117, torch.float32), kwargs = {})
#   %mul_175 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_281, %rsqrt_41), kwargs = {})
#   return %mul_191,%mul_175,%sum_11,%add_151,%view_239
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_14 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 805306368}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (11))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    x0 = xindex
    tmp5 = tl.load(in_ptr2 + (11))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (10))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.load(in_ptr2 + (10))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp25 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    _tmp42 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr5 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr10 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp2 * tmp3
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp4 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp11 * tmp12
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp16 * tmp17
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21 * tmp8
        tmp23 = tmp18 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tmp27 = r0_1
        tmp28 = tl.full([1, 1], 12, tl.int64)
        tmp29 = tmp27 < tmp28
        tmp30 = tl.load(in_ptr7 + (r0_1 + 12*x0), r0_mask & tmp29, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = 0.0
        tmp32 = tl.where(tmp29, tmp30, tmp31)
        tmp34 = tmp32 + tmp33
        tmp36 = tmp34 + tmp35
        tmp38 = tmp36 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp13 * tmp39
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(r0_mask, tmp43, _tmp42)
        tl.store(out_ptr0 + (r0_1 + 768*x0), tmp13, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 768*x0), tmp26, r0_mask)
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp66 = tl.load(in_ptr0 + (11))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp44 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp51 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp55 = tl.load(in_ptr10 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp58 = tl.load(out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp45 = r0_1
        tmp46 = tl.full([1, 1], 12, tl.int64)
        tmp47 = tmp45 < tmp46
        tmp48 = tl.load(in_ptr7 + (r0_1 + 12*x0), r0_mask & tmp47, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp49 = 0.0
        tmp50 = tl.where(tmp47, tmp48, tmp49)
        tmp52 = tmp50 + tmp51
        tmp54 = tmp52 + tmp53
        tmp56 = tmp54 + tmp55
        tmp57 = tmp56.to(tl.float32)
        tmp59 = 0.0013020833333333333
        tmp60 = tmp58 * tmp59
        tmp61 = tmp60 * tmp42
        tmp62 = tmp57 - tmp61
        tmp63 = tmp62 * tmp12
        tmp64 = tmp63.to(tl.float32)
        tmp65 = tmp44 + tmp64
        tmp68 = tmp67.to(tl.float32)
        tmp69 = tmp65 * tmp68
        tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp65, r0_mask)
        tl.store(out_ptr3 + (r0_1 + 768*x0), tmp69, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/ym/cym3p3sdzip56ez7gm7jscdegrpzt2igxwjk4hymj3jedwb5ywwl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf54 : Tensor "f32[32768, 768][768, 1]cuda:0" = PlaceHolder[target=buf54]
#   %convert_element_type_405 : Tensor "bf16[32768, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%index_put, torch.bfloat16), kwargs = {})
#   return %convert_element_type_405
triton_poi_fused_embedding_dense_backward_15 = async_compile.triton('triton_poi_fused_embedding_dense_backward_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/g4/cg4jq5baxyxoub6n73mwdhdqjwsginf3a6fzrcdvfeyroddfd3pi.py
# Topologically Sorted Source Nodes: [getitem_77, clone_11, rms_norm_44], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   clone_11 => convert_element_type_418
#   getitem_77 => select_22
#   rms_norm_44 => convert_element_type_299, mul_188
# Graph fragment:
#   %add_125 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_125]
#   %rsqrt_44 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_44]
#   %mm_99 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_99]
#   %add_151 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_151]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %sum_14 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_14]
#   %select_22 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 11), kwargs = {})
#   %mul_256 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_151, %select_22), kwargs = {})
#   %view_242 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_99, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_418 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_242, torch.float32), kwargs = {})
#   %convert_element_type_299 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_125, torch.float32), kwargs = {})
#   %mul_188 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_299, %rsqrt_44), kwargs = {})
#   %mul_261 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_188, %convert_element_type_418), kwargs = {})
#   %sum_14 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_261, [2], True), kwargs = {})
#   %div_9 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_188, 768), kwargs = {})
#   %mul_262 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %sum_14), kwargs = {})
#   %sub_11 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_418, %mul_262), kwargs = {})
#   %mul_263 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_44), kwargs = {})
#   %convert_element_type_420 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_263, torch.bfloat16), kwargs = {})
#   %add_152 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_256, %convert_element_type_420), kwargs = {})
#   return %sum_14,%add_152
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_16 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (11))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0013020833333333333
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/kx/ckxk7ld4lgwnv44qa3ihvltx6sys3vm4osa6sep6ugcr33dfrqtq.py
# Topologically Sorted Source Nodes: [clone_17, getitem_71], Original ATen: [aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   clone_17 => convert_element_type_447
#   getitem_71 => select_20
# Graph fragment:
#   %mul_175 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=mul_175]
#   %mm_103 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_103]
#   %mm_105 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_105]
#   %mm_107 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_107]
#   %add_152 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_152]
#   %sum_17 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_17]
#   %rsqrt_41 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_41]
#   %add_161 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_161]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_151 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_151]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %add_127 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_127]
#   %add_116 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_116]
#   %mul_255 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_151, %cat), kwargs = {})
#   %sum_12 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_255,), kwargs = {})
#   %mul_257 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_151, %add_127), kwargs = {})
#   %sum_13 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_257,), kwargs = {})
#   %view_248 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_103, [16, 2048, 768]), kwargs = {})
#   %view_251 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_105, [16, 2048, 768]), kwargs = {})
#   %add_159 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_248, %view_251), kwargs = {})
#   %view_254 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_107, [16, 2048, 768]), kwargs = {})
#   %add_160 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_159, %view_254), kwargs = {})
#   %convert_element_type_447 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_160, torch.float32), kwargs = {})
#   %select_20 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 10), kwargs = {})
#   %mul_283 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %convert_element_type_447), kwargs = {})
#   %sum_17 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_283, [2], True), kwargs = {})
#   %div_12 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_175, 768), kwargs = {})
#   %mul_284 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, %sum_17), kwargs = {})
#   %sub_14 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_447, %mul_284), kwargs = {})
#   %mul_285 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_41), kwargs = {})
#   %convert_element_type_449 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_285, torch.bfloat16), kwargs = {})
#   %add_161 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_152, %convert_element_type_449), kwargs = {})
#   %mul_287 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %cat), kwargs = {})
#   %sum_18 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_287,), kwargs = {})
#   %mul_288 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %select_20), kwargs = {})
#   %mul_289 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %add_116), kwargs = {})
#   %sum_19 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_289,), kwargs = {})
#   %view_255 : Tensor "bf16[32768, 768][768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_288, [32768, 768]), kwargs = {})
#   return %sum_17,%add_161,%view_255,%buf56,%buf58,%buf98,%buf100
triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_17 = async_compile.triton('triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 11, 'num_reduction': 5, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1179648, 'r0_': 704643072}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (10))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp25 = tl.load(in_ptr6 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr7 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp43 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = 0.0013020833333333333
    tmp14 = tmp0 * tmp13
    tmp15 = tmp14 * tmp11
    tmp16 = tmp6 - tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp12 + tmp19
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 * tmp23
    tmp27 = tmp25 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.where(r0_mask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
    tmp36 = tl.where(r0_mask, tmp34, 0)
    tmp37 = tl.sum(tmp36, 1)[:, None].to(tl.float32)
    tmp38 = tmp20 * tmp26
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.where(r0_mask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None].to(tl.float32)
    tmp44 = tmp20 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
    tmp47 = tl.where(r0_mask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp20, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp24, r0_mask)
    tl.store(out_ptr2 + (x0), tmp31, None)
    tl.store(out_ptr3 + (x0), tmp37, None)
    tl.store(out_ptr4 + (x0), tmp42, None)
    tl.store(out_ptr5 + (x0), tmp48, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/aq/caqychqofcavx7czjxovzstfoei4bqkap7ez657nsqxm335y22j7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf56 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=buf56]
#   %mul_255 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_151, %cat), kwargs = {})
#   %sum_12 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_255,), kwargs = {})
#   return %sum_12
triton_red_fused_mul_sum_18 = async_compile.triton('triton_red_fused_mul_sum_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 131072}}
)
@triton.jit
def triton_red_fused_mul_sum_18(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 32768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp2, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/lv/clvvjzdvu6aa4njshz2ypv3fs7xda5j2napfxwyqvc7735v4ejxk.py
# Topologically Sorted Source Nodes: [getitem_71, clone_19, rms_norm_40], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   clone_19 => convert_element_type_462
#   getitem_71 => select_20
#   rms_norm_40 => convert_element_type_273, mul_172
# Graph fragment:
#   %add_114 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_114]
#   %rsqrt_40 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_40]
#   %mm_111 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_111]
#   %add_161 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_161]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %sum_20 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_20]
#   %select_20 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 10), kwargs = {})
#   %mul_288 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %select_20), kwargs = {})
#   %view_258 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_111, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_462 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_258, torch.float32), kwargs = {})
#   %convert_element_type_273 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_114, torch.float32), kwargs = {})
#   %mul_172 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_273, %rsqrt_40), kwargs = {})
#   %mul_293 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %convert_element_type_462), kwargs = {})
#   %sum_20 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_293, [2], True), kwargs = {})
#   %div_13 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_172, 768), kwargs = {})
#   %mul_294 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_13, %sum_20), kwargs = {})
#   %sub_15 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_462, %mul_294), kwargs = {})
#   %mul_295 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %rsqrt_40), kwargs = {})
#   %convert_element_type_464 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_295, torch.bfloat16), kwargs = {})
#   %add_165 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_288, %convert_element_type_464), kwargs = {})
#   return %sum_20,%add_165
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_19 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (10))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0013020833333333333
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/ya/cyaalde4ipj6qoxy3romknlkuecfkm4iugkjr2u66u4o473wyfxw.py
# Topologically Sorted Source Nodes: [full_5, _generalized_scatter_17, clone_25, getitem_64, mul_118, getitem_65, mul_119, x_58, rms_norm_37, getitem_58, mul_106, getitem_59, mul_107, x_52, rms_norm_33], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   _generalized_scatter_17 => slice_scatter_default_13
#   clone_25 => convert_element_type_499
#   full_5 => full_default_10
#   getitem_58 => select_16
#   getitem_59 => select_17
#   getitem_64 => select_18
#   getitem_65 => select_19
#   mul_106 => mul_139
#   mul_107 => mul_140
#   mul_118 => mul_155
#   mul_119 => mul_156
#   rms_norm_33 => convert_element_type_226, mul_141
#   rms_norm_37 => convert_element_type_252, mul_157
#   x_52 => add_94
#   x_58 => add_105
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_104 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_104]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %rsqrt_37 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_37]
#   %add_93 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_93]
#   %rsqrt_33 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_33]
#   %mul_157 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=mul_157]
#   %mm_115 : Tensor "bf16[32768, 12][12, 1]cuda:0" = PlaceHolder[target=mm_115]
#   %mm_117 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_117]
#   %mm_119 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_119]
#   %mm_121 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_121]
#   %add_165 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_165]
#   %sum_24 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_24]
#   %add_175 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_175]
#   %full_default_10 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([16, 2048, 768], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_263 : Tensor "bf16[16, 2048, 12][24576, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_115, [16, 2048, 12]), kwargs = {})
#   %slice_scatter_default_13 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_263, 2, 0, 12), kwargs = {})
#   %view_267 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_117, [16, 2048, 768]), kwargs = {})
#   %add_172 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_13, %view_267), kwargs = {})
#   %view_270 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_119, [16, 2048, 768]), kwargs = {})
#   %add_173 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_172, %view_270), kwargs = {})
#   %view_273 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_121, [16, 2048, 768]), kwargs = {})
#   %add_174 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_173, %view_273), kwargs = {})
#   %convert_element_type_499 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_174, torch.float32), kwargs = {})
#   %select_18 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 9), kwargs = {})
#   %mul_155 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_18, %add_104), kwargs = {})
#   %select_19 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 9), kwargs = {})
#   %mul_156 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_19, %cat), kwargs = {})
#   %add_105 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %mul_156), kwargs = {})
#   %convert_element_type_252 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_105, torch.float32), kwargs = {})
#   %mul_157 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_252, %rsqrt_37), kwargs = {})
#   %mul_320 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_157, %convert_element_type_499), kwargs = {})
#   %sum_24 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_320, [2], True), kwargs = {})
#   %div_16 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_157, 768), kwargs = {})
#   %mul_321 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_16, %sum_24), kwargs = {})
#   %sub_19 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_499, %mul_321), kwargs = {})
#   %mul_322 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %rsqrt_37), kwargs = {})
#   %convert_element_type_501 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_322, torch.bfloat16), kwargs = {})
#   %add_175 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_165, %convert_element_type_501), kwargs = {})
#   %mul_325 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_175, %select_18), kwargs = {})
#   %view_274 : Tensor "bf16[32768, 768][768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_325, [32768, 768]), kwargs = {})
#   %select_16 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 8), kwargs = {})
#   %mul_139 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_16, %add_93), kwargs = {})
#   %select_17 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 8), kwargs = {})
#   %mul_140 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_17, %cat), kwargs = {})
#   %add_94 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_139, %mul_140), kwargs = {})
#   %convert_element_type_226 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_94, torch.float32), kwargs = {})
#   %mul_141 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_226, %rsqrt_33), kwargs = {})
#   return %mul_157,%mul_141,%sum_24,%add_175,%view_274
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_20 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 805306368}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (9))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    x0 = xindex
    tmp5 = tl.load(in_ptr2 + (9))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (8))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.load(in_ptr2 + (8))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp25 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    _tmp42 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr5 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr10 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp2 * tmp3
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp4 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp11 * tmp12
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp16 * tmp17
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21 * tmp8
        tmp23 = tmp18 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tmp27 = r0_1
        tmp28 = tl.full([1, 1], 12, tl.int64)
        tmp29 = tmp27 < tmp28
        tmp30 = tl.load(in_ptr7 + (r0_1 + 12*x0), r0_mask & tmp29, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = 0.0
        tmp32 = tl.where(tmp29, tmp30, tmp31)
        tmp34 = tmp32 + tmp33
        tmp36 = tmp34 + tmp35
        tmp38 = tmp36 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp13 * tmp39
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(r0_mask, tmp43, _tmp42)
        tl.store(out_ptr0 + (r0_1 + 768*x0), tmp13, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 768*x0), tmp26, r0_mask)
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp66 = tl.load(in_ptr0 + (9))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp44 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp51 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp55 = tl.load(in_ptr10 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp58 = tl.load(out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp45 = r0_1
        tmp46 = tl.full([1, 1], 12, tl.int64)
        tmp47 = tmp45 < tmp46
        tmp48 = tl.load(in_ptr7 + (r0_1 + 12*x0), r0_mask & tmp47, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp49 = 0.0
        tmp50 = tl.where(tmp47, tmp48, tmp49)
        tmp52 = tmp50 + tmp51
        tmp54 = tmp52 + tmp53
        tmp56 = tmp54 + tmp55
        tmp57 = tmp56.to(tl.float32)
        tmp59 = 0.0013020833333333333
        tmp60 = tmp58 * tmp59
        tmp61 = tmp60 * tmp42
        tmp62 = tmp57 - tmp61
        tmp63 = tmp62 * tmp12
        tmp64 = tmp63.to(tl.float32)
        tmp65 = tmp44 + tmp64
        tmp68 = tmp67.to(tl.float32)
        tmp69 = tmp65 * tmp68
        tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp65, r0_mask)
        tl.store(out_ptr3 + (r0_1 + 768*x0), tmp69, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/4t/c4t2xicrogih4ltqvd47sbvtpjmgvtv46nxhtpjcailxslzebdkh.py
# Topologically Sorted Source Nodes: [getitem_64, clone_27, rms_norm_36], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   clone_27 => convert_element_type_516
#   getitem_64 => select_18
#   rms_norm_36 => convert_element_type_244, mul_154
# Graph fragment:
#   %add_102 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_102]
#   %rsqrt_36 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_36]
#   %mm_125 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_125]
#   %add_175 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_175]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %sum_27 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_27]
#   %select_18 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 9), kwargs = {})
#   %mul_325 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_175, %select_18), kwargs = {})
#   %view_277 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_125, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_516 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_277, torch.float32), kwargs = {})
#   %convert_element_type_244 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_102, torch.float32), kwargs = {})
#   %mul_154 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_244, %rsqrt_36), kwargs = {})
#   %mul_330 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_154, %convert_element_type_516), kwargs = {})
#   %sum_27 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_330, [2], True), kwargs = {})
#   %div_17 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_154, 768), kwargs = {})
#   %mul_331 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_17, %sum_27), kwargs = {})
#   %sub_20 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_516, %mul_331), kwargs = {})
#   %mul_332 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %rsqrt_36), kwargs = {})
#   %convert_element_type_518 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_332, torch.bfloat16), kwargs = {})
#   %add_179 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_325, %convert_element_type_518), kwargs = {})
#   return %sum_27,%add_179
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_21 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (9))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0013020833333333333
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/tm/ctmqxf4ufwx7muass43ml3xngzeyiaeird5h3gde55vmkd4vjupf.py
# Topologically Sorted Source Nodes: [clone_33, getitem_58], Original ATen: [aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   clone_33 => convert_element_type_545
#   getitem_58 => select_16
# Graph fragment:
#   %mul_141 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=mul_141]
#   %mm_129 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_129]
#   %mm_131 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_131]
#   %mm_133 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_133]
#   %add_179 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_179]
#   %sum_30 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_30]
#   %rsqrt_33 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_33]
#   %add_188 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_188]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_175 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_175]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %add_104 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_104]
#   %add_93 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_93]
#   %mul_324 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_175, %cat), kwargs = {})
#   %sum_25 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_324,), kwargs = {})
#   %mul_326 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_175, %add_104), kwargs = {})
#   %sum_26 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_326,), kwargs = {})
#   %view_283 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_129, [16, 2048, 768]), kwargs = {})
#   %view_286 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_131, [16, 2048, 768]), kwargs = {})
#   %add_186 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_283, %view_286), kwargs = {})
#   %view_289 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_133, [16, 2048, 768]), kwargs = {})
#   %add_187 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_186, %view_289), kwargs = {})
#   %convert_element_type_545 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_187, torch.float32), kwargs = {})
#   %select_16 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 8), kwargs = {})
#   %mul_352 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_141, %convert_element_type_545), kwargs = {})
#   %sum_30 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_352, [2], True), kwargs = {})
#   %div_20 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_141, 768), kwargs = {})
#   %mul_353 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_20, %sum_30), kwargs = {})
#   %sub_23 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_545, %mul_353), kwargs = {})
#   %mul_354 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %rsqrt_33), kwargs = {})
#   %convert_element_type_547 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_354, torch.bfloat16), kwargs = {})
#   %add_188 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_179, %convert_element_type_547), kwargs = {})
#   %mul_356 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_188, %cat), kwargs = {})
#   %sum_31 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_356,), kwargs = {})
#   %mul_357 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_188, %select_16), kwargs = {})
#   %mul_358 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_188, %add_93), kwargs = {})
#   %sum_32 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_358,), kwargs = {})
#   %view_290 : Tensor "bf16[32768, 768][768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_357, [32768, 768]), kwargs = {})
#   return %sum_30,%add_188,%view_290,%buf148,%buf150,%buf190,%buf192
triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_22 = async_compile.triton('triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 11, 'num_reduction': 5, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1179648, 'r0_': 704643072}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (8))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp25 = tl.load(in_ptr6 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr7 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp43 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = 0.0013020833333333333
    tmp14 = tmp0 * tmp13
    tmp15 = tmp14 * tmp11
    tmp16 = tmp6 - tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp12 + tmp19
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 * tmp23
    tmp27 = tmp25 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.where(r0_mask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
    tmp36 = tl.where(r0_mask, tmp34, 0)
    tmp37 = tl.sum(tmp36, 1)[:, None].to(tl.float32)
    tmp38 = tmp20 * tmp26
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.where(r0_mask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None].to(tl.float32)
    tmp44 = tmp20 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
    tmp47 = tl.where(r0_mask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp20, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp24, r0_mask)
    tl.store(out_ptr2 + (x0), tmp31, None)
    tl.store(out_ptr3 + (x0), tmp37, None)
    tl.store(out_ptr4 + (x0), tmp42, None)
    tl.store(out_ptr5 + (x0), tmp48, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/6r/c6riihzmpcfi4omopgapkbig5b2kbe67dmrtzjsrgy6i3e6rojyh.py
# Topologically Sorted Source Nodes: [getitem_58, clone_35, rms_norm_32], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   clone_35 => convert_element_type_560
#   getitem_58 => select_16
#   rms_norm_32 => convert_element_type_218, mul_138
# Graph fragment:
#   %add_91 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_91]
#   %rsqrt_32 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_32]
#   %mm_137 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_137]
#   %add_188 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_188]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %sum_33 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_33]
#   %select_16 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 8), kwargs = {})
#   %mul_357 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_188, %select_16), kwargs = {})
#   %view_293 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_137, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_560 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_293, torch.float32), kwargs = {})
#   %convert_element_type_218 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_91, torch.float32), kwargs = {})
#   %mul_138 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_218, %rsqrt_32), kwargs = {})
#   %mul_362 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_138, %convert_element_type_560), kwargs = {})
#   %sum_33 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_362, [2], True), kwargs = {})
#   %div_21 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_138, 768), kwargs = {})
#   %mul_363 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_21, %sum_33), kwargs = {})
#   %sub_24 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_560, %mul_363), kwargs = {})
#   %mul_364 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_32), kwargs = {})
#   %convert_element_type_562 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_364, torch.bfloat16), kwargs = {})
#   %add_192 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_357, %convert_element_type_562), kwargs = {})
#   return %sum_33,%add_192
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_23 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (8))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0013020833333333333
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/uc/cuc2tcblza34qvcxckkun4yu2dkhiufxu2qehst6py34i4hrykuj.py
# Topologically Sorted Source Nodes: [clone_1, x_77, to_87, full_5, getitem_78, getitem_72, getitem_65, getitem_59, _generalized_scatter_30, clone_41, getitem_51, mul_92, getitem_52, mul_93, x_46, rms_norm_29, getitem_45, mul_80, getitem_46, mul_81, x_40, rms_norm_25], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.neg, aten.slice_backward, aten.select, aten.add]
# Source node to ATen node mapping:
#   _generalized_scatter_30 => slice_scatter_default_22
#   clone_1 => convert_element_type_350
#   clone_41 => convert_element_type_597
#   full_5 => full_default_10
#   getitem_45 => select_12
#   getitem_46 => select_13
#   getitem_51 => select_14
#   getitem_52 => select_15
#   getitem_59 => select_17
#   getitem_65 => select_19
#   getitem_72 => select_21
#   getitem_78 => select_23
#   mul_80 => mul_105
#   mul_81 => mul_106
#   mul_92 => mul_121
#   mul_93 => mul_122
#   rms_norm_25 => convert_element_type_171, mul_107
#   rms_norm_29 => convert_element_type_197, mul_123
#   to_87 => convert_element_type_336
#   x_40 => add_71
#   x_46 => add_82
#   x_77 => convert_element_type_337, mul_208
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_81 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_81]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %rsqrt_29 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_29]
#   %add_70 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_70]
#   %rsqrt_25 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_25]
#   %mul_123 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=mul_123]
#   %mm_141 : Tensor "bf16[32768, 12][12, 1]cuda:0" = PlaceHolder[target=mm_141]
#   %mm_143 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_143]
#   %mm_145 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_145]
#   %mm_147 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_147]
#   %add_192 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_192]
#   %sum_37 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_37]
#   %add_151 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_151]
#   %add_161 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_161]
#   %add_175 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_175]
#   %add_188 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_188]
#   %add_202 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_202]
#   %mm_81 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_81]
#   %sub : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=sub]
#   %rsqrt_49 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_49]
#   %sum_5 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_5]
#   %primals_93 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_93]
#   %view_218 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_81, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_350 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_218, torch.float32), kwargs = {})
#   %convert_element_type_337 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.float32), kwargs = {})
#   %mul_208 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_337, %rsqrt_49), kwargs = {})
#   %div_4 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_208, 768), kwargs = {})
#   %mul_217 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sum_5), kwargs = {})
#   %sub_5 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_350, %mul_217), kwargs = {})
#   %mul_218 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_49), kwargs = {})
#   %convert_element_type_352 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_218, torch.bfloat16), kwargs = {})
#   %neg_25 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_352,), kwargs = {})
#   %convert_element_type_336 : Tensor "bf16[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_93, torch.bfloat16), kwargs = {})
#   %mul_219 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_25, %convert_element_type_336), kwargs = {})
#   %full_default_10 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([16, 2048, 768], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_23 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 11), kwargs = {})
#   %mul_254 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_151, %select_23), kwargs = {})
#   %select_21 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 10), kwargs = {})
#   %mul_286 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %select_21), kwargs = {})
#   %add_162 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_254, %mul_286), kwargs = {})
#   %select_19 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 9), kwargs = {})
#   %mul_323 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_175, %select_19), kwargs = {})
#   %add_176 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_162, %mul_323), kwargs = {})
#   %select_17 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 8), kwargs = {})
#   %mul_355 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_188, %select_17), kwargs = {})
#   %add_189 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_176, %mul_355), kwargs = {})
#   %view_298 : Tensor "bf16[16, 2048, 12][24576, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_141, [16, 2048, 12]), kwargs = {})
#   %slice_scatter_default_22 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_298, 2, 0, 12), kwargs = {})
#   %view_302 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_143, [16, 2048, 768]), kwargs = {})
#   %add_199 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_22, %view_302), kwargs = {})
#   %view_305 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_145, [16, 2048, 768]), kwargs = {})
#   %add_200 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_199, %view_305), kwargs = {})
#   %view_308 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_147, [16, 2048, 768]), kwargs = {})
#   %add_201 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_200, %view_308), kwargs = {})
#   %convert_element_type_597 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_201, torch.float32), kwargs = {})
#   %select_14 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 7), kwargs = {})
#   %mul_121 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_14, %add_81), kwargs = {})
#   %select_15 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 7), kwargs = {})
#   %mul_122 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_15, %cat), kwargs = {})
#   %add_82 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_121, %mul_122), kwargs = {})
#   %convert_element_type_197 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_82, torch.float32), kwargs = {})
#   %mul_123 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_197, %rsqrt_29), kwargs = {})
#   %mul_389 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_123, %convert_element_type_597), kwargs = {})
#   %sum_37 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_389, [2], True), kwargs = {})
#   %div_24 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_123, 768), kwargs = {})
#   %mul_390 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_24, %sum_37), kwargs = {})
#   %sub_28 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_597, %mul_390), kwargs = {})
#   %mul_391 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_29), kwargs = {})
#   %convert_element_type_599 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_391, torch.bfloat16), kwargs = {})
#   %add_202 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_192, %convert_element_type_599), kwargs = {})
#   %mul_392 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_202, %select_15), kwargs = {})
#   %add_203 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_189, %mul_392), kwargs = {})
#   %mul_394 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_202, %select_14), kwargs = {})
#   %add_205 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_219, %mul_394), kwargs = {})
#   %select_12 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 6), kwargs = {})
#   %mul_105 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_12, %add_70), kwargs = {})
#   %select_13 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 6), kwargs = {})
#   %mul_106 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_13, %cat), kwargs = {})
#   %add_71 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_105, %mul_106), kwargs = {})
#   %convert_element_type_171 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_71, torch.float32), kwargs = {})
#   %mul_107 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_171, %rsqrt_25), kwargs = {})
#   return %mul_123,%mul_107,%sum_37,%add_202,%add_203,%add_205
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_neg_select_slice_backward_view_24 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_neg_select_slice_backward_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_out_ptr2': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*bf16', 'in_ptr12': '*bf16', 'in_ptr13': '*bf16', 'in_ptr14': '*bf16', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]], (23,): [['tt.divisibility', 16]], (24,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_neg_select_slice_backward_view_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 34, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288, 'r0_': 1207959552}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_neg_select_slice_backward_view_24(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (7))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    x0 = xindex
    tmp5 = tl.load(in_ptr2 + (7))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (6))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.load(in_ptr2 + (6))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp25 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    _tmp42 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr5 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr10 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp2 * tmp3
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp4 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp11 * tmp12
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp16 * tmp17
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21 * tmp8
        tmp23 = tmp18 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tmp27 = r0_1
        tmp28 = tl.full([1, 1], 12, tl.int64)
        tmp29 = tmp27 < tmp28
        tmp30 = tl.load(in_ptr7 + (r0_1 + 12*x0), r0_mask & tmp29, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = 0.0
        tmp32 = tl.where(tmp29, tmp30, tmp31)
        tmp34 = tmp32 + tmp33
        tmp36 = tmp34 + tmp35
        tmp38 = tmp36 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp13 * tmp39
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(r0_mask, tmp43, _tmp42)
        tl.store(out_ptr0 + (r0_1 + 768*x0), tmp13, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 768*x0), tmp26, r0_mask)
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp67 = tl.load(in_ptr2 + (11))
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK, R0_BLOCK])
    tmp72 = tl.load(in_ptr2 + (10))
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, R0_BLOCK])
    tmp78 = tl.load(in_ptr2 + (9))
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK, R0_BLOCK])
    tmp84 = tl.load(in_ptr2 + (8))
    tmp85 = tl.broadcast_to(tmp84, [XBLOCK, R0_BLOCK])
    tmp89 = tl.load(in_ptr2 + (7))
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, R0_BLOCK])
    tmp98 = tl.load(in_ptr15 + (x0), None, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr16 + (x0), None, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr17 + (0))
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK, R0_BLOCK])
    tmp111 = tl.load(in_ptr0 + (7))
    tmp112 = tl.broadcast_to(tmp111, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp44 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp51 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp55 = tl.load(in_ptr10 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp58 = tl.load(out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp66 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp71 = tl.load(in_ptr11 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp77 = tl.load(in_ptr12 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp83 = tl.load(in_ptr13 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp94 = tl.load(in_out_ptr2 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp96 = tl.load(in_ptr14 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp45 = r0_1
        tmp46 = tl.full([1, 1], 12, tl.int64)
        tmp47 = tmp45 < tmp46
        tmp48 = tl.load(in_ptr7 + (r0_1 + 12*x0), r0_mask & tmp47, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp49 = 0.0
        tmp50 = tl.where(tmp47, tmp48, tmp49)
        tmp52 = tmp50 + tmp51
        tmp54 = tmp52 + tmp53
        tmp56 = tmp54 + tmp55
        tmp57 = tmp56.to(tl.float32)
        tmp59 = 0.0013020833333333333
        tmp60 = tmp58 * tmp59
        tmp61 = tmp60 * tmp42
        tmp62 = tmp57 - tmp61
        tmp63 = tmp62 * tmp12
        tmp64 = tmp63.to(tl.float32)
        tmp65 = tmp44 + tmp64
        tmp69 = tmp68.to(tl.float32)
        tmp70 = tmp66 * tmp69
        tmp74 = tmp73.to(tl.float32)
        tmp75 = tmp71 * tmp74
        tmp76 = tmp70 + tmp75
        tmp80 = tmp79.to(tl.float32)
        tmp81 = tmp77 * tmp80
        tmp82 = tmp76 + tmp81
        tmp86 = tmp85.to(tl.float32)
        tmp87 = tmp83 * tmp86
        tmp88 = tmp82 + tmp87
        tmp91 = tmp90.to(tl.float32)
        tmp92 = tmp65 * tmp91
        tmp93 = tmp88 + tmp92
        tmp95 = tmp94.to(tl.float32)
        tmp97 = tmp96.to(tl.float32)
        tmp99 = tmp97 * tmp98
        tmp100 = tmp99 * tmp59
        tmp102 = tmp100 * tmp101
        tmp103 = tmp95 - tmp102
        tmp104 = tmp103 * tmp98
        tmp105 = tmp104.to(tl.float32)
        tmp106 = -tmp105
        tmp109 = tmp108.to(tl.float32)
        tmp110 = tmp106 * tmp109
        tmp113 = tmp112.to(tl.float32)
        tmp114 = tmp65 * tmp113
        tmp115 = tmp110 + tmp114
        tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp65, r0_mask)
        tl.store(in_out_ptr1 + (r0_1 + 768*x0), tmp93, r0_mask)
        tl.store(in_out_ptr2 + (r0_1 + 768*x0), tmp115, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/zk/czk4kq3vmm22a6sojma5tgamwev5ymqj62s7k2hcym4flcngt6h2.py
# Topologically Sorted Source Nodes: [clone_43, rms_norm_28], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
# Source node to ATen node mapping:
#   clone_43 => convert_element_type_614
#   rms_norm_28 => convert_element_type_189, mul_120
# Graph fragment:
#   %add_79 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_79]
#   %rsqrt_28 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_28]
#   %mm_151 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_151]
#   %add_205 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_205]
#   %sum_40 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_40]
#   %view_312 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_151, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_614 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_312, torch.float32), kwargs = {})
#   %convert_element_type_189 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_79, torch.float32), kwargs = {})
#   %mul_120 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_189, %rsqrt_28), kwargs = {})
#   %mul_399 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_120, %convert_element_type_614), kwargs = {})
#   %sum_40 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_399, [2], True), kwargs = {})
#   %div_25 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_120, 768), kwargs = {})
#   %mul_400 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_25, %sum_40), kwargs = {})
#   %sub_29 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_614, %mul_400), kwargs = {})
#   %mul_401 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %rsqrt_28), kwargs = {})
#   %convert_element_type_616 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_401, torch.bfloat16), kwargs = {})
#   %add_207 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_205, %convert_element_type_616), kwargs = {})
#   return %sum_40,%add_207
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_25 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp12 = 0.0013020833333333333
    tmp13 = tmp3 * tmp12
    tmp14 = tmp13 * tmp10
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp2
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp11 + tmp17
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp18, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/3u/c3ujy3wrfyvu3xl7sazgzb7mt4tm7c3nkl32ojts7hvcp6pyzst5.py
# Topologically Sorted Source Nodes: [clone_49, getitem_45], Original ATen: [aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   clone_49 => convert_element_type_643
#   getitem_45 => select_12
# Graph fragment:
#   %mul_107 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=mul_107]
#   %mm_155 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_155]
#   %mm_157 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_157]
#   %mm_159 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_159]
#   %add_207 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_207]
#   %sum_43 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_43]
#   %rsqrt_25 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_25]
#   %add_216 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_216]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_202 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_202]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %add_81 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_81]
#   %add_70 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_70]
#   %mul_393 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_202, %cat), kwargs = {})
#   %sum_38 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_393,), kwargs = {})
#   %mul_395 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_202, %add_81), kwargs = {})
#   %sum_39 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_395,), kwargs = {})
#   %view_318 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_155, [16, 2048, 768]), kwargs = {})
#   %view_321 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_157, [16, 2048, 768]), kwargs = {})
#   %add_214 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_318, %view_321), kwargs = {})
#   %view_324 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_159, [16, 2048, 768]), kwargs = {})
#   %add_215 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_214, %view_324), kwargs = {})
#   %convert_element_type_643 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_215, torch.float32), kwargs = {})
#   %select_12 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 6), kwargs = {})
#   %mul_421 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_107, %convert_element_type_643), kwargs = {})
#   %sum_43 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_421, [2], True), kwargs = {})
#   %div_28 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_107, 768), kwargs = {})
#   %mul_422 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_28, %sum_43), kwargs = {})
#   %sub_32 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_643, %mul_422), kwargs = {})
#   %mul_423 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %rsqrt_25), kwargs = {})
#   %convert_element_type_645 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_423, torch.bfloat16), kwargs = {})
#   %add_216 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_207, %convert_element_type_645), kwargs = {})
#   %mul_425 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_216, %cat), kwargs = {})
#   %sum_44 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_425,), kwargs = {})
#   %mul_426 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_216, %select_12), kwargs = {})
#   %mul_427 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_216, %add_70), kwargs = {})
#   %sum_45 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_427,), kwargs = {})
#   %view_325 : Tensor "bf16[32768, 768][768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_426, [32768, 768]), kwargs = {})
#   return %sum_43,%add_216,%view_325,%buf240,%buf243,%buf283,%buf285
triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_26 = async_compile.triton('triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 11, 'num_reduction': 5, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1179648, 'r0_': 704643072}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (6))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp25 = tl.load(in_ptr6 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr7 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp43 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = 0.0013020833333333333
    tmp14 = tmp0 * tmp13
    tmp15 = tmp14 * tmp11
    tmp16 = tmp6 - tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp12 + tmp19
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 * tmp23
    tmp27 = tmp25 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.where(r0_mask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
    tmp36 = tl.where(r0_mask, tmp34, 0)
    tmp37 = tl.sum(tmp36, 1)[:, None].to(tl.float32)
    tmp38 = tmp20 * tmp26
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.where(r0_mask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None].to(tl.float32)
    tmp44 = tmp20 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
    tmp47 = tl.where(r0_mask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp20, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp24, r0_mask)
    tl.store(out_ptr2 + (x0), tmp31, None)
    tl.store(out_ptr3 + (x0), tmp37, None)
    tl.store(out_ptr4 + (x0), tmp42, None)
    tl.store(out_ptr5 + (x0), tmp48, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/oh/coheq5nziw6su2o3lr4icpyguezsgk26plz2vw2c6sihlx325s2b.py
# Topologically Sorted Source Nodes: [getitem_45, clone_51, rms_norm_24], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   clone_51 => convert_element_type_658
#   getitem_45 => select_12
#   rms_norm_24 => convert_element_type_163, mul_104
# Graph fragment:
#   %add_68 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_68]
#   %rsqrt_24 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_24]
#   %mm_163 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_163]
#   %add_216 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_216]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %sum_46 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_46]
#   %select_12 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 6), kwargs = {})
#   %mul_426 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_216, %select_12), kwargs = {})
#   %view_328 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_163, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_658 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_328, torch.float32), kwargs = {})
#   %convert_element_type_163 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_68, torch.float32), kwargs = {})
#   %mul_104 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_163, %rsqrt_24), kwargs = {})
#   %mul_431 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_104, %convert_element_type_658), kwargs = {})
#   %sum_46 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_431, [2], True), kwargs = {})
#   %div_29 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_104, 768), kwargs = {})
#   %mul_432 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_29, %sum_46), kwargs = {})
#   %sub_33 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_658, %mul_432), kwargs = {})
#   %mul_433 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %rsqrt_24), kwargs = {})
#   %convert_element_type_660 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_433, torch.bfloat16), kwargs = {})
#   %add_220 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_426, %convert_element_type_660), kwargs = {})
#   return %sum_46,%add_220
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_27 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (6))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0013020833333333333
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/ps/cpseduydkgpqyfc6ueqowyrbqmgffafpxf3n2nye3tfnikr6ytbh.py
# Topologically Sorted Source Nodes: [full_5, _generalized_scatter_43, clone_57, getitem_38, mul_66, getitem_39, mul_67, x_34, rms_norm_21, getitem_32, mul_54, getitem_33, mul_55, x_28, rms_norm_17], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   _generalized_scatter_43 => slice_scatter_default_31
#   clone_57 => convert_element_type_695
#   full_5 => full_default_10
#   getitem_32 => select_8
#   getitem_33 => select_9
#   getitem_38 => select_10
#   getitem_39 => select_11
#   mul_54 => mul_71
#   mul_55 => mul_72
#   mul_66 => mul_87
#   mul_67 => mul_88
#   rms_norm_17 => convert_element_type_116, mul_73
#   rms_norm_21 => convert_element_type_142, mul_89
#   x_28 => add_48
#   x_34 => add_59
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_58 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_58]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %rsqrt_21 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_21]
#   %add_47 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_47]
#   %rsqrt_17 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_17]
#   %mul_89 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=mul_89]
#   %mm_167 : Tensor "bf16[32768, 12][12, 1]cuda:0" = PlaceHolder[target=mm_167]
#   %mm_169 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_169]
#   %mm_171 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_171]
#   %mm_173 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_173]
#   %add_220 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_220]
#   %sum_50 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_50]
#   %add_230 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_230]
#   %full_default_10 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([16, 2048, 768], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_333 : Tensor "bf16[16, 2048, 12][24576, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_167, [16, 2048, 12]), kwargs = {})
#   %slice_scatter_default_31 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_333, 2, 0, 12), kwargs = {})
#   %view_337 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_169, [16, 2048, 768]), kwargs = {})
#   %add_227 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_31, %view_337), kwargs = {})
#   %view_340 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_171, [16, 2048, 768]), kwargs = {})
#   %add_228 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_227, %view_340), kwargs = {})
#   %view_343 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_173, [16, 2048, 768]), kwargs = {})
#   %add_229 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_228, %view_343), kwargs = {})
#   %convert_element_type_695 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_229, torch.float32), kwargs = {})
#   %select_10 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 5), kwargs = {})
#   %mul_87 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_10, %add_58), kwargs = {})
#   %select_11 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 5), kwargs = {})
#   %mul_88 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_11, %cat), kwargs = {})
#   %add_59 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %mul_88), kwargs = {})
#   %convert_element_type_142 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_59, torch.float32), kwargs = {})
#   %mul_89 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_142, %rsqrt_21), kwargs = {})
#   %mul_458 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, %convert_element_type_695), kwargs = {})
#   %sum_50 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_458, [2], True), kwargs = {})
#   %div_32 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_89, 768), kwargs = {})
#   %mul_459 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_32, %sum_50), kwargs = {})
#   %sub_37 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_695, %mul_459), kwargs = {})
#   %mul_460 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %rsqrt_21), kwargs = {})
#   %convert_element_type_697 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_460, torch.bfloat16), kwargs = {})
#   %add_230 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_220, %convert_element_type_697), kwargs = {})
#   %mul_463 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, %select_10), kwargs = {})
#   %view_344 : Tensor "bf16[32768, 768][768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_463, [32768, 768]), kwargs = {})
#   %select_8 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 4), kwargs = {})
#   %mul_71 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %add_47), kwargs = {})
#   %select_9 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 4), kwargs = {})
#   %mul_72 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, %cat), kwargs = {})
#   %add_48 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %mul_72), kwargs = {})
#   %convert_element_type_116 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_48, torch.float32), kwargs = {})
#   %mul_73 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_116, %rsqrt_17), kwargs = {})
#   return %mul_89,%mul_73,%sum_50,%add_230,%view_344
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_28 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 805306368}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (5))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    x0 = xindex
    tmp5 = tl.load(in_ptr2 + (5))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (4))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.load(in_ptr2 + (4))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp25 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    _tmp42 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr5 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr10 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp2 * tmp3
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp4 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp11 * tmp12
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp16 * tmp17
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21 * tmp8
        tmp23 = tmp18 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tmp27 = r0_1
        tmp28 = tl.full([1, 1], 12, tl.int64)
        tmp29 = tmp27 < tmp28
        tmp30 = tl.load(in_ptr7 + (r0_1 + 12*x0), r0_mask & tmp29, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = 0.0
        tmp32 = tl.where(tmp29, tmp30, tmp31)
        tmp34 = tmp32 + tmp33
        tmp36 = tmp34 + tmp35
        tmp38 = tmp36 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp13 * tmp39
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(r0_mask, tmp43, _tmp42)
        tl.store(out_ptr0 + (r0_1 + 768*x0), tmp13, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 768*x0), tmp26, r0_mask)
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp66 = tl.load(in_ptr0 + (5))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp44 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp51 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp55 = tl.load(in_ptr10 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp58 = tl.load(out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp45 = r0_1
        tmp46 = tl.full([1, 1], 12, tl.int64)
        tmp47 = tmp45 < tmp46
        tmp48 = tl.load(in_ptr7 + (r0_1 + 12*x0), r0_mask & tmp47, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp49 = 0.0
        tmp50 = tl.where(tmp47, tmp48, tmp49)
        tmp52 = tmp50 + tmp51
        tmp54 = tmp52 + tmp53
        tmp56 = tmp54 + tmp55
        tmp57 = tmp56.to(tl.float32)
        tmp59 = 0.0013020833333333333
        tmp60 = tmp58 * tmp59
        tmp61 = tmp60 * tmp42
        tmp62 = tmp57 - tmp61
        tmp63 = tmp62 * tmp12
        tmp64 = tmp63.to(tl.float32)
        tmp65 = tmp44 + tmp64
        tmp68 = tmp67.to(tl.float32)
        tmp69 = tmp65 * tmp68
        tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp65, r0_mask)
        tl.store(out_ptr3 + (r0_1 + 768*x0), tmp69, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/ri/criami75fwtzkjwie4t42w5yt2kscf2bas6x7suw76uxaxvlp3ze.py
# Topologically Sorted Source Nodes: [getitem_38, clone_59, rms_norm_20], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   clone_59 => convert_element_type_712
#   getitem_38 => select_10
#   rms_norm_20 => convert_element_type_134, mul_86
# Graph fragment:
#   %add_56 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_56]
#   %rsqrt_20 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_20]
#   %mm_177 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_177]
#   %add_230 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_230]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %sum_53 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_53]
#   %select_10 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 5), kwargs = {})
#   %mul_463 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, %select_10), kwargs = {})
#   %view_347 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_177, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_712 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_347, torch.float32), kwargs = {})
#   %convert_element_type_134 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_56, torch.float32), kwargs = {})
#   %mul_86 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_134, %rsqrt_20), kwargs = {})
#   %mul_468 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_86, %convert_element_type_712), kwargs = {})
#   %sum_53 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_468, [2], True), kwargs = {})
#   %div_33 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_86, 768), kwargs = {})
#   %mul_469 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_33, %sum_53), kwargs = {})
#   %sub_38 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_712, %mul_469), kwargs = {})
#   %mul_470 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %rsqrt_20), kwargs = {})
#   %convert_element_type_714 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_470, torch.bfloat16), kwargs = {})
#   %add_234 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_463, %convert_element_type_714), kwargs = {})
#   return %sum_53,%add_234
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (5))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0013020833333333333
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/pd/cpddoyfcdsl2lsw6odl3pkop6vr4wl5r366xskxxzaijqzm4kecl.py
# Topologically Sorted Source Nodes: [clone_65, getitem_32], Original ATen: [aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   clone_65 => convert_element_type_741
#   getitem_32 => select_8
# Graph fragment:
#   %mul_73 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=mul_73]
#   %mm_181 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_181]
#   %mm_183 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_183]
#   %mm_185 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_185]
#   %add_234 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_234]
#   %sum_56 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_56]
#   %rsqrt_17 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_17]
#   %add_243 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_243]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_230 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_230]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %add_58 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_58]
#   %add_47 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_47]
#   %mul_462 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, %cat), kwargs = {})
#   %sum_51 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_462,), kwargs = {})
#   %mul_464 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, %add_58), kwargs = {})
#   %sum_52 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_464,), kwargs = {})
#   %view_353 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_181, [16, 2048, 768]), kwargs = {})
#   %view_356 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_183, [16, 2048, 768]), kwargs = {})
#   %add_241 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_353, %view_356), kwargs = {})
#   %view_359 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_185, [16, 2048, 768]), kwargs = {})
#   %add_242 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_241, %view_359), kwargs = {})
#   %convert_element_type_741 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_242, torch.float32), kwargs = {})
#   %select_8 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 4), kwargs = {})
#   %mul_490 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %convert_element_type_741), kwargs = {})
#   %sum_56 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_490, [2], True), kwargs = {})
#   %div_36 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_73, 768), kwargs = {})
#   %mul_491 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_36, %sum_56), kwargs = {})
#   %sub_41 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_741, %mul_491), kwargs = {})
#   %mul_492 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %rsqrt_17), kwargs = {})
#   %convert_element_type_743 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_492, torch.bfloat16), kwargs = {})
#   %add_243 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_234, %convert_element_type_743), kwargs = {})
#   %mul_494 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_243, %cat), kwargs = {})
#   %sum_57 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_494,), kwargs = {})
#   %mul_495 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_243, %select_8), kwargs = {})
#   %mul_496 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_243, %add_47), kwargs = {})
#   %sum_58 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_496,), kwargs = {})
#   %view_360 : Tensor "bf16[32768, 768][768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_495, [32768, 768]), kwargs = {})
#   return %sum_56,%add_243,%view_360,%buf333,%buf335,%buf375,%buf377
triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_30 = async_compile.triton('triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 11, 'num_reduction': 5, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1179648, 'r0_': 704643072}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (4))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp25 = tl.load(in_ptr6 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr7 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp43 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = 0.0013020833333333333
    tmp14 = tmp0 * tmp13
    tmp15 = tmp14 * tmp11
    tmp16 = tmp6 - tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp12 + tmp19
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 * tmp23
    tmp27 = tmp25 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.where(r0_mask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
    tmp36 = tl.where(r0_mask, tmp34, 0)
    tmp37 = tl.sum(tmp36, 1)[:, None].to(tl.float32)
    tmp38 = tmp20 * tmp26
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.where(r0_mask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None].to(tl.float32)
    tmp44 = tmp20 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
    tmp47 = tl.where(r0_mask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp20, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp24, r0_mask)
    tl.store(out_ptr2 + (x0), tmp31, None)
    tl.store(out_ptr3 + (x0), tmp37, None)
    tl.store(out_ptr4 + (x0), tmp42, None)
    tl.store(out_ptr5 + (x0), tmp48, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/qa/cqaqckk7jf6gtwyxze4sowtnas4twvsciur3obmbqqn2ejjmemgm.py
# Topologically Sorted Source Nodes: [getitem_32, clone_67, rms_norm_16], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   clone_67 => convert_element_type_756
#   getitem_32 => select_8
#   rms_norm_16 => convert_element_type_108, mul_70
# Graph fragment:
#   %add_45 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_45]
#   %rsqrt_16 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_16]
#   %mm_189 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_189]
#   %add_243 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_243]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %sum_59 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_59]
#   %select_8 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 4), kwargs = {})
#   %mul_495 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_243, %select_8), kwargs = {})
#   %view_363 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_189, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_756 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_363, torch.float32), kwargs = {})
#   %convert_element_type_108 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_45, torch.float32), kwargs = {})
#   %mul_70 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_108, %rsqrt_16), kwargs = {})
#   %mul_500 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %convert_element_type_756), kwargs = {})
#   %sum_59 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_500, [2], True), kwargs = {})
#   %div_37 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_70, 768), kwargs = {})
#   %mul_501 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_37, %sum_59), kwargs = {})
#   %sub_42 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_756, %mul_501), kwargs = {})
#   %mul_502 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %rsqrt_16), kwargs = {})
#   %convert_element_type_758 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_502, torch.bfloat16), kwargs = {})
#   %add_247 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_495, %convert_element_type_758), kwargs = {})
#   return %sum_59,%add_247
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_31 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (4))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0013020833333333333
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/ak/cakbbdnxe5zwbzu7pz7va4jqrqnoksexmxfberot4c3jyci644nl.py
# Topologically Sorted Source Nodes: [full_5, getitem_46, getitem_39, getitem_33, _generalized_scatter_56, clone_73, getitem_25, mul_40, getitem_26, mul_41, x_22, rms_norm_13, getitem_19, mul_28, getitem_20, mul_29, x_16, rms_norm_9], Original ATen: [aten.slice_backward, aten.select, aten.mul, aten.add, aten.view, aten._fused_rms_norm_backward, aten._to_copy]
# Source node to ATen node mapping:
#   _generalized_scatter_56 => slice_scatter_default_40
#   clone_73 => convert_element_type_793
#   full_5 => full_default_10
#   getitem_19 => select_4
#   getitem_20 => select_5
#   getitem_25 => select_6
#   getitem_26 => select_7
#   getitem_33 => select_9
#   getitem_39 => select_11
#   getitem_46 => select_13
#   mul_28 => mul_37
#   mul_29 => mul_38
#   mul_40 => mul_53
#   mul_41 => mul_54
#   rms_norm_13 => convert_element_type_87, mul_55
#   rms_norm_9 => convert_element_type_61, mul_39
#   x_16 => add_25
#   x_22 => add_36
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_35 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_35]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %rsqrt_13 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_13]
#   %add_24 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_24]
#   %rsqrt_9 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_9]
#   %mul_55 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=mul_55]
#   %mm_193 : Tensor "bf16[32768, 12][12, 1]cuda:0" = PlaceHolder[target=mm_193]
#   %mm_195 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_195]
#   %mm_197 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_197]
#   %mm_199 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_199]
#   %add_247 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_247]
#   %sum_63 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_63]
#   %add_203 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_203]
#   %add_216 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_216]
#   %add_230 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_230]
#   %add_243 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_243]
#   %add_257 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_257]
#   %full_default_10 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([16, 2048, 768], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_13 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 6), kwargs = {})
#   %mul_424 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_216, %select_13), kwargs = {})
#   %add_217 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_203, %mul_424), kwargs = {})
#   %select_11 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 5), kwargs = {})
#   %mul_461 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, %select_11), kwargs = {})
#   %add_231 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_217, %mul_461), kwargs = {})
#   %select_9 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 4), kwargs = {})
#   %mul_493 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_243, %select_9), kwargs = {})
#   %add_244 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_231, %mul_493), kwargs = {})
#   %view_368 : Tensor "bf16[16, 2048, 12][24576, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_193, [16, 2048, 12]), kwargs = {})
#   %slice_scatter_default_40 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_368, 2, 0, 12), kwargs = {})
#   %view_372 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_195, [16, 2048, 768]), kwargs = {})
#   %add_254 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_40, %view_372), kwargs = {})
#   %view_375 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_197, [16, 2048, 768]), kwargs = {})
#   %add_255 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_254, %view_375), kwargs = {})
#   %view_378 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_199, [16, 2048, 768]), kwargs = {})
#   %add_256 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_255, %view_378), kwargs = {})
#   %convert_element_type_793 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_256, torch.float32), kwargs = {})
#   %select_6 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 3), kwargs = {})
#   %mul_53 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_6, %add_35), kwargs = {})
#   %select_7 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 3), kwargs = {})
#   %mul_54 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_7, %cat), kwargs = {})
#   %add_36 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %mul_54), kwargs = {})
#   %convert_element_type_87 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_36, torch.float32), kwargs = {})
#   %mul_55 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_87, %rsqrt_13), kwargs = {})
#   %mul_527 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %convert_element_type_793), kwargs = {})
#   %sum_63 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_527, [2], True), kwargs = {})
#   %div_40 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_55, 768), kwargs = {})
#   %mul_528 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_40, %sum_63), kwargs = {})
#   %sub_46 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_793, %mul_528), kwargs = {})
#   %mul_529 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %rsqrt_13), kwargs = {})
#   %convert_element_type_795 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_529, torch.bfloat16), kwargs = {})
#   %add_257 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_247, %convert_element_type_795), kwargs = {})
#   %mul_530 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_257, %select_7), kwargs = {})
#   %add_258 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_244, %mul_530), kwargs = {})
#   %mul_532 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_257, %select_6), kwargs = {})
#   %view_379 : Tensor "bf16[32768, 768][768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_532, [32768, 768]), kwargs = {})
#   %select_4 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 2), kwargs = {})
#   %mul_37 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, %add_24), kwargs = {})
#   %select_5 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 2), kwargs = {})
#   %mul_38 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %cat), kwargs = {})
#   %add_25 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %mul_38), kwargs = {})
#   %convert_element_type_61 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_25, torch.float32), kwargs = {})
#   %mul_39 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_61, %rsqrt_9), kwargs = {})
#   return %mul_55,%mul_39,%sum_63,%add_257,%add_258,%view_379
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_32 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*bf16', 'in_ptr12': '*bf16', 'in_ptr13': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_32', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 28, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 1107296256}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_32(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (3))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    x0 = xindex
    tmp5 = tl.load(in_ptr2 + (3))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (2))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.load(in_ptr2 + (2))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp25 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    _tmp42 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr5 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr10 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp2 * tmp3
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp4 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp11 * tmp12
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp16 * tmp17
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21 * tmp8
        tmp23 = tmp18 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tmp27 = r0_1
        tmp28 = tl.full([1, 1], 12, tl.int64)
        tmp29 = tmp27 < tmp28
        tmp30 = tl.load(in_ptr7 + (r0_1 + 12*x0), r0_mask & tmp29, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = 0.0
        tmp32 = tl.where(tmp29, tmp30, tmp31)
        tmp34 = tmp32 + tmp33
        tmp36 = tmp34 + tmp35
        tmp38 = tmp36 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp13 * tmp39
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(r0_mask, tmp43, _tmp42)
        tl.store(out_ptr0 + (r0_1 + 768*x0), tmp13, r0_mask)
        tl.store(out_ptr1 + (r0_1 + 768*x0), tmp26, r0_mask)
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp68 = tl.load(in_ptr2 + (6))
    tmp69 = tl.broadcast_to(tmp68, [XBLOCK, R0_BLOCK])
    tmp74 = tl.load(in_ptr2 + (5))
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK, R0_BLOCK])
    tmp80 = tl.load(in_ptr2 + (4))
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, R0_BLOCK])
    tmp85 = tl.load(in_ptr2 + (3))
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK, R0_BLOCK])
    tmp90 = tl.load(in_ptr0 + (3))
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp44 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp51 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp55 = tl.load(in_ptr10 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp58 = tl.load(out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp66 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp67 = tl.load(in_ptr11 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp73 = tl.load(in_ptr12 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp79 = tl.load(in_ptr13 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp45 = r0_1
        tmp46 = tl.full([1, 1], 12, tl.int64)
        tmp47 = tmp45 < tmp46
        tmp48 = tl.load(in_ptr7 + (r0_1 + 12*x0), r0_mask & tmp47, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp49 = 0.0
        tmp50 = tl.where(tmp47, tmp48, tmp49)
        tmp52 = tmp50 + tmp51
        tmp54 = tmp52 + tmp53
        tmp56 = tmp54 + tmp55
        tmp57 = tmp56.to(tl.float32)
        tmp59 = 0.0013020833333333333
        tmp60 = tmp58 * tmp59
        tmp61 = tmp60 * tmp42
        tmp62 = tmp57 - tmp61
        tmp63 = tmp62 * tmp12
        tmp64 = tmp63.to(tl.float32)
        tmp65 = tmp44 + tmp64
        tmp70 = tmp69.to(tl.float32)
        tmp71 = tmp67 * tmp70
        tmp72 = tmp66 + tmp71
        tmp76 = tmp75.to(tl.float32)
        tmp77 = tmp73 * tmp76
        tmp78 = tmp72 + tmp77
        tmp82 = tmp81.to(tl.float32)
        tmp83 = tmp79 * tmp82
        tmp84 = tmp78 + tmp83
        tmp87 = tmp86.to(tl.float32)
        tmp88 = tmp65 * tmp87
        tmp89 = tmp84 + tmp88
        tmp92 = tmp91.to(tl.float32)
        tmp93 = tmp65 * tmp92
        tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp65, r0_mask)
        tl.store(in_out_ptr1 + (r0_1 + 768*x0), tmp89, r0_mask)
        tl.store(out_ptr3 + (r0_1 + 768*x0), tmp93, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/ls/clsu23tsye6auxmsfer2rf3l6xvidign6skrr2s2yq7fbj5s7g56.py
# Topologically Sorted Source Nodes: [getitem_25, clone_75, rms_norm_12], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   clone_75 => convert_element_type_810
#   getitem_25 => select_6
#   rms_norm_12 => convert_element_type_79, mul_52
# Graph fragment:
#   %add_33 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_33]
#   %rsqrt_12 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_12]
#   %mm_203 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_203]
#   %add_257 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_257]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %sum_66 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_66]
#   %select_6 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 3), kwargs = {})
#   %mul_532 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_257, %select_6), kwargs = {})
#   %view_382 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_203, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_810 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_382, torch.float32), kwargs = {})
#   %convert_element_type_79 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.float32), kwargs = {})
#   %mul_52 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_79, %rsqrt_12), kwargs = {})
#   %mul_537 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %convert_element_type_810), kwargs = {})
#   %sum_66 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_537, [2], True), kwargs = {})
#   %div_41 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_52, 768), kwargs = {})
#   %mul_538 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_41, %sum_66), kwargs = {})
#   %sub_47 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_810, %mul_538), kwargs = {})
#   %mul_539 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %rsqrt_12), kwargs = {})
#   %convert_element_type_812 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_539, torch.bfloat16), kwargs = {})
#   %add_261 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_532, %convert_element_type_812), kwargs = {})
#   return %sum_66,%add_261
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_33 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (3))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0013020833333333333
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/se/cse3iqu4eqkyp5wcogn7aflwgqhwov56thlyi7otjq6vcfxwyapk.py
# Topologically Sorted Source Nodes: [clone_81, getitem_19], Original ATen: [aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
# Source node to ATen node mapping:
#   clone_81 => convert_element_type_839
#   getitem_19 => select_4
# Graph fragment:
#   %mul_39 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=mul_39]
#   %mm_207 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_207]
#   %mm_209 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_209]
#   %mm_211 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_211]
#   %add_261 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_261]
#   %sum_69 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_69]
#   %rsqrt_9 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_9]
#   %add_270 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_270]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_257 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_257]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %add_35 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_35]
#   %add_24 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_24]
#   %mul_531 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_257, %cat), kwargs = {})
#   %sum_64 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_531,), kwargs = {})
#   %mul_533 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_257, %add_35), kwargs = {})
#   %sum_65 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_533,), kwargs = {})
#   %view_388 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_207, [16, 2048, 768]), kwargs = {})
#   %view_391 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_209, [16, 2048, 768]), kwargs = {})
#   %add_268 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_388, %view_391), kwargs = {})
#   %view_394 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_211, [16, 2048, 768]), kwargs = {})
#   %add_269 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_268, %view_394), kwargs = {})
#   %convert_element_type_839 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_269, torch.float32), kwargs = {})
#   %select_4 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 2), kwargs = {})
#   %mul_559 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_39, %convert_element_type_839), kwargs = {})
#   %sum_69 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_559, [2], True), kwargs = {})
#   %div_44 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_39, 768), kwargs = {})
#   %mul_560 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_44, %sum_69), kwargs = {})
#   %sub_50 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_839, %mul_560), kwargs = {})
#   %mul_561 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %rsqrt_9), kwargs = {})
#   %convert_element_type_841 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_561, torch.bfloat16), kwargs = {})
#   %add_270 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_261, %convert_element_type_841), kwargs = {})
#   %mul_563 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_270, %cat), kwargs = {})
#   %sum_70 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_563,), kwargs = {})
#   %mul_564 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_270, %select_4), kwargs = {})
#   %mul_565 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_270, %add_24), kwargs = {})
#   %sum_71 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_565,), kwargs = {})
#   %view_395 : Tensor "bf16[32768, 768][768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_564, [32768, 768]), kwargs = {})
#   return %sum_69,%add_270,%view_395,%buf425,%buf428,%buf468,%buf470
triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_34 = async_compile.triton('triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 11, 'num_reduction': 5, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1179648, 'r0_': 704643072}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (2))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp25 = tl.load(in_ptr6 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr7 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp43 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = 0.0013020833333333333
    tmp14 = tmp0 * tmp13
    tmp15 = tmp14 * tmp11
    tmp16 = tmp6 - tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp12 + tmp19
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 * tmp23
    tmp27 = tmp25 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.where(r0_mask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
    tmp36 = tl.where(r0_mask, tmp34, 0)
    tmp37 = tl.sum(tmp36, 1)[:, None].to(tl.float32)
    tmp38 = tmp20 * tmp26
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.where(r0_mask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None].to(tl.float32)
    tmp44 = tmp20 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
    tmp47 = tl.where(r0_mask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp20, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp24, r0_mask)
    tl.store(out_ptr2 + (x0), tmp31, None)
    tl.store(out_ptr3 + (x0), tmp37, None)
    tl.store(out_ptr4 + (x0), tmp42, None)
    tl.store(out_ptr5 + (x0), tmp48, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/35/c357qlmq56otutji37ophpfzz2bypj2e7sh2azc55z5xwzkhuksh.py
# Topologically Sorted Source Nodes: [getitem_19, clone_83, rms_norm_8], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   clone_83 => convert_element_type_854
#   getitem_19 => select_4
#   rms_norm_8 => convert_element_type_53, mul_36
# Graph fragment:
#   %add_22 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_22]
#   %rsqrt_8 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_8]
#   %mm_215 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_215]
#   %add_270 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_270]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %sum_72 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_72]
#   %select_4 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 2), kwargs = {})
#   %mul_564 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_270, %select_4), kwargs = {})
#   %view_398 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_215, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_854 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_398, torch.float32), kwargs = {})
#   %convert_element_type_53 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_22, torch.float32), kwargs = {})
#   %mul_36 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_53, %rsqrt_8), kwargs = {})
#   %mul_569 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %convert_element_type_854), kwargs = {})
#   %sum_72 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_569, [2], True), kwargs = {})
#   %div_45 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_36, 768), kwargs = {})
#   %mul_570 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_45, %sum_72), kwargs = {})
#   %sub_51 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_854, %mul_570), kwargs = {})
#   %mul_571 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %rsqrt_8), kwargs = {})
#   %convert_element_type_856 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_571, torch.bfloat16), kwargs = {})
#   %add_274 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_564, %convert_element_type_856), kwargs = {})
#   return %sum_72,%add_274
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_35 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (2))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0013020833333333333
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/yw/cywutvrbdw5utoqzkb3bzrb3ki4zclufzoos7jf3ej5pbdjg2fzz.py
# Topologically Sorted Source Nodes: [full_5, _generalized_scatter_69, clone_89, getitem_12, mul_14, getitem_13, mul_15, x_10, rms_norm_5], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   _generalized_scatter_69 => slice_scatter_default_49
#   clone_89 => convert_element_type_891
#   full_5 => full_default_10
#   getitem_12 => select_2
#   getitem_13 => select_3
#   mul_14 => mul_19
#   mul_15 => mul_20
#   rms_norm_5 => convert_element_type_32, mul_21
#   x_10 => add_13
# Graph fragment:
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %add_12 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_12]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %rsqrt_5 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_5]
#   %mul_21 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=mul_21]
#   %mm_219 : Tensor "bf16[32768, 12][12, 1]cuda:0" = PlaceHolder[target=mm_219]
#   %mm_221 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_221]
#   %mm_223 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_223]
#   %mm_225 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_225]
#   %add_274 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_274]
#   %sum_76 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_76]
#   %add_284 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_284]
#   %full_default_10 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([16, 2048, 768], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_403 : Tensor "bf16[16, 2048, 12][24576, 12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_219, [16, 2048, 12]), kwargs = {})
#   %slice_scatter_default_49 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %view_403, 2, 0, 12), kwargs = {})
#   %view_407 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_221, [16, 2048, 768]), kwargs = {})
#   %add_281 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_49, %view_407), kwargs = {})
#   %view_410 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_223, [16, 2048, 768]), kwargs = {})
#   %add_282 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_281, %view_410), kwargs = {})
#   %view_413 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_225, [16, 2048, 768]), kwargs = {})
#   %add_283 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_282, %view_413), kwargs = {})
#   %convert_element_type_891 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_283, torch.float32), kwargs = {})
#   %select_2 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 1), kwargs = {})
#   %mul_19 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %add_12), kwargs = {})
#   %select_3 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 1), kwargs = {})
#   %mul_20 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %cat), kwargs = {})
#   %add_13 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %mul_20), kwargs = {})
#   %convert_element_type_32 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_13, torch.float32), kwargs = {})
#   %mul_21 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_32, %rsqrt_5), kwargs = {})
#   %mul_596 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %convert_element_type_891), kwargs = {})
#   %sum_76 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_596, [2], True), kwargs = {})
#   %div_48 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_21, 768), kwargs = {})
#   %mul_597 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_48, %sum_76), kwargs = {})
#   %sub_55 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_891, %mul_597), kwargs = {})
#   %mul_598 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %rsqrt_5), kwargs = {})
#   %convert_element_type_893 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_598, torch.bfloat16), kwargs = {})
#   %add_284 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_274, %convert_element_type_893), kwargs = {})
#   %mul_601 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_284, %select_2), kwargs = {})
#   %view_414 : Tensor "bf16[32768, 768][768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_601, [32768, 768]), kwargs = {})
#   return %mul_21,%sum_76,%add_284,%view_414
triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_36 = async_compile.triton('triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 553648128}}
)
@triton.jit
def triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (1))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    x0 = xindex
    tmp5 = tl.load(in_ptr2 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp29 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr6 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr7 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp2 * tmp3
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp4 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp11 * tmp12
        tmp14 = r0_1
        tmp15 = tl.full([1, 1], 12, tl.int64)
        tmp16 = tmp14 < tmp15
        tmp17 = tl.load(in_ptr5 + (r0_1 + 12*x0), r0_mask & tmp16, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = 0.0
        tmp19 = tl.where(tmp16, tmp17, tmp18)
        tmp21 = tmp19 + tmp20
        tmp23 = tmp21 + tmp22
        tmp25 = tmp23 + tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp13 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(r0_mask, tmp30, _tmp29)
        tl.store(out_ptr0 + (r0_1 + 768*x0), tmp13, r0_mask)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tmp53 = tl.load(in_ptr0 + (1))
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp31 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp38 = tl.load(in_ptr6 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp40 = tl.load(in_ptr7 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp42 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp45 = tl.load(out_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp32 = r0_1
        tmp33 = tl.full([1, 1], 12, tl.int64)
        tmp34 = tmp32 < tmp33
        tmp35 = tl.load(in_ptr5 + (r0_1 + 12*x0), r0_mask & tmp34, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp36 = 0.0
        tmp37 = tl.where(tmp34, tmp35, tmp36)
        tmp39 = tmp37 + tmp38
        tmp41 = tmp39 + tmp40
        tmp43 = tmp41 + tmp42
        tmp44 = tmp43.to(tl.float32)
        tmp46 = 0.0013020833333333333
        tmp47 = tmp45 * tmp46
        tmp48 = tmp47 * tmp29
        tmp49 = tmp44 - tmp48
        tmp50 = tmp49 * tmp12
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp31 + tmp51
        tmp55 = tmp54.to(tl.float32)
        tmp56 = tmp52 * tmp55
        tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp52, r0_mask)
        tl.store(out_ptr2 + (r0_1 + 768*x0), tmp56, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/6w/c6wtzcgwkagclbqwvjiwwqcqusz3cm56ixxtwemv724ikzfelkeh.py
# Topologically Sorted Source Nodes: [getitem_12, clone_91, rms_norm_4], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
#   clone_91 => convert_element_type_908
#   getitem_12 => select_2
#   rms_norm_4 => convert_element_type_24, mul_18
# Graph fragment:
#   %add_10 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_10]
#   %rsqrt_4 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_4]
#   %mm_229 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_229]
#   %add_284 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_284]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %sum_79 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_79]
#   %select_2 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 1), kwargs = {})
#   %mul_601 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_284, %select_2), kwargs = {})
#   %view_417 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_229, [16, 2048, 768]), kwargs = {})
#   %convert_element_type_908 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_417, torch.float32), kwargs = {})
#   %convert_element_type_24 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.float32), kwargs = {})
#   %mul_18 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %rsqrt_4), kwargs = {})
#   %mul_606 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %convert_element_type_908), kwargs = {})
#   %sum_79 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_606, [2], True), kwargs = {})
#   %div_49 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_18, 768), kwargs = {})
#   %mul_607 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_49, %sum_79), kwargs = {})
#   %sub_56 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_908, %mul_607), kwargs = {})
#   %mul_608 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %rsqrt_4), kwargs = {})
#   %convert_element_type_910 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_608, torch.bfloat16), kwargs = {})
#   %add_288 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_601, %convert_element_type_910), kwargs = {})
#   return %sum_79,%add_288
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_37 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251658240}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (1))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0013020833333333333
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp5 - tmp18
    tmp20 = tmp19 * tmp2
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp15 + tmp21
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/rj/crjueqpe5uhbrt4mc55my44hjpkuhtccrur5xp6laibq7naqnxbt.py
# Topologically Sorted Source Nodes: [getitem_20, getitem_13, clone_97, getitem_6, mul_2, getitem_7, mul_3, x_4, rms_norm_1], Original ATen: [aten.select, aten.mul, aten.add, aten.sum, aten.view, aten._fused_rms_norm_backward, aten._to_copy]
# Source node to ATen node mapping:
#   clone_97 => convert_element_type_937
#   getitem_13 => select_3
#   getitem_20 => select_5
#   getitem_6 => select
#   getitem_7 => select_1
#   mul_2 => mul_3
#   mul_3 => mul_4
#   rms_norm_1 => convert_element_type_6, mul_5
#   x_4 => add_2
# Graph fragment:
#   %add_284 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_284]
#   %cat : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=cat]
#   %add_12 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_12]
#   %primals_7 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_7]
#   %primals_8 : Tensor "f32[12][1]cuda:0" = PlaceHolder[target=primals_8]
#   %rsqrt_1 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %mm_233 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_233]
#   %mm_235 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_235]
#   %mm_237 : Tensor "bf16[32768, 768][768, 1]cuda:0" = PlaceHolder[target=mm_237]
#   %add_288 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_288]
#   %sum_82 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_82]
#   %add_258 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_258]
#   %add_270 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_270]
#   %add_297 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_297]
#   %select_5 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 2), kwargs = {})
#   %mul_562 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_270, %select_5), kwargs = {})
#   %add_271 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_258, %mul_562), kwargs = {})
#   %select_3 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 1), kwargs = {})
#   %mul_599 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_284, %select_3), kwargs = {})
#   %mul_600 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_284, %cat), kwargs = {})
#   %sum_77 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_600,), kwargs = {})
#   %add_285 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_271, %mul_599), kwargs = {})
#   %mul_602 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_284, %add_12), kwargs = {})
#   %sum_78 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_602,), kwargs = {})
#   %view_423 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_233, [16, 2048, 768]), kwargs = {})
#   %view_426 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_235, [16, 2048, 768]), kwargs = {})
#   %add_295 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_423, %view_426), kwargs = {})
#   %view_429 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_237, [16, 2048, 768]), kwargs = {})
#   %add_296 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_295, %view_429), kwargs = {})
#   %convert_element_type_937 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_296, torch.float32), kwargs = {})
#   %select : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_7, 0, 0), kwargs = {})
#   %mul_3 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %cat), kwargs = {})
#   %select_1 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%primals_8, 0, 0), kwargs = {})
#   %mul_4 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %cat), kwargs = {})
#   %add_2 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_4), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float32), kwargs = {})
#   %mul_5 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %rsqrt_1), kwargs = {})
#   %mul_628 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %convert_element_type_937), kwargs = {})
#   %sum_82 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_628, [2], True), kwargs = {})
#   %div_52 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_5, 768), kwargs = {})
#   %mul_629 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_52, %sum_82), kwargs = {})
#   %sub_59 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_937, %mul_629), kwargs = {})
#   %mul_630 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %rsqrt_1), kwargs = {})
#   %convert_element_type_939 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_630, torch.bfloat16), kwargs = {})
#   %add_297 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_288, %convert_element_type_939), kwargs = {})
#   %mul_631 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_297, %select_1), kwargs = {})
#   %mul_632 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_297, %cat), kwargs = {})
#   %sum_83 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_632,), kwargs = {})
#   %add_298 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_285, %mul_631), kwargs = {})
#   %mul_633 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_297, %select), kwargs = {})
#   %add_300 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_298, %mul_633), kwargs = {})
#   return %buf518,%buf520,%sum_82,%add_297,%add_300,%buf559
triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_38 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_38', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 14, 'num_reduction': 4, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 917504, 'r0_': 553648128}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_38(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr3 + (0))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp17 = tl.load(in_ptr4 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp23 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr6 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr7 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr8 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp44 = tl.load(in_out_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp45 = tl.load(in_ptr9 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp46 = tl.load(in_ptr4 + (2))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, R0_BLOCK])
    tmp51 = tl.load(in_ptr4 + (1))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK, R0_BLOCK])
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp1
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp1
    tmp21 = tmp16 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp22 * tmp23
    tmp27 = tmp25 + tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp24 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(r0_mask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None].to(tl.float32)
    tmp37 = 0.0013020833333333333
    tmp38 = tmp24 * tmp37
    tmp39 = tmp38 * tmp35
    tmp40 = tmp30 - tmp39
    tmp41 = tmp40 * tmp23
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp36 + tmp42
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp45 * tmp48
    tmp50 = tmp44 + tmp49
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp0 * tmp53
    tmp55 = tmp50 + tmp54
    tmp56 = tmp43 * tmp19
    tmp57 = tmp55 + tmp56
    tmp58 = tmp43 * tmp15
    tmp59 = tmp57 + tmp58
    tmp60 = tmp43 * tmp1
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK, R0_BLOCK])
    tmp63 = tl.where(r0_mask, tmp61, 0)
    tmp64 = tl.sum(tmp63, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr1 + (r0_1 + 768*x0), tmp59, r0_mask)
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp12, None)
    tl.store(out_ptr3 + (x0), tmp64, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/ik/cikwubd2elot3zco5hdmpz3s7kwygwks7rzzbwoo5iuflky6jrdf.py
# Topologically Sorted Source Nodes: [full_7, _generalized_scatter_5, _generalized_scatter_11, _generalized_scatter_18, _generalized_scatter_24, _generalized_scatter_31, _generalized_scatter_37, _generalized_scatter_44, _generalized_scatter_50, _generalized_scatter_57, _generalized_scatter_63, _generalized_scatter_70, _generalized_scatter_76], Original ATen: [aten._to_copy, aten.select_backward, aten.add]
# Source node to ATen node mapping:
#   _generalized_scatter_11 => select_scatter_default_2
#   _generalized_scatter_18 => select_scatter_default_4
#   _generalized_scatter_24 => select_scatter_default_6
#   _generalized_scatter_31 => select_scatter_default_8
#   _generalized_scatter_37 => select_scatter_default_10
#   _generalized_scatter_44 => select_scatter_default_12
#   _generalized_scatter_5 => select_scatter_default
#   _generalized_scatter_50 => select_scatter_default_14
#   _generalized_scatter_57 => select_scatter_default_16
#   _generalized_scatter_63 => select_scatter_default_18
#   _generalized_scatter_70 => select_scatter_default_20
#   _generalized_scatter_76 => select_scatter_default_22
#   full_7 => full_default_13
# Graph fragment:
#   %sum_12 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=sum_12]
#   %sum_18 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=sum_18]
#   %sum_25 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=sum_25]
#   %sum_31 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=sum_31]
#   %sum_38 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=sum_38]
#   %sum_44 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=sum_44]
#   %sum_51 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=sum_51]
#   %sum_57 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=sum_57]
#   %sum_64 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=sum_64]
#   %sum_70 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=sum_70]
#   %sum_77 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=sum_77]
#   %sum_83 : Tensor "bf16[][]cuda:0" = PlaceHolder[target=sum_83]
#   %convert_element_type_406 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_12, torch.float32), kwargs = {})
#   %full_default_13 : Tensor "f32[12][1]cuda:0"[num_users=23] = call_function[target=torch.ops.aten.full.default](args = ([12], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %convert_element_type_406, 0, 11), kwargs = {})
#   %convert_element_type_450 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_18, torch.float32), kwargs = {})
#   %select_scatter_default_2 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %convert_element_type_450, 0, 10), kwargs = {})
#   %add_163 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_2), kwargs = {})
#   %convert_element_type_504 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_25, torch.float32), kwargs = {})
#   %select_scatter_default_4 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %convert_element_type_504, 0, 9), kwargs = {})
#   %add_177 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_163, %select_scatter_default_4), kwargs = {})
#   %convert_element_type_548 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_31, torch.float32), kwargs = {})
#   %select_scatter_default_6 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %convert_element_type_548, 0, 8), kwargs = {})
#   %add_190 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_177, %select_scatter_default_6), kwargs = {})
#   %convert_element_type_602 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_38, torch.float32), kwargs = {})
#   %select_scatter_default_8 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %convert_element_type_602, 0, 7), kwargs = {})
#   %add_204 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_190, %select_scatter_default_8), kwargs = {})
#   %convert_element_type_646 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_44, torch.float32), kwargs = {})
#   %select_scatter_default_10 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %convert_element_type_646, 0, 6), kwargs = {})
#   %add_218 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_204, %select_scatter_default_10), kwargs = {})
#   %convert_element_type_700 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_51, torch.float32), kwargs = {})
#   %select_scatter_default_12 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %convert_element_type_700, 0, 5), kwargs = {})
#   %add_232 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_218, %select_scatter_default_12), kwargs = {})
#   %convert_element_type_744 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_57, torch.float32), kwargs = {})
#   %select_scatter_default_14 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %convert_element_type_744, 0, 4), kwargs = {})
#   %add_245 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_232, %select_scatter_default_14), kwargs = {})
#   %convert_element_type_798 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_64, torch.float32), kwargs = {})
#   %select_scatter_default_16 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %convert_element_type_798, 0, 3), kwargs = {})
#   %add_259 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_245, %select_scatter_default_16), kwargs = {})
#   %convert_element_type_842 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_70, torch.float32), kwargs = {})
#   %select_scatter_default_18 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %convert_element_type_842, 0, 2), kwargs = {})
#   %add_272 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_259, %select_scatter_default_18), kwargs = {})
#   %convert_element_type_896 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_77, torch.float32), kwargs = {})
#   %select_scatter_default_20 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %convert_element_type_896, 0, 1), kwargs = {})
#   %add_286 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_272, %select_scatter_default_20), kwargs = {})
#   %convert_element_type_940 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_83, torch.float32), kwargs = {})
#   %select_scatter_default_22 : Tensor "f32[12][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_13, %convert_element_type_940, 0, 0), kwargs = {})
#   %add_299 : Tensor "f32[12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_286, %select_scatter_default_22), kwargs = {})
#   return %add_299
triton_poi_fused__to_copy_add_select_backward_39 = async_compile.triton('triton_poi_fused__to_copy_add_select_backward_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*bf16', 'in_ptr10': '*bf16', 'in_ptr11': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_select_backward_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 96}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_select_backward_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp3 = tl.load(in_ptr0 + (0)).to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp10 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp17 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp24 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp31 = tl.load(in_ptr4 + (0)).to(tl.float32)
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp38 = tl.load(in_ptr5 + (0)).to(tl.float32)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp45 = tl.load(in_ptr6 + (0)).to(tl.float32)
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp52 = tl.load(in_ptr7 + (0)).to(tl.float32)
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp59 = tl.load(in_ptr8 + (0)).to(tl.float32)
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK])
    tmp66 = tl.load(in_ptr9 + (0)).to(tl.float32)
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp73 = tl.load(in_ptr10 + (0)).to(tl.float32)
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK])
    tmp80 = tl.load(in_ptr11 + (0)).to(tl.float32)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 11, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tl.full([1], 10, tl.int32)
    tmp9 = tmp0 == tmp8
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.where(tmp9, tmp12, tmp6)
    tmp14 = tmp7 + tmp13
    tmp15 = tl.full([1], 9, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.where(tmp16, tmp19, tmp6)
    tmp21 = tmp14 + tmp20
    tmp22 = tl.full([1], 8, tl.int32)
    tmp23 = tmp0 == tmp22
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.where(tmp23, tmp26, tmp6)
    tmp28 = tmp21 + tmp27
    tmp29 = tl.full([1], 7, tl.int32)
    tmp30 = tmp0 == tmp29
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.where(tmp30, tmp33, tmp6)
    tmp35 = tmp28 + tmp34
    tmp36 = tl.full([1], 6, tl.int32)
    tmp37 = tmp0 == tmp36
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tl.where(tmp37, tmp40, tmp6)
    tmp42 = tmp35 + tmp41
    tmp43 = tl.full([1], 5, tl.int32)
    tmp44 = tmp0 == tmp43
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tl.where(tmp44, tmp47, tmp6)
    tmp49 = tmp42 + tmp48
    tmp50 = tl.full([1], 4, tl.int32)
    tmp51 = tmp0 == tmp50
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.where(tmp51, tmp54, tmp6)
    tmp56 = tmp49 + tmp55
    tmp57 = tl.full([1], 3, tl.int32)
    tmp58 = tmp0 == tmp57
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tl.where(tmp58, tmp61, tmp6)
    tmp63 = tmp56 + tmp62
    tmp64 = tl.full([1], 2, tl.int32)
    tmp65 = tmp0 == tmp64
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tl.where(tmp65, tmp68, tmp6)
    tmp70 = tmp63 + tmp69
    tmp71 = tl.full([1], 1, tl.int32)
    tmp72 = tmp0 == tmp71
    tmp75 = tmp74.to(tl.float32)
    tmp76 = tl.where(tmp72, tmp75, tmp6)
    tmp77 = tmp70 + tmp76
    tmp78 = tl.full([1], 0, tl.int32)
    tmp79 = tmp0 == tmp78
    tmp82 = tmp81.to(tl.float32)
    tmp83 = tl.where(tmp79, tmp82, tmp6)
    tmp84 = tmp77 + tmp83
    tl.store(out_ptr0 + (x0), tmp84, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/4e/c4ebzujjxmmfnkcsgovanktn5iyqdzd5bobimosde5s6la5t2nq7.py
# Topologically Sorted Source Nodes: [x_2, getitem_5], Original ATen: [aten.slice, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   getitem_5 => slice_7
#   x_2 => convert_element_type, convert_element_type_1, mul
# Graph fragment:
#   %add_300 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_300]
#   %embedding : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %slice_111 : Tensor "bf16[16, 2047, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_300, 1, 1, 2048), kwargs = {})
#   %convert_element_type : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %slice_7 : Tensor "bf16[16, 2047, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_1, 1, 0, -1), kwargs = {})
#   %mul_636 : Tensor "bf16[16, 2047, 768][1572096, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_111, %slice_7), kwargs = {})
#   %sum_85 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_636, [2], True), kwargs = {})
#   return %sum_85
triton_red_fused__to_copy_mul_slice_sum_40 = async_compile.triton('triton_red_fused__to_copy_mul_slice_sum_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_slice_sum_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262016, 'r0_': 100614144}}
)
@triton.jit
def triton_red_fused__to_copy_mul_slice_sum_40(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32752
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 2047)
    x1 = xindex // 2047
    tmp3 = tl.load(in_ptr2 + (x0 + 2048*x1), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (768 + r0_2 + 768*x0 + 1572864*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 768*x0 + 1572864*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 2048*x1), tmp8, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/ud/cudqjmdmfxz55rcp7hdl73msnmpdq5ki2ukbhtr7otme6x25emzw.py
# Topologically Sorted Source Nodes: [to_1, linear, sigmoid], Original ATen: [aten._to_copy, aten._unsafe_view, aten.sigmoid, aten.mul, aten.sum, aten.sigmoid_backward]
# Source node to ATen node mapping:
#   linear => view_1
#   sigmoid => sigmoid
#   to_1 => convert_element_type_2
# Graph fragment:
#   %sum_85 : Tensor "bf16[16, 2047, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_85]
#   %mm : Tensor "bf16[32752, 1][1, 1]cuda:0" = PlaceHolder[target=mm]
#   %primals_5 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_5]
#   %convert_element_type_2 : Tensor "bf16[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.bfloat16), kwargs = {})
#   %view_1 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [16, 2047, 1]), kwargs = {})
#   %sigmoid : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_1,), kwargs = {})
#   %mul_637 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_85, %convert_element_type_2), kwargs = {})
#   %mul_638 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_85, %sigmoid), kwargs = {})
#   %sum_86 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_638, [0, 1], True), kwargs = {})
#   %convert_element_type_942 : Tensor "f32[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_637, torch.float32), kwargs = {})
#   %convert_element_type_943 : Tensor "f32[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid, torch.float32), kwargs = {})
#   %sub_60 : Tensor "f32[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_943), kwargs = {})
#   %mul_639 : Tensor "f32[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_943, %sub_60), kwargs = {})
#   %mul_640 : Tensor "f32[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_942, %mul_639), kwargs = {})
#   %convert_element_type_944 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_640, torch.bfloat16), kwargs = {})
#   return %buf566,%convert_element_type_944
triton_red_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_sum_41 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_sum_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_sum_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16, 'r0_': 131008}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_sum_41(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 8188
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp8 = tl.load(in_ptr2 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        r0_2 = (r0_index % 2047)
        r0_3 = r0_index // 2047
        tmp0 = tl.load(in_ptr0 + (2048*(r0_1 // 2047) + 8192*x0 + ((r0_1 % 2047))), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 8188*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr0 + (r0_2 + 2048*r0_3 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp7 * tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp2.to(tl.float32)
        tmp14 = 1.0
        tmp15 = tmp14 - tmp13
        tmp16 = tmp13 * tmp15
        tmp17 = tmp12 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tl.store(out_ptr1 + (r0_2 + 2048*r0_3 + 8192*x0), tmp18, r0_mask & xmask)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/3w/c3wbvahhrc2ztldcrbyfnxoymffoxan5m7k6yzwslgonguowodef.py
# Topologically Sorted Source Nodes: [linear, sigmoid], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.mul, aten.sum, aten.view, aten._to_copy]
# Source node to ATen node mapping:
#   linear => view_1
#   sigmoid => sigmoid
# Graph fragment:
#   %buf566 : Tensor "f32[1, 1, 1, 4][4, 4, 4, 1]cuda:0" = PlaceHolder[target=buf566]
#   %sum_86 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=sum_86]
#   %view_1 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [16, 2047, 1]), kwargs = {})
#   %sigmoid : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_1,), kwargs = {})
#   %mul_638 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_85, %sigmoid), kwargs = {})
#   %sum_86 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_638, [0, 1], True), kwargs = {})
#   %view_430 : Tensor "bf16[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_86, [1]), kwargs = {})
#   %convert_element_type_950 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_430, torch.float32), kwargs = {})
#   return %sum_86,%convert_element_type_950
triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sum_view_42 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sum_view_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sum_view_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 8}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sum_view_42(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None].to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/4n/c4nxkjes3q5sst6p3fgh4vgxb55hx4jvr666m2a4jktp7bsl5z2g.py
# Topologically Sorted Source Nodes: [to_1, linear, sigmoid], Original ATen: [aten._to_copy, aten._unsafe_view, aten.sigmoid, aten.mul, aten.sigmoid_backward, aten.view]
# Source node to ATen node mapping:
#   linear => view_1
#   sigmoid => sigmoid
#   to_1 => convert_element_type_2
# Graph fragment:
#   %convert_element_type_944 : Tensor "bf16[16, 2047, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=convert_element_type_944]
#   %convert_element_type_2 : Tensor "bf16[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.bfloat16), kwargs = {})
#   %view_1 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [16, 2047, 1]), kwargs = {})
#   %sigmoid : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_1,), kwargs = {})
#   %mul_637 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_85, %convert_element_type_2), kwargs = {})
#   %convert_element_type_942 : Tensor "f32[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_637, torch.float32), kwargs = {})
#   %convert_element_type_943 : Tensor "f32[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sigmoid, torch.float32), kwargs = {})
#   %sub_60 : Tensor "f32[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type_943), kwargs = {})
#   %mul_639 : Tensor "f32[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_943, %sub_60), kwargs = {})
#   %mul_640 : Tensor "f32[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_942, %mul_639), kwargs = {})
#   %convert_element_type_944 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_640, torch.bfloat16), kwargs = {})
#   %view_431 : Tensor "bf16[32752, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_944, [32752, 1]), kwargs = {})
#   return %view_431
triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_view_43 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_view_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_view_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 196512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_view_43(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2048*(x0 // 2047) + ((x0 % 2047))), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/gk/cgkfq6dtfqckxgwpulg3djiuvx63bpanhv552afxzzdumbkjzoa5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_238 : Tensor "bf16[1, 24][24, 1]cuda:0" = PlaceHolder[target=mm_238]
#   %convert_element_type_949 : Tensor "f32[1, 24][24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_238, torch.float32), kwargs = {})
#   return %convert_element_type_949
triton_poi_fused__to_copy_44 = async_compile.triton('triton_poi_fused__to_copy_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_44(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpzdkiugnf/no/cnonrk4qkim57vg55j6qwnlndammfz5kksnzlxixz4dwfyhrm3n5.py
# Topologically Sorted Source Nodes: [loss, full_5, full_6, to_1, linear, sigmoid, gate, x_2, _generalized_scatter_77, _generalized_scatter_78, _generalized_scatter_79, full_88, _generalized_scatter_80, _generalized_scatter_81, clone_99, convert_element_type_default], Original ATen: [aten.nll_loss_forward, aten.slice_backward, aten.embedding_dense_backward, aten.slice, aten._to_copy, aten._unsafe_view, aten.sigmoid, aten.mul, aten.add, aten.view, aten._fused_rms_norm_backward]
# Source node to ATen node mapping:
#   _generalized_scatter_77 => slice_scatter_default_54
#   _generalized_scatter_78 => slice_scatter_default_55
#   _generalized_scatter_79 => slice_scatter_default_56
#   _generalized_scatter_80 => slice_scatter_default_57
#   _generalized_scatter_81 => slice_scatter_default_58
#   clone_99 => convert_element_type_951
#   convert_element_type_default => mul_644
#   full_5 => full_default_10
#   full_6 => full_default_12
#   full_88 => full_default_110
#   gate => mul_1
#   linear => view_1
#   loss => full_default_1
#   sigmoid => sigmoid
#   to_1 => convert_element_type_2
#   x_2 => convert_element_type, mul
# Graph fragment:
#   %add_300 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_300]
#   %primals_5 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_5]
#   %mm : Tensor "bf16[32752, 1][1, 1]cuda:0" = PlaceHolder[target=mm]
#   %mm_239 : Tensor "bf16[32752, 24][24, 1]cuda:0" = PlaceHolder[target=mm_239]
#   %embedding : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=embedding]
#   %rsqrt : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_304 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0" = PlaceHolder[target=add_304]
#   %primals_1 : Tensor "i64[16, 2048][2048, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %sum_87 : Tensor "f32[16, 2048, 1][2048, 1, 32768]cuda:0" = PlaceHolder[target=sum_87]
#   %index_put_6 : Tensor "f32[32768, 768][768, 1]cuda:0" = PlaceHolder[target=index_put_6]
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_10 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([16, 2048, 768], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq : Tensor "b8[16, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, -1), kwargs = {})
#   %unsqueeze_8 : Tensor "b8[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %full_default_12 : Tensor "f32[32768, 768][768, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([32768, 768], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_110 : Tensor "bf16[16, 1, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_300, 1, 0, 1), kwargs = {})
#   %slice_111 : Tensor "bf16[16, 2047, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.slice.Tensor](args = (%add_300, 1, 1, 2048), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.bfloat16), kwargs = {})
#   %view_1 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [16, 2047, 1]), kwargs = {})
#   %sigmoid : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_1,), kwargs = {})
#   %mul_1 : Tensor "bf16[16, 2047, 1][2047, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %sigmoid), kwargs = {})
#   %mul_635 : Tensor "bf16[16, 2047, 768][1572096, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_111, %mul_1), kwargs = {})
#   %convert_element_type : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %slice_scatter_default_54 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %mul_635, 1, 0, -1), kwargs = {})
#   %slice_scatter_default_55 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %slice_111, 1, 1, 9223372036854775807), kwargs = {})
#   %add_302 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_54, %slice_scatter_default_55), kwargs = {})
#   %slice_scatter_default_56 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %slice_110, 1, 0, 1), kwargs = {})
#   %add_303 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_302, %slice_scatter_default_56), kwargs = {})
#   %view_432 : Tensor "bf16[16, 2047, 24][49128, 24, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_239, [16, 2047, 24]), kwargs = {})
#   %full_default_110 : Tensor "bf16[16, 2047, 768][1572096, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([16, 2047, 768], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_57 : Tensor "bf16[16, 2047, 768][1572096, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_110, %view_432, 2, 0, 24), kwargs = {})
#   %slice_scatter_default_58 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %slice_scatter_default_57, 1, 1, 9223372036854775807), kwargs = {})
#   %add_304 : Tensor "bf16[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_303, %slice_scatter_default_58), kwargs = {})
#   %convert_element_type_951 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_304, torch.float32), kwargs = {})
#   %mul_642 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %convert_element_type_951), kwargs = {})
#   %sum_87 : Tensor "f32[16, 2048, 1][2048, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_642, [2], True), kwargs = {})
#   %div_53 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 768), kwargs = {})
#   %mul_643 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_53, %sum_87), kwargs = {})
#   %sub_61 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_951, %mul_643), kwargs = {})
#   %mul_644 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %rsqrt), kwargs = {})
#   %where_22 : Tensor "f32[16, 2048, 768][1572864, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_8, %full_default_1, %mul_644), kwargs = {})
#   %index_put_6 : Tensor "f32[32768, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_12, [%primals_1], %where_22, True), kwargs = {})
#   return %add_304,%sum_87,%buf577
triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_sigmoid_slice_slice_backward_view_45 = async_compile.triton('triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_sigmoid_slice_slice_backward_view_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_sigmoid_slice_slice_backward_view_45', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 9, 'num_reduction': 1, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_sigmoid_slice_slice_backward_view_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x3 = xindex
    x1 = xindex // 2048
    tmp36 = tl.load(in_ptr4 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp38 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 2047, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (768 + r0_2 + 768*x3), r0_mask & tmp2, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(tmp2, tmp5, 0.0)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(x0 + 2047*x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp7 * tmp9
    tmp11 = tmp3 * tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = 0.0
    tmp15 = tl.where(tmp2, tmp13, tmp14)
    tmp16 = tl.full([1, 1], 1, tl.int64)
    tmp17 = tmp0 >= tmp16
    tmp18 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask & tmp17, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, tmp14)
    tmp20 = tmp15 + tmp19
    tmp21 = tmp0 < tmp16
    tmp22 = tl.load(in_ptr0 + (r0_2 + 1572864*x1), r0_mask & tmp21, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tl.where(tmp21, tmp22, tmp14)
    tmp24 = tmp20 + tmp23
    tmp25 = tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])
    tmp26 = tl.full([1, 1], 24, tl.int64)
    tmp27 = tmp25 < tmp26
    tmp28 = tmp27 & tmp17
    tmp29 = tl.load(in_ptr3 + ((-24) + r0_2 + 24*x0 + 49128*x1), r0_mask & tmp28, other=0.0).to(tl.float32)
    tmp30 = 0.0
    tmp31 = tl.where(tmp27, tmp29, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp17, tmp31, tmp32)
    tmp34 = tl.where(tmp17, tmp33, tmp14)
    tmp35 = tmp24 + tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp39 = tmp37 * tmp38
    tmp40 = tmp35.to(tl.float32)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, R0_BLOCK])
    tmp44 = tl.where(r0_mask, tmp42, 0)
    tmp45 = tl.sum(tmp44, 1)[:, None].to(tl.float32)
    tmp47 = tl.full([XBLOCK, R0_BLOCK], 32768, tl.int32)
    tmp48 = tmp46 + tmp47
    tmp49 = tmp46 < 0
    tmp50 = tl.where(tmp49, tmp48, tmp46)
    tl.device_assert((0 <= tmp50) & (tmp50 < 32768), "index out of bounds: 0 <= tmp50 < 32768")
    tmp52 = tl.full([1, 1], -1, tl.int64)
    tmp53 = tmp46 == tmp52
    tmp54 = 0.0013020833333333333
    tmp55 = tmp39 * tmp54
    tmp56 = tmp55 * tmp45
    tmp57 = tmp40 - tmp56
    tmp58 = tmp57 * tmp38
    tmp59 = tl.where(tmp53, tmp14, tmp58)
    tl.atomic_add(out_ptr2 + (tl.broadcast_to(r0_2 + 768*tmp50, [XBLOCK, R0_BLOCK])), tmp59, r0_mask, sem='relaxed')
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
        primals_1, primals_2, primals_3, primals_5, primals_7, primals_8, primals_93, primals_95, embedding, rsqrt, view, mm, cat, rsqrt_1, view_2, view_10, cat_1, cat_2, rsqrt_2, rsqrt_3, mul_16, mul_17, getitem, getitem_1, add_10, rsqrt_4, view_14, mm_5, view_16, add_12, embedding_1, rsqrt_5, view_18, view_28, mm_10, add_15, cat_3, cat_4, rsqrt_6, rsqrt_7, mul_34, mul_35, getitem_4, getitem_5, add_22, rsqrt_8, view_33, mm_12, view_35, add_24, rsqrt_9, view_37, view_45, cat_5, cat_6, rsqrt_10, rsqrt_11, mul_50, mul_51, getitem_8, getitem_9, add_33, rsqrt_12, view_49, mm_18, view_51, add_35, embedding_2, rsqrt_13, view_53, view_63, mm_23, add_38, cat_7, cat_8, rsqrt_14, rsqrt_15, mul_68, mul_69, getitem_12, getitem_13, add_45, rsqrt_16, view_68, mm_25, view_70, add_47, rsqrt_17, view_72, view_80, cat_9, cat_10, rsqrt_18, rsqrt_19, mul_84, mul_85, getitem_16, getitem_17, add_56, rsqrt_20, view_84, mm_31, view_86, add_58, embedding_3, rsqrt_21, view_88, view_98, mm_36, add_61, cat_11, cat_12, rsqrt_22, rsqrt_23, mul_102, mul_103, getitem_20, getitem_21, add_68, rsqrt_24, view_103, mm_38, view_105, add_70, rsqrt_25, view_107, view_115, cat_13, cat_14, rsqrt_26, rsqrt_27, mul_118, mul_119, getitem_24, getitem_25, add_79, rsqrt_28, view_119, mm_44, view_121, add_81, embedding_4, rsqrt_29, view_123, view_133, mm_49, add_84, cat_15, cat_16, rsqrt_30, rsqrt_31, mul_136, mul_137, getitem_28, getitem_29, add_91, rsqrt_32, view_138, mm_51, view_140, add_93, rsqrt_33, view_142, view_150, cat_17, cat_18, rsqrt_34, rsqrt_35, mul_152, mul_153, getitem_32, getitem_33, add_102, rsqrt_36, view_154, mm_57, view_156, add_104, embedding_5, rsqrt_37, view_158, view_168, mm_62, add_107, cat_19, cat_20, rsqrt_38, rsqrt_39, mul_170, mul_171, getitem_36, getitem_37, add_114, rsqrt_40, view_173, mm_64, view_175, add_116, rsqrt_41, view_177, view_185, cat_21, cat_22, rsqrt_42, rsqrt_43, mul_186, mul_187, getitem_40, getitem_41, add_125, rsqrt_44, view_189, mm_70, view_191, add_127, embedding_6, rsqrt_45, view_193, view_203, mm_75, add_130, cat_23, cat_24, rsqrt_46, rsqrt_47, mul_204, mul_205, getitem_44, getitem_45, add_137, rsqrt_48, view_208, mm_77, view_210, sub, rsqrt_49, view_212, mm_79, amax, log, convert_element_type_343, permute_82, permute_86, permute_90, permute_94, permute_98, permute_102, permute_106, permute_110, permute_114, permute_118, permute_122, permute_126, permute_130, permute_134, permute_138, permute_142, permute_146, permute_150, permute_154, permute_158, permute_162, permute_166, permute_170, permute_174, permute_178, permute_182, permute_186, permute_190, permute_194, permute_198, permute_202, permute_206, permute_210, permute_214, permute_218, permute_222, permute_226, permute_230, permute_234, permute_238, permute_242, permute_246, permute_250, permute_254, permute_258, permute_262, permute_266, permute_270, permute_274, permute_278, permute_282, permute_286, permute_290, permute_294, permute_298, permute_302, permute_306, permute_310, permute_314, permute_318, permute_322, permute_326, permute_330, permute_334, permute_338, permute_342, permute_346, permute_350, permute_354, permute_358, permute_362, permute_366, permute_370, permute_374, permute_378, permute_382, permute_386, permute_390, permute_394, permute_398, tangents_1 = args
        args.clear()
        assert_size_stride(primals_1, (16, 2048), (2048, 1))
        assert_size_stride(primals_2, (1, 20480, 1, 64), (1310720, 64, 64, 1))
        assert_size_stride(primals_3, (1, 20480, 1, 64), (1310720, 64, 64, 1))
        assert_size_stride(primals_5, (1, ), (1, ))
        assert_size_stride(primals_7, (12, ), (1, ))
        assert_size_stride(primals_8, (12, ), (1, ))
        assert_size_stride(primals_93, (1, ), (1, ))
        assert_size_stride(primals_95, (16, 2048), (2048, 1))
        assert_size_stride(embedding, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view, (32752, 24), (24, 1))
        assert_size_stride(mm, (32752, 1), (1, 1))
        assert_size_stride(cat, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_1, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_2, (32768, 768), (768, 1))
        assert_size_stride(view_10, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_1, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_2, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(rsqrt_2, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(rsqrt_3, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(mul_16, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(mul_17, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_1, (16, 6, 2048), (12288, 2048, 1))
        assert_size_stride(add_10, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_4, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_14, (32768, 768), (768, 1))
        assert_size_stride(mm_5, (32768, 3072), (3072, 1))
        assert_size_stride(view_16, (32768, 3072), (3072, 1))
        assert_size_stride(add_12, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(embedding_1, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_5, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_18, (32768, 768), (768, 1))
        assert_size_stride(view_28, (32768, 12), (768, 1))
        assert_size_stride(mm_10, (32768, 6), (6, 1))
        assert_size_stride(add_15, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_3, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_4, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(rsqrt_6, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(rsqrt_7, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(mul_34, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(mul_35, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_4, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_5, (16, 6, 2048), (12288, 2048, 1))
        assert_size_stride(add_22, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_8, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_33, (32768, 768), (768, 1))
        assert_size_stride(mm_12, (32768, 3072), (3072, 1))
        assert_size_stride(view_35, (32768, 3072), (3072, 1))
        assert_size_stride(add_24, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_9, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_37, (32768, 768), (768, 1))
        assert_size_stride(view_45, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_5, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_6, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(rsqrt_10, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(rsqrt_11, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(mul_50, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(mul_51, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_8, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_9, (16, 6, 2048), (12288, 2048, 1))
        assert_size_stride(add_33, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_12, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_49, (32768, 768), (768, 1))
        assert_size_stride(mm_18, (32768, 3072), (3072, 1))
        assert_size_stride(view_51, (32768, 3072), (3072, 1))
        assert_size_stride(add_35, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(embedding_2, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_13, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_53, (32768, 768), (768, 1))
        assert_size_stride(view_63, (32768, 12), (768, 1))
        assert_size_stride(mm_23, (32768, 6), (6, 1))
        assert_size_stride(add_38, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_7, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_8, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(rsqrt_14, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(rsqrt_15, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(mul_68, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(mul_69, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_12, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_13, (16, 6, 2048), (12288, 2048, 1))
        assert_size_stride(add_45, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_16, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_68, (32768, 768), (768, 1))
        assert_size_stride(mm_25, (32768, 3072), (3072, 1))
        assert_size_stride(view_70, (32768, 3072), (3072, 1))
        assert_size_stride(add_47, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_17, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_72, (32768, 768), (768, 1))
        assert_size_stride(view_80, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_9, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_10, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(rsqrt_18, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(rsqrt_19, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(mul_84, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(mul_85, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_16, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_17, (16, 6, 2048), (12288, 2048, 1))
        assert_size_stride(add_56, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_20, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_84, (32768, 768), (768, 1))
        assert_size_stride(mm_31, (32768, 3072), (3072, 1))
        assert_size_stride(view_86, (32768, 3072), (3072, 1))
        assert_size_stride(add_58, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(embedding_3, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_21, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_88, (32768, 768), (768, 1))
        assert_size_stride(view_98, (32768, 12), (768, 1))
        assert_size_stride(mm_36, (32768, 6), (6, 1))
        assert_size_stride(add_61, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_11, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_12, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(rsqrt_22, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(rsqrt_23, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(mul_102, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(mul_103, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_20, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_21, (16, 6, 2048), (12288, 2048, 1))
        assert_size_stride(add_68, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_24, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_103, (32768, 768), (768, 1))
        assert_size_stride(mm_38, (32768, 3072), (3072, 1))
        assert_size_stride(view_105, (32768, 3072), (3072, 1))
        assert_size_stride(add_70, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_25, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_107, (32768, 768), (768, 1))
        assert_size_stride(view_115, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_13, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_14, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(rsqrt_26, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(rsqrt_27, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(mul_118, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(mul_119, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_24, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_25, (16, 6, 2048), (12288, 2048, 1))
        assert_size_stride(add_79, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_28, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_119, (32768, 768), (768, 1))
        assert_size_stride(mm_44, (32768, 3072), (3072, 1))
        assert_size_stride(view_121, (32768, 3072), (3072, 1))
        assert_size_stride(add_81, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(embedding_4, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_29, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_123, (32768, 768), (768, 1))
        assert_size_stride(view_133, (32768, 12), (768, 1))
        assert_size_stride(mm_49, (32768, 6), (6, 1))
        assert_size_stride(add_84, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_15, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_16, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(rsqrt_30, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(rsqrt_31, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(mul_136, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(mul_137, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_28, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_29, (16, 6, 2048), (12288, 2048, 1))
        assert_size_stride(add_91, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_32, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_138, (32768, 768), (768, 1))
        assert_size_stride(mm_51, (32768, 3072), (3072, 1))
        assert_size_stride(view_140, (32768, 3072), (3072, 1))
        assert_size_stride(add_93, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_33, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_142, (32768, 768), (768, 1))
        assert_size_stride(view_150, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_17, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_18, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(rsqrt_34, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(rsqrt_35, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(mul_152, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(mul_153, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_32, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_33, (16, 6, 2048), (12288, 2048, 1))
        assert_size_stride(add_102, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_36, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_154, (32768, 768), (768, 1))
        assert_size_stride(mm_57, (32768, 3072), (3072, 1))
        assert_size_stride(view_156, (32768, 3072), (3072, 1))
        assert_size_stride(add_104, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(embedding_5, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_37, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_158, (32768, 768), (768, 1))
        assert_size_stride(view_168, (32768, 12), (768, 1))
        assert_size_stride(mm_62, (32768, 6), (6, 1))
        assert_size_stride(add_107, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_19, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_20, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(rsqrt_38, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(rsqrt_39, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(mul_170, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(mul_171, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_36, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_37, (16, 6, 2048), (12288, 2048, 1))
        assert_size_stride(add_114, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_40, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_173, (32768, 768), (768, 1))
        assert_size_stride(mm_64, (32768, 3072), (3072, 1))
        assert_size_stride(view_175, (32768, 3072), (3072, 1))
        assert_size_stride(add_116, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_41, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_177, (32768, 768), (768, 1))
        assert_size_stride(view_185, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_21, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_22, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(rsqrt_42, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(rsqrt_43, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(mul_186, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(mul_187, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_40, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_41, (16, 6, 2048), (12288, 2048, 1))
        assert_size_stride(add_125, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_44, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_189, (32768, 768), (768, 1))
        assert_size_stride(mm_70, (32768, 3072), (3072, 1))
        assert_size_stride(view_191, (32768, 3072), (3072, 1))
        assert_size_stride(add_127, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(embedding_6, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_45, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_193, (32768, 768), (768, 1))
        assert_size_stride(view_203, (32768, 12), (768, 1))
        assert_size_stride(mm_75, (32768, 6), (6, 1))
        assert_size_stride(add_130, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_23, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(cat_24, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(rsqrt_46, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(rsqrt_47, (16, 2048, 6, 1), (12288, 6, 1, 1))
        assert_size_stride(mul_204, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(mul_205, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_44, (16, 2048, 6, 128), (1572864, 768, 128, 1))
        assert_size_stride(getitem_45, (16, 6, 2048), (12288, 2048, 1))
        assert_size_stride(add_137, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_48, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_208, (32768, 768), (768, 1))
        assert_size_stride(mm_77, (32768, 3072), (3072, 1))
        assert_size_stride(view_210, (32768, 3072), (3072, 1))
        assert_size_stride(sub, (16, 2048, 768), (1572864, 768, 1))
        assert_size_stride(rsqrt_49, (16, 2048, 1), (2048, 1, 1))
        assert_size_stride(view_212, (32768, 768), (768, 1))
        assert_size_stride(mm_79, (32768, 32768), (32768, 1))
        assert_size_stride(amax, (32768, 1), (1, 1))
        assert_size_stride(log, (32768, 1), (1, 1))
        assert_size_stride(convert_element_type_343, (), ())
        assert_size_stride(permute_82, (32768, 768), (768, 1))
        assert_size_stride(permute_86, (768, 3072), (3072, 1))
        assert_size_stride(permute_90, (3072, 768), (768, 1))
        assert_size_stride(permute_94, (768, 768), (768, 1))
        assert_size_stride(permute_98, (6, 12), (12, 1))
        assert_size_stride(permute_102, (768, 768), (768, 1))
        assert_size_stride(permute_106, (768, 768), (768, 1))
        assert_size_stride(permute_110, (768, 768), (768, 1))
        assert_size_stride(permute_114, (768, 3072), (3072, 1))
        assert_size_stride(permute_118, (3072, 768), (768, 1))
        assert_size_stride(permute_122, (768, 768), (768, 1))
        assert_size_stride(permute_126, (768, 768), (768, 1))
        assert_size_stride(permute_130, (768, 768), (768, 1))
        assert_size_stride(permute_134, (768, 768), (768, 1))
        assert_size_stride(permute_138, (768, 3072), (3072, 1))
        assert_size_stride(permute_142, (3072, 768), (768, 1))
        assert_size_stride(permute_146, (768, 768), (768, 1))
        assert_size_stride(permute_150, (6, 12), (12, 1))
        assert_size_stride(permute_154, (768, 768), (768, 1))
        assert_size_stride(permute_158, (768, 768), (768, 1))
        assert_size_stride(permute_162, (768, 768), (768, 1))
        assert_size_stride(permute_166, (768, 3072), (3072, 1))
        assert_size_stride(permute_170, (3072, 768), (768, 1))
        assert_size_stride(permute_174, (768, 768), (768, 1))
        assert_size_stride(permute_178, (768, 768), (768, 1))
        assert_size_stride(permute_182, (768, 768), (768, 1))
        assert_size_stride(permute_186, (768, 768), (768, 1))
        assert_size_stride(permute_190, (768, 3072), (3072, 1))
        assert_size_stride(permute_194, (3072, 768), (768, 1))
        assert_size_stride(permute_198, (768, 768), (768, 1))
        assert_size_stride(permute_202, (6, 12), (12, 1))
        assert_size_stride(permute_206, (768, 768), (768, 1))
        assert_size_stride(permute_210, (768, 768), (768, 1))
        assert_size_stride(permute_214, (768, 768), (768, 1))
        assert_size_stride(permute_218, (768, 3072), (3072, 1))
        assert_size_stride(permute_222, (3072, 768), (768, 1))
        assert_size_stride(permute_226, (768, 768), (768, 1))
        assert_size_stride(permute_230, (768, 768), (768, 1))
        assert_size_stride(permute_234, (768, 768), (768, 1))
        assert_size_stride(permute_238, (768, 768), (768, 1))
        assert_size_stride(permute_242, (768, 3072), (3072, 1))
        assert_size_stride(permute_246, (3072, 768), (768, 1))
        assert_size_stride(permute_250, (768, 768), (768, 1))
        assert_size_stride(permute_254, (6, 12), (12, 1))
        assert_size_stride(permute_258, (768, 768), (768, 1))
        assert_size_stride(permute_262, (768, 768), (768, 1))
        assert_size_stride(permute_266, (768, 768), (768, 1))
        assert_size_stride(permute_270, (768, 3072), (3072, 1))
        assert_size_stride(permute_274, (3072, 768), (768, 1))
        assert_size_stride(permute_278, (768, 768), (768, 1))
        assert_size_stride(permute_282, (768, 768), (768, 1))
        assert_size_stride(permute_286, (768, 768), (768, 1))
        assert_size_stride(permute_290, (768, 768), (768, 1))
        assert_size_stride(permute_294, (768, 3072), (3072, 1))
        assert_size_stride(permute_298, (3072, 768), (768, 1))
        assert_size_stride(permute_302, (768, 768), (768, 1))
        assert_size_stride(permute_306, (6, 12), (12, 1))
        assert_size_stride(permute_310, (768, 768), (768, 1))
        assert_size_stride(permute_314, (768, 768), (768, 1))
        assert_size_stride(permute_318, (768, 768), (768, 1))
        assert_size_stride(permute_322, (768, 3072), (3072, 1))
        assert_size_stride(permute_326, (3072, 768), (768, 1))
        assert_size_stride(permute_330, (768, 768), (768, 1))
        assert_size_stride(permute_334, (768, 768), (768, 1))
        assert_size_stride(permute_338, (768, 768), (768, 1))
        assert_size_stride(permute_342, (768, 768), (768, 1))
        assert_size_stride(permute_346, (768, 3072), (3072, 1))
        assert_size_stride(permute_350, (3072, 768), (768, 1))
        assert_size_stride(permute_354, (768, 768), (768, 1))
        assert_size_stride(permute_358, (6, 12), (12, 1))
        assert_size_stride(permute_362, (768, 768), (768, 1))
        assert_size_stride(permute_366, (768, 768), (768, 1))
        assert_size_stride(permute_370, (768, 768), (768, 1))
        assert_size_stride(permute_374, (768, 3072), (3072, 1))
        assert_size_stride(permute_378, (3072, 768), (768, 1))
        assert_size_stride(permute_382, (768, 768), (768, 1))
        assert_size_stride(permute_386, (768, 768), (768, 1))
        assert_size_stride(permute_390, (768, 768), (768, 1))
        assert_size_stride(permute_394, (768, 768), (768, 1))
        assert_size_stride(permute_398, (1, 24), (24, 1))
        assert_size_stride(tangents_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf2 = reinterpret_tensor(mm_79, (16, 2048, 32768), (67108864, 32768, 1), 0); del mm_79  # reuse
            # Topologically Sorted Source Nodes: [view_55, loss, , logits, logits_2, truediv, tanh, logits_3, view_54], Original ATen: [aten.nll_loss_backward, aten.view, aten.nll_loss_forward, aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.mul, aten._log_softmax, aten._log_softmax_backward_data, aten.tanh_backward]
            # [Provenance debug handles] triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_view_0:222
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_view_0.run(buf2, primals_95, tangents_1, convert_element_type_343, amax, log, 32768, 32768, stream=stream0)
            del amax
            del convert_element_type_343
            del log
            del primals_95
            del tangents_1
            buf3 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [logits, logits_2, truediv, tanh, permute_83], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.view, aten.mul, aten.tanh_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:451
            extern_kernels.mm(reinterpret_tensor(buf2, (32768, 32768), (1, 32768), 0), view_212, out=buf3)
            del view_212
            buf4 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [logits, logits_2, truediv, tanh], Original ATen: [aten._unsafe_view, aten._to_copy, aten.div, aten.tanh, aten.view, aten.mul, aten.tanh_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:452
            extern_kernels.mm(reinterpret_tensor(buf2, (32768, 32768), (32768, 1), 0), permute_82, out=buf4)
            del buf2
            del permute_82
            buf5 = empty_strided_cuda((32768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_1:223
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(buf3, buf5, 25165824, stream=stream0)
            buf6 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf7 = empty_strided_cuda((1, 1, 1, 16, 2048, 1), (32768, 32768, 32768, 2048, 1, 32768), torch.float32)
            buf10 = reinterpret_tensor(buf3, (16, 2048, 768), (1572864, 768, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [clone_1, x_77], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.neg, aten.sum]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_2:224
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_2.run(sub, rsqrt_49, buf4, add_81, buf6, buf7, buf10, 32768, 768, stream=stream0)
            buf9 = empty_strided_cuda((1, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [clone_1, x_77], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.neg, aten.sum]
            # [Provenance debug handles] triton_red_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_3:225
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_mul_neg_sum_view_3.run(buf7, buf9, 1, 32768, stream=stream0)
            buf11 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [clone_1, x_77, permute_87], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:453
            extern_kernels.mm(reinterpret_tensor(buf10, (768, 32768), (1, 768), 0), view_210, out=buf11)
            del view_210
            buf12 = empty_strided_cuda((32768, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [clone_1, x_77], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:454
            extern_kernels.mm(reinterpret_tensor(buf10, (32768, 768), (768, 1), 0), permute_86, out=buf12)
            del permute_86
            buf13 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:226
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf11, buf13, 2359296, stream=stream0)
            buf14 = reinterpret_tensor(mm_77, (16, 2048, 3072), (6291456, 3072, 1), 0); del mm_77  # reuse
            # Topologically Sorted Source Nodes: [x_72, relu_11, scalar_tensor_4], Original ATen: [aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.threshold_backward]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5:227
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5.run(buf14, buf12, 100663296, stream=stream0)
            del buf12
            buf15 = reinterpret_tensor(buf11, (3072, 768), (768, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [x_72, relu_11, scalar_tensor_4, permute_91], Original ATen: [aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.threshold_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:455
            extern_kernels.mm(reinterpret_tensor(buf14, (3072, 32768), (1, 3072), 0), view_208, out=buf15)
            del view_208
            buf16 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_72, relu_11, scalar_tensor_4], Original ATen: [aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.threshold_backward, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:456
            extern_kernels.mm(reinterpret_tensor(buf14, (32768, 3072), (3072, 1), 0), permute_90, out=buf16)
            del permute_90
            buf17 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:228
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf15, buf17, 2359296, stream=stream0)
            buf19 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [clone_3, rms_norm_48], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_6:229
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_6.run(buf19, add_137, rsqrt_48, buf16, 32768, 768, stream=stream0)
            del add_137
            del rsqrt_48
            buf20 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [y_34, y_35, permute_95], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:457
            extern_kernels.mm(reinterpret_tensor(buf19, (768, 32768), (1, 768), 0), reinterpret_tensor(getitem_44, (32768, 768), (768, 1), 0), out=buf20)
            buf21 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:458
            extern_kernels.mm(reinterpret_tensor(buf19, (32768, 768), (768, 1), 0), permute_94, out=buf21)
            del permute_94
            buf22 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:230
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf20, buf22, 589824, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf23 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf21, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), mul_204, mul_205, add_130, getitem_44, getitem_45, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_130
            del getitem_44
            del getitem_45
            del mul_204
            del mul_205
            buf24 = buf23[0]
            assert_size_stride(buf24, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf24, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf25 = buf23[1]
            assert_size_stride(buf25, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf25, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf26 = buf23[2]
            assert_size_stride(buf26, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf26, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf23
            buf28 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            # Topologically Sorted Source Nodes: [clone_5, k_46], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:231
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_24, rsqrt_47, buf25, buf28, 196608, 128, stream=stream0)
            buf29 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 196608), torch.float32)
            # Topologically Sorted Source Nodes: [clone_7, q_46], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:232
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_23, rsqrt_46, buf24, buf29, 196608, 128, stream=stream0)
            buf30 = empty_strided_cuda((16, 2048, 6, 64), (786432, 384, 64, 1), torch.bfloat16)
            buf31 = empty_strided_cuda((16, 2048, 6, 64), (786432, 384, 64, 1), torch.bfloat16)
            buf32 = empty_strided_cuda((16, 2048, 6, 64), (786432, 384, 64, 1), torch.bfloat16)
            buf33 = empty_strided_cuda((16, 2048, 6, 64), (786432, 384, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [clone_5, k_46, clone_7, q_46, cos, sin, neg], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.slice, aten.neg, aten.add]
            # [Provenance debug handles] triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9:233
            stream0 = get_raw_stream(0)
            triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9.run(buf25, cat_24, rsqrt_47, buf28, primals_2, primals_3, buf24, cat_23, rsqrt_46, buf29, buf30, buf31, buf32, buf33, 196608, 64, stream=stream0)
            del cat_23
            del cat_24
            del rsqrt_46
            del rsqrt_47
            buf53 = empty_strided_cuda((32768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [loss, linear_75, sigmoid_6, gate_6, unsqueeze_5, full_6], Original ATen: [aten.nll_loss_forward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.embedding_dense_backward]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10:234
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10.run(buf53, 25165824, stream=stream0)
            buf34 = empty_strided_cuda((16, 2048, 6, 1), (12288, 6, 1, 1), torch.bfloat16)
            buf35 = reinterpret_tensor(buf34, (16, 2048, 6), (12288, 6, 1), 0); del buf34  # reuse
            # Topologically Sorted Source Nodes: [loss, linear_75, sigmoid_6, gate_6, unsqueeze_5, ve_11, full_6], Original ATen: [aten.nll_loss_forward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten.squeeze, aten.sigmoid_backward, aten.embedding_dense_backward]
            # [Provenance debug handles] triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11:235
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11.run(buf35, buf26, embedding_6, primals_1, mm_75, buf53, 196608, 128, stream=stream0)
            del embedding_6
            del mm_75
            buf36 = empty_strided_cuda((6, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_75, sigmoid_6, permute_99], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:459
            extern_kernels.mm(reinterpret_tensor(buf35, (6, 32768), (1, 6), 0), view_203, out=buf36)
            del view_203
            buf37 = empty_strided_cuda((32768, 12), (12, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_75, sigmoid_6], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:460
            extern_kernels.mm(reinterpret_tensor(buf35, (32768, 6), (6, 1), 0), permute_98, out=buf37)
            del permute_98
            buf38 = empty_strided_cuda((6, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_12:236
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(buf36, buf38, 72, stream=stream0)
            buf39 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [permute_103], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:461
            extern_kernels.mm(reinterpret_tensor(buf26, (768, 32768), (1, 768), 0), view_193, out=buf39)
            buf40 = reinterpret_tensor(buf25, (32768, 768), (768, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:462
            extern_kernels.mm(reinterpret_tensor(buf26, (32768, 768), (768, 1), 0), permute_102, out=buf40)
            del permute_102
            buf41 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:237
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf39, buf41, 589824, stream=stream0)
            buf42 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter, _generalized_scatter_1], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:238
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf30, buf31, buf42, 25165824, stream=stream0)
            buf43 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter, _generalized_scatter_1, permute_107], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:463
            extern_kernels.mm(reinterpret_tensor(buf42, (768, 32768), (1, 768), 0), view_193, out=buf43)
            buf44 = reinterpret_tensor(buf24, (32768, 768), (768, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter, _generalized_scatter_1], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:464
            extern_kernels.mm(reinterpret_tensor(buf42, (32768, 768), (768, 1), 0), permute_106, out=buf44)
            del permute_106
            buf45 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:239
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf43, buf45, 589824, stream=stream0)
            buf46 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_2, _generalized_scatter_3], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:240
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf32, buf33, buf46, 25165824, stream=stream0)
            buf47 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_2, _generalized_scatter_3, permute_111], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:465
            extern_kernels.mm(reinterpret_tensor(buf46, (768, 32768), (1, 768), 0), view_193, out=buf47)
            del view_193
            buf48 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_2, _generalized_scatter_3], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:466
            extern_kernels.mm(reinterpret_tensor(buf46, (32768, 768), (768, 1), 0), permute_110, out=buf48)
            del permute_110
            buf49 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:241
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf47, buf49, 589824, stream=stream0)
            buf50 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.float32)
            buf95 = empty_strided_cuda((16, 2048, 768), (1572864, 768, 1), torch.float32)
            buf52 = buf19; del buf19  # reuse
            buf60 = reinterpret_tensor(buf46, (32768, 768), (768, 1), 0); del buf46  # reuse
            # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_4, clone_9, getitem_77, mul_144, getitem_78, mul_145, x_70, rms_norm_45, getitem_71, mul_132, getitem_72, mul_133, x_64, rms_norm_41], Original ATen: [aten.view, aten.slice_backward, aten.add, aten._fused_rms_norm_backward, aten.select, aten.mul, aten._to_copy]
            # [Provenance debug handles] triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_14:242
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_14.run(buf52, primals_7, add_127, primals_8, cat, rsqrt_45, add_116, rsqrt_41, buf37, buf40, buf44, buf48, buf50, buf95, buf60, 32768, 768, stream=stream0)
            del rsqrt_45
            buf55 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            # [Provenance debug handles] triton_poi_fused_embedding_dense_backward_15:243
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_15.run(buf53, buf55, 25165824, stream=stream0)
            buf62 = reinterpret_tensor(buf14, (32768, 3072), (3072, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:467
            extern_kernels.mm(buf60, permute_114, out=buf62)
            del permute_114
            buf64 = reinterpret_tensor(mm_70, (16, 2048, 3072), (6291456, 3072, 1), 0); del mm_70  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_66, relu_10], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5:244
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5.run(buf64, buf62, 100663296, stream=stream0)
            del buf62
            buf66 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_66, relu_10], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:468
            extern_kernels.mm(reinterpret_tensor(buf64, (32768, 3072), (3072, 1), 0), permute_118, out=buf66)
            del permute_118
            buf69 = reinterpret_tensor(buf66, (16, 2048, 768), (1572864, 768, 1), 0); del buf66  # reuse
            # Topologically Sorted Source Nodes: [getitem_77, clone_11, rms_norm_44], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_16:245
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_16.run(buf69, add_125, rsqrt_44, buf52, primals_7, 32768, 768, stream=stream0)
            del add_125
            del rsqrt_44
            buf71 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:469
            extern_kernels.mm(reinterpret_tensor(buf69, (32768, 768), (768, 1), 0), permute_122, out=buf71)
            del permute_122
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf73 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf71, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), mul_186, mul_187, view_185, getitem_40, getitem_41, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del getitem_41
            del mul_186
            del mul_187
            del view_185
            buf76 = buf73[2]
            assert_size_stride(buf76, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf76, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf85 = buf71; del buf71  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:470
            extern_kernels.mm(reinterpret_tensor(buf76, (32768, 768), (768, 1), 0), permute_126, out=buf85)
            del permute_126
            buf74 = buf73[0]
            assert_size_stride(buf74, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf74, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf75 = buf73[1]
            assert_size_stride(buf75, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf75, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf73
            buf78 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [clone_13, k_42], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:246
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_22, rsqrt_43, buf75, buf78, 196608, 128, stream=stream0)
            buf79 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [clone_15, q_42], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:247
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_21, rsqrt_42, buf74, buf79, 196608, 128, stream=stream0)
            buf80 = buf33; del buf33  # reuse
            buf81 = buf32; del buf32  # reuse
            buf82 = buf31; del buf31  # reuse
            buf83 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, clone_13, k_42, clone_15, q_42], Original ATen: [aten.slice, aten.neg, aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9:248
            stream0 = get_raw_stream(0)
            triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9.run(buf75, cat_22, rsqrt_43, buf78, primals_2, primals_3, buf74, cat_21, rsqrt_42, buf79, buf80, buf81, buf82, buf83, 196608, 64, stream=stream0)
            del cat_21
            del cat_22
            del rsqrt_42
            del rsqrt_43
            buf87 = buf75; del buf75  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_7, _generalized_scatter_8], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:249
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf80, buf81, buf87, 25165824, stream=stream0)
            buf89 = reinterpret_tensor(buf74, (32768, 768), (768, 1), 0); del buf74  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_7, _generalized_scatter_8], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:471
            extern_kernels.mm(reinterpret_tensor(buf87, (32768, 768), (768, 1), 0), permute_130, out=buf89)
            del permute_130
            buf91 = empty_strided_cuda((16, 2048, 6, 128), (1572864, 768, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_9, _generalized_scatter_10], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:250
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf82, buf83, buf91, 25165824, stream=stream0)
            buf93 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_9, _generalized_scatter_10], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:472
            extern_kernels.mm(reinterpret_tensor(buf91, (32768, 768), (768, 1), 0), permute_134, out=buf93)
            del permute_134
            buf97 = reinterpret_tensor(buf85, (16, 2048, 768), (1572864, 768, 1), 0); del buf85  # reuse
            buf102 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
            buf56 = reinterpret_tensor(buf7, (16, 2048, 1), (2048, 1, 32768), 0); del buf7  # reuse
            buf58 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf98 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            buf100 = empty_strided_cuda((16, 2048, 1), (2048, 1, 32768), torch.float32)
            # Topologically Sorted Source Nodes: [clone_17, getitem_71], Original ATen: [aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_17:251
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_17.run(buf97, buf95, buf89, buf93, buf69, rsqrt_41, primals_7, buf52, cat, add_127, add_116, buf102, buf56, buf58, buf98, buf100, 32768, 768, stream=stream0)
            del add_116
            del add_127
            del buf89
            del buf93
            del rsqrt_41
            buf57 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:252
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf56, buf57, 1, 32768, stream=stream0)
            buf59 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:253
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf58, buf59, 1, 32768, stream=stream0)
            buf61 = reinterpret_tensor(buf15, (768, 3072), (3072, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [permute_115], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:473
            extern_kernels.mm(reinterpret_tensor(buf60, (768, 32768), (1, 768), 0), view_191, out=buf61)
            del buf60
            del view_191
            buf63 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:254
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf61, buf63, 2359296, stream=stream0)
            buf65 = reinterpret_tensor(buf61, (3072, 768), (768, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_66, relu_10, permute_119], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:474
            extern_kernels.mm(reinterpret_tensor(buf64, (3072, 32768), (1, 3072), 0), view_189, out=buf65)
            del view_189
            buf67 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:255
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf65, buf67, 2359296, stream=stream0)
            buf70 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [y_31, y_32, permute_123], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:475
            extern_kernels.mm(reinterpret_tensor(buf69, (768, 32768), (1, 768), 0), reinterpret_tensor(getitem_40, (32768, 768), (768, 1), 0), out=buf70)
            del getitem_40
            buf72 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:256
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf70, buf72, 589824, stream=stream0)
            buf84 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [permute_127], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:476
            extern_kernels.mm(reinterpret_tensor(buf76, (768, 32768), (1, 768), 0), view_177, out=buf84)
            buf86 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:257
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf84, buf86, 589824, stream=stream0)
            buf88 = buf84; del buf84  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_7, _generalized_scatter_8, permute_131], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:477
            extern_kernels.mm(reinterpret_tensor(buf87, (768, 32768), (1, 768), 0), view_177, out=buf88)
            buf90 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:258
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf88, buf90, 589824, stream=stream0)
            buf92 = buf88; del buf88  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_9, _generalized_scatter_10, permute_135], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:478
            extern_kernels.mm(reinterpret_tensor(buf91, (768, 32768), (1, 768), 0), view_177, out=buf92)
            del view_177
            buf94 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:259
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf92, buf94, 589824, stream=stream0)
            buf99 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:260
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf98, buf99, 1, 32768, stream=stream0)
            buf101 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:261
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf100, buf101, 1, 32768, stream=stream0)
            buf103 = reinterpret_tensor(buf65, (768, 3072), (3072, 1), 0); del buf65  # reuse
            # Topologically Sorted Source Nodes: [permute_139], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:479
            extern_kernels.mm(reinterpret_tensor(buf102, (768, 32768), (1, 768), 0), view_175, out=buf103)
            del view_175
            buf104 = reinterpret_tensor(buf64, (32768, 3072), (3072, 1), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:480
            extern_kernels.mm(buf102, permute_138, out=buf104)
            del permute_138
            buf105 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:262
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf103, buf105, 2359296, stream=stream0)
            buf106 = reinterpret_tensor(mm_64, (16, 2048, 3072), (6291456, 3072, 1), 0); del mm_64  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_60, relu_9], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5:263
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5.run(buf106, buf104, 100663296, stream=stream0)
            del buf104
            buf107 = reinterpret_tensor(buf103, (3072, 768), (768, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_60, relu_9, permute_143], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:481
            extern_kernels.mm(reinterpret_tensor(buf106, (3072, 32768), (1, 3072), 0), view_173, out=buf107)
            del view_173
            buf108 = buf102; del buf102  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_60, relu_9], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:482
            extern_kernels.mm(reinterpret_tensor(buf106, (32768, 3072), (3072, 1), 0), permute_142, out=buf108)
            del permute_142
            buf109 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:264
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf107, buf109, 2359296, stream=stream0)
            buf111 = reinterpret_tensor(buf108, (16, 2048, 768), (1572864, 768, 1), 0); del buf108  # reuse
            # Topologically Sorted Source Nodes: [getitem_71, clone_19, rms_norm_40], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_19:265
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_19.run(buf111, add_114, rsqrt_40, buf97, primals_7, 32768, 768, stream=stream0)
            del add_114
            del rsqrt_40
            buf112 = buf92; del buf92  # reuse
            # Topologically Sorted Source Nodes: [y_28, y_29, permute_147], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:483
            extern_kernels.mm(reinterpret_tensor(buf111, (768, 32768), (1, 768), 0), reinterpret_tensor(getitem_36, (32768, 768), (768, 1), 0), out=buf112)
            buf113 = reinterpret_tensor(buf91, (32768, 768), (768, 1), 0); del buf91  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:484
            extern_kernels.mm(reinterpret_tensor(buf111, (32768, 768), (768, 1), 0), permute_146, out=buf113)
            del permute_146
            buf114 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:266
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf112, buf114, 589824, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf115 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf113, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), mul_170, mul_171, add_107, getitem_36, getitem_37, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_107
            del getitem_36
            del getitem_37
            del mul_170
            del mul_171
            buf116 = buf115[0]
            assert_size_stride(buf116, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf116, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf117 = buf115[1]
            assert_size_stride(buf117, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf117, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf118 = buf115[2]
            assert_size_stride(buf118, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf118, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf115
            buf120 = buf79; del buf79  # reuse
            # Topologically Sorted Source Nodes: [clone_21, k_38], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:267
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_20, rsqrt_39, buf117, buf120, 196608, 128, stream=stream0)
            buf121 = buf78; del buf78  # reuse
            # Topologically Sorted Source Nodes: [clone_23, q_38], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:268
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_19, rsqrt_38, buf116, buf121, 196608, 128, stream=stream0)
            buf122 = buf83; del buf83  # reuse
            buf123 = buf82; del buf82  # reuse
            buf124 = buf81; del buf81  # reuse
            buf125 = buf80; del buf80  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, clone_21, k_38, clone_23, q_38], Original ATen: [aten.slice, aten.neg, aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9:269
            stream0 = get_raw_stream(0)
            triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9.run(buf117, cat_20, rsqrt_39, buf120, primals_2, primals_3, buf116, cat_19, rsqrt_38, buf121, buf122, buf123, buf124, buf125, 196608, 64, stream=stream0)
            del cat_19
            del cat_20
            del rsqrt_38
            del rsqrt_39
            buf145 = reinterpret_tensor(buf95, (32768, 768), (768, 1), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [loss, full_6, linear_62, sigmoid_5, gate_5, unsqueeze_4], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10:270
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10.run(buf145, 25165824, stream=stream0)
            buf126 = reinterpret_tensor(buf35, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf35  # reuse
            buf127 = reinterpret_tensor(buf126, (16, 2048, 6), (12288, 6, 1), 0); del buf126  # reuse
            # Topologically Sorted Source Nodes: [loss, full_6, linear_62, sigmoid_5, gate_5, unsqueeze_4, ve_9], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten.squeeze, aten.sigmoid_backward]
            # [Provenance debug handles] triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11:271
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11.run(buf127, buf118, embedding_5, primals_1, mm_62, buf145, 196608, 128, stream=stream0)
            del embedding_5
            del mm_62
            buf128 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [linear_62, sigmoid_5, permute_151], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:485
            extern_kernels.mm(reinterpret_tensor(buf127, (6, 32768), (1, 6), 0), view_168, out=buf128)
            del view_168
            buf129 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [linear_62, sigmoid_5], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:486
            extern_kernels.mm(reinterpret_tensor(buf127, (32768, 6), (6, 1), 0), permute_150, out=buf129)
            del permute_150
            buf130 = empty_strided_cuda((6, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_12:272
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(buf128, buf130, 72, stream=stream0)
            buf131 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [permute_155], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:487
            extern_kernels.mm(reinterpret_tensor(buf118, (768, 32768), (1, 768), 0), view_158, out=buf131)
            buf132 = reinterpret_tensor(buf117, (32768, 768), (768, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:488
            extern_kernels.mm(reinterpret_tensor(buf118, (32768, 768), (768, 1), 0), permute_154, out=buf132)
            del permute_154
            buf133 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:273
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf131, buf133, 589824, stream=stream0)
            buf134 = buf118; del buf118  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_13, _generalized_scatter_14], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:274
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf122, buf123, buf134, 25165824, stream=stream0)
            buf135 = buf131; del buf131  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_13, _generalized_scatter_14, permute_159], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:489
            extern_kernels.mm(reinterpret_tensor(buf134, (768, 32768), (1, 768), 0), view_158, out=buf135)
            buf136 = reinterpret_tensor(buf116, (32768, 768), (768, 1), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_13, _generalized_scatter_14], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:490
            extern_kernels.mm(reinterpret_tensor(buf134, (32768, 768), (768, 1), 0), permute_158, out=buf136)
            del permute_158
            buf137 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:275
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf135, buf137, 589824, stream=stream0)
            buf138 = buf134; del buf134  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_15, _generalized_scatter_16], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:276
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf124, buf125, buf138, 25165824, stream=stream0)
            buf139 = buf135; del buf135  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_15, _generalized_scatter_16, permute_163], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:491
            extern_kernels.mm(reinterpret_tensor(buf138, (768, 32768), (1, 768), 0), view_158, out=buf139)
            del view_158
            buf140 = buf113; del buf113  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_15, _generalized_scatter_16], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:492
            extern_kernels.mm(reinterpret_tensor(buf138, (32768, 768), (768, 1), 0), permute_162, out=buf140)
            del permute_162
            buf141 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:277
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf139, buf141, 589824, stream=stream0)
            buf142 = reinterpret_tensor(buf53, (16, 2048, 768), (1572864, 768, 1), 0); del buf53  # reuse
            buf187 = buf50; del buf50  # reuse
            buf144 = buf111; del buf111  # reuse
            buf152 = reinterpret_tensor(buf138, (32768, 768), (768, 1), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_17, clone_25, getitem_64, mul_118, getitem_65, mul_119, x_58, rms_norm_37, getitem_58, mul_106, getitem_59, mul_107, x_52, rms_norm_33], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select, aten.mul, aten._to_copy]
            # [Provenance debug handles] triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_20:278
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_20.run(buf144, primals_7, add_104, primals_8, cat, rsqrt_37, add_93, rsqrt_33, buf129, buf132, buf136, buf140, buf142, buf187, buf152, 32768, 768, stream=stream0)
            del rsqrt_37
            buf147 = buf140; del buf140  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            # [Provenance debug handles] triton_poi_fused_embedding_dense_backward_15:279
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_15.run(buf145, buf147, 25165824, stream=stream0)
            buf154 = reinterpret_tensor(buf106, (32768, 3072), (3072, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:493
            extern_kernels.mm(buf152, permute_166, out=buf154)
            del permute_166
            buf156 = reinterpret_tensor(mm_57, (16, 2048, 3072), (6291456, 3072, 1), 0); del mm_57  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_54, relu_8], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5:280
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5.run(buf156, buf154, 100663296, stream=stream0)
            del buf154
            buf158 = buf136; del buf136  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_54, relu_8], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:494
            extern_kernels.mm(reinterpret_tensor(buf156, (32768, 3072), (3072, 1), 0), permute_170, out=buf158)
            del permute_170
            buf161 = reinterpret_tensor(buf158, (16, 2048, 768), (1572864, 768, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [getitem_64, clone_27, rms_norm_36], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_21:281
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_21.run(buf161, add_102, rsqrt_36, buf144, primals_7, 32768, 768, stream=stream0)
            del add_102
            del rsqrt_36
            buf163 = buf132; del buf132  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:495
            extern_kernels.mm(reinterpret_tensor(buf161, (32768, 768), (768, 1), 0), permute_174, out=buf163)
            del permute_174
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf165 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf163, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), mul_152, mul_153, view_150, getitem_32, getitem_33, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del getitem_33
            del mul_152
            del mul_153
            del view_150
            buf168 = buf165[2]
            assert_size_stride(buf168, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf168, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf177 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:496
            extern_kernels.mm(reinterpret_tensor(buf168, (32768, 768), (768, 1), 0), permute_178, out=buf177)
            del permute_178
            buf166 = buf165[0]
            assert_size_stride(buf166, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf166, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf167 = buf165[1]
            assert_size_stride(buf167, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf167, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf165
            buf170 = buf121; del buf121  # reuse
            # Topologically Sorted Source Nodes: [clone_29, k_34], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:282
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_18, rsqrt_35, buf167, buf170, 196608, 128, stream=stream0)
            buf171 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [clone_31, q_34], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:283
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_17, rsqrt_34, buf166, buf171, 196608, 128, stream=stream0)
            buf172 = buf125; del buf125  # reuse
            buf173 = buf124; del buf124  # reuse
            buf174 = buf123; del buf123  # reuse
            buf175 = buf122; del buf122  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, clone_29, k_34, clone_31, q_34], Original ATen: [aten.slice, aten.neg, aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9:284
            stream0 = get_raw_stream(0)
            triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9.run(buf167, cat_18, rsqrt_35, buf170, primals_2, primals_3, buf166, cat_17, rsqrt_34, buf171, buf172, buf173, buf174, buf175, 196608, 64, stream=stream0)
            del cat_17
            del cat_18
            del rsqrt_34
            del rsqrt_35
            buf179 = buf167; del buf167  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_20, _generalized_scatter_21], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:285
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf172, buf173, buf179, 25165824, stream=stream0)
            buf181 = reinterpret_tensor(buf166, (32768, 768), (768, 1), 0); del buf166  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_20, _generalized_scatter_21], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:497
            extern_kernels.mm(reinterpret_tensor(buf179, (32768, 768), (768, 1), 0), permute_182, out=buf181)
            del permute_182
            buf183 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_22, _generalized_scatter_23], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:286
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf174, buf175, buf183, 25165824, stream=stream0)
            buf185 = reinterpret_tensor(buf76, (32768, 768), (768, 1), 0); del buf76  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_22, _generalized_scatter_23], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:498
            extern_kernels.mm(reinterpret_tensor(buf183, (32768, 768), (768, 1), 0), permute_186, out=buf185)
            del permute_186
            buf189 = reinterpret_tensor(buf177, (16, 2048, 768), (1572864, 768, 1), 0); del buf177  # reuse
            buf194 = reinterpret_tensor(buf69, (32768, 768), (768, 1), 0); del buf69  # reuse
            buf148 = buf100; del buf100  # reuse
            buf150 = buf98; del buf98  # reuse
            buf190 = buf58; del buf58  # reuse
            buf192 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [clone_33, getitem_58], Original ATen: [aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_22:287
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_22.run(buf189, buf187, buf181, buf185, buf161, rsqrt_33, primals_7, buf144, cat, add_104, add_93, buf194, buf148, buf150, buf190, buf192, 32768, 768, stream=stream0)
            del add_104
            del add_93
            del buf181
            del buf185
            del rsqrt_33
            buf149 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:288
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf148, buf149, 1, 32768, stream=stream0)
            del buf148
            buf151 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:289
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf150, buf151, 1, 32768, stream=stream0)
            buf153 = reinterpret_tensor(buf107, (768, 3072), (3072, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [permute_167], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:499
            extern_kernels.mm(reinterpret_tensor(buf152, (768, 32768), (1, 768), 0), view_156, out=buf153)
            del buf152
            del view_156
            buf155 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:290
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf153, buf155, 2359296, stream=stream0)
            buf157 = reinterpret_tensor(buf153, (3072, 768), (768, 1), 0); del buf153  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_54, relu_8, permute_171], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:500
            extern_kernels.mm(reinterpret_tensor(buf156, (3072, 32768), (1, 3072), 0), view_154, out=buf157)
            del view_154
            buf159 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:291
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf157, buf159, 2359296, stream=stream0)
            buf162 = buf139; del buf139  # reuse
            # Topologically Sorted Source Nodes: [y_25, y_26, permute_175], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:501
            extern_kernels.mm(reinterpret_tensor(buf161, (768, 32768), (1, 768), 0), reinterpret_tensor(getitem_32, (32768, 768), (768, 1), 0), out=buf162)
            del buf161
            del getitem_32
            buf164 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:292
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf162, buf164, 589824, stream=stream0)
            buf176 = buf162; del buf162  # reuse
            # Topologically Sorted Source Nodes: [permute_179], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:502
            extern_kernels.mm(reinterpret_tensor(buf168, (768, 32768), (1, 768), 0), view_142, out=buf176)
            del buf168
            buf178 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:293
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf176, buf178, 589824, stream=stream0)
            buf180 = buf176; del buf176  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_20, _generalized_scatter_21, permute_183], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:503
            extern_kernels.mm(reinterpret_tensor(buf179, (768, 32768), (1, 768), 0), view_142, out=buf180)
            del buf179
            buf182 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:294
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf180, buf182, 589824, stream=stream0)
            buf184 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_22, _generalized_scatter_23, permute_187], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:504
            extern_kernels.mm(reinterpret_tensor(buf183, (768, 32768), (1, 768), 0), view_142, out=buf184)
            del view_142
            buf186 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:295
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf184, buf186, 589824, stream=stream0)
            buf191 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:296
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf190, buf191, 1, 32768, stream=stream0)
            buf193 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:297
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf192, buf193, 1, 32768, stream=stream0)
            buf195 = reinterpret_tensor(buf157, (768, 3072), (3072, 1), 0); del buf157  # reuse
            # Topologically Sorted Source Nodes: [permute_191], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:505
            extern_kernels.mm(reinterpret_tensor(buf194, (768, 32768), (1, 768), 0), view_140, out=buf195)
            del view_140
            buf196 = reinterpret_tensor(buf156, (32768, 3072), (3072, 1), 0); del buf156  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:506
            extern_kernels.mm(buf194, permute_190, out=buf196)
            del permute_190
            buf197 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:298
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf195, buf197, 2359296, stream=stream0)
            buf198 = reinterpret_tensor(mm_51, (16, 2048, 3072), (6291456, 3072, 1), 0); del mm_51  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_48, relu_7], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5:299
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5.run(buf198, buf196, 100663296, stream=stream0)
            del buf196
            buf199 = reinterpret_tensor(buf195, (3072, 768), (768, 1), 0); del buf195  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_48, relu_7, permute_195], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:507
            extern_kernels.mm(reinterpret_tensor(buf198, (3072, 32768), (1, 3072), 0), view_138, out=buf199)
            del view_138
            buf200 = buf194; del buf194  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_48, relu_7], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:508
            extern_kernels.mm(reinterpret_tensor(buf198, (32768, 3072), (3072, 1), 0), permute_194, out=buf200)
            del permute_194
            buf201 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:300
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf199, buf201, 2359296, stream=stream0)
            buf203 = reinterpret_tensor(buf200, (16, 2048, 768), (1572864, 768, 1), 0); del buf200  # reuse
            # Topologically Sorted Source Nodes: [getitem_58, clone_35, rms_norm_32], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_23:301
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_23.run(buf203, add_91, rsqrt_32, buf189, primals_7, 32768, 768, stream=stream0)
            del add_91
            del rsqrt_32
            buf204 = buf184; del buf184  # reuse
            # Topologically Sorted Source Nodes: [y_22, y_23, permute_199], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:509
            extern_kernels.mm(reinterpret_tensor(buf203, (768, 32768), (1, 768), 0), reinterpret_tensor(getitem_28, (32768, 768), (768, 1), 0), out=buf204)
            buf205 = reinterpret_tensor(buf183, (32768, 768), (768, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:510
            extern_kernels.mm(reinterpret_tensor(buf203, (32768, 768), (768, 1), 0), permute_198, out=buf205)
            del permute_198
            buf206 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:302
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf204, buf206, 589824, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf207 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf205, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), mul_136, mul_137, add_84, getitem_28, getitem_29, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_84
            del getitem_28
            del getitem_29
            del mul_136
            del mul_137
            buf208 = buf207[0]
            assert_size_stride(buf208, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf208, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf209 = buf207[1]
            assert_size_stride(buf209, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf209, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf210 = buf207[2]
            assert_size_stride(buf210, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf210, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf207
            buf212 = buf171; del buf171  # reuse
            # Topologically Sorted Source Nodes: [clone_37, k_30], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:303
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_16, rsqrt_31, buf209, buf212, 196608, 128, stream=stream0)
            buf213 = buf170; del buf170  # reuse
            # Topologically Sorted Source Nodes: [clone_39, q_30], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:304
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_15, rsqrt_30, buf208, buf213, 196608, 128, stream=stream0)
            buf214 = buf175; del buf175  # reuse
            buf215 = buf174; del buf174  # reuse
            buf216 = buf173; del buf173  # reuse
            buf217 = buf172; del buf172  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, clone_37, k_30, clone_39, q_30], Original ATen: [aten.slice, aten.neg, aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9:305
            stream0 = get_raw_stream(0)
            triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9.run(buf209, cat_16, rsqrt_31, buf212, primals_2, primals_3, buf208, cat_15, rsqrt_30, buf213, buf214, buf215, buf216, buf217, 196608, 64, stream=stream0)
            del cat_15
            del cat_16
            del rsqrt_30
            del rsqrt_31
            buf237 = reinterpret_tensor(buf187, (32768, 768), (768, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [loss, full_6, linear_49, sigmoid_4, gate_4, unsqueeze_3], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10:306
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10.run(buf237, 25165824, stream=stream0)
            buf218 = reinterpret_tensor(buf127, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf127  # reuse
            buf219 = reinterpret_tensor(buf218, (16, 2048, 6), (12288, 6, 1), 0); del buf218  # reuse
            # Topologically Sorted Source Nodes: [loss, full_6, linear_49, sigmoid_4, gate_4, unsqueeze_3, ve_7], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten.squeeze, aten.sigmoid_backward]
            # [Provenance debug handles] triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11:307
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11.run(buf219, buf210, embedding_4, primals_1, mm_49, buf237, 196608, 128, stream=stream0)
            del embedding_4
            del mm_49
            buf220 = buf128; del buf128  # reuse
            # Topologically Sorted Source Nodes: [linear_49, sigmoid_4, permute_203], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:511
            extern_kernels.mm(reinterpret_tensor(buf219, (6, 32768), (1, 6), 0), view_133, out=buf220)
            del view_133
            buf221 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [linear_49, sigmoid_4], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:512
            extern_kernels.mm(reinterpret_tensor(buf219, (32768, 6), (6, 1), 0), permute_202, out=buf221)
            del permute_202
            buf222 = empty_strided_cuda((6, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_12:308
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(buf220, buf222, 72, stream=stream0)
            buf223 = buf204; del buf204  # reuse
            # Topologically Sorted Source Nodes: [permute_207], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:513
            extern_kernels.mm(reinterpret_tensor(buf210, (768, 32768), (1, 768), 0), view_123, out=buf223)
            buf224 = reinterpret_tensor(buf209, (32768, 768), (768, 1), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:514
            extern_kernels.mm(reinterpret_tensor(buf210, (32768, 768), (768, 1), 0), permute_206, out=buf224)
            del permute_206
            buf225 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:309
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf223, buf225, 589824, stream=stream0)
            buf226 = buf210; del buf210  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_26, _generalized_scatter_27], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:310
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf214, buf215, buf226, 25165824, stream=stream0)
            buf227 = buf223; del buf223  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_26, _generalized_scatter_27, permute_211], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:515
            extern_kernels.mm(reinterpret_tensor(buf226, (768, 32768), (1, 768), 0), view_123, out=buf227)
            buf228 = reinterpret_tensor(buf208, (32768, 768), (768, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_26, _generalized_scatter_27], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:516
            extern_kernels.mm(reinterpret_tensor(buf226, (32768, 768), (768, 1), 0), permute_210, out=buf228)
            del permute_210
            buf229 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:311
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf227, buf229, 589824, stream=stream0)
            buf230 = buf226; del buf226  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_28, _generalized_scatter_29], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:312
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf216, buf217, buf230, 25165824, stream=stream0)
            buf231 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_28, _generalized_scatter_29, permute_215], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:517
            extern_kernels.mm(reinterpret_tensor(buf230, (768, 32768), (1, 768), 0), view_123, out=buf231)
            del view_123
            buf232 = buf205; del buf205  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_28, _generalized_scatter_29], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:518
            extern_kernels.mm(reinterpret_tensor(buf230, (32768, 768), (768, 1), 0), permute_214, out=buf232)
            del buf230
            del permute_214
            buf233 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:313
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf231, buf233, 589824, stream=stream0)
            buf234 = reinterpret_tensor(buf145, (16, 2048, 768), (1572864, 768, 1), 0); del buf145  # reuse
            buf280 = buf142; del buf142  # reuse
            buf236 = buf203; del buf203  # reuse
            buf242 = buf52; del buf52  # reuse
            buf245 = reinterpret_tensor(buf4, (16, 2048, 768), (1572864, 768, 1), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [clone_1, x_77, to_87, full_5, getitem_78, getitem_72, getitem_65, getitem_59, _generalized_scatter_30, clone_41, getitem_51, mul_92, getitem_52, mul_93, x_46, rms_norm_29, getitem_45, mul_80, getitem_46, mul_81, x_40, rms_norm_25], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.neg, aten.slice_backward, aten.select, aten.add]
            # [Provenance debug handles] triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_neg_select_slice_backward_view_24:314
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_neg_select_slice_backward_view_24.run(buf236, buf242, buf245, primals_7, add_81, primals_8, cat, rsqrt_29, add_70, rsqrt_25, buf221, buf224, buf228, buf232, buf97, buf144, buf189, sub, rsqrt_49, buf6, primals_93, buf234, buf280, 32768, 768, stream=stream0)
            del primals_93
            del rsqrt_29
            del rsqrt_49
            del sub
            buf239 = reinterpret_tensor(buf97, (32768, 768), (768, 1), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            # [Provenance debug handles] triton_poi_fused_embedding_dense_backward_15:315
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_15.run(buf237, buf239, 25165824, stream=stream0)
            buf247 = reinterpret_tensor(buf198, (32768, 3072), (3072, 1), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:519
            extern_kernels.mm(reinterpret_tensor(buf245, (32768, 768), (768, 1), 0), permute_218, out=buf247)
            del permute_218
            buf249 = reinterpret_tensor(mm_44, (16, 2048, 3072), (6291456, 3072, 1), 0); del mm_44  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_42, relu_6], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5:316
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5.run(buf249, buf247, 100663296, stream=stream0)
            del buf247
            buf251 = buf232; del buf232  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_42, relu_6], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:520
            extern_kernels.mm(reinterpret_tensor(buf249, (32768, 3072), (3072, 1), 0), permute_222, out=buf251)
            del permute_222
            buf254 = reinterpret_tensor(buf251, (16, 2048, 768), (1572864, 768, 1), 0); del buf251  # reuse
            # Topologically Sorted Source Nodes: [clone_43, rms_norm_28], Original ATen: [aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.mul, aten.add]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_25:317
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_view_25.run(buf254, add_79, rsqrt_28, buf245, 32768, 768, stream=stream0)
            del add_79
            del rsqrt_28
            buf256 = buf228; del buf228  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:521
            extern_kernels.mm(reinterpret_tensor(buf254, (32768, 768), (768, 1), 0), permute_226, out=buf256)
            del permute_226
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf258 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf256, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), mul_118, mul_119, view_115, getitem_24, getitem_25, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del getitem_25
            del mul_118
            del mul_119
            del view_115
            buf261 = buf258[2]
            assert_size_stride(buf261, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf261, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf270 = buf256; del buf256  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:522
            extern_kernels.mm(reinterpret_tensor(buf261, (32768, 768), (768, 1), 0), permute_230, out=buf270)
            del permute_230
            buf259 = buf258[0]
            assert_size_stride(buf259, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf259, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf260 = buf258[1]
            assert_size_stride(buf260, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf260, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf258
            buf263 = buf213; del buf213  # reuse
            # Topologically Sorted Source Nodes: [clone_45, k_26], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:318
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_14, rsqrt_27, buf260, buf263, 196608, 128, stream=stream0)
            buf264 = buf212; del buf212  # reuse
            # Topologically Sorted Source Nodes: [clone_47, q_26], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:319
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_13, rsqrt_26, buf259, buf264, 196608, 128, stream=stream0)
            buf265 = buf217; del buf217  # reuse
            buf266 = buf216; del buf216  # reuse
            buf267 = buf215; del buf215  # reuse
            buf268 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, clone_45, k_26, clone_47, q_26], Original ATen: [aten.slice, aten.neg, aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9:320
            stream0 = get_raw_stream(0)
            triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9.run(buf260, cat_14, rsqrt_27, buf263, primals_2, primals_3, buf259, cat_13, rsqrt_26, buf264, buf265, buf266, buf267, buf268, 196608, 64, stream=stream0)
            del cat_13
            del cat_14
            del rsqrt_26
            del rsqrt_27
            buf272 = buf260; del buf260  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_33, _generalized_scatter_34], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:321
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf265, buf266, buf272, 25165824, stream=stream0)
            buf274 = reinterpret_tensor(buf259, (32768, 768), (768, 1), 0); del buf259  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_33, _generalized_scatter_34], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:523
            extern_kernels.mm(reinterpret_tensor(buf272, (32768, 768), (768, 1), 0), permute_234, out=buf274)
            del permute_234
            buf276 = reinterpret_tensor(buf224, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_35, _generalized_scatter_36], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:322
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf267, buf268, buf276, 25165824, stream=stream0)
            buf278 = reinterpret_tensor(buf189, (32768, 768), (768, 1), 0); del buf189  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_35, _generalized_scatter_36], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:524
            extern_kernels.mm(reinterpret_tensor(buf276, (32768, 768), (768, 1), 0), permute_238, out=buf278)
            del permute_238
            buf282 = reinterpret_tensor(buf270, (16, 2048, 768), (1572864, 768, 1), 0); del buf270  # reuse
            buf287 = reinterpret_tensor(buf144, (32768, 768), (768, 1), 0); del buf144  # reuse
            buf240 = buf6; del buf6  # reuse
            buf243 = buf192; del buf192  # reuse
            buf283 = buf190; del buf190  # reuse
            buf285 = buf150; del buf150  # reuse
            # Topologically Sorted Source Nodes: [clone_49, getitem_45], Original ATen: [aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_26:323
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_26.run(buf282, buf280, buf274, buf278, buf254, rsqrt_25, primals_7, buf236, cat, add_81, add_70, buf287, buf240, buf243, buf283, buf285, 32768, 768, stream=stream0)
            del add_70
            del add_81
            del buf236
            del buf274
            del buf278
            del rsqrt_25
            buf241 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:324
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf240, buf241, 1, 32768, stream=stream0)
            buf244 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:325
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf243, buf244, 1, 32768, stream=stream0)
            buf246 = reinterpret_tensor(buf199, (768, 3072), (3072, 1), 0); del buf199  # reuse
            # Topologically Sorted Source Nodes: [permute_219], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:525
            extern_kernels.mm(reinterpret_tensor(buf245, (768, 32768), (1, 768), 0), view_121, out=buf246)
            del buf245
            del view_121
            buf248 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:326
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf246, buf248, 2359296, stream=stream0)
            buf250 = reinterpret_tensor(buf246, (3072, 768), (768, 1), 0); del buf246  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_42, relu_6, permute_223], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:526
            extern_kernels.mm(reinterpret_tensor(buf249, (3072, 32768), (1, 3072), 0), view_119, out=buf250)
            del view_119
            buf252 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:327
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf250, buf252, 2359296, stream=stream0)
            buf255 = buf231; del buf231  # reuse
            # Topologically Sorted Source Nodes: [y_19, y_20, permute_227], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:527
            extern_kernels.mm(reinterpret_tensor(buf254, (768, 32768), (1, 768), 0), reinterpret_tensor(getitem_24, (32768, 768), (768, 1), 0), out=buf255)
            del getitem_24
            buf257 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:328
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf255, buf257, 589824, stream=stream0)
            buf269 = buf255; del buf255  # reuse
            # Topologically Sorted Source Nodes: [permute_231], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:528
            extern_kernels.mm(reinterpret_tensor(buf261, (768, 32768), (1, 768), 0), view_107, out=buf269)
            buf271 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:329
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf269, buf271, 589824, stream=stream0)
            buf273 = buf269; del buf269  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_33, _generalized_scatter_34, permute_235], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:529
            extern_kernels.mm(reinterpret_tensor(buf272, (768, 32768), (1, 768), 0), view_107, out=buf273)
            buf275 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:330
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf273, buf275, 589824, stream=stream0)
            buf277 = buf273; del buf273  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_35, _generalized_scatter_36, permute_239], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:530
            extern_kernels.mm(reinterpret_tensor(buf276, (768, 32768), (1, 768), 0), view_107, out=buf277)
            del view_107
            buf279 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:331
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf277, buf279, 589824, stream=stream0)
            buf284 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:332
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf283, buf284, 1, 32768, stream=stream0)
            buf286 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:333
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf285, buf286, 1, 32768, stream=stream0)
            buf288 = reinterpret_tensor(buf250, (768, 3072), (3072, 1), 0); del buf250  # reuse
            # Topologically Sorted Source Nodes: [permute_243], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:531
            extern_kernels.mm(reinterpret_tensor(buf287, (768, 32768), (1, 768), 0), view_105, out=buf288)
            del view_105
            buf289 = reinterpret_tensor(buf249, (32768, 3072), (3072, 1), 0); del buf249  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:532
            extern_kernels.mm(buf287, permute_242, out=buf289)
            del permute_242
            buf290 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:334
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf288, buf290, 2359296, stream=stream0)
            buf291 = reinterpret_tensor(mm_38, (16, 2048, 3072), (6291456, 3072, 1), 0); del mm_38  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_36, relu_5], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5:335
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5.run(buf291, buf289, 100663296, stream=stream0)
            del buf289
            buf292 = reinterpret_tensor(buf288, (3072, 768), (768, 1), 0); del buf288  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_36, relu_5, permute_247], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:533
            extern_kernels.mm(reinterpret_tensor(buf291, (3072, 32768), (1, 3072), 0), view_103, out=buf292)
            del view_103
            buf293 = buf287; del buf287  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_36, relu_5], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:534
            extern_kernels.mm(reinterpret_tensor(buf291, (32768, 3072), (3072, 1), 0), permute_246, out=buf293)
            del permute_246
            buf294 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:336
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf292, buf294, 2359296, stream=stream0)
            buf296 = reinterpret_tensor(buf293, (16, 2048, 768), (1572864, 768, 1), 0); del buf293  # reuse
            # Topologically Sorted Source Nodes: [getitem_45, clone_51, rms_norm_24], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_27:337
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_27.run(buf296, add_68, rsqrt_24, buf282, primals_7, 32768, 768, stream=stream0)
            del add_68
            del rsqrt_24
            buf297 = buf277; del buf277  # reuse
            # Topologically Sorted Source Nodes: [y_16, y_17, permute_251], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:535
            extern_kernels.mm(reinterpret_tensor(buf296, (768, 32768), (1, 768), 0), reinterpret_tensor(getitem_20, (32768, 768), (768, 1), 0), out=buf297)
            buf298 = reinterpret_tensor(buf276, (32768, 768), (768, 1), 0); del buf276  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:536
            extern_kernels.mm(reinterpret_tensor(buf296, (32768, 768), (768, 1), 0), permute_250, out=buf298)
            del permute_250
            buf299 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:338
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf297, buf299, 589824, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf300 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf298, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), mul_102, mul_103, add_61, getitem_20, getitem_21, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_61
            del getitem_20
            del getitem_21
            del mul_102
            del mul_103
            buf301 = buf300[0]
            assert_size_stride(buf301, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf301, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf302 = buf300[1]
            assert_size_stride(buf302, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf302, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf303 = buf300[2]
            assert_size_stride(buf303, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf303, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf300
            buf305 = buf264; del buf264  # reuse
            # Topologically Sorted Source Nodes: [clone_53, k_22], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:339
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_12, rsqrt_23, buf302, buf305, 196608, 128, stream=stream0)
            buf306 = buf263; del buf263  # reuse
            # Topologically Sorted Source Nodes: [clone_55, q_22], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:340
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_11, rsqrt_22, buf301, buf306, 196608, 128, stream=stream0)
            buf307 = buf268; del buf268  # reuse
            buf308 = buf267; del buf267  # reuse
            buf309 = buf266; del buf266  # reuse
            buf310 = buf265; del buf265  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, clone_53, k_22, clone_55, q_22], Original ATen: [aten.slice, aten.neg, aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9:341
            stream0 = get_raw_stream(0)
            triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9.run(buf302, cat_12, rsqrt_23, buf305, primals_2, primals_3, buf301, cat_11, rsqrt_22, buf306, buf307, buf308, buf309, buf310, 196608, 64, stream=stream0)
            del cat_11
            del cat_12
            del rsqrt_22
            del rsqrt_23
            buf330 = reinterpret_tensor(buf280, (32768, 768), (768, 1), 0); del buf280  # reuse
            # Topologically Sorted Source Nodes: [loss, full_6, linear_36, sigmoid_3, gate_3, unsqueeze_2], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10:342
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10.run(buf330, 25165824, stream=stream0)
            buf311 = reinterpret_tensor(buf219, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf219  # reuse
            buf312 = reinterpret_tensor(buf311, (16, 2048, 6), (12288, 6, 1), 0); del buf311  # reuse
            # Topologically Sorted Source Nodes: [loss, full_6, linear_36, sigmoid_3, gate_3, unsqueeze_2, ve_5], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten.squeeze, aten.sigmoid_backward]
            # [Provenance debug handles] triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11:343
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11.run(buf312, buf303, embedding_3, primals_1, mm_36, buf330, 196608, 128, stream=stream0)
            del embedding_3
            del mm_36
            buf313 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [linear_36, sigmoid_3, permute_255], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:537
            extern_kernels.mm(reinterpret_tensor(buf312, (6, 32768), (1, 6), 0), view_98, out=buf313)
            del view_98
            buf314 = buf221; del buf221  # reuse
            # Topologically Sorted Source Nodes: [linear_36, sigmoid_3], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:538
            extern_kernels.mm(reinterpret_tensor(buf312, (32768, 6), (6, 1), 0), permute_254, out=buf314)
            del permute_254
            buf315 = empty_strided_cuda((6, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_12:344
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(buf313, buf315, 72, stream=stream0)
            buf316 = buf297; del buf297  # reuse
            # Topologically Sorted Source Nodes: [permute_259], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:539
            extern_kernels.mm(reinterpret_tensor(buf303, (768, 32768), (1, 768), 0), view_88, out=buf316)
            buf317 = reinterpret_tensor(buf302, (32768, 768), (768, 1), 0); del buf302  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:540
            extern_kernels.mm(reinterpret_tensor(buf303, (32768, 768), (768, 1), 0), permute_258, out=buf317)
            del permute_258
            buf318 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:345
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf316, buf318, 589824, stream=stream0)
            buf319 = buf303; del buf303  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_39, _generalized_scatter_40], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:346
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf307, buf308, buf319, 25165824, stream=stream0)
            buf320 = buf316; del buf316  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_39, _generalized_scatter_40, permute_263], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:541
            extern_kernels.mm(reinterpret_tensor(buf319, (768, 32768), (1, 768), 0), view_88, out=buf320)
            buf321 = reinterpret_tensor(buf301, (32768, 768), (768, 1), 0); del buf301  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_39, _generalized_scatter_40], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:542
            extern_kernels.mm(reinterpret_tensor(buf319, (32768, 768), (768, 1), 0), permute_262, out=buf321)
            del permute_262
            buf322 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:347
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf320, buf322, 589824, stream=stream0)
            buf323 = buf319; del buf319  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_41, _generalized_scatter_42], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:348
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf309, buf310, buf323, 25165824, stream=stream0)
            buf324 = buf320; del buf320  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_41, _generalized_scatter_42, permute_267], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:543
            extern_kernels.mm(reinterpret_tensor(buf323, (768, 32768), (1, 768), 0), view_88, out=buf324)
            del view_88
            buf325 = buf298; del buf298  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_41, _generalized_scatter_42], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:544
            extern_kernels.mm(reinterpret_tensor(buf323, (32768, 768), (768, 1), 0), permute_266, out=buf325)
            del permute_266
            buf326 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:349
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf324, buf326, 589824, stream=stream0)
            buf327 = reinterpret_tensor(buf237, (16, 2048, 768), (1572864, 768, 1), 0); del buf237  # reuse
            buf372 = buf234; del buf234  # reuse
            buf329 = buf296; del buf296  # reuse
            buf337 = reinterpret_tensor(buf323, (32768, 768), (768, 1), 0); del buf323  # reuse
            # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_43, clone_57, getitem_38, mul_66, getitem_39, mul_67, x_34, rms_norm_21, getitem_32, mul_54, getitem_33, mul_55, x_28, rms_norm_17], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select, aten.mul, aten._to_copy]
            # [Provenance debug handles] triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_28:350
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_28.run(buf329, primals_7, add_58, primals_8, cat, rsqrt_21, add_47, rsqrt_17, buf314, buf317, buf321, buf325, buf327, buf372, buf337, 32768, 768, stream=stream0)
            del rsqrt_21
            buf332 = buf325; del buf325  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            # [Provenance debug handles] triton_poi_fused_embedding_dense_backward_15:351
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_15.run(buf330, buf332, 25165824, stream=stream0)
            buf339 = reinterpret_tensor(buf291, (32768, 3072), (3072, 1), 0); del buf291  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:545
            extern_kernels.mm(buf337, permute_270, out=buf339)
            del permute_270
            buf341 = reinterpret_tensor(mm_31, (16, 2048, 3072), (6291456, 3072, 1), 0); del mm_31  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_30, relu_4], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5:352
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5.run(buf341, buf339, 100663296, stream=stream0)
            del buf339
            buf343 = buf321; del buf321  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_30, relu_4], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:546
            extern_kernels.mm(reinterpret_tensor(buf341, (32768, 3072), (3072, 1), 0), permute_274, out=buf343)
            del permute_274
            buf346 = reinterpret_tensor(buf343, (16, 2048, 768), (1572864, 768, 1), 0); del buf343  # reuse
            # Topologically Sorted Source Nodes: [getitem_38, clone_59, rms_norm_20], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29:353
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_29.run(buf346, add_56, rsqrt_20, buf329, primals_7, 32768, 768, stream=stream0)
            del add_56
            del rsqrt_20
            buf348 = buf317; del buf317  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:547
            extern_kernels.mm(reinterpret_tensor(buf346, (32768, 768), (768, 1), 0), permute_278, out=buf348)
            del permute_278
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf350 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf348, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), mul_84, mul_85, view_80, getitem_16, getitem_17, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del getitem_17
            del mul_84
            del mul_85
            del view_80
            buf353 = buf350[2]
            assert_size_stride(buf353, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf353, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf362 = buf348; del buf348  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:548
            extern_kernels.mm(reinterpret_tensor(buf353, (32768, 768), (768, 1), 0), permute_282, out=buf362)
            del permute_282
            buf351 = buf350[0]
            assert_size_stride(buf351, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf351, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf352 = buf350[1]
            assert_size_stride(buf352, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf352, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf350
            buf355 = buf306; del buf306  # reuse
            # Topologically Sorted Source Nodes: [clone_61, k_18], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:354
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_10, rsqrt_19, buf352, buf355, 196608, 128, stream=stream0)
            buf356 = buf305; del buf305  # reuse
            # Topologically Sorted Source Nodes: [clone_63, q_18], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:355
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_9, rsqrt_18, buf351, buf356, 196608, 128, stream=stream0)
            buf357 = buf310; del buf310  # reuse
            buf358 = buf309; del buf309  # reuse
            buf359 = buf308; del buf308  # reuse
            buf360 = buf307; del buf307  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, clone_61, k_18, clone_63, q_18], Original ATen: [aten.slice, aten.neg, aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9:356
            stream0 = get_raw_stream(0)
            triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9.run(buf352, cat_10, rsqrt_19, buf355, primals_2, primals_3, buf351, cat_9, rsqrt_18, buf356, buf357, buf358, buf359, buf360, 196608, 64, stream=stream0)
            del cat_10
            del cat_9
            del rsqrt_18
            del rsqrt_19
            buf364 = buf352; del buf352  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_46, _generalized_scatter_47], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:357
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf357, buf358, buf364, 25165824, stream=stream0)
            buf366 = reinterpret_tensor(buf351, (32768, 768), (768, 1), 0); del buf351  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_46, _generalized_scatter_47], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:549
            extern_kernels.mm(reinterpret_tensor(buf364, (32768, 768), (768, 1), 0), permute_286, out=buf366)
            del permute_286
            buf368 = buf272; del buf272  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_48, _generalized_scatter_49], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:358
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf359, buf360, buf368, 25165824, stream=stream0)
            buf370 = reinterpret_tensor(buf261, (32768, 768), (768, 1), 0); del buf261  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_48, _generalized_scatter_49], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:550
            extern_kernels.mm(reinterpret_tensor(buf368, (32768, 768), (768, 1), 0), permute_290, out=buf370)
            del permute_290
            buf374 = reinterpret_tensor(buf362, (16, 2048, 768), (1572864, 768, 1), 0); del buf362  # reuse
            buf379 = reinterpret_tensor(buf254, (32768, 768), (768, 1), 0); del buf254  # reuse
            buf333 = buf285; del buf285  # reuse
            buf335 = buf283; del buf283  # reuse
            buf375 = buf243; del buf243  # reuse
            buf377 = buf240; del buf240  # reuse
            # Topologically Sorted Source Nodes: [clone_65, getitem_32], Original ATen: [aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_30:359
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_30.run(buf374, buf372, buf366, buf370, buf346, rsqrt_17, primals_7, buf329, cat, add_58, add_47, buf379, buf333, buf335, buf375, buf377, 32768, 768, stream=stream0)
            del add_47
            del add_58
            del buf366
            del buf370
            del rsqrt_17
            buf334 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:360
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf333, buf334, 1, 32768, stream=stream0)
            buf336 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:361
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf335, buf336, 1, 32768, stream=stream0)
            buf338 = reinterpret_tensor(buf292, (768, 3072), (3072, 1), 0); del buf292  # reuse
            # Topologically Sorted Source Nodes: [permute_271], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:551
            extern_kernels.mm(reinterpret_tensor(buf337, (768, 32768), (1, 768), 0), view_86, out=buf338)
            del buf337
            del view_86
            buf340 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:362
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf338, buf340, 2359296, stream=stream0)
            buf342 = reinterpret_tensor(buf338, (3072, 768), (768, 1), 0); del buf338  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_30, relu_4, permute_275], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:552
            extern_kernels.mm(reinterpret_tensor(buf341, (3072, 32768), (1, 3072), 0), view_84, out=buf342)
            del view_84
            buf344 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:363
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf342, buf344, 2359296, stream=stream0)
            buf347 = buf324; del buf324  # reuse
            # Topologically Sorted Source Nodes: [y_13, y_14, permute_279], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:553
            extern_kernels.mm(reinterpret_tensor(buf346, (768, 32768), (1, 768), 0), reinterpret_tensor(getitem_16, (32768, 768), (768, 1), 0), out=buf347)
            del buf346
            del getitem_16
            buf349 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:364
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf347, buf349, 589824, stream=stream0)
            buf361 = buf347; del buf347  # reuse
            # Topologically Sorted Source Nodes: [permute_283], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:554
            extern_kernels.mm(reinterpret_tensor(buf353, (768, 32768), (1, 768), 0), view_72, out=buf361)
            del buf353
            buf363 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:365
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf361, buf363, 589824, stream=stream0)
            buf365 = buf361; del buf361  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_46, _generalized_scatter_47, permute_287], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:555
            extern_kernels.mm(reinterpret_tensor(buf364, (768, 32768), (1, 768), 0), view_72, out=buf365)
            del buf364
            buf367 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:366
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf365, buf367, 589824, stream=stream0)
            buf369 = buf365; del buf365  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_48, _generalized_scatter_49, permute_291], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:556
            extern_kernels.mm(reinterpret_tensor(buf368, (768, 32768), (1, 768), 0), view_72, out=buf369)
            del view_72
            buf371 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:367
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf369, buf371, 589824, stream=stream0)
            buf376 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:368
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf375, buf376, 1, 32768, stream=stream0)
            buf378 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:369
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf377, buf378, 1, 32768, stream=stream0)
            buf380 = reinterpret_tensor(buf342, (768, 3072), (3072, 1), 0); del buf342  # reuse
            # Topologically Sorted Source Nodes: [permute_295], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:557
            extern_kernels.mm(reinterpret_tensor(buf379, (768, 32768), (1, 768), 0), view_70, out=buf380)
            del view_70
            buf381 = reinterpret_tensor(buf341, (32768, 3072), (3072, 1), 0); del buf341  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:558
            extern_kernels.mm(buf379, permute_294, out=buf381)
            del permute_294
            buf382 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:370
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf380, buf382, 2359296, stream=stream0)
            buf383 = reinterpret_tensor(mm_25, (16, 2048, 3072), (6291456, 3072, 1), 0); del mm_25  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_24, relu_3], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5:371
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5.run(buf383, buf381, 100663296, stream=stream0)
            del buf381
            buf384 = reinterpret_tensor(buf380, (3072, 768), (768, 1), 0); del buf380  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_24, relu_3, permute_299], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:559
            extern_kernels.mm(reinterpret_tensor(buf383, (3072, 32768), (1, 3072), 0), view_68, out=buf384)
            del view_68
            buf385 = buf379; del buf379  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_24, relu_3], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:560
            extern_kernels.mm(reinterpret_tensor(buf383, (32768, 3072), (3072, 1), 0), permute_298, out=buf385)
            del permute_298
            buf386 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:372
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf384, buf386, 2359296, stream=stream0)
            buf388 = reinterpret_tensor(buf385, (16, 2048, 768), (1572864, 768, 1), 0); del buf385  # reuse
            # Topologically Sorted Source Nodes: [getitem_32, clone_67, rms_norm_16], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_31:373
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_31.run(buf388, add_45, rsqrt_16, buf374, primals_7, 32768, 768, stream=stream0)
            del add_45
            del rsqrt_16
            buf389 = buf369; del buf369  # reuse
            # Topologically Sorted Source Nodes: [y_10, y_11, permute_303], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:561
            extern_kernels.mm(reinterpret_tensor(buf388, (768, 32768), (1, 768), 0), reinterpret_tensor(getitem_12, (32768, 768), (768, 1), 0), out=buf389)
            buf390 = reinterpret_tensor(buf368, (32768, 768), (768, 1), 0); del buf368  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:562
            extern_kernels.mm(reinterpret_tensor(buf388, (32768, 768), (768, 1), 0), permute_302, out=buf390)
            del permute_302
            buf391 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:374
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf389, buf391, 589824, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf392 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf390, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), mul_68, mul_69, add_38, getitem_12, getitem_13, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_38
            del getitem_12
            del getitem_13
            del mul_68
            del mul_69
            buf393 = buf392[0]
            assert_size_stride(buf393, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf393, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf394 = buf392[1]
            assert_size_stride(buf394, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf394, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf395 = buf392[2]
            assert_size_stride(buf395, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf395, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf392
            buf397 = buf356; del buf356  # reuse
            # Topologically Sorted Source Nodes: [clone_69, k_14], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:375
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_8, rsqrt_15, buf394, buf397, 196608, 128, stream=stream0)
            buf398 = buf355; del buf355  # reuse
            # Topologically Sorted Source Nodes: [clone_71, q_14], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:376
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_7, rsqrt_14, buf393, buf398, 196608, 128, stream=stream0)
            buf399 = buf360; del buf360  # reuse
            buf400 = buf359; del buf359  # reuse
            buf401 = buf358; del buf358  # reuse
            buf402 = buf357; del buf357  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, clone_69, k_14, clone_71, q_14], Original ATen: [aten.slice, aten.neg, aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9:377
            stream0 = get_raw_stream(0)
            triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9.run(buf394, cat_8, rsqrt_15, buf397, primals_2, primals_3, buf393, cat_7, rsqrt_14, buf398, buf399, buf400, buf401, buf402, 196608, 64, stream=stream0)
            del cat_7
            del cat_8
            del rsqrt_14
            del rsqrt_15
            buf422 = reinterpret_tensor(buf372, (32768, 768), (768, 1), 0); del buf372  # reuse
            # Topologically Sorted Source Nodes: [loss, full_6, linear_23, sigmoid_2, gate_2, unsqueeze_1], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10:378
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10.run(buf422, 25165824, stream=stream0)
            buf403 = reinterpret_tensor(buf312, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf312  # reuse
            buf404 = reinterpret_tensor(buf403, (16, 2048, 6), (12288, 6, 1), 0); del buf403  # reuse
            # Topologically Sorted Source Nodes: [loss, full_6, linear_23, sigmoid_2, gate_2, unsqueeze_1, ve_3], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten.squeeze, aten.sigmoid_backward]
            # [Provenance debug handles] triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11:379
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11.run(buf404, buf395, embedding_2, primals_1, mm_23, buf422, 196608, 128, stream=stream0)
            del embedding_2
            del mm_23
            buf405 = buf313; del buf313  # reuse
            # Topologically Sorted Source Nodes: [linear_23, sigmoid_2, permute_307], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:563
            extern_kernels.mm(reinterpret_tensor(buf404, (6, 32768), (1, 6), 0), view_63, out=buf405)
            del view_63
            buf406 = buf314; del buf314  # reuse
            # Topologically Sorted Source Nodes: [linear_23, sigmoid_2], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:564
            extern_kernels.mm(reinterpret_tensor(buf404, (32768, 6), (6, 1), 0), permute_306, out=buf406)
            del permute_306
            buf407 = empty_strided_cuda((6, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_12:380
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(buf405, buf407, 72, stream=stream0)
            buf408 = buf389; del buf389  # reuse
            # Topologically Sorted Source Nodes: [permute_311], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:565
            extern_kernels.mm(reinterpret_tensor(buf395, (768, 32768), (1, 768), 0), view_53, out=buf408)
            buf409 = reinterpret_tensor(buf394, (32768, 768), (768, 1), 0); del buf394  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:566
            extern_kernels.mm(reinterpret_tensor(buf395, (32768, 768), (768, 1), 0), permute_310, out=buf409)
            del permute_310
            buf410 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:381
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf408, buf410, 589824, stream=stream0)
            buf411 = buf395; del buf395  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_52, _generalized_scatter_53], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:382
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf399, buf400, buf411, 25165824, stream=stream0)
            buf412 = buf408; del buf408  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_52, _generalized_scatter_53, permute_315], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:567
            extern_kernels.mm(reinterpret_tensor(buf411, (768, 32768), (1, 768), 0), view_53, out=buf412)
            buf413 = reinterpret_tensor(buf393, (32768, 768), (768, 1), 0); del buf393  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_52, _generalized_scatter_53], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:568
            extern_kernels.mm(reinterpret_tensor(buf411, (32768, 768), (768, 1), 0), permute_314, out=buf413)
            del permute_314
            buf414 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:383
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf412, buf414, 589824, stream=stream0)
            buf415 = buf411; del buf411  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_54, _generalized_scatter_55], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:384
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf401, buf402, buf415, 25165824, stream=stream0)
            buf416 = buf412; del buf412  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_54, _generalized_scatter_55, permute_319], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:569
            extern_kernels.mm(reinterpret_tensor(buf415, (768, 32768), (1, 768), 0), view_53, out=buf416)
            del view_53
            buf417 = buf390; del buf390  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_54, _generalized_scatter_55], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:570
            extern_kernels.mm(reinterpret_tensor(buf415, (32768, 768), (768, 1), 0), permute_318, out=buf417)
            del permute_318
            buf418 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:385
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf416, buf418, 589824, stream=stream0)
            buf419 = reinterpret_tensor(buf330, (16, 2048, 768), (1572864, 768, 1), 0); del buf330  # reuse
            buf465 = buf327; del buf327  # reuse
            buf421 = buf388; del buf388  # reuse
            buf427 = buf242; del buf242  # reuse
            buf430 = reinterpret_tensor(buf415, (32768, 768), (768, 1), 0); del buf415  # reuse
            # Topologically Sorted Source Nodes: [full_5, getitem_46, getitem_39, getitem_33, _generalized_scatter_56, clone_73, getitem_25, mul_40, getitem_26, mul_41, x_22, rms_norm_13, getitem_19, mul_28, getitem_20, mul_29, x_16, rms_norm_9], Original ATen: [aten.slice_backward, aten.select, aten.mul, aten.add, aten.view, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_32:386
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_32.run(buf421, buf427, primals_7, add_35, primals_8, cat, rsqrt_13, add_24, rsqrt_9, buf406, buf409, buf413, buf417, buf282, buf329, buf374, buf419, buf465, buf430, 32768, 768, stream=stream0)
            del buf419
            del rsqrt_13
            buf424 = buf417; del buf417  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            # [Provenance debug handles] triton_poi_fused_embedding_dense_backward_15:387
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_15.run(buf422, buf424, 25165824, stream=stream0)
            buf432 = reinterpret_tensor(buf383, (32768, 3072), (3072, 1), 0); del buf383  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:571
            extern_kernels.mm(buf430, permute_322, out=buf432)
            del permute_322
            buf434 = reinterpret_tensor(mm_18, (16, 2048, 3072), (6291456, 3072, 1), 0); del mm_18  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_18, relu_2], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5:388
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5.run(buf434, buf432, 100663296, stream=stream0)
            del buf432
            buf436 = buf413; del buf413  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_18, relu_2], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:572
            extern_kernels.mm(reinterpret_tensor(buf434, (32768, 3072), (3072, 1), 0), permute_326, out=buf436)
            del permute_326
            buf439 = reinterpret_tensor(buf436, (16, 2048, 768), (1572864, 768, 1), 0); del buf436  # reuse
            # Topologically Sorted Source Nodes: [getitem_25, clone_75, rms_norm_12], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_33:389
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_33.run(buf439, add_33, rsqrt_12, buf421, primals_7, 32768, 768, stream=stream0)
            del add_33
            del rsqrt_12
            buf441 = buf409; del buf409  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:573
            extern_kernels.mm(reinterpret_tensor(buf439, (32768, 768), (768, 1), 0), permute_330, out=buf441)
            del permute_330
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf443 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf441, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), mul_50, mul_51, view_45, getitem_8, getitem_9, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del getitem_9
            del mul_50
            del mul_51
            del view_45
            buf446 = buf443[2]
            assert_size_stride(buf446, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf446, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf455 = buf441; del buf441  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:574
            extern_kernels.mm(reinterpret_tensor(buf446, (32768, 768), (768, 1), 0), permute_334, out=buf455)
            del permute_334
            buf444 = buf443[0]
            assert_size_stride(buf444, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf444, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf445 = buf443[1]
            assert_size_stride(buf445, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf445, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf443
            buf448 = buf398; del buf398  # reuse
            # Topologically Sorted Source Nodes: [clone_77, k_10], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:390
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_6, rsqrt_11, buf445, buf448, 196608, 128, stream=stream0)
            buf449 = buf397; del buf397  # reuse
            # Topologically Sorted Source Nodes: [clone_79, q_10], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:391
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_5, rsqrt_10, buf444, buf449, 196608, 128, stream=stream0)
            buf450 = buf402; del buf402  # reuse
            buf451 = buf401; del buf401  # reuse
            buf452 = buf400; del buf400  # reuse
            buf453 = buf399; del buf399  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, clone_77, k_10, clone_79, q_10], Original ATen: [aten.slice, aten.neg, aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9:392
            stream0 = get_raw_stream(0)
            triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9.run(buf445, cat_6, rsqrt_11, buf448, primals_2, primals_3, buf444, cat_5, rsqrt_10, buf449, buf450, buf451, buf452, buf453, 196608, 64, stream=stream0)
            del cat_5
            del cat_6
            del rsqrt_10
            del rsqrt_11
            buf457 = buf445; del buf445  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_59, _generalized_scatter_60], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:393
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf450, buf451, buf457, 25165824, stream=stream0)
            buf459 = reinterpret_tensor(buf444, (32768, 768), (768, 1), 0); del buf444  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_59, _generalized_scatter_60], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:575
            extern_kernels.mm(reinterpret_tensor(buf457, (32768, 768), (768, 1), 0), permute_338, out=buf459)
            del permute_338
            buf461 = reinterpret_tensor(buf374, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0); del buf374  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_61, _generalized_scatter_62], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:394
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf452, buf453, buf461, 25165824, stream=stream0)
            buf463 = reinterpret_tensor(buf329, (32768, 768), (768, 1), 0); del buf329  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_61, _generalized_scatter_62], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:576
            extern_kernels.mm(reinterpret_tensor(buf461, (32768, 768), (768, 1), 0), permute_342, out=buf463)
            del permute_342
            buf467 = reinterpret_tensor(buf455, (16, 2048, 768), (1572864, 768, 1), 0); del buf455  # reuse
            buf472 = reinterpret_tensor(buf282, (32768, 768), (768, 1), 0); del buf282  # reuse
            buf425 = buf377; del buf377  # reuse
            buf428 = buf375; del buf375  # reuse
            buf468 = buf335; del buf335  # reuse
            buf470 = buf333; del buf333  # reuse
            # Topologically Sorted Source Nodes: [clone_81, getitem_19], Original ATen: [aten.mul, aten.sum, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_34:395
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward_add_mul_select_sum_view_34.run(buf467, buf465, buf459, buf463, buf439, rsqrt_9, primals_7, buf421, cat, add_35, add_24, buf472, buf425, buf428, buf468, buf470, 32768, 768, stream=stream0)
            del add_24
            del add_35
            del buf421
            del buf459
            del buf463
            del rsqrt_9
            buf426 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:396
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf425, buf426, 1, 32768, stream=stream0)
            del buf425
            buf429 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:397
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf428, buf429, 1, 32768, stream=stream0)
            buf431 = reinterpret_tensor(buf384, (768, 3072), (3072, 1), 0); del buf384  # reuse
            # Topologically Sorted Source Nodes: [permute_323], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:577
            extern_kernels.mm(reinterpret_tensor(buf430, (768, 32768), (1, 768), 0), view_51, out=buf431)
            del buf430
            del view_51
            buf433 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:398
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf431, buf433, 2359296, stream=stream0)
            buf435 = reinterpret_tensor(buf431, (3072, 768), (768, 1), 0); del buf431  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_18, relu_2, permute_327], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:578
            extern_kernels.mm(reinterpret_tensor(buf434, (3072, 32768), (1, 3072), 0), view_49, out=buf435)
            del view_49
            buf437 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:399
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf435, buf437, 2359296, stream=stream0)
            buf440 = buf416; del buf416  # reuse
            # Topologically Sorted Source Nodes: [y_7, y_8, permute_331], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:579
            extern_kernels.mm(reinterpret_tensor(buf439, (768, 32768), (1, 768), 0), reinterpret_tensor(getitem_8, (32768, 768), (768, 1), 0), out=buf440)
            del buf439
            del getitem_8
            buf442 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:400
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf440, buf442, 589824, stream=stream0)
            buf454 = buf440; del buf440  # reuse
            # Topologically Sorted Source Nodes: [permute_335], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:580
            extern_kernels.mm(reinterpret_tensor(buf446, (768, 32768), (1, 768), 0), view_37, out=buf454)
            buf456 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:401
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf454, buf456, 589824, stream=stream0)
            buf458 = buf454; del buf454  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_59, _generalized_scatter_60, permute_339], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:581
            extern_kernels.mm(reinterpret_tensor(buf457, (768, 32768), (1, 768), 0), view_37, out=buf458)
            buf460 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:402
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf458, buf460, 589824, stream=stream0)
            buf462 = buf458; del buf458  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_61, _generalized_scatter_62, permute_343], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:582
            extern_kernels.mm(reinterpret_tensor(buf461, (768, 32768), (1, 768), 0), view_37, out=buf462)
            del view_37
            buf464 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:403
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf462, buf464, 589824, stream=stream0)
            buf469 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:404
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf468, buf469, 1, 32768, stream=stream0)
            buf471 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:405
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf470, buf471, 1, 32768, stream=stream0)
            buf473 = reinterpret_tensor(buf435, (768, 3072), (3072, 1), 0); del buf435  # reuse
            # Topologically Sorted Source Nodes: [permute_347], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:583
            extern_kernels.mm(reinterpret_tensor(buf472, (768, 32768), (1, 768), 0), view_35, out=buf473)
            del view_35
            buf474 = reinterpret_tensor(buf434, (32768, 3072), (3072, 1), 0); del buf434  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:584
            extern_kernels.mm(buf472, permute_346, out=buf474)
            del permute_346
            buf475 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:406
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf473, buf475, 2359296, stream=stream0)
            buf476 = reinterpret_tensor(mm_12, (16, 2048, 3072), (6291456, 3072, 1), 0); del mm_12  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_12, relu_1], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5:407
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5.run(buf476, buf474, 100663296, stream=stream0)
            del buf474
            buf477 = reinterpret_tensor(buf473, (3072, 768), (768, 1), 0); del buf473  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_12, relu_1, permute_351], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:585
            extern_kernels.mm(reinterpret_tensor(buf476, (3072, 32768), (1, 3072), 0), view_33, out=buf477)
            del view_33
            buf478 = buf472; del buf472  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_12, relu_1], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:586
            extern_kernels.mm(reinterpret_tensor(buf476, (32768, 3072), (3072, 1), 0), permute_350, out=buf478)
            del permute_350
            buf479 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:408
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf477, buf479, 2359296, stream=stream0)
            buf481 = reinterpret_tensor(buf478, (16, 2048, 768), (1572864, 768, 1), 0); del buf478  # reuse
            # Topologically Sorted Source Nodes: [getitem_19, clone_83, rms_norm_8], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_35:409
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_35.run(buf481, add_22, rsqrt_8, buf467, primals_7, 32768, 768, stream=stream0)
            del add_22
            del rsqrt_8
            buf482 = buf462; del buf462  # reuse
            # Topologically Sorted Source Nodes: [y_4, y_5, permute_355], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:587
            extern_kernels.mm(reinterpret_tensor(buf481, (768, 32768), (1, 768), 0), reinterpret_tensor(getitem_4, (32768, 768), (768, 1), 0), out=buf482)
            buf483 = reinterpret_tensor(buf461, (32768, 768), (768, 1), 0); del buf461  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:588
            extern_kernels.mm(reinterpret_tensor(buf481, (32768, 768), (768, 1), 0), permute_354, out=buf483)
            del permute_354
            buf484 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:410
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf482, buf484, 589824, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf485 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf483, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), mul_34, mul_35, add_15, getitem_4, getitem_5, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del add_15
            del getitem_4
            del getitem_5
            del mul_34
            del mul_35
            buf486 = buf485[0]
            assert_size_stride(buf486, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf486, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf487 = buf485[1]
            assert_size_stride(buf487, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf487, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf488 = buf485[2]
            assert_size_stride(buf488, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf488, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf485
            buf490 = buf449; del buf449  # reuse
            # Topologically Sorted Source Nodes: [clone_85, k_6], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:411
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_4, rsqrt_7, buf487, buf490, 196608, 128, stream=stream0)
            buf491 = buf448; del buf448  # reuse
            # Topologically Sorted Source Nodes: [clone_87, q_6], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:412
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_3, rsqrt_6, buf486, buf491, 196608, 128, stream=stream0)
            buf492 = buf453; del buf453  # reuse
            buf493 = buf452; del buf452  # reuse
            buf494 = buf451; del buf451  # reuse
            buf495 = buf450; del buf450  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, clone_85, k_6, clone_87, q_6], Original ATen: [aten.slice, aten.neg, aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9:413
            stream0 = get_raw_stream(0)
            triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9.run(buf487, cat_4, rsqrt_7, buf490, primals_2, primals_3, buf486, cat_3, rsqrt_6, buf491, buf492, buf493, buf494, buf495, 196608, 64, stream=stream0)
            del cat_3
            del cat_4
            del rsqrt_6
            del rsqrt_7
            buf515 = reinterpret_tensor(buf465, (32768, 768), (768, 1), 0); del buf465  # reuse
            # Topologically Sorted Source Nodes: [loss, full_6, linear_10, sigmoid_1, gate_1, unsqueeze], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10:414
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10.run(buf515, 25165824, stream=stream0)
            buf496 = reinterpret_tensor(buf404, (16, 2048, 6, 1), (12288, 6, 1, 1), 0); del buf404  # reuse
            buf497 = reinterpret_tensor(buf496, (16, 2048, 6), (12288, 6, 1), 0); del buf496  # reuse
            # Topologically Sorted Source Nodes: [loss, full_6, linear_10, sigmoid_1, gate_1, unsqueeze, ve_1], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward, aten._unsafe_view, aten.sigmoid, aten.mul, aten.unsqueeze, aten.view, aten.sum, aten.squeeze, aten.sigmoid_backward]
            # [Provenance debug handles] triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11:415
            stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_sigmoid_backward_squeeze_sum_unsqueeze_view_11.run(buf497, buf488, embedding_1, primals_1, mm_10, buf515, 196608, 128, stream=stream0)
            del embedding_1
            del mm_10
            buf498 = buf405; del buf405  # reuse
            # Topologically Sorted Source Nodes: [linear_10, sigmoid_1, permute_359], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:589
            extern_kernels.mm(reinterpret_tensor(buf497, (6, 32768), (1, 6), 0), view_28, out=buf498)
            del view_28
            buf499 = buf406; del buf406  # reuse
            # Topologically Sorted Source Nodes: [linear_10, sigmoid_1], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.squeeze, aten.mul, aten.sigmoid_backward, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:590
            extern_kernels.mm(reinterpret_tensor(buf497, (32768, 6), (6, 1), 0), permute_358, out=buf499)
            del buf497
            del permute_358
            buf500 = empty_strided_cuda((6, 12), (12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_12:416
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_12.run(buf498, buf500, 72, stream=stream0)
            del buf498
            buf501 = buf482; del buf482  # reuse
            # Topologically Sorted Source Nodes: [permute_363], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:591
            extern_kernels.mm(reinterpret_tensor(buf488, (768, 32768), (1, 768), 0), view_18, out=buf501)
            buf502 = reinterpret_tensor(buf487, (32768, 768), (768, 1), 0); del buf487  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:592
            extern_kernels.mm(reinterpret_tensor(buf488, (32768, 768), (768, 1), 0), permute_362, out=buf502)
            del permute_362
            buf503 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:417
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf501, buf503, 589824, stream=stream0)
            buf504 = buf488; del buf488  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_65, _generalized_scatter_66], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:418
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf492, buf493, buf504, 25165824, stream=stream0)
            buf505 = buf501; del buf501  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_65, _generalized_scatter_66, permute_367], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:593
            extern_kernels.mm(reinterpret_tensor(buf504, (768, 32768), (1, 768), 0), view_18, out=buf505)
            buf506 = reinterpret_tensor(buf486, (32768, 768), (768, 1), 0); del buf486  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_65, _generalized_scatter_66], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:594
            extern_kernels.mm(reinterpret_tensor(buf504, (32768, 768), (768, 1), 0), permute_366, out=buf506)
            del permute_366
            buf507 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:419
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf505, buf507, 589824, stream=stream0)
            buf508 = buf504; del buf504  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_67, _generalized_scatter_68], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:420
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf494, buf495, buf508, 25165824, stream=stream0)
            buf509 = buf505; del buf505  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_67, _generalized_scatter_68, permute_371], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:595
            extern_kernels.mm(reinterpret_tensor(buf508, (768, 32768), (1, 768), 0), view_18, out=buf509)
            del view_18
            buf510 = buf483; del buf483  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_67, _generalized_scatter_68], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:596
            extern_kernels.mm(reinterpret_tensor(buf508, (32768, 768), (768, 1), 0), permute_370, out=buf510)
            del permute_370
            buf511 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:421
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf509, buf511, 589824, stream=stream0)
            buf512 = reinterpret_tensor(buf422, (16, 2048, 768), (1572864, 768, 1), 0); del buf422  # reuse
            buf514 = buf481; del buf481  # reuse
            buf522 = reinterpret_tensor(buf508, (32768, 768), (768, 1), 0); del buf508  # reuse
            # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_69, clone_89, getitem_12, mul_14, getitem_13, mul_15, x_10, rms_norm_5], Original ATen: [aten.slice_backward, aten.view, aten.add, aten._fused_rms_norm_backward, aten.select, aten.mul, aten._to_copy]
            # [Provenance debug handles] triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_36:422
            stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_select_slice_backward_view_36.run(buf514, primals_7, add_12, primals_8, cat, rsqrt_5, buf499, buf502, buf506, buf510, buf512, buf522, 32768, 768, stream=stream0)
            del buf499
            del buf512
            del rsqrt_5
            buf517 = buf510; del buf510  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            # [Provenance debug handles] triton_poi_fused_embedding_dense_backward_15:423
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_15.run(buf515, buf517, 25165824, stream=stream0)
            buf524 = reinterpret_tensor(buf476, (32768, 3072), (3072, 1), 0); del buf476  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:597
            extern_kernels.mm(buf522, permute_374, out=buf524)
            del permute_374
            buf526 = reinterpret_tensor(mm_5, (16, 2048, 3072), (6291456, 3072, 1), 0); del mm_5  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_6, relu], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5:424
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_pow_relu_threshold_backward_view_5.run(buf526, buf524, 100663296, stream=stream0)
            del buf524
            buf528 = buf506; del buf506  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_6, relu], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:598
            extern_kernels.mm(reinterpret_tensor(buf526, (32768, 3072), (3072, 1), 0), permute_378, out=buf528)
            del permute_378
            buf531 = reinterpret_tensor(buf528, (16, 2048, 768), (1572864, 768, 1), 0); del buf528  # reuse
            # Topologically Sorted Source Nodes: [getitem_12, clone_91, rms_norm_4], Original ATen: [aten.select, aten.mul, aten.view, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_37:425
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_view_37.run(buf531, add_10, rsqrt_4, buf514, primals_7, 32768, 768, stream=stream0)
            del add_10
            del rsqrt_4
            buf533 = buf502; del buf502  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:599
            extern_kernels.mm(reinterpret_tensor(buf531, (32768, 768), (768, 1), 0), permute_382, out=buf533)
            del permute_382
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, flash_attn_3._flash_attn_backward]
            buf535 = torch.ops.flash_attn_3._flash_attn_backward.default(reinterpret_tensor(buf533, (16, 2048, 6, 128), (1572864, 768, 128, 1), 0), mul_16, mul_17, view_10, getitem, getitem_1, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0, softcap=0.0, deterministic=False, sm_margin=0)
            del getitem_1
            del mul_16
            del mul_17
            del view_10
            buf538 = buf535[2]
            assert_size_stride(buf538, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf538, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf547 = buf533; del buf533  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:600
            extern_kernels.mm(reinterpret_tensor(buf538, (32768, 768), (768, 1), 0), permute_386, out=buf547)
            del permute_386
            buf536 = buf535[0]
            assert_size_stride(buf536, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf536, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            buf537 = buf535[1]
            assert_size_stride(buf537, (16, 2048, 6, 128), (1572864, 768, 128, 1), 'torch.ops.flash_attn_3._flash_attn_backward.default')
            assert_alignment(buf537, 16, 'torch.ops.flash_attn_3._flash_attn_backward.default')
            del buf535
            buf540 = buf491; del buf491  # reuse
            # Topologically Sorted Source Nodes: [clone_93, k_2], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:426
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_2, rsqrt_3, buf537, buf540, 196608, 128, stream=stream0)
            buf541 = buf490; del buf490  # reuse
            # Topologically Sorted Source Nodes: [clone_95, q_2], Original ATen: [aten.mul, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_mul_8:427
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_mul_8.run(cat_1, rsqrt_2, buf536, buf541, 196608, 128, stream=stream0)
            buf542 = buf495; del buf495  # reuse
            buf543 = buf494; del buf494  # reuse
            buf544 = buf493; del buf493  # reuse
            buf545 = buf492; del buf492  # reuse
            # Topologically Sorted Source Nodes: [cos, sin, neg, clone_93, k_2, clone_95, q_2], Original ATen: [aten.slice, aten.neg, aten.mul, aten._fused_rms_norm_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9:428
            stream0 = get_raw_stream(0)
            triton_poi_fused__fused_rms_norm_backward__to_copy_add_mul_neg_slice_9.run(buf537, cat_2, rsqrt_3, buf540, primals_2, primals_3, buf536, cat_1, rsqrt_2, buf541, buf542, buf543, buf544, buf545, 196608, 64, stream=stream0)
            del buf540
            del buf541
            del cat_1
            del cat_2
            del primals_2
            del primals_3
            del rsqrt_2
            del rsqrt_3
            buf549 = buf537; del buf537  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_72, _generalized_scatter_73], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:429
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf542, buf543, buf549, 25165824, stream=stream0)
            del buf542
            del buf543
            buf551 = reinterpret_tensor(buf536, (32768, 768), (768, 1), 0); del buf536  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_72, _generalized_scatter_73], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:601
            extern_kernels.mm(reinterpret_tensor(buf549, (32768, 768), (768, 1), 0), permute_390, out=buf551)
            del permute_390
            buf553 = buf457; del buf457  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_74, _generalized_scatter_75], Original ATen: [aten.slice_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_slice_backward_13:430
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_backward_13.run(buf544, buf545, buf553, 25165824, stream=stream0)
            del buf544
            del buf545
            buf555 = reinterpret_tensor(buf446, (32768, 768), (768, 1), 0); del buf446  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_74, _generalized_scatter_75], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:602
            extern_kernels.mm(reinterpret_tensor(buf553, (32768, 768), (768, 1), 0), permute_394, out=buf555)
            del permute_394
            buf518 = buf470; del buf470  # reuse
            buf520 = buf468; del buf468  # reuse
            buf558 = reinterpret_tensor(buf547, (16, 2048, 768), (1572864, 768, 1), 0); del buf547  # reuse
            buf562 = buf427; del buf427  # reuse
            buf559 = buf428; del buf428  # reuse
            # Topologically Sorted Source Nodes: [getitem_20, getitem_13, clone_97, getitem_6, mul_2, getitem_7, mul_3, x_4, rms_norm_1], Original ATen: [aten.select, aten.mul, aten.add, aten.sum, aten.view, aten._fused_rms_norm_backward, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_38:431
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy_add_mul_select_sum_view_38.run(buf558, buf562, buf514, cat, add_12, primals_7, primals_8, rsqrt_1, buf551, buf555, buf531, buf467, buf518, buf520, buf559, 32768, 768, stream=stream0)
            del add_12
            del buf467
            del buf514
            del buf551
            del buf555
            del buf558
            del cat
            del primals_7
            del primals_8
            del rsqrt_1
            buf519 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:432
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf518, buf519, 1, 32768, stream=stream0)
            del buf518
            buf521 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:433
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf520, buf521, 1, 32768, stream=stream0)
            del buf520
            buf523 = reinterpret_tensor(buf477, (768, 3072), (3072, 1), 0); del buf477  # reuse
            # Topologically Sorted Source Nodes: [permute_375], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:603
            extern_kernels.mm(reinterpret_tensor(buf522, (768, 32768), (1, 768), 0), view_16, out=buf523)
            del buf522
            del view_16
            buf525 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:434
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf523, buf525, 2359296, stream=stream0)
            buf527 = reinterpret_tensor(buf523, (3072, 768), (768, 1), 0); del buf523  # reuse
            # Topologically Sorted Source Nodes: [scalar_tensor_4, x_6, relu, permute_379], Original ATen: [aten.threshold_backward, aten.view, aten._unsafe_view, aten.relu, aten.pow, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:604
            extern_kernels.mm(reinterpret_tensor(buf526, (3072, 32768), (1, 3072), 0), view_14, out=buf527)
            del buf526
            del view_14
            buf529 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_4:435
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(buf527, buf529, 2359296, stream=stream0)
            del buf527
            buf532 = buf509; del buf509  # reuse
            # Topologically Sorted Source Nodes: [y_1, y_2, permute_383], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:605
            extern_kernels.mm(reinterpret_tensor(buf531, (768, 32768), (1, 768), 0), reinterpret_tensor(getitem, (32768, 768), (768, 1), 0), out=buf532)
            del buf531
            del getitem
            buf534 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:436
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf532, buf534, 589824, stream=stream0)
            buf546 = buf532; del buf532  # reuse
            # Topologically Sorted Source Nodes: [permute_387], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:606
            extern_kernels.mm(reinterpret_tensor(buf538, (768, 32768), (1, 768), 0), view_2, out=buf546)
            del buf538
            buf548 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:437
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf546, buf548, 589824, stream=stream0)
            buf550 = buf546; del buf546  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_72, _generalized_scatter_73, permute_391], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:607
            extern_kernels.mm(reinterpret_tensor(buf549, (768, 32768), (1, 768), 0), view_2, out=buf550)
            del buf549
            buf552 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:438
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf550, buf552, 589824, stream=stream0)
            buf554 = buf550; del buf550  # reuse
            # Topologically Sorted Source Nodes: [full_1, _generalized_scatter_74, _generalized_scatter_75, permute_395], Original ATen: [aten.slice_backward, aten.add, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:608
            extern_kernels.mm(reinterpret_tensor(buf553, (768, 32768), (1, 768), 0), view_2, out=buf554)
            del buf553
            del view_2
            buf556 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:439
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf554, buf556, 589824, stream=stream0)
            del buf554
            buf560 = empty_strided_cuda((), (), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_18:440
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_18.run(buf559, buf560, 1, 32768, stream=stream0)
            del buf559
            buf561 = empty_strided_cuda((12, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [full_7, _generalized_scatter_5, _generalized_scatter_11, _generalized_scatter_18, _generalized_scatter_24, _generalized_scatter_31, _generalized_scatter_37, _generalized_scatter_44, _generalized_scatter_50, _generalized_scatter_57, _generalized_scatter_63, _generalized_scatter_70, _generalized_scatter_76], Original ATen: [aten._to_copy, aten.select_backward, aten.add]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_select_backward_39:441
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_select_backward_39.run(buf57, buf99, buf149, buf191, buf241, buf284, buf334, buf376, buf426, buf469, buf519, buf560, buf561, 12, stream=stream0)
            del buf149
            del buf191
            del buf241
            del buf284
            del buf334
            del buf376
            del buf426
            del buf469
            del buf519
            del buf57
            del buf99
            buf563 = empty_strided_cuda((12, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [full_7, _generalized_scatter_6, _generalized_scatter_12, _generalized_scatter_19, _generalized_scatter_25, _generalized_scatter_32, _generalized_scatter_38, _generalized_scatter_45, _generalized_scatter_51, _generalized_scatter_58, _generalized_scatter_64, _generalized_scatter_71, _generalized_scatter_76], Original ATen: [aten.select_backward, aten._to_copy, aten.add]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_select_backward_39:442
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_select_backward_39.run(buf59, buf101, buf151, buf193, buf244, buf286, buf336, buf378, buf429, buf471, buf521, buf560, buf563, 12, stream=stream0)
            del buf101
            del buf151
            del buf193
            del buf244
            del buf286
            del buf336
            del buf378
            del buf429
            del buf471
            del buf521
            del buf560
            del buf59
            buf564 = empty_strided_cuda((16, 2047, 1), (2048, 1, 32768), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_2, getitem_5], Original ATen: [aten.slice, aten._to_copy, aten.mul, aten.sum]
            # [Provenance debug handles] triton_red_fused__to_copy_mul_slice_sum_40:443
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_slice_sum_40.run(buf562, embedding, rsqrt, buf564, 32752, 768, stream=stream0)
            buf566 = empty_strided_cuda((1, 1, 1, 4), (4, 4, 4, 1), torch.float32)
            buf568 = empty_strided_cuda((16, 2047, 1), (2048, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_1, linear, sigmoid], Original ATen: [aten._to_copy, aten._unsafe_view, aten.sigmoid, aten.mul, aten.sum, aten.sigmoid_backward]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_sum_41:444
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_sum_41.run(buf564, mm, primals_5, buf566, buf568, 4, 8188, stream=stream0)
            del buf564
            buf574 = empty_strided_cuda((1, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [linear, sigmoid], Original ATen: [aten._unsafe_view, aten.sigmoid, aten.mul, aten.sum, aten.view, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sum_view_42:445
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_mul_sigmoid_sum_view_42.run(buf566, buf574, 1, 4, stream=stream0)
            del buf566
            buf569 = empty_strided_cuda((32752, 1), (1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to_1, linear, sigmoid], Original ATen: [aten._to_copy, aten._unsafe_view, aten.sigmoid, aten.mul, aten.sigmoid_backward, aten.view]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_view_43:446
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_mul_sigmoid_sigmoid_backward_view_43.run(buf568, buf569, 32752, stream=stream0)
            del buf568
            buf570 = empty_strided_cuda((1, 24), (24, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_399], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:609
            extern_kernels.mm(reinterpret_tensor(buf569, (1, 32752), (1, 1), 0), view, out=buf570)
            del view
            buf571 = empty_strided_cuda((32752, 24), (24, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:610
            extern_kernels.mm(buf569, permute_398, out=buf571)
            del buf569
            del permute_398
            buf572 = empty_strided_cuda((1, 24), (24, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_44:447
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_44.run(buf570, buf572, 24, stream=stream0)
            del buf570
            buf576 = buf515; del buf515  # reuse
            # Topologically Sorted Source Nodes: [full_6], Original ATen: [aten.embedding_dense_backward]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10:448
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_embedding_dense_backward_mul_nll_loss_forward_sigmoid_unsqueeze_view_10.run(buf576, 25165824, stream=stream0)
            # Topologically Sorted Source Nodes: [loss, full_5, full_6, to_1, linear, sigmoid, gate, x_2, _generalized_scatter_77, _generalized_scatter_78, _generalized_scatter_79, full_88, _generalized_scatter_80, _generalized_scatter_81, clone_99, convert_element_type_default], Original ATen: [aten.nll_loss_forward, aten.slice_backward, aten.embedding_dense_backward, aten.slice, aten._to_copy, aten._unsafe_view, aten.sigmoid, aten.mul, aten.add, aten.view, aten._fused_rms_norm_backward]
            # [Provenance debug handles] triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_sigmoid_slice_slice_backward_view_45:449
            stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_backward__to_copy__unsafe_view_add_embedding_dense_backward_mul_nll_loss_forward_sigmoid_slice_slice_backward_view_45.run(buf562, primals_5, mm, buf571, embedding, rsqrt, primals_1, buf576, 32768, 768, stream=stream0)
            del buf571
            del embedding
            del mm
            del primals_1
            del primals_5
            del rsqrt
            buf578 = reinterpret_tensor(buf562, (32768, 768), (768, 1), 0); del buf562  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
            # [Provenance debug handles] triton_poi_fused_embedding_dense_backward_15:450
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_15.run(buf576, buf578, 25165824, stream=stream0)
            del buf576
        return (None, None, None, buf578, buf574, buf572, buf563, buf561, buf556, buf552, buf548, buf534, buf529, buf525, buf517, buf511, buf507, buf503, buf500, buf484, buf479, buf475, buf464, buf460, buf456, buf442, buf437, buf433, buf424, buf418, buf414, buf410, buf407, buf391, buf386, buf382, buf371, buf367, buf363, buf349, buf344, buf340, buf332, buf326, buf322, buf318, buf315, buf299, buf294, buf290, buf279, buf275, buf271, buf257, buf252, buf248, buf239, buf233, buf229, buf225, buf222, buf206, buf201, buf197, buf186, buf182, buf178, buf164, buf159, buf155, buf147, buf141, buf137, buf133, buf130, buf114, buf109, buf105, buf94, buf90, buf86, buf72, buf67, buf63, buf55, buf49, buf45, buf41, buf38, buf22, buf17, buf13, buf9, buf5, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((1, 20480, 1, 64), (1310720, 64, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((1, 20480, 1, 64), (1310720, 64, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((16, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    embedding = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((32752, 24), (24, 1), device='cuda:0', dtype=torch.bfloat16)
    mm = rand_strided((32752, 1), (1, 1), device='cuda:0', dtype=torch.bfloat16)
    cat = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_1 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_10 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_1 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_2 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_2 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_17 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((16, 6, 2048), (12288, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_10 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_4 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_5 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_16 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_12 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_1 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_5 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_28 = rand_strided((32768, 12), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_10 = rand_strided((32768, 6), (6, 1), device='cuda:0', dtype=torch.bfloat16)
    add_15 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_3 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_4 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_6 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_7 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_34 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_35 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_4 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_5 = rand_strided((16, 6, 2048), (12288, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_22 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_8 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_33 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_12 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_35 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_24 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_9 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_45 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_5 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_6 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_10 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_11 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_50 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_51 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_8 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_9 = rand_strided((16, 6, 2048), (12288, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_33 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_12 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_18 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_51 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_35 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_2 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_13 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_53 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_63 = rand_strided((32768, 12), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_23 = rand_strided((32768, 6), (6, 1), device='cuda:0', dtype=torch.bfloat16)
    add_38 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_7 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_8 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_14 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_15 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_68 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_69 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_12 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_13 = rand_strided((16, 6, 2048), (12288, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_45 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_16 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_25 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_70 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_47 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_17 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_80 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_9 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_10 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_18 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_19 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_84 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_85 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_16 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_17 = rand_strided((16, 6, 2048), (12288, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_56 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_20 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_31 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_86 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_58 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_3 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_21 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_98 = rand_strided((32768, 12), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_36 = rand_strided((32768, 6), (6, 1), device='cuda:0', dtype=torch.bfloat16)
    add_61 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_11 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_12 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_22 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_23 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_102 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_103 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_20 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_21 = rand_strided((16, 6, 2048), (12288, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_68 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_24 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_38 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_105 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_70 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_25 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_115 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_13 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_14 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_26 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_27 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_118 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_119 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_24 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_25 = rand_strided((16, 6, 2048), (12288, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_79 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_28 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_44 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_121 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_81 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_4 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_29 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_123 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_133 = rand_strided((32768, 12), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_49 = rand_strided((32768, 6), (6, 1), device='cuda:0', dtype=torch.bfloat16)
    add_84 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_15 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_16 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_30 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_31 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_136 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_137 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_28 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_29 = rand_strided((16, 6, 2048), (12288, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_91 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_32 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_138 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_51 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_140 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_93 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_33 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_142 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_150 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_17 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_18 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_34 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_35 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_152 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_153 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_32 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_33 = rand_strided((16, 6, 2048), (12288, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_102 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_36 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_57 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_156 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_104 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_5 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_37 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_158 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_168 = rand_strided((32768, 12), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_62 = rand_strided((32768, 6), (6, 1), device='cuda:0', dtype=torch.bfloat16)
    add_107 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_19 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_20 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_38 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_39 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_170 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_171 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_36 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_37 = rand_strided((16, 6, 2048), (12288, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_114 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_40 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_173 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_64 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_175 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_116 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_41 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_177 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_185 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_21 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_22 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_42 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_43 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_186 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_187 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_40 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_41 = rand_strided((16, 6, 2048), (12288, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_125 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_44 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_189 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_70 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_191 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_127 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    embedding_6 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_45 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_193 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_203 = rand_strided((32768, 12), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_75 = rand_strided((32768, 6), (6, 1), device='cuda:0', dtype=torch.bfloat16)
    add_130 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_23 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_24 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_46 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_47 = rand_strided((16, 2048, 6, 1), (12288, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_204 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_205 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_44 = rand_strided((16, 2048, 6, 128), (1572864, 768, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_45 = rand_strided((16, 6, 2048), (12288, 2048, 1), device='cuda:0', dtype=torch.float32)
    add_137 = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_48 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_208 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_77 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_210 = rand_strided((32768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    sub = rand_strided((16, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_49 = rand_strided((16, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_212 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_79 = rand_strided((32768, 32768), (32768, 1), device='cuda:0', dtype=torch.bfloat16)
    amax = rand_strided((32768, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((32768, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_343 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_82 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_86 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_90 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_94 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_98 = rand_strided((6, 12), (12, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_102 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_106 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_110 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_114 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_118 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_122 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_126 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_130 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_134 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_138 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_142 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_146 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_150 = rand_strided((6, 12), (12, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_154 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_158 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_162 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_166 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_170 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_174 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_178 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_182 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_186 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_190 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_194 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_198 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_202 = rand_strided((6, 12), (12, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_206 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_210 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_214 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_218 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_222 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_226 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_230 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_234 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_238 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_242 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_246 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_250 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_254 = rand_strided((6, 12), (12, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_258 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_262 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_266 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_270 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_274 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_278 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_282 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_286 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_290 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_294 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_298 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_302 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_306 = rand_strided((6, 12), (12, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_310 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_314 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_318 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_322 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_326 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_330 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_334 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_338 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_342 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_346 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_350 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_354 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_358 = rand_strided((6, 12), (12, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_362 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_366 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_370 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_374 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_378 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_382 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_386 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_390 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_394 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_398 = rand_strided((1, 24), (24, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_5, primals_7, primals_8, primals_93, primals_95, embedding, rsqrt, view, mm, cat, rsqrt_1, view_2, view_10, cat_1, cat_2, rsqrt_2, rsqrt_3, mul_16, mul_17, getitem, getitem_1, add_10, rsqrt_4, view_14, mm_5, view_16, add_12, embedding_1, rsqrt_5, view_18, view_28, mm_10, add_15, cat_3, cat_4, rsqrt_6, rsqrt_7, mul_34, mul_35, getitem_4, getitem_5, add_22, rsqrt_8, view_33, mm_12, view_35, add_24, rsqrt_9, view_37, view_45, cat_5, cat_6, rsqrt_10, rsqrt_11, mul_50, mul_51, getitem_8, getitem_9, add_33, rsqrt_12, view_49, mm_18, view_51, add_35, embedding_2, rsqrt_13, view_53, view_63, mm_23, add_38, cat_7, cat_8, rsqrt_14, rsqrt_15, mul_68, mul_69, getitem_12, getitem_13, add_45, rsqrt_16, view_68, mm_25, view_70, add_47, rsqrt_17, view_72, view_80, cat_9, cat_10, rsqrt_18, rsqrt_19, mul_84, mul_85, getitem_16, getitem_17, add_56, rsqrt_20, view_84, mm_31, view_86, add_58, embedding_3, rsqrt_21, view_88, view_98, mm_36, add_61, cat_11, cat_12, rsqrt_22, rsqrt_23, mul_102, mul_103, getitem_20, getitem_21, add_68, rsqrt_24, view_103, mm_38, view_105, add_70, rsqrt_25, view_107, view_115, cat_13, cat_14, rsqrt_26, rsqrt_27, mul_118, mul_119, getitem_24, getitem_25, add_79, rsqrt_28, view_119, mm_44, view_121, add_81, embedding_4, rsqrt_29, view_123, view_133, mm_49, add_84, cat_15, cat_16, rsqrt_30, rsqrt_31, mul_136, mul_137, getitem_28, getitem_29, add_91, rsqrt_32, view_138, mm_51, view_140, add_93, rsqrt_33, view_142, view_150, cat_17, cat_18, rsqrt_34, rsqrt_35, mul_152, mul_153, getitem_32, getitem_33, add_102, rsqrt_36, view_154, mm_57, view_156, add_104, embedding_5, rsqrt_37, view_158, view_168, mm_62, add_107, cat_19, cat_20, rsqrt_38, rsqrt_39, mul_170, mul_171, getitem_36, getitem_37, add_114, rsqrt_40, view_173, mm_64, view_175, add_116, rsqrt_41, view_177, view_185, cat_21, cat_22, rsqrt_42, rsqrt_43, mul_186, mul_187, getitem_40, getitem_41, add_125, rsqrt_44, view_189, mm_70, view_191, add_127, embedding_6, rsqrt_45, view_193, view_203, mm_75, add_130, cat_23, cat_24, rsqrt_46, rsqrt_47, mul_204, mul_205, getitem_44, getitem_45, add_137, rsqrt_48, view_208, mm_77, view_210, sub, rsqrt_49, view_212, mm_79, amax, log, convert_element_type_343, permute_82, permute_86, permute_90, permute_94, permute_98, permute_102, permute_106, permute_110, permute_114, permute_118, permute_122, permute_126, permute_130, permute_134, permute_138, permute_142, permute_146, permute_150, permute_154, permute_158, permute_162, permute_166, permute_170, permute_174, permute_178, permute_182, permute_186, permute_190, permute_194, permute_198, permute_202, permute_206, permute_210, permute_214, permute_218, permute_222, permute_226, permute_230, permute_234, permute_238, permute_242, permute_246, permute_250, permute_254, permute_258, permute_262, permute_266, permute_270, permute_274, permute_278, permute_282, permute_286, permute_290, permute_294, permute_298, permute_302, permute_306, permute_310, permute_314, permute_318, permute_322, permute_326, permute_330, permute_334, permute_338, permute_342, permute_346, permute_350, permute_354, permute_358, permute_362, permute_366, permute_370, permute_374, permute_378, permute_382, permute_386, permute_390, permute_394, permute_398, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
