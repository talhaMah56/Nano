# AOT ID: ['5_inference']
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


# kernel path: /work/nvme/bffp/tmahmood1/cisc662/cache/inductor/tmpz6px867g/xm/cxmd2z5hayh2hy3zhsi7r34xr74w7eg3rf3l444amulyx5ra54os.py
# Topologically Sorted Source Nodes: [mul, sub, mul_, sub_1, lerp_, sub_2, lerp__1, square, pow_2, bias2, truediv, sqrt, denom, truediv_2, pow_1, bias1, step_size, neg, mul_1, add_], Original ATen: [aten.mul, aten.rsub, aten.lerp, aten.pow, aten.div, aten.sqrt, aten.add, aten.neg, aten.copy_]
# Source node to ATen node mapping:
#   add_ => add_3
#   bias1 => sub_7
#   bias2 => sub_8
#   denom => add_2
#   lerp_ => abs_1, add, ge, mul_2, sub_2, sub_3, where, where_1
#   lerp__1 => abs_2, add_1, ge_1, mul_3, sub_5, sub_6, where_2, where_3
#   mul => mul
#   mul_ => mul_1
#   mul_1 => mul_4
#   neg => neg
#   pow_1 => pow_2
#   pow_2 => pow_3
#   sqrt => sqrt
#   square => pow_1
#   step_size => div_1
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_4
#   truediv => div
#   truediv_2 => div_2
# Graph fragment:
#   %copy_ : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=copy_]
#   %arg4_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg6_1]
#   %arg3_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %copy__1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=copy__1]
#   %arg7_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg7_1]
#   %copy__2 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=copy__2]
#   %arg8_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f32[][]cpu" = PlaceHolder[target=arg9_1]
#   %add_3 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=add_3]
#   %add : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=add]
#   %add_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=add_1]
#   %mul : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, %arg5_1), kwargs = {})
#   %sub : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul), kwargs = {})
#   %mul_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %sub), kwargs = {})
#   %sub_1 : Tensor "f32[][]cpu"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg6_1), kwargs = {})
#   %abs_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_1,), kwargs = {})
#   %ge : Tensor "b8[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_1, 0.5), kwargs = {})
#   %sub_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, 1), kwargs = {})
#   %where : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %sub_2, %sub_1), kwargs = {})
#   %sub_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg3_1, %arg1_1), kwargs = {})
#   %mul_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %sub_3), kwargs = {})
#   %where_1 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %arg3_1, %arg1_1), kwargs = {})
#   %add : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %where_1), kwargs = {})
#   %sub_4 : Tensor "f32[][]cpu"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %arg7_1), kwargs = {})
#   %abs_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_4,), kwargs = {})
#   %ge_1 : Tensor "b8[][]cpu"[num_users=2] = call_function[target=torch.ops.aten.ge.Scalar](args = (%abs_2, 0.5), kwargs = {})
#   %sub_5 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_4, 1), kwargs = {})
#   %where_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %sub_5, %sub_4), kwargs = {})
#   %pow_1 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg3_1, 2), kwargs = {})
#   %sub_6 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_1, %arg2_1), kwargs = {})
#   %mul_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_2, %sub_6), kwargs = {})
#   %where_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_1, %pow_1, %arg2_1), kwargs = {})
#   %add_1 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %where_3), kwargs = {})
#   %pow_3 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%arg7_1, %arg8_1), kwargs = {})
#   %sub_8 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %pow_3), kwargs = {})
#   %div : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1, %sub_8), kwargs = {})
#   %sqrt : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%div,), kwargs = {})
#   %add_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sqrt, %arg9_1), kwargs = {})
#   %div_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add, %add_2), kwargs = {})
#   %pow_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%arg6_1, %arg8_1), kwargs = {})
#   %sub_7 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %pow_2), kwargs = {})
#   %div_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg4_1, %sub_7), kwargs = {})
#   %neg : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div_1,), kwargs = {})
#   %mul_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %neg), kwargs = {})
#   %add_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_4), kwargs = {})
#   %copy_ : Tensor "f32[1][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg0_1, %add_3), kwargs = {})
#   %copy__1 : Tensor "f32[1][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg1_1, %add), kwargs = {})
#   %copy__2 : Tensor "f32[1][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg2_1, %add_1), kwargs = {})
#   return %add_3,%add,%add_1,%buf2,%buf9,%buf17
triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0 = async_compile.triton('triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'fp32', 'in_ptr2': 'fp32', 'in_ptr3': 'fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': 'fp32', 'in_ptr7': '*fp32', 'in_ptr8': 'fp32', 'in_ptr9': 'fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0', 'mutated_arg_names': ['in_ptr0', 'in_ptr5', 'in_ptr7', 'out_ptr3', 'out_ptr4', 'out_ptr5'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '4BCE242A979B338BE0244007D01AE8CB950366134EFD2ACEDAF485C417B3268A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = in_ptr1
    tmp3 = in_ptr2
    tmp8 = in_ptr3
    tmp15 = tl.load(in_ptr4 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp17 = tl.load(in_ptr5 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp23 = in_ptr6
    tmp30 = tl.load(in_ptr7 + (0))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp36 = in_ptr8
    tmp41 = in_ptr9
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp5 - tmp4
    tmp7 = tmp1 * tmp6
    tmp9 = tmp5 - tmp8
    tmp10 = tl_math.abs(tmp9)
    tmp11 = 0.5
    tmp12 = tmp10 >= tmp11
    tmp13 = tmp9 - tmp5
    tmp14 = tl.where(tmp12, tmp13, tmp9)
    tmp19 = tmp16 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tl.where(tmp12, tmp16, tmp18)
    tmp22 = tmp20 + tmp21
    tmp24 = tmp5 - tmp23
    tmp25 = tl_math.abs(tmp24)
    tmp26 = tmp25 >= tmp11
    tmp27 = tmp24 - tmp5
    tmp28 = tl.where(tmp26, tmp27, tmp24)
    tmp29 = tmp16 * tmp16
    tmp32 = tmp29 - tmp31
    tmp33 = tmp28 * tmp32
    tmp34 = tl.where(tmp26, tmp29, tmp31)
    tmp35 = tmp33 + tmp34
    tmp37 = libdevice.pow(tmp23, tmp36)
    tmp38 = tmp5 - tmp37
    tmp39 = (tmp35 / tmp38)
    tmp40 = libdevice.sqrt(tmp39)
    tmp42 = tmp40 + tmp41
    tmp43 = (tmp22 / tmp42)
    tmp44 = libdevice.pow(tmp8, tmp36)
    tmp45 = tmp5 - tmp44
    tmp46 = (tmp2 / tmp45)
    tmp47 = -tmp46
    tmp48 = tmp43 * tmp47
    tmp49 = tmp7 + tmp48
    tl.store(out_ptr3 + (tl.full([XBLOCK], 0, tl.int32)), tmp49, None)
    tl.store(out_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp22, None)
    tl.store(out_ptr5 + (tl.full([XBLOCK], 0, tl.int32)), tmp35, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, ), (1, ))
        assert_size_stride(arg1_1, (1, ), (1, ))
        assert_size_stride(arg2_1, (1, ), (1, ))
        assert_size_stride(arg3_1, (1, ), (1, ))
        assert_size_stride(arg4_1, (), ())
        assert_size_stride(arg5_1, (), ())
        assert_size_stride(arg6_1, (), ())
        assert_size_stride(arg7_1, (), ())
        assert_size_stride(arg8_1, (), ())
        assert_size_stride(arg9_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [mul, sub, mul_, sub_1, lerp_, sub_2, lerp__1, square, pow_2, bias2, truediv, sqrt, denom, truediv_2, pow_1, bias1, step_size, neg, mul_1, add_], Original ATen: [aten.mul, aten.rsub, aten.lerp, aten.pow, aten.div, aten.sqrt, aten.add, aten.neg, aten.copy_]
            # [Provenance debug handles] triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0:615
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__div_lerp_mul_neg_pow_rsub_sqrt_0.run(arg0_1, arg4_1.item(), arg5_1.item(), arg6_1.item(), arg3_1, arg1_1, arg7_1.item(), arg2_1, arg8_1.item(), arg9_1.item(), arg0_1, arg1_1, arg2_1, 1, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            del arg7_1
            del arg8_1
            del arg9_1
        return ()

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
