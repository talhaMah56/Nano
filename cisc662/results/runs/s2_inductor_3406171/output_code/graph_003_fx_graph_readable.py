class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[32768, 768]", arg1_1: "f32[32768, 768]", arg2_1: "f32[32768, 768]", arg3_1: "f32[32768, 768]", arg4_1: "f32[]", arg5_1: "f32[]", arg6_1: "f32[]", arg7_1: "f32[]", arg8_1: "f32[]", arg9_1: "f32[]"):
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:49 in adamw_step_fused, code: p32.mul_(1 - lr_t * wd_t)
        mul: "f32[]" = torch.ops.aten.mul.Tensor(arg4_1, arg5_1);  arg5_1 = None
        sub: "f32[]" = torch.ops.aten.sub.Tensor(1, mul);  mul = None
        mul_1: "f32[32768, 768]" = torch.ops.aten.mul.Tensor(arg0_1, sub);  sub = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:51 in adamw_step_fused, code: exp_avg32.lerp_(grad32, 1 - beta1_t)
        sub_1: "f32[]" = torch.ops.aten.sub.Tensor(1, arg6_1)
        abs_1: "f32[]" = torch.ops.aten.abs.default(sub_1)
        ge: "b8[]" = torch.ops.aten.ge.Scalar(abs_1, 0.5);  abs_1 = None
        sub_2: "f32[]" = torch.ops.aten.sub.Tensor(sub_1, 1)
        where: "f32[]" = torch.ops.aten.where.self(ge, sub_2, sub_1);  sub_2 = sub_1 = None
        where_1: "f32[32768, 768]" = torch.ops.aten.where.self(ge, arg3_1, arg1_1);  ge = None
        sub_3: "f32[32768, 768]" = torch.ops.aten.sub.Tensor(arg3_1, arg1_1)
        mul_2: "f32[32768, 768]" = torch.ops.aten.mul.Tensor(where, sub_3);  where = sub_3 = None
        add: "f32[32768, 768]" = torch.ops.aten.add.Tensor(mul_2, where_1);  mul_2 = where_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:52 in adamw_step_fused, code: exp_avg_sq32.lerp_(grad32.square(), 1 - beta2_t)
        pow_1: "f32[32768, 768]" = torch.ops.aten.pow.Tensor_Scalar(arg3_1, 2);  arg3_1 = None
        sub_4: "f32[]" = torch.ops.aten.sub.Tensor(1, arg7_1)
        abs_2: "f32[]" = torch.ops.aten.abs.default(sub_4)
        ge_1: "b8[]" = torch.ops.aten.ge.Scalar(abs_2, 0.5);  abs_2 = None
        sub_5: "f32[]" = torch.ops.aten.sub.Tensor(sub_4, 1)
        where_2: "f32[]" = torch.ops.aten.where.self(ge_1, sub_5, sub_4);  sub_5 = sub_4 = None
        where_3: "f32[32768, 768]" = torch.ops.aten.where.self(ge_1, pow_1, arg2_1);  ge_1 = None
        sub_6: "f32[32768, 768]" = torch.ops.aten.sub.Tensor(pow_1, arg2_1);  pow_1 = None
        mul_3: "f32[32768, 768]" = torch.ops.aten.mul.Tensor(where_2, sub_6);  where_2 = sub_6 = None
        add_1: "f32[32768, 768]" = torch.ops.aten.add.Tensor(mul_3, where_3);  mul_3 = where_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:54 in adamw_step_fused, code: bias1 = 1 - beta1_t ** step_t
        pow_2: "f32[]" = torch.ops.aten.pow.Tensor_Tensor(arg6_1, arg8_1);  arg6_1 = None
        sub_7: "f32[]" = torch.ops.aten.sub.Tensor(1, pow_2);  pow_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:55 in adamw_step_fused, code: bias2 = 1 - beta2_t ** step_t
        pow_3: "f32[]" = torch.ops.aten.pow.Tensor_Tensor(arg7_1, arg8_1);  arg7_1 = arg8_1 = None
        sub_8: "f32[]" = torch.ops.aten.sub.Tensor(1, pow_3);  pow_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:57 in adamw_step_fused, code: denom = (exp_avg_sq32 / bias2).sqrt() + eps_t
        div: "f32[32768, 768]" = torch.ops.aten.div.Tensor(add_1, sub_8);  sub_8 = None
        sqrt: "f32[32768, 768]" = torch.ops.aten.sqrt.default(div);  div = None
        add_2: "f32[32768, 768]" = torch.ops.aten.add.Tensor(sqrt, arg9_1);  sqrt = arg9_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:58 in adamw_step_fused, code: step_size = lr_t / bias1
        div_1: "f32[]" = torch.ops.aten.div.Tensor(arg4_1, sub_7);  arg4_1 = sub_7 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:59 in adamw_step_fused, code: p32.add_(exp_avg32 / denom, alpha=-step_size)
        div_2: "f32[32768, 768]" = torch.ops.aten.div.Tensor(add, add_2);  add_2 = None
        neg: "f32[]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        mul_4: "f32[32768, 768]" = torch.ops.aten.mul.Tensor(div_2, neg);  div_2 = neg = None
        add_3: "f32[32768, 768]" = torch.ops.aten.add.Tensor(mul_1, mul_4);  mul_1 = mul_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:61 in adamw_step_fused, code: p.copy_(p32)
        copy_: "f32[32768, 768]" = torch.ops.aten.copy_.default(arg0_1, add_3);  arg0_1 = add_3 = copy_ = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:62 in adamw_step_fused, code: exp_avg.copy_(exp_avg32)
        copy__1: "f32[32768, 768]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy__1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:63 in adamw_step_fused, code: exp_avg_sq.copy_(exp_avg_sq32)
        copy__2: "f32[32768, 768]" = torch.ops.aten.copy_.default(arg2_1, add_1);  arg2_1 = add_1 = copy__2 = None
        return ()
        