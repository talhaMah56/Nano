class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[12, 768, 3072]", arg2_1: "f32[12, 768, 3072]", arg3_1: "f32[12, 768, 3072]", arg4_1: "f32[]", arg5_1: "f32[12, 1, 3072]", arg6_1: "f32[]", arg7_1: "f32[]"):
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:132 in muon_step_fused, code: momentum_buffer.lerp_(stacked_grads, 1 - momentum)
        sub: "f32[]" = torch.ops.aten.sub.Tensor(1, arg0_1)
        abs_1: "f32[]" = torch.ops.aten.abs.default(sub)
        ge: "b8[]" = torch.ops.aten.ge.Scalar(abs_1, 0.5);  abs_1 = None
        sub_1: "f32[]" = torch.ops.aten.sub.Tensor(sub, 1)
        where: "f32[]" = torch.ops.aten.where.self(ge, sub_1, sub);  sub_1 = sub = None
        where_1: "f32[12, 768, 3072]" = torch.ops.aten.where.self(ge, arg1_1, arg2_1);  ge = None
        sub_2: "f32[12, 768, 3072]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1)
        mul: "f32[12, 768, 3072]" = torch.ops.aten.mul.Tensor(where, sub_2);  where = sub_2 = None
        add: "f32[12, 768, 3072]" = torch.ops.aten.add.Tensor(mul, where_1);  mul = where_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:133 in muon_step_fused, code: g = stacked_grads.lerp_(momentum_buffer, momentum)
        abs_2: "f32[]" = torch.ops.aten.abs.default(arg0_1)
        ge_1: "b8[]" = torch.ops.aten.ge.Scalar(abs_2, 0.5);  abs_2 = None
        sub_3: "f32[]" = torch.ops.aten.sub.Tensor(arg0_1, 1)
        where_2: "f32[]" = torch.ops.aten.where.self(ge_1, sub_3, arg0_1);  sub_3 = arg0_1 = None
        where_3: "f32[12, 768, 3072]" = torch.ops.aten.where.self(ge_1, add, arg1_1);  ge_1 = None
        sub_4: "f32[12, 768, 3072]" = torch.ops.aten.sub.Tensor(add, arg1_1)
        mul_1: "f32[12, 768, 3072]" = torch.ops.aten.mul.Tensor(where_2, sub_4);  where_2 = sub_4 = None
        add_1: "f32[12, 768, 3072]" = torch.ops.aten.add.Tensor(mul_1, where_3);  mul_1 = where_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:136 in muon_step_fused, code: X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
        convert_element_type: "bf16[12, 768, 3072]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:139 in muon_step_fused, code: target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
        convert_element_type_1: "f32[12, 768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32)
        pow_1: "f32[12, 768, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 2);  convert_element_type_1 = None
        sum_1: "f32[12, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_1, [-2, -1], True);  pow_1 = None
        pow_2: "f32[12, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        div: "f32[12, 1, 1]" = torch.ops.aten.div.Tensor(pow_2, 27.712812921102035);  pow_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:140 in muon_step_fused, code: row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        convert_element_type_2: "f32[12, 768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32)
        pow_3: "f32[12, 768, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2);  convert_element_type_2 = None
        sum_2: "f32[12, 768, 1]" = torch.ops.aten.sum.dim_IntList(pow_3, [-1], True);  pow_3 = None
        pow_4: "f32[12, 768, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_2, 0.5);  sum_2 = None
        clamp_min: "f32[12, 768, 1]" = torch.ops.aten.clamp_min.default(pow_4, 1e-06);  pow_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:141 in muon_step_fused, code: X = X * (target / row_norm).to(X.dtype)
        div_1: "f32[12, 768, 1]" = torch.ops.aten.div.Tensor(div, clamp_min);  div = clamp_min = None
        convert_element_type_3: "bf16[12, 768, 1]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        mul_2: "bf16[12, 768, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type, convert_element_type_3);  convert_element_type = convert_element_type_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:144 in muon_step_fused, code: X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        convert_element_type_4: "f32[12, 768, 3072]" = torch.ops.prims.convert_element_type.default(mul_2, torch.float32)
        pow_5: "f32[12, 768, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_4, 2);  convert_element_type_4 = None
        sum_3: "f32[12, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_5, [-2, -1], True);  pow_5 = None
        pow_6: "f32[12, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_3, 0.5);  sum_3 = None
        convert_element_type_5: "bf16[12, 1, 1]" = torch.ops.prims.convert_element_type.default(pow_6, torch.bfloat16);  pow_6 = None
        mul_3: "bf16[12, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.01);  convert_element_type_5 = None
        add_2: "bf16[12, 1, 1]" = torch.ops.aten.add.Tensor(mul_3, 1e-06);  mul_3 = None
        div_2: "bf16[12, 768, 3072]" = torch.ops.aten.div.Tensor(mul_2, add_2);  mul_2 = add_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:152 in muon_step_fused, code: A = X @ X.mT
        permute: "bf16[12, 3072, 768]" = torch.ops.aten.permute.default(div_2, [0, 2, 1])
        expand: "bf16[12, 768, 3072]" = torch.ops.aten.expand.default(div_2, [12, 768, 3072])
        expand_1: "bf16[12, 3072, 768]" = torch.ops.aten.expand.default(permute, [12, 3072, 768]);  permute = None
        bmm: "bf16[12, 768, 768]" = torch.ops.aten.bmm.default(expand, expand_1);  expand = expand_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:153 in muon_step_fused, code: B = b * A + c * (A @ A)
        mul_4: "bf16[12, 768, 768]" = torch.ops.aten.mul.Tensor(bmm, -22.48329292557795)
        expand_2: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(bmm, [12, 768, 768])
        expand_3: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(bmm, [12, 768, 768]);  bmm = None
        bmm_1: "bf16[12, 768, 768]" = torch.ops.aten.bmm.default(expand_2, expand_3);  expand_2 = expand_3 = None
        mul_5: "bf16[12, 768, 768]" = torch.ops.aten.mul.Tensor(bmm_1, 15.878769915207462);  bmm_1 = None
        add_3: "bf16[12, 768, 768]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:154 in muon_step_fused, code: X = a * X + B @ X
        mul_6: "bf16[12, 768, 3072]" = torch.ops.aten.mul.Tensor(div_2, 8.156554524902461)
        expand_4: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(add_3, [12, 768, 768]);  add_3 = None
        expand_5: "bf16[12, 768, 3072]" = torch.ops.aten.expand.default(div_2, [12, 768, 3072]);  div_2 = None
        bmm_2: "bf16[12, 768, 3072]" = torch.ops.aten.bmm.default(expand_4, expand_5);  expand_4 = expand_5 = None
        add_4: "bf16[12, 768, 3072]" = torch.ops.aten.add.Tensor(mul_6, bmm_2);  mul_6 = bmm_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:152 in muon_step_fused, code: A = X @ X.mT
        permute_1: "bf16[12, 3072, 768]" = torch.ops.aten.permute.default(add_4, [0, 2, 1])
        expand_6: "bf16[12, 768, 3072]" = torch.ops.aten.expand.default(add_4, [12, 768, 3072])
        expand_7: "bf16[12, 3072, 768]" = torch.ops.aten.expand.default(permute_1, [12, 3072, 768]);  permute_1 = None
        bmm_3: "bf16[12, 768, 768]" = torch.ops.aten.bmm.default(expand_6, expand_7);  expand_6 = expand_7 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:153 in muon_step_fused, code: B = b * A + c * (A @ A)
        mul_7: "bf16[12, 768, 768]" = torch.ops.aten.mul.Tensor(bmm_3, -2.808917465908714)
        expand_8: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(bmm_3, [12, 768, 768])
        expand_9: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(bmm_3, [12, 768, 768]);  bmm_3 = None
        bmm_4: "bf16[12, 768, 768]" = torch.ops.aten.bmm.default(expand_8, expand_9);  expand_8 = expand_9 = None
        mul_8: "bf16[12, 768, 768]" = torch.ops.aten.mul.Tensor(bmm_4, 0.5000178451051316);  bmm_4 = None
        add_5: "bf16[12, 768, 768]" = torch.ops.aten.add.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:154 in muon_step_fused, code: X = a * X + B @ X
        mul_9: "bf16[12, 768, 3072]" = torch.ops.aten.mul.Tensor(add_4, 4.042929935166739)
        expand_10: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(add_5, [12, 768, 768]);  add_5 = None
        expand_11: "bf16[12, 768, 3072]" = torch.ops.aten.expand.default(add_4, [12, 768, 3072]);  add_4 = None
        bmm_5: "bf16[12, 768, 3072]" = torch.ops.aten.bmm.default(expand_10, expand_11);  expand_10 = expand_11 = None
        add_6: "bf16[12, 768, 3072]" = torch.ops.aten.add.Tensor(mul_9, bmm_5);  mul_9 = bmm_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:152 in muon_step_fused, code: A = X @ X.mT
        permute_2: "bf16[12, 3072, 768]" = torch.ops.aten.permute.default(add_6, [0, 2, 1])
        expand_12: "bf16[12, 768, 3072]" = torch.ops.aten.expand.default(add_6, [12, 768, 3072])
        expand_13: "bf16[12, 3072, 768]" = torch.ops.aten.expand.default(permute_2, [12, 3072, 768]);  permute_2 = None
        bmm_6: "bf16[12, 768, 768]" = torch.ops.aten.bmm.default(expand_12, expand_13);  expand_12 = expand_13 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:153 in muon_step_fused, code: B = b * A + c * (A @ A)
        mul_10: "bf16[12, 768, 768]" = torch.ops.aten.mul.Tensor(bmm_6, -2.772484153217685)
        expand_14: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(bmm_6, [12, 768, 768])
        expand_15: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(bmm_6, [12, 768, 768]);  bmm_6 = None
        bmm_7: "bf16[12, 768, 768]" = torch.ops.aten.bmm.default(expand_14, expand_15);  expand_14 = expand_15 = None
        mul_11: "bf16[12, 768, 768]" = torch.ops.aten.mul.Tensor(bmm_7, 0.5060648178503393);  bmm_7 = None
        add_7: "bf16[12, 768, 768]" = torch.ops.aten.add.Tensor(mul_10, mul_11);  mul_10 = mul_11 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:154 in muon_step_fused, code: X = a * X + B @ X
        mul_12: "bf16[12, 768, 3072]" = torch.ops.aten.mul.Tensor(add_6, 3.8916678022926607)
        expand_16: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(add_7, [12, 768, 768]);  add_7 = None
        expand_17: "bf16[12, 768, 3072]" = torch.ops.aten.expand.default(add_6, [12, 768, 3072]);  add_6 = None
        bmm_8: "bf16[12, 768, 3072]" = torch.ops.aten.bmm.default(expand_16, expand_17);  expand_16 = expand_17 = None
        add_8: "bf16[12, 768, 3072]" = torch.ops.aten.add.Tensor(mul_12, bmm_8);  mul_12 = bmm_8 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:152 in muon_step_fused, code: A = X @ X.mT
        permute_3: "bf16[12, 3072, 768]" = torch.ops.aten.permute.default(add_8, [0, 2, 1])
        expand_18: "bf16[12, 768, 3072]" = torch.ops.aten.expand.default(add_8, [12, 768, 3072])
        expand_19: "bf16[12, 3072, 768]" = torch.ops.aten.expand.default(permute_3, [12, 3072, 768]);  permute_3 = None
        bmm_9: "bf16[12, 768, 768]" = torch.ops.aten.bmm.default(expand_18, expand_19);  expand_18 = expand_19 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:153 in muon_step_fused, code: B = b * A + c * (A @ A)
        mul_13: "bf16[12, 768, 768]" = torch.ops.aten.mul.Tensor(bmm_9, -2.3681294933425376)
        expand_20: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(bmm_9, [12, 768, 768])
        expand_21: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(bmm_9, [12, 768, 768]);  bmm_9 = None
        bmm_10: "bf16[12, 768, 768]" = torch.ops.aten.bmm.default(expand_20, expand_21);  expand_20 = expand_21 = None
        mul_14: "bf16[12, 768, 768]" = torch.ops.aten.mul.Tensor(bmm_10, 0.46449024233003106);  bmm_10 = None
        add_9: "bf16[12, 768, 768]" = torch.ops.aten.add.Tensor(mul_13, mul_14);  mul_13 = mul_14 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:154 in muon_step_fused, code: X = a * X + B @ X
        mul_15: "bf16[12, 768, 3072]" = torch.ops.aten.mul.Tensor(add_8, 3.285753657755655)
        expand_22: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(add_9, [12, 768, 768]);  add_9 = None
        expand_23: "bf16[12, 768, 3072]" = torch.ops.aten.expand.default(add_8, [12, 768, 3072]);  add_8 = None
        bmm_11: "bf16[12, 768, 3072]" = torch.ops.aten.bmm.default(expand_22, expand_23);  expand_22 = expand_23 = None
        add_10: "bf16[12, 768, 3072]" = torch.ops.aten.add.Tensor(mul_15, bmm_11);  mul_15 = bmm_11 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:152 in muon_step_fused, code: A = X @ X.mT
        permute_4: "bf16[12, 3072, 768]" = torch.ops.aten.permute.default(add_10, [0, 2, 1])
        expand_24: "bf16[12, 768, 3072]" = torch.ops.aten.expand.default(add_10, [12, 768, 3072])
        expand_25: "bf16[12, 3072, 768]" = torch.ops.aten.expand.default(permute_4, [12, 3072, 768]);  permute_4 = None
        bmm_12: "bf16[12, 768, 768]" = torch.ops.aten.bmm.default(expand_24, expand_25);  expand_24 = expand_25 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:153 in muon_step_fused, code: B = b * A + c * (A @ A)
        mul_16: "bf16[12, 768, 768]" = torch.ops.aten.mul.Tensor(bmm_12, -1.7097828382687081)
        expand_26: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(bmm_12, [12, 768, 768])
        expand_27: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(bmm_12, [12, 768, 768]);  bmm_12 = None
        bmm_13: "bf16[12, 768, 768]" = torch.ops.aten.bmm.default(expand_26, expand_27);  expand_26 = expand_27 = None
        mul_17: "bf16[12, 768, 768]" = torch.ops.aten.mul.Tensor(bmm_13, 0.42323551169305323);  bmm_13 = None
        add_11: "bf16[12, 768, 768]" = torch.ops.aten.add.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:154 in muon_step_fused, code: X = a * X + B @ X
        mul_18: "bf16[12, 768, 3072]" = torch.ops.aten.mul.Tensor(add_10, 2.3465413258596377)
        expand_28: "bf16[12, 768, 768]" = torch.ops.aten.expand.default(add_11, [12, 768, 768]);  add_11 = None
        expand_29: "bf16[12, 768, 3072]" = torch.ops.aten.expand.default(add_10, [12, 768, 3072]);  add_10 = None
        bmm_14: "bf16[12, 768, 3072]" = torch.ops.aten.bmm.default(expand_28, expand_29);  expand_28 = expand_29 = None
        add_12: "bf16[12, 768, 3072]" = torch.ops.aten.add.Tensor(mul_18, bmm_14);  mul_18 = bmm_14 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:156 in muon_step_fused, code: g = X.to(stacked_params.dtype)
        convert_element_type_36: "f32[12, 768, 3072]" = torch.ops.prims.convert_element_type.default(add_12, torch.float32);  add_12 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:160 in muon_step_fused, code: current_norm = g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        pow_7: "f32[12, 768, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_36, 2)
        sum_4: "f32[12, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_7, [-2, -1], True);  pow_7 = None
        pow_8: "f32[12, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None
        clamp_min_1: "f32[12, 1, 1]" = torch.ops.aten.clamp_min.default(pow_8, 1e-06);  pow_8 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:161 in muon_step_fused, code: g = g * (target_norm / current_norm).to(g.dtype)
        reciprocal: "f32[12, 1, 1]" = torch.ops.aten.reciprocal.default(clamp_min_1);  clamp_min_1 = None
        mul_19: "f32[12, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal, 27.712812921102035);  reciprocal = None
        mul_20: "f32[12, 768, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_36, mul_19);  convert_element_type_36 = mul_19 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:165 in muon_step_fused, code: v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        pow_9: "f32[12, 768, 3072]" = torch.ops.aten.pow.Tensor_Scalar(mul_20, 2)
        mean: "f32[12, 1, 3072]" = torch.ops.aten.mean.dim(pow_9, [-2], True);  pow_9 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:167 in muon_step_fused, code: v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
        sum_5: "f32[12, 1, 1]" = torch.ops.aten.sum.dim_IntList(mean, [-2, -1], True)
        mul_21: "f32[12, 1, 1]" = torch.ops.aten.mul.Tensor(sum_5, 768);  sum_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:168 in muon_step_fused, code: v_norm = v_norm_sq.sqrt()
        sqrt: "f32[12, 1, 1]" = torch.ops.aten.sqrt.default(mul_21);  mul_21 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:169 in muon_step_fused, code: second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
        sub_5: "f32[]" = torch.ops.aten.sub.Tensor(1, arg4_1);  arg4_1 = None
        abs_3: "f32[]" = torch.ops.aten.abs.default(sub_5)
        ge_2: "b8[]" = torch.ops.aten.ge.Scalar(abs_3, 0.5);  abs_3 = None
        sub_6: "f32[]" = torch.ops.aten.sub.Tensor(sub_5, 1)
        where_4: "f32[]" = torch.ops.aten.where.self(ge_2, sub_6, sub_5);  sub_6 = sub_5 = None
        where_5: "f32[12, 1, 3072]" = torch.ops.aten.where.self(ge_2, mean, arg5_1);  ge_2 = None
        sub_7: "f32[12, 1, 3072]" = torch.ops.aten.sub.Tensor(mean, arg5_1)
        mul_22: "f32[12, 1, 3072]" = torch.ops.aten.mul.Tensor(where_4, sub_7);  where_4 = sub_7 = None
        add_13: "f32[12, 1, 3072]" = torch.ops.aten.add.Tensor(mul_22, where_5);  mul_22 = where_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:170 in muon_step_fused, code: step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
        clamp_min_2: "f32[12, 1, 3072]" = torch.ops.aten.clamp_min.default(add_13, 1e-10)
        rsqrt: "f32[12, 1, 3072]" = torch.ops.aten.rsqrt.default(clamp_min_2);  clamp_min_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:171 in muon_step_fused, code: scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        mul_23: "f32[12, 1, 3072]" = torch.ops.aten.mul.Tensor(mean, 768);  mean = None
        pow_10: "f32[12, 1, 3072]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 2)
        mul_24: "f32[12, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_23, pow_10);  mul_23 = pow_10 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:172 in muon_step_fused, code: v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
        sum_6: "f32[12, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_24, [-2, -1], True);  mul_24 = None
        sqrt_1: "f32[12, 1, 1]" = torch.ops.aten.sqrt.default(sum_6);  sum_6 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:173 in muon_step_fused, code: final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
        clamp_min_3: "f32[12, 1, 1]" = torch.ops.aten.clamp_min.default(sqrt_1, 1e-10);  sqrt_1 = None
        div_3: "f32[12, 1, 1]" = torch.ops.aten.div.Tensor(sqrt, clamp_min_3);  sqrt = clamp_min_3 = None
        mul_25: "f32[12, 1, 3072]" = torch.ops.aten.mul.Tensor(rsqrt, div_3);  rsqrt = div_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:174 in muon_step_fused, code: g = g * final_scale.to(g.dtype)
        mul_26: "f32[12, 768, 3072]" = torch.ops.aten.mul.Tensor(mul_20, mul_25);  mul_20 = mul_25 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:179 in muon_step_fused, code: mask = (g * stacked_params) >= 0
        mul_27: "f32[12, 768, 3072]" = torch.ops.aten.mul.Tensor(mul_26, arg3_1)
        ge_3: "b8[12, 768, 3072]" = torch.ops.aten.ge.Scalar(mul_27, 0);  mul_27 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:180 in muon_step_fused, code: stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
        mul_28: "f32[12, 768, 3072]" = torch.ops.aten.mul.Tensor(arg6_1, mul_26);  mul_26 = None
        mul_29: "f32[]" = torch.ops.aten.mul.Tensor(arg6_1, arg7_1);  arg6_1 = arg7_1 = None
        mul_30: "f32[12, 768, 3072]" = torch.ops.aten.mul.Tensor(mul_29, arg3_1);  mul_29 = None
        mul_31: "f32[12, 768, 3072]" = torch.ops.aten.mul.Tensor(mul_30, ge_3);  mul_30 = ge_3 = None
        add_14: "f32[12, 768, 3072]" = torch.ops.aten.add.Tensor(mul_28, mul_31);  mul_28 = mul_31 = None
        sub_8: "f32[12, 768, 3072]" = torch.ops.aten.sub.Tensor(arg3_1, add_14);  add_14 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:133 in muon_step_fused, code: g = stacked_grads.lerp_(momentum_buffer, momentum)
        copy_: "f32[12, 768, 3072]" = torch.ops.aten.copy_.default(arg1_1, add_1);  arg1_1 = add_1 = copy_ = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:132 in muon_step_fused, code: momentum_buffer.lerp_(stacked_grads, 1 - momentum)
        copy__1: "f32[12, 768, 3072]" = torch.ops.aten.copy_.default(arg2_1, add);  arg2_1 = add = copy__1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:180 in muon_step_fused, code: stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
        copy__2: "f32[12, 768, 3072]" = torch.ops.aten.copy_.default(arg3_1, sub_8);  arg3_1 = sub_8 = copy__2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/optim.py:169 in muon_step_fused, code: second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
        copy__3: "f32[12, 1, 3072]" = torch.ops.aten.copy_.default(arg5_1, add_13);  arg5_1 = add_13 = copy__3 = None
        return ()
        