class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[16, 2048]", primals_2: "bf16[1, 20480, 1, 64]", primals_3: "bf16[1, 20480, 1, 64]", primals_4: "bf16[32768, 768]", primals_5: "f32[1]", primals_6: "f32[1, 24]", primals_7: "f32[12]", primals_8: "f32[12]", primals_9: "f32[768, 768]", primals_10: "f32[768, 768]", primals_11: "f32[768, 768]", primals_12: "f32[768, 768]", primals_13: "f32[3072, 768]", primals_14: "f32[768, 3072]", primals_15: "bf16[32768, 768]", primals_16: "f32[768, 768]", primals_17: "f32[768, 768]", primals_18: "f32[768, 768]", primals_19: "f32[6, 12]", primals_20: "f32[768, 768]", primals_21: "f32[3072, 768]", primals_22: "f32[768, 3072]", primals_23: "f32[768, 768]", primals_24: "f32[768, 768]", primals_25: "f32[768, 768]", primals_26: "f32[768, 768]", primals_27: "f32[3072, 768]", primals_28: "f32[768, 3072]", primals_29: "bf16[32768, 768]", primals_30: "f32[768, 768]", primals_31: "f32[768, 768]", primals_32: "f32[768, 768]", primals_33: "f32[6, 12]", primals_34: "f32[768, 768]", primals_35: "f32[3072, 768]", primals_36: "f32[768, 3072]", primals_37: "f32[768, 768]", primals_38: "f32[768, 768]", primals_39: "f32[768, 768]", primals_40: "f32[768, 768]", primals_41: "f32[3072, 768]", primals_42: "f32[768, 3072]", primals_43: "bf16[32768, 768]", primals_44: "f32[768, 768]", primals_45: "f32[768, 768]", primals_46: "f32[768, 768]", primals_47: "f32[6, 12]", primals_48: "f32[768, 768]", primals_49: "f32[3072, 768]", primals_50: "f32[768, 3072]", primals_51: "f32[768, 768]", primals_52: "f32[768, 768]", primals_53: "f32[768, 768]", primals_54: "f32[768, 768]", primals_55: "f32[3072, 768]", primals_56: "f32[768, 3072]", primals_57: "bf16[32768, 768]", primals_58: "f32[768, 768]", primals_59: "f32[768, 768]", primals_60: "f32[768, 768]", primals_61: "f32[6, 12]", primals_62: "f32[768, 768]", primals_63: "f32[3072, 768]", primals_64: "f32[768, 3072]", primals_65: "f32[768, 768]", primals_66: "f32[768, 768]", primals_67: "f32[768, 768]", primals_68: "f32[768, 768]", primals_69: "f32[3072, 768]", primals_70: "f32[768, 3072]", primals_71: "bf16[32768, 768]", primals_72: "f32[768, 768]", primals_73: "f32[768, 768]", primals_74: "f32[768, 768]", primals_75: "f32[6, 12]", primals_76: "f32[768, 768]", primals_77: "f32[3072, 768]", primals_78: "f32[768, 3072]", primals_79: "f32[768, 768]", primals_80: "f32[768, 768]", primals_81: "f32[768, 768]", primals_82: "f32[768, 768]", primals_83: "f32[3072, 768]", primals_84: "f32[768, 3072]", primals_85: "bf16[32768, 768]", primals_86: "f32[768, 768]", primals_87: "f32[768, 768]", primals_88: "f32[768, 768]", primals_89: "f32[6, 12]", primals_90: "f32[768, 768]", primals_91: "f32[3072, 768]", primals_92: "f32[768, 3072]", primals_93: "f32[1]", primals_94: "f32[32768, 768]", primals_95: "i64[16, 2048]"):
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:468 in forward, code: cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length
        slice_1: "bf16[1, 2048, 1, 64]" = torch.ops.aten.slice.Tensor(primals_2, 1, 0, 2048)
        slice_2: "bf16[1, 2048, 1, 64]" = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 2048)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:471 in forward, code: x = self.transformer.wte(idx) # embed current token
        embedding: "bf16[16, 2048, 768]" = torch.ops.aten.embedding.default(primals_4, primals_1);  primals_4 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)
        pow_1: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_1, [2], True);  pow_1 = None
        add: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean, 1.1920928955078125e-07);  mean = None
        rsqrt: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = None
        convert_element_type_1: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:479 in forward, code: gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        convert_element_type_2: "bf16[1]" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16)
        slice_3: "bf16[16, 2047, 768]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 1, 1, 9223372036854775807)
        slice_4: "bf16[16, 2047, 24]" = torch.ops.aten.slice.Tensor(slice_3, 2, 0, 24)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_3: "bf16[1, 24]" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute: "bf16[24, 1]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        clone: "bf16[16, 2047, 24]" = torch.ops.aten.clone.default(slice_4, memory_format = torch.contiguous_format);  slice_4 = None
        view: "bf16[32752, 24]" = torch.ops.aten.view.default(clone, [32752, 24]);  clone = None
        mm: "bf16[32752, 1]" = torch.ops.aten.mm.default(view, permute)
        view_1: "bf16[16, 2047, 1]" = torch.ops.aten.view.default(mm, [16, 2047, 1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:479 in forward, code: gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        sigmoid: "bf16[16, 2047, 1]" = torch.ops.aten.sigmoid.default(view_1);  view_1 = None
        mul_1: "bf16[16, 2047, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sigmoid);  convert_element_type_2 = sigmoid = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:480 in forward, code: x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        slice_5: "bf16[16, 1, 768]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 1, 0, 1)
        slice_7: "bf16[16, 2047, 768]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 1, 0, -1);  convert_element_type_1 = None
        mul_2: "bf16[16, 2047, 768]" = torch.ops.aten.mul.Tensor(mul_1, slice_7);  mul_1 = slice_7 = None
        add_1: "bf16[16, 2047, 768]" = torch.ops.aten.add.Tensor(slice_3, mul_2);  slice_3 = mul_2 = None
        cat: "bf16[16, 2048, 768]" = torch.ops.aten.cat.default([slice_5, add_1], 1);  slice_5 = add_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 0)
        mul_3: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select, cat);  select = None
        select_1: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 0)
        mul_4: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_1, cat);  select_1 = None
        add_2: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_6: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
        pow_2: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2)
        mean_1: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_2, [2], True);  pow_2 = None
        add_3: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_1, 1.1920928955078125e-07);  mean_1 = None
        rsqrt_1: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_5: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, rsqrt_1);  convert_element_type_6 = None
        convert_element_type_7: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_8: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        permute_1: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        view_2: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_7, [32768, 768]);  convert_element_type_7 = None
        mm_1: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_2, permute_1)
        view_3: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_1, [16, 2048, 768]);  mm_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_4: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_3, [16, 2048, 6, 128]);  view_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_11: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        permute_2: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        mm_2: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_2, permute_2)
        view_6: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_2, [16, 2048, 768]);  mm_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_7: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_6, [16, 2048, 6, 128]);  view_6 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_14: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        permute_3: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_14, [1, 0]);  convert_element_type_14 = None
        mm_3: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_2, permute_3)
        view_9: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_3, [16, 2048, 768]);  mm_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_10: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_9, [16, 2048, 6, 128]);  view_9 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_8: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_4, 3, 0, 64)
        slice_9: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_4, 3, 64, 9223372036854775807);  view_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_6: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_8, slice_1)
        mul_7: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_9, slice_2)
        add_4: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        neg: "bf16[1, 2048, 1, 64]" = torch.ops.aten.neg.default(slice_2)
        mul_8: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_8, neg);  slice_8 = None
        mul_9: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_9, slice_1);  slice_9 = None
        add_5: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_1: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_4, add_5], 3);  add_4 = add_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_10: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_7, 3, 0, 64)
        slice_11: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_7, 3, 64, 9223372036854775807);  view_7 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_10: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_10, slice_1)
        mul_11: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_11, slice_2)
        add_6: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_10, mul_11);  mul_10 = mul_11 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_12: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_10, neg);  slice_10 = None
        mul_13: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_11, slice_1);  slice_11 = None
        add_7: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_2: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_6, add_7], 3);  add_6 = add_7 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_17: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_1, torch.float32)
        pow_3: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_17, 2)
        mean_2: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_3, [3], True);  pow_3 = None
        add_8: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_2, 1.1920928955078125e-07);  mean_2 = None
        rsqrt_2: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_14: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_17, rsqrt_2);  convert_element_type_17 = None
        convert_element_type_18: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        convert_element_type_19: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_2, torch.float32)
        pow_4: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_19, 2)
        mean_3: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_4, [3], True);  pow_4 = None
        add_9: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_3, 1.1920928955078125e-07);  mean_3 = None
        rsqrt_3: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_15: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_19, rsqrt_3);  convert_element_type_19 = None
        convert_element_type_20: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_16: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.2);  convert_element_type_18 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_17: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.2);  convert_element_type_20 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward = torch.ops.flash_attn_3._flash_attn_forward.default(mul_16, mul_17, view_10, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0)
        getitem: "bf16[16, 2048, 6, 128]" = _flash_attn_forward[0]
        getitem_1: "f32[16, 6, 2048]" = _flash_attn_forward[1];  _flash_attn_forward = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_11: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_21: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        permute_4: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_21, [1, 0]);  convert_element_type_21 = None
        view_12: "bf16[32768, 768]" = torch.ops.aten.view.default(view_11, [32768, 768]);  view_11 = None
        mm_4: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_12, permute_4);  view_12 = None
        view_13: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_4, [16, 2048, 768]);  mm_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:151 in forward, code: x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        add_10: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_2, view_13);  add_2 = view_13 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_24: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32)
        pow_5: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_24, 2)
        mean_4: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_5, [2], True);  pow_5 = None
        add_11: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_4, 1.1920928955078125e-07);  mean_4 = None
        rsqrt_4: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_18: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_24, rsqrt_4);  convert_element_type_24 = None
        convert_element_type_25: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_26: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        permute_5: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_26, [1, 0]);  convert_element_type_26 = None
        view_14: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_25, [32768, 768]);  convert_element_type_25 = None
        mm_5: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_14, permute_5)
        view_15: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_5, [16, 2048, 3072])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_15);  view_15 = None
        pow_6: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu, 2);  relu = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_29: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        permute_6: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        view_16: "bf16[32768, 3072]" = torch.ops.aten.view.default(pow_6, [32768, 3072]);  pow_6 = None
        mm_6: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_16, permute_6)
        view_17: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_6, [16, 2048, 768]);  mm_6 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:152 in forward, code: x = x + self.mlp(norm(x))
        add_12: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_10, view_17);  view_17 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_2: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 1)
        mul_19: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_2, add_12);  select_2 = None
        select_3: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 1)
        mul_20: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_3, cat);  select_3 = None
        add_13: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_19, mul_20);  mul_19 = mul_20 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:501 in forward, code: ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        embedding_1: "bf16[16, 2048, 768]" = torch.ops.aten.embedding.default(primals_15, primals_1);  primals_15 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_32: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32)
        pow_7: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_32, 2)
        mean_5: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_7, [2], True);  pow_7 = None
        add_14: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_5, 1.1920928955078125e-07);  mean_5 = None
        rsqrt_5: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_21: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, rsqrt_5);  convert_element_type_32 = None
        convert_element_type_33: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_34: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        permute_7: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        view_18: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_33, [32768, 768])
        mm_7: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_18, permute_7)
        view_19: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_7, [16, 2048, 768]);  mm_7 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_20: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_19, [16, 2048, 6, 128]);  view_19 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_37: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        permute_8: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_37, [1, 0]);  convert_element_type_37 = None
        mm_8: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_18, permute_8)
        view_22: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_8, [16, 2048, 768]);  mm_8 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_23: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_22, [16, 2048, 6, 128]);  view_22 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_40: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        permute_9: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_40, [1, 0]);  convert_element_type_40 = None
        mm_9: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_18, permute_9)
        view_25: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_9, [16, 2048, 768]);  mm_9 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_26: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_25, [16, 2048, 6, 128]);  view_25 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_27: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(embedding_1, [16, 2048, 6, 128])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        slice_12: "bf16[16, 2048, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_33, 2, 0, 12);  convert_element_type_33 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_43: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        permute_10: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_43, [1, 0]);  convert_element_type_43 = None
        view_28: "bf16[32768, 12]" = torch.ops.aten.view.default(slice_12, [32768, 12]);  slice_12 = None
        mm_10: "bf16[32768, 6]" = torch.ops.aten.mm.default(view_28, permute_10)
        view_29: "bf16[16, 2048, 6]" = torch.ops.aten.view.default(mm_10, [16, 2048, 6])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        sigmoid_1: "bf16[16, 2048, 6]" = torch.ops.aten.sigmoid.default(view_29);  view_29 = None
        mul_22: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(sigmoid_1, 3);  sigmoid_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        unsqueeze: "bf16[16, 2048, 6, 1]" = torch.ops.aten.unsqueeze.default(mul_22, -1);  mul_22 = None
        mul_23: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(unsqueeze, view_27);  unsqueeze = view_27 = None
        add_15: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(view_26, mul_23);  view_26 = mul_23 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_13: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_20, 3, 0, 64)
        slice_14: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_20, 3, 64, 9223372036854775807);  view_20 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_24: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_13, slice_1)
        mul_25: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_14, slice_2)
        add_16: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_26: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_13, neg);  slice_13 = None
        mul_27: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_14, slice_1);  slice_14 = None
        add_17: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_3: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_16, add_17], 3);  add_16 = add_17 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_15: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_23, 3, 0, 64)
        slice_16: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_23, 3, 64, 9223372036854775807);  view_23 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_28: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_15, slice_1)
        mul_29: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_16, slice_2)
        add_18: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_30: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_15, neg);  slice_15 = None
        mul_31: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_16, slice_1);  slice_16 = None
        add_19: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_4: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_18, add_19], 3);  add_18 = add_19 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_46: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_3, torch.float32)
        pow_8: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_46, 2)
        mean_6: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_8, [3], True);  pow_8 = None
        add_20: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_6, 1.1920928955078125e-07);  mean_6 = None
        rsqrt_6: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_32: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_46, rsqrt_6);  convert_element_type_46 = None
        convert_element_type_47: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        convert_element_type_48: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_4, torch.float32)
        pow_9: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_48, 2)
        mean_7: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_9, [3], True);  pow_9 = None
        add_21: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_7, 1.1920928955078125e-07);  mean_7 = None
        rsqrt_7: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_33: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_48, rsqrt_7);  convert_element_type_48 = None
        convert_element_type_49: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_34: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.2);  convert_element_type_47 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_35: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.2);  convert_element_type_49 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_1 = torch.ops.flash_attn_3._flash_attn_forward.default(mul_34, mul_35, add_15, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0)
        getitem_4: "bf16[16, 2048, 6, 128]" = _flash_attn_forward_1[0]
        getitem_5: "f32[16, 6, 2048]" = _flash_attn_forward_1[1];  _flash_attn_forward_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_30: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_4, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_50: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        permute_11: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_50, [1, 0]);  convert_element_type_50 = None
        view_31: "bf16[32768, 768]" = torch.ops.aten.view.default(view_30, [32768, 768]);  view_30 = None
        mm_11: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_31, permute_11);  view_31 = None
        view_32: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_11, [16, 2048, 768]);  mm_11 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:151 in forward, code: x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        add_22: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_13, view_32);  add_13 = view_32 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_53: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_22, torch.float32)
        pow_10: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_53, 2)
        mean_8: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_10, [2], True);  pow_10 = None
        add_23: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_8, 1.1920928955078125e-07);  mean_8 = None
        rsqrt_8: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_36: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_53, rsqrt_8);  convert_element_type_53 = None
        convert_element_type_54: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_36, torch.bfloat16);  mul_36 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_55: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        permute_12: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_55, [1, 0]);  convert_element_type_55 = None
        view_33: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_54, [32768, 768]);  convert_element_type_54 = None
        mm_12: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_33, permute_12)
        view_34: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_12, [16, 2048, 3072])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_1: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_34);  view_34 = None
        pow_11: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_1, 2);  relu_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_58: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        permute_13: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_58, [1, 0]);  convert_element_type_58 = None
        view_35: "bf16[32768, 3072]" = torch.ops.aten.view.default(pow_11, [32768, 3072]);  pow_11 = None
        mm_13: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_35, permute_13)
        view_36: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_13, [16, 2048, 768]);  mm_13 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:152 in forward, code: x = x + self.mlp(norm(x))
        add_24: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_22, view_36);  view_36 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_4: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 2)
        mul_37: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_4, add_24);  select_4 = None
        select_5: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 2)
        mul_38: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_5, cat);  select_5 = None
        add_25: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_61: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_25, torch.float32)
        pow_12: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_61, 2)
        mean_9: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_12, [2], True);  pow_12 = None
        add_26: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_9, 1.1920928955078125e-07);  mean_9 = None
        rsqrt_9: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_39: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_61, rsqrt_9);  convert_element_type_61 = None
        convert_element_type_62: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_63: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        permute_14: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_63, [1, 0]);  convert_element_type_63 = None
        view_37: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_62, [32768, 768]);  convert_element_type_62 = None
        mm_14: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_37, permute_14)
        view_38: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_14, [16, 2048, 768]);  mm_14 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_39: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_38, [16, 2048, 6, 128]);  view_38 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_66: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        permute_15: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        mm_15: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_37, permute_15)
        view_41: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_15, [16, 2048, 768]);  mm_15 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_42: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_41, [16, 2048, 6, 128]);  view_41 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_69: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        permute_16: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_69, [1, 0]);  convert_element_type_69 = None
        mm_16: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_37, permute_16)
        view_44: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_16, [16, 2048, 768]);  mm_16 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_45: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_44, [16, 2048, 6, 128]);  view_44 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_17: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_39, 3, 0, 64)
        slice_18: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_39, 3, 64, 9223372036854775807);  view_39 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_40: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_17, slice_1)
        mul_41: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_18, slice_2)
        add_27: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_42: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_17, neg);  slice_17 = None
        mul_43: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_18, slice_1);  slice_18 = None
        add_28: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_5: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_27, add_28], 3);  add_27 = add_28 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_19: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_42, 3, 0, 64)
        slice_20: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_42, 3, 64, 9223372036854775807);  view_42 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_44: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_19, slice_1)
        mul_45: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_20, slice_2)
        add_29: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_46: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_19, neg);  slice_19 = None
        mul_47: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_20, slice_1);  slice_20 = None
        add_30: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_6: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_29, add_30], 3);  add_29 = add_30 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_72: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_5, torch.float32)
        pow_13: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_72, 2)
        mean_10: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_13, [3], True);  pow_13 = None
        add_31: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_10, 1.1920928955078125e-07);  mean_10 = None
        rsqrt_10: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_48: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_72, rsqrt_10);  convert_element_type_72 = None
        convert_element_type_73: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        convert_element_type_74: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_6, torch.float32)
        pow_14: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_74, 2)
        mean_11: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_14, [3], True);  pow_14 = None
        add_32: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_11, 1.1920928955078125e-07);  mean_11 = None
        rsqrt_11: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_49: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_74, rsqrt_11);  convert_element_type_74 = None
        convert_element_type_75: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_50: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.2);  convert_element_type_73 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_51: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 1.2);  convert_element_type_75 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_2 = torch.ops.flash_attn_3._flash_attn_forward.default(mul_50, mul_51, view_45, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0)
        getitem_8: "bf16[16, 2048, 6, 128]" = _flash_attn_forward_2[0]
        getitem_9: "f32[16, 6, 2048]" = _flash_attn_forward_2[1];  _flash_attn_forward_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_46: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_8, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_76: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        permute_17: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_76, [1, 0]);  convert_element_type_76 = None
        view_47: "bf16[32768, 768]" = torch.ops.aten.view.default(view_46, [32768, 768]);  view_46 = None
        mm_17: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_47, permute_17);  view_47 = None
        view_48: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_17, [16, 2048, 768]);  mm_17 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:151 in forward, code: x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        add_33: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_25, view_48);  add_25 = view_48 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_79: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_33, torch.float32)
        pow_15: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_79, 2)
        mean_12: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_15, [2], True);  pow_15 = None
        add_34: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_12, 1.1920928955078125e-07);  mean_12 = None
        rsqrt_12: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_52: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_79, rsqrt_12);  convert_element_type_79 = None
        convert_element_type_80: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_81: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        permute_18: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_81, [1, 0]);  convert_element_type_81 = None
        view_49: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_80, [32768, 768]);  convert_element_type_80 = None
        mm_18: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_49, permute_18)
        view_50: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_18, [16, 2048, 3072])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_2: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_50);  view_50 = None
        pow_16: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_2, 2);  relu_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_84: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        permute_19: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None
        view_51: "bf16[32768, 3072]" = torch.ops.aten.view.default(pow_16, [32768, 3072]);  pow_16 = None
        mm_19: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_51, permute_19)
        view_52: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_19, [16, 2048, 768]);  mm_19 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:152 in forward, code: x = x + self.mlp(norm(x))
        add_35: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_33, view_52);  view_52 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_6: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 3)
        mul_53: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_6, add_35);  select_6 = None
        select_7: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 3)
        mul_54: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_7, cat);  select_7 = None
        add_36: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:501 in forward, code: ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        embedding_2: "bf16[16, 2048, 768]" = torch.ops.aten.embedding.default(primals_29, primals_1);  primals_29 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_87: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_36, torch.float32)
        pow_17: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_87, 2)
        mean_13: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_17, [2], True);  pow_17 = None
        add_37: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_13, 1.1920928955078125e-07);  mean_13 = None
        rsqrt_13: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_55: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_87, rsqrt_13);  convert_element_type_87 = None
        convert_element_type_88: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_89: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        permute_20: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_89, [1, 0]);  convert_element_type_89 = None
        view_53: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_88, [32768, 768])
        mm_20: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_53, permute_20)
        view_54: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_20, [16, 2048, 768]);  mm_20 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_55: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_54, [16, 2048, 6, 128]);  view_54 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_92: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        permute_21: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_92, [1, 0]);  convert_element_type_92 = None
        mm_21: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_53, permute_21)
        view_57: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_21, [16, 2048, 768]);  mm_21 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_58: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_57, [16, 2048, 6, 128]);  view_57 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_95: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        permute_22: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_95, [1, 0]);  convert_element_type_95 = None
        mm_22: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_53, permute_22)
        view_60: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_22, [16, 2048, 768]);  mm_22 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_61: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_60, [16, 2048, 6, 128]);  view_60 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_62: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(embedding_2, [16, 2048, 6, 128])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        slice_21: "bf16[16, 2048, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_88, 2, 0, 12);  convert_element_type_88 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_98: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        permute_23: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_98, [1, 0]);  convert_element_type_98 = None
        view_63: "bf16[32768, 12]" = torch.ops.aten.view.default(slice_21, [32768, 12]);  slice_21 = None
        mm_23: "bf16[32768, 6]" = torch.ops.aten.mm.default(view_63, permute_23)
        view_64: "bf16[16, 2048, 6]" = torch.ops.aten.view.default(mm_23, [16, 2048, 6])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        sigmoid_2: "bf16[16, 2048, 6]" = torch.ops.aten.sigmoid.default(view_64);  view_64 = None
        mul_56: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(sigmoid_2, 3);  sigmoid_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        unsqueeze_1: "bf16[16, 2048, 6, 1]" = torch.ops.aten.unsqueeze.default(mul_56, -1);  mul_56 = None
        mul_57: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(unsqueeze_1, view_62);  unsqueeze_1 = view_62 = None
        add_38: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(view_61, mul_57);  view_61 = mul_57 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_22: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_55, 3, 0, 64)
        slice_23: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_55, 3, 64, 9223372036854775807);  view_55 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_58: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_22, slice_1)
        mul_59: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_23, slice_2)
        add_39: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_58, mul_59);  mul_58 = mul_59 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_60: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_22, neg);  slice_22 = None
        mul_61: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_23, slice_1);  slice_23 = None
        add_40: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_7: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_39, add_40], 3);  add_39 = add_40 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_24: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_58, 3, 0, 64)
        slice_25: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_58, 3, 64, 9223372036854775807);  view_58 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_62: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_24, slice_1)
        mul_63: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_25, slice_2)
        add_41: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_64: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_24, neg);  slice_24 = None
        mul_65: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_25, slice_1);  slice_25 = None
        add_42: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_8: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_41, add_42], 3);  add_41 = add_42 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_101: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_7, torch.float32)
        pow_18: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_101, 2)
        mean_14: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_18, [3], True);  pow_18 = None
        add_43: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_14, 1.1920928955078125e-07);  mean_14 = None
        rsqrt_14: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_66: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_101, rsqrt_14);  convert_element_type_101 = None
        convert_element_type_102: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_66, torch.bfloat16);  mul_66 = None
        convert_element_type_103: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_8, torch.float32)
        pow_19: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_103, 2)
        mean_15: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_19, [3], True);  pow_19 = None
        add_44: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_15, 1.1920928955078125e-07);  mean_15 = None
        rsqrt_15: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_67: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_103, rsqrt_15);  convert_element_type_103 = None
        convert_element_type_104: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_67, torch.bfloat16);  mul_67 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_68: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_102, 1.2);  convert_element_type_102 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_69: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_104, 1.2);  convert_element_type_104 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_3 = torch.ops.flash_attn_3._flash_attn_forward.default(mul_68, mul_69, add_38, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0)
        getitem_12: "bf16[16, 2048, 6, 128]" = _flash_attn_forward_3[0]
        getitem_13: "f32[16, 6, 2048]" = _flash_attn_forward_3[1];  _flash_attn_forward_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_65: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_12, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_105: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        permute_24: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_105, [1, 0]);  convert_element_type_105 = None
        view_66: "bf16[32768, 768]" = torch.ops.aten.view.default(view_65, [32768, 768]);  view_65 = None
        mm_24: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_66, permute_24);  view_66 = None
        view_67: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_24, [16, 2048, 768]);  mm_24 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:151 in forward, code: x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        add_45: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_36, view_67);  add_36 = view_67 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_108: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_45, torch.float32)
        pow_20: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_108, 2)
        mean_16: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_20, [2], True);  pow_20 = None
        add_46: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_16, 1.1920928955078125e-07);  mean_16 = None
        rsqrt_16: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_70: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_108, rsqrt_16);  convert_element_type_108 = None
        convert_element_type_109: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_110: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        permute_25: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_110, [1, 0]);  convert_element_type_110 = None
        view_68: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_109, [32768, 768]);  convert_element_type_109 = None
        mm_25: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_68, permute_25)
        view_69: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_25, [16, 2048, 3072])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_3: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_69);  view_69 = None
        pow_21: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_3, 2);  relu_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_113: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        permute_26: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_113, [1, 0]);  convert_element_type_113 = None
        view_70: "bf16[32768, 3072]" = torch.ops.aten.view.default(pow_21, [32768, 3072]);  pow_21 = None
        mm_26: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_70, permute_26)
        view_71: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_26, [16, 2048, 768]);  mm_26 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:152 in forward, code: x = x + self.mlp(norm(x))
        add_47: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_45, view_71);  view_71 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_8: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 4)
        mul_71: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_8, add_47);  select_8 = None
        select_9: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 4)
        mul_72: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_9, cat);  select_9 = None
        add_48: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_116: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)
        pow_22: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_116, 2)
        mean_17: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_22, [2], True);  pow_22 = None
        add_49: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_17, 1.1920928955078125e-07);  mean_17 = None
        rsqrt_17: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_73: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_116, rsqrt_17);  convert_element_type_116 = None
        convert_element_type_117: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_118: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        permute_27: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_118, [1, 0]);  convert_element_type_118 = None
        view_72: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_117, [32768, 768]);  convert_element_type_117 = None
        mm_27: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_72, permute_27)
        view_73: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_27, [16, 2048, 768]);  mm_27 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_74: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_73, [16, 2048, 6, 128]);  view_73 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_121: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        permute_28: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        mm_28: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_72, permute_28)
        view_76: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_28, [16, 2048, 768]);  mm_28 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_77: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_76, [16, 2048, 6, 128]);  view_76 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_124: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        permute_29: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_124, [1, 0]);  convert_element_type_124 = None
        mm_29: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_72, permute_29)
        view_79: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_29, [16, 2048, 768]);  mm_29 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_80: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_79, [16, 2048, 6, 128]);  view_79 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_26: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_74, 3, 0, 64)
        slice_27: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_74, 3, 64, 9223372036854775807);  view_74 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_74: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_26, slice_1)
        mul_75: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_27, slice_2)
        add_50: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_76: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_26, neg);  slice_26 = None
        mul_77: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_27, slice_1);  slice_27 = None
        add_51: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_76, mul_77);  mul_76 = mul_77 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_9: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_50, add_51], 3);  add_50 = add_51 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_28: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_77, 3, 0, 64)
        slice_29: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_77, 3, 64, 9223372036854775807);  view_77 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_78: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_28, slice_1)
        mul_79: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_29, slice_2)
        add_52: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_80: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_28, neg);  slice_28 = None
        mul_81: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_29, slice_1);  slice_29 = None
        add_53: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_80, mul_81);  mul_80 = mul_81 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_10: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_52, add_53], 3);  add_52 = add_53 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_127: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_9, torch.float32)
        pow_23: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_127, 2)
        mean_18: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_23, [3], True);  pow_23 = None
        add_54: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_18, 1.1920928955078125e-07);  mean_18 = None
        rsqrt_18: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_82: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_127, rsqrt_18);  convert_element_type_127 = None
        convert_element_type_128: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_82, torch.bfloat16);  mul_82 = None
        convert_element_type_129: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_10, torch.float32)
        pow_24: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_129, 2)
        mean_19: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_24, [3], True);  pow_24 = None
        add_55: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_19, 1.1920928955078125e-07);  mean_19 = None
        rsqrt_19: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        mul_83: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_129, rsqrt_19);  convert_element_type_129 = None
        convert_element_type_130: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_84: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_128, 1.2);  convert_element_type_128 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_85: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_130, 1.2);  convert_element_type_130 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_4 = torch.ops.flash_attn_3._flash_attn_forward.default(mul_84, mul_85, view_80, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0)
        getitem_16: "bf16[16, 2048, 6, 128]" = _flash_attn_forward_4[0]
        getitem_17: "f32[16, 6, 2048]" = _flash_attn_forward_4[1];  _flash_attn_forward_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_81: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_16, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_131: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        permute_30: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_131, [1, 0]);  convert_element_type_131 = None
        view_82: "bf16[32768, 768]" = torch.ops.aten.view.default(view_81, [32768, 768]);  view_81 = None
        mm_30: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_82, permute_30);  view_82 = None
        view_83: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_30, [16, 2048, 768]);  mm_30 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:151 in forward, code: x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        add_56: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_48, view_83);  add_48 = view_83 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_134: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_56, torch.float32)
        pow_25: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_134, 2)
        mean_20: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_25, [2], True);  pow_25 = None
        add_57: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_20, 1.1920928955078125e-07);  mean_20 = None
        rsqrt_20: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_86: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_134, rsqrt_20);  convert_element_type_134 = None
        convert_element_type_135: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_136: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        permute_31: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_136, [1, 0]);  convert_element_type_136 = None
        view_84: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_135, [32768, 768]);  convert_element_type_135 = None
        mm_31: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_84, permute_31)
        view_85: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_31, [16, 2048, 3072])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_4: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_85);  view_85 = None
        pow_26: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_4, 2);  relu_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_139: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        permute_32: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        view_86: "bf16[32768, 3072]" = torch.ops.aten.view.default(pow_26, [32768, 3072]);  pow_26 = None
        mm_32: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_86, permute_32)
        view_87: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_32, [16, 2048, 768]);  mm_32 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:152 in forward, code: x = x + self.mlp(norm(x))
        add_58: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_56, view_87);  view_87 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_10: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 5)
        mul_87: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_10, add_58);  select_10 = None
        select_11: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 5)
        mul_88: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_11, cat);  select_11 = None
        add_59: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:501 in forward, code: ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        embedding_3: "bf16[16, 2048, 768]" = torch.ops.aten.embedding.default(primals_43, primals_1);  primals_43 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_142: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        pow_27: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_142, 2)
        mean_21: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_27, [2], True);  pow_27 = None
        add_60: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_21, 1.1920928955078125e-07);  mean_21 = None
        rsqrt_21: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_89: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_142, rsqrt_21);  convert_element_type_142 = None
        convert_element_type_143: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_144: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        permute_33: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        view_88: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_143, [32768, 768])
        mm_33: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_88, permute_33)
        view_89: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_33, [16, 2048, 768]);  mm_33 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_90: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_89, [16, 2048, 6, 128]);  view_89 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_147: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        permute_34: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_147, [1, 0]);  convert_element_type_147 = None
        mm_34: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_88, permute_34)
        view_92: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_34, [16, 2048, 768]);  mm_34 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_93: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_92, [16, 2048, 6, 128]);  view_92 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_150: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        permute_35: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_150, [1, 0]);  convert_element_type_150 = None
        mm_35: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_88, permute_35)
        view_95: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_35, [16, 2048, 768]);  mm_35 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_96: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_95, [16, 2048, 6, 128]);  view_95 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_97: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(embedding_3, [16, 2048, 6, 128])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        slice_30: "bf16[16, 2048, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_143, 2, 0, 12);  convert_element_type_143 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_153: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        permute_36: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_153, [1, 0]);  convert_element_type_153 = None
        view_98: "bf16[32768, 12]" = torch.ops.aten.view.default(slice_30, [32768, 12]);  slice_30 = None
        mm_36: "bf16[32768, 6]" = torch.ops.aten.mm.default(view_98, permute_36)
        view_99: "bf16[16, 2048, 6]" = torch.ops.aten.view.default(mm_36, [16, 2048, 6])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        sigmoid_3: "bf16[16, 2048, 6]" = torch.ops.aten.sigmoid.default(view_99);  view_99 = None
        mul_90: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(sigmoid_3, 3);  sigmoid_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        unsqueeze_2: "bf16[16, 2048, 6, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
        mul_91: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(unsqueeze_2, view_97);  unsqueeze_2 = view_97 = None
        add_61: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(view_96, mul_91);  view_96 = mul_91 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_31: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_90, 3, 0, 64)
        slice_32: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_90, 3, 64, 9223372036854775807);  view_90 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_92: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_31, slice_1)
        mul_93: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_32, slice_2)
        add_62: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_94: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_31, neg);  slice_31 = None
        mul_95: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_32, slice_1);  slice_32 = None
        add_63: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_11: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_62, add_63], 3);  add_62 = add_63 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_33: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_93, 3, 0, 64)
        slice_34: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_93, 3, 64, 9223372036854775807);  view_93 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_96: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_33, slice_1)
        mul_97: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_34, slice_2)
        add_64: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_96, mul_97);  mul_96 = mul_97 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_98: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_33, neg);  slice_33 = None
        mul_99: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_34, slice_1);  slice_34 = None
        add_65: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_98, mul_99);  mul_98 = mul_99 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_12: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_64, add_65], 3);  add_64 = add_65 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_156: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_11, torch.float32)
        pow_28: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_156, 2)
        mean_22: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_28, [3], True);  pow_28 = None
        add_66: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_22, 1.1920928955078125e-07);  mean_22 = None
        rsqrt_22: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_100: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_156, rsqrt_22);  convert_element_type_156 = None
        convert_element_type_157: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_100, torch.bfloat16);  mul_100 = None
        convert_element_type_158: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_12, torch.float32)
        pow_29: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_158, 2)
        mean_23: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_29, [3], True);  pow_29 = None
        add_67: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_23, 1.1920928955078125e-07);  mean_23 = None
        rsqrt_23: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_101: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_158, rsqrt_23);  convert_element_type_158 = None
        convert_element_type_159: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_102: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_157, 1.2);  convert_element_type_157 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_103: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_159, 1.2);  convert_element_type_159 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_5 = torch.ops.flash_attn_3._flash_attn_forward.default(mul_102, mul_103, add_61, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0)
        getitem_20: "bf16[16, 2048, 6, 128]" = _flash_attn_forward_5[0]
        getitem_21: "f32[16, 6, 2048]" = _flash_attn_forward_5[1];  _flash_attn_forward_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_100: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_20, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_160: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        permute_37: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        view_101: "bf16[32768, 768]" = torch.ops.aten.view.default(view_100, [32768, 768]);  view_100 = None
        mm_37: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_101, permute_37);  view_101 = None
        view_102: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_37, [16, 2048, 768]);  mm_37 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:151 in forward, code: x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        add_68: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_59, view_102);  add_59 = view_102 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_163: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_68, torch.float32)
        pow_30: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_163, 2)
        mean_24: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_30, [2], True);  pow_30 = None
        add_69: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_24, 1.1920928955078125e-07);  mean_24 = None
        rsqrt_24: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_104: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_163, rsqrt_24);  convert_element_type_163 = None
        convert_element_type_164: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_165: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        permute_38: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_165, [1, 0]);  convert_element_type_165 = None
        view_103: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_164, [32768, 768]);  convert_element_type_164 = None
        mm_38: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_103, permute_38)
        view_104: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_38, [16, 2048, 3072])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_5: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_104);  view_104 = None
        pow_31: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_5, 2);  relu_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_168: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        permute_39: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_168, [1, 0]);  convert_element_type_168 = None
        view_105: "bf16[32768, 3072]" = torch.ops.aten.view.default(pow_31, [32768, 3072]);  pow_31 = None
        mm_39: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_105, permute_39)
        view_106: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_39, [16, 2048, 768]);  mm_39 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:152 in forward, code: x = x + self.mlp(norm(x))
        add_70: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_68, view_106);  view_106 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_12: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 6)
        mul_105: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_12, add_70);  select_12 = None
        select_13: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 6)
        mul_106: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_13, cat);  select_13 = None
        add_71: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_171: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32)
        pow_32: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_171, 2)
        mean_25: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_32, [2], True);  pow_32 = None
        add_72: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_25, 1.1920928955078125e-07);  mean_25 = None
        rsqrt_25: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_107: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_171, rsqrt_25);  convert_element_type_171 = None
        convert_element_type_172: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_107, torch.bfloat16);  mul_107 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_173: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        permute_40: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_173, [1, 0]);  convert_element_type_173 = None
        view_107: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_172, [32768, 768]);  convert_element_type_172 = None
        mm_40: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_107, permute_40)
        view_108: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_40, [16, 2048, 768]);  mm_40 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_109: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_108, [16, 2048, 6, 128]);  view_108 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_176: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        permute_41: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_176, [1, 0]);  convert_element_type_176 = None
        mm_41: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_107, permute_41)
        view_111: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_41, [16, 2048, 768]);  mm_41 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_112: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_111, [16, 2048, 6, 128]);  view_111 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_179: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        permute_42: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_179, [1, 0]);  convert_element_type_179 = None
        mm_42: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_107, permute_42)
        view_114: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_42, [16, 2048, 768]);  mm_42 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_115: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_114, [16, 2048, 6, 128]);  view_114 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_35: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_109, 3, 0, 64)
        slice_36: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_109, 3, 64, 9223372036854775807);  view_109 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_108: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_35, slice_1)
        mul_109: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_36, slice_2)
        add_73: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_110: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_35, neg);  slice_35 = None
        mul_111: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_36, slice_1);  slice_36 = None
        add_74: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_13: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_73, add_74], 3);  add_73 = add_74 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_37: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_112, 3, 0, 64)
        slice_38: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_112, 3, 64, 9223372036854775807);  view_112 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_112: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_37, slice_1)
        mul_113: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_38, slice_2)
        add_75: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_114: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_37, neg);  slice_37 = None
        mul_115: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_38, slice_1);  slice_38 = None
        add_76: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_14: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_75, add_76], 3);  add_75 = add_76 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_182: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_13, torch.float32)
        pow_33: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_182, 2)
        mean_26: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_33, [3], True);  pow_33 = None
        add_77: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_26, 1.1920928955078125e-07);  mean_26 = None
        rsqrt_26: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_116: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_182, rsqrt_26);  convert_element_type_182 = None
        convert_element_type_183: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_116, torch.bfloat16);  mul_116 = None
        convert_element_type_184: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_14, torch.float32)
        pow_34: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_184, 2)
        mean_27: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_34, [3], True);  pow_34 = None
        add_78: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_27, 1.1920928955078125e-07);  mean_27 = None
        rsqrt_27: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_117: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_184, rsqrt_27);  convert_element_type_184 = None
        convert_element_type_185: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_118: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_183, 1.2);  convert_element_type_183 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_119: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_185, 1.2);  convert_element_type_185 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_6 = torch.ops.flash_attn_3._flash_attn_forward.default(mul_118, mul_119, view_115, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0)
        getitem_24: "bf16[16, 2048, 6, 128]" = _flash_attn_forward_6[0]
        getitem_25: "f32[16, 6, 2048]" = _flash_attn_forward_6[1];  _flash_attn_forward_6 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_116: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_24, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_186: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        permute_43: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_186, [1, 0]);  convert_element_type_186 = None
        view_117: "bf16[32768, 768]" = torch.ops.aten.view.default(view_116, [32768, 768]);  view_116 = None
        mm_43: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_117, permute_43);  view_117 = None
        view_118: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_43, [16, 2048, 768]);  mm_43 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:151 in forward, code: x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        add_79: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_71, view_118);  add_71 = view_118 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_189: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32)
        pow_35: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_189, 2)
        mean_28: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_35, [2], True);  pow_35 = None
        add_80: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_28, 1.1920928955078125e-07);  mean_28 = None
        rsqrt_28: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_120: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_189, rsqrt_28);  convert_element_type_189 = None
        convert_element_type_190: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_191: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        permute_44: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_191, [1, 0]);  convert_element_type_191 = None
        view_119: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_190, [32768, 768]);  convert_element_type_190 = None
        mm_44: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_119, permute_44)
        view_120: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_44, [16, 2048, 3072])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_6: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_120);  view_120 = None
        pow_36: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_6, 2);  relu_6 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_194: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        permute_45: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_194, [1, 0]);  convert_element_type_194 = None
        view_121: "bf16[32768, 3072]" = torch.ops.aten.view.default(pow_36, [32768, 3072]);  pow_36 = None
        mm_45: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_121, permute_45)
        view_122: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_45, [16, 2048, 768]);  mm_45 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:152 in forward, code: x = x + self.mlp(norm(x))
        add_81: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_79, view_122);  view_122 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_14: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 7)
        mul_121: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_14, add_81);  select_14 = None
        select_15: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 7)
        mul_122: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_15, cat);  select_15 = None
        add_82: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:501 in forward, code: ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        embedding_4: "bf16[16, 2048, 768]" = torch.ops.aten.embedding.default(primals_57, primals_1);  primals_57 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_197: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_82, torch.float32)
        pow_37: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_197, 2)
        mean_29: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_37, [2], True);  pow_37 = None
        add_83: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_29, 1.1920928955078125e-07);  mean_29 = None
        rsqrt_29: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_123: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_197, rsqrt_29);  convert_element_type_197 = None
        convert_element_type_198: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_123, torch.bfloat16);  mul_123 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_199: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        permute_46: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_199, [1, 0]);  convert_element_type_199 = None
        view_123: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_198, [32768, 768])
        mm_46: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_123, permute_46)
        view_124: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_46, [16, 2048, 768]);  mm_46 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_125: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_124, [16, 2048, 6, 128]);  view_124 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_202: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        permute_47: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_202, [1, 0]);  convert_element_type_202 = None
        mm_47: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_123, permute_47)
        view_127: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_47, [16, 2048, 768]);  mm_47 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_128: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_127, [16, 2048, 6, 128]);  view_127 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_205: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        permute_48: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_205, [1, 0]);  convert_element_type_205 = None
        mm_48: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_123, permute_48)
        view_130: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_48, [16, 2048, 768]);  mm_48 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_131: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_130, [16, 2048, 6, 128]);  view_130 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_132: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(embedding_4, [16, 2048, 6, 128])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        slice_39: "bf16[16, 2048, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_198, 2, 0, 12);  convert_element_type_198 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_208: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        permute_49: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_208, [1, 0]);  convert_element_type_208 = None
        view_133: "bf16[32768, 12]" = torch.ops.aten.view.default(slice_39, [32768, 12]);  slice_39 = None
        mm_49: "bf16[32768, 6]" = torch.ops.aten.mm.default(view_133, permute_49)
        view_134: "bf16[16, 2048, 6]" = torch.ops.aten.view.default(mm_49, [16, 2048, 6])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        sigmoid_4: "bf16[16, 2048, 6]" = torch.ops.aten.sigmoid.default(view_134);  view_134 = None
        mul_124: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(sigmoid_4, 3);  sigmoid_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        unsqueeze_3: "bf16[16, 2048, 6, 1]" = torch.ops.aten.unsqueeze.default(mul_124, -1);  mul_124 = None
        mul_125: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(unsqueeze_3, view_132);  unsqueeze_3 = view_132 = None
        add_84: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(view_131, mul_125);  view_131 = mul_125 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_40: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_125, 3, 0, 64)
        slice_41: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_125, 3, 64, 9223372036854775807);  view_125 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_126: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_40, slice_1)
        mul_127: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_41, slice_2)
        add_85: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_126, mul_127);  mul_126 = mul_127 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_128: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_40, neg);  slice_40 = None
        mul_129: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_41, slice_1);  slice_41 = None
        add_86: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_128, mul_129);  mul_128 = mul_129 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_15: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_85, add_86], 3);  add_85 = add_86 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_42: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_128, 3, 0, 64)
        slice_43: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_128, 3, 64, 9223372036854775807);  view_128 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_130: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_42, slice_1)
        mul_131: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_43, slice_2)
        add_87: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_132: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_42, neg);  slice_42 = None
        mul_133: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_43, slice_1);  slice_43 = None
        add_88: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_16: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_87, add_88], 3);  add_87 = add_88 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_211: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_15, torch.float32)
        pow_38: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_211, 2)
        mean_30: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_38, [3], True);  pow_38 = None
        add_89: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_30, 1.1920928955078125e-07);  mean_30 = None
        rsqrt_30: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_134: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_211, rsqrt_30);  convert_element_type_211 = None
        convert_element_type_212: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None
        convert_element_type_213: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_16, torch.float32)
        pow_39: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_213, 2)
        mean_31: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_39, [3], True);  pow_39 = None
        add_90: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_31, 1.1920928955078125e-07);  mean_31 = None
        rsqrt_31: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_135: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_213, rsqrt_31);  convert_element_type_213 = None
        convert_element_type_214: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_135, torch.bfloat16);  mul_135 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_136: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_212, 1.2);  convert_element_type_212 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_137: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_214, 1.2);  convert_element_type_214 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_7 = torch.ops.flash_attn_3._flash_attn_forward.default(mul_136, mul_137, add_84, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0)
        getitem_28: "bf16[16, 2048, 6, 128]" = _flash_attn_forward_7[0]
        getitem_29: "f32[16, 6, 2048]" = _flash_attn_forward_7[1];  _flash_attn_forward_7 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_135: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_28, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_215: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        permute_50: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_215, [1, 0]);  convert_element_type_215 = None
        view_136: "bf16[32768, 768]" = torch.ops.aten.view.default(view_135, [32768, 768]);  view_135 = None
        mm_50: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_136, permute_50);  view_136 = None
        view_137: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_50, [16, 2048, 768]);  mm_50 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:151 in forward, code: x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        add_91: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_82, view_137);  add_82 = view_137 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_218: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_91, torch.float32)
        pow_40: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_218, 2)
        mean_32: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_40, [2], True);  pow_40 = None
        add_92: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_32, 1.1920928955078125e-07);  mean_32 = None
        rsqrt_32: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_138: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_218, rsqrt_32);  convert_element_type_218 = None
        convert_element_type_219: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_138, torch.bfloat16);  mul_138 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_220: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        permute_51: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_220, [1, 0]);  convert_element_type_220 = None
        view_138: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_219, [32768, 768]);  convert_element_type_219 = None
        mm_51: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_138, permute_51)
        view_139: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_51, [16, 2048, 3072])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_7: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_139);  view_139 = None
        pow_41: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_7, 2);  relu_7 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_223: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        permute_52: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        view_140: "bf16[32768, 3072]" = torch.ops.aten.view.default(pow_41, [32768, 3072]);  pow_41 = None
        mm_52: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_140, permute_52)
        view_141: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_52, [16, 2048, 768]);  mm_52 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:152 in forward, code: x = x + self.mlp(norm(x))
        add_93: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_91, view_141);  view_141 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_16: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 8)
        mul_139: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_16, add_93);  select_16 = None
        select_17: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 8)
        mul_140: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_17, cat);  select_17 = None
        add_94: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_226: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_94, torch.float32)
        pow_42: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_226, 2)
        mean_33: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_42, [2], True);  pow_42 = None
        add_95: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_33, 1.1920928955078125e-07);  mean_33 = None
        rsqrt_33: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_141: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_226, rsqrt_33);  convert_element_type_226 = None
        convert_element_type_227: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16);  mul_141 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_228: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        permute_53: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_228, [1, 0]);  convert_element_type_228 = None
        view_142: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_227, [32768, 768]);  convert_element_type_227 = None
        mm_53: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_142, permute_53)
        view_143: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_53, [16, 2048, 768]);  mm_53 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_144: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_143, [16, 2048, 6, 128]);  view_143 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_231: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        permute_54: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_231, [1, 0]);  convert_element_type_231 = None
        mm_54: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_142, permute_54)
        view_146: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_54, [16, 2048, 768]);  mm_54 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_147: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_146, [16, 2048, 6, 128]);  view_146 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_234: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        permute_55: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_234, [1, 0]);  convert_element_type_234 = None
        mm_55: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_142, permute_55)
        view_149: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_55, [16, 2048, 768]);  mm_55 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_150: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_149, [16, 2048, 6, 128]);  view_149 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_44: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_144, 3, 0, 64)
        slice_45: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_144, 3, 64, 9223372036854775807);  view_144 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_142: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_44, slice_1)
        mul_143: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_45, slice_2)
        add_96: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_144: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_44, neg);  slice_44 = None
        mul_145: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_45, slice_1);  slice_45 = None
        add_97: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_17: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_96, add_97], 3);  add_96 = add_97 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_46: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_147, 3, 0, 64)
        slice_47: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_147, 3, 64, 9223372036854775807);  view_147 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_146: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_46, slice_1)
        mul_147: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_47, slice_2)
        add_98: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_146, mul_147);  mul_146 = mul_147 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_148: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_46, neg);  slice_46 = None
        mul_149: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_47, slice_1);  slice_47 = None
        add_99: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_18: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_98, add_99], 3);  add_98 = add_99 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_237: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_17, torch.float32)
        pow_43: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_237, 2)
        mean_34: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_43, [3], True);  pow_43 = None
        add_100: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_34, 1.1920928955078125e-07);  mean_34 = None
        rsqrt_34: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_150: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_237, rsqrt_34);  convert_element_type_237 = None
        convert_element_type_238: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None
        convert_element_type_239: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_18, torch.float32)
        pow_44: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_239, 2)
        mean_35: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_44, [3], True);  pow_44 = None
        add_101: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_35, 1.1920928955078125e-07);  mean_35 = None
        rsqrt_35: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        mul_151: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_239, rsqrt_35);  convert_element_type_239 = None
        convert_element_type_240: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_151, torch.bfloat16);  mul_151 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_152: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_238, 1.2);  convert_element_type_238 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_153: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_240, 1.2);  convert_element_type_240 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_8 = torch.ops.flash_attn_3._flash_attn_forward.default(mul_152, mul_153, view_150, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0)
        getitem_32: "bf16[16, 2048, 6, 128]" = _flash_attn_forward_8[0]
        getitem_33: "f32[16, 6, 2048]" = _flash_attn_forward_8[1];  _flash_attn_forward_8 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_151: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_32, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_241: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        permute_56: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_241, [1, 0]);  convert_element_type_241 = None
        view_152: "bf16[32768, 768]" = torch.ops.aten.view.default(view_151, [32768, 768]);  view_151 = None
        mm_56: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_152, permute_56);  view_152 = None
        view_153: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_56, [16, 2048, 768]);  mm_56 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:151 in forward, code: x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        add_102: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_94, view_153);  add_94 = view_153 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_244: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_102, torch.float32)
        pow_45: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_244, 2)
        mean_36: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_45, [2], True);  pow_45 = None
        add_103: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_36, 1.1920928955078125e-07);  mean_36 = None
        rsqrt_36: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_154: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_244, rsqrt_36);  convert_element_type_244 = None
        convert_element_type_245: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_154, torch.bfloat16);  mul_154 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_246: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        permute_57: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_246, [1, 0]);  convert_element_type_246 = None
        view_154: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_245, [32768, 768]);  convert_element_type_245 = None
        mm_57: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_154, permute_57)
        view_155: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_57, [16, 2048, 3072])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_8: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_155);  view_155 = None
        pow_46: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_8, 2);  relu_8 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_249: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        permute_58: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_249, [1, 0]);  convert_element_type_249 = None
        view_156: "bf16[32768, 3072]" = torch.ops.aten.view.default(pow_46, [32768, 3072]);  pow_46 = None
        mm_58: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_156, permute_58)
        view_157: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_58, [16, 2048, 768]);  mm_58 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:152 in forward, code: x = x + self.mlp(norm(x))
        add_104: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_102, view_157);  view_157 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_18: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 9)
        mul_155: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_18, add_104);  select_18 = None
        select_19: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 9)
        mul_156: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_19, cat);  select_19 = None
        add_105: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:501 in forward, code: ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        embedding_5: "bf16[16, 2048, 768]" = torch.ops.aten.embedding.default(primals_71, primals_1);  primals_71 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_252: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_105, torch.float32)
        pow_47: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_252, 2)
        mean_37: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_47, [2], True);  pow_47 = None
        add_106: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_37, 1.1920928955078125e-07);  mean_37 = None
        rsqrt_37: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_157: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_252, rsqrt_37);  convert_element_type_252 = None
        convert_element_type_253: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_157, torch.bfloat16);  mul_157 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_254: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        permute_59: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_254, [1, 0]);  convert_element_type_254 = None
        view_158: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_253, [32768, 768])
        mm_59: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_158, permute_59)
        view_159: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_59, [16, 2048, 768]);  mm_59 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_160: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_159, [16, 2048, 6, 128]);  view_159 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_257: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        permute_60: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_257, [1, 0]);  convert_element_type_257 = None
        mm_60: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_158, permute_60)
        view_162: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_60, [16, 2048, 768]);  mm_60 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_163: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_162, [16, 2048, 6, 128]);  view_162 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_260: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        permute_61: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_260, [1, 0]);  convert_element_type_260 = None
        mm_61: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_158, permute_61)
        view_165: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_61, [16, 2048, 768]);  mm_61 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_166: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_165, [16, 2048, 6, 128]);  view_165 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_167: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(embedding_5, [16, 2048, 6, 128])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        slice_48: "bf16[16, 2048, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_253, 2, 0, 12);  convert_element_type_253 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_263: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        permute_62: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_263, [1, 0]);  convert_element_type_263 = None
        view_168: "bf16[32768, 12]" = torch.ops.aten.view.default(slice_48, [32768, 12]);  slice_48 = None
        mm_62: "bf16[32768, 6]" = torch.ops.aten.mm.default(view_168, permute_62)
        view_169: "bf16[16, 2048, 6]" = torch.ops.aten.view.default(mm_62, [16, 2048, 6])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        sigmoid_5: "bf16[16, 2048, 6]" = torch.ops.aten.sigmoid.default(view_169);  view_169 = None
        mul_158: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(sigmoid_5, 3);  sigmoid_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        unsqueeze_4: "bf16[16, 2048, 6, 1]" = torch.ops.aten.unsqueeze.default(mul_158, -1);  mul_158 = None
        mul_159: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(unsqueeze_4, view_167);  unsqueeze_4 = view_167 = None
        add_107: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(view_166, mul_159);  view_166 = mul_159 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_49: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_160, 3, 0, 64)
        slice_50: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_160, 3, 64, 9223372036854775807);  view_160 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_160: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_49, slice_1)
        mul_161: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_50, slice_2)
        add_108: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_162: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_49, neg);  slice_49 = None
        mul_163: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_50, slice_1);  slice_50 = None
        add_109: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_19: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_108, add_109], 3);  add_108 = add_109 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_51: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_163, 3, 0, 64)
        slice_52: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_163, 3, 64, 9223372036854775807);  view_163 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_164: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_51, slice_1)
        mul_165: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_52, slice_2)
        add_110: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_166: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_51, neg);  slice_51 = None
        mul_167: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_52, slice_1);  slice_52 = None
        add_111: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_166, mul_167);  mul_166 = mul_167 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_20: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_110, add_111], 3);  add_110 = add_111 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_266: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_19, torch.float32)
        pow_48: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_266, 2)
        mean_38: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_48, [3], True);  pow_48 = None
        add_112: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_38, 1.1920928955078125e-07);  mean_38 = None
        rsqrt_38: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_168: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_266, rsqrt_38);  convert_element_type_266 = None
        convert_element_type_267: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None
        convert_element_type_268: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_20, torch.float32)
        pow_49: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_268, 2)
        mean_39: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_49, [3], True);  pow_49 = None
        add_113: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_39, 1.1920928955078125e-07);  mean_39 = None
        rsqrt_39: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_169: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_268, rsqrt_39);  convert_element_type_268 = None
        convert_element_type_269: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_169, torch.bfloat16);  mul_169 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_170: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_267, 1.2);  convert_element_type_267 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_171: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_269, 1.2);  convert_element_type_269 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_9 = torch.ops.flash_attn_3._flash_attn_forward.default(mul_170, mul_171, add_107, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0)
        getitem_36: "bf16[16, 2048, 6, 128]" = _flash_attn_forward_9[0]
        getitem_37: "f32[16, 6, 2048]" = _flash_attn_forward_9[1];  _flash_attn_forward_9 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_170: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_36, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_270: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        permute_63: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        view_171: "bf16[32768, 768]" = torch.ops.aten.view.default(view_170, [32768, 768]);  view_170 = None
        mm_63: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_171, permute_63);  view_171 = None
        view_172: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_63, [16, 2048, 768]);  mm_63 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:151 in forward, code: x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        add_114: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_105, view_172);  add_105 = view_172 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_273: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_114, torch.float32)
        pow_50: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_273, 2)
        mean_40: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_50, [2], True);  pow_50 = None
        add_115: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_40, 1.1920928955078125e-07);  mean_40 = None
        rsqrt_40: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_172: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_273, rsqrt_40);  convert_element_type_273 = None
        convert_element_type_274: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_172, torch.bfloat16);  mul_172 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_275: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        permute_64: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_275, [1, 0]);  convert_element_type_275 = None
        view_173: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_274, [32768, 768]);  convert_element_type_274 = None
        mm_64: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_173, permute_64)
        view_174: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_64, [16, 2048, 3072])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_9: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_174);  view_174 = None
        pow_51: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_9, 2);  relu_9 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_278: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        permute_65: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_278, [1, 0]);  convert_element_type_278 = None
        view_175: "bf16[32768, 3072]" = torch.ops.aten.view.default(pow_51, [32768, 3072]);  pow_51 = None
        mm_65: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_175, permute_65)
        view_176: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_65, [16, 2048, 768]);  mm_65 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:152 in forward, code: x = x + self.mlp(norm(x))
        add_116: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_114, view_176);  view_176 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_20: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 10)
        mul_173: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_20, add_116);  select_20 = None
        select_21: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 10)
        mul_174: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_21, cat);  select_21 = None
        add_117: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_281: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_117, torch.float32)
        pow_52: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_281, 2)
        mean_41: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_52, [2], True);  pow_52 = None
        add_118: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_41, 1.1920928955078125e-07);  mean_41 = None
        rsqrt_41: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_175: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_281, rsqrt_41);  convert_element_type_281 = None
        convert_element_type_282: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_283: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        permute_66: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_283, [1, 0]);  convert_element_type_283 = None
        view_177: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_282, [32768, 768]);  convert_element_type_282 = None
        mm_66: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_177, permute_66)
        view_178: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_66, [16, 2048, 768]);  mm_66 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_179: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_178, [16, 2048, 6, 128]);  view_178 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_286: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        permute_67: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_286, [1, 0]);  convert_element_type_286 = None
        mm_67: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_177, permute_67)
        view_181: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_67, [16, 2048, 768]);  mm_67 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_182: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_181, [16, 2048, 6, 128]);  view_181 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_289: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        permute_68: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_289, [1, 0]);  convert_element_type_289 = None
        mm_68: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_177, permute_68)
        view_184: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_68, [16, 2048, 768]);  mm_68 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_185: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_184, [16, 2048, 6, 128]);  view_184 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_53: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_179, 3, 0, 64)
        slice_54: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_179, 3, 64, 9223372036854775807);  view_179 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_176: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_53, slice_1)
        mul_177: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_54, slice_2)
        add_119: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_178: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_53, neg);  slice_53 = None
        mul_179: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_54, slice_1);  slice_54 = None
        add_120: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_21: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_119, add_120], 3);  add_119 = add_120 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_55: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_182, 3, 0, 64)
        slice_56: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_182, 3, 64, 9223372036854775807);  view_182 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_180: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_55, slice_1)
        mul_181: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_56, slice_2)
        add_121: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_182: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_55, neg);  slice_55 = None
        mul_183: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_56, slice_1);  slice_56 = None
        add_122: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_22: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_121, add_122], 3);  add_121 = add_122 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_292: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_21, torch.float32)
        pow_53: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_292, 2)
        mean_42: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_53, [3], True);  pow_53 = None
        add_123: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_42, 1.1920928955078125e-07);  mean_42 = None
        rsqrt_42: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        mul_184: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_292, rsqrt_42);  convert_element_type_292 = None
        convert_element_type_293: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_184, torch.bfloat16);  mul_184 = None
        convert_element_type_294: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_22, torch.float32)
        pow_54: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_294, 2)
        mean_43: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_54, [3], True);  pow_54 = None
        add_124: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_43, 1.1920928955078125e-07);  mean_43 = None
        rsqrt_43: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        mul_185: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_294, rsqrt_43);  convert_element_type_294 = None
        convert_element_type_295: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_185, torch.bfloat16);  mul_185 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_186: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_293, 1.2);  convert_element_type_293 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_187: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_295, 1.2);  convert_element_type_295 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_10 = torch.ops.flash_attn_3._flash_attn_forward.default(mul_186, mul_187, view_185, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0)
        getitem_40: "bf16[16, 2048, 6, 128]" = _flash_attn_forward_10[0]
        getitem_41: "f32[16, 6, 2048]" = _flash_attn_forward_10[1];  _flash_attn_forward_10 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_186: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_40, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_296: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        permute_69: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_296, [1, 0]);  convert_element_type_296 = None
        view_187: "bf16[32768, 768]" = torch.ops.aten.view.default(view_186, [32768, 768]);  view_186 = None
        mm_69: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_187, permute_69);  view_187 = None
        view_188: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_69, [16, 2048, 768]);  mm_69 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:151 in forward, code: x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        add_125: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_117, view_188);  add_117 = view_188 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_299: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_125, torch.float32)
        pow_55: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_299, 2)
        mean_44: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_55, [2], True);  pow_55 = None
        add_126: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_44, 1.1920928955078125e-07);  mean_44 = None
        rsqrt_44: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_188: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_299, rsqrt_44);  convert_element_type_299 = None
        convert_element_type_300: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_188, torch.bfloat16);  mul_188 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_301: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        permute_70: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_301, [1, 0]);  convert_element_type_301 = None
        view_189: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_300, [32768, 768]);  convert_element_type_300 = None
        mm_70: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_189, permute_70)
        view_190: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_70, [16, 2048, 3072])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_10: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_190);  view_190 = None
        pow_56: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_10, 2);  relu_10 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_304: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        permute_71: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_304, [1, 0]);  convert_element_type_304 = None
        view_191: "bf16[32768, 3072]" = torch.ops.aten.view.default(pow_56, [32768, 3072]);  pow_56 = None
        mm_71: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_191, permute_71)
        view_192: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_71, [16, 2048, 768]);  mm_71 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:152 in forward, code: x = x + self.mlp(norm(x))
        add_127: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_125, view_192);  view_192 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_22: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 11)
        mul_189: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_22, add_127);  select_22 = None
        select_23: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 11)
        mul_190: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_23, cat);  select_23 = None
        add_128: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:501 in forward, code: ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        embedding_6: "bf16[16, 2048, 768]" = torch.ops.aten.embedding.default(primals_85, primals_1);  primals_85 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_307: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_128, torch.float32)
        pow_57: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_307, 2)
        mean_45: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_57, [2], True);  pow_57 = None
        add_129: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_45, 1.1920928955078125e-07);  mean_45 = None
        rsqrt_45: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_191: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_307, rsqrt_45);  convert_element_type_307 = None
        convert_element_type_308: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_191, torch.bfloat16);  mul_191 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_309: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        permute_72: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_309, [1, 0]);  convert_element_type_309 = None
        view_193: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_308, [32768, 768])
        mm_72: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_193, permute_72)
        view_194: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_72, [16, 2048, 768]);  mm_72 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_195: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_194, [16, 2048, 6, 128]);  view_194 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_312: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        permute_73: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_312, [1, 0]);  convert_element_type_312 = None
        mm_73: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_193, permute_73)
        view_197: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_73, [16, 2048, 768]);  mm_73 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_198: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_197, [16, 2048, 6, 128]);  view_197 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_315: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        permute_74: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_315, [1, 0]);  convert_element_type_315 = None
        mm_74: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_193, permute_74)
        view_200: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_74, [16, 2048, 768]);  mm_74 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_201: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_200, [16, 2048, 6, 128]);  view_200 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_202: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(embedding_6, [16, 2048, 6, 128])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        slice_57: "bf16[16, 2048, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_308, 2, 0, 12);  convert_element_type_308 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_318: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        permute_75: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_318, [1, 0]);  convert_element_type_318 = None
        view_203: "bf16[32768, 12]" = torch.ops.aten.view.default(slice_57, [32768, 12]);  slice_57 = None
        mm_75: "bf16[32768, 6]" = torch.ops.aten.mm.default(view_203, permute_75)
        view_204: "bf16[16, 2048, 6]" = torch.ops.aten.view.default(mm_75, [16, 2048, 6])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        sigmoid_6: "bf16[16, 2048, 6]" = torch.ops.aten.sigmoid.default(view_204);  view_204 = None
        mul_192: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(sigmoid_6, 3);  sigmoid_6 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        unsqueeze_5: "bf16[16, 2048, 6, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
        mul_193: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(unsqueeze_5, view_202);  unsqueeze_5 = view_202 = None
        add_130: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(view_201, mul_193);  view_201 = mul_193 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_58: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_195, 3, 0, 64)
        slice_59: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_195, 3, 64, 9223372036854775807);  view_195 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_194: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_58, slice_1)
        mul_195: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_59, slice_2)
        add_131: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_196: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_58, neg);  slice_58 = None
        mul_197: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_59, slice_1);  slice_59 = None
        add_132: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_23: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_131, add_132], 3);  add_131 = add_132 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_60: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_198, 3, 0, 64)
        slice_61: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(view_198, 3, 64, 9223372036854775807);  view_198 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_198: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_60, slice_1)
        mul_199: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_61, slice_2);  slice_2 = None
        add_133: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_200: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_60, neg);  slice_60 = neg = None
        mul_201: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_61, slice_1);  slice_61 = slice_1 = None
        add_134: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        cat_24: "bf16[16, 2048, 6, 128]" = torch.ops.aten.cat.default([add_133, add_134], 3);  add_133 = add_134 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_321: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_23, torch.float32)
        pow_58: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_321, 2)
        mean_46: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_58, [3], True);  pow_58 = None
        add_135: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_46, 1.1920928955078125e-07);  mean_46 = None
        rsqrt_46: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        mul_202: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_321, rsqrt_46);  convert_element_type_321 = None
        convert_element_type_322: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_202, torch.bfloat16);  mul_202 = None
        convert_element_type_323: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_24, torch.float32)
        pow_59: "f32[16, 2048, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_323, 2)
        mean_47: "f32[16, 2048, 6, 1]" = torch.ops.aten.mean.dim(pow_59, [3], True);  pow_59 = None
        add_136: "f32[16, 2048, 6, 1]" = torch.ops.aten.add.Scalar(mean_47, 1.1920928955078125e-07);  mean_47 = None
        rsqrt_47: "f32[16, 2048, 6, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_203: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_323, rsqrt_47);  convert_element_type_323 = None
        convert_element_type_324: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_203, torch.bfloat16);  mul_203 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_204: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_322, 1.2);  convert_element_type_322 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_205: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_324, 1.2);  convert_element_type_324 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_11 = torch.ops.flash_attn_3._flash_attn_forward.default(mul_204, mul_205, add_130, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0)
        getitem_44: "bf16[16, 2048, 6, 128]" = _flash_attn_forward_11[0]
        getitem_45: "f32[16, 6, 2048]" = _flash_attn_forward_11[1];  _flash_attn_forward_11 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_205: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_44, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_325: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        permute_76: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_325, [1, 0]);  convert_element_type_325 = None
        view_206: "bf16[32768, 768]" = torch.ops.aten.view.default(view_205, [32768, 768]);  view_205 = None
        mm_76: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_206, permute_76);  view_206 = None
        view_207: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_76, [16, 2048, 768]);  mm_76 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:151 in forward, code: x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        add_137: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_128, view_207);  add_128 = view_207 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_328: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_137, torch.float32)
        pow_60: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_328, 2)
        mean_48: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_60, [2], True);  pow_60 = None
        add_138: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_48, 1.1920928955078125e-07);  mean_48 = None
        rsqrt_48: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_206: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_328, rsqrt_48);  convert_element_type_328 = None
        convert_element_type_329: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_206, torch.bfloat16);  mul_206 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_330: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        permute_77: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_330, [1, 0]);  convert_element_type_330 = None
        view_208: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_329, [32768, 768]);  convert_element_type_329 = None
        mm_77: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_208, permute_77)
        view_209: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_77, [16, 2048, 3072])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_11: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_209);  view_209 = None
        pow_61: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_11, 2);  relu_11 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_333: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        permute_78: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_333, [1, 0]);  convert_element_type_333 = None
        view_210: "bf16[32768, 3072]" = torch.ops.aten.view.default(pow_61, [32768, 3072]);  pow_61 = None
        mm_78: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_210, permute_78)
        view_211: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_78, [16, 2048, 768]);  mm_78 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:152 in forward, code: x = x + self.mlp(norm(x))
        add_139: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_137, view_211);  view_211 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:507 in forward, code: x = x - self.backout_lambda.to(x.dtype) * x_backout
        convert_element_type_336: "bf16[1]" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16)
        mul_207: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_336, add_81);  convert_element_type_336 = None
        sub: "bf16[16, 2048, 768]" = torch.ops.aten.sub.Tensor(add_139, mul_207);  add_139 = mul_207 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_337: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(sub, torch.float32)
        pow_62: "f32[16, 2048, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_337, 2)
        mean_49: "f32[16, 2048, 1]" = torch.ops.aten.mean.dim(pow_62, [2], True);  pow_62 = None
        add_140: "f32[16, 2048, 1]" = torch.ops.aten.add.Scalar(mean_49, 1.1920928955078125e-07);  mean_49 = None
        rsqrt_49: "f32[16, 2048, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_208: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_337, rsqrt_49);  convert_element_type_337 = None
        convert_element_type_338: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_208, torch.bfloat16);  mul_208 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_339: "bf16[32768, 768]" = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        permute_79: "bf16[768, 32768]" = torch.ops.aten.permute.default(convert_element_type_339, [1, 0]);  convert_element_type_339 = None
        view_212: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_338, [32768, 768]);  convert_element_type_338 = None
        mm_79: "bf16[32768, 32768]" = torch.ops.aten.mm.default(view_212, permute_79)
        view_213: "bf16[16, 2048, 32768]" = torch.ops.aten.view.default(mm_79, [16, 2048, 32768])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:514 in forward, code: logits = logits.float() # switch to fp32 for logit softcap and loss computation
        convert_element_type_342: "f32[16, 2048, 32768]" = torch.ops.prims.convert_element_type.default(view_213, torch.float32);  view_213 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:515 in forward, code: logits = softcap * torch.tanh(logits / softcap) # squash the logits
        div: "f32[16, 2048, 32768]" = torch.ops.aten.div.Tensor(convert_element_type_342, 15);  convert_element_type_342 = None
        tanh: "f32[16, 2048, 32768]" = torch.ops.aten.tanh.default(div);  div = None
        mul_209: "f32[16, 2048, 32768]" = torch.ops.aten.mul.Tensor(tanh, 15);  tanh = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:520 in forward, code: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        view_214: "f32[32768, 32768]" = torch.ops.aten.view.default(mul_209, [-1, 32768]);  mul_209 = None
        view_215: "i64[32768]" = torch.ops.aten.view.default(primals_95, [-1])
        amax: "f32[32768, 1]" = torch.ops.aten.amax.default(view_214, [1], True)
        sub_1: "f32[32768, 32768]" = torch.ops.aten.sub.Tensor(view_214, amax);  view_214 = None
        exp: "f32[32768, 32768]" = torch.ops.aten.exp.default(sub_1)
        sum_1: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[32768, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_2: "f32[32768, 32768]" = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = None
        ne: "b8[32768]" = torch.ops.aten.ne.Scalar(view_215, -1)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[32768]" = torch.ops.aten.where.self(ne, view_215, full_default);  view_215 = full_default = None
        unsqueeze_6: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[32768, 1]" = torch.ops.aten.gather.default(sub_2, 1, unsqueeze_6);  sub_2 = unsqueeze_6 = None
        squeeze: "f32[32768]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_24: "f32[32768]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32768]" = torch.ops.aten.where.self(ne, neg_24, full_default_1);  neg_24 = full_default_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_343: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_1: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_343);  sum_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_82: "bf16[32768, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_86: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_90: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_94: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_98: "bf16[6, 12]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_102: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_106: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_110: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_114: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_118: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_122: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_126: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_130: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_134: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_138: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_142: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_146: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_150: "bf16[6, 12]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_154: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_158: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_162: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_166: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_170: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_174: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_178: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_182: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_186: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_190: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_194: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_198: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_202: "bf16[6, 12]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_206: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_210: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_214: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_218: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_222: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_226: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_230: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_234: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_238: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_242: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_246: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_250: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_254: "bf16[6, 12]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_258: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_262: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_266: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_270: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_274: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_278: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_282: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_286: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_290: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_294: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_298: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_302: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_306: "bf16[6, 12]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_310: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_314: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_318: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_322: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_326: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_330: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_334: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_338: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_342: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_346: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_350: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_354: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_358: "bf16[6, 12]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_362: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_366: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_370: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_374: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_378: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_382: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_386: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_390: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_394: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        permute_398: "bf16[1, 24]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (div_1, primals_1, primals_2, primals_3, primals_5, primals_7, primals_8, primals_93, primals_95, embedding, rsqrt, view, mm, cat, rsqrt_1, view_2, view_10, cat_1, cat_2, rsqrt_2, rsqrt_3, mul_16, mul_17, getitem, getitem_1, add_10, rsqrt_4, view_14, mm_5, view_16, add_12, embedding_1, rsqrt_5, view_18, view_28, mm_10, add_15, cat_3, cat_4, rsqrt_6, rsqrt_7, mul_34, mul_35, getitem_4, getitem_5, add_22, rsqrt_8, view_33, mm_12, view_35, add_24, rsqrt_9, view_37, view_45, cat_5, cat_6, rsqrt_10, rsqrt_11, mul_50, mul_51, getitem_8, getitem_9, add_33, rsqrt_12, view_49, mm_18, view_51, add_35, embedding_2, rsqrt_13, view_53, view_63, mm_23, add_38, cat_7, cat_8, rsqrt_14, rsqrt_15, mul_68, mul_69, getitem_12, getitem_13, add_45, rsqrt_16, view_68, mm_25, view_70, add_47, rsqrt_17, view_72, view_80, cat_9, cat_10, rsqrt_18, rsqrt_19, mul_84, mul_85, getitem_16, getitem_17, add_56, rsqrt_20, view_84, mm_31, view_86, add_58, embedding_3, rsqrt_21, view_88, view_98, mm_36, add_61, cat_11, cat_12, rsqrt_22, rsqrt_23, mul_102, mul_103, getitem_20, getitem_21, add_68, rsqrt_24, view_103, mm_38, view_105, add_70, rsqrt_25, view_107, view_115, cat_13, cat_14, rsqrt_26, rsqrt_27, mul_118, mul_119, getitem_24, getitem_25, add_79, rsqrt_28, view_119, mm_44, view_121, add_81, embedding_4, rsqrt_29, view_123, view_133, mm_49, add_84, cat_15, cat_16, rsqrt_30, rsqrt_31, mul_136, mul_137, getitem_28, getitem_29, add_91, rsqrt_32, view_138, mm_51, view_140, add_93, rsqrt_33, view_142, view_150, cat_17, cat_18, rsqrt_34, rsqrt_35, mul_152, mul_153, getitem_32, getitem_33, add_102, rsqrt_36, view_154, mm_57, view_156, add_104, embedding_5, rsqrt_37, view_158, view_168, mm_62, add_107, cat_19, cat_20, rsqrt_38, rsqrt_39, mul_170, mul_171, getitem_36, getitem_37, add_114, rsqrt_40, view_173, mm_64, view_175, add_116, rsqrt_41, view_177, view_185, cat_21, cat_22, rsqrt_42, rsqrt_43, mul_186, mul_187, getitem_40, getitem_41, add_125, rsqrt_44, view_189, mm_70, view_191, add_127, embedding_6, rsqrt_45, view_193, view_203, mm_75, add_130, cat_23, cat_24, rsqrt_46, rsqrt_47, mul_204, mul_205, getitem_44, getitem_45, add_137, rsqrt_48, view_208, mm_77, view_210, sub, rsqrt_49, view_212, mm_79, amax, log, convert_element_type_343, permute_82, permute_86, permute_90, permute_94, permute_98, permute_102, permute_106, permute_110, permute_114, permute_118, permute_122, permute_126, permute_130, permute_134, permute_138, permute_142, permute_146, permute_150, permute_154, permute_158, permute_162, permute_166, permute_170, permute_174, permute_178, permute_182, permute_186, permute_190, permute_194, permute_198, permute_202, permute_206, permute_210, permute_214, permute_218, permute_222, permute_226, permute_230, permute_234, permute_238, permute_242, permute_246, permute_250, permute_254, permute_258, permute_262, permute_266, permute_270, permute_274, permute_278, permute_282, permute_286, permute_290, permute_294, permute_298, permute_302, permute_306, permute_310, permute_314, permute_318, permute_322, permute_326, permute_330, permute_334, permute_338, permute_342, permute_346, permute_350, permute_354, permute_358, permute_362, permute_366, permute_370, permute_374, permute_378, permute_382, permute_386, permute_390, permute_394, permute_398)
        