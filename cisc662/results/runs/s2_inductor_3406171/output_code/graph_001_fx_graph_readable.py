class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[16, 2048]", primals_2: "bf16[1, 20480, 1, 64]", primals_3: "bf16[1, 20480, 1, 64]", primals_5: "f32[1]", primals_7: "f32[12]", primals_8: "f32[12]", primals_93: "f32[1]", primals_95: "i64[16, 2048]", embedding: "bf16[16, 2048, 768]", rsqrt: "f32[16, 2048, 1]", view: "bf16[32752, 24]", mm: "bf16[32752, 1]", cat: "bf16[16, 2048, 768]", rsqrt_1: "f32[16, 2048, 1]", view_2: "bf16[32768, 768]", view_10: "bf16[16, 2048, 6, 128]", cat_1: "bf16[16, 2048, 6, 128]", cat_2: "bf16[16, 2048, 6, 128]", rsqrt_2: "f32[16, 2048, 6, 1]", rsqrt_3: "f32[16, 2048, 6, 1]", mul_16: "bf16[16, 2048, 6, 128]", mul_17: "bf16[16, 2048, 6, 128]", getitem: "bf16[16, 2048, 6, 128]", getitem_1: "f32[16, 6, 2048]", add_10: "bf16[16, 2048, 768]", rsqrt_4: "f32[16, 2048, 1]", view_14: "bf16[32768, 768]", mm_5: "bf16[32768, 3072]", view_16: "bf16[32768, 3072]", add_12: "bf16[16, 2048, 768]", embedding_1: "bf16[16, 2048, 768]", rsqrt_5: "f32[16, 2048, 1]", view_18: "bf16[32768, 768]", view_28: "bf16[32768, 12]", mm_10: "bf16[32768, 6]", add_15: "bf16[16, 2048, 6, 128]", cat_3: "bf16[16, 2048, 6, 128]", cat_4: "bf16[16, 2048, 6, 128]", rsqrt_6: "f32[16, 2048, 6, 1]", rsqrt_7: "f32[16, 2048, 6, 1]", mul_34: "bf16[16, 2048, 6, 128]", mul_35: "bf16[16, 2048, 6, 128]", getitem_4: "bf16[16, 2048, 6, 128]", getitem_5: "f32[16, 6, 2048]", add_22: "bf16[16, 2048, 768]", rsqrt_8: "f32[16, 2048, 1]", view_33: "bf16[32768, 768]", mm_12: "bf16[32768, 3072]", view_35: "bf16[32768, 3072]", add_24: "bf16[16, 2048, 768]", rsqrt_9: "f32[16, 2048, 1]", view_37: "bf16[32768, 768]", view_45: "bf16[16, 2048, 6, 128]", cat_5: "bf16[16, 2048, 6, 128]", cat_6: "bf16[16, 2048, 6, 128]", rsqrt_10: "f32[16, 2048, 6, 1]", rsqrt_11: "f32[16, 2048, 6, 1]", mul_50: "bf16[16, 2048, 6, 128]", mul_51: "bf16[16, 2048, 6, 128]", getitem_8: "bf16[16, 2048, 6, 128]", getitem_9: "f32[16, 6, 2048]", add_33: "bf16[16, 2048, 768]", rsqrt_12: "f32[16, 2048, 1]", view_49: "bf16[32768, 768]", mm_18: "bf16[32768, 3072]", view_51: "bf16[32768, 3072]", add_35: "bf16[16, 2048, 768]", embedding_2: "bf16[16, 2048, 768]", rsqrt_13: "f32[16, 2048, 1]", view_53: "bf16[32768, 768]", view_63: "bf16[32768, 12]", mm_23: "bf16[32768, 6]", add_38: "bf16[16, 2048, 6, 128]", cat_7: "bf16[16, 2048, 6, 128]", cat_8: "bf16[16, 2048, 6, 128]", rsqrt_14: "f32[16, 2048, 6, 1]", rsqrt_15: "f32[16, 2048, 6, 1]", mul_68: "bf16[16, 2048, 6, 128]", mul_69: "bf16[16, 2048, 6, 128]", getitem_12: "bf16[16, 2048, 6, 128]", getitem_13: "f32[16, 6, 2048]", add_45: "bf16[16, 2048, 768]", rsqrt_16: "f32[16, 2048, 1]", view_68: "bf16[32768, 768]", mm_25: "bf16[32768, 3072]", view_70: "bf16[32768, 3072]", add_47: "bf16[16, 2048, 768]", rsqrt_17: "f32[16, 2048, 1]", view_72: "bf16[32768, 768]", view_80: "bf16[16, 2048, 6, 128]", cat_9: "bf16[16, 2048, 6, 128]", cat_10: "bf16[16, 2048, 6, 128]", rsqrt_18: "f32[16, 2048, 6, 1]", rsqrt_19: "f32[16, 2048, 6, 1]", mul_84: "bf16[16, 2048, 6, 128]", mul_85: "bf16[16, 2048, 6, 128]", getitem_16: "bf16[16, 2048, 6, 128]", getitem_17: "f32[16, 6, 2048]", add_56: "bf16[16, 2048, 768]", rsqrt_20: "f32[16, 2048, 1]", view_84: "bf16[32768, 768]", mm_31: "bf16[32768, 3072]", view_86: "bf16[32768, 3072]", add_58: "bf16[16, 2048, 768]", embedding_3: "bf16[16, 2048, 768]", rsqrt_21: "f32[16, 2048, 1]", view_88: "bf16[32768, 768]", view_98: "bf16[32768, 12]", mm_36: "bf16[32768, 6]", add_61: "bf16[16, 2048, 6, 128]", cat_11: "bf16[16, 2048, 6, 128]", cat_12: "bf16[16, 2048, 6, 128]", rsqrt_22: "f32[16, 2048, 6, 1]", rsqrt_23: "f32[16, 2048, 6, 1]", mul_102: "bf16[16, 2048, 6, 128]", mul_103: "bf16[16, 2048, 6, 128]", getitem_20: "bf16[16, 2048, 6, 128]", getitem_21: "f32[16, 6, 2048]", add_68: "bf16[16, 2048, 768]", rsqrt_24: "f32[16, 2048, 1]", view_103: "bf16[32768, 768]", mm_38: "bf16[32768, 3072]", view_105: "bf16[32768, 3072]", add_70: "bf16[16, 2048, 768]", rsqrt_25: "f32[16, 2048, 1]", view_107: "bf16[32768, 768]", view_115: "bf16[16, 2048, 6, 128]", cat_13: "bf16[16, 2048, 6, 128]", cat_14: "bf16[16, 2048, 6, 128]", rsqrt_26: "f32[16, 2048, 6, 1]", rsqrt_27: "f32[16, 2048, 6, 1]", mul_118: "bf16[16, 2048, 6, 128]", mul_119: "bf16[16, 2048, 6, 128]", getitem_24: "bf16[16, 2048, 6, 128]", getitem_25: "f32[16, 6, 2048]", add_79: "bf16[16, 2048, 768]", rsqrt_28: "f32[16, 2048, 1]", view_119: "bf16[32768, 768]", mm_44: "bf16[32768, 3072]", view_121: "bf16[32768, 3072]", add_81: "bf16[16, 2048, 768]", embedding_4: "bf16[16, 2048, 768]", rsqrt_29: "f32[16, 2048, 1]", view_123: "bf16[32768, 768]", view_133: "bf16[32768, 12]", mm_49: "bf16[32768, 6]", add_84: "bf16[16, 2048, 6, 128]", cat_15: "bf16[16, 2048, 6, 128]", cat_16: "bf16[16, 2048, 6, 128]", rsqrt_30: "f32[16, 2048, 6, 1]", rsqrt_31: "f32[16, 2048, 6, 1]", mul_136: "bf16[16, 2048, 6, 128]", mul_137: "bf16[16, 2048, 6, 128]", getitem_28: "bf16[16, 2048, 6, 128]", getitem_29: "f32[16, 6, 2048]", add_91: "bf16[16, 2048, 768]", rsqrt_32: "f32[16, 2048, 1]", view_138: "bf16[32768, 768]", mm_51: "bf16[32768, 3072]", view_140: "bf16[32768, 3072]", add_93: "bf16[16, 2048, 768]", rsqrt_33: "f32[16, 2048, 1]", view_142: "bf16[32768, 768]", view_150: "bf16[16, 2048, 6, 128]", cat_17: "bf16[16, 2048, 6, 128]", cat_18: "bf16[16, 2048, 6, 128]", rsqrt_34: "f32[16, 2048, 6, 1]", rsqrt_35: "f32[16, 2048, 6, 1]", mul_152: "bf16[16, 2048, 6, 128]", mul_153: "bf16[16, 2048, 6, 128]", getitem_32: "bf16[16, 2048, 6, 128]", getitem_33: "f32[16, 6, 2048]", add_102: "bf16[16, 2048, 768]", rsqrt_36: "f32[16, 2048, 1]", view_154: "bf16[32768, 768]", mm_57: "bf16[32768, 3072]", view_156: "bf16[32768, 3072]", add_104: "bf16[16, 2048, 768]", embedding_5: "bf16[16, 2048, 768]", rsqrt_37: "f32[16, 2048, 1]", view_158: "bf16[32768, 768]", view_168: "bf16[32768, 12]", mm_62: "bf16[32768, 6]", add_107: "bf16[16, 2048, 6, 128]", cat_19: "bf16[16, 2048, 6, 128]", cat_20: "bf16[16, 2048, 6, 128]", rsqrt_38: "f32[16, 2048, 6, 1]", rsqrt_39: "f32[16, 2048, 6, 1]", mul_170: "bf16[16, 2048, 6, 128]", mul_171: "bf16[16, 2048, 6, 128]", getitem_36: "bf16[16, 2048, 6, 128]", getitem_37: "f32[16, 6, 2048]", add_114: "bf16[16, 2048, 768]", rsqrt_40: "f32[16, 2048, 1]", view_173: "bf16[32768, 768]", mm_64: "bf16[32768, 3072]", view_175: "bf16[32768, 3072]", add_116: "bf16[16, 2048, 768]", rsqrt_41: "f32[16, 2048, 1]", view_177: "bf16[32768, 768]", view_185: "bf16[16, 2048, 6, 128]", cat_21: "bf16[16, 2048, 6, 128]", cat_22: "bf16[16, 2048, 6, 128]", rsqrt_42: "f32[16, 2048, 6, 1]", rsqrt_43: "f32[16, 2048, 6, 1]", mul_186: "bf16[16, 2048, 6, 128]", mul_187: "bf16[16, 2048, 6, 128]", getitem_40: "bf16[16, 2048, 6, 128]", getitem_41: "f32[16, 6, 2048]", add_125: "bf16[16, 2048, 768]", rsqrt_44: "f32[16, 2048, 1]", view_189: "bf16[32768, 768]", mm_70: "bf16[32768, 3072]", view_191: "bf16[32768, 3072]", add_127: "bf16[16, 2048, 768]", embedding_6: "bf16[16, 2048, 768]", rsqrt_45: "f32[16, 2048, 1]", view_193: "bf16[32768, 768]", view_203: "bf16[32768, 12]", mm_75: "bf16[32768, 6]", add_130: "bf16[16, 2048, 6, 128]", cat_23: "bf16[16, 2048, 6, 128]", cat_24: "bf16[16, 2048, 6, 128]", rsqrt_46: "f32[16, 2048, 6, 1]", rsqrt_47: "f32[16, 2048, 6, 1]", mul_204: "bf16[16, 2048, 6, 128]", mul_205: "bf16[16, 2048, 6, 128]", getitem_44: "bf16[16, 2048, 6, 128]", getitem_45: "f32[16, 6, 2048]", add_137: "bf16[16, 2048, 768]", rsqrt_48: "f32[16, 2048, 1]", view_208: "bf16[32768, 768]", mm_77: "bf16[32768, 3072]", view_210: "bf16[32768, 3072]", sub: "bf16[16, 2048, 768]", rsqrt_49: "f32[16, 2048, 1]", view_212: "bf16[32768, 768]", mm_79: "bf16[32768, 32768]", amax: "f32[32768, 1]", log: "f32[32768, 1]", convert_element_type_343: "f32[]", permute_82: "bf16[32768, 768]", permute_86: "bf16[768, 3072]", permute_90: "bf16[3072, 768]", permute_94: "bf16[768, 768]", permute_98: "bf16[6, 12]", permute_102: "bf16[768, 768]", permute_106: "bf16[768, 768]", permute_110: "bf16[768, 768]", permute_114: "bf16[768, 3072]", permute_118: "bf16[3072, 768]", permute_122: "bf16[768, 768]", permute_126: "bf16[768, 768]", permute_130: "bf16[768, 768]", permute_134: "bf16[768, 768]", permute_138: "bf16[768, 3072]", permute_142: "bf16[3072, 768]", permute_146: "bf16[768, 768]", permute_150: "bf16[6, 12]", permute_154: "bf16[768, 768]", permute_158: "bf16[768, 768]", permute_162: "bf16[768, 768]", permute_166: "bf16[768, 3072]", permute_170: "bf16[3072, 768]", permute_174: "bf16[768, 768]", permute_178: "bf16[768, 768]", permute_182: "bf16[768, 768]", permute_186: "bf16[768, 768]", permute_190: "bf16[768, 3072]", permute_194: "bf16[3072, 768]", permute_198: "bf16[768, 768]", permute_202: "bf16[6, 12]", permute_206: "bf16[768, 768]", permute_210: "bf16[768, 768]", permute_214: "bf16[768, 768]", permute_218: "bf16[768, 3072]", permute_222: "bf16[3072, 768]", permute_226: "bf16[768, 768]", permute_230: "bf16[768, 768]", permute_234: "bf16[768, 768]", permute_238: "bf16[768, 768]", permute_242: "bf16[768, 3072]", permute_246: "bf16[3072, 768]", permute_250: "bf16[768, 768]", permute_254: "bf16[6, 12]", permute_258: "bf16[768, 768]", permute_262: "bf16[768, 768]", permute_266: "bf16[768, 768]", permute_270: "bf16[768, 3072]", permute_274: "bf16[3072, 768]", permute_278: "bf16[768, 768]", permute_282: "bf16[768, 768]", permute_286: "bf16[768, 768]", permute_290: "bf16[768, 768]", permute_294: "bf16[768, 3072]", permute_298: "bf16[3072, 768]", permute_302: "bf16[768, 768]", permute_306: "bf16[6, 12]", permute_310: "bf16[768, 768]", permute_314: "bf16[768, 768]", permute_318: "bf16[768, 768]", permute_322: "bf16[768, 3072]", permute_326: "bf16[3072, 768]", permute_330: "bf16[768, 768]", permute_334: "bf16[768, 768]", permute_338: "bf16[768, 768]", permute_342: "bf16[768, 768]", permute_346: "bf16[768, 3072]", permute_350: "bf16[3072, 768]", permute_354: "bf16[768, 768]", permute_358: "bf16[6, 12]", permute_362: "bf16[768, 768]", permute_366: "bf16[768, 768]", permute_370: "bf16[768, 768]", permute_374: "bf16[768, 3072]", permute_378: "bf16[3072, 768]", permute_382: "bf16[768, 768]", permute_386: "bf16[768, 768]", permute_390: "bf16[768, 768]", permute_394: "bf16[768, 768]", permute_398: "bf16[1, 24]", tangents_1: "f32[]"):
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:520 in forward, code: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        div_2: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_343);  tangents_1 = convert_element_type_343 = None
        view_215: "i64[32768]" = torch.ops.aten.view.default(primals_95, [-1]);  primals_95 = None
        unsqueeze_7: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(view_215, 1);  view_215 = None
        ne_3: "b8[32768, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_7, -1)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[32768, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_7, full_default);  unsqueeze_7 = full_default = None
        full_default_3: "f32[32768, 32768]" = torch.ops.aten.full.default([32768, 32768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter: "f32[32768, 32768]" = torch.ops.aten.scatter.value(full_default_3, 1, where_2, -1.0);  full_default_3 = where_2 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[32768, 1]" = torch.ops.aten.where.self(ne_3, div_2, full_default_1);  ne_3 = div_2 = None
        mul_210: "f32[32768, 32768]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_213: "bf16[16, 2048, 32768]" = torch.ops.aten.view.default(mm_79, [16, 2048, 32768]);  mm_79 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:514 in forward, code: logits = logits.float() # switch to fp32 for logit softcap and loss computation
        convert_element_type_342: "f32[16, 2048, 32768]" = torch.ops.prims.convert_element_type.default(view_213, torch.float32);  view_213 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:515 in forward, code: logits = softcap * torch.tanh(logits / softcap) # squash the logits
        div: "f32[16, 2048, 32768]" = torch.ops.aten.div.Tensor(convert_element_type_342, 15);  convert_element_type_342 = None
        tanh: "f32[16, 2048, 32768]" = torch.ops.aten.tanh.default(div);  div = None
        mul_209: "f32[16, 2048, 32768]" = torch.ops.aten.mul.Tensor(tanh, 15)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:520 in forward, code: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        view_214: "f32[32768, 32768]" = torch.ops.aten.view.default(mul_209, [-1, 32768]);  mul_209 = None
        sub_1: "f32[32768, 32768]" = torch.ops.aten.sub.Tensor(view_214, amax);  view_214 = amax = None
        sub_2: "f32[32768, 32768]" = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = log = None
        exp_1: "f32[32768, 32768]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_4: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [1], True)
        mul_211: "f32[32768, 32768]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_3: "f32[32768, 32768]" = torch.ops.aten.sub.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
        view_216: "f32[16, 2048, 32768]" = torch.ops.aten.view.default(sub_3, [16, 2048, 32768]);  sub_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:515 in forward, code: logits = softcap * torch.tanh(logits / softcap) # squash the logits
        mul_212: "f32[16, 2048, 32768]" = torch.ops.aten.mul.Tensor(view_216, 15);  view_216 = None
        mul_213: "f32[16, 2048, 32768]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_4: "f32[16, 2048, 32768]" = torch.ops.aten.sub.Tensor(1, mul_213);  mul_213 = None
        mul_214: "f32[16, 2048, 32768]" = torch.ops.aten.mul.Tensor(mul_212, sub_4);  mul_212 = sub_4 = None
        div_3: "f32[16, 2048, 32768]" = torch.ops.aten.div.Tensor(mul_214, 15);  mul_214 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:514 in forward, code: logits = logits.float() # switch to fp32 for logit softcap and loss computation
        convert_element_type_344: "bf16[16, 2048, 32768]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_217: "bf16[32768, 32768]" = torch.ops.aten.view.default(convert_element_type_344, [32768, 32768]);  convert_element_type_344 = None
        permute_80: "bf16[32768, 32768]" = torch.ops.aten.permute.default(view_217, [1, 0])
        mm_80: "bf16[32768, 768]" = torch.ops.aten.mm.default(permute_80, view_212);  permute_80 = view_212 = None
        mm_81: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_217, permute_82);  view_217 = permute_82 = None
        view_218: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_81, [16, 2048, 768]);  mm_81 = None
        convert_element_type_349: "f32[32768, 768]" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_350: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_218, torch.float32);  view_218 = None
        convert_element_type_337: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(sub, torch.float32);  sub = None
        mul_208: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_337, rsqrt_49);  convert_element_type_337 = None
        mul_216: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_208, convert_element_type_350)
        sum_5: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_216, [2], True);  mul_216 = None
        div_4: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_208, 768);  mul_208 = None
        mul_217: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_4, sum_5);  div_4 = sum_5 = None
        sub_5: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_350, mul_217);  convert_element_type_350 = mul_217 = None
        mul_218: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_49);  sub_5 = rsqrt_49 = None
        convert_element_type_352: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_218, torch.bfloat16);  mul_218 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:507 in forward, code: x = x - self.backout_lambda.to(x.dtype) * x_backout
        neg_25: "bf16[16, 2048, 768]" = torch.ops.aten.neg.default(convert_element_type_352)
        convert_element_type_336: "bf16[1]" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        mul_219: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(neg_25, convert_element_type_336);  convert_element_type_336 = None
        mul_220: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(neg_25, add_81);  neg_25 = None
        sum_6: "bf16[1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1, 2], True);  mul_220 = None
        view_219: "bf16[1]" = torch.ops.aten.view.default(sum_6, [1]);  sum_6 = None
        convert_element_type_353: "f32[1]" = torch.ops.prims.convert_element_type.default(view_219, torch.float32);  view_219 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_220: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_352, [32768, 768])
        permute_84: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_220, [1, 0])
        mm_82: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_84, view_210);  permute_84 = view_210 = None
        mm_83: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_220, permute_86);  view_220 = permute_86 = None
        view_221: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_83, [16, 2048, 3072]);  mm_83 = None
        convert_element_type_358: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_209: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_77, [16, 2048, 3072]);  mm_77 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_11: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_209);  view_209 = None
        pow_63: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_11, 1.0)
        mul_221: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Scalar(pow_63, 2.0);  pow_63 = None
        mul_222: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Tensor(view_221, mul_221);  view_221 = mul_221 = None
        le: "b8[16, 2048, 3072]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        full_default_5: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[16, 2048, 3072]" = torch.ops.aten.where.self(le, full_default_5, mul_222);  le = mul_222 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_222: "bf16[32768, 3072]" = torch.ops.aten.view.default(where_4, [32768, 3072]);  where_4 = None
        permute_88: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_222, [1, 0])
        mm_84: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_88, view_208);  permute_88 = view_208 = None
        mm_85: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_222, permute_90);  view_222 = permute_90 = None
        view_223: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_85, [16, 2048, 768]);  mm_85 = None
        convert_element_type_363: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_364: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_223, torch.float32);  view_223 = None
        convert_element_type_328: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_137, torch.float32);  add_137 = None
        mul_206: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_328, rsqrt_48);  convert_element_type_328 = None
        mul_224: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_206, convert_element_type_364)
        sum_7: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
        div_5: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_206, 768);  mul_206 = None
        mul_225: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_5, sum_7);  div_5 = sum_7 = None
        sub_6: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_364, mul_225);  convert_element_type_364 = mul_225 = None
        mul_226: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_48);  sub_6 = rsqrt_48 = None
        convert_element_type_366: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_226, torch.bfloat16);  mul_226 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_141: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(convert_element_type_352, convert_element_type_366);  convert_element_type_352 = convert_element_type_366 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_224: "bf16[32768, 768]" = torch.ops.aten.view.default(add_141, [32768, 768])
        permute_92: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_224, [1, 0])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_205: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_44, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_206: "bf16[32768, 768]" = torch.ops.aten.view.default(view_205, [32768, 768]);  view_205 = None
        mm_86: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_92, view_206);  permute_92 = view_206 = None
        mm_87: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_224, permute_94);  view_224 = permute_94 = None
        view_225: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_87, [16, 2048, 768]);  mm_87 = None
        convert_element_type_371: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_86, torch.float32);  mm_86 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_226: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_225, [16, 2048, 6, 128]);  view_225 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_backward = torch.ops.flash_attn_3._flash_attn_backward.default(view_226, mul_204, mul_205, add_130, getitem_44, getitem_45, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0);  view_226 = mul_204 = mul_205 = add_130 = getitem_44 = getitem_45 = None
        getitem_48: "bf16[16, 2048, 6, 128]" = _flash_attn_backward[0]
        getitem_49: "bf16[16, 2048, 6, 128]" = _flash_attn_backward[1]
        getitem_50: "bf16[16, 2048, 6, 128]" = _flash_attn_backward[2];  _flash_attn_backward = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_227: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_49, 1.2);  getitem_49 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_228: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_48, 1.2);  getitem_48 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_372: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_227, torch.float32);  mul_227 = None
        convert_element_type_323: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_24, torch.float32);  cat_24 = None
        mul_203: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_323, rsqrt_47);  convert_element_type_323 = None
        mul_230: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_203, convert_element_type_372)
        sum_8: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_230, [3], True);  mul_230 = None
        div_6: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_203, 128);  mul_203 = None
        mul_231: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_6, sum_8);  div_6 = sum_8 = None
        sub_7: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_372, mul_231);  convert_element_type_372 = mul_231 = None
        mul_232: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_47);  sub_7 = rsqrt_47 = None
        convert_element_type_374: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_232, torch.bfloat16);  mul_232 = None
        convert_element_type_375: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_228, torch.float32);  mul_228 = None
        convert_element_type_321: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_23, torch.float32);  cat_23 = None
        mul_202: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_321, rsqrt_46);  convert_element_type_321 = None
        mul_234: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_202, convert_element_type_375)
        sum_9: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [3], True);  mul_234 = None
        div_7: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_202, 128);  mul_202 = None
        mul_235: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_7, sum_9);  div_7 = sum_9 = None
        sub_8: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_375, mul_235);  convert_element_type_375 = mul_235 = None
        mul_236: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_46);  sub_8 = rsqrt_46 = None
        convert_element_type_377: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_236, torch.bfloat16);  mul_236 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_62: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_374, 3, 0, 64)
        slice_63: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_374, 3, 64, 128);  convert_element_type_374 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:468 in forward, code: cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length
        slice_1: "bf16[1, 2048, 1, 64]" = torch.ops.aten.slice.Tensor(primals_2, 1, 0, 2048);  primals_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_237: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_63, slice_1)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:468 in forward, code: cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length
        slice_2: "bf16[1, 2048, 1, 64]" = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 2048);  primals_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        neg: "bf16[1, 2048, 1, 64]" = torch.ops.aten.neg.default(slice_2)
        mul_238: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_63, neg);  slice_63 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_239: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_62, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_142: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_237, mul_239);  mul_237 = mul_239 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_240: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_62, slice_1);  slice_62 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_143: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_238, mul_240);  mul_238 = mul_240 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        full_default_6: "bf16[16, 2048, 6, 128]" = torch.ops.aten.full.default([16, 2048, 6, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_142, 3, 64, 9223372036854775807);  add_142 = None
        slice_scatter_1: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_143, 3, 0, 64);  add_143 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_144: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_64: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_377, 3, 0, 64)
        slice_65: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_377, 3, 64, 128);  convert_element_type_377 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_241: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_65, slice_1)
        mul_242: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_65, neg);  slice_65 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_243: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_64, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_145: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_241, mul_243);  mul_241 = mul_243 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_244: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_64, slice_1);  slice_64 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_146: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_242, mul_244);  mul_242 = mul_244 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_2: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_145, 3, 64, 9223372036854775807);  add_145 = None
        slice_scatter_3: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_146, 3, 0, 64);  add_146 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_147: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_204: "bf16[16, 2048, 6]" = torch.ops.aten.view.default(mm_75, [16, 2048, 6]);  mm_75 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        sigmoid_6: "bf16[16, 2048, 6]" = torch.ops.aten.sigmoid.default(view_204);  view_204 = None
        mul_192: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(sigmoid_6, 3)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        unsqueeze_5: "bf16[16, 2048, 6, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
        mul_245: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_50, unsqueeze_5);  unsqueeze_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_202: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(embedding_6, [16, 2048, 6, 128]);  embedding_6 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        mul_246: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_50, view_202);  view_202 = None
        sum_10: "bf16[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_246, [3], True);  mul_246 = None
        squeeze_1: "bf16[16, 2048, 6]" = torch.ops.aten.squeeze.dim(sum_10, -1);  sum_10 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        mul_247: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(squeeze_1, 3);  squeeze_1 = None
        convert_element_type_378: "f32[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(mul_247, torch.float32);  mul_247 = None
        convert_element_type_379: "f32[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_6, torch.float32);  sigmoid_6 = None
        sub_9: "f32[16, 2048, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_379)
        mul_248: "f32[16, 2048, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_379, sub_9);  convert_element_type_379 = sub_9 = None
        mul_249: "f32[16, 2048, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_378, mul_248);  convert_element_type_378 = mul_248 = None
        convert_element_type_380: "bf16[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(mul_249, torch.bfloat16);  mul_249 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_227: "bf16[32768, 6]" = torch.ops.aten.view.default(convert_element_type_380, [32768, 6]);  convert_element_type_380 = None
        permute_96: "bf16[6, 32768]" = torch.ops.aten.permute.default(view_227, [1, 0])
        mm_88: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_96, view_203);  permute_96 = view_203 = None
        mm_89: "bf16[32768, 12]" = torch.ops.aten.mm.default(view_227, permute_98);  view_227 = permute_98 = None
        view_228: "bf16[16, 2048, 12]" = torch.ops.aten.view.default(mm_89, [16, 2048, 12]);  mm_89 = None
        convert_element_type_385: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_88, torch.float32);  mm_88 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        full_default_10: "bf16[16, 2048, 768]" = torch.ops.aten.full.default([16, 2048, 768], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_4: "bf16[16, 2048, 768]" = torch.ops.aten.slice_scatter.default(full_default_10, view_228, 2, 0, 12);  view_228 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_229: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mul_245, [16, 2048, 768]);  mul_245 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_230: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_50, [16, 2048, 768]);  getitem_50 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_231: "bf16[32768, 768]" = torch.ops.aten.view.default(view_230, [32768, 768]);  view_230 = None
        permute_100: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_231, [1, 0])
        mm_90: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_100, view_193);  permute_100 = None
        mm_91: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_231, permute_102);  view_231 = permute_102 = None
        view_232: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_91, [16, 2048, 768]);  mm_91 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_148: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(slice_scatter_4, view_232);  slice_scatter_4 = view_232 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_390: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_233: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_144, [16, 2048, 768]);  add_144 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_234: "bf16[32768, 768]" = torch.ops.aten.view.default(view_233, [32768, 768]);  view_233 = None
        permute_104: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_234, [1, 0])
        mm_92: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_104, view_193);  permute_104 = None
        mm_93: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_234, permute_106);  view_234 = permute_106 = None
        view_235: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_93, [16, 2048, 768]);  mm_93 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_149: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_148, view_235);  add_148 = view_235 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_395: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_236: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_147, [16, 2048, 768]);  add_147 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_237: "bf16[32768, 768]" = torch.ops.aten.view.default(view_236, [32768, 768]);  view_236 = None
        permute_108: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_237, [1, 0])
        mm_94: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_108, view_193);  permute_108 = view_193 = None
        mm_95: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_237, permute_110);  view_237 = permute_110 = None
        view_238: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_95, [16, 2048, 768]);  mm_95 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_150: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_149, view_238);  add_149 = view_238 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_400: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_401: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_150, torch.float32);  add_150 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_22: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 11)
        mul_189: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_22, add_127)
        select_23: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 11)
        mul_190: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_23, cat)
        add_128: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_307: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_128, torch.float32);  add_128 = None
        mul_191: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_307, rsqrt_45);  convert_element_type_307 = None
        mul_251: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_191, convert_element_type_401)
        sum_11: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True);  mul_251 = None
        div_8: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_191, 768);  mul_191 = None
        mul_252: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_8, sum_11);  div_8 = sum_11 = None
        sub_10: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_401, mul_252);  convert_element_type_401 = mul_252 = None
        mul_253: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_45);  sub_10 = rsqrt_45 = None
        convert_element_type_403: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_253, torch.bfloat16);  mul_253 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_151: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_141, convert_element_type_403);  add_141 = convert_element_type_403 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:501 in forward, code: ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        convert_element_type_404: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_229, torch.float32);  view_229 = None
        eq: "b8[16, 2048]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_8: "b8[16, 2048, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        where_5: "f32[16, 2048, 768]" = torch.ops.aten.where.self(unsqueeze_8, full_default_1, convert_element_type_404);  convert_element_type_404 = None
        full_default_12: "f32[32768, 768]" = torch.ops.aten.full.default([32768, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[32768, 768]" = torch.ops.aten.index_put.default(full_default_12, [primals_1], where_5, True);  where_5 = None
        convert_element_type_405: "bf16[32768, 768]" = torch.ops.prims.convert_element_type.default(index_put, torch.bfloat16);  index_put = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_254: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_151, select_23);  select_23 = None
        mul_255: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_151, cat)
        sum_12: "bf16[]" = torch.ops.aten.sum.default(mul_255);  mul_255 = None
        convert_element_type_406: "f32[]" = torch.ops.prims.convert_element_type.default(sum_12, torch.float32);  sum_12 = None
        full_default_13: "f32[12]" = torch.ops.aten.full.default([12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_406, 0, 11);  convert_element_type_406 = None
        mul_256: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_151, select_22);  select_22 = None
        mul_257: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_151, add_127);  add_151 = add_127 = None
        sum_13: "bf16[]" = torch.ops.aten.sum.default(mul_257);  mul_257 = None
        convert_element_type_407: "f32[]" = torch.ops.prims.convert_element_type.default(sum_13, torch.float32);  sum_13 = None
        select_scatter_1: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_407, 0, 11);  convert_element_type_407 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_239: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_256, [32768, 768])
        permute_112: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_239, [1, 0])
        mm_96: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_112, view_191);  permute_112 = view_191 = None
        mm_97: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_239, permute_114);  view_239 = permute_114 = None
        view_240: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_97, [16, 2048, 3072]);  mm_97 = None
        convert_element_type_412: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_190: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_70, [16, 2048, 3072]);  mm_70 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_10: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_190);  view_190 = None
        pow_64: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_10, 1.0)
        mul_258: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Scalar(pow_64, 2.0);  pow_64 = None
        mul_259: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Tensor(view_240, mul_258);  view_240 = mul_258 = None
        le_1: "b8[16, 2048, 3072]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_6: "bf16[16, 2048, 3072]" = torch.ops.aten.where.self(le_1, full_default_5, mul_259);  le_1 = mul_259 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_241: "bf16[32768, 3072]" = torch.ops.aten.view.default(where_6, [32768, 3072]);  where_6 = None
        permute_116: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_241, [1, 0])
        mm_98: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_116, view_189);  permute_116 = view_189 = None
        mm_99: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_241, permute_118);  view_241 = permute_118 = None
        view_242: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_99, [16, 2048, 768]);  mm_99 = None
        convert_element_type_417: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_98, torch.float32);  mm_98 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_418: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_242, torch.float32);  view_242 = None
        convert_element_type_299: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_125, torch.float32);  add_125 = None
        mul_188: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_299, rsqrt_44);  convert_element_type_299 = None
        mul_261: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_188, convert_element_type_418)
        sum_14: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [2], True);  mul_261 = None
        div_9: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_188, 768);  mul_188 = None
        mul_262: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_9, sum_14);  div_9 = sum_14 = None
        sub_11: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_418, mul_262);  convert_element_type_418 = mul_262 = None
        mul_263: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_44);  sub_11 = rsqrt_44 = None
        convert_element_type_420: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_263, torch.bfloat16);  mul_263 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_152: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_256, convert_element_type_420);  mul_256 = convert_element_type_420 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_243: "bf16[32768, 768]" = torch.ops.aten.view.default(add_152, [32768, 768])
        permute_120: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_243, [1, 0])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_186: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_40, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_187: "bf16[32768, 768]" = torch.ops.aten.view.default(view_186, [32768, 768]);  view_186 = None
        mm_100: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_120, view_187);  permute_120 = view_187 = None
        mm_101: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_243, permute_122);  view_243 = permute_122 = None
        view_244: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_101, [16, 2048, 768]);  mm_101 = None
        convert_element_type_425: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_100, torch.float32);  mm_100 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_245: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_244, [16, 2048, 6, 128]);  view_244 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_backward_1 = torch.ops.flash_attn_3._flash_attn_backward.default(view_245, mul_186, mul_187, view_185, getitem_40, getitem_41, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0);  view_245 = mul_186 = mul_187 = view_185 = getitem_40 = getitem_41 = None
        getitem_52: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_1[0]
        getitem_53: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_1[1]
        getitem_54: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_1[2];  _flash_attn_backward_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_264: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_53, 1.2);  getitem_53 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_265: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_52, 1.2);  getitem_52 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_426: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_264, torch.float32);  mul_264 = None
        convert_element_type_294: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_22, torch.float32);  cat_22 = None
        mul_185: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_294, rsqrt_43);  convert_element_type_294 = None
        mul_267: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_185, convert_element_type_426)
        sum_15: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [3], True);  mul_267 = None
        div_10: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_185, 128);  mul_185 = None
        mul_268: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_10, sum_15);  div_10 = sum_15 = None
        sub_12: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_426, mul_268);  convert_element_type_426 = mul_268 = None
        mul_269: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_43);  sub_12 = rsqrt_43 = None
        convert_element_type_428: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16);  mul_269 = None
        convert_element_type_429: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_265, torch.float32);  mul_265 = None
        convert_element_type_292: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_21, torch.float32);  cat_21 = None
        mul_184: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_292, rsqrt_42);  convert_element_type_292 = None
        mul_271: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_184, convert_element_type_429)
        sum_16: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [3], True);  mul_271 = None
        div_11: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_184, 128);  mul_184 = None
        mul_272: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_11, sum_16);  div_11 = sum_16 = None
        sub_13: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_429, mul_272);  convert_element_type_429 = mul_272 = None
        mul_273: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_42);  sub_13 = rsqrt_42 = None
        convert_element_type_431: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_273, torch.bfloat16);  mul_273 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_66: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_428, 3, 0, 64)
        slice_67: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_428, 3, 64, 128);  convert_element_type_428 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_274: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_67, slice_1)
        mul_275: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_67, neg);  slice_67 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_276: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_66, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_153: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_274, mul_276);  mul_274 = mul_276 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_277: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_66, slice_1);  slice_66 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_154: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_275, mul_277);  mul_275 = mul_277 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_5: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_153, 3, 64, 9223372036854775807);  add_153 = None
        slice_scatter_6: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_154, 3, 0, 64);  add_154 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_155: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_5, slice_scatter_6);  slice_scatter_5 = slice_scatter_6 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_68: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_431, 3, 0, 64)
        slice_69: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_431, 3, 64, 128);  convert_element_type_431 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_278: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_69, slice_1)
        mul_279: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_69, neg);  slice_69 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_280: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_68, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_156: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_278, mul_280);  mul_278 = mul_280 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_281: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_68, slice_1);  slice_68 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_157: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_279, mul_281);  mul_279 = mul_281 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_7: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_156, 3, 64, 9223372036854775807);  add_156 = None
        slice_scatter_8: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_157, 3, 0, 64);  add_157 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_158: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_7, slice_scatter_8);  slice_scatter_7 = slice_scatter_8 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_246: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_54, [16, 2048, 768]);  getitem_54 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_247: "bf16[32768, 768]" = torch.ops.aten.view.default(view_246, [32768, 768]);  view_246 = None
        permute_124: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_247, [1, 0])
        mm_102: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_124, view_177);  permute_124 = None
        mm_103: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_247, permute_126);  view_247 = permute_126 = None
        view_248: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_103, [16, 2048, 768]);  mm_103 = None
        convert_element_type_436: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_249: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_155, [16, 2048, 768]);  add_155 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_250: "bf16[32768, 768]" = torch.ops.aten.view.default(view_249, [32768, 768]);  view_249 = None
        permute_128: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_250, [1, 0])
        mm_104: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_128, view_177);  permute_128 = None
        mm_105: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_250, permute_130);  view_250 = permute_130 = None
        view_251: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_105, [16, 2048, 768]);  mm_105 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_159: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(view_248, view_251);  view_248 = view_251 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_441: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_252: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_158, [16, 2048, 768]);  add_158 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_253: "bf16[32768, 768]" = torch.ops.aten.view.default(view_252, [32768, 768]);  view_252 = None
        permute_132: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_253, [1, 0])
        mm_106: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_132, view_177);  permute_132 = view_177 = None
        mm_107: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_253, permute_134);  view_253 = permute_134 = None
        view_254: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_107, [16, 2048, 768]);  mm_107 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_160: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_159, view_254);  add_159 = view_254 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_446: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_447: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_160, torch.float32);  add_160 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_20: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 10)
        mul_173: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_20, add_116)
        select_21: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 10)
        mul_174: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_21, cat)
        add_117: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_281: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_117, torch.float32);  add_117 = None
        mul_175: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_281, rsqrt_41);  convert_element_type_281 = None
        mul_283: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_175, convert_element_type_447)
        sum_17: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True);  mul_283 = None
        div_12: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_175, 768);  mul_175 = None
        mul_284: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_12, sum_17);  div_12 = sum_17 = None
        sub_14: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_447, mul_284);  convert_element_type_447 = mul_284 = None
        mul_285: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_41);  sub_14 = rsqrt_41 = None
        convert_element_type_449: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_285, torch.bfloat16);  mul_285 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_161: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_152, convert_element_type_449);  add_152 = convert_element_type_449 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_286: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_161, select_21);  select_21 = None
        mul_287: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_161, cat)
        sum_18: "bf16[]" = torch.ops.aten.sum.default(mul_287);  mul_287 = None
        convert_element_type_450: "f32[]" = torch.ops.prims.convert_element_type.default(sum_18, torch.float32);  sum_18 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_162: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_254, mul_286);  mul_254 = mul_286 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_scatter_2: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_450, 0, 10);  convert_element_type_450 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_163: "f32[12]" = torch.ops.aten.add.Tensor(select_scatter, select_scatter_2);  select_scatter = select_scatter_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_288: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_161, select_20);  select_20 = None
        mul_289: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_161, add_116);  add_161 = add_116 = None
        sum_19: "bf16[]" = torch.ops.aten.sum.default(mul_289);  mul_289 = None
        convert_element_type_451: "f32[]" = torch.ops.prims.convert_element_type.default(sum_19, torch.float32);  sum_19 = None
        select_scatter_3: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_451, 0, 10);  convert_element_type_451 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_164: "f32[12]" = torch.ops.aten.add.Tensor(select_scatter_1, select_scatter_3);  select_scatter_1 = select_scatter_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_255: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_288, [32768, 768])
        permute_136: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_255, [1, 0])
        mm_108: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_136, view_175);  permute_136 = view_175 = None
        mm_109: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_255, permute_138);  view_255 = permute_138 = None
        view_256: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_109, [16, 2048, 3072]);  mm_109 = None
        convert_element_type_456: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_174: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_64, [16, 2048, 3072]);  mm_64 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_9: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_174);  view_174 = None
        pow_65: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_9, 1.0)
        mul_290: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Scalar(pow_65, 2.0);  pow_65 = None
        mul_291: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Tensor(view_256, mul_290);  view_256 = mul_290 = None
        le_2: "b8[16, 2048, 3072]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_7: "bf16[16, 2048, 3072]" = torch.ops.aten.where.self(le_2, full_default_5, mul_291);  le_2 = mul_291 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_257: "bf16[32768, 3072]" = torch.ops.aten.view.default(where_7, [32768, 3072]);  where_7 = None
        permute_140: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_257, [1, 0])
        mm_110: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_140, view_173);  permute_140 = view_173 = None
        mm_111: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_257, permute_142);  view_257 = permute_142 = None
        view_258: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_111, [16, 2048, 768]);  mm_111 = None
        convert_element_type_461: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_110, torch.float32);  mm_110 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_462: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_258, torch.float32);  view_258 = None
        convert_element_type_273: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_114, torch.float32);  add_114 = None
        mul_172: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_273, rsqrt_40);  convert_element_type_273 = None
        mul_293: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_172, convert_element_type_462)
        sum_20: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True);  mul_293 = None
        div_13: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_172, 768);  mul_172 = None
        mul_294: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_13, sum_20);  div_13 = sum_20 = None
        sub_15: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_462, mul_294);  convert_element_type_462 = mul_294 = None
        mul_295: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_40);  sub_15 = rsqrt_40 = None
        convert_element_type_464: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_295, torch.bfloat16);  mul_295 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_165: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_288, convert_element_type_464);  mul_288 = convert_element_type_464 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_259: "bf16[32768, 768]" = torch.ops.aten.view.default(add_165, [32768, 768])
        permute_144: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_259, [1, 0])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_170: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_36, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_171: "bf16[32768, 768]" = torch.ops.aten.view.default(view_170, [32768, 768]);  view_170 = None
        mm_112: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_144, view_171);  permute_144 = view_171 = None
        mm_113: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_259, permute_146);  view_259 = permute_146 = None
        view_260: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_113, [16, 2048, 768]);  mm_113 = None
        convert_element_type_469: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_112, torch.float32);  mm_112 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_261: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_260, [16, 2048, 6, 128]);  view_260 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_backward_2 = torch.ops.flash_attn_3._flash_attn_backward.default(view_261, mul_170, mul_171, add_107, getitem_36, getitem_37, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0);  view_261 = mul_170 = mul_171 = add_107 = getitem_36 = getitem_37 = None
        getitem_56: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_2[0]
        getitem_57: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_2[1]
        getitem_58: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_2[2];  _flash_attn_backward_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_296: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_57, 1.2);  getitem_57 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_297: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_56, 1.2);  getitem_56 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_470: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_296, torch.float32);  mul_296 = None
        convert_element_type_268: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_20, torch.float32);  cat_20 = None
        mul_169: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_268, rsqrt_39);  convert_element_type_268 = None
        mul_299: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_169, convert_element_type_470)
        sum_21: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [3], True);  mul_299 = None
        div_14: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_169, 128);  mul_169 = None
        mul_300: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_14, sum_21);  div_14 = sum_21 = None
        sub_16: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_470, mul_300);  convert_element_type_470 = mul_300 = None
        mul_301: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_39);  sub_16 = rsqrt_39 = None
        convert_element_type_472: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_301, torch.bfloat16);  mul_301 = None
        convert_element_type_473: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_297, torch.float32);  mul_297 = None
        convert_element_type_266: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_19, torch.float32);  cat_19 = None
        mul_168: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_266, rsqrt_38);  convert_element_type_266 = None
        mul_303: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_168, convert_element_type_473)
        sum_22: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [3], True);  mul_303 = None
        div_15: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_168, 128);  mul_168 = None
        mul_304: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_15, sum_22);  div_15 = sum_22 = None
        sub_17: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_473, mul_304);  convert_element_type_473 = mul_304 = None
        mul_305: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_38);  sub_17 = rsqrt_38 = None
        convert_element_type_475: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_305, torch.bfloat16);  mul_305 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_70: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_472, 3, 0, 64)
        slice_71: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_472, 3, 64, 128);  convert_element_type_472 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_306: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_71, slice_1)
        mul_307: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_71, neg);  slice_71 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_308: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_70, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_166: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_306, mul_308);  mul_306 = mul_308 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_309: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_70, slice_1);  slice_70 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_167: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_307, mul_309);  mul_307 = mul_309 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_9: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_166, 3, 64, 9223372036854775807);  add_166 = None
        slice_scatter_10: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_167, 3, 0, 64);  add_167 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_168: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_9, slice_scatter_10);  slice_scatter_9 = slice_scatter_10 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_72: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_475, 3, 0, 64)
        slice_73: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_475, 3, 64, 128);  convert_element_type_475 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_310: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_73, slice_1)
        mul_311: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_73, neg);  slice_73 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_312: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_72, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_169: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_310, mul_312);  mul_310 = mul_312 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_313: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_72, slice_1);  slice_72 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_170: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_311, mul_313);  mul_311 = mul_313 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_11: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_169, 3, 64, 9223372036854775807);  add_169 = None
        slice_scatter_12: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_170, 3, 0, 64);  add_170 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_171: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_11, slice_scatter_12);  slice_scatter_11 = slice_scatter_12 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_169: "bf16[16, 2048, 6]" = torch.ops.aten.view.default(mm_62, [16, 2048, 6]);  mm_62 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        sigmoid_5: "bf16[16, 2048, 6]" = torch.ops.aten.sigmoid.default(view_169);  view_169 = None
        mul_158: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(sigmoid_5, 3)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        unsqueeze_4: "bf16[16, 2048, 6, 1]" = torch.ops.aten.unsqueeze.default(mul_158, -1);  mul_158 = None
        mul_314: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_58, unsqueeze_4);  unsqueeze_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_167: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(embedding_5, [16, 2048, 6, 128]);  embedding_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        mul_315: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_58, view_167);  view_167 = None
        sum_23: "bf16[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [3], True);  mul_315 = None
        squeeze_2: "bf16[16, 2048, 6]" = torch.ops.aten.squeeze.dim(sum_23, -1);  sum_23 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        mul_316: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(squeeze_2, 3);  squeeze_2 = None
        convert_element_type_476: "f32[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(mul_316, torch.float32);  mul_316 = None
        convert_element_type_477: "f32[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_5, torch.float32);  sigmoid_5 = None
        sub_18: "f32[16, 2048, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_477)
        mul_317: "f32[16, 2048, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_477, sub_18);  convert_element_type_477 = sub_18 = None
        mul_318: "f32[16, 2048, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_476, mul_317);  convert_element_type_476 = mul_317 = None
        convert_element_type_478: "bf16[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(mul_318, torch.bfloat16);  mul_318 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_262: "bf16[32768, 6]" = torch.ops.aten.view.default(convert_element_type_478, [32768, 6]);  convert_element_type_478 = None
        permute_148: "bf16[6, 32768]" = torch.ops.aten.permute.default(view_262, [1, 0])
        mm_114: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_148, view_168);  permute_148 = view_168 = None
        mm_115: "bf16[32768, 12]" = torch.ops.aten.mm.default(view_262, permute_150);  view_262 = permute_150 = None
        view_263: "bf16[16, 2048, 12]" = torch.ops.aten.view.default(mm_115, [16, 2048, 12]);  mm_115 = None
        convert_element_type_483: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_114, torch.float32);  mm_114 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        slice_scatter_13: "bf16[16, 2048, 768]" = torch.ops.aten.slice_scatter.default(full_default_10, view_263, 2, 0, 12);  view_263 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_264: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mul_314, [16, 2048, 768]);  mul_314 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_265: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_58, [16, 2048, 768]);  getitem_58 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_266: "bf16[32768, 768]" = torch.ops.aten.view.default(view_265, [32768, 768]);  view_265 = None
        permute_152: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_266, [1, 0])
        mm_116: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_152, view_158);  permute_152 = None
        mm_117: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_266, permute_154);  view_266 = permute_154 = None
        view_267: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_117, [16, 2048, 768]);  mm_117 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_172: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(slice_scatter_13, view_267);  slice_scatter_13 = view_267 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_488: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_268: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_168, [16, 2048, 768]);  add_168 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_269: "bf16[32768, 768]" = torch.ops.aten.view.default(view_268, [32768, 768]);  view_268 = None
        permute_156: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_269, [1, 0])
        mm_118: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_156, view_158);  permute_156 = None
        mm_119: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_269, permute_158);  view_269 = permute_158 = None
        view_270: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_119, [16, 2048, 768]);  mm_119 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_173: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_172, view_270);  add_172 = view_270 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_493: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_271: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_171, [16, 2048, 768]);  add_171 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_272: "bf16[32768, 768]" = torch.ops.aten.view.default(view_271, [32768, 768]);  view_271 = None
        permute_160: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_272, [1, 0])
        mm_120: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_160, view_158);  permute_160 = view_158 = None
        mm_121: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_272, permute_162);  view_272 = permute_162 = None
        view_273: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_121, [16, 2048, 768]);  mm_121 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_174: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_173, view_273);  add_173 = view_273 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_498: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_499: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_174, torch.float32);  add_174 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_18: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 9)
        mul_155: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_18, add_104)
        select_19: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 9)
        mul_156: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_19, cat)
        add_105: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_252: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_105, torch.float32);  add_105 = None
        mul_157: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_252, rsqrt_37);  convert_element_type_252 = None
        mul_320: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_157, convert_element_type_499)
        sum_24: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
        div_16: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_157, 768);  mul_157 = None
        mul_321: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_16, sum_24);  div_16 = sum_24 = None
        sub_19: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_499, mul_321);  convert_element_type_499 = mul_321 = None
        mul_322: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_37);  sub_19 = rsqrt_37 = None
        convert_element_type_501: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_322, torch.bfloat16);  mul_322 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_175: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_165, convert_element_type_501);  add_165 = convert_element_type_501 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:501 in forward, code: ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        convert_element_type_502: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_264, torch.float32);  view_264 = None
        where_8: "f32[16, 2048, 768]" = torch.ops.aten.where.self(unsqueeze_8, full_default_1, convert_element_type_502);  convert_element_type_502 = None
        index_put_1: "f32[32768, 768]" = torch.ops.aten.index_put.default(full_default_12, [primals_1], where_8, True);  where_8 = None
        convert_element_type_503: "bf16[32768, 768]" = torch.ops.prims.convert_element_type.default(index_put_1, torch.bfloat16);  index_put_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_323: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_175, select_19);  select_19 = None
        mul_324: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_175, cat)
        sum_25: "bf16[]" = torch.ops.aten.sum.default(mul_324);  mul_324 = None
        convert_element_type_504: "f32[]" = torch.ops.prims.convert_element_type.default(sum_25, torch.float32);  sum_25 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_176: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_162, mul_323);  add_162 = mul_323 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_scatter_4: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_504, 0, 9);  convert_element_type_504 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_177: "f32[12]" = torch.ops.aten.add.Tensor(add_163, select_scatter_4);  add_163 = select_scatter_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_325: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_175, select_18);  select_18 = None
        mul_326: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_175, add_104);  add_175 = add_104 = None
        sum_26: "bf16[]" = torch.ops.aten.sum.default(mul_326);  mul_326 = None
        convert_element_type_505: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        select_scatter_5: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_505, 0, 9);  convert_element_type_505 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_178: "f32[12]" = torch.ops.aten.add.Tensor(add_164, select_scatter_5);  add_164 = select_scatter_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_274: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_325, [32768, 768])
        permute_164: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_274, [1, 0])
        mm_122: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_164, view_156);  permute_164 = view_156 = None
        mm_123: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_274, permute_166);  view_274 = permute_166 = None
        view_275: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_123, [16, 2048, 3072]);  mm_123 = None
        convert_element_type_510: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_122, torch.float32);  mm_122 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_155: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_57, [16, 2048, 3072]);  mm_57 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_8: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_155);  view_155 = None
        pow_66: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_8, 1.0)
        mul_327: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Scalar(pow_66, 2.0);  pow_66 = None
        mul_328: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Tensor(view_275, mul_327);  view_275 = mul_327 = None
        le_3: "b8[16, 2048, 3072]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_9: "bf16[16, 2048, 3072]" = torch.ops.aten.where.self(le_3, full_default_5, mul_328);  le_3 = mul_328 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_276: "bf16[32768, 3072]" = torch.ops.aten.view.default(where_9, [32768, 3072]);  where_9 = None
        permute_168: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_276, [1, 0])
        mm_124: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_168, view_154);  permute_168 = view_154 = None
        mm_125: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_276, permute_170);  view_276 = permute_170 = None
        view_277: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_125, [16, 2048, 768]);  mm_125 = None
        convert_element_type_515: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_124, torch.float32);  mm_124 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_516: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_277, torch.float32);  view_277 = None
        convert_element_type_244: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_102, torch.float32);  add_102 = None
        mul_154: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_244, rsqrt_36);  convert_element_type_244 = None
        mul_330: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_154, convert_element_type_516)
        sum_27: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [2], True);  mul_330 = None
        div_17: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_154, 768);  mul_154 = None
        mul_331: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_17, sum_27);  div_17 = sum_27 = None
        sub_20: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_516, mul_331);  convert_element_type_516 = mul_331 = None
        mul_332: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_36);  sub_20 = rsqrt_36 = None
        convert_element_type_518: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_332, torch.bfloat16);  mul_332 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_179: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_325, convert_element_type_518);  mul_325 = convert_element_type_518 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_278: "bf16[32768, 768]" = torch.ops.aten.view.default(add_179, [32768, 768])
        permute_172: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_278, [1, 0])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_151: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_32, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_152: "bf16[32768, 768]" = torch.ops.aten.view.default(view_151, [32768, 768]);  view_151 = None
        mm_126: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_172, view_152);  permute_172 = view_152 = None
        mm_127: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_278, permute_174);  view_278 = permute_174 = None
        view_279: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_127, [16, 2048, 768]);  mm_127 = None
        convert_element_type_523: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_280: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_279, [16, 2048, 6, 128]);  view_279 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_backward_3 = torch.ops.flash_attn_3._flash_attn_backward.default(view_280, mul_152, mul_153, view_150, getitem_32, getitem_33, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0);  view_280 = mul_152 = mul_153 = view_150 = getitem_32 = getitem_33 = None
        getitem_60: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_3[0]
        getitem_61: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_3[1]
        getitem_62: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_3[2];  _flash_attn_backward_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_333: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_61, 1.2);  getitem_61 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_334: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_60, 1.2);  getitem_60 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_524: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_333, torch.float32);  mul_333 = None
        convert_element_type_239: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_18, torch.float32);  cat_18 = None
        mul_151: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_239, rsqrt_35);  convert_element_type_239 = None
        mul_336: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_151, convert_element_type_524)
        sum_28: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [3], True);  mul_336 = None
        div_18: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_151, 128);  mul_151 = None
        mul_337: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_18, sum_28);  div_18 = sum_28 = None
        sub_21: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_524, mul_337);  convert_element_type_524 = mul_337 = None
        mul_338: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_35);  sub_21 = rsqrt_35 = None
        convert_element_type_526: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_338, torch.bfloat16);  mul_338 = None
        convert_element_type_527: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_334, torch.float32);  mul_334 = None
        convert_element_type_237: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_17, torch.float32);  cat_17 = None
        mul_150: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_237, rsqrt_34);  convert_element_type_237 = None
        mul_340: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_150, convert_element_type_527)
        sum_29: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [3], True);  mul_340 = None
        div_19: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_150, 128);  mul_150 = None
        mul_341: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_19, sum_29);  div_19 = sum_29 = None
        sub_22: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_527, mul_341);  convert_element_type_527 = mul_341 = None
        mul_342: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_34);  sub_22 = rsqrt_34 = None
        convert_element_type_529: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_342, torch.bfloat16);  mul_342 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_74: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_526, 3, 0, 64)
        slice_75: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_526, 3, 64, 128);  convert_element_type_526 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_343: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_75, slice_1)
        mul_344: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_75, neg);  slice_75 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_345: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_74, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_180: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_343, mul_345);  mul_343 = mul_345 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_346: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_74, slice_1);  slice_74 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_181: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_344, mul_346);  mul_344 = mul_346 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_14: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_180, 3, 64, 9223372036854775807);  add_180 = None
        slice_scatter_15: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_181, 3, 0, 64);  add_181 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_182: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_14, slice_scatter_15);  slice_scatter_14 = slice_scatter_15 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_76: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_529, 3, 0, 64)
        slice_77: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_529, 3, 64, 128);  convert_element_type_529 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_347: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_77, slice_1)
        mul_348: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_77, neg);  slice_77 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_349: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_76, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_183: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_347, mul_349);  mul_347 = mul_349 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_350: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_76, slice_1);  slice_76 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_184: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_348, mul_350);  mul_348 = mul_350 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_16: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_183, 3, 64, 9223372036854775807);  add_183 = None
        slice_scatter_17: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_184, 3, 0, 64);  add_184 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_185: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_16, slice_scatter_17);  slice_scatter_16 = slice_scatter_17 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_281: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_62, [16, 2048, 768]);  getitem_62 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_282: "bf16[32768, 768]" = torch.ops.aten.view.default(view_281, [32768, 768]);  view_281 = None
        permute_176: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_282, [1, 0])
        mm_128: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_176, view_142);  permute_176 = None
        mm_129: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_282, permute_178);  view_282 = permute_178 = None
        view_283: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_129, [16, 2048, 768]);  mm_129 = None
        convert_element_type_534: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_284: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_182, [16, 2048, 768]);  add_182 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_285: "bf16[32768, 768]" = torch.ops.aten.view.default(view_284, [32768, 768]);  view_284 = None
        permute_180: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_130: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_180, view_142);  permute_180 = None
        mm_131: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_285, permute_182);  view_285 = permute_182 = None
        view_286: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_131, [16, 2048, 768]);  mm_131 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_186: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(view_283, view_286);  view_283 = view_286 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_539: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_287: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_185, [16, 2048, 768]);  add_185 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_288: "bf16[32768, 768]" = torch.ops.aten.view.default(view_287, [32768, 768]);  view_287 = None
        permute_184: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_132: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_184, view_142);  permute_184 = view_142 = None
        mm_133: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_288, permute_186);  view_288 = permute_186 = None
        view_289: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_133, [16, 2048, 768]);  mm_133 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_187: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_186, view_289);  add_186 = view_289 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_544: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_545: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_187, torch.float32);  add_187 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_16: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 8)
        mul_139: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_16, add_93)
        select_17: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 8)
        mul_140: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_17, cat)
        add_94: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_226: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_94, torch.float32);  add_94 = None
        mul_141: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_226, rsqrt_33);  convert_element_type_226 = None
        mul_352: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_141, convert_element_type_545)
        sum_30: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True);  mul_352 = None
        div_20: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_141, 768);  mul_141 = None
        mul_353: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_20, sum_30);  div_20 = sum_30 = None
        sub_23: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_545, mul_353);  convert_element_type_545 = mul_353 = None
        mul_354: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_33);  sub_23 = rsqrt_33 = None
        convert_element_type_547: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_354, torch.bfloat16);  mul_354 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_188: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_179, convert_element_type_547);  add_179 = convert_element_type_547 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_355: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_188, select_17);  select_17 = None
        mul_356: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_188, cat)
        sum_31: "bf16[]" = torch.ops.aten.sum.default(mul_356);  mul_356 = None
        convert_element_type_548: "f32[]" = torch.ops.prims.convert_element_type.default(sum_31, torch.float32);  sum_31 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_189: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_176, mul_355);  add_176 = mul_355 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_scatter_6: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_548, 0, 8);  convert_element_type_548 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_190: "f32[12]" = torch.ops.aten.add.Tensor(add_177, select_scatter_6);  add_177 = select_scatter_6 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_357: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_188, select_16);  select_16 = None
        mul_358: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_188, add_93);  add_188 = add_93 = None
        sum_32: "bf16[]" = torch.ops.aten.sum.default(mul_358);  mul_358 = None
        convert_element_type_549: "f32[]" = torch.ops.prims.convert_element_type.default(sum_32, torch.float32);  sum_32 = None
        select_scatter_7: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_549, 0, 8);  convert_element_type_549 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_191: "f32[12]" = torch.ops.aten.add.Tensor(add_178, select_scatter_7);  add_178 = select_scatter_7 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_290: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_357, [32768, 768])
        permute_188: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_290, [1, 0])
        mm_134: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_188, view_140);  permute_188 = view_140 = None
        mm_135: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_290, permute_190);  view_290 = permute_190 = None
        view_291: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_135, [16, 2048, 3072]);  mm_135 = None
        convert_element_type_554: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_134, torch.float32);  mm_134 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_139: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_51, [16, 2048, 3072]);  mm_51 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_7: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_139);  view_139 = None
        pow_67: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_7, 1.0)
        mul_359: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Scalar(pow_67, 2.0);  pow_67 = None
        mul_360: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Tensor(view_291, mul_359);  view_291 = mul_359 = None
        le_4: "b8[16, 2048, 3072]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_10: "bf16[16, 2048, 3072]" = torch.ops.aten.where.self(le_4, full_default_5, mul_360);  le_4 = mul_360 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_292: "bf16[32768, 3072]" = torch.ops.aten.view.default(where_10, [32768, 3072]);  where_10 = None
        permute_192: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_292, [1, 0])
        mm_136: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_192, view_138);  permute_192 = view_138 = None
        mm_137: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_292, permute_194);  view_292 = permute_194 = None
        view_293: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_137, [16, 2048, 768]);  mm_137 = None
        convert_element_type_559: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_560: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_293, torch.float32);  view_293 = None
        convert_element_type_218: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_91, torch.float32);  add_91 = None
        mul_138: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_218, rsqrt_32);  convert_element_type_218 = None
        mul_362: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_138, convert_element_type_560)
        sum_33: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_362, [2], True);  mul_362 = None
        div_21: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_138, 768);  mul_138 = None
        mul_363: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_21, sum_33);  div_21 = sum_33 = None
        sub_24: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_560, mul_363);  convert_element_type_560 = mul_363 = None
        mul_364: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_32);  sub_24 = rsqrt_32 = None
        convert_element_type_562: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_364, torch.bfloat16);  mul_364 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_192: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_357, convert_element_type_562);  mul_357 = convert_element_type_562 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_294: "bf16[32768, 768]" = torch.ops.aten.view.default(add_192, [32768, 768])
        permute_196: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_294, [1, 0])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_135: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_28, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_136: "bf16[32768, 768]" = torch.ops.aten.view.default(view_135, [32768, 768]);  view_135 = None
        mm_138: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_196, view_136);  permute_196 = view_136 = None
        mm_139: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_294, permute_198);  view_294 = permute_198 = None
        view_295: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_139, [16, 2048, 768]);  mm_139 = None
        convert_element_type_567: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_296: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_295, [16, 2048, 6, 128]);  view_295 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_backward_4 = torch.ops.flash_attn_3._flash_attn_backward.default(view_296, mul_136, mul_137, add_84, getitem_28, getitem_29, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0);  view_296 = mul_136 = mul_137 = add_84 = getitem_28 = getitem_29 = None
        getitem_64: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_4[0]
        getitem_65: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_4[1]
        getitem_66: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_4[2];  _flash_attn_backward_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_365: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_65, 1.2);  getitem_65 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_366: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_64, 1.2);  getitem_64 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_568: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_365, torch.float32);  mul_365 = None
        convert_element_type_213: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_16, torch.float32);  cat_16 = None
        mul_135: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_213, rsqrt_31);  convert_element_type_213 = None
        mul_368: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_135, convert_element_type_568)
        sum_34: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [3], True);  mul_368 = None
        div_22: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_135, 128);  mul_135 = None
        mul_369: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_22, sum_34);  div_22 = sum_34 = None
        sub_25: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_568, mul_369);  convert_element_type_568 = mul_369 = None
        mul_370: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_31);  sub_25 = rsqrt_31 = None
        convert_element_type_570: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_370, torch.bfloat16);  mul_370 = None
        convert_element_type_571: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_366, torch.float32);  mul_366 = None
        convert_element_type_211: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_15, torch.float32);  cat_15 = None
        mul_134: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_211, rsqrt_30);  convert_element_type_211 = None
        mul_372: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_134, convert_element_type_571)
        sum_35: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [3], True);  mul_372 = None
        div_23: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_134, 128);  mul_134 = None
        mul_373: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_23, sum_35);  div_23 = sum_35 = None
        sub_26: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_571, mul_373);  convert_element_type_571 = mul_373 = None
        mul_374: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_30);  sub_26 = rsqrt_30 = None
        convert_element_type_573: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_374, torch.bfloat16);  mul_374 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_78: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_570, 3, 0, 64)
        slice_79: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_570, 3, 64, 128);  convert_element_type_570 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_375: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_79, slice_1)
        mul_376: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_79, neg);  slice_79 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_377: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_78, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_193: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_375, mul_377);  mul_375 = mul_377 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_378: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_78, slice_1);  slice_78 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_194: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_376, mul_378);  mul_376 = mul_378 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_18: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_193, 3, 64, 9223372036854775807);  add_193 = None
        slice_scatter_19: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_194, 3, 0, 64);  add_194 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_195: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_18, slice_scatter_19);  slice_scatter_18 = slice_scatter_19 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_80: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_573, 3, 0, 64)
        slice_81: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_573, 3, 64, 128);  convert_element_type_573 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_379: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_81, slice_1)
        mul_380: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_81, neg);  slice_81 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_381: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_80, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_196: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_379, mul_381);  mul_379 = mul_381 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_382: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_80, slice_1);  slice_80 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_197: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_380, mul_382);  mul_380 = mul_382 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_20: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_196, 3, 64, 9223372036854775807);  add_196 = None
        slice_scatter_21: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_197, 3, 0, 64);  add_197 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_198: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_20, slice_scatter_21);  slice_scatter_20 = slice_scatter_21 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_134: "bf16[16, 2048, 6]" = torch.ops.aten.view.default(mm_49, [16, 2048, 6]);  mm_49 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        sigmoid_4: "bf16[16, 2048, 6]" = torch.ops.aten.sigmoid.default(view_134);  view_134 = None
        mul_124: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(sigmoid_4, 3)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        unsqueeze_3: "bf16[16, 2048, 6, 1]" = torch.ops.aten.unsqueeze.default(mul_124, -1);  mul_124 = None
        mul_383: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_66, unsqueeze_3);  unsqueeze_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_132: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(embedding_4, [16, 2048, 6, 128]);  embedding_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        mul_384: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_66, view_132);  view_132 = None
        sum_36: "bf16[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_384, [3], True);  mul_384 = None
        squeeze_3: "bf16[16, 2048, 6]" = torch.ops.aten.squeeze.dim(sum_36, -1);  sum_36 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        mul_385: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(squeeze_3, 3);  squeeze_3 = None
        convert_element_type_574: "f32[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(mul_385, torch.float32);  mul_385 = None
        convert_element_type_575: "f32[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_4, torch.float32);  sigmoid_4 = None
        sub_27: "f32[16, 2048, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_575)
        mul_386: "f32[16, 2048, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_575, sub_27);  convert_element_type_575 = sub_27 = None
        mul_387: "f32[16, 2048, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_574, mul_386);  convert_element_type_574 = mul_386 = None
        convert_element_type_576: "bf16[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(mul_387, torch.bfloat16);  mul_387 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_297: "bf16[32768, 6]" = torch.ops.aten.view.default(convert_element_type_576, [32768, 6]);  convert_element_type_576 = None
        permute_200: "bf16[6, 32768]" = torch.ops.aten.permute.default(view_297, [1, 0])
        mm_140: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_200, view_133);  permute_200 = view_133 = None
        mm_141: "bf16[32768, 12]" = torch.ops.aten.mm.default(view_297, permute_202);  view_297 = permute_202 = None
        view_298: "bf16[16, 2048, 12]" = torch.ops.aten.view.default(mm_141, [16, 2048, 12]);  mm_141 = None
        convert_element_type_581: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        slice_scatter_22: "bf16[16, 2048, 768]" = torch.ops.aten.slice_scatter.default(full_default_10, view_298, 2, 0, 12);  view_298 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_299: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mul_383, [16, 2048, 768]);  mul_383 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_300: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_66, [16, 2048, 768]);  getitem_66 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_301: "bf16[32768, 768]" = torch.ops.aten.view.default(view_300, [32768, 768]);  view_300 = None
        permute_204: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_301, [1, 0])
        mm_142: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_204, view_123);  permute_204 = None
        mm_143: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_301, permute_206);  view_301 = permute_206 = None
        view_302: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_143, [16, 2048, 768]);  mm_143 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_199: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(slice_scatter_22, view_302);  slice_scatter_22 = view_302 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_586: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_303: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_195, [16, 2048, 768]);  add_195 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_304: "bf16[32768, 768]" = torch.ops.aten.view.default(view_303, [32768, 768]);  view_303 = None
        permute_208: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_304, [1, 0])
        mm_144: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_208, view_123);  permute_208 = None
        mm_145: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_304, permute_210);  view_304 = permute_210 = None
        view_305: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_145, [16, 2048, 768]);  mm_145 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_200: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_199, view_305);  add_199 = view_305 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_591: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_306: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_198, [16, 2048, 768]);  add_198 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_307: "bf16[32768, 768]" = torch.ops.aten.view.default(view_306, [32768, 768]);  view_306 = None
        permute_212: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_307, [1, 0])
        mm_146: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_212, view_123);  permute_212 = view_123 = None
        mm_147: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_307, permute_214);  view_307 = permute_214 = None
        view_308: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_147, [16, 2048, 768]);  mm_147 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_201: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_200, view_308);  add_200 = view_308 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_596: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_146, torch.float32);  mm_146 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_597: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_201, torch.float32);  add_201 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_14: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 7)
        mul_121: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_14, add_81)
        select_15: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 7)
        mul_122: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_15, cat)
        add_82: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_197: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_82, torch.float32);  add_82 = None
        mul_123: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_197, rsqrt_29);  convert_element_type_197 = None
        mul_389: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_123, convert_element_type_597)
        sum_37: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [2], True);  mul_389 = None
        div_24: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_123, 768);  mul_123 = None
        mul_390: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_24, sum_37);  div_24 = sum_37 = None
        sub_28: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_597, mul_390);  convert_element_type_597 = mul_390 = None
        mul_391: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_29);  sub_28 = rsqrt_29 = None
        convert_element_type_599: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_391, torch.bfloat16);  mul_391 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_202: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_192, convert_element_type_599);  add_192 = convert_element_type_599 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:501 in forward, code: ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        convert_element_type_600: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        where_11: "f32[16, 2048, 768]" = torch.ops.aten.where.self(unsqueeze_8, full_default_1, convert_element_type_600);  convert_element_type_600 = None
        index_put_2: "f32[32768, 768]" = torch.ops.aten.index_put.default(full_default_12, [primals_1], where_11, True);  where_11 = None
        convert_element_type_601: "bf16[32768, 768]" = torch.ops.prims.convert_element_type.default(index_put_2, torch.bfloat16);  index_put_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_392: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_202, select_15);  select_15 = None
        mul_393: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_202, cat)
        sum_38: "bf16[]" = torch.ops.aten.sum.default(mul_393);  mul_393 = None
        convert_element_type_602: "f32[]" = torch.ops.prims.convert_element_type.default(sum_38, torch.float32);  sum_38 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_203: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_189, mul_392);  add_189 = mul_392 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_scatter_8: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_602, 0, 7);  convert_element_type_602 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_204: "f32[12]" = torch.ops.aten.add.Tensor(add_190, select_scatter_8);  add_190 = select_scatter_8 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_394: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_202, select_14);  select_14 = None
        mul_395: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_202, add_81);  add_202 = add_81 = None
        sum_39: "bf16[]" = torch.ops.aten.sum.default(mul_395);  mul_395 = None
        convert_element_type_603: "f32[]" = torch.ops.prims.convert_element_type.default(sum_39, torch.float32);  sum_39 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_205: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_219, mul_394);  mul_219 = mul_394 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_scatter_9: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_603, 0, 7);  convert_element_type_603 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_206: "f32[12]" = torch.ops.aten.add.Tensor(add_191, select_scatter_9);  add_191 = select_scatter_9 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_309: "bf16[32768, 768]" = torch.ops.aten.view.default(add_205, [32768, 768])
        permute_216: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_309, [1, 0])
        mm_148: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_216, view_121);  permute_216 = view_121 = None
        mm_149: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_309, permute_218);  view_309 = permute_218 = None
        view_310: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_149, [16, 2048, 3072]);  mm_149 = None
        convert_element_type_608: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_148, torch.float32);  mm_148 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_120: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_44, [16, 2048, 3072]);  mm_44 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_6: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_120);  view_120 = None
        pow_68: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_6, 1.0)
        mul_396: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Scalar(pow_68, 2.0);  pow_68 = None
        mul_397: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Tensor(view_310, mul_396);  view_310 = mul_396 = None
        le_5: "b8[16, 2048, 3072]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_12: "bf16[16, 2048, 3072]" = torch.ops.aten.where.self(le_5, full_default_5, mul_397);  le_5 = mul_397 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_311: "bf16[32768, 3072]" = torch.ops.aten.view.default(where_12, [32768, 3072]);  where_12 = None
        permute_220: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_150: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_220, view_119);  permute_220 = view_119 = None
        mm_151: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_311, permute_222);  view_311 = permute_222 = None
        view_312: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_151, [16, 2048, 768]);  mm_151 = None
        convert_element_type_613: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_614: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_312, torch.float32);  view_312 = None
        convert_element_type_189: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32);  add_79 = None
        mul_120: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_189, rsqrt_28);  convert_element_type_189 = None
        mul_399: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_120, convert_element_type_614)
        sum_40: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True);  mul_399 = None
        div_25: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_120, 768);  mul_120 = None
        mul_400: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_25, sum_40);  div_25 = sum_40 = None
        sub_29: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_614, mul_400);  convert_element_type_614 = mul_400 = None
        mul_401: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_28);  sub_29 = rsqrt_28 = None
        convert_element_type_616: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_401, torch.bfloat16);  mul_401 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_207: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_205, convert_element_type_616);  add_205 = convert_element_type_616 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_313: "bf16[32768, 768]" = torch.ops.aten.view.default(add_207, [32768, 768])
        permute_224: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_313, [1, 0])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_116: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_24, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_117: "bf16[32768, 768]" = torch.ops.aten.view.default(view_116, [32768, 768]);  view_116 = None
        mm_152: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_224, view_117);  permute_224 = view_117 = None
        mm_153: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_313, permute_226);  view_313 = permute_226 = None
        view_314: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_153, [16, 2048, 768]);  mm_153 = None
        convert_element_type_621: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_315: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_314, [16, 2048, 6, 128]);  view_314 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_backward_5 = torch.ops.flash_attn_3._flash_attn_backward.default(view_315, mul_118, mul_119, view_115, getitem_24, getitem_25, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0);  view_315 = mul_118 = mul_119 = view_115 = getitem_24 = getitem_25 = None
        getitem_68: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_5[0]
        getitem_69: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_5[1]
        getitem_70: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_5[2];  _flash_attn_backward_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_402: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_69, 1.2);  getitem_69 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_403: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_68, 1.2);  getitem_68 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_622: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_402, torch.float32);  mul_402 = None
        convert_element_type_184: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_14, torch.float32);  cat_14 = None
        mul_117: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_184, rsqrt_27);  convert_element_type_184 = None
        mul_405: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_117, convert_element_type_622)
        sum_41: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_405, [3], True);  mul_405 = None
        div_26: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_117, 128);  mul_117 = None
        mul_406: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_26, sum_41);  div_26 = sum_41 = None
        sub_30: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_622, mul_406);  convert_element_type_622 = mul_406 = None
        mul_407: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_27);  sub_30 = rsqrt_27 = None
        convert_element_type_624: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_407, torch.bfloat16);  mul_407 = None
        convert_element_type_625: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_403, torch.float32);  mul_403 = None
        convert_element_type_182: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_13, torch.float32);  cat_13 = None
        mul_116: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_182, rsqrt_26);  convert_element_type_182 = None
        mul_409: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_116, convert_element_type_625)
        sum_42: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_409, [3], True);  mul_409 = None
        div_27: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_116, 128);  mul_116 = None
        mul_410: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_27, sum_42);  div_27 = sum_42 = None
        sub_31: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_625, mul_410);  convert_element_type_625 = mul_410 = None
        mul_411: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_26);  sub_31 = rsqrt_26 = None
        convert_element_type_627: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_411, torch.bfloat16);  mul_411 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_82: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_624, 3, 0, 64)
        slice_83: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_624, 3, 64, 128);  convert_element_type_624 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_412: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_83, slice_1)
        mul_413: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_83, neg);  slice_83 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_414: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_82, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_208: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_412, mul_414);  mul_412 = mul_414 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_415: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_82, slice_1);  slice_82 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_209: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_413, mul_415);  mul_413 = mul_415 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_23: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_208, 3, 64, 9223372036854775807);  add_208 = None
        slice_scatter_24: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_209, 3, 0, 64);  add_209 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_210: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_23, slice_scatter_24);  slice_scatter_23 = slice_scatter_24 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_84: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_627, 3, 0, 64)
        slice_85: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_627, 3, 64, 128);  convert_element_type_627 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_416: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_85, slice_1)
        mul_417: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_85, neg);  slice_85 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_418: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_84, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_211: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_416, mul_418);  mul_416 = mul_418 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_419: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_84, slice_1);  slice_84 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_212: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_417, mul_419);  mul_417 = mul_419 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_25: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_211, 3, 64, 9223372036854775807);  add_211 = None
        slice_scatter_26: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_212, 3, 0, 64);  add_212 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_213: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_25, slice_scatter_26);  slice_scatter_25 = slice_scatter_26 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_316: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_70, [16, 2048, 768]);  getitem_70 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_317: "bf16[32768, 768]" = torch.ops.aten.view.default(view_316, [32768, 768]);  view_316 = None
        permute_228: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_317, [1, 0])
        mm_154: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_228, view_107);  permute_228 = None
        mm_155: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_317, permute_230);  view_317 = permute_230 = None
        view_318: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_155, [16, 2048, 768]);  mm_155 = None
        convert_element_type_632: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_319: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_210, [16, 2048, 768]);  add_210 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_320: "bf16[32768, 768]" = torch.ops.aten.view.default(view_319, [32768, 768]);  view_319 = None
        permute_232: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_320, [1, 0])
        mm_156: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_232, view_107);  permute_232 = None
        mm_157: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_320, permute_234);  view_320 = permute_234 = None
        view_321: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_157, [16, 2048, 768]);  mm_157 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_214: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(view_318, view_321);  view_318 = view_321 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_637: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_322: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_213, [16, 2048, 768]);  add_213 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_323: "bf16[32768, 768]" = torch.ops.aten.view.default(view_322, [32768, 768]);  view_322 = None
        permute_236: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_323, [1, 0])
        mm_158: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_236, view_107);  permute_236 = view_107 = None
        mm_159: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_323, permute_238);  view_323 = permute_238 = None
        view_324: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_159, [16, 2048, 768]);  mm_159 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_215: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_214, view_324);  add_214 = view_324 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_642: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_158, torch.float32);  mm_158 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_643: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_215, torch.float32);  add_215 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_12: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 6)
        mul_105: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_12, add_70)
        select_13: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 6)
        mul_106: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_13, cat)
        add_71: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_171: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        mul_107: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_171, rsqrt_25);  convert_element_type_171 = None
        mul_421: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_107, convert_element_type_643)
        sum_43: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True);  mul_421 = None
        div_28: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_107, 768);  mul_107 = None
        mul_422: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_28, sum_43);  div_28 = sum_43 = None
        sub_32: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_643, mul_422);  convert_element_type_643 = mul_422 = None
        mul_423: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_25);  sub_32 = rsqrt_25 = None
        convert_element_type_645: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_423, torch.bfloat16);  mul_423 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_216: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_207, convert_element_type_645);  add_207 = convert_element_type_645 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_424: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_216, select_13);  select_13 = None
        mul_425: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_216, cat)
        sum_44: "bf16[]" = torch.ops.aten.sum.default(mul_425);  mul_425 = None
        convert_element_type_646: "f32[]" = torch.ops.prims.convert_element_type.default(sum_44, torch.float32);  sum_44 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_217: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_203, mul_424);  add_203 = mul_424 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_scatter_10: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_646, 0, 6);  convert_element_type_646 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_218: "f32[12]" = torch.ops.aten.add.Tensor(add_204, select_scatter_10);  add_204 = select_scatter_10 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_426: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_216, select_12);  select_12 = None
        mul_427: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_216, add_70);  add_216 = add_70 = None
        sum_45: "bf16[]" = torch.ops.aten.sum.default(mul_427);  mul_427 = None
        convert_element_type_647: "f32[]" = torch.ops.prims.convert_element_type.default(sum_45, torch.float32);  sum_45 = None
        select_scatter_11: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_647, 0, 6);  convert_element_type_647 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_219: "f32[12]" = torch.ops.aten.add.Tensor(add_206, select_scatter_11);  add_206 = select_scatter_11 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_325: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_426, [32768, 768])
        permute_240: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_325, [1, 0])
        mm_160: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_240, view_105);  permute_240 = view_105 = None
        mm_161: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_325, permute_242);  view_325 = permute_242 = None
        view_326: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_161, [16, 2048, 3072]);  mm_161 = None
        convert_element_type_652: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_104: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_38, [16, 2048, 3072]);  mm_38 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_5: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_104);  view_104 = None
        pow_69: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_5, 1.0)
        mul_428: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Scalar(pow_69, 2.0);  pow_69 = None
        mul_429: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Tensor(view_326, mul_428);  view_326 = mul_428 = None
        le_6: "b8[16, 2048, 3072]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_13: "bf16[16, 2048, 3072]" = torch.ops.aten.where.self(le_6, full_default_5, mul_429);  le_6 = mul_429 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_327: "bf16[32768, 3072]" = torch.ops.aten.view.default(where_13, [32768, 3072]);  where_13 = None
        permute_244: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_327, [1, 0])
        mm_162: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_244, view_103);  permute_244 = view_103 = None
        mm_163: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_327, permute_246);  view_327 = permute_246 = None
        view_328: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_163, [16, 2048, 768]);  mm_163 = None
        convert_element_type_657: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_658: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_328, torch.float32);  view_328 = None
        convert_element_type_163: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_68, torch.float32);  add_68 = None
        mul_104: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_163, rsqrt_24);  convert_element_type_163 = None
        mul_431: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_104, convert_element_type_658)
        sum_46: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [2], True);  mul_431 = None
        div_29: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_104, 768);  mul_104 = None
        mul_432: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_29, sum_46);  div_29 = sum_46 = None
        sub_33: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_658, mul_432);  convert_element_type_658 = mul_432 = None
        mul_433: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_24);  sub_33 = rsqrt_24 = None
        convert_element_type_660: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_433, torch.bfloat16);  mul_433 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_220: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_426, convert_element_type_660);  mul_426 = convert_element_type_660 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_329: "bf16[32768, 768]" = torch.ops.aten.view.default(add_220, [32768, 768])
        permute_248: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_329, [1, 0])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_100: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_20, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_101: "bf16[32768, 768]" = torch.ops.aten.view.default(view_100, [32768, 768]);  view_100 = None
        mm_164: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_248, view_101);  permute_248 = view_101 = None
        mm_165: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_329, permute_250);  view_329 = permute_250 = None
        view_330: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_165, [16, 2048, 768]);  mm_165 = None
        convert_element_type_665: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_164, torch.float32);  mm_164 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_331: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_330, [16, 2048, 6, 128]);  view_330 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_backward_6 = torch.ops.flash_attn_3._flash_attn_backward.default(view_331, mul_102, mul_103, add_61, getitem_20, getitem_21, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0);  view_331 = mul_102 = mul_103 = add_61 = getitem_20 = getitem_21 = None
        getitem_72: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_6[0]
        getitem_73: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_6[1]
        getitem_74: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_6[2];  _flash_attn_backward_6 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_434: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_73, 1.2);  getitem_73 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_435: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_72, 1.2);  getitem_72 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_666: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_434, torch.float32);  mul_434 = None
        convert_element_type_158: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_12, torch.float32);  cat_12 = None
        mul_101: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_158, rsqrt_23);  convert_element_type_158 = None
        mul_437: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_101, convert_element_type_666)
        sum_47: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_437, [3], True);  mul_437 = None
        div_30: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_101, 128);  mul_101 = None
        mul_438: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_30, sum_47);  div_30 = sum_47 = None
        sub_34: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_666, mul_438);  convert_element_type_666 = mul_438 = None
        mul_439: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_23);  sub_34 = rsqrt_23 = None
        convert_element_type_668: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_439, torch.bfloat16);  mul_439 = None
        convert_element_type_669: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_435, torch.float32);  mul_435 = None
        convert_element_type_156: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_11, torch.float32);  cat_11 = None
        mul_100: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_156, rsqrt_22);  convert_element_type_156 = None
        mul_441: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_100, convert_element_type_669)
        sum_48: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [3], True);  mul_441 = None
        div_31: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_100, 128);  mul_100 = None
        mul_442: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_31, sum_48);  div_31 = sum_48 = None
        sub_35: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_669, mul_442);  convert_element_type_669 = mul_442 = None
        mul_443: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_22);  sub_35 = rsqrt_22 = None
        convert_element_type_671: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_443, torch.bfloat16);  mul_443 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_86: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_668, 3, 0, 64)
        slice_87: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_668, 3, 64, 128);  convert_element_type_668 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_444: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_87, slice_1)
        mul_445: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_87, neg);  slice_87 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_446: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_86, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_221: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_444, mul_446);  mul_444 = mul_446 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_447: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_86, slice_1);  slice_86 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_222: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_445, mul_447);  mul_445 = mul_447 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_27: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_221, 3, 64, 9223372036854775807);  add_221 = None
        slice_scatter_28: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_222, 3, 0, 64);  add_222 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_223: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_27, slice_scatter_28);  slice_scatter_27 = slice_scatter_28 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_88: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_671, 3, 0, 64)
        slice_89: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_671, 3, 64, 128);  convert_element_type_671 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_448: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_89, slice_1)
        mul_449: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_89, neg);  slice_89 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_450: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_88, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_224: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_448, mul_450);  mul_448 = mul_450 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_451: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_88, slice_1);  slice_88 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_225: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_449, mul_451);  mul_449 = mul_451 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_29: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_224, 3, 64, 9223372036854775807);  add_224 = None
        slice_scatter_30: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_225, 3, 0, 64);  add_225 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_226: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_29, slice_scatter_30);  slice_scatter_29 = slice_scatter_30 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_99: "bf16[16, 2048, 6]" = torch.ops.aten.view.default(mm_36, [16, 2048, 6]);  mm_36 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        sigmoid_3: "bf16[16, 2048, 6]" = torch.ops.aten.sigmoid.default(view_99);  view_99 = None
        mul_90: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(sigmoid_3, 3)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        unsqueeze_2: "bf16[16, 2048, 6, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
        mul_452: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_74, unsqueeze_2);  unsqueeze_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_97: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(embedding_3, [16, 2048, 6, 128]);  embedding_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        mul_453: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_74, view_97);  view_97 = None
        sum_49: "bf16[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [3], True);  mul_453 = None
        squeeze_4: "bf16[16, 2048, 6]" = torch.ops.aten.squeeze.dim(sum_49, -1);  sum_49 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        mul_454: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(squeeze_4, 3);  squeeze_4 = None
        convert_element_type_672: "f32[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(mul_454, torch.float32);  mul_454 = None
        convert_element_type_673: "f32[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_3, torch.float32);  sigmoid_3 = None
        sub_36: "f32[16, 2048, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_673)
        mul_455: "f32[16, 2048, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_673, sub_36);  convert_element_type_673 = sub_36 = None
        mul_456: "f32[16, 2048, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_672, mul_455);  convert_element_type_672 = mul_455 = None
        convert_element_type_674: "bf16[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(mul_456, torch.bfloat16);  mul_456 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_332: "bf16[32768, 6]" = torch.ops.aten.view.default(convert_element_type_674, [32768, 6]);  convert_element_type_674 = None
        permute_252: "bf16[6, 32768]" = torch.ops.aten.permute.default(view_332, [1, 0])
        mm_166: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_252, view_98);  permute_252 = view_98 = None
        mm_167: "bf16[32768, 12]" = torch.ops.aten.mm.default(view_332, permute_254);  view_332 = permute_254 = None
        view_333: "bf16[16, 2048, 12]" = torch.ops.aten.view.default(mm_167, [16, 2048, 12]);  mm_167 = None
        convert_element_type_679: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        slice_scatter_31: "bf16[16, 2048, 768]" = torch.ops.aten.slice_scatter.default(full_default_10, view_333, 2, 0, 12);  view_333 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_334: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mul_452, [16, 2048, 768]);  mul_452 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_335: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_74, [16, 2048, 768]);  getitem_74 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_336: "bf16[32768, 768]" = torch.ops.aten.view.default(view_335, [32768, 768]);  view_335 = None
        permute_256: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_336, [1, 0])
        mm_168: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_256, view_88);  permute_256 = None
        mm_169: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_336, permute_258);  view_336 = permute_258 = None
        view_337: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_169, [16, 2048, 768]);  mm_169 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_227: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(slice_scatter_31, view_337);  slice_scatter_31 = view_337 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_684: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_338: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_223, [16, 2048, 768]);  add_223 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_339: "bf16[32768, 768]" = torch.ops.aten.view.default(view_338, [32768, 768]);  view_338 = None
        permute_260: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_339, [1, 0])
        mm_170: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_260, view_88);  permute_260 = None
        mm_171: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_339, permute_262);  view_339 = permute_262 = None
        view_340: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_171, [16, 2048, 768]);  mm_171 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_228: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_227, view_340);  add_227 = view_340 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_689: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_170, torch.float32);  mm_170 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_341: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_226, [16, 2048, 768]);  add_226 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_342: "bf16[32768, 768]" = torch.ops.aten.view.default(view_341, [32768, 768]);  view_341 = None
        permute_264: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_342, [1, 0])
        mm_172: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_264, view_88);  permute_264 = view_88 = None
        mm_173: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_342, permute_266);  view_342 = permute_266 = None
        view_343: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_173, [16, 2048, 768]);  mm_173 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_229: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_228, view_343);  add_228 = view_343 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_694: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_172, torch.float32);  mm_172 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_695: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_229, torch.float32);  add_229 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_10: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 5)
        mul_87: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_10, add_58)
        select_11: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 5)
        mul_88: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_11, cat)
        add_59: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_142: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        mul_89: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_142, rsqrt_21);  convert_element_type_142 = None
        mul_458: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_89, convert_element_type_695)
        sum_50: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_458, [2], True);  mul_458 = None
        div_32: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_89, 768);  mul_89 = None
        mul_459: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_32, sum_50);  div_32 = sum_50 = None
        sub_37: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_695, mul_459);  convert_element_type_695 = mul_459 = None
        mul_460: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_21);  sub_37 = rsqrt_21 = None
        convert_element_type_697: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_460, torch.bfloat16);  mul_460 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_230: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_220, convert_element_type_697);  add_220 = convert_element_type_697 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:501 in forward, code: ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        convert_element_type_698: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_334, torch.float32);  view_334 = None
        where_14: "f32[16, 2048, 768]" = torch.ops.aten.where.self(unsqueeze_8, full_default_1, convert_element_type_698);  convert_element_type_698 = None
        index_put_3: "f32[32768, 768]" = torch.ops.aten.index_put.default(full_default_12, [primals_1], where_14, True);  where_14 = None
        convert_element_type_699: "bf16[32768, 768]" = torch.ops.prims.convert_element_type.default(index_put_3, torch.bfloat16);  index_put_3 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_461: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_230, select_11);  select_11 = None
        mul_462: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_230, cat)
        sum_51: "bf16[]" = torch.ops.aten.sum.default(mul_462);  mul_462 = None
        convert_element_type_700: "f32[]" = torch.ops.prims.convert_element_type.default(sum_51, torch.float32);  sum_51 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_231: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_217, mul_461);  add_217 = mul_461 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_scatter_12: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_700, 0, 5);  convert_element_type_700 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_232: "f32[12]" = torch.ops.aten.add.Tensor(add_218, select_scatter_12);  add_218 = select_scatter_12 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_463: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_230, select_10);  select_10 = None
        mul_464: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_230, add_58);  add_230 = add_58 = None
        sum_52: "bf16[]" = torch.ops.aten.sum.default(mul_464);  mul_464 = None
        convert_element_type_701: "f32[]" = torch.ops.prims.convert_element_type.default(sum_52, torch.float32);  sum_52 = None
        select_scatter_13: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_701, 0, 5);  convert_element_type_701 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_233: "f32[12]" = torch.ops.aten.add.Tensor(add_219, select_scatter_13);  add_219 = select_scatter_13 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_344: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_463, [32768, 768])
        permute_268: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_344, [1, 0])
        mm_174: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_268, view_86);  permute_268 = view_86 = None
        mm_175: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_344, permute_270);  view_344 = permute_270 = None
        view_345: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_175, [16, 2048, 3072]);  mm_175 = None
        convert_element_type_706: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_174, torch.float32);  mm_174 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_85: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_31, [16, 2048, 3072]);  mm_31 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_4: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_85);  view_85 = None
        pow_70: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_4, 1.0)
        mul_465: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Scalar(pow_70, 2.0);  pow_70 = None
        mul_466: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Tensor(view_345, mul_465);  view_345 = mul_465 = None
        le_7: "b8[16, 2048, 3072]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_15: "bf16[16, 2048, 3072]" = torch.ops.aten.where.self(le_7, full_default_5, mul_466);  le_7 = mul_466 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_346: "bf16[32768, 3072]" = torch.ops.aten.view.default(where_15, [32768, 3072]);  where_15 = None
        permute_272: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_346, [1, 0])
        mm_176: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_272, view_84);  permute_272 = view_84 = None
        mm_177: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_346, permute_274);  view_346 = permute_274 = None
        view_347: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_177, [16, 2048, 768]);  mm_177 = None
        convert_element_type_711: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_712: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_347, torch.float32);  view_347 = None
        convert_element_type_134: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_56, torch.float32);  add_56 = None
        mul_86: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_134, rsqrt_20);  convert_element_type_134 = None
        mul_468: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_86, convert_element_type_712)
        sum_53: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_468, [2], True);  mul_468 = None
        div_33: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_86, 768);  mul_86 = None
        mul_469: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_33, sum_53);  div_33 = sum_53 = None
        sub_38: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_712, mul_469);  convert_element_type_712 = mul_469 = None
        mul_470: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_20);  sub_38 = rsqrt_20 = None
        convert_element_type_714: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_470, torch.bfloat16);  mul_470 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_234: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_463, convert_element_type_714);  mul_463 = convert_element_type_714 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_348: "bf16[32768, 768]" = torch.ops.aten.view.default(add_234, [32768, 768])
        permute_276: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_348, [1, 0])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_81: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_16, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_82: "bf16[32768, 768]" = torch.ops.aten.view.default(view_81, [32768, 768]);  view_81 = None
        mm_178: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_276, view_82);  permute_276 = view_82 = None
        mm_179: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_348, permute_278);  view_348 = permute_278 = None
        view_349: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_179, [16, 2048, 768]);  mm_179 = None
        convert_element_type_719: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_350: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_349, [16, 2048, 6, 128]);  view_349 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_backward_7 = torch.ops.flash_attn_3._flash_attn_backward.default(view_350, mul_84, mul_85, view_80, getitem_16, getitem_17, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0);  view_350 = mul_84 = mul_85 = view_80 = getitem_16 = getitem_17 = None
        getitem_76: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_7[0]
        getitem_77: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_7[1]
        getitem_78: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_7[2];  _flash_attn_backward_7 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_471: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_77, 1.2);  getitem_77 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_472: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_76, 1.2);  getitem_76 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_720: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_471, torch.float32);  mul_471 = None
        convert_element_type_129: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_10, torch.float32);  cat_10 = None
        mul_83: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_129, rsqrt_19);  convert_element_type_129 = None
        mul_474: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_83, convert_element_type_720)
        sum_54: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_474, [3], True);  mul_474 = None
        div_34: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_83, 128);  mul_83 = None
        mul_475: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_34, sum_54);  div_34 = sum_54 = None
        sub_39: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_720, mul_475);  convert_element_type_720 = mul_475 = None
        mul_476: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_19);  sub_39 = rsqrt_19 = None
        convert_element_type_722: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_476, torch.bfloat16);  mul_476 = None
        convert_element_type_723: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_472, torch.float32);  mul_472 = None
        convert_element_type_127: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_9, torch.float32);  cat_9 = None
        mul_82: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_127, rsqrt_18);  convert_element_type_127 = None
        mul_478: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_82, convert_element_type_723)
        sum_55: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [3], True);  mul_478 = None
        div_35: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_82, 128);  mul_82 = None
        mul_479: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_35, sum_55);  div_35 = sum_55 = None
        sub_40: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_723, mul_479);  convert_element_type_723 = mul_479 = None
        mul_480: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_18);  sub_40 = rsqrt_18 = None
        convert_element_type_725: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_480, torch.bfloat16);  mul_480 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_90: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_722, 3, 0, 64)
        slice_91: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_722, 3, 64, 128);  convert_element_type_722 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_481: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_91, slice_1)
        mul_482: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_91, neg);  slice_91 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_483: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_90, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_235: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_481, mul_483);  mul_481 = mul_483 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_484: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_90, slice_1);  slice_90 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_236: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_482, mul_484);  mul_482 = mul_484 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_32: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_235, 3, 64, 9223372036854775807);  add_235 = None
        slice_scatter_33: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_236, 3, 0, 64);  add_236 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_237: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_32, slice_scatter_33);  slice_scatter_32 = slice_scatter_33 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_92: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_725, 3, 0, 64)
        slice_93: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_725, 3, 64, 128);  convert_element_type_725 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_485: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_93, slice_1)
        mul_486: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_93, neg);  slice_93 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_487: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_92, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_238: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_485, mul_487);  mul_485 = mul_487 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_488: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_92, slice_1);  slice_92 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_239: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_486, mul_488);  mul_486 = mul_488 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_34: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_238, 3, 64, 9223372036854775807);  add_238 = None
        slice_scatter_35: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_239, 3, 0, 64);  add_239 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_240: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_34, slice_scatter_35);  slice_scatter_34 = slice_scatter_35 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_351: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_78, [16, 2048, 768]);  getitem_78 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_352: "bf16[32768, 768]" = torch.ops.aten.view.default(view_351, [32768, 768]);  view_351 = None
        permute_280: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_352, [1, 0])
        mm_180: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_280, view_72);  permute_280 = None
        mm_181: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_352, permute_282);  view_352 = permute_282 = None
        view_353: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_181, [16, 2048, 768]);  mm_181 = None
        convert_element_type_730: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_180, torch.float32);  mm_180 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_354: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_237, [16, 2048, 768]);  add_237 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_355: "bf16[32768, 768]" = torch.ops.aten.view.default(view_354, [32768, 768]);  view_354 = None
        permute_284: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_355, [1, 0])
        mm_182: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_284, view_72);  permute_284 = None
        mm_183: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_355, permute_286);  view_355 = permute_286 = None
        view_356: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_183, [16, 2048, 768]);  mm_183 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_241: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(view_353, view_356);  view_353 = view_356 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_735: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_182, torch.float32);  mm_182 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_357: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_240, [16, 2048, 768]);  add_240 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_358: "bf16[32768, 768]" = torch.ops.aten.view.default(view_357, [32768, 768]);  view_357 = None
        permute_288: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_358, [1, 0])
        mm_184: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_288, view_72);  permute_288 = view_72 = None
        mm_185: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_358, permute_290);  view_358 = permute_290 = None
        view_359: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_185, [16, 2048, 768]);  mm_185 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_242: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_241, view_359);  add_241 = view_359 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_740: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_741: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_242, torch.float32);  add_242 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_8: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 4)
        mul_71: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_8, add_47)
        select_9: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 4)
        mul_72: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_9, cat)
        add_48: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_116: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None
        mul_73: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_116, rsqrt_17);  convert_element_type_116 = None
        mul_490: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_73, convert_element_type_741)
        sum_56: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [2], True);  mul_490 = None
        div_36: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_73, 768);  mul_73 = None
        mul_491: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_36, sum_56);  div_36 = sum_56 = None
        sub_41: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_741, mul_491);  convert_element_type_741 = mul_491 = None
        mul_492: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_17);  sub_41 = rsqrt_17 = None
        convert_element_type_743: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_492, torch.bfloat16);  mul_492 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_243: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_234, convert_element_type_743);  add_234 = convert_element_type_743 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_493: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_243, select_9);  select_9 = None
        mul_494: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_243, cat)
        sum_57: "bf16[]" = torch.ops.aten.sum.default(mul_494);  mul_494 = None
        convert_element_type_744: "f32[]" = torch.ops.prims.convert_element_type.default(sum_57, torch.float32);  sum_57 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_244: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_231, mul_493);  add_231 = mul_493 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_scatter_14: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_744, 0, 4);  convert_element_type_744 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_245: "f32[12]" = torch.ops.aten.add.Tensor(add_232, select_scatter_14);  add_232 = select_scatter_14 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_495: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_243, select_8);  select_8 = None
        mul_496: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_243, add_47);  add_243 = add_47 = None
        sum_58: "bf16[]" = torch.ops.aten.sum.default(mul_496);  mul_496 = None
        convert_element_type_745: "f32[]" = torch.ops.prims.convert_element_type.default(sum_58, torch.float32);  sum_58 = None
        select_scatter_15: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_745, 0, 4);  convert_element_type_745 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_246: "f32[12]" = torch.ops.aten.add.Tensor(add_233, select_scatter_15);  add_233 = select_scatter_15 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_360: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_495, [32768, 768])
        permute_292: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_360, [1, 0])
        mm_186: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_292, view_70);  permute_292 = view_70 = None
        mm_187: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_360, permute_294);  view_360 = permute_294 = None
        view_361: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_187, [16, 2048, 3072]);  mm_187 = None
        convert_element_type_750: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_186, torch.float32);  mm_186 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_69: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_25, [16, 2048, 3072]);  mm_25 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_3: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_69);  view_69 = None
        pow_71: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_3, 1.0)
        mul_497: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Scalar(pow_71, 2.0);  pow_71 = None
        mul_498: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Tensor(view_361, mul_497);  view_361 = mul_497 = None
        le_8: "b8[16, 2048, 3072]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_16: "bf16[16, 2048, 3072]" = torch.ops.aten.where.self(le_8, full_default_5, mul_498);  le_8 = mul_498 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_362: "bf16[32768, 3072]" = torch.ops.aten.view.default(where_16, [32768, 3072]);  where_16 = None
        permute_296: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_362, [1, 0])
        mm_188: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_296, view_68);  permute_296 = view_68 = None
        mm_189: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_362, permute_298);  view_362 = permute_298 = None
        view_363: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_189, [16, 2048, 768]);  mm_189 = None
        convert_element_type_755: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_756: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_363, torch.float32);  view_363 = None
        convert_element_type_108: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_45, torch.float32);  add_45 = None
        mul_70: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_108, rsqrt_16);  convert_element_type_108 = None
        mul_500: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_70, convert_element_type_756)
        sum_59: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [2], True);  mul_500 = None
        div_37: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_70, 768);  mul_70 = None
        mul_501: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_37, sum_59);  div_37 = sum_59 = None
        sub_42: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_756, mul_501);  convert_element_type_756 = mul_501 = None
        mul_502: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_16);  sub_42 = rsqrt_16 = None
        convert_element_type_758: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_502, torch.bfloat16);  mul_502 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_247: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_495, convert_element_type_758);  mul_495 = convert_element_type_758 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_364: "bf16[32768, 768]" = torch.ops.aten.view.default(add_247, [32768, 768])
        permute_300: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_364, [1, 0])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_65: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_12, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_66: "bf16[32768, 768]" = torch.ops.aten.view.default(view_65, [32768, 768]);  view_65 = None
        mm_190: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_300, view_66);  permute_300 = view_66 = None
        mm_191: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_364, permute_302);  view_364 = permute_302 = None
        view_365: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_191, [16, 2048, 768]);  mm_191 = None
        convert_element_type_763: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_366: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_365, [16, 2048, 6, 128]);  view_365 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_backward_8 = torch.ops.flash_attn_3._flash_attn_backward.default(view_366, mul_68, mul_69, add_38, getitem_12, getitem_13, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0);  view_366 = mul_68 = mul_69 = add_38 = getitem_12 = getitem_13 = None
        getitem_80: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_8[0]
        getitem_81: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_8[1]
        getitem_82: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_8[2];  _flash_attn_backward_8 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_503: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_81, 1.2);  getitem_81 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_504: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_80, 1.2);  getitem_80 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_764: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_503, torch.float32);  mul_503 = None
        convert_element_type_103: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_8, torch.float32);  cat_8 = None
        mul_67: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_103, rsqrt_15);  convert_element_type_103 = None
        mul_506: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_67, convert_element_type_764)
        sum_60: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_506, [3], True);  mul_506 = None
        div_38: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_67, 128);  mul_67 = None
        mul_507: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_38, sum_60);  div_38 = sum_60 = None
        sub_43: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_764, mul_507);  convert_element_type_764 = mul_507 = None
        mul_508: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_15);  sub_43 = rsqrt_15 = None
        convert_element_type_766: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_508, torch.bfloat16);  mul_508 = None
        convert_element_type_767: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_504, torch.float32);  mul_504 = None
        convert_element_type_101: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_7, torch.float32);  cat_7 = None
        mul_66: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_101, rsqrt_14);  convert_element_type_101 = None
        mul_510: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_66, convert_element_type_767)
        sum_61: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_510, [3], True);  mul_510 = None
        div_39: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_66, 128);  mul_66 = None
        mul_511: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_39, sum_61);  div_39 = sum_61 = None
        sub_44: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_767, mul_511);  convert_element_type_767 = mul_511 = None
        mul_512: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_14);  sub_44 = rsqrt_14 = None
        convert_element_type_769: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_512, torch.bfloat16);  mul_512 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_94: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_766, 3, 0, 64)
        slice_95: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_766, 3, 64, 128);  convert_element_type_766 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_513: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_95, slice_1)
        mul_514: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_95, neg);  slice_95 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_515: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_94, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_248: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_513, mul_515);  mul_513 = mul_515 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_516: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_94, slice_1);  slice_94 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_249: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_514, mul_516);  mul_514 = mul_516 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_36: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_248, 3, 64, 9223372036854775807);  add_248 = None
        slice_scatter_37: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_249, 3, 0, 64);  add_249 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_250: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_36, slice_scatter_37);  slice_scatter_36 = slice_scatter_37 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_96: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_769, 3, 0, 64)
        slice_97: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_769, 3, 64, 128);  convert_element_type_769 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_517: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_97, slice_1)
        mul_518: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_97, neg);  slice_97 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_519: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_96, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_251: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_517, mul_519);  mul_517 = mul_519 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_520: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_96, slice_1);  slice_96 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_252: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_518, mul_520);  mul_518 = mul_520 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_38: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_251, 3, 64, 9223372036854775807);  add_251 = None
        slice_scatter_39: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_252, 3, 0, 64);  add_252 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_253: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_38, slice_scatter_39);  slice_scatter_38 = slice_scatter_39 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_64: "bf16[16, 2048, 6]" = torch.ops.aten.view.default(mm_23, [16, 2048, 6]);  mm_23 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        sigmoid_2: "bf16[16, 2048, 6]" = torch.ops.aten.sigmoid.default(view_64);  view_64 = None
        mul_56: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(sigmoid_2, 3)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        unsqueeze_1: "bf16[16, 2048, 6, 1]" = torch.ops.aten.unsqueeze.default(mul_56, -1);  mul_56 = None
        mul_521: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_82, unsqueeze_1);  unsqueeze_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_62: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(embedding_2, [16, 2048, 6, 128]);  embedding_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        mul_522: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_82, view_62);  view_62 = None
        sum_62: "bf16[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [3], True);  mul_522 = None
        squeeze_5: "bf16[16, 2048, 6]" = torch.ops.aten.squeeze.dim(sum_62, -1);  sum_62 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        mul_523: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(squeeze_5, 3);  squeeze_5 = None
        convert_element_type_770: "f32[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(mul_523, torch.float32);  mul_523 = None
        convert_element_type_771: "f32[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_2, torch.float32);  sigmoid_2 = None
        sub_45: "f32[16, 2048, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_771)
        mul_524: "f32[16, 2048, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_771, sub_45);  convert_element_type_771 = sub_45 = None
        mul_525: "f32[16, 2048, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_770, mul_524);  convert_element_type_770 = mul_524 = None
        convert_element_type_772: "bf16[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(mul_525, torch.bfloat16);  mul_525 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_367: "bf16[32768, 6]" = torch.ops.aten.view.default(convert_element_type_772, [32768, 6]);  convert_element_type_772 = None
        permute_304: "bf16[6, 32768]" = torch.ops.aten.permute.default(view_367, [1, 0])
        mm_192: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_304, view_63);  permute_304 = view_63 = None
        mm_193: "bf16[32768, 12]" = torch.ops.aten.mm.default(view_367, permute_306);  view_367 = permute_306 = None
        view_368: "bf16[16, 2048, 12]" = torch.ops.aten.view.default(mm_193, [16, 2048, 12]);  mm_193 = None
        convert_element_type_777: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        slice_scatter_40: "bf16[16, 2048, 768]" = torch.ops.aten.slice_scatter.default(full_default_10, view_368, 2, 0, 12);  view_368 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_369: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mul_521, [16, 2048, 768]);  mul_521 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_370: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_82, [16, 2048, 768]);  getitem_82 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_371: "bf16[32768, 768]" = torch.ops.aten.view.default(view_370, [32768, 768]);  view_370 = None
        permute_308: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_371, [1, 0])
        mm_194: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_308, view_53);  permute_308 = None
        mm_195: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_371, permute_310);  view_371 = permute_310 = None
        view_372: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_195, [16, 2048, 768]);  mm_195 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_254: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(slice_scatter_40, view_372);  slice_scatter_40 = view_372 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_782: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_194, torch.float32);  mm_194 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_373: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_250, [16, 2048, 768]);  add_250 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_374: "bf16[32768, 768]" = torch.ops.aten.view.default(view_373, [32768, 768]);  view_373 = None
        permute_312: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_374, [1, 0])
        mm_196: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_312, view_53);  permute_312 = None
        mm_197: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_374, permute_314);  view_374 = permute_314 = None
        view_375: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_197, [16, 2048, 768]);  mm_197 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_255: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_254, view_375);  add_254 = view_375 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_787: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_196, torch.float32);  mm_196 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_376: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_253, [16, 2048, 768]);  add_253 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_377: "bf16[32768, 768]" = torch.ops.aten.view.default(view_376, [32768, 768]);  view_376 = None
        permute_316: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_377, [1, 0])
        mm_198: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_316, view_53);  permute_316 = view_53 = None
        mm_199: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_377, permute_318);  view_377 = permute_318 = None
        view_378: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_199, [16, 2048, 768]);  mm_199 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_256: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_255, view_378);  add_255 = view_378 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_792: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_198, torch.float32);  mm_198 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_793: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_256, torch.float32);  add_256 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_6: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 3)
        mul_53: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_6, add_35)
        select_7: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 3)
        mul_54: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_7, cat)
        add_36: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_87: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_36, torch.float32);  add_36 = None
        mul_55: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_87, rsqrt_13);  convert_element_type_87 = None
        mul_527: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_55, convert_element_type_793)
        sum_63: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True);  mul_527 = None
        div_40: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_55, 768);  mul_55 = None
        mul_528: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_40, sum_63);  div_40 = sum_63 = None
        sub_46: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_793, mul_528);  convert_element_type_793 = mul_528 = None
        mul_529: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_13);  sub_46 = rsqrt_13 = None
        convert_element_type_795: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_529, torch.bfloat16);  mul_529 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_257: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_247, convert_element_type_795);  add_247 = convert_element_type_795 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:501 in forward, code: ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        convert_element_type_796: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_369, torch.float32);  view_369 = None
        where_17: "f32[16, 2048, 768]" = torch.ops.aten.where.self(unsqueeze_8, full_default_1, convert_element_type_796);  convert_element_type_796 = None
        index_put_4: "f32[32768, 768]" = torch.ops.aten.index_put.default(full_default_12, [primals_1], where_17, True);  where_17 = None
        convert_element_type_797: "bf16[32768, 768]" = torch.ops.prims.convert_element_type.default(index_put_4, torch.bfloat16);  index_put_4 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_530: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_257, select_7);  select_7 = None
        mul_531: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_257, cat)
        sum_64: "bf16[]" = torch.ops.aten.sum.default(mul_531);  mul_531 = None
        convert_element_type_798: "f32[]" = torch.ops.prims.convert_element_type.default(sum_64, torch.float32);  sum_64 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_258: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_244, mul_530);  add_244 = mul_530 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_scatter_16: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_798, 0, 3);  convert_element_type_798 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_259: "f32[12]" = torch.ops.aten.add.Tensor(add_245, select_scatter_16);  add_245 = select_scatter_16 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_532: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_257, select_6);  select_6 = None
        mul_533: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_257, add_35);  add_257 = add_35 = None
        sum_65: "bf16[]" = torch.ops.aten.sum.default(mul_533);  mul_533 = None
        convert_element_type_799: "f32[]" = torch.ops.prims.convert_element_type.default(sum_65, torch.float32);  sum_65 = None
        select_scatter_17: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_799, 0, 3);  convert_element_type_799 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_260: "f32[12]" = torch.ops.aten.add.Tensor(add_246, select_scatter_17);  add_246 = select_scatter_17 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_379: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_532, [32768, 768])
        permute_320: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_379, [1, 0])
        mm_200: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_320, view_51);  permute_320 = view_51 = None
        mm_201: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_379, permute_322);  view_379 = permute_322 = None
        view_380: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_201, [16, 2048, 3072]);  mm_201 = None
        convert_element_type_804: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_200, torch.float32);  mm_200 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_50: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_18, [16, 2048, 3072]);  mm_18 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_2: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_50);  view_50 = None
        pow_72: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_2, 1.0)
        mul_534: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Scalar(pow_72, 2.0);  pow_72 = None
        mul_535: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Tensor(view_380, mul_534);  view_380 = mul_534 = None
        le_9: "b8[16, 2048, 3072]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_18: "bf16[16, 2048, 3072]" = torch.ops.aten.where.self(le_9, full_default_5, mul_535);  le_9 = mul_535 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_381: "bf16[32768, 3072]" = torch.ops.aten.view.default(where_18, [32768, 3072]);  where_18 = None
        permute_324: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_381, [1, 0])
        mm_202: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_324, view_49);  permute_324 = view_49 = None
        mm_203: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_381, permute_326);  view_381 = permute_326 = None
        view_382: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_203, [16, 2048, 768]);  mm_203 = None
        convert_element_type_809: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_202, torch.float32);  mm_202 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_810: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        convert_element_type_79: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        mul_52: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_79, rsqrt_12);  convert_element_type_79 = None
        mul_537: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_52, convert_element_type_810)
        sum_66: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_537, [2], True);  mul_537 = None
        div_41: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_52, 768);  mul_52 = None
        mul_538: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_41, sum_66);  div_41 = sum_66 = None
        sub_47: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_810, mul_538);  convert_element_type_810 = mul_538 = None
        mul_539: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_12);  sub_47 = rsqrt_12 = None
        convert_element_type_812: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_539, torch.bfloat16);  mul_539 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_261: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_532, convert_element_type_812);  mul_532 = convert_element_type_812 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_383: "bf16[32768, 768]" = torch.ops.aten.view.default(add_261, [32768, 768])
        permute_328: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_383, [1, 0])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_46: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_8, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_47: "bf16[32768, 768]" = torch.ops.aten.view.default(view_46, [32768, 768]);  view_46 = None
        mm_204: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_328, view_47);  permute_328 = view_47 = None
        mm_205: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_383, permute_330);  view_383 = permute_330 = None
        view_384: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_205, [16, 2048, 768]);  mm_205 = None
        convert_element_type_817: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_204, torch.float32);  mm_204 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_385: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_384, [16, 2048, 6, 128]);  view_384 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_backward_9 = torch.ops.flash_attn_3._flash_attn_backward.default(view_385, mul_50, mul_51, view_45, getitem_8, getitem_9, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0);  view_385 = mul_50 = mul_51 = view_45 = getitem_8 = getitem_9 = None
        getitem_84: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_9[0]
        getitem_85: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_9[1]
        getitem_86: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_9[2];  _flash_attn_backward_9 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_540: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_85, 1.2);  getitem_85 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_541: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_84, 1.2);  getitem_84 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_818: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_540, torch.float32);  mul_540 = None
        convert_element_type_74: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_6, torch.float32);  cat_6 = None
        mul_49: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_74, rsqrt_11);  convert_element_type_74 = None
        mul_543: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_49, convert_element_type_818)
        sum_67: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_543, [3], True);  mul_543 = None
        div_42: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_49, 128);  mul_49 = None
        mul_544: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_42, sum_67);  div_42 = sum_67 = None
        sub_48: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_818, mul_544);  convert_element_type_818 = mul_544 = None
        mul_545: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_11);  sub_48 = rsqrt_11 = None
        convert_element_type_820: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_545, torch.bfloat16);  mul_545 = None
        convert_element_type_821: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_541, torch.float32);  mul_541 = None
        convert_element_type_72: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_5, torch.float32);  cat_5 = None
        mul_48: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_72, rsqrt_10);  convert_element_type_72 = None
        mul_547: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_48, convert_element_type_821)
        sum_68: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_547, [3], True);  mul_547 = None
        div_43: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_48, 128);  mul_48 = None
        mul_548: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_43, sum_68);  div_43 = sum_68 = None
        sub_49: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_821, mul_548);  convert_element_type_821 = mul_548 = None
        mul_549: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_10);  sub_49 = rsqrt_10 = None
        convert_element_type_823: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_549, torch.bfloat16);  mul_549 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_98: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_820, 3, 0, 64)
        slice_99: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_820, 3, 64, 128);  convert_element_type_820 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_550: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_99, slice_1)
        mul_551: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_99, neg);  slice_99 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_552: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_98, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_262: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_550, mul_552);  mul_550 = mul_552 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_553: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_98, slice_1);  slice_98 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_263: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_551, mul_553);  mul_551 = mul_553 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_41: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_262, 3, 64, 9223372036854775807);  add_262 = None
        slice_scatter_42: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_263, 3, 0, 64);  add_263 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_264: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_41, slice_scatter_42);  slice_scatter_41 = slice_scatter_42 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_100: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_823, 3, 0, 64)
        slice_101: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_823, 3, 64, 128);  convert_element_type_823 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_554: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_101, slice_1)
        mul_555: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_101, neg);  slice_101 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_556: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_100, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_265: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_554, mul_556);  mul_554 = mul_556 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_557: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_100, slice_1);  slice_100 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_266: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_555, mul_557);  mul_555 = mul_557 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_43: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_265, 3, 64, 9223372036854775807);  add_265 = None
        slice_scatter_44: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_266, 3, 0, 64);  add_266 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_267: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_43, slice_scatter_44);  slice_scatter_43 = slice_scatter_44 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_386: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_86, [16, 2048, 768]);  getitem_86 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_387: "bf16[32768, 768]" = torch.ops.aten.view.default(view_386, [32768, 768]);  view_386 = None
        permute_332: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_387, [1, 0])
        mm_206: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_332, view_37);  permute_332 = None
        mm_207: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_387, permute_334);  view_387 = permute_334 = None
        view_388: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_207, [16, 2048, 768]);  mm_207 = None
        convert_element_type_828: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_206, torch.float32);  mm_206 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_389: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_264, [16, 2048, 768]);  add_264 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_390: "bf16[32768, 768]" = torch.ops.aten.view.default(view_389, [32768, 768]);  view_389 = None
        permute_336: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_390, [1, 0])
        mm_208: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_336, view_37);  permute_336 = None
        mm_209: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_390, permute_338);  view_390 = permute_338 = None
        view_391: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_209, [16, 2048, 768]);  mm_209 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_268: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(view_388, view_391);  view_388 = view_391 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_833: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_208, torch.float32);  mm_208 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_392: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_267, [16, 2048, 768]);  add_267 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_393: "bf16[32768, 768]" = torch.ops.aten.view.default(view_392, [32768, 768]);  view_392 = None
        permute_340: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_393, [1, 0])
        mm_210: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_340, view_37);  permute_340 = view_37 = None
        mm_211: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_393, permute_342);  view_393 = permute_342 = None
        view_394: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_211, [16, 2048, 768]);  mm_211 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_269: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_268, view_394);  add_268 = view_394 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_838: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_210, torch.float32);  mm_210 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_839: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_269, torch.float32);  add_269 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_4: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 2)
        mul_37: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_4, add_24)
        select_5: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 2)
        mul_38: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_5, cat)
        add_25: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_61: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_25, torch.float32);  add_25 = None
        mul_39: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_61, rsqrt_9);  convert_element_type_61 = None
        mul_559: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_39, convert_element_type_839)
        sum_69: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_559, [2], True);  mul_559 = None
        div_44: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_39, 768);  mul_39 = None
        mul_560: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_44, sum_69);  div_44 = sum_69 = None
        sub_50: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_839, mul_560);  convert_element_type_839 = mul_560 = None
        mul_561: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_9);  sub_50 = rsqrt_9 = None
        convert_element_type_841: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_561, torch.bfloat16);  mul_561 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_270: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_261, convert_element_type_841);  add_261 = convert_element_type_841 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_562: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_270, select_5);  select_5 = None
        mul_563: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_270, cat)
        sum_70: "bf16[]" = torch.ops.aten.sum.default(mul_563);  mul_563 = None
        convert_element_type_842: "f32[]" = torch.ops.prims.convert_element_type.default(sum_70, torch.float32);  sum_70 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_271: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_258, mul_562);  add_258 = mul_562 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_scatter_18: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_842, 0, 2);  convert_element_type_842 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_272: "f32[12]" = torch.ops.aten.add.Tensor(add_259, select_scatter_18);  add_259 = select_scatter_18 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_564: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_270, select_4);  select_4 = None
        mul_565: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_270, add_24);  add_270 = add_24 = None
        sum_71: "bf16[]" = torch.ops.aten.sum.default(mul_565);  mul_565 = None
        convert_element_type_843: "f32[]" = torch.ops.prims.convert_element_type.default(sum_71, torch.float32);  sum_71 = None
        select_scatter_19: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_843, 0, 2);  convert_element_type_843 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_273: "f32[12]" = torch.ops.aten.add.Tensor(add_260, select_scatter_19);  add_260 = select_scatter_19 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_395: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_564, [32768, 768])
        permute_344: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_212: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_344, view_35);  permute_344 = view_35 = None
        mm_213: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_395, permute_346);  view_395 = permute_346 = None
        view_396: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_213, [16, 2048, 3072]);  mm_213 = None
        convert_element_type_848: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_212, torch.float32);  mm_212 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_34: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_12, [16, 2048, 3072]);  mm_12 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu_1: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_34);  view_34 = None
        pow_73: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_1, 1.0)
        mul_566: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Scalar(pow_73, 2.0);  pow_73 = None
        mul_567: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Tensor(view_396, mul_566);  view_396 = mul_566 = None
        le_10: "b8[16, 2048, 3072]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_19: "bf16[16, 2048, 3072]" = torch.ops.aten.where.self(le_10, full_default_5, mul_567);  le_10 = mul_567 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_397: "bf16[32768, 3072]" = torch.ops.aten.view.default(where_19, [32768, 3072]);  where_19 = None
        permute_348: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_397, [1, 0])
        mm_214: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_348, view_33);  permute_348 = view_33 = None
        mm_215: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_397, permute_350);  view_397 = permute_350 = None
        view_398: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_215, [16, 2048, 768]);  mm_215 = None
        convert_element_type_853: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_214, torch.float32);  mm_214 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_854: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_398, torch.float32);  view_398 = None
        convert_element_type_53: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_22, torch.float32);  add_22 = None
        mul_36: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_53, rsqrt_8);  convert_element_type_53 = None
        mul_569: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_36, convert_element_type_854)
        sum_72: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_569, [2], True);  mul_569 = None
        div_45: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_36, 768);  mul_36 = None
        mul_570: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_45, sum_72);  div_45 = sum_72 = None
        sub_51: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_854, mul_570);  convert_element_type_854 = mul_570 = None
        mul_571: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_8);  sub_51 = rsqrt_8 = None
        convert_element_type_856: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_571, torch.bfloat16);  mul_571 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_274: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_564, convert_element_type_856);  mul_564 = convert_element_type_856 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_399: "bf16[32768, 768]" = torch.ops.aten.view.default(add_274, [32768, 768])
        permute_352: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_399, [1, 0])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_30: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_4, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_31: "bf16[32768, 768]" = torch.ops.aten.view.default(view_30, [32768, 768]);  view_30 = None
        mm_216: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_352, view_31);  permute_352 = view_31 = None
        mm_217: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_399, permute_354);  view_399 = permute_354 = None
        view_400: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_217, [16, 2048, 768]);  mm_217 = None
        convert_element_type_861: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_216, torch.float32);  mm_216 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_401: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_400, [16, 2048, 6, 128]);  view_400 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_backward_10 = torch.ops.flash_attn_3._flash_attn_backward.default(view_401, mul_34, mul_35, add_15, getitem_4, getitem_5, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0);  view_401 = mul_34 = mul_35 = add_15 = getitem_4 = getitem_5 = None
        getitem_88: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_10[0]
        getitem_89: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_10[1]
        getitem_90: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_10[2];  _flash_attn_backward_10 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_572: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_89, 1.2);  getitem_89 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_573: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_88, 1.2);  getitem_88 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_862: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_572, torch.float32);  mul_572 = None
        convert_element_type_48: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_4, torch.float32);  cat_4 = None
        mul_33: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_48, rsqrt_7);  convert_element_type_48 = None
        mul_575: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_33, convert_element_type_862)
        sum_73: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_575, [3], True);  mul_575 = None
        div_46: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_33, 128);  mul_33 = None
        mul_576: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_46, sum_73);  div_46 = sum_73 = None
        sub_52: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_862, mul_576);  convert_element_type_862 = mul_576 = None
        mul_577: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_7);  sub_52 = rsqrt_7 = None
        convert_element_type_864: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_577, torch.bfloat16);  mul_577 = None
        convert_element_type_865: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_573, torch.float32);  mul_573 = None
        convert_element_type_46: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_3, torch.float32);  cat_3 = None
        mul_32: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_46, rsqrt_6);  convert_element_type_46 = None
        mul_579: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_32, convert_element_type_865)
        sum_74: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [3], True);  mul_579 = None
        div_47: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_32, 128);  mul_32 = None
        mul_580: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_47, sum_74);  div_47 = sum_74 = None
        sub_53: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_865, mul_580);  convert_element_type_865 = mul_580 = None
        mul_581: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_6);  sub_53 = rsqrt_6 = None
        convert_element_type_867: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_581, torch.bfloat16);  mul_581 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_102: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_864, 3, 0, 64)
        slice_103: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_864, 3, 64, 128);  convert_element_type_864 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_582: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_103, slice_1)
        mul_583: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_103, neg);  slice_103 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_584: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_102, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_275: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_582, mul_584);  mul_582 = mul_584 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_585: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_102, slice_1);  slice_102 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_276: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_583, mul_585);  mul_583 = mul_585 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_45: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_275, 3, 64, 9223372036854775807);  add_275 = None
        slice_scatter_46: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_276, 3, 0, 64);  add_276 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_277: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_45, slice_scatter_46);  slice_scatter_45 = slice_scatter_46 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_104: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_867, 3, 0, 64)
        slice_105: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_867, 3, 64, 128);  convert_element_type_867 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_586: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_105, slice_1)
        mul_587: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_105, neg);  slice_105 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_588: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_104, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_278: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_586, mul_588);  mul_586 = mul_588 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_589: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_104, slice_1);  slice_104 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_279: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_587, mul_589);  mul_587 = mul_589 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_47: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_278, 3, 64, 9223372036854775807);  add_278 = None
        slice_scatter_48: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_279, 3, 0, 64);  add_279 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_280: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_47, slice_scatter_48);  slice_scatter_47 = slice_scatter_48 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_29: "bf16[16, 2048, 6]" = torch.ops.aten.view.default(mm_10, [16, 2048, 6]);  mm_10 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        sigmoid_1: "bf16[16, 2048, 6]" = torch.ops.aten.sigmoid.default(view_29);  view_29 = None
        mul_22: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(sigmoid_1, 3)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        unsqueeze: "bf16[16, 2048, 6, 1]" = torch.ops.aten.unsqueeze.default(mul_22, -1);  mul_22 = None
        mul_590: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_90, unsqueeze);  unsqueeze = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_27: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(embedding_1, [16, 2048, 6, 128]);  embedding_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:97 in forward, code: v = v + gate.unsqueeze(-1) * ve
        mul_591: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_90, view_27);  view_27 = None
        sum_75: "bf16[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_591, [3], True);  mul_591 = None
        squeeze_6: "bf16[16, 2048, 6]" = torch.ops.aten.squeeze.dim(sum_75, -1);  sum_75 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        mul_592: "bf16[16, 2048, 6]" = torch.ops.aten.mul.Tensor(squeeze_6, 3);  squeeze_6 = None
        convert_element_type_868: "f32[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(mul_592, torch.float32);  mul_592 = None
        convert_element_type_869: "f32[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(sigmoid_1, torch.float32);  sigmoid_1 = None
        sub_54: "f32[16, 2048, 6]" = torch.ops.aten.sub.Tensor(1, convert_element_type_869)
        mul_593: "f32[16, 2048, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_869, sub_54);  convert_element_type_869 = sub_54 = None
        mul_594: "f32[16, 2048, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_868, mul_593);  convert_element_type_868 = mul_593 = None
        convert_element_type_870: "bf16[16, 2048, 6]" = torch.ops.prims.convert_element_type.default(mul_594, torch.bfloat16);  mul_594 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_402: "bf16[32768, 6]" = torch.ops.aten.view.default(convert_element_type_870, [32768, 6]);  convert_element_type_870 = None
        permute_356: "bf16[6, 32768]" = torch.ops.aten.permute.default(view_402, [1, 0])
        mm_218: "bf16[6, 12]" = torch.ops.aten.mm.default(permute_356, view_28);  permute_356 = view_28 = None
        mm_219: "bf16[32768, 12]" = torch.ops.aten.mm.default(view_402, permute_358);  view_402 = permute_358 = None
        view_403: "bf16[16, 2048, 12]" = torch.ops.aten.view.default(mm_219, [16, 2048, 12]);  mm_219 = None
        convert_element_type_875: "f32[6, 12]" = torch.ops.prims.convert_element_type.default(mm_218, torch.float32);  mm_218 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:96 in forward, code: gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
        slice_scatter_49: "bf16[16, 2048, 768]" = torch.ops.aten.slice_scatter.default(full_default_10, view_403, 2, 0, 12);  view_403 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:95 in forward, code: ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        view_404: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mul_590, [16, 2048, 768]);  mul_590 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_405: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_90, [16, 2048, 768]);  getitem_90 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_406: "bf16[32768, 768]" = torch.ops.aten.view.default(view_405, [32768, 768]);  view_405 = None
        permute_360: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_406, [1, 0])
        mm_220: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_360, view_18);  permute_360 = None
        mm_221: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_406, permute_362);  view_406 = permute_362 = None
        view_407: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_221, [16, 2048, 768]);  mm_221 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_281: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(slice_scatter_49, view_407);  slice_scatter_49 = view_407 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_880: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_220, torch.float32);  mm_220 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_408: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_277, [16, 2048, 768]);  add_277 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_409: "bf16[32768, 768]" = torch.ops.aten.view.default(view_408, [32768, 768]);  view_408 = None
        permute_364: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_409, [1, 0])
        mm_222: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_364, view_18);  permute_364 = None
        mm_223: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_409, permute_366);  view_409 = permute_366 = None
        view_410: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_223, [16, 2048, 768]);  mm_223 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_282: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_281, view_410);  add_281 = view_410 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_885: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_222, torch.float32);  mm_222 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_411: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_280, [16, 2048, 768]);  add_280 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_412: "bf16[32768, 768]" = torch.ops.aten.view.default(view_411, [32768, 768]);  view_411 = None
        permute_368: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_412, [1, 0])
        mm_224: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_368, view_18);  permute_368 = view_18 = None
        mm_225: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_412, permute_370);  view_412 = permute_370 = None
        view_413: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_225, [16, 2048, 768]);  mm_225 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_283: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_282, view_413);  add_282 = view_413 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_890: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_224, torch.float32);  mm_224 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_891: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_283, torch.float32);  add_283 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_2: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 1)
        mul_19: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_2, add_12)
        select_3: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 1)
        mul_20: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_3, cat)
        add_13: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_19, mul_20);  mul_19 = mul_20 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_32: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None
        mul_21: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, rsqrt_5);  convert_element_type_32 = None
        mul_596: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_21, convert_element_type_891)
        sum_76: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_596, [2], True);  mul_596 = None
        div_48: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_21, 768);  mul_21 = None
        mul_597: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_48, sum_76);  div_48 = sum_76 = None
        sub_55: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_891, mul_597);  convert_element_type_891 = mul_597 = None
        mul_598: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_5);  sub_55 = rsqrt_5 = None
        convert_element_type_893: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_598, torch.bfloat16);  mul_598 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_284: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_274, convert_element_type_893);  add_274 = convert_element_type_893 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:501 in forward, code: ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        convert_element_type_894: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_404, torch.float32);  view_404 = None
        where_20: "f32[16, 2048, 768]" = torch.ops.aten.where.self(unsqueeze_8, full_default_1, convert_element_type_894);  convert_element_type_894 = None
        index_put_5: "f32[32768, 768]" = torch.ops.aten.index_put.default(full_default_12, [primals_1], where_20, True);  where_20 = None
        convert_element_type_895: "bf16[32768, 768]" = torch.ops.prims.convert_element_type.default(index_put_5, torch.bfloat16);  index_put_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_599: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_284, select_3);  select_3 = None
        mul_600: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_284, cat)
        sum_77: "bf16[]" = torch.ops.aten.sum.default(mul_600);  mul_600 = None
        convert_element_type_896: "f32[]" = torch.ops.prims.convert_element_type.default(sum_77, torch.float32);  sum_77 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_285: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_271, mul_599);  add_271 = mul_599 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_scatter_20: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_896, 0, 1);  convert_element_type_896 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_286: "f32[12]" = torch.ops.aten.add.Tensor(add_272, select_scatter_20);  add_272 = select_scatter_20 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_601: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_284, select_2);  select_2 = None
        mul_602: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_284, add_12);  add_284 = add_12 = None
        sum_78: "bf16[]" = torch.ops.aten.sum.default(mul_602);  mul_602 = None
        convert_element_type_897: "f32[]" = torch.ops.prims.convert_element_type.default(sum_78, torch.float32);  sum_78 = None
        select_scatter_21: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_897, 0, 1);  convert_element_type_897 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_287: "f32[12]" = torch.ops.aten.add.Tensor(add_273, select_scatter_21);  add_273 = select_scatter_21 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_414: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_601, [32768, 768])
        permute_372: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_414, [1, 0])
        mm_226: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_372, view_16);  permute_372 = view_16 = None
        mm_227: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_414, permute_374);  view_414 = permute_374 = None
        view_415: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_227, [16, 2048, 3072]);  mm_227 = None
        convert_element_type_902: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_226, torch.float32);  mm_226 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_15: "bf16[16, 2048, 3072]" = torch.ops.aten.view.default(mm_5, [16, 2048, 3072]);  mm_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:139 in forward, code: x = F.relu(x).square()
        relu: "bf16[16, 2048, 3072]" = torch.ops.aten.relu.default(view_15);  view_15 = None
        pow_74: "bf16[16, 2048, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu, 1.0)
        mul_603: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Scalar(pow_74, 2.0);  pow_74 = None
        mul_604: "bf16[16, 2048, 3072]" = torch.ops.aten.mul.Tensor(view_415, mul_603);  view_415 = mul_603 = None
        le_11: "b8[16, 2048, 3072]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_21: "bf16[16, 2048, 3072]" = torch.ops.aten.where.self(le_11, full_default_5, mul_604);  le_11 = full_default_5 = mul_604 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_416: "bf16[32768, 3072]" = torch.ops.aten.view.default(where_21, [32768, 3072]);  where_21 = None
        permute_376: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_416, [1, 0])
        mm_228: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_376, view_14);  permute_376 = view_14 = None
        mm_229: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_416, permute_378);  view_416 = permute_378 = None
        view_417: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_229, [16, 2048, 768]);  mm_229 = None
        convert_element_type_907: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_228, torch.float32);  mm_228 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_908: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_417, torch.float32);  view_417 = None
        convert_element_type_24: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32);  add_10 = None
        mul_18: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_24, rsqrt_4);  convert_element_type_24 = None
        mul_606: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_18, convert_element_type_908)
        sum_79: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_606, [2], True);  mul_606 = None
        div_49: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_18, 768);  mul_18 = None
        mul_607: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_49, sum_79);  div_49 = sum_79 = None
        sub_56: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_908, mul_607);  convert_element_type_908 = mul_607 = None
        mul_608: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_4);  sub_56 = rsqrt_4 = None
        convert_element_type_910: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_608, torch.bfloat16);  mul_608 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_288: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_601, convert_element_type_910);  mul_601 = convert_element_type_910 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_418: "bf16[32768, 768]" = torch.ops.aten.view.default(add_288, [32768, 768])
        permute_380: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_418, [1, 0])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_11: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem, [16, 2048, -1])
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_12: "bf16[32768, 768]" = torch.ops.aten.view.default(view_11, [32768, 768]);  view_11 = None
        mm_230: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_380, view_12);  permute_380 = view_12 = None
        mm_231: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_418, permute_382);  view_418 = permute_382 = None
        view_419: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_231, [16, 2048, 768]);  mm_231 = None
        convert_element_type_915: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_230, torch.float32);  mm_230 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:126 in forward, code: y = y.contiguous().view(B, T, -1)
        view_420: "bf16[16, 2048, 6, 128]" = torch.ops.aten.view.default(view_419, [16, 2048, 6, 128]);  view_419 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/cache/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-aarch64-linux/flash_attention_3/flash_attn_interface.py:826 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_backward_11 = torch.ops.flash_attn_3._flash_attn_backward.default(view_420, mul_16, mul_17, view_10, getitem, getitem_1, None, None, None, None, None, None, 0.08838834764831845, True, 2048, 0);  view_420 = mul_16 = mul_17 = view_10 = getitem = getitem_1 = None
        getitem_92: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_11[0]
        getitem_93: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_11[1]
        getitem_94: "bf16[16, 2048, 6, 128]" = _flash_attn_backward_11[2];  _flash_attn_backward_11 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:104 in forward, code: k = k * 1.2
        mul_609: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_93, 1.2);  getitem_93 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:103 in forward, code: q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        mul_610: "bf16[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(getitem_92, 1.2);  getitem_92 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_916: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_609, torch.float32);  mul_609 = None
        convert_element_type_19: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_2, torch.float32);  cat_2 = None
        mul_15: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_19, rsqrt_3);  convert_element_type_19 = None
        mul_612: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_15, convert_element_type_916)
        sum_80: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_612, [3], True);  mul_612 = None
        div_50: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_15, 128);  mul_15 = None
        mul_613: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_50, sum_80);  div_50 = sum_80 = None
        sub_57: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_916, mul_613);  convert_element_type_916 = mul_613 = None
        mul_614: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_3);  sub_57 = rsqrt_3 = None
        convert_element_type_918: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_614, torch.bfloat16);  mul_614 = None
        convert_element_type_919: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_610, torch.float32);  mul_610 = None
        convert_element_type_17: "f32[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_1, torch.float32);  cat_1 = None
        mul_14: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_17, rsqrt_2);  convert_element_type_17 = None
        mul_616: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(mul_14, convert_element_type_919)
        sum_81: "f32[16, 2048, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_616, [3], True);  mul_616 = None
        div_51: "f32[16, 2048, 6, 128]" = torch.ops.aten.div.Tensor(mul_14, 128);  mul_14 = None
        mul_617: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(div_51, sum_81);  div_51 = sum_81 = None
        sub_58: "f32[16, 2048, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_919, mul_617);  convert_element_type_919 = mul_617 = None
        mul_618: "f32[16, 2048, 6, 128]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_2);  sub_58 = rsqrt_2 = None
        convert_element_type_921: "bf16[16, 2048, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_618, torch.bfloat16);  mul_618 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_106: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_918, 3, 0, 64)
        slice_107: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_918, 3, 64, 128);  convert_element_type_918 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_619: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_107, slice_1)
        mul_620: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_107, neg);  slice_107 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_621: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_106, slice_2)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_289: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_619, mul_621);  mul_619 = mul_621 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_622: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_106, slice_1);  slice_106 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_290: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_620, mul_622);  mul_620 = mul_622 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_50: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_289, 3, 64, 9223372036854775807);  add_289 = None
        slice_scatter_51: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_290, 3, 0, 64);  add_290 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_291: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_50, slice_scatter_51);  slice_scatter_50 = slice_scatter_51 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:65 in apply_rotary_emb, code: return torch.cat([y1, y2], 3)
        slice_108: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_921, 3, 0, 64)
        slice_109: "bf16[16, 2048, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_921, 3, 64, 128);  convert_element_type_921 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:64 in apply_rotary_emb, code: y2 = x1 * (-sin) + x2 * cos
        mul_623: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_109, slice_1)
        mul_624: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_109, neg);  slice_109 = neg = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_625: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_108, slice_2);  slice_2 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_292: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_623, mul_625);  mul_623 = mul_625 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        mul_626: "bf16[16, 2048, 6, 64]" = torch.ops.aten.mul.Tensor(slice_108, slice_1);  slice_108 = slice_1 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:63 in apply_rotary_emb, code: y1 = x1 * cos + x2 * sin # rotate pairs of dims
        add_293: "bf16[16, 2048, 6, 64]" = torch.ops.aten.add.Tensor(mul_624, mul_626);  mul_624 = mul_626 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        slice_scatter_52: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_292, 3, 64, 9223372036854775807);  add_292 = None
        slice_scatter_53: "bf16[16, 2048, 6, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, add_293, 3, 0, 64);  full_default_6 = add_293 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:62 in apply_rotary_emb, code: x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
        add_294: "bf16[16, 2048, 6, 128]" = torch.ops.aten.add.Tensor(slice_scatter_52, slice_scatter_53);  slice_scatter_52 = slice_scatter_53 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:91 in forward, code: v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        view_421: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(getitem_94, [16, 2048, 768]);  getitem_94 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_422: "bf16[32768, 768]" = torch.ops.aten.view.default(view_421, [32768, 768]);  view_421 = None
        permute_384: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_422, [1, 0])
        mm_232: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_384, view_2);  permute_384 = None
        mm_233: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_422, permute_386);  view_422 = permute_386 = None
        view_423: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_233, [16, 2048, 768]);  mm_233 = None
        convert_element_type_926: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_232, torch.float32);  mm_232 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:90 in forward, code: k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        view_424: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_291, [16, 2048, 768]);  add_291 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_425: "bf16[32768, 768]" = torch.ops.aten.view.default(view_424, [32768, 768]);  view_424 = None
        permute_388: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_425, [1, 0])
        mm_234: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_388, view_2);  permute_388 = None
        mm_235: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_425, permute_390);  view_425 = permute_390 = None
        view_426: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_235, [16, 2048, 768]);  mm_235 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_295: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(view_423, view_426);  view_423 = view_426 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_931: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_234, torch.float32);  mm_234 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:89 in forward, code: q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        view_427: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(add_294, [16, 2048, 768]);  add_294 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_428: "bf16[32768, 768]" = torch.ops.aten.view.default(view_427, [32768, 768]);  view_427 = None
        permute_392: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_428, [1, 0])
        mm_236: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_392, view_2);  permute_392 = view_2 = None
        mm_237: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_428, permute_394);  view_428 = permute_394 = None
        view_429: "bf16[16, 2048, 768]" = torch.ops.aten.view.default(mm_237, [16, 2048, 768]);  mm_237 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        add_296: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_295, view_429);  add_295 = view_429 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        convert_element_type_936: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_236, torch.float32);  mm_236 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_937: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_296, torch.float32);  add_296 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select: "f32[]" = torch.ops.aten.select.int(primals_7, 0, 0);  primals_7 = None
        mul_3: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select, cat)
        select_1: "f32[]" = torch.ops.aten.select.int(primals_8, 0, 0);  primals_8 = None
        mul_4: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(select_1, cat)
        add_2: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_6: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.float32);  add_2 = None
        mul_5: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, rsqrt_1);  convert_element_type_6 = None
        mul_628: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_5, convert_element_type_937)
        sum_82: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_628, [2], True);  mul_628 = None
        div_52: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul_5, 768);  mul_5 = None
        mul_629: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_52, sum_82);  div_52 = sum_82 = None
        sub_59: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_937, mul_629);  convert_element_type_937 = mul_629 = None
        mul_630: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_1);  sub_59 = rsqrt_1 = None
        convert_element_type_939: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_630, torch.bfloat16);  mul_630 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        add_297: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_288, convert_element_type_939);  add_288 = convert_element_type_939 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_631: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_297, select_1);  select_1 = None
        mul_632: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_297, cat);  cat = None
        sum_83: "bf16[]" = torch.ops.aten.sum.default(mul_632);  mul_632 = None
        convert_element_type_940: "f32[]" = torch.ops.prims.convert_element_type.default(sum_83, torch.float32);  sum_83 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_298: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_285, mul_631);  add_285 = mul_631 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        select_scatter_22: "f32[12]" = torch.ops.aten.select_scatter.default(full_default_13, convert_element_type_940, 0, 0);  full_default_13 = convert_element_type_940 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_299: "f32[12]" = torch.ops.aten.add.Tensor(add_286, select_scatter_22);  add_286 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        mul_633: "bf16[16, 2048, 768]" = torch.ops.aten.mul.Tensor(add_297, select);  add_297 = select = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:500 in forward, code: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        add_300: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_298, mul_633);  add_298 = mul_633 = None
        add_301: "f32[12]" = torch.ops.aten.add.Tensor(add_287, select_scatter_22);  add_287 = select_scatter_22 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:480 in forward, code: x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        slice_110: "bf16[16, 1, 768]" = torch.ops.aten.slice.Tensor(add_300, 1, 0, 1)
        slice_111: "bf16[16, 2047, 768]" = torch.ops.aten.slice.Tensor(add_300, 1, 1, 2048);  add_300 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:479 in forward, code: gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        convert_element_type_2: "bf16[1]" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_1: "bf16[16, 2047, 1]" = torch.ops.aten.view.default(mm, [16, 2047, 1]);  mm = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:479 in forward, code: gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        sigmoid: "bf16[16, 2047, 1]" = torch.ops.aten.sigmoid.default(view_1);  view_1 = None
        mul_1: "bf16[16, 2047, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sigmoid)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:480 in forward, code: x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        mul_635: "bf16[16, 2047, 768]" = torch.ops.aten.mul.Tensor(slice_111, mul_1);  mul_1 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32);  embedding = None
        mul: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = None
        convert_element_type_1: "bf16[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16)
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:480 in forward, code: x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        slice_7: "bf16[16, 2047, 768]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 1, 0, -1);  convert_element_type_1 = None
        mul_636: "bf16[16, 2047, 768]" = torch.ops.aten.mul.Tensor(slice_111, slice_7);  slice_7 = None
        sum_85: "bf16[16, 2047, 1]" = torch.ops.aten.sum.dim_IntList(mul_636, [2], True);  mul_636 = None
        slice_scatter_54: "bf16[16, 2048, 768]" = torch.ops.aten.slice_scatter.default(full_default_10, mul_635, 1, 0, -1);  mul_635 = None
        slice_scatter_55: "bf16[16, 2048, 768]" = torch.ops.aten.slice_scatter.default(full_default_10, slice_111, 1, 1, 9223372036854775807);  slice_111 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:480 in forward, code: x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        add_302: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(slice_scatter_54, slice_scatter_55);  slice_scatter_54 = slice_scatter_55 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:480 in forward, code: x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        slice_scatter_56: "bf16[16, 2048, 768]" = torch.ops.aten.slice_scatter.default(full_default_10, slice_110, 1, 0, 1);  slice_110 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:480 in forward, code: x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        add_303: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_302, slice_scatter_56);  add_302 = slice_scatter_56 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:479 in forward, code: gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        mul_637: "bf16[16, 2047, 1]" = torch.ops.aten.mul.Tensor(sum_85, convert_element_type_2);  convert_element_type_2 = None
        mul_638: "bf16[16, 2047, 1]" = torch.ops.aten.mul.Tensor(sum_85, sigmoid);  sum_85 = None
        sum_86: "bf16[1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_638, [0, 1], True);  mul_638 = None
        view_430: "bf16[1]" = torch.ops.aten.view.default(sum_86, [1]);  sum_86 = None
        convert_element_type_942: "f32[16, 2047, 1]" = torch.ops.prims.convert_element_type.default(mul_637, torch.float32);  mul_637 = None
        convert_element_type_943: "f32[16, 2047, 1]" = torch.ops.prims.convert_element_type.default(sigmoid, torch.float32);  sigmoid = None
        sub_60: "f32[16, 2047, 1]" = torch.ops.aten.sub.Tensor(1, convert_element_type_943)
        mul_639: "f32[16, 2047, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_943, sub_60);  convert_element_type_943 = sub_60 = None
        mul_640: "f32[16, 2047, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_942, mul_639);  convert_element_type_942 = mul_639 = None
        convert_element_type_944: "bf16[16, 2047, 1]" = torch.ops.prims.convert_element_type.default(mul_640, torch.bfloat16);  mul_640 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:50 in forward, code: return F.linear(x, self.weight.to(dtype=x.dtype))
        view_431: "bf16[32752, 1]" = torch.ops.aten.view.default(convert_element_type_944, [32752, 1]);  convert_element_type_944 = None
        permute_396: "bf16[1, 32752]" = torch.ops.aten.permute.default(view_431, [1, 0])
        mm_238: "bf16[1, 24]" = torch.ops.aten.mm.default(permute_396, view);  permute_396 = view = None
        mm_239: "bf16[32752, 24]" = torch.ops.aten.mm.default(view_431, permute_398);  view_431 = permute_398 = None
        view_432: "bf16[16, 2047, 24]" = torch.ops.aten.view.default(mm_239, [16, 2047, 24]);  mm_239 = None
        convert_element_type_949: "f32[1, 24]" = torch.ops.prims.convert_element_type.default(mm_238, torch.float32);  mm_238 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:479 in forward, code: gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        full_default_110: "bf16[16, 2047, 768]" = torch.ops.aten.full.default([16, 2047, 768], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_57: "bf16[16, 2047, 768]" = torch.ops.aten.slice_scatter.default(full_default_110, view_432, 2, 0, 24);  full_default_110 = view_432 = None
        slice_scatter_58: "bf16[16, 2048, 768]" = torch.ops.aten.slice_scatter.default(full_default_10, slice_scatter_57, 1, 1, 9223372036854775807);  full_default_10 = slice_scatter_57 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:479 in forward, code: gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        add_304: "bf16[16, 2048, 768]" = torch.ops.aten.add.Tensor(add_303, slice_scatter_58);  add_303 = slice_scatter_58 = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:479 in forward, code: gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        convert_element_type_950: "f32[1]" = torch.ops.prims.convert_element_type.default(view_430, torch.float32);  view_430 = None
        
         # File: /work/nvme/bffp/tmahmood1/cisc662/venv/lib/python3.10/site-packages/torch/nn/functional.py:2920 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_951: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(add_304, torch.float32);  add_304 = None
        mul_642: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(mul, convert_element_type_951)
        sum_87: "f32[16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [2], True);  mul_642 = None
        div_53: "f32[16, 2048, 768]" = torch.ops.aten.div.Tensor(mul, 768);  mul = None
        mul_643: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(div_53, sum_87);  div_53 = sum_87 = None
        sub_61: "f32[16, 2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_951, mul_643);  convert_element_type_951 = mul_643 = None
        mul_644: "f32[16, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt);  sub_61 = rsqrt = None
        
         # File: /u/tmahmood1/project/nanochat/nanochat/gpt.py:471 in forward, code: x = self.transformer.wte(idx) # embed current token
        convert_element_type_default: "f32[16, 2048, 768]" = torch.ops.prims.convert_element_type.default(mul_644, torch.float32);  mul_644 = None
        where_22: "f32[16, 2048, 768]" = torch.ops.aten.where.self(unsqueeze_8, full_default_1, convert_element_type_default);  unsqueeze_8 = full_default_1 = convert_element_type_default = None
        index_put_6: "f32[32768, 768]" = torch.ops.aten.index_put.default(full_default_12, [primals_1], where_22, True);  full_default_12 = primals_1 = where_22 = None
        convert_element_type_955: "bf16[32768, 768]" = torch.ops.prims.convert_element_type.default(index_put_6, torch.bfloat16);  index_put_6 = None
        return (None, None, None, convert_element_type_955, convert_element_type_950, convert_element_type_949, add_301, add_299, convert_element_type_936, convert_element_type_931, convert_element_type_926, convert_element_type_915, convert_element_type_907, convert_element_type_902, convert_element_type_895, convert_element_type_890, convert_element_type_885, convert_element_type_880, convert_element_type_875, convert_element_type_861, convert_element_type_853, convert_element_type_848, convert_element_type_838, convert_element_type_833, convert_element_type_828, convert_element_type_817, convert_element_type_809, convert_element_type_804, convert_element_type_797, convert_element_type_792, convert_element_type_787, convert_element_type_782, convert_element_type_777, convert_element_type_763, convert_element_type_755, convert_element_type_750, convert_element_type_740, convert_element_type_735, convert_element_type_730, convert_element_type_719, convert_element_type_711, convert_element_type_706, convert_element_type_699, convert_element_type_694, convert_element_type_689, convert_element_type_684, convert_element_type_679, convert_element_type_665, convert_element_type_657, convert_element_type_652, convert_element_type_642, convert_element_type_637, convert_element_type_632, convert_element_type_621, convert_element_type_613, convert_element_type_608, convert_element_type_601, convert_element_type_596, convert_element_type_591, convert_element_type_586, convert_element_type_581, convert_element_type_567, convert_element_type_559, convert_element_type_554, convert_element_type_544, convert_element_type_539, convert_element_type_534, convert_element_type_523, convert_element_type_515, convert_element_type_510, convert_element_type_503, convert_element_type_498, convert_element_type_493, convert_element_type_488, convert_element_type_483, convert_element_type_469, convert_element_type_461, convert_element_type_456, convert_element_type_446, convert_element_type_441, convert_element_type_436, convert_element_type_425, convert_element_type_417, convert_element_type_412, convert_element_type_405, convert_element_type_400, convert_element_type_395, convert_element_type_390, convert_element_type_385, convert_element_type_371, convert_element_type_363, convert_element_type_358, convert_element_type_353, convert_element_type_349, None)
        