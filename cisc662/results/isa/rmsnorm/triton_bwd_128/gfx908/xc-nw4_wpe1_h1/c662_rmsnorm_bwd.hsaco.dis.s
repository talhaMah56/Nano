
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_bwd_128/gfx908/xc-nw4_wpe1_h1/c662_rmsnorm_bwd.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <c662_rmsnorm_bwd>:
	s_load_dword s0, s[4:5], 0x20                              // 000000001700: C0020002 00000020
	v_lshrrev_b32_e32 v1, 4, v0                                // 000000001708: 20020084
	v_lshl_or_b32 v9, s6, 4, v1                                // 00000000170C: D2000009 04050806
	v_mov_b32_e32 v1, 0                                        // 000000001714: 7E020280
	v_mov_b32_e32 v10, 0                                       // 000000001718: 7E140280
	s_waitcnt lgkmcnt(0)                                       // 00000000171C: BF8CC07F
	v_cmp_gt_i32_e32 vcc, s0, v9                               // 000000001720: 7D881200
	v_mov_b32_e32 v11, 0                                       // 000000001724: 7E160280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001728: BE82206A
	s_cbranch_execz 15                                         // 00000000172C: BF88000F <c662_rmsnorm_bwd+0x6c>
	s_load_dwordx2 s[0:1], s[4:5], 0x10                        // 000000001730: C0060002 00000010
	v_ashrrev_i32_e32 v10, 31, v9                              // 000000001738: 2214129F
	v_lshlrev_b64 v[2:3], 2, v[9:10]                           // 00000000173C: D28F0002 00021282
	s_waitcnt lgkmcnt(0)                                       // 000000001744: BF8CC07F
	v_mov_b32_e32 v4, s1                                       // 000000001748: 7E080201
	v_add_co_u32_e64 v2, s[0:1], s0, v2                        // 00000000174C: D1190002 00020400
	v_addc_co_u32_e64 v3, s[0:1], v4, v3, s[0:1]               // 000000001754: D11C0003 00020704
	global_load_dword v10, v[2:3], off                         // 00000000175C: DC508000 0A7F0002
	s_waitcnt vmcnt(0)                                         // 000000001764: BF8C0F70
	v_mov_b32_e32 v11, v10                                     // 000000001768: 7E16030A
	s_or_b64 exec, exec, s[2:3]                                // 00000000176C: 87FE027E
	v_lshlrev_b32_e32 v0, 3, v0                                // 000000001770: 24000083
	v_and_b32_e32 v0, 0x78, v0                                 // 000000001774: 260000FF 00000078
	v_lshlrev_b32_e32 v0, 1, v0                                // 00000000177C: 24000081
	v_mov_b32_e32 v2, 0                                        // 000000001780: 7E040280
	v_mov_b32_e32 v3, 0                                        // 000000001784: 7E060280
	v_mov_b32_e32 v4, 0                                        // 000000001788: 7E080280
	s_and_saveexec_b64 s[2:3], vcc                             // 00000000178C: BE82206A
	s_cbranch_execz 21                                         // 000000001790: BF880015 <c662_rmsnorm_bwd+0xe8>
	s_load_dword s6, s[4:5], 0x28                              // 000000001794: C0020182 00000028
	s_load_dwordx2 s[0:1], s[4:5], 0x8                         // 00000000179C: C0060002 00000008
	s_waitcnt lgkmcnt(0)                                       // 0000000017A4: BF8CC07F
	v_mul_lo_u32 v1, s6, v9                                    // 0000000017A8: D2850001 00021206
	v_mov_b32_e32 v3, s1                                       // 0000000017B0: 7E060201
	v_ashrrev_i32_e32 v2, 31, v1                               // 0000000017B4: 2204029F
	v_lshlrev_b64 v[1:2], 1, v[1:2]                            // 0000000017B8: D28F0001 00020281
	v_add_co_u32_e64 v1, s[0:1], s0, v1                        // 0000000017C0: D1190001 00020200
	v_addc_co_u32_e64 v2, s[0:1], v3, v2, s[0:1]               // 0000000017C8: D11C0002 00020503
	v_add_co_u32_e64 v1, s[0:1], v1, v0                        // 0000000017D0: D1190001 00020101
	v_addc_co_u32_e64 v2, s[0:1], 0, v2, s[0:1]                // 0000000017D8: D11C0002 00020480
	global_load_dwordx4 v[1:4], v[1:2], off                    // 0000000017E0: DC5C8000 017F0001
	s_or_b64 exec, exec, s[2:3]                                // 0000000017E8: 87FE027E
	v_mov_b32_e32 v5, 0                                        // 0000000017EC: 7E0A0280
	v_mov_b32_e32 v6, 0                                        // 0000000017F0: 7E0C0280
	v_mov_b32_e32 v7, 0                                        // 0000000017F4: 7E0E0280
	v_mov_b32_e32 v8, 0                                        // 0000000017F8: 7E100280
	s_and_saveexec_b64 s[2:3], vcc                             // 0000000017FC: BE82206A
	s_cbranch_execz 21                                         // 000000001800: BF880015 <c662_rmsnorm_bwd+0x158>
	s_load_dword s6, s[4:5], 0x24                              // 000000001804: C0020182 00000024
	s_load_dwordx2 s[0:1], s[4:5], 0x0                         // 00000000180C: C0060002 00000000
	s_waitcnt lgkmcnt(0)                                       // 000000001814: BF8CC07F
	v_mul_lo_u32 v5, s6, v9                                    // 000000001818: D2850005 00021206
	v_mov_b32_e32 v7, s1                                       // 000000001820: 7E0E0201
	v_ashrrev_i32_e32 v6, 31, v5                               // 000000001824: 220C0A9F
	v_lshlrev_b64 v[5:6], 1, v[5:6]                            // 000000001828: D28F0005 00020A81
	v_add_co_u32_e64 v5, s[0:1], s0, v5                        // 000000001830: D1190005 00020A00
	v_addc_co_u32_e64 v6, s[0:1], v7, v6, s[0:1]               // 000000001838: D11C0006 00020D07
	v_add_co_u32_e64 v5, s[0:1], v5, v0                        // 000000001840: D1190005 00020105
	v_addc_co_u32_e64 v6, s[0:1], 0, v6, s[0:1]                // 000000001848: D11C0006 00020C80
	global_load_dwordx4 v[5:8], v[5:6], off                    // 000000001850: DC5C8000 057F0005
	s_or_b64 exec, exec, s[2:3]                                // 000000001858: 87FE027E
	s_waitcnt vmcnt(0)                                         // 00000000185C: BF8C0F70
	v_and_b32_e32 v13, 0xffff0000, v1                          // 000000001860: 261A02FF FFFF0000
	v_lshlrev_b32_e32 v14, 16, v1                              // 000000001868: 241C0290
	v_and_b32_e32 v20, 0xffff0000, v3                          // 00000000186C: 262806FF FFFF0000
	v_and_b32_e32 v12, 0xffff0000, v5                          // 000000001874: 26180AFF FFFF0000
	v_mul_f32_e32 v17, v11, v13                                // 00000000187C: 0A221B0B
	v_lshlrev_b32_e32 v18, 16, v2                              // 000000001880: 24240490
	v_lshlrev_b32_e32 v21, 16, v3                              // 000000001884: 242A0690
	v_lshlrev_b32_e32 v16, 16, v5                              // 000000001888: 24200A90
	v_and_b32_e32 v1, 0xffff0000, v8                           // 00000000188C: 260210FF FFFF0000
	v_lshlrev_b32_e32 v3, 16, v8                               // 000000001894: 24061090
	v_mul_f32_e32 v19, v10, v14                                // 000000001898: 0A261D0A
	v_mul_f32_e32 v8, v11, v20                                 // 00000000189C: 0A10290B
	v_mul_f32_e32 v20, v17, v12                                // 0000000018A0: 0A281911
	v_and_b32_e32 v15, 0xffff0000, v2                          // 0000000018A4: 261E04FF FFFF0000
	v_and_b32_e32 v22, 0xffff0000, v4                          // 0000000018AC: 262C08FF FFFF0000
	v_lshlrev_b32_e32 v23, 16, v4                              // 0000000018B4: 242E0890
	v_and_b32_e32 v4, 0xffff0000, v6                           // 0000000018B8: 26080CFF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v6                               // 0000000018C0: 240C0C90
	v_mul_f32_e32 v18, v10, v18                                // 0000000018C4: 0A24250A
	v_fmac_f32_e32 v20, v19, v16                               // 0000000018C8: 76282113
	v_mul_f32_e32 v14, v11, v15                                // 0000000018CC: 0A1C1F0B
	v_fmac_f32_e32 v20, v18, v6                                // 0000000018D0: 76280D12
	v_lshlrev_b32_e32 v5, 16, v7                               // 0000000018D4: 240A0E90
	v_mul_f32_e32 v15, v10, v21                                // 0000000018D8: 0A1E2B0A
	v_fmac_f32_e32 v20, v14, v4                                // 0000000018DC: 7628090E
	v_and_b32_e32 v2, 0xffff0000, v7                           // 0000000018E0: 26040EFF FFFF0000
	v_fmac_f32_e32 v20, v15, v5                                // 0000000018E8: 76280B0F
	v_mul_f32_e32 v13, v10, v23                                // 0000000018EC: 0A1A2F0A
	v_fmac_f32_e32 v20, v8, v2                                 // 0000000018F0: 76280508
	v_mul_f32_e32 v7, v11, v22                                 // 0000000018F4: 0A0E2D0B
	v_fmac_f32_e32 v20, v13, v3                                // 0000000018F8: 7628070D
	v_fmac_f32_e32 v20, v7, v1                                 // 0000000018FC: 76280307
	ds_swizzle_b32 v21, v20 offset:swizzle(SWAP,8)             // 000000001900: D87A201F 15000014
	s_waitcnt lgkmcnt(0)                                       // 000000001908: BF8CC07F
	v_add_f32_e32 v20, v20, v21                                // 00000000190C: 02282B14
	ds_swizzle_b32 v21, v20 offset:swizzle(SWAP,4)             // 000000001910: D87A101F 15000014
	s_waitcnt lgkmcnt(0)                                       // 000000001918: BF8CC07F
	v_add_f32_e32 v20, v20, v21                                // 00000000191C: 02282B14
	ds_swizzle_b32 v21, v20 offset:swizzle(SWAP,2)             // 000000001920: D87A081F 15000014
	s_waitcnt lgkmcnt(0)                                       // 000000001928: BF8CC07F
	v_add_f32_e32 v20, v20, v21                                // 00000000192C: 02282B14
	ds_swizzle_b32 v21, v20 offset:swizzle(SWAP,1)             // 000000001930: D87A041F 15000014
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001938: BE80206A
	s_cbranch_execz 119                                        // 00000000193C: BF880077 <c662_rmsnorm_bwd+0x41c>
	s_load_dword s2, s[4:5], 0x2c                              // 000000001940: C0020082 0000002C
	s_load_dwordx2 s[0:1], s[4:5], 0x18                        // 000000001948: C0060002 00000018
	s_waitcnt lgkmcnt(0)                                       // 000000001950: BF8CC07F
	v_add_f32_e32 v21, v20, v21                                // 000000001954: 022A2B14
	s_mov_b32 s3, 0x5040100                                    // 000000001958: BE8300FF 05040100
	v_mul_lo_u32 v20, s2, v9                                   // 000000001960: D2850014 00021202
	v_mul_f32_e32 v9, 0x3c000000, v21                          // 000000001968: 0A122AFF 3C000000
	v_fma_f32 v16, -v19, v9, v16                               // 000000001970: D1CB0010 24421313
	v_mov_b32_e32 v22, s1                                      // 000000001978: 7E2C0201
	v_ashrrev_i32_e32 v21, 31, v20                             // 00000000197C: 222A289F
	v_lshlrev_b64 v[19:20], 1, v[20:21]                        // 000000001980: D28F0013 00022881
	v_fma_f32 v3, -v13, v9, v3                                 // 000000001988: D1CB0003 240E130D
	v_add_co_u32_e32 v19, vcc, s0, v19                         // 000000001990: 32262600
	v_addc_co_u32_e32 v20, vcc, v22, v20, vcc                  // 000000001994: 38282916
	v_fma_f32 v6, -v18, v9, v6                                 // 000000001998: D1CB0006 241A1312
	v_fma_f32 v14, -v14, v9, v4                                // 0000000019A0: D1CB000E 2412130E
	v_fma_f32 v15, -v15, v9, v5                                // 0000000019A8: D1CB000F 2416130F
	v_fma_f32 v2, -v8, v9, v2                                  // 0000000019B0: D1CB0002 240A1308
	v_fma_f32 v1, -v7, v9, v1                                  // 0000000019B8: D1CB0001 24061307
	v_add_co_u32_e32 v4, vcc, v19, v0                          // 0000000019C0: 32080113
	v_mul_f32_e32 v3, v10, v3                                  // 0000000019C4: 0A06070A
	v_fma_f32 v12, -v17, v9, v12                               // 0000000019C8: D1CB000C 24321311
	v_addc_co_u32_e32 v5, vcc, 0, v20, vcc                     // 0000000019D0: 380A2880
	v_mul_f32_e32 v7, v10, v16                                 // 0000000019D4: 0A0E210A
	v_mul_f32_e32 v6, v10, v6                                  // 0000000019D8: 0A0C0D0A
	v_mul_f32_e32 v2, v11, v2                                  // 0000000019DC: 0A04050B
	v_mul_f32_e32 v9, v10, v15                                 // 0000000019E0: 0A121F0A
	v_mul_f32_e32 v1, v11, v1                                  // 0000000019E4: 0A02030B
	v_bfe_u32 v10, v3, 16, 1                                   // 0000000019E8: D1C8000A 02052103
	s_movk_i32 s2, 0x7fff                                      // 0000000019F0: B0027FFF
	v_mul_f32_e32 v0, v11, v12                                 // 0000000019F4: 0A00190B
	v_mul_f32_e32 v8, v11, v14                                 // 0000000019F8: 0A101D0B
	v_bfe_u32 v11, v1, 16, 1                                   // 0000000019FC: D1C8000B 02052101
	v_add3_u32 v10, v3, v10, s2                                // 000000001A04: D1FF000A 000A1503
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000001A0C: 7C8E0703
	v_bfe_u32 v3, v2, 16, 1                                    // 000000001A10: D1C80003 02052102
	v_add3_u32 v13, v2, v3, s2                                 // 000000001A18: D1FF000D 000A0702
	v_add3_u32 v3, v1, v11, s2                                 // 000000001A20: D1FF0003 000A1701
	v_cmp_o_f32_e64 s[0:1], v1, v1                             // 000000001A28: D0470000 00020301
	v_mov_b32_e32 v11, 0x7fff                                  // 000000001A30: 7E1602FF 00007FFF
	v_bfe_u32 v12, v9, 16, 1                                   // 000000001A38: D1C8000C 02052109
	v_cndmask_b32_sdwa v1, v11, v10, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001A40: 000214F9 0506060B
	s_mov_b64 vcc, s[0:1]                                      // 000000001A48: BEEA0100
	v_add3_u32 v12, v9, v12, s2                                // 000000001A4C: D1FF000C 000A1909
	v_cndmask_b32_sdwa v3, v11, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001A54: 000606F9 0506060B
	v_cmp_o_f32_e32 vcc, v9, v9                                // 000000001A5C: 7C8E1309
	v_cmp_o_f32_e64 s[0:1], v2, v2                             // 000000001A60: D0470000 00020502
	v_perm_b32 v3, v3, v1, s3                                  // 000000001A68: D1ED0003 000E0303
	v_cndmask_b32_sdwa v1, v11, v12, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001A70: 000218F9 0506060B
	s_mov_b64 vcc, s[0:1]                                      // 000000001A78: BEEA0100
	v_cndmask_b32_sdwa v2, v11, v13, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001A7C: 00041AF9 0506060B
	v_perm_b32 v2, v2, v1, s3                                  // 000000001A84: D1ED0002 000E0302
	v_bfe_u32 v1, v6, 16, 1                                    // 000000001A8C: D1C80001 02052106
	v_bfe_u32 v9, v8, 16, 1                                    // 000000001A94: D1C80009 02052108
	v_add3_u32 v1, v6, v1, s2                                  // 000000001A9C: D1FF0001 000A0306
	v_cmp_o_f32_e32 vcc, v6, v6                                // 000000001AA4: 7C8E0D06
	v_cmp_o_f32_e64 s[0:1], v8, v8                             // 000000001AA8: D0470000 00021108
	v_bfe_u32 v6, v7, 16, 1                                    // 000000001AB0: D1C80006 02052107
	v_add3_u32 v9, v8, v9, s2                                  // 000000001AB8: D1FF0009 000A1308
	v_cndmask_b32_sdwa v1, v11, v1, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001AC0: 000202F9 0506060B
	s_mov_b64 vcc, s[0:1]                                      // 000000001AC8: BEEA0100
	v_bfe_u32 v8, v0, 16, 1                                    // 000000001ACC: D1C80008 02052100
	v_add3_u32 v6, v7, v6, s2                                  // 000000001AD4: D1FF0006 000A0D07
	v_cndmask_b32_sdwa v9, v11, v9, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001ADC: 001212F9 0506060B
	v_cmp_o_f32_e32 vcc, v7, v7                                // 000000001AE4: 7C8E0F07
	v_add3_u32 v8, v0, v8, s2                                  // 000000001AE8: D1FF0008 000A1100
	v_cndmask_b32_sdwa v6, v11, v6, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001AF0: 000C0CF9 0506060B
	v_cmp_o_f32_e32 vcc, v0, v0                                // 000000001AF8: 7C8E0100
	v_cndmask_b32_sdwa v0, v11, v8, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001AFC: 000010F9 0506060B
	v_perm_b32 v1, v9, v1, s3                                  // 000000001B04: D1ED0001 000E0309
	v_perm_b32 v0, v0, v6, s3                                  // 000000001B0C: D1ED0000 000E0D00
	global_store_dwordx4 v[4:5], v[0:3], off                   // 000000001B14: DC7C8000 007F0004
	s_endpgm                                                   // 000000001B1C: BF810000
