
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_v2_768/gfx908/xc-nw4_wpe1_h1/c662_rmsnorm_fwd_v2.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <c662_rmsnorm_fwd_v2>:
	s_load_dwordx2 s[2:3], s[4:5], 0x18                        // 000000001700: C0060082 00000018
	s_lshl_b32 s8, s6, 3                                       // 000000001708: 8E088306
	v_lshrrev_b32_e32 v1, 5, v0                                // 00000000170C: 20020085
	v_or_b32_e32 v15, s8, v1                                   // 000000001710: 281E0208
	s_load_dwordx2 s[6:7], s[4:5], 0x0                         // 000000001714: C0060182 00000000
	s_waitcnt lgkmcnt(0)                                       // 00000000171C: BF8CC07F
	v_mul_lo_u32 v1, s3, v15                                   // 000000001720: D2850001 00021E03
	v_cmp_gt_i32_e64 s[0:1], s2, v15                           // 000000001728: D0C40000 00021E02
	v_mov_b32_e32 v5, 0                                        // 000000001730: 7E0A0280
	v_mov_b32_e32 v4, 0                                        // 000000001734: 7E080280
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001738: 2204029F
	v_lshlrev_b64 v[1:2], 1, v[1:2]                            // 00000000173C: D28F0001 00020281
	v_mov_b32_e32 v3, s7                                       // 000000001744: 7E060207
	v_add_co_u32_e32 v1, vcc, s6, v1                           // 000000001748: 32020206
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 00000000174C: 38040503
	v_lshlrev_b32_e32 v3, 3, v0                                // 000000001750: 24060083
	v_and_b32_e32 v3, 0xf8, v3                                 // 000000001754: 260606FF 000000F8
	v_lshlrev_b32_e32 v22, 1, v3                               // 00000000175C: 242C0681
	v_add_co_u32_e32 v13, vcc, v1, v22                         // 000000001760: 321A2D01
	v_addc_co_u32_e32 v14, vcc, 0, v2, vcc                     // 000000001764: 381C0480
	v_mov_b32_e32 v1, 0                                        // 000000001768: 7E020280
	v_mov_b32_e32 v2, 0                                        // 00000000176C: 7E040280
	v_mov_b32_e32 v3, 0                                        // 000000001770: 7E060280
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 000000001774: BE862000
	s_cbranch_execz 2                                          // 000000001778: BF880002 <c662_rmsnorm_fwd_v2+0x84>
	global_load_dwordx4 v[1:4], v[13:14], off                  // 00000000177C: DC5C8000 017F000D
	s_or_b64 exec, exec, s[6:7]                                // 000000001784: 87FE067E
	v_mov_b32_e32 v6, 0                                        // 000000001788: 7E0C0280
	v_mov_b32_e32 v7, 0                                        // 00000000178C: 7E0E0280
	v_mov_b32_e32 v8, 0                                        // 000000001790: 7E100280
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 000000001794: BE862000
	s_cbranch_execz 2                                          // 000000001798: BF880002 <c662_rmsnorm_fwd_v2+0xa4>
	global_load_dwordx4 v[5:8], v[13:14], off offset:512       // 00000000179C: DC5C8200 057F000D
	s_or_b64 exec, exec, s[6:7]                                // 0000000017A4: 87FE067E
	s_load_dword s3, s[4:5], 0x24                              // 0000000017A8: C00200C2 00000024
	v_mov_b32_e32 v9, 0                                        // 0000000017B0: 7E120280
	v_mov_b32_e32 v10, 0                                       // 0000000017B4: 7E140280
	v_mov_b32_e32 v11, 0                                       // 0000000017B8: 7E160280
	v_mov_b32_e32 v12, 0                                       // 0000000017BC: 7E180280
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 0000000017C0: BE862000
	s_cbranch_execz 2                                          // 0000000017C4: BF880002 <c662_rmsnorm_fwd_v2+0xd0>
	global_load_dwordx4 v[9:12], v[13:14], off offset:1024     // 0000000017C8: DC5C8400 097F000D
	s_or_b64 exec, exec, s[6:7]                                // 0000000017D0: 87FE067E
	s_waitcnt vmcnt(0)                                         // 0000000017D4: BF8C0F70
	v_and_b32_e32 v13, 0xffff0000, v5                          // 0000000017D8: 261A0AFF FFFF0000
	v_lshlrev_b32_e32 v14, 16, v5                              // 0000000017E0: 241C0A90
	v_and_b32_e32 v23, 0xffff0000, v1                          // 0000000017E4: 262E02FF FFFF0000
	v_lshlrev_b32_e32 v24, 16, v1                              // 0000000017EC: 24300290
	v_and_b32_e32 v25, 0xffff0000, v2                          // 0000000017F0: 263204FF FFFF0000
	v_lshlrev_b32_e32 v26, 16, v2                              // 0000000017F8: 24340490
	v_and_b32_e32 v27, 0xffff0000, v3                          // 0000000017FC: 263606FF FFFF0000
	v_lshlrev_b32_e32 v28, 16, v3                              // 000000001804: 24380690
	v_and_b32_e32 v29, 0xffff0000, v4                          // 000000001808: 263A08FF FFFF0000
	v_lshlrev_b32_e32 v30, 16, v4                              // 000000001810: 243C0890
	v_and_b32_e32 v1, 0xffff0000, v9                           // 000000001814: 260212FF FFFF0000
	v_lshlrev_b32_e32 v2, 16, v9                               // 00000000181C: 24041290
	v_and_b32_e32 v3, 0xffff0000, v10                          // 000000001820: 260614FF FFFF0000
	v_lshlrev_b32_e32 v4, 16, v10                              // 000000001828: 24081490
	v_mul_f32_e32 v9, v14, v14                                 // 00000000182C: 0A121D0E
	v_mul_f32_e32 v10, v13, v13                                // 000000001830: 0A141B0D
	v_fmac_f32_e32 v9, v24, v24                                // 000000001834: 76123118
	v_fmac_f32_e32 v10, v23, v23                               // 000000001838: 76142F17
	v_lshlrev_b32_e32 v17, 16, v6                              // 00000000183C: 24220C90
	v_fmac_f32_e32 v9, v2, v2                                  // 000000001840: 76120502
	v_fmac_f32_e32 v10, v1, v1                                 // 000000001844: 76140301
	v_add_f32_e32 v9, v9, v10                                  // 000000001848: 02121509
	v_mul_f32_e32 v10, v17, v17                                // 00000000184C: 0A142311
	v_fmac_f32_e32 v10, v26, v26                               // 000000001850: 7614351A
	v_and_b32_e32 v16, 0xffff0000, v6                          // 000000001854: 26200CFF FFFF0000
	v_fmac_f32_e32 v10, v4, v4                                 // 00000000185C: 76140904
	v_add_f32_e32 v9, v10, v9                                  // 000000001860: 0212130A
	v_mul_f32_e32 v10, v16, v16                                // 000000001864: 0A142110
	v_fmac_f32_e32 v10, v25, v25                               // 000000001868: 76143319
	v_lshlrev_b32_e32 v19, 16, v7                              // 00000000186C: 24260E90
	v_fmac_f32_e32 v10, v3, v3                                 // 000000001870: 76140703
	v_add_f32_e32 v9, v10, v9                                  // 000000001874: 0212130A
	v_mul_f32_e32 v10, v19, v19                                // 000000001878: 0A142713
	v_lshlrev_b32_e32 v6, 16, v11                              // 00000000187C: 240C1690
	v_fmac_f32_e32 v10, v28, v28                               // 000000001880: 7614391C
	v_and_b32_e32 v18, 0xffff0000, v7                          // 000000001884: 26240EFF FFFF0000
	v_fmac_f32_e32 v10, v6, v6                                 // 00000000188C: 76140D06
	v_add_f32_e32 v9, v10, v9                                  // 000000001890: 0212130A
	v_mul_f32_e32 v10, v18, v18                                // 000000001894: 0A142512
	v_and_b32_e32 v5, 0xffff0000, v11                          // 000000001898: 260A16FF FFFF0000
	v_fmac_f32_e32 v10, v27, v27                               // 0000000018A0: 7614371B
	v_lshlrev_b32_e32 v21, 16, v8                              // 0000000018A4: 242A1090
	v_fmac_f32_e32 v10, v5, v5                                 // 0000000018A8: 76140B05
	v_add_f32_e32 v9, v10, v9                                  // 0000000018AC: 0212130A
	v_mul_f32_e32 v10, v21, v21                                // 0000000018B0: 0A142B15
	v_and_b32_e32 v20, 0xffff0000, v8                          // 0000000018B4: 262810FF FFFF0000
	v_lshlrev_b32_e32 v8, 16, v12                              // 0000000018BC: 24101890
	v_fmac_f32_e32 v10, v30, v30                               // 0000000018C0: 76143D1E
	v_fmac_f32_e32 v10, v8, v8                                 // 0000000018C4: 76141108
	v_add_f32_e32 v9, v10, v9                                  // 0000000018C8: 0212130A
	v_mul_f32_e32 v10, v20, v20                                // 0000000018CC: 0A142914
	v_and_b32_e32 v7, 0xffff0000, v12                          // 0000000018D0: 260E18FF FFFF0000
	v_fmac_f32_e32 v10, v29, v29                               // 0000000018D8: 76143B1D
	v_fmac_f32_e32 v10, v7, v7                                 // 0000000018DC: 76140F07
	v_add_f32_e32 v9, v10, v9                                  // 0000000018E0: 0212130A
	ds_swizzle_b32 v10, v9 offset:swizzle(SWAP,16)             // 0000000018E4: D87A401F 0A000009
	s_mov_b32 s9, 0x44400000                                   // 0000000018EC: BE8900FF 44400000
	s_waitcnt lgkmcnt(0)                                       // 0000000018F4: BF8CC07F
	v_add_f32_e32 v9, v9, v10                                  // 0000000018F8: 02121509
	ds_swizzle_b32 v10, v9 offset:swizzle(SWAP,8)              // 0000000018FC: D87A201F 0A000009
	s_waitcnt lgkmcnt(0)                                       // 000000001904: BF8CC07F
	v_add_f32_e32 v9, v9, v10                                  // 000000001908: 02121509
	ds_swizzle_b32 v10, v9 offset:swizzle(SWAP,4)              // 00000000190C: D87A101F 0A000009
	s_waitcnt lgkmcnt(0)                                       // 000000001914: BF8CC07F
	v_add_f32_e32 v9, v9, v10                                  // 000000001918: 02121509
	ds_swizzle_b32 v10, v9 offset:swizzle(SWAP,2)              // 00000000191C: D87A081F 0A000009
	s_waitcnt lgkmcnt(0)                                       // 000000001924: BF8CC07F
	v_add_f32_e32 v9, v9, v10                                  // 000000001928: 02121509
	ds_swizzle_b32 v10, v9 offset:swizzle(SWAP,1)              // 00000000192C: D87A041F 0A000009
	s_waitcnt lgkmcnt(0)                                       // 000000001934: BF8CC07F
	v_add_f32_e32 v9, v9, v10                                  // 000000001938: 02121509
	v_div_scale_f32 v10, s[6:7], s9, s9, v9                    // 00000000193C: D1E0060A 04241209
	v_div_scale_f32 v11, vcc, v9, s9, v9                       // 000000001944: D1E06A0B 04241309
	v_rcp_f32_e32 v12, v10                                     // 00000000194C: 7E18450A
	v_fma_f32 v31, -v10, v12, 1.0                              // 000000001950: D1CB001F 23CA190A
	v_fmac_f32_e32 v12, v31, v12                               // 000000001958: 7618191F
	v_mul_f32_e32 v31, v11, v12                                // 00000000195C: 0A3E190B
	v_fma_f32 v32, -v10, v31, v11                              // 000000001960: D1CB0020 242E3F0A
	v_fmac_f32_e32 v31, v32, v12                               // 000000001968: 763E1920
	v_fma_f32 v10, -v10, v31, v11                              // 00000000196C: D1CB000A 242E3F0A
	v_div_fmas_f32 v10, v10, v12, v31                          // 000000001974: D1E2000A 047E190A
	v_div_fixup_f32 v9, v10, s9, v9                            // 00000000197C: D1DE0009 0424130A
	v_add_f32_e32 v9, s3, v9                                   // 000000001984: 02121203
	v_rsq_f32_e32 v9, v9                                       // 000000001988: 7E124909
	v_and_b32_e32 v10, 0xe0, v0                                // 00000000198C: 261400FF 000000E0
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 000000001994: BE862000
	s_cbranch_execz 246                                        // 000000001998: BF8800F6 <c662_rmsnorm_fwd_v2+0x674>
	s_load_dword s3, s[4:5], 0x20                              // 00000000199C: C00200C2 00000020
	s_load_dwordx2 s[0:1], s[4:5], 0x8                         // 0000000019A4: C0060002 00000008
	v_mov_b32_e32 v33, 0x7fff                                  // 0000000019AC: 7E4202FF 00007FFF
	s_mov_b32 s9, 0x5040100                                    // 0000000019B4: BE8900FF 05040100
	v_mul_f32_e32 v1, v9, v1                                   // 0000000019BC: 0A020309
	s_waitcnt lgkmcnt(0)                                       // 0000000019C0: BF8CC07F
	v_mul_lo_u32 v11, s3, v15                                  // 0000000019C4: D285000B 00021E03
	v_mov_b32_e32 v15, s1                                      // 0000000019CC: 7E1E0201
	s_movk_i32 s3, 0x7fff                                      // 0000000019D0: B0037FFF
	v_ashrrev_i32_e32 v12, 31, v11                             // 0000000019D4: 2218169F
	v_lshlrev_b64 v[11:12], 1, v[11:12]                        // 0000000019D8: D28F000B 00021681
	v_add_co_u32_e32 v11, vcc, s0, v11                         // 0000000019E0: 32161600
	v_addc_co_u32_e32 v12, vcc, v15, v12, vcc                  // 0000000019E4: 3818190F
	v_add_co_u32_e32 v31, vcc, v11, v22                        // 0000000019E8: 323E2D0B
	v_mul_f32_e32 v15, v9, v25                                 // 0000000019EC: 0A1E3309
	v_mul_f32_e32 v22, v9, v26                                 // 0000000019F0: 0A2C3509
	v_mul_f32_e32 v25, v9, v29                                 // 0000000019F4: 0A323B09
	v_mul_f32_e32 v26, v9, v30                                 // 0000000019F8: 0A343D09
	v_addc_co_u32_e32 v32, vcc, 0, v12, vcc                    // 0000000019FC: 38401880
	v_mul_f32_e32 v11, v9, v23                                 // 000000001A00: 0A162F09
	v_mul_f32_e32 v12, v9, v24                                 // 000000001A04: 0A183109
	v_mul_f32_e32 v23, v9, v27                                 // 000000001A08: 0A2E3709
	v_mul_f32_e32 v24, v9, v28                                 // 000000001A0C: 0A303909
	v_bfe_u32 v27, v26, 16, 1                                  // 000000001A10: D1C8001B 0205211A
	v_bfe_u32 v28, v25, 16, 1                                  // 000000001A18: D1C8001C 02052119
	v_bfe_u32 v29, v24, 16, 1                                  // 000000001A20: D1C8001D 02052118
	v_bfe_u32 v30, v23, 16, 1                                  // 000000001A28: D1C8001E 02052117
	v_add3_u32 v27, v26, v27, s3                               // 000000001A30: D1FF001B 000E371A
	v_add3_u32 v28, v25, v28, s3                               // 000000001A38: D1FF001C 000E3919
	v_lshrrev_b32_e32 v27, 16, v27                             // 000000001A40: 20363690
	v_cmp_o_f32_e32 vcc, v26, v26                              // 000000001A44: 7C8E351A
	v_lshrrev_b32_e32 v28, 16, v28                             // 000000001A48: 20383890
	v_cmp_o_f32_e64 s[0:1], v25, v25                           // 000000001A4C: D0470000 00023319
	v_add3_u32 v25, v23, v30, s3                               // 000000001A54: D1FF0019 000E3D17
	v_add3_u32 v29, v24, v29, s3                               // 000000001A5C: D1FF001D 000E3B18
	v_bfe_u32 v26, v22, 16, 1                                  // 000000001A64: D1C8001A 02052116
	v_lshrrev_b32_e32 v29, 16, v29                             // 000000001A6C: 203A3A90
	v_lshrrev_b32_e32 v30, 16, v25                             // 000000001A70: 203C3290
	v_cndmask_b32_e32 v25, v33, v27, vcc                       // 000000001A74: 00323721
	v_cndmask_b32_e64 v27, v33, v28, s[0:1]                    // 000000001A78: D100001B 00023921
	v_cmp_o_f32_e32 vcc, v24, v24                              // 000000001A80: 7C8E3118
	v_cmp_o_f32_e64 s[0:1], v23, v23                           // 000000001A84: D0470000 00022F17
	v_bfe_u32 v34, v15, 16, 1                                  // 000000001A8C: D1C80022 0205210F
	v_cndmask_b32_e32 v23, v33, v29, vcc                       // 000000001A94: 002E3B21
	v_cndmask_b32_e64 v24, v33, v30, s[0:1]                    // 000000001A98: D1000018 00023D21
	v_add3_u32 v26, v22, v26, s3                               // 000000001AA0: D1FF001A 000E3516
	v_perm_b32 v24, v24, v23, s9                               // 000000001AA8: D1ED0018 00262F18
	v_bfe_u32 v23, v12, 16, 1                                  // 000000001AB0: D1C80017 0205210C
	v_add3_u32 v28, v15, v34, s3                               // 000000001AB8: D1FF001C 000E450F
	v_lshrrev_b32_e32 v26, 16, v26                             // 000000001AC0: 20343490
	v_cmp_o_f32_e32 vcc, v22, v22                              // 000000001AC4: 7C8E2D16
	v_perm_b32 v25, v27, v25, s9                               // 000000001AC8: D1ED0019 0026331B
	v_bfe_u32 v27, v11, 16, 1                                  // 000000001AD0: D1C8001B 0205210B
	v_add3_u32 v23, v12, v23, s3                               // 000000001AD8: D1FF0017 000E2F0C
	v_lshrrev_b32_e32 v28, 16, v28                             // 000000001AE0: 20383890
	v_cndmask_b32_e32 v22, v33, v26, vcc                       // 000000001AE4: 002C3521
	v_cmp_o_f32_e32 vcc, v15, v15                              // 000000001AE8: 7C8E1F0F
	v_add3_u32 v27, v11, v27, s3                               // 000000001AEC: D1FF001B 000E370B
	v_lshrrev_b32_e32 v29, 16, v23                             // 000000001AF4: 203A2E90
	v_cndmask_b32_e32 v15, v33, v28, vcc                       // 000000001AF8: 001E3921
	v_cmp_o_f32_e32 vcc, v12, v12                              // 000000001AFC: 7C8E190C
	v_lshrrev_b32_e32 v27, 16, v27                             // 000000001B00: 20363690
	v_cndmask_b32_e32 v12, v33, v29, vcc                       // 000000001B04: 00183B21
	v_cmp_o_f32_e32 vcc, v11, v11                              // 000000001B08: 7C8E170B
	v_cndmask_b32_e32 v11, v33, v27, vcc                       // 000000001B0C: 00163721
	v_perm_b32 v23, v15, v22, s9                               // 000000001B10: D1ED0017 00262D0F
	v_perm_b32 v22, v11, v12, s9                               // 000000001B18: D1ED0016 0026190B
	v_mul_f32_e32 v11, v9, v13                                 // 000000001B20: 0A161B09
	v_mul_f32_e32 v13, v9, v18                                 // 000000001B24: 0A1A2509
	v_mul_f32_e32 v18, v9, v21                                 // 000000001B28: 0A242B09
	v_mul_f32_e32 v15, v9, v14                                 // 000000001B2C: 0A1E1D09
	v_mul_f32_e32 v12, v9, v16                                 // 000000001B30: 0A182109
	v_mul_f32_e32 v16, v9, v17                                 // 000000001B34: 0A202309
	v_mul_f32_e32 v17, v9, v19                                 // 000000001B38: 0A222709
	v_mul_f32_e32 v14, v9, v20                                 // 000000001B3C: 0A1C2909
	v_bfe_u32 v19, v18, 16, 1                                  // 000000001B40: D1C80013 02052112
	v_bfe_u32 v20, v14, 16, 1                                  // 000000001B48: D1C80014 0205210E
	v_add3_u32 v19, v18, v19, s3                               // 000000001B50: D1FF0013 000E2712
	v_bfe_u32 v21, v17, 16, 1                                  // 000000001B58: D1C80015 02052111
	v_add3_u32 v20, v14, v20, s3                               // 000000001B60: D1FF0014 000E290E
	v_lshrrev_b32_e32 v19, 16, v19                             // 000000001B68: 20262690
	v_cmp_o_f32_e32 vcc, v18, v18                              // 000000001B6C: 7C8E2512
	global_store_dwordx4 v[31:32], v[22:25], off               // 000000001B70: DC7C8000 007F161F
	v_add3_u32 v21, v17, v21, s3                               // 000000001B78: D1FF0015 000E2B11
	v_bfe_u32 v22, v13, 16, 1                                  // 000000001B80: D1C80016 0205210D
	v_lshrrev_b32_e32 v20, 16, v20                             // 000000001B88: 20282890
	v_cndmask_b32_e32 v18, v33, v19, vcc                       // 000000001B8C: 00242721
	v_cmp_o_f32_e32 vcc, v14, v14                              // 000000001B90: 7C8E1D0E
	v_bfe_u32 v23, v16, 16, 1                                  // 000000001B94: D1C80017 02052110
	v_add3_u32 v22, v13, v22, s3                               // 000000001B9C: D1FF0016 000E2D0D
	v_lshrrev_b32_e32 v21, 16, v21                             // 000000001BA4: 202A2A90
	v_cndmask_b32_e32 v14, v33, v20, vcc                       // 000000001BA8: 001C2921
	v_cmp_o_f32_e32 vcc, v17, v17                              // 000000001BAC: 7C8E2311
	v_bfe_u32 v24, v12, 16, 1                                  // 000000001BB0: D1C80018 0205210C
	v_add3_u32 v23, v16, v23, s3                               // 000000001BB8: D1FF0017 000E2F10
	v_lshrrev_b32_e32 v22, 16, v22                             // 000000001BC0: 202C2C90
	v_cndmask_b32_e32 v17, v33, v21, vcc                       // 000000001BC4: 00222B21
	v_cmp_o_f32_e32 vcc, v13, v13                              // 000000001BC8: 7C8E1B0D
	v_bfe_u32 v25, v15, 16, 1                                  // 000000001BCC: D1C80019 0205210F
	v_add3_u32 v24, v12, v24, s3                               // 000000001BD4: D1FF0018 000E310C
	v_lshrrev_b32_e32 v23, 16, v23                             // 000000001BDC: 202E2E90
	v_cndmask_b32_e32 v13, v33, v22, vcc                       // 000000001BE0: 001A2D21
	v_cmp_o_f32_e32 vcc, v16, v16                              // 000000001BE4: 7C8E2110
	v_bfe_u32 v26, v11, 16, 1                                  // 000000001BE8: D1C8001A 0205210B
	v_add3_u32 v25, v15, v25, s3                               // 000000001BF0: D1FF0019 000E330F
	v_lshrrev_b32_e32 v24, 16, v24                             // 000000001BF8: 20303090
	v_cndmask_b32_e32 v16, v33, v23, vcc                       // 000000001BFC: 00202F21
	v_cmp_o_f32_e32 vcc, v12, v12                              // 000000001C00: 7C8E190C
	v_add3_u32 v26, v11, v26, s3                               // 000000001C04: D1FF001A 000E350B
	v_lshrrev_b32_e32 v25, 16, v25                             // 000000001C0C: 20323290
	v_cndmask_b32_e32 v12, v33, v24, vcc                       // 000000001C10: 00183121
	v_cmp_o_f32_e32 vcc, v15, v15                              // 000000001C14: 7C8E1F0F
	v_lshrrev_b32_e32 v26, 16, v26                             // 000000001C18: 20343490
	v_cndmask_b32_e32 v15, v33, v25, vcc                       // 000000001C1C: 001E3321
	v_cmp_o_f32_e32 vcc, v11, v11                              // 000000001C20: 7C8E170B
	v_cndmask_b32_e32 v11, v33, v26, vcc                       // 000000001C24: 00163521
	v_perm_b32 v14, v14, v18, s9                               // 000000001C28: D1ED000E 0026250E
	v_perm_b32 v13, v13, v17, s9                               // 000000001C30: D1ED000D 0026230D
	v_perm_b32 v12, v12, v16, s9                               // 000000001C38: D1ED000C 0026210C
	v_perm_b32 v11, v11, v15, s9                               // 000000001C40: D1ED000B 00261F0B
	global_store_dwordx4 v[31:32], v[11:14], off offset:512    // 000000001C48: DC7C8200 007F0B1F
	v_bfe_u32 v18, v1, 16, 1                                   // 000000001C50: D1C80012 02052101
	v_mul_f32_e32 v11, v9, v2                                  // 000000001C58: 0A160509
	v_mul_f32_e32 v2, v9, v3                                   // 000000001C5C: 0A040709
	v_mul_f32_e32 v3, v9, v5                                   // 000000001C60: 0A060B09
	v_mul_f32_e32 v5, v9, v6                                   // 000000001C64: 0A0A0D09
	v_mul_f32_e32 v6, v9, v8                                   // 000000001C68: 0A0C1109
	v_mul_f32_e32 v12, v9, v4                                  // 000000001C6C: 0A180909
	v_mul_f32_e32 v4, v9, v7                                   // 000000001C70: 0A080F09
	v_bfe_u32 v7, v6, 16, 1                                    // 000000001C74: D1C80007 02052106
	v_bfe_u32 v8, v4, 16, 1                                    // 000000001C7C: D1C80008 02052104
	v_add3_u32 v7, v6, v7, s3                                  // 000000001C84: D1FF0007 000E0F06
	v_cmp_o_f32_e32 vcc, v6, v6                                // 000000001C8C: 7C8E0D06
	v_bfe_u32 v13, v5, 16, 1                                   // 000000001C90: D1C8000D 02052105
	v_add3_u32 v8, v4, v8, s3                                  // 000000001C98: D1FF0008 000E1104
	v_cndmask_b32_sdwa v6, v33, v7, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001CA0: 000C0EF9 05060621
	v_cmp_o_f32_e32 vcc, v4, v4                                // 000000001CA8: 7C8E0904
	v_bfe_u32 v14, v3, 16, 1                                   // 000000001CAC: D1C8000E 02052103
	v_add3_u32 v13, v5, v13, s3                                // 000000001CB4: D1FF000D 000E1B05
	v_cndmask_b32_sdwa v4, v33, v8, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001CBC: 000810F9 05060621
	v_cmp_o_f32_e32 vcc, v5, v5                                // 000000001CC4: 7C8E0B05
	v_bfe_u32 v15, v12, 16, 1                                  // 000000001CC8: D1C8000F 0205210C
	v_add3_u32 v14, v3, v14, s3                                // 000000001CD0: D1FF000E 000E1D03
	v_cndmask_b32_sdwa v5, v33, v13, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001CD8: 000A1AF9 05060621
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000001CE0: 7C8E0703
	v_bfe_u32 v16, v2, 16, 1                                   // 000000001CE4: D1C80010 02052102
	v_add3_u32 v15, v12, v15, s3                               // 000000001CEC: D1FF000F 000E1F0C
	v_cndmask_b32_sdwa v3, v33, v14, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001CF4: 00061CF9 05060621
	v_cmp_o_f32_e32 vcc, v12, v12                              // 000000001CFC: 7C8E190C
	v_bfe_u32 v17, v11, 16, 1                                  // 000000001D00: D1C80011 0205210B
	v_add3_u32 v16, v2, v16, s3                                // 000000001D08: D1FF0010 000E2102
	v_perm_b32 v3, v3, v5, s9                                  // 000000001D10: D1ED0003 00260B03
	v_cndmask_b32_sdwa v5, v33, v15, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D18: 000A1EF9 05060621
	v_cmp_o_f32_e32 vcc, v2, v2                                // 000000001D20: 7C8E0502
	v_add3_u32 v17, v11, v17, s3                               // 000000001D24: D1FF0011 000E230B
	v_cndmask_b32_sdwa v2, v33, v16, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D2C: 000420F9 05060621
	v_cmp_o_f32_e32 vcc, v11, v11                              // 000000001D34: 7C8E170B
	v_add3_u32 v18, v1, v18, s3                                // 000000001D38: D1FF0012 000E2501
	v_perm_b32 v2, v2, v5, s9                                  // 000000001D40: D1ED0002 00260B02
	v_cndmask_b32_sdwa v5, v33, v17, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D48: 000A22F9 05060621
	v_cmp_o_f32_e32 vcc, v1, v1                                // 000000001D50: 7C8E0301
	v_cndmask_b32_sdwa v1, v33, v18, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D54: 000224F9 05060621
	v_perm_b32 v4, v4, v6, s9                                  // 000000001D5C: D1ED0004 00260D04
	v_perm_b32 v1, v1, v5, s9                                  // 000000001D64: D1ED0001 00260B01
	global_store_dwordx4 v[31:32], v[1:4], off offset:1024     // 000000001D6C: DC7C8400 007F011F
	s_or_b64 exec, exec, s[6:7]                                // 000000001D74: 87FE067E
	v_and_b32_e32 v3, 7, v0                                    // 000000001D78: 26060087
	v_or_b32_e32 v1, s8, v3                                    // 000000001D7C: 28020608
	v_and_b32_e32 v0, 0xf8, v0                                 // 000000001D80: 260000FF 000000F8
	v_cmp_gt_i32_e32 vcc, s2, v1                               // 000000001D88: 7D880202
	v_lshrrev_b32_e32 v2, 3, v10                               // 000000001D8C: 20041483
	v_cmp_eq_u32_e64 s[0:1], 0, v0                             // 000000001D90: D0CA0000 00020080
	v_add_u32_e32 v2, 0, v2                                    // 000000001D98: 68040480
	s_and_b64 s[0:1], s[0:1], vcc                              // 000000001D9C: 86806A00
	ds_write_b32 v2, v9                                        // 000000001DA0: D81A0000 00000902
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000001DA8: BF8C0070
	s_barrier                                                  // 000000001DAC: BF8A0000
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001DB0: BE822000
	s_cbranch_execz 15                                         // 000000001DB4: BF88000F <c662_rmsnorm_fwd_v2+0x6f4>
	s_load_dwordx2 s[0:1], s[4:5], 0x10                        // 000000001DB8: C0060002 00000010
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001DC0: 2204029F
	v_lshl_add_u32 v3, v3, 2, 0                                // 000000001DC4: D1FD0003 02010503
	v_lshlrev_b64 v[0:1], 2, v[1:2]                            // 000000001DCC: D28F0000 00020282
	ds_read_b32 v2, v3                                         // 000000001DD4: D86C0000 02000003
	s_waitcnt lgkmcnt(0)                                       // 000000001DDC: BF8CC07F
	v_mov_b32_e32 v3, s1                                       // 000000001DE0: 7E060201
	v_add_co_u32_e32 v0, vcc, s0, v0                           // 000000001DE4: 32000000
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc                     // 000000001DE8: 38020303
	global_store_dword v[0:1], v2, off                         // 000000001DEC: DC708000 007F0200
	s_endpgm                                                   // 000000001DF4: BF810000
