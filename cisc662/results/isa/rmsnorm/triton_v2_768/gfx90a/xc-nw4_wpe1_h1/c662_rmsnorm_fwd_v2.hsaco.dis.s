
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_v2_768/gfx90a/xc-nw4_wpe1_h1/c662_rmsnorm_fwd_v2.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <c662_rmsnorm_fwd_v2>:
	s_load_dwordx2 s[6:7], s[4:5], 0x0                         // 000000001700: C0060182 00000000
	s_load_dwordx8 s[8:15], s[4:5], 0x8                        // 000000001708: C00E0202 00000008
	s_waitcnt lgkmcnt(0)                                       // 000000001710: BF8CC07F
	s_branch 58                                                // 000000001714: BF82003A <c662_rmsnorm_fwd_v2+0x100>
	s_nop 0                                                    // 000000001718: BF800000
	s_nop 0                                                    // 00000000171C: BF800000
	s_nop 0                                                    // 000000001720: BF800000
	s_nop 0                                                    // 000000001724: BF800000
	s_nop 0                                                    // 000000001728: BF800000
	s_nop 0                                                    // 00000000172C: BF800000
	s_nop 0                                                    // 000000001730: BF800000
	s_nop 0                                                    // 000000001734: BF800000
	s_nop 0                                                    // 000000001738: BF800000
	s_nop 0                                                    // 00000000173C: BF800000
	s_nop 0                                                    // 000000001740: BF800000
	s_nop 0                                                    // 000000001744: BF800000
	s_nop 0                                                    // 000000001748: BF800000
	s_nop 0                                                    // 00000000174C: BF800000
	s_nop 0                                                    // 000000001750: BF800000
	s_nop 0                                                    // 000000001754: BF800000
	s_nop 0                                                    // 000000001758: BF800000
	s_nop 0                                                    // 00000000175C: BF800000
	s_nop 0                                                    // 000000001760: BF800000
	s_nop 0                                                    // 000000001764: BF800000
	s_nop 0                                                    // 000000001768: BF800000
	s_nop 0                                                    // 00000000176C: BF800000
	s_nop 0                                                    // 000000001770: BF800000
	s_nop 0                                                    // 000000001774: BF800000
	s_nop 0                                                    // 000000001778: BF800000
	s_nop 0                                                    // 00000000177C: BF800000
	s_nop 0                                                    // 000000001780: BF800000
	s_nop 0                                                    // 000000001784: BF800000
	s_nop 0                                                    // 000000001788: BF800000
	s_nop 0                                                    // 00000000178C: BF800000
	s_nop 0                                                    // 000000001790: BF800000
	s_nop 0                                                    // 000000001794: BF800000
	s_nop 0                                                    // 000000001798: BF800000
	s_nop 0                                                    // 00000000179C: BF800000
	s_nop 0                                                    // 0000000017A0: BF800000
	s_nop 0                                                    // 0000000017A4: BF800000
	s_nop 0                                                    // 0000000017A8: BF800000
	s_nop 0                                                    // 0000000017AC: BF800000
	s_nop 0                                                    // 0000000017B0: BF800000
	s_nop 0                                                    // 0000000017B4: BF800000
	s_nop 0                                                    // 0000000017B8: BF800000
	s_nop 0                                                    // 0000000017BC: BF800000
	s_nop 0                                                    // 0000000017C0: BF800000
	s_nop 0                                                    // 0000000017C4: BF800000
	s_nop 0                                                    // 0000000017C8: BF800000
	s_nop 0                                                    // 0000000017CC: BF800000
	s_nop 0                                                    // 0000000017D0: BF800000
	s_nop 0                                                    // 0000000017D4: BF800000
	s_nop 0                                                    // 0000000017D8: BF800000
	s_nop 0                                                    // 0000000017DC: BF800000
	s_nop 0                                                    // 0000000017E0: BF800000
	s_nop 0                                                    // 0000000017E4: BF800000
	s_nop 0                                                    // 0000000017E8: BF800000
	s_nop 0                                                    // 0000000017EC: BF800000
	s_nop 0                                                    // 0000000017F0: BF800000
	s_nop 0                                                    // 0000000017F4: BF800000
	s_nop 0                                                    // 0000000017F8: BF800000
	s_nop 0                                                    // 0000000017FC: BF800000
	s_lshl_b32 s4, s16, 3                                      // 000000001800: 8E048310
	v_lshrrev_b32_e32 v1, 5, v0                                // 000000001804: 20020085
	v_or_b32_e32 v1, s4, v1                                    // 000000001808: 28020204
	v_mul_lo_u32 v2, s13, v1                                   // 00000000180C: D2850002 0002020D
	v_ashrrev_i32_e32 v3, 31, v2                               // 000000001814: 2206049F
	v_lshlrev_b64 v[2:3], 1, v[2:3]                            // 000000001818: D28F0002 00020481
	v_mov_b32_e32 v4, s7                                       // 000000001820: 7E080207
	v_add_co_u32_e32 v2, vcc, s6, v2                           // 000000001824: 32040406
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc                     // 000000001828: 38060704
	v_lshlrev_b32_e32 v4, 3, v0                                // 00000000182C: 24080083
	v_and_b32_e32 v4, 0xf8, v4                                 // 000000001830: 260808FF 000000F8
	v_lshlrev_b32_e32 v30, 1, v4                               // 000000001838: 243C0881
	v_add_co_u32_e32 v14, vcc, v2, v30                         // 00000000183C: 321C3D02
	v_cmp_gt_i32_e64 s[0:1], s12, v1                           // 000000001840: D0C40000 0002020C
	v_addc_co_u32_e32 v15, vcc, 0, v3, vcc                     // 000000001848: 381E0680
	v_mov_b32_e32 v6, 0                                        // 00000000184C: 7E0C0280
	v_mov_b32_e32 v2, 0                                        // 000000001850: 7E040280
	v_mov_b32_e32 v3, 0                                        // 000000001854: 7E060280
	v_mov_b32_e32 v4, 0                                        // 000000001858: 7E080280
	v_mov_b32_e32 v5, 0                                        // 00000000185C: 7E0A0280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001860: BE822000
	s_cbranch_execz 2                                          // 000000001864: BF880002 <c662_rmsnorm_fwd_v2+0x170>
	global_load_dwordx4 v[2:5], v[14:15], off                  // 000000001868: DC5C8000 027F000E
	s_or_b64 exec, exec, s[2:3]                                // 000000001870: 87FE027E
	v_mov_b32_e32 v7, 0                                        // 000000001874: 7E0E0280
	v_mov_b32_e32 v8, 0                                        // 000000001878: 7E100280
	v_mov_b32_e32 v9, 0                                        // 00000000187C: 7E120280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001880: BE822000
	s_cbranch_execz 2                                          // 000000001884: BF880002 <c662_rmsnorm_fwd_v2+0x190>
	global_load_dwordx4 v[6:9], v[14:15], off offset:512       // 000000001888: DC5C8200 067F000E
	s_or_b64 exec, exec, s[2:3]                                // 000000001890: 87FE027E
	v_mov_b32_e32 v10, 0                                       // 000000001894: 7E140280
	v_mov_b32_e32 v11, 0                                       // 000000001898: 7E160280
	v_mov_b32_e32 v12, 0                                       // 00000000189C: 7E180280
	v_mov_b32_e32 v13, 0                                       // 0000000018A0: 7E1A0280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 0000000018A4: BE822000
	s_cbranch_execz 2                                          // 0000000018A8: BF880002 <c662_rmsnorm_fwd_v2+0x1b4>
	global_load_dwordx4 v[10:13], v[14:15], off offset:1024    // 0000000018AC: DC5C8400 0A7F000E
	s_or_b64 exec, exec, s[2:3]                                // 0000000018B4: 87FE027E
	s_waitcnt vmcnt(0)                                         // 0000000018B8: BF8C0F70
	v_and_b32_e32 v15, 0xffff0000, v6                          // 0000000018BC: 261E0CFF FFFF0000
	v_lshlrev_b32_e32 v14, 16, v6                              // 0000000018C4: 241C0C90
	v_and_b32_e32 v17, 0xffff0000, v7                          // 0000000018C8: 26220EFF FFFF0000
	v_lshlrev_b32_e32 v16, 16, v7                              // 0000000018D0: 24200E90
	v_and_b32_e32 v19, 0xffff0000, v8                          // 0000000018D4: 262610FF FFFF0000
	v_lshlrev_b32_e32 v18, 16, v8                              // 0000000018DC: 24241090
	v_and_b32_e32 v21, 0xffff0000, v9                          // 0000000018E0: 262A12FF FFFF0000
	v_lshlrev_b32_e32 v20, 16, v9                              // 0000000018E8: 24281290
	v_and_b32_e32 v23, 0xffff0000, v2                          // 0000000018EC: 262E04FF FFFF0000
	v_lshlrev_b32_e32 v22, 16, v2                              // 0000000018F4: 242C0490
	v_pk_mul_f32 v[8:9], v[14:15], v[14:15]                    // 0000000018F8: D3B14008 18021D0E
	v_and_b32_e32 v25, 0xffff0000, v3                          // 000000001900: 263206FF FFFF0000
	v_lshlrev_b32_e32 v24, 16, v3                              // 000000001908: 24300690
	v_and_b32_e32 v27, 0xffff0000, v4                          // 00000000190C: 263608FF FFFF0000
	v_lshlrev_b32_e32 v26, 16, v4                              // 000000001914: 24340890
	v_and_b32_e32 v29, 0xffff0000, v5                          // 000000001918: 263A0AFF FFFF0000
	v_lshlrev_b32_e32 v28, 16, v5                              // 000000001920: 24380A90
	v_and_b32_e32 v3, 0xffff0000, v10                          // 000000001924: 260614FF FFFF0000
	v_lshlrev_b32_e32 v2, 16, v10                              // 00000000192C: 24041490
	v_and_b32_e32 v5, 0xffff0000, v11                          // 000000001930: 260A16FF FFFF0000
	v_lshlrev_b32_e32 v4, 16, v11                              // 000000001938: 24081690
	v_and_b32_e32 v7, 0xffff0000, v12                          // 00000000193C: 260E18FF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v12                              // 000000001944: 240C1890
	v_and_b32_e32 v11, 0xffff0000, v13                         // 000000001948: 26161AFF FFFF0000
	v_lshlrev_b32_e32 v10, 16, v13                             // 000000001950: 24141A90
	v_pk_mul_f32 v[12:13], v[16:17], v[16:17]                  // 000000001954: D3B1400C 18022110
	v_pk_fma_f32 v[8:9], v[22:23], v[22:23], v[8:9]            // 00000000195C: D3B04008 1C222D16
	v_pk_fma_f32 v[12:13], v[24:25], v[24:25], v[12:13]        // 000000001964: D3B0400C 1C323118
	v_pk_fma_f32 v[8:9], v[2:3], v[2:3], v[8:9]                // 00000000196C: D3B04008 1C220502
	v_pk_mul_f32 v[32:33], v[18:19], v[18:19]                  // 000000001974: D3B14020 18022512
	v_pk_fma_f32 v[12:13], v[4:5], v[4:5], v[12:13]            // 00000000197C: D3B0400C 1C320904
	v_add_f32_e32 v8, v8, v9                                   // 000000001984: 02101308
	v_pk_fma_f32 v[32:33], v[26:27], v[26:27], v[32:33]        // 000000001988: D3B04020 1C82351A
	v_add_f32_e32 v8, v12, v8                                  // 000000001990: 0210110C
	v_pk_mul_f32 v[34:35], v[20:21], v[20:21]                  // 000000001994: D3B14022 18022914
	v_pk_fma_f32 v[32:33], v[6:7], v[6:7], v[32:33]            // 00000000199C: D3B04020 1C820D06
	v_add_f32_e32 v8, v13, v8                                  // 0000000019A4: 0210110D
	v_pk_fma_f32 v[34:35], v[28:29], v[28:29], v[34:35]        // 0000000019A8: D3B04022 1C8A391C
	v_add_f32_e32 v8, v32, v8                                  // 0000000019B0: 02101120
	v_pk_fma_f32 v[34:35], v[10:11], v[10:11], v[34:35]        // 0000000019B4: D3B04022 1C8A150A
	v_add_f32_e32 v8, v33, v8                                  // 0000000019BC: 02101121
	v_add_f32_e32 v8, v34, v8                                  // 0000000019C0: 02101122
	v_add_f32_e32 v8, v35, v8                                  // 0000000019C4: 02101123
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,16)              // 0000000019C8: D87A401F 09000008
	s_mov_b32 s5, 0x44400000                                   // 0000000019D0: BE8500FF 44400000
	s_waitcnt lgkmcnt(0)                                       // 0000000019D8: BF8CC07F
	v_add_f32_e32 v8, v8, v9                                   // 0000000019DC: 02101308
	v_mov_b32_e32 v9, v8                                       // 0000000019E0: 7E120308
	s_nop 1                                                    // 0000000019E4: BF800001
	v_mov_b32_dpp v9, v9 row_shr:8 row_mask:0xf bank_mask:0xc  // 0000000019E8: 7E1202FA FC011809
	s_nop 1                                                    // 0000000019F0: BF800001
	v_mov_b32_dpp v9, v8 row_shl:8 row_mask:0xf bank_mask:0x3  // 0000000019F4: 7E1202FA F3010808
	v_add_f32_e32 v8, v8, v9                                   // 0000000019FC: 02101308
	v_mov_b32_e32 v9, v8                                       // 000000001A00: 7E120308
	s_nop 1                                                    // 000000001A04: BF800001
	v_mov_b32_dpp v9, v9 row_shr:4 row_mask:0xf bank_mask:0xa  // 000000001A08: 7E1202FA FA011409
	s_nop 1                                                    // 000000001A10: BF800001
	v_mov_b32_dpp v9, v8 row_shl:4 row_mask:0xf bank_mask:0x5  // 000000001A14: 7E1202FA F5010408
	v_add_f32_e32 v8, v8, v9                                   // 000000001A1C: 02101308
	v_mov_b32_e32 v9, v8                                       // 000000001A20: 7E120308
	s_nop 1                                                    // 000000001A24: BF800001
	v_mov_b32_dpp v9, v9 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000001A28: 7E1202FA FF004E09
	v_add_f32_e32 v8, v8, v9                                   // 000000001A30: 02101308
	v_mov_b32_e32 v9, v8                                       // 000000001A34: 7E120308
	s_nop 1                                                    // 000000001A38: BF800001
	v_mov_b32_dpp v9, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 000000001A3C: 7E1202FA FF00B109
	v_add_f32_e32 v8, v8, v9                                   // 000000001A44: 02101308
	v_div_scale_f32 v9, s[2:3], s5, s5, v8                     // 000000001A48: D1E00209 04200A05
	v_rcp_f32_e32 v12, v9                                      // 000000001A50: 7E184509
	v_fma_f32 v13, -v9, v12, 1.0                               // 000000001A54: D1CB000D 23CA1909
	v_fmac_f32_e32 v12, v13, v12                               // 000000001A5C: 7618190D
	v_div_scale_f32 v13, vcc, v8, s5, v8                       // 000000001A60: D1E06A0D 04200B08
	v_mul_f32_e32 v31, v13, v12                                // 000000001A68: 0A3E190D
	v_fma_f32 v32, -v9, v31, v13                               // 000000001A6C: D1CB0020 24363F09
	v_fmac_f32_e32 v31, v32, v12                               // 000000001A74: 763E1920
	v_fma_f32 v9, -v9, v31, v13                                // 000000001A78: D1CB0009 24363F09
	v_div_fmas_f32 v9, v9, v12, v31                            // 000000001A80: D1E20009 047E1909
	v_div_fixup_f32 v8, v9, s5, v8                             // 000000001A88: D1DE0008 04200B09
	v_add_f32_e32 v8, s15, v8                                  // 000000001A90: 0210100F
	v_rsq_f32_e32 v8, v8                                       // 000000001A94: 7E104908
	v_and_b32_e32 v9, 0xe0, v0                                 // 000000001A98: 261200FF 000000E0
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001AA0: BE822000
	s_cbranch_execz 237                                        // 000000001AA4: BF8800ED <c662_rmsnorm_fwd_v2+0x75c>
	v_mul_lo_u32 v12, s14, v1                                  // 000000001AA8: D285000C 0002020E
	v_ashrrev_i32_e32 v13, 31, v12                             // 000000001AB0: 221A189F
	v_lshlrev_b64 v[12:13], 1, v[12:13]                        // 000000001AB4: D28F000C 00021881
	v_mov_b32_e32 v1, s9                                       // 000000001ABC: 7E020209
	v_add_co_u32_e32 v12, vcc, s8, v12                         // 000000001AC0: 32181808
	v_addc_co_u32_e32 v1, vcc, v1, v13, vcc                    // 000000001AC4: 38021B01
	v_add_co_u32_e32 v30, vcc, v12, v30                        // 000000001AC8: 323C3D0C
	v_pk_mul_f32 v[12:13], v[8:9], v[22:23] op_sel_hi:[0,1]    // 000000001ACC: D3B1400C 10022D08
	v_pk_mul_f32 v[22:23], v[8:9], v[24:25] op_sel_hi:[0,1]    // 000000001AD4: D3B14016 10023108
	v_pk_mul_f32 v[24:25], v[8:9], v[28:29] op_sel_hi:[0,1]    // 000000001ADC: D3B14018 10023908
	v_addc_co_u32_e32 v31, vcc, 0, v1, vcc                     // 000000001AE4: 383E0280
	v_bfe_u32 v1, v24, 16, 1                                   // 000000001AE8: D1C80001 02052118
	s_movk_i32 s0, 0x7fff                                      // 000000001AF0: B0007FFF
	v_pk_mul_f32 v[26:27], v[8:9], v[26:27] op_sel_hi:[0,1]    // 000000001AF4: D3B1401A 10023508
	v_bfe_u32 v28, v25, 16, 1                                  // 000000001AFC: D1C8001C 02052119
	v_add3_u32 v1, v24, v1, s0                                 // 000000001B04: D1FF0001 00020318
	v_bfe_u32 v29, v26, 16, 1                                  // 000000001B0C: D1C8001D 0205211A
	v_add3_u32 v28, v25, v28, s0                               // 000000001B14: D1FF001C 00023919
	v_lshrrev_b32_e32 v1, 16, v1                               // 000000001B1C: 20020290
	v_mov_b32_e32 v37, 0x7fff                                  // 000000001B20: 7E4A02FF 00007FFF
	v_cmp_o_f32_e32 vcc, v24, v24                              // 000000001B28: 7C8E3118
	v_bfe_u32 v32, v27, 16, 1                                  // 000000001B2C: D1C80020 0205211B
	v_add3_u32 v29, v26, v29, s0                               // 000000001B34: D1FF001D 00023B1A
	v_lshrrev_b32_e32 v28, 16, v28                             // 000000001B3C: 20383890
	v_cndmask_b32_e32 v1, v37, v1, vcc                         // 000000001B40: 00020325
	v_cmp_o_f32_e32 vcc, v25, v25                              // 000000001B44: 7C8E3319
	v_bfe_u32 v33, v22, 16, 1                                  // 000000001B48: D1C80021 02052116
	v_add3_u32 v32, v27, v32, s0                               // 000000001B50: D1FF0020 0002411B
	v_lshrrev_b32_e32 v29, 16, v29                             // 000000001B58: 203A3A90
	v_cndmask_b32_e32 v24, v37, v28, vcc                       // 000000001B5C: 00303925
	s_mov_b32 s1, 0x5040100                                    // 000000001B60: BE8100FF 05040100
	v_cmp_o_f32_e32 vcc, v26, v26                              // 000000001B68: 7C8E351A
	v_bfe_u32 v34, v23, 16, 1                                  // 000000001B6C: D1C80022 02052117
	v_add3_u32 v33, v22, v33, s0                               // 000000001B74: D1FF0021 00024316
	v_lshrrev_b32_e32 v32, 16, v32                             // 000000001B7C: 20404090
	v_perm_b32 v25, v24, v1, s1                                // 000000001B80: D1ED0019 00060318
	v_cndmask_b32_e32 v1, v37, v29, vcc                        // 000000001B88: 00023B25
	v_cmp_o_f32_e32 vcc, v27, v27                              // 000000001B8C: 7C8E371B
	v_bfe_u32 v35, v12, 16, 1                                  // 000000001B90: D1C80023 0205210C
	v_add3_u32 v34, v23, v34, s0                               // 000000001B98: D1FF0022 00024517
	v_lshrrev_b32_e32 v33, 16, v33                             // 000000001BA0: 20424290
	v_cndmask_b32_e32 v24, v37, v32, vcc                       // 000000001BA4: 00304125
	v_cmp_o_f32_e32 vcc, v22, v22                              // 000000001BA8: 7C8E2D16
	v_bfe_u32 v36, v13, 16, 1                                  // 000000001BAC: D1C80024 0205210D
	v_add3_u32 v35, v12, v35, s0                               // 000000001BB4: D1FF0023 0002470C
	v_lshrrev_b32_e32 v34, 16, v34                             // 000000001BBC: 20444490
	v_perm_b32 v24, v24, v1, s1                                // 000000001BC0: D1ED0018 00060318
	v_cndmask_b32_e32 v1, v37, v33, vcc                        // 000000001BC8: 00024325
	v_cmp_o_f32_e32 vcc, v23, v23                              // 000000001BCC: 7C8E2F17
	v_add3_u32 v36, v13, v36, s0                               // 000000001BD0: D1FF0024 0002490D
	v_lshrrev_b32_e32 v35, 16, v35                             // 000000001BD8: 20464690
	v_cndmask_b32_e32 v22, v37, v34, vcc                       // 000000001BDC: 002C4525
	v_cmp_o_f32_e32 vcc, v12, v12                              // 000000001BE0: 7C8E190C
	v_lshrrev_b32_e32 v36, 16, v36                             // 000000001BE4: 20484890
	v_perm_b32 v23, v22, v1, s1                                // 000000001BE8: D1ED0017 00060316
	v_cndmask_b32_e32 v1, v37, v35, vcc                        // 000000001BF0: 00024725
	v_cmp_o_f32_e32 vcc, v13, v13                              // 000000001BF4: 7C8E1B0D
	v_cndmask_b32_e32 v12, v37, v36, vcc                       // 000000001BF8: 00184925
	v_perm_b32 v22, v12, v1, s1                                // 000000001BFC: D1ED0016 0006030C
	global_store_dwordx4 v[30:31], v[22:25], off               // 000000001C04: DC7C8000 007F161E
	v_pk_mul_f32 v[12:13], v[8:9], v[16:17] op_sel_hi:[0,1]    // 000000001C0C: D3B1400C 10022108
	v_pk_mul_f32 v[22:23], v[8:9], v[14:15] op_sel_hi:[0,1]    // 000000001C14: D3B14016 10021D08
	v_pk_mul_f32 v[14:15], v[8:9], v[20:21] op_sel_hi:[0,1]    // 000000001C1C: D3B1400E 10022908
	v_bfe_u32 v1, v14, 16, 1                                   // 000000001C24: D1C80001 0205210E
	v_pk_mul_f32 v[16:17], v[8:9], v[18:19] op_sel_hi:[0,1]    // 000000001C2C: D3B14010 10022508
	v_bfe_u32 v18, v15, 16, 1                                  // 000000001C34: D1C80012 0205210F
	v_add3_u32 v1, v14, v1, s0                                 // 000000001C3C: D1FF0001 0002030E
	v_bfe_u32 v19, v16, 16, 1                                  // 000000001C44: D1C80013 02052110
	v_add3_u32 v18, v15, v18, s0                               // 000000001C4C: D1FF0012 0002250F
	v_lshrrev_b32_e32 v1, 16, v1                               // 000000001C54: 20020290
	v_cmp_o_f32_e32 vcc, v14, v14                              // 000000001C58: 7C8E1D0E
	v_bfe_u32 v20, v17, 16, 1                                  // 000000001C5C: D1C80014 02052111
	v_add3_u32 v19, v16, v19, s0                               // 000000001C64: D1FF0013 00022710
	v_lshrrev_b32_e32 v18, 16, v18                             // 000000001C6C: 20242490
	v_cndmask_b32_e32 v1, v37, v1, vcc                         // 000000001C70: 00020325
	v_cmp_o_f32_e32 vcc, v15, v15                              // 000000001C74: 7C8E1F0F
	v_bfe_u32 v21, v12, 16, 1                                  // 000000001C78: D1C80015 0205210C
	v_add3_u32 v20, v17, v20, s0                               // 000000001C80: D1FF0014 00022911
	v_lshrrev_b32_e32 v19, 16, v19                             // 000000001C88: 20262690
	v_cndmask_b32_e32 v14, v37, v18, vcc                       // 000000001C8C: 001C2525
	v_cmp_o_f32_e32 vcc, v16, v16                              // 000000001C90: 7C8E2110
	v_bfe_u32 v24, v13, 16, 1                                  // 000000001C94: D1C80018 0205210D
	v_add3_u32 v21, v12, v21, s0                               // 000000001C9C: D1FF0015 00022B0C
	v_lshrrev_b32_e32 v20, 16, v20                             // 000000001CA4: 20282890
	v_perm_b32 v15, v14, v1, s1                                // 000000001CA8: D1ED000F 0006030E
	v_cndmask_b32_e32 v1, v37, v19, vcc                        // 000000001CB0: 00022725
	v_cmp_o_f32_e32 vcc, v17, v17                              // 000000001CB4: 7C8E2311
	v_bfe_u32 v25, v22, 16, 1                                  // 000000001CB8: D1C80019 02052116
	v_add3_u32 v24, v13, v24, s0                               // 000000001CC0: D1FF0018 0002310D
	v_lshrrev_b32_e32 v21, 16, v21                             // 000000001CC8: 202A2A90
	v_cndmask_b32_e32 v14, v37, v20, vcc                       // 000000001CCC: 001C2925
	v_cmp_o_f32_e32 vcc, v12, v12                              // 000000001CD0: 7C8E190C
	v_bfe_u32 v26, v23, 16, 1                                  // 000000001CD4: D1C8001A 02052117
	v_add3_u32 v25, v22, v25, s0                               // 000000001CDC: D1FF0019 00023316
	v_lshrrev_b32_e32 v24, 16, v24                             // 000000001CE4: 20303090
	v_perm_b32 v14, v14, v1, s1                                // 000000001CE8: D1ED000E 0006030E
	v_cndmask_b32_e32 v1, v37, v21, vcc                        // 000000001CF0: 00022B25
	v_cmp_o_f32_e32 vcc, v13, v13                              // 000000001CF4: 7C8E1B0D
	v_add3_u32 v26, v23, v26, s0                               // 000000001CF8: D1FF001A 00023517
	v_lshrrev_b32_e32 v25, 16, v25                             // 000000001D00: 20323290
	v_cndmask_b32_e32 v12, v37, v24, vcc                       // 000000001D04: 00183125
	v_cmp_o_f32_e32 vcc, v22, v22                              // 000000001D08: 7C8E2D16
	v_lshrrev_b32_e32 v26, 16, v26                             // 000000001D0C: 20343490
	v_perm_b32 v13, v12, v1, s1                                // 000000001D10: D1ED000D 0006030C
	v_cndmask_b32_e32 v1, v37, v25, vcc                        // 000000001D18: 00023325
	v_cmp_o_f32_e32 vcc, v23, v23                              // 000000001D1C: 7C8E2F17
	v_cndmask_b32_e32 v12, v37, v26, vcc                       // 000000001D20: 00183525
	v_perm_b32 v12, v12, v1, s1                                // 000000001D24: D1ED000C 0006030C
	global_store_dwordx4 v[30:31], v[12:15], off offset:512    // 000000001D2C: DC7C8200 007F0C1E
	v_pk_mul_f32 v[6:7], v[8:9], v[6:7] op_sel_hi:[0,1]        // 000000001D34: D3B14006 10020D08
	v_pk_mul_f32 v[12:13], v[8:9], v[2:3] op_sel_hi:[0,1]      // 000000001D3C: D3B1400C 10020508
	v_pk_mul_f32 v[2:3], v[8:9], v[4:5] op_sel_hi:[0,1]        // 000000001D44: D3B14002 10020908
	v_pk_mul_f32 v[4:5], v[8:9], v[10:11] op_sel_hi:[0,1]      // 000000001D4C: D3B14004 10021508
	v_bfe_u32 v1, v4, 16, 1                                    // 000000001D54: D1C80001 02052104
	v_bfe_u32 v10, v5, 16, 1                                   // 000000001D5C: D1C8000A 02052105
	v_add3_u32 v1, v4, v1, s0                                  // 000000001D64: D1FF0001 00020304
	v_cmp_o_f32_e32 vcc, v4, v4                                // 000000001D6C: 7C8E0904
	v_bfe_u32 v11, v6, 16, 1                                   // 000000001D70: D1C8000B 02052106
	v_add3_u32 v10, v5, v10, s0                                // 000000001D78: D1FF000A 00021505
	v_cndmask_b32_sdwa v1, v37, v1, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D80: 000202F9 05060625
	v_cmp_o_f32_e32 vcc, v5, v5                                // 000000001D88: 7C8E0B05
	v_bfe_u32 v14, v7, 16, 1                                   // 000000001D8C: D1C8000E 02052107
	v_add3_u32 v11, v6, v11, s0                                // 000000001D94: D1FF000B 00021706
	v_cndmask_b32_sdwa v4, v37, v10, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D9C: 000814F9 05060625
	v_cmp_o_f32_e32 vcc, v6, v6                                // 000000001DA4: 7C8E0D06
	v_bfe_u32 v15, v2, 16, 1                                   // 000000001DA8: D1C8000F 02052102
	v_add3_u32 v14, v7, v14, s0                                // 000000001DB0: D1FF000E 00021D07
	v_perm_b32 v5, v4, v1, s1                                  // 000000001DB8: D1ED0005 00060304
	v_cndmask_b32_sdwa v1, v37, v11, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001DC0: 000216F9 05060625
	v_cmp_o_f32_e32 vcc, v7, v7                                // 000000001DC8: 7C8E0F07
	v_bfe_u32 v16, v3, 16, 1                                   // 000000001DCC: D1C80010 02052103
	v_add3_u32 v15, v2, v15, s0                                // 000000001DD4: D1FF000F 00021F02
	v_cndmask_b32_sdwa v4, v37, v14, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001DDC: 00081CF9 05060625
	v_cmp_o_f32_e32 vcc, v2, v2                                // 000000001DE4: 7C8E0502
	v_bfe_u32 v17, v12, 16, 1                                  // 000000001DE8: D1C80011 0205210C
	v_add3_u32 v16, v3, v16, s0                                // 000000001DF0: D1FF0010 00022103
	v_perm_b32 v4, v4, v1, s1                                  // 000000001DF8: D1ED0004 00060304
	v_cndmask_b32_sdwa v1, v37, v15, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001E00: 00021EF9 05060625
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000001E08: 7C8E0703
	v_bfe_u32 v18, v13, 16, 1                                  // 000000001E0C: D1C80012 0205210D
	v_add3_u32 v17, v12, v17, s0                               // 000000001E14: D1FF0011 0002230C
	v_cndmask_b32_sdwa v2, v37, v16, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001E1C: 000420F9 05060625
	v_cmp_o_f32_e32 vcc, v12, v12                              // 000000001E24: 7C8E190C
	v_add3_u32 v18, v13, v18, s0                               // 000000001E28: D1FF0012 0002250D
	v_perm_b32 v3, v2, v1, s1                                  // 000000001E30: D1ED0003 00060302
	v_cndmask_b32_sdwa v1, v37, v17, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001E38: 000222F9 05060625
	v_cmp_o_f32_e32 vcc, v13, v13                              // 000000001E40: 7C8E1B0D
	v_cndmask_b32_sdwa v2, v37, v18, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001E44: 000424F9 05060625
	v_perm_b32 v2, v2, v1, s1                                  // 000000001E4C: D1ED0002 00060302
	global_store_dwordx4 v[30:31], v[2:5], off offset:1024     // 000000001E54: DC7C8400 007F021E
	s_or_b64 exec, exec, s[2:3]                                // 000000001E5C: 87FE027E
	v_and_b32_e32 v1, 7, v0                                    // 000000001E60: 26020087
	v_or_b32_e32 v2, s4, v1                                    // 000000001E64: 28040204
	v_and_b32_e32 v0, 0xf8, v0                                 // 000000001E68: 260000FF 000000F8
	v_cmp_gt_i32_e32 vcc, s12, v2                              // 000000001E70: 7D88040C
	v_lshrrev_b32_e32 v3, 3, v9                                // 000000001E74: 20061283
	v_cmp_eq_u32_e64 s[0:1], 0, v0                             // 000000001E78: D0CA0000 00020080
	v_add_u32_e32 v3, 0, v3                                    // 000000001E80: 68060680
	s_and_b64 s[0:1], s[0:1], vcc                              // 000000001E84: 86806A00
	ds_write_b32 v3, v8                                        // 000000001E88: D81A0000 00000803
	s_waitcnt lgkmcnt(0)                                       // 000000001E90: BF8CC07F
	s_barrier                                                  // 000000001E94: BF8A0000
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001E98: BE822000
	s_cbranch_execz 13                                         // 000000001E9C: BF88000D <c662_rmsnorm_fwd_v2+0x7d4>
	v_lshl_add_u32 v0, v1, 2, 0                                // 000000001EA0: D1FD0000 02010501
	ds_read_b32 v5, v0                                         // 000000001EA8: D86C0000 05000000
	v_ashrrev_i32_e32 v3, 31, v2                               // 000000001EB0: 2206049F
	v_lshlrev_b64 v[2:3], 2, v[2:3]                            // 000000001EB4: D28F0002 00020482
	v_mov_b32_e32 v4, s11                                      // 000000001EBC: 7E08020B
	v_add_co_u32_e32 v0, vcc, s10, v2                          // 000000001EC0: 3200040A
	v_addc_co_u32_e32 v1, vcc, v4, v3, vcc                     // 000000001EC4: 38020704
	s_waitcnt lgkmcnt(0)                                       // 000000001EC8: BF8CC07F
	global_store_dword v[0:1], v5, off                         // 000000001ECC: DC708000 007F0500
	s_endpgm                                                   // 000000001ED4: BF810000
	s_nop 0                                                    // 000000001ED8: BF800000
	s_nop 0                                                    // 000000001EDC: BF800000
	s_nop 0                                                    // 000000001EE0: BF800000
	s_nop 0                                                    // 000000001EE4: BF800000
	s_nop 0                                                    // 000000001EE8: BF800000
	s_nop 0                                                    // 000000001EEC: BF800000
	s_nop 0                                                    // 000000001EF0: BF800000
	s_nop 0                                                    // 000000001EF4: BF800000
	s_nop 0                                                    // 000000001EF8: BF800000
	s_nop 0                                                    // 000000001EFC: BF800000
	s_nop 0                                                    // 000000001F00: BF800000
	s_nop 0                                                    // 000000001F04: BF800000
	s_nop 0                                                    // 000000001F08: BF800000
	s_nop 0                                                    // 000000001F0C: BF800000
	s_nop 0                                                    // 000000001F10: BF800000
	s_nop 0                                                    // 000000001F14: BF800000
	s_nop 0                                                    // 000000001F18: BF800000
	s_nop 0                                                    // 000000001F1C: BF800000
	s_nop 0                                                    // 000000001F20: BF800000
	s_nop 0                                                    // 000000001F24: BF800000
	s_nop 0                                                    // 000000001F28: BF800000
	s_nop 0                                                    // 000000001F2C: BF800000
	s_nop 0                                                    // 000000001F30: BF800000
	s_nop 0                                                    // 000000001F34: BF800000
	s_nop 0                                                    // 000000001F38: BF800000
	s_nop 0                                                    // 000000001F3C: BF800000
	s_nop 0                                                    // 000000001F40: BF800000
	s_nop 0                                                    // 000000001F44: BF800000
	s_nop 0                                                    // 000000001F48: BF800000
	s_nop 0                                                    // 000000001F4C: BF800000
	s_nop 0                                                    // 000000001F50: BF800000
	s_nop 0                                                    // 000000001F54: BF800000
	s_nop 0                                                    // 000000001F58: BF800000
	s_nop 0                                                    // 000000001F5C: BF800000
	s_nop 0                                                    // 000000001F60: BF800000
	s_nop 0                                                    // 000000001F64: BF800000
	s_nop 0                                                    // 000000001F68: BF800000
	s_nop 0                                                    // 000000001F6C: BF800000
	s_nop 0                                                    // 000000001F70: BF800000
	s_nop 0                                                    // 000000001F74: BF800000
	s_nop 0                                                    // 000000001F78: BF800000
	s_nop 0                                                    // 000000001F7C: BF800000
	s_nop 0                                                    // 000000001F80: BF800000
	s_nop 0                                                    // 000000001F84: BF800000
	s_nop 0                                                    // 000000001F88: BF800000
	s_nop 0                                                    // 000000001F8C: BF800000
	s_nop 0                                                    // 000000001F90: BF800000
	s_nop 0                                                    // 000000001F94: BF800000
	s_nop 0                                                    // 000000001F98: BF800000
	s_nop 0                                                    // 000000001F9C: BF800000
	s_nop 0                                                    // 000000001FA0: BF800000
	s_nop 0                                                    // 000000001FA4: BF800000
	s_nop 0                                                    // 000000001FA8: BF800000
	s_nop 0                                                    // 000000001FAC: BF800000
	s_nop 0                                                    // 000000001FB0: BF800000
	s_nop 0                                                    // 000000001FB4: BF800000
	s_nop 0                                                    // 000000001FB8: BF800000
	s_nop 0                                                    // 000000001FBC: BF800000
	s_nop 0                                                    // 000000001FC0: BF800000
	s_nop 0                                                    // 000000001FC4: BF800000
	s_nop 0                                                    // 000000001FC8: BF800000
	s_nop 0                                                    // 000000001FCC: BF800000
	s_nop 0                                                    // 000000001FD0: BF800000
	s_nop 0                                                    // 000000001FD4: BF800000
	s_nop 0                                                    // 000000001FD8: BF800000
	s_nop 0                                                    // 000000001FDC: BF800000
	s_nop 0                                                    // 000000001FE0: BF800000
	s_nop 0                                                    // 000000001FE4: BF800000
	s_nop 0                                                    // 000000001FE8: BF800000
	s_nop 0                                                    // 000000001FEC: BF800000
	s_nop 0                                                    // 000000001FF0: BF800000
	s_nop 0                                                    // 000000001FF4: BF800000
	s_nop 0                                                    // 000000001FF8: BF800000
	s_nop 0                                                    // 000000001FFC: BF800000
	s_nop 0                                                    // 000000002000: BF800000
	s_nop 0                                                    // 000000002004: BF800000
	s_nop 0                                                    // 000000002008: BF800000
	s_nop 0                                                    // 00000000200C: BF800000
	s_nop 0                                                    // 000000002010: BF800000
	s_nop 0                                                    // 000000002014: BF800000
	s_nop 0                                                    // 000000002018: BF800000
	s_nop 0                                                    // 00000000201C: BF800000
	s_nop 0                                                    // 000000002020: BF800000
	s_nop 0                                                    // 000000002024: BF800000
	s_nop 0                                                    // 000000002028: BF800000
	s_nop 0                                                    // 00000000202C: BF800000
	s_nop 0                                                    // 000000002030: BF800000
	s_nop 0                                                    // 000000002034: BF800000
	s_nop 0                                                    // 000000002038: BF800000
	s_nop 0                                                    // 00000000203C: BF800000
	s_nop 0                                                    // 000000002040: BF800000
	s_nop 0                                                    // 000000002044: BF800000
	s_nop 0                                                    // 000000002048: BF800000
	s_nop 0                                                    // 00000000204C: BF800000
	s_nop 0                                                    // 000000002050: BF800000
	s_nop 0                                                    // 000000002054: BF800000
	s_nop 0                                                    // 000000002058: BF800000
	s_nop 0                                                    // 00000000205C: BF800000
	s_nop 0                                                    // 000000002060: BF800000
	s_nop 0                                                    // 000000002064: BF800000
	s_nop 0                                                    // 000000002068: BF800000
	s_nop 0                                                    // 00000000206C: BF800000
	s_nop 0                                                    // 000000002070: BF800000
	s_nop 0                                                    // 000000002074: BF800000
	s_nop 0                                                    // 000000002078: BF800000
	s_nop 0                                                    // 00000000207C: BF800000
	s_nop 0                                                    // 000000002080: BF800000
	s_nop 0                                                    // 000000002084: BF800000
	s_nop 0                                                    // 000000002088: BF800000
	s_nop 0                                                    // 00000000208C: BF800000
	s_nop 0                                                    // 000000002090: BF800000
	s_nop 0                                                    // 000000002094: BF800000
	s_nop 0                                                    // 000000002098: BF800000
	s_nop 0                                                    // 00000000209C: BF800000
	s_nop 0                                                    // 0000000020A0: BF800000
	s_nop 0                                                    // 0000000020A4: BF800000
	s_nop 0                                                    // 0000000020A8: BF800000
	s_nop 0                                                    // 0000000020AC: BF800000
	s_nop 0                                                    // 0000000020B0: BF800000
	s_nop 0                                                    // 0000000020B4: BF800000
	s_nop 0                                                    // 0000000020B8: BF800000
	s_nop 0                                                    // 0000000020BC: BF800000
	s_nop 0                                                    // 0000000020C0: BF800000
	s_nop 0                                                    // 0000000020C4: BF800000
	s_nop 0                                                    // 0000000020C8: BF800000
	s_nop 0                                                    // 0000000020CC: BF800000
	s_nop 0                                                    // 0000000020D0: BF800000
	s_nop 0                                                    // 0000000020D4: BF800000
	s_nop 0                                                    // 0000000020D8: BF800000
	s_nop 0                                                    // 0000000020DC: BF800000
	s_nop 0                                                    // 0000000020E0: BF800000
	s_nop 0                                                    // 0000000020E4: BF800000
	s_nop 0                                                    // 0000000020E8: BF800000
	s_nop 0                                                    // 0000000020EC: BF800000
	s_nop 0                                                    // 0000000020F0: BF800000
	s_nop 0                                                    // 0000000020F4: BF800000
	s_nop 0                                                    // 0000000020F8: BF800000
	s_nop 0                                                    // 0000000020FC: BF800000
	s_nop 0                                                    // 000000002100: BF800000
	s_nop 0                                                    // 000000002104: BF800000
	s_nop 0                                                    // 000000002108: BF800000
	s_nop 0                                                    // 00000000210C: BF800000
	s_nop 0                                                    // 000000002110: BF800000
	s_nop 0                                                    // 000000002114: BF800000
	s_nop 0                                                    // 000000002118: BF800000
	s_nop 0                                                    // 00000000211C: BF800000
	s_nop 0                                                    // 000000002120: BF800000
	s_nop 0                                                    // 000000002124: BF800000
	s_nop 0                                                    // 000000002128: BF800000
	s_nop 0                                                    // 00000000212C: BF800000
	s_nop 0                                                    // 000000002130: BF800000
	s_nop 0                                                    // 000000002134: BF800000
	s_nop 0                                                    // 000000002138: BF800000
	s_nop 0                                                    // 00000000213C: BF800000
	s_nop 0                                                    // 000000002140: BF800000
	s_nop 0                                                    // 000000002144: BF800000
	s_nop 0                                                    // 000000002148: BF800000
	s_nop 0                                                    // 00000000214C: BF800000
	s_nop 0                                                    // 000000002150: BF800000
	s_nop 0                                                    // 000000002154: BF800000
	s_nop 0                                                    // 000000002158: BF800000
	s_nop 0                                                    // 00000000215C: BF800000
	s_nop 0                                                    // 000000002160: BF800000
	s_nop 0                                                    // 000000002164: BF800000
	s_nop 0                                                    // 000000002168: BF800000
	s_nop 0                                                    // 00000000216C: BF800000
	s_nop 0                                                    // 000000002170: BF800000
	s_nop 0                                                    // 000000002174: BF800000
	s_nop 0                                                    // 000000002178: BF800000
	s_nop 0                                                    // 00000000217C: BF800000
	s_nop 0                                                    // 000000002180: BF800000
	s_nop 0                                                    // 000000002184: BF800000
	s_nop 0                                                    // 000000002188: BF800000
	s_nop 0                                                    // 00000000218C: BF800000
	s_nop 0                                                    // 000000002190: BF800000
	s_nop 0                                                    // 000000002194: BF800000
	s_nop 0                                                    // 000000002198: BF800000
	s_nop 0                                                    // 00000000219C: BF800000
	s_nop 0                                                    // 0000000021A0: BF800000
	s_nop 0                                                    // 0000000021A4: BF800000
	s_nop 0                                                    // 0000000021A8: BF800000
	s_nop 0                                                    // 0000000021AC: BF800000
	s_nop 0                                                    // 0000000021B0: BF800000
	s_nop 0                                                    // 0000000021B4: BF800000
	s_nop 0                                                    // 0000000021B8: BF800000
	s_nop 0                                                    // 0000000021BC: BF800000
	s_nop 0                                                    // 0000000021C0: BF800000
	s_nop 0                                                    // 0000000021C4: BF800000
	s_nop 0                                                    // 0000000021C8: BF800000
	s_nop 0                                                    // 0000000021CC: BF800000
	s_nop 0                                                    // 0000000021D0: BF800000
	s_nop 0                                                    // 0000000021D4: BF800000
	s_nop 0                                                    // 0000000021D8: BF800000
	s_nop 0                                                    // 0000000021DC: BF800000
	s_nop 0                                                    // 0000000021E0: BF800000
	s_nop 0                                                    // 0000000021E4: BF800000
	s_nop 0                                                    // 0000000021E8: BF800000
	s_nop 0                                                    // 0000000021EC: BF800000
	s_nop 0                                                    // 0000000021F0: BF800000
	s_nop 0                                                    // 0000000021F4: BF800000
	s_nop 0                                                    // 0000000021F8: BF800000
	s_nop 0                                                    // 0000000021FC: BF800000
	s_nop 0                                                    // 000000002200: BF800000
	s_nop 0                                                    // 000000002204: BF800000
	s_nop 0                                                    // 000000002208: BF800000
	s_nop 0                                                    // 00000000220C: BF800000
	s_nop 0                                                    // 000000002210: BF800000
	s_nop 0                                                    // 000000002214: BF800000
	s_nop 0                                                    // 000000002218: BF800000
	s_nop 0                                                    // 00000000221C: BF800000
	s_nop 0                                                    // 000000002220: BF800000
	s_nop 0                                                    // 000000002224: BF800000
	s_nop 0                                                    // 000000002228: BF800000
	s_nop 0                                                    // 00000000222C: BF800000
	s_nop 0                                                    // 000000002230: BF800000
	s_nop 0                                                    // 000000002234: BF800000
	s_nop 0                                                    // 000000002238: BF800000
	s_nop 0                                                    // 00000000223C: BF800000
	s_nop 0                                                    // 000000002240: BF800000
	s_nop 0                                                    // 000000002244: BF800000
	s_nop 0                                                    // 000000002248: BF800000
	s_nop 0                                                    // 00000000224C: BF800000
	s_nop 0                                                    // 000000002250: BF800000
	s_nop 0                                                    // 000000002254: BF800000
	s_nop 0                                                    // 000000002258: BF800000
	s_nop 0                                                    // 00000000225C: BF800000
	s_nop 0                                                    // 000000002260: BF800000
	s_nop 0                                                    // 000000002264: BF800000
	s_nop 0                                                    // 000000002268: BF800000
	s_nop 0                                                    // 00000000226C: BF800000
	s_nop 0                                                    // 000000002270: BF800000
	s_nop 0                                                    // 000000002274: BF800000
	s_nop 0                                                    // 000000002278: BF800000
	s_nop 0                                                    // 00000000227C: BF800000
	s_nop 0                                                    // 000000002280: BF800000
	s_nop 0                                                    // 000000002284: BF800000
	s_nop 0                                                    // 000000002288: BF800000
	s_nop 0                                                    // 00000000228C: BF800000
	s_nop 0                                                    // 000000002290: BF800000
	s_nop 0                                                    // 000000002294: BF800000
	s_nop 0                                                    // 000000002298: BF800000
	s_nop 0                                                    // 00000000229C: BF800000
	s_nop 0                                                    // 0000000022A0: BF800000
	s_nop 0                                                    // 0000000022A4: BF800000
	s_nop 0                                                    // 0000000022A8: BF800000
	s_nop 0                                                    // 0000000022AC: BF800000
	s_nop 0                                                    // 0000000022B0: BF800000
	s_nop 0                                                    // 0000000022B4: BF800000
	s_nop 0                                                    // 0000000022B8: BF800000
	s_nop 0                                                    // 0000000022BC: BF800000
	s_nop 0                                                    // 0000000022C0: BF800000
	s_nop 0                                                    // 0000000022C4: BF800000
	s_nop 0                                                    // 0000000022C8: BF800000
	s_nop 0                                                    // 0000000022CC: BF800000
	s_nop 0                                                    // 0000000022D0: BF800000
	s_nop 0                                                    // 0000000022D4: BF800000
	s_nop 0                                                    // 0000000022D8: BF800000
	s_nop 0                                                    // 0000000022DC: BF800000
	s_nop 0                                                    // 0000000022E0: BF800000
	s_nop 0                                                    // 0000000022E4: BF800000
	s_nop 0                                                    // 0000000022E8: BF800000
	s_nop 0                                                    // 0000000022EC: BF800000
	s_nop 0                                                    // 0000000022F0: BF800000
	s_nop 0                                                    // 0000000022F4: BF800000
	s_nop 0                                                    // 0000000022F8: BF800000
	s_nop 0                                                    // 0000000022FC: BF800000
