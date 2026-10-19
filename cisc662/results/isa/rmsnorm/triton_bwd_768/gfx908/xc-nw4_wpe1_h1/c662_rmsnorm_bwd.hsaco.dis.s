
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_bwd_768/gfx908/xc-nw4_wpe1_h1/c662_rmsnorm_bwd.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <c662_rmsnorm_bwd>:
	s_load_dword s0, s[4:5], 0x20                              // 000000001700: C0020002 00000020
	s_load_dwordx4 s[8:11], s[4:5], 0x0                        // 000000001708: C00A0202 00000000
	s_load_dword s7, s[4:5], 0x28                              // 000000001710: C00201C2 00000028
	v_lshrrev_b32_e32 v1, 6, v0                                // 000000001718: 20020086
	v_lshl_or_b32 v1, s6, 2, v1                                // 00000000171C: D2000001 04050406
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF8CC07F
	v_cmp_gt_i32_e32 vcc, s0, v1                               // 000000001728: 7D880200
	v_mov_b32_e32 v3, 0                                        // 00000000172C: 7E060280
	v_mov_b32_e32 v2, 0                                        // 000000001730: 7E040280
	v_mov_b32_e32 v17, 0                                       // 000000001734: 7E220280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001738: BE82206A
	s_cbranch_execz 15                                         // 00000000173C: BF88000F <c662_rmsnorm_bwd+0x7c>
	s_load_dwordx2 s[0:1], s[4:5], 0x10                        // 000000001740: C0060002 00000010
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001748: 2204029F
	v_lshlrev_b64 v[4:5], 2, v[1:2]                            // 00000000174C: D28F0004 00020282
	s_waitcnt lgkmcnt(0)                                       // 000000001754: BF8CC07F
	v_mov_b32_e32 v2, s1                                       // 000000001758: 7E040201
	v_add_co_u32_e64 v4, s[0:1], s0, v4                        // 00000000175C: D1190004 00020800
	v_addc_co_u32_e64 v5, s[0:1], v2, v5, s[0:1]               // 000000001764: D11C0005 00020B02
	global_load_dword v2, v[4:5], off                          // 00000000176C: DC508000 027F0004
	s_waitcnt vmcnt(0)                                         // 000000001774: BF8C0F70
	v_mov_b32_e32 v17, v2                                      // 000000001778: 7E220302
	s_or_b64 exec, exec, s[2:3]                                // 00000000177C: 87FE027E
	v_mul_lo_u32 v4, s7, v1                                    // 000000001780: D2850004 00020207
	s_load_dword s2, s[4:5], 0x24                              // 000000001788: C0020082 00000024
	v_lshlrev_b32_e32 v18, 2, v0                               // 000000001790: 24240082
	v_mov_b32_e32 v6, s11                                      // 000000001794: 7E0C020B
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000001798: 220A089F
	v_lshlrev_b64 v[4:5], 1, v[4:5]                            // 00000000179C: D28F0004 00020881
	v_and_b32_e32 v0, 0xfc, v18                                // 0000000017A4: 260024FF 000000FC
	v_add_co_u32_e64 v4, s[0:1], s10, v4                       // 0000000017AC: D1190004 0002080A
	v_addc_co_u32_e64 v5, s[0:1], v6, v5, s[0:1]               // 0000000017B4: D11C0005 00020B06
	v_lshlrev_b32_e32 v0, 1, v0                                // 0000000017BC: 24000081
	v_add_co_u32_e64 v9, s[0:1], v4, v0                        // 0000000017C0: D1190009 00020104
	v_addc_co_u32_e64 v10, s[0:1], 0, v5, s[0:1]               // 0000000017C8: D11C000A 00020A80
	v_mov_b32_e32 v4, 0                                        // 0000000017D0: 7E080280
	s_and_saveexec_b64 s[0:1], vcc                             // 0000000017D4: BE80206A
	s_cbranch_execz 2                                          // 0000000017D8: BF880002 <c662_rmsnorm_bwd+0xe4>
	global_load_dwordx2 v[3:4], v[9:10], off                   // 0000000017DC: DC548000 037F0009
	s_or_b64 exec, exec, s[0:1]                                // 0000000017E4: 87FE007E
	s_waitcnt lgkmcnt(0)                                       // 0000000017E8: BF8CC07F
	v_mul_lo_u32 v5, s2, v1                                    // 0000000017EC: D2850005 00020202
	v_mov_b32_e32 v7, s9                                       // 0000000017F4: 7E0E0209
	v_ashrrev_i32_e32 v6, 31, v5                               // 0000000017F8: 220C0A9F
	v_lshlrev_b64 v[5:6], 1, v[5:6]                            // 0000000017FC: D28F0005 00020A81
	v_add_co_u32_e64 v5, s[0:1], s8, v5                        // 000000001804: D1190005 00020A08
	v_addc_co_u32_e64 v6, s[0:1], v7, v6, s[0:1]               // 00000000180C: D11C0006 00020D07
	v_add_co_u32_e64 v11, s[0:1], v5, v0                       // 000000001814: D119000B 00020105
	v_addc_co_u32_e64 v12, s[0:1], 0, v6, s[0:1]               // 00000000181C: D11C000C 00020C80
	v_mov_b32_e32 v7, 0                                        // 000000001824: 7E0E0280
	v_mov_b32_e32 v5, 0                                        // 000000001828: 7E0A0280
	v_mov_b32_e32 v6, 0                                        // 00000000182C: 7E0C0280
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001830: BE80206A
	s_cbranch_execz 2                                          // 000000001834: BF880002 <c662_rmsnorm_bwd+0x140>
	global_load_dwordx2 v[5:6], v[11:12], off                  // 000000001838: DC548000 057F000B
	s_or_b64 exec, exec, s[0:1]                                // 000000001840: 87FE007E
	v_mov_b32_e32 v8, 0                                        // 000000001844: 7E100280
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001848: BE80206A
	s_cbranch_execz 2                                          // 00000000184C: BF880002 <c662_rmsnorm_bwd+0x158>
	global_load_dwordx2 v[7:8], v[9:10], off offset:512        // 000000001850: DC548200 077F0009
	s_or_b64 exec, exec, s[0:1]                                // 000000001858: 87FE007E
	v_mov_b32_e32 v15, 0                                       // 00000000185C: 7E1E0280
	v_mov_b32_e32 v13, 0                                       // 000000001860: 7E1A0280
	v_mov_b32_e32 v14, 0                                       // 000000001864: 7E1C0280
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001868: BE80206A
	s_cbranch_execz 2                                          // 00000000186C: BF880002 <c662_rmsnorm_bwd+0x178>
	global_load_dwordx2 v[13:14], v[11:12], off offset:512     // 000000001870: DC548200 0D7F000B
	s_or_b64 exec, exec, s[0:1]                                // 000000001878: 87FE007E
	v_mov_b32_e32 v16, 0                                       // 00000000187C: 7E200280
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001880: BE80206A
	s_cbranch_execz 2                                          // 000000001884: BF880002 <c662_rmsnorm_bwd+0x190>
	global_load_dwordx2 v[15:16], v[9:10], off offset:1024     // 000000001888: DC548400 0F7F0009
	s_or_b64 exec, exec, s[0:1]                                // 000000001890: 87FE007E
	v_mov_b32_e32 v9, 0                                        // 000000001894: 7E120280
	v_mov_b32_e32 v10, 0                                       // 000000001898: 7E140280
	s_and_saveexec_b64 s[0:1], vcc                             // 00000000189C: BE80206A
	s_cbranch_execz 2                                          // 0000000018A0: BF880002 <c662_rmsnorm_bwd+0x1ac>
	global_load_dwordx2 v[9:10], v[11:12], off offset:1024     // 0000000018A4: DC548400 097F000B
	s_or_b64 exec, exec, s[0:1]                                // 0000000018AC: 87FE007E
	s_waitcnt vmcnt(0)                                         // 0000000018B0: BF8C0F70
	v_and_b32_e32 v23, 0xffff0000, v15                         // 0000000018B4: 262E1EFF FFFF0000
	v_lshlrev_b32_e32 v26, 16, v15                             // 0000000018BC: 24341E90
	v_and_b32_e32 v29, 0xffff0000, v16                         // 0000000018C0: 263A20FF FFFF0000
	v_lshlrev_b32_e32 v28, 16, v16                             // 0000000018C8: 24382090
	v_and_b32_e32 v12, 0xffff0000, v13                         // 0000000018CC: 26181AFF FFFF0000
	v_lshlrev_b32_e32 v15, 16, v13                             // 0000000018D4: 241E1A90
	v_and_b32_e32 v11, 0xffff0000, v14                         // 0000000018D8: 26161CFF FFFF0000
	v_lshlrev_b32_e32 v13, 16, v14                             // 0000000018E0: 241A1C90
	v_and_b32_e32 v14, 0xffff0000, v7                          // 0000000018E4: 261C0EFF FFFF0000
	v_lshlrev_b32_e32 v7, 16, v7                               // 0000000018EC: 240E0E90
	v_lshlrev_b32_e32 v16, 16, v3                              // 0000000018F0: 24200690
	v_and_b32_e32 v30, 0xffff0000, v8                          // 0000000018F4: 263C10FF FFFF0000
	v_lshlrev_b32_e32 v31, 16, v8                              // 0000000018FC: 243E1090
	v_and_b32_e32 v8, 0xffff0000, v3                           // 000000001900: 261006FF FFFF0000
	v_mul_f32_e32 v25, v2, v16                                 // 000000001908: 0A322102
	v_mul_f32_e32 v14, v17, v14                                // 00000000190C: 0A1C1D11
	v_mul_f32_e32 v16, v2, v7                                  // 000000001910: 0A200F02
	v_and_b32_e32 v19, 0xffff0000, v5                          // 000000001914: 26260AFF FFFF0000
	v_lshlrev_b32_e32 v21, 16, v5                              // 00000000191C: 242A0A90
	v_and_b32_e32 v20, 0xffff0000, v6                          // 000000001920: 26280CFF FFFF0000
	v_lshlrev_b32_e32 v22, 16, v6                              // 000000001928: 242C0C90
	v_and_b32_e32 v32, 0xffff0000, v4                          // 00000000192C: 264008FF FFFF0000
	v_lshlrev_b32_e32 v27, 16, v4                              // 000000001934: 24360890
	v_and_b32_e32 v4, 0xffff0000, v9                           // 000000001938: 260812FF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v9                               // 000000001940: 240C1290
	v_and_b32_e32 v3, 0xffff0000, v10                          // 000000001944: 260614FF FFFF0000
	v_lshlrev_b32_e32 v5, 16, v10                              // 00000000194C: 240A1490
	v_mul_f32_e32 v24, v17, v8                                 // 000000001950: 0A301111
	v_mul_f32_e32 v9, v16, v15                                 // 000000001954: 0A121F10
	v_mul_f32_e32 v10, v14, v12                                // 000000001958: 0A14190E
	v_fmac_f32_e32 v9, v25, v21                                // 00000000195C: 76122B19
	v_fmac_f32_e32 v10, v24, v19                               // 000000001960: 76142718
	v_mul_f32_e32 v7, v17, v23                                 // 000000001964: 0A0E2F11
	v_mul_f32_e32 v8, v2, v26                                  // 000000001968: 0A103502
	v_mul_f32_e32 v23, v2, v31                                 // 00000000196C: 0A2E3F02
	v_fmac_f32_e32 v9, v8, v6                                  // 000000001970: 76120D08
	v_fmac_f32_e32 v10, v7, v4                                 // 000000001974: 76140907
	v_mul_f32_e32 v27, v2, v27                                 // 000000001978: 0A363702
	v_mul_f32_e32 v26, v23, v13                                // 00000000197C: 0A341B17
	v_add_f32_e32 v10, v9, v10                                 // 000000001980: 02141509
	v_fmac_f32_e32 v26, v27, v22                               // 000000001984: 76342D1B
	v_mul_f32_e32 v9, v2, v28                                  // 000000001988: 0A123902
	v_fmac_f32_e32 v26, v9, v5                                 // 00000000198C: 76340B09
	v_add_f32_e32 v31, v26, v10                                // 000000001990: 023E151A
	v_mul_f32_e32 v26, v17, v30                                // 000000001994: 0A343D11
	v_mul_f32_e32 v28, v17, v32                                // 000000001998: 0A384111
	v_mul_f32_e32 v30, v26, v11                                // 00000000199C: 0A3C171A
	v_fmac_f32_e32 v30, v28, v20                               // 0000000019A0: 763C291C
	v_mul_f32_e32 v10, v17, v29                                // 0000000019A4: 0A143B11
	v_fmac_f32_e32 v30, v10, v3                                // 0000000019A8: 763C070A
	v_add_f32_e32 v29, v30, v31                                // 0000000019AC: 023A3F1E
	v_xor_b32_e32 v18, 0x80, v18                               // 0000000019B0: 2A2424FF 00000080
	ds_bpermute_b32 v18, v18, v29                              // 0000000019B8: D87E0000 12001D12
	s_waitcnt lgkmcnt(0)                                       // 0000000019C0: BF8CC07F
	v_add_f32_e32 v18, v29, v18                                // 0000000019C4: 0224251D
	ds_swizzle_b32 v29, v18 offset:swizzle(SWAP,16)            // 0000000019C8: D87A401F 1D000012
	s_waitcnt lgkmcnt(0)                                       // 0000000019D0: BF8CC07F
	v_add_f32_e32 v18, v18, v29                                // 0000000019D4: 02243B12
	ds_swizzle_b32 v29, v18 offset:swizzle(SWAP,8)             // 0000000019D8: D87A201F 1D000012
	s_waitcnt lgkmcnt(0)                                       // 0000000019E0: BF8CC07F
	v_add_f32_e32 v18, v18, v29                                // 0000000019E4: 02243B12
	ds_swizzle_b32 v29, v18 offset:swizzle(SWAP,4)             // 0000000019E8: D87A101F 1D000012
	s_waitcnt lgkmcnt(0)                                       // 0000000019F0: BF8CC07F
	v_add_f32_e32 v18, v18, v29                                // 0000000019F4: 02243B12
	ds_swizzle_b32 v29, v18 offset:swizzle(SWAP,2)             // 0000000019F8: D87A081F 1D000012
	s_waitcnt lgkmcnt(0)                                       // 000000001A00: BF8CC07F
	v_add_f32_e32 v18, v18, v29                                // 000000001A04: 02243B12
	ds_swizzle_b32 v29, v18 offset:swizzle(SWAP,1)             // 000000001A08: D87A041F 1D000012
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001A10: BE80206A
	s_cbranch_execz 180                                        // 000000001A14: BF8800B4 <c662_rmsnorm_bwd+0x5e8>
	s_waitcnt lgkmcnt(0)                                       // 000000001A18: BF8CC07F
	v_add_f32_e32 v18, v18, v29                                // 000000001A1C: 02243B12
	s_mov_b32 s2, 0x44400000                                   // 000000001A20: BE8200FF 44400000
	v_div_scale_f32 v29, s[0:1], s2, s2, v18                   // 000000001A28: D1E0001D 04480402
	s_load_dword s3, s[4:5], 0x2c                              // 000000001A30: C00200C2 0000002C
	s_load_dwordx2 s[0:1], s[4:5], 0x18                        // 000000001A38: C0060002 00000018
	v_rcp_f32_e32 v30, v29                                     // 000000001A40: 7E3C451D
	v_fma_f32 v31, -v29, v30, 1.0                              // 000000001A44: D1CB001F 23CA3D1D
	v_fmac_f32_e32 v30, v31, v30                               // 000000001A4C: 763C3D1F
	v_div_scale_f32 v31, vcc, v18, s2, v18                     // 000000001A50: D1E06A1F 04480512
	v_mul_f32_e32 v32, v31, v30                                // 000000001A58: 0A403D1F
	v_fma_f32 v33, -v29, v32, v31                              // 000000001A5C: D1CB0021 247E411D
	v_fmac_f32_e32 v32, v33, v30                               // 000000001A64: 76403D21
	v_fma_f32 v29, -v29, v32, v31                              // 000000001A68: D1CB001D 247E411D
	v_div_fmas_f32 v31, v29, v30, v32                          // 000000001A70: D1E2001F 04823D1D
	s_waitcnt lgkmcnt(0)                                       // 000000001A78: BF8CC07F
	v_mul_lo_u32 v29, s3, v1                                   // 000000001A7C: D285001D 00020203
	v_mov_b32_e32 v1, s1                                       // 000000001A84: 7E020201
	s_mov_b32 s1, 0x5040100                                    // 000000001A88: BE8100FF 05040100
	v_ashrrev_i32_e32 v30, 31, v29                             // 000000001A90: 223C3A9F
	v_lshlrev_b64 v[29:30], 1, v[29:30]                        // 000000001A94: D28F001D 00023A81
	v_add_co_u32_e32 v29, vcc, s0, v29                         // 000000001A9C: 323A3A00
	s_movk_i32 s0, 0x7fff                                      // 000000001AA0: B0007FFF
	v_addc_co_u32_e32 v1, vcc, v1, v30, vcc                    // 000000001AA4: 38023D01
	v_mov_b32_e32 v30, 0x7fff                                  // 000000001AA8: 7E3C02FF 00007FFF
	v_div_fixup_f32 v31, v31, s2, v18                          // 000000001AB0: D1DE001F 0448051F
	v_fma_f32 v18, -v25, v31, v21                              // 000000001AB8: D1CB0012 24563F19
	v_fma_f32 v21, -v27, v31, v22                              // 000000001AC0: D1CB0015 245A3F1B
	v_fma_f32 v19, -v24, v31, v19                              // 000000001AC8: D1CB0013 244E3F18
	v_fma_f32 v20, -v28, v31, v20                              // 000000001AD0: D1CB0014 24523F1C
	v_mul_f32_e32 v21, v2, v21                                 // 000000001AD8: 0A2A2B02
	v_mul_f32_e32 v20, v17, v20                                // 000000001ADC: 0A282911
	v_mul_f32_e32 v22, v17, v19                                // 000000001AE0: 0A2C2711
	v_bfe_u32 v19, v21, 16, 1                                  // 000000001AE4: D1C80013 02052115
	v_mul_f32_e32 v18, v2, v18                                 // 000000001AEC: 0A242502
	v_bfe_u32 v24, v20, 16, 1                                  // 000000001AF0: D1C80018 02052114
	v_add3_u32 v19, v21, v19, s0                               // 000000001AF8: D1FF0013 00022715
	v_bfe_u32 v27, v18, 16, 1                                  // 000000001B00: D1C8001B 02052112
	v_add3_u32 v24, v20, v24, s0                               // 000000001B08: D1FF0018 00023114
	v_lshrrev_b32_e32 v19, 16, v19                             // 000000001B10: 20262690
	v_cmp_o_f32_e32 vcc, v21, v21                              // 000000001B14: 7C8E2B15
	v_bfe_u32 v25, v22, 16, 1                                  // 000000001B18: D1C80019 02052116
	v_add3_u32 v27, v18, v27, s0                               // 000000001B20: D1FF001B 00023712
	v_lshrrev_b32_e32 v24, 16, v24                             // 000000001B28: 20303090
	v_cndmask_b32_e32 v19, v30, v19, vcc                       // 000000001B2C: 0026271E
	v_cmp_o_f32_e32 vcc, v20, v20                              // 000000001B30: 7C8E2914
	v_fma_f32 v12, -v14, v31, v12                              // 000000001B34: D1CB000C 24323F0E
	v_add3_u32 v25, v22, v25, s0                               // 000000001B3C: D1FF0019 00023316
	v_lshrrev_b32_e32 v27, 16, v27                             // 000000001B44: 20363690
	v_cndmask_b32_e32 v20, v30, v24, vcc                       // 000000001B48: 0028311E
	v_cmp_o_f32_e32 vcc, v18, v18                              // 000000001B4C: 7C8E2512
	v_mul_f32_e32 v14, v17, v12                                // 000000001B50: 0A1C1911
	v_fma_f32 v12, -v23, v31, v13                              // 000000001B54: D1CB000C 24363F17
	v_lshrrev_b32_e32 v25, 16, v25                             // 000000001B5C: 20323290
	v_cndmask_b32_e32 v18, v30, v27, vcc                       // 000000001B60: 0024371E
	v_cmp_o_f32_e32 vcc, v22, v22                              // 000000001B64: 7C8E2D16
	v_fma_f32 v11, -v26, v31, v11                              // 000000001B68: D1CB000B 242E3F1A
	v_mul_f32_e32 v12, v2, v12                                 // 000000001B70: 0A181902
	v_perm_b32 v19, v20, v19, s1                               // 000000001B74: D1ED0013 00062714
	v_cndmask_b32_e32 v20, v30, v25, vcc                       // 000000001B7C: 0028331E
	v_add_co_u32_e32 v0, vcc, v29, v0                          // 000000001B80: 3200011D
	v_fma_f32 v15, -v16, v31, v15                              // 000000001B84: D1CB000F 243E3F10
	v_mul_f32_e32 v11, v17, v11                                // 000000001B8C: 0A161711
	v_bfe_u32 v13, v12, 16, 1                                  // 000000001B90: D1C8000D 0205210C
	v_perm_b32 v18, v20, v18, s1                               // 000000001B98: D1ED0012 00062514
	v_addc_co_u32_e32 v1, vcc, 0, v1, vcc                      // 000000001BA0: 38020280
	v_mul_f32_e32 v15, v2, v15                                 // 000000001BA4: 0A1E1F02
	v_bfe_u32 v16, v11, 16, 1                                  // 000000001BA8: D1C80010 0205210B
	v_add3_u32 v13, v12, v13, s0                               // 000000001BB0: D1FF000D 00021B0C
	global_store_dwordx2 v[0:1], v[18:19], off                 // 000000001BB8: DC748000 007F1200
	v_bfe_u32 v18, v15, 16, 1                                  // 000000001BC0: D1C80012 0205210F
	v_add3_u32 v16, v11, v16, s0                               // 000000001BC8: D1FF0010 0002210B
	v_lshrrev_b32_e32 v13, 16, v13                             // 000000001BD0: 201A1A90
	v_cmp_o_f32_e32 vcc, v12, v12                              // 000000001BD4: 7C8E190C
	v_bfe_u32 v19, v14, 16, 1                                  // 000000001BD8: D1C80013 0205210E
	v_add3_u32 v18, v15, v18, s0                               // 000000001BE0: D1FF0012 0002250F
	v_lshrrev_b32_e32 v16, 16, v16                             // 000000001BE8: 20202090
	v_cndmask_b32_e32 v12, v30, v13, vcc                       // 000000001BEC: 00181B1E
	v_cmp_o_f32_e32 vcc, v11, v11                              // 000000001BF0: 7C8E170B
	v_fma_f32 v6, -v8, v31, v6                                 // 000000001BF4: D1CB0006 241A3F08
	v_fma_f32 v5, -v9, v31, v5                                 // 000000001BFC: D1CB0005 24163F09
	v_add3_u32 v19, v14, v19, s0                               // 000000001C04: D1FF0013 0002270E
	v_lshrrev_b32_e32 v18, 16, v18                             // 000000001C0C: 20242490
	v_cndmask_b32_e32 v11, v30, v16, vcc                       // 000000001C10: 0016211E
	v_cmp_o_f32_e32 vcc, v15, v15                              // 000000001C14: 7C8E1F0F
	v_mul_f32_e32 v6, v2, v6                                   // 000000001C18: 0A0C0D02
	v_fma_f32 v3, -v10, v31, v3                                // 000000001C1C: D1CB0003 240E3F0A
	v_mul_f32_e32 v2, v2, v5                                   // 000000001C24: 0A040B02
	v_lshrrev_b32_e32 v19, 16, v19                             // 000000001C28: 20262690
	v_perm_b32 v12, v11, v12, s1                               // 000000001C2C: D1ED000C 0006190B
	v_cndmask_b32_e32 v11, v30, v18, vcc                       // 000000001C34: 0016251E
	v_cmp_o_f32_e32 vcc, v14, v14                              // 000000001C38: 7C8E1D0E
	v_mul_f32_e32 v3, v17, v3                                  // 000000001C3C: 0A060711
	v_bfe_u32 v5, v2, 16, 1                                    // 000000001C40: D1C80005 02052102
	v_cndmask_b32_e32 v13, v30, v19, vcc                       // 000000001C48: 001A271E
	v_fma_f32 v4, -v7, v31, v4                                 // 000000001C4C: D1CB0004 24123F07
	v_bfe_u32 v7, v3, 16, 1                                    // 000000001C54: D1C80007 02052103
	v_add3_u32 v5, v2, v5, s0                                  // 000000001C5C: D1FF0005 00020B02
	v_cmp_o_f32_e32 vcc, v2, v2                                // 000000001C64: 7C8E0502
	v_mul_f32_e32 v4, v17, v4                                  // 000000001C68: 0A080911
	v_bfe_u32 v8, v6, 16, 1                                    // 000000001C6C: D1C80008 02052106
	v_add3_u32 v7, v3, v7, s0                                  // 000000001C74: D1FF0007 00020F03
	v_cndmask_b32_sdwa v2, v30, v5, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001C7C: 00040AF9 0506061E
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000001C84: 7C8E0703
	v_bfe_u32 v9, v4, 16, 1                                    // 000000001C88: D1C80009 02052104
	v_add3_u32 v8, v6, v8, s0                                  // 000000001C90: D1FF0008 00021106
	v_cndmask_b32_sdwa v3, v30, v7, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001C98: 00060EF9 0506061E
	v_cmp_o_f32_e32 vcc, v6, v6                                // 000000001CA0: 7C8E0D06
	v_add3_u32 v9, v4, v9, s0                                  // 000000001CA4: D1FF0009 00021304
	v_perm_b32 v3, v3, v2, s1                                  // 000000001CAC: D1ED0003 00060503
	v_cndmask_b32_sdwa v2, v30, v8, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001CB4: 000410F9 0506061E
	v_cmp_o_f32_e32 vcc, v4, v4                                // 000000001CBC: 7C8E0904
	v_cndmask_b32_sdwa v4, v30, v9, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001CC0: 000812F9 0506061E
	v_perm_b32 v11, v13, v11, s1                               // 000000001CC8: D1ED000B 0006170D
	v_perm_b32 v2, v4, v2, s1                                  // 000000001CD0: D1ED0002 00060504
	global_store_dwordx2 v[0:1], v[11:12], off offset:512      // 000000001CD8: DC748200 007F0B00
	global_store_dwordx2 v[0:1], v[2:3], off offset:1024       // 000000001CE0: DC748400 007F0200
	s_endpgm                                                   // 000000001CE8: BF810000
