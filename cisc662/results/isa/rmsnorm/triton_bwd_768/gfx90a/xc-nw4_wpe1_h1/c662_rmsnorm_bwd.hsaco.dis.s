
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_bwd_768/gfx90a/xc-nw4_wpe1_h1/c662_rmsnorm_bwd.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <c662_rmsnorm_bwd>:
	s_load_dwordx2 s[6:7], s[4:5], 0x0                         // 000000001700: C0060182 00000000
	s_load_dwordx8 s[8:15], s[4:5], 0x8                        // 000000001708: C00E0202 00000008
	s_waitcnt lgkmcnt(0)                                       // 000000001710: BF8CC07F
	s_branch 58                                                // 000000001714: BF82003A <c662_rmsnorm_bwd+0x100>
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
	s_load_dword s17, s[4:5], 0x28                             // 000000001800: C0020442 00000028
	v_lshrrev_b32_e32 v1, 6, v0                                // 000000001808: 20020086
	v_lshl_or_b32 v4, s16, 2, v1                               // 00000000180C: D2000004 04050410
	v_mov_b32_e32 v6, 0                                        // 000000001814: 7E0C0280
	v_cmp_gt_i32_e32 vcc, s14, v4                              // 000000001818: 7D88080E
	v_mov_b32_e32 v2, 0                                        // 00000000181C: 7E040280
	v_mov_b32_e32 v3, v6                                       // 000000001820: 7E060306
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001824: BE82206A
	s_cbranch_execz 12                                         // 000000001828: BF88000C <c662_rmsnorm_bwd+0x15c>
	v_ashrrev_i32_e32 v5, 31, v4                               // 00000000182C: 220A089F
	v_lshlrev_b64 v[2:3], 2, v[4:5]                            // 000000001830: D28F0002 00020882
	v_mov_b32_e32 v1, s11                                      // 000000001838: 7E02020B
	v_add_co_u32_e64 v2, s[0:1], s10, v2                       // 00000000183C: D1190002 0002040A
	v_addc_co_u32_e64 v3, s[0:1], v1, v3, s[0:1]               // 000000001844: D11C0003 00020701
	global_load_dword v2, v[2:3], off                          // 00000000184C: DC508000 027F0002
	s_waitcnt vmcnt(0)                                         // 000000001854: BF8C0F70
	v_mov_b32_e32 v3, v2                                       // 000000001858: 7E060302
	s_or_b64 exec, exec, s[2:3]                                // 00000000185C: 87FE027E
	s_waitcnt lgkmcnt(0)                                       // 000000001860: BF8CC07F
	v_mul_lo_u32 v8, s17, v4                                   // 000000001864: D2850008 00020811
	v_ashrrev_i32_e32 v9, 31, v8                               // 00000000186C: 2212109F
	v_lshlrev_b64 v[8:9], 1, v[8:9]                            // 000000001870: D28F0008 00021081
	v_lshlrev_b32_e32 v0, 2, v0                                // 000000001878: 24000082
	v_mov_b32_e32 v1, s9                                       // 00000000187C: 7E020209
	v_add_co_u32_e64 v7, s[0:1], s8, v8                        // 000000001880: D1190007 00021008
	v_and_b32_e32 v0, 0xfc, v0                                 // 000000001888: 260000FF 000000FC
	v_addc_co_u32_e64 v1, s[0:1], v1, v9, s[0:1]               // 000000001890: D11C0001 00021301
	v_lshlrev_b32_e32 v5, 1, v0                                // 000000001898: 240A0081
	v_add_co_u32_e64 v0, s[0:1], v7, v5                        // 00000000189C: D1190000 00020B07
	v_addc_co_u32_e64 v1, s[0:1], 0, v1, s[0:1]                // 0000000018A4: D11C0001 00020280
	v_mov_b32_e32 v7, 0                                        // 0000000018AC: 7E0E0280
	s_and_saveexec_b64 s[0:1], vcc                             // 0000000018B0: BE80206A
	s_cbranch_execz 2                                          // 0000000018B4: BF880002 <c662_rmsnorm_bwd+0x1c0>
	global_load_dwordx2 v[6:7], v[0:1], off                    // 0000000018B8: DC548000 067F0000
	s_or_b64 exec, exec, s[0:1]                                // 0000000018C0: 87FE007E
	v_mul_lo_u32 v8, s15, v4                                   // 0000000018C4: D2850008 0002080F
	v_ashrrev_i32_e32 v9, 31, v8                               // 0000000018CC: 2212109F
	v_lshlrev_b64 v[8:9], 1, v[8:9]                            // 0000000018D0: D28F0008 00021081
	v_mov_b32_e32 v10, s7                                      // 0000000018D8: 7E140207
	v_add_co_u32_e64 v8, s[0:1], s6, v8                        // 0000000018DC: D1190008 00021006
	v_addc_co_u32_e64 v9, s[0:1], v10, v9, s[0:1]              // 0000000018E4: D11C0009 0002130A
	v_add_co_u32_e64 v8, s[0:1], v8, v5                        // 0000000018EC: D1190008 00020B08
	v_addc_co_u32_e64 v9, s[0:1], 0, v9, s[0:1]                // 0000000018F4: D11C0009 00021280
	v_mov_b32_e32 v12, 0                                       // 0000000018FC: 7E180280
	v_mov_b32_e32 v10, 0                                       // 000000001900: 7E140280
	v_mov_b32_e32 v11, 0                                       // 000000001904: 7E160280
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001908: BE80206A
	s_cbranch_execz 2                                          // 00000000190C: BF880002 <c662_rmsnorm_bwd+0x218>
	global_load_dwordx2 v[10:11], v[8:9], off                  // 000000001910: DC548000 0A7F0008
	s_or_b64 exec, exec, s[0:1]                                // 000000001918: 87FE007E
	v_mov_b32_e32 v13, 0                                       // 00000000191C: 7E1A0280
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001920: BE80206A
	s_cbranch_execz 2                                          // 000000001924: BF880002 <c662_rmsnorm_bwd+0x230>
	global_load_dwordx2 v[12:13], v[0:1], off offset:512       // 000000001928: DC548200 0C7F0000
	s_or_b64 exec, exec, s[0:1]                                // 000000001930: 87FE007E
	v_mov_b32_e32 v18, 0                                       // 000000001934: 7E240280
	v_mov_b32_e32 v14, 0                                       // 000000001938: 7E1C0280
	v_mov_b32_e32 v15, 0                                       // 00000000193C: 7E1E0280
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001940: BE80206A
	s_cbranch_execz 2                                          // 000000001944: BF880002 <c662_rmsnorm_bwd+0x250>
	global_load_dwordx2 v[14:15], v[8:9], off offset:512       // 000000001948: DC548200 0E7F0008
	s_or_b64 exec, exec, s[0:1]                                // 000000001950: 87FE007E
	v_mov_b32_e32 v19, 0                                       // 000000001954: 7E260280
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001958: BE80206A
	s_cbranch_execz 2                                          // 00000000195C: BF880002 <c662_rmsnorm_bwd+0x268>
	global_load_dwordx2 v[18:19], v[0:1], off offset:1024      // 000000001960: DC548400 127F0000
	s_or_b64 exec, exec, s[0:1]                                // 000000001968: 87FE007E
	v_mov_b32_e32 v16, 0                                       // 00000000196C: 7E200280
	v_mov_b32_e32 v17, 0                                       // 000000001970: 7E220280
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001974: BE80206A
	s_cbranch_execz 2                                          // 000000001978: BF880002 <c662_rmsnorm_bwd+0x284>
	global_load_dwordx2 v[16:17], v[8:9], off offset:1024      // 00000000197C: DC548400 107F0008
	s_or_b64 exec, exec, s[0:1]                                // 000000001984: 87FE007E
	s_waitcnt vmcnt(0)                                         // 000000001988: BF8C0F70
	v_and_b32_e32 v29, 0xffff0000, v18                         // 00000000198C: 263A24FF FFFF0000
	v_lshlrev_b32_e32 v28, 16, v18                             // 000000001994: 24382490
	v_and_b32_e32 v31, 0xffff0000, v19                         // 000000001998: 263E26FF FFFF0000
	v_lshlrev_b32_e32 v30, 16, v19                             // 0000000019A0: 243C2690
	v_and_b32_e32 v1, 0xffff0000, v14                          // 0000000019A4: 26021CFF FFFF0000
	v_lshlrev_b32_e32 v0, 16, v14                              // 0000000019AC: 24001C90
	v_and_b32_e32 v9, 0xffff0000, v15                          // 0000000019B0: 26121EFF FFFF0000
	v_lshlrev_b32_e32 v8, 16, v15                              // 0000000019B8: 24101E90
	v_and_b32_e32 v15, 0xffff0000, v12                         // 0000000019BC: 261E18FF FFFF0000
	v_lshlrev_b32_e32 v14, 16, v12                             // 0000000019C4: 241C1890
	v_and_b32_e32 v33, 0xffff0000, v13                         // 0000000019C8: 26421AFF FFFF0000
	v_lshlrev_b32_e32 v32, 16, v13                             // 0000000019D0: 24401A90
	v_and_b32_e32 v19, 0xffff0000, v7                          // 0000000019D4: 26260EFF FFFF0000
	v_lshlrev_b32_e32 v18, 16, v7                              // 0000000019DC: 24240E90
	v_and_b32_e32 v21, 0xffff0000, v10                         // 0000000019E0: 262A14FF FFFF0000
	v_lshlrev_b32_e32 v20, 16, v10                             // 0000000019E8: 24281490
	v_and_b32_e32 v23, 0xffff0000, v11                         // 0000000019EC: 262E16FF FFFF0000
	v_lshlrev_b32_e32 v22, 16, v11                             // 0000000019F4: 242C1690
	v_and_b32_e32 v13, 0xffff0000, v6                          // 0000000019F8: 261A0CFF FFFF0000
	v_lshlrev_b32_e32 v12, 16, v6                              // 000000001A00: 24180C90
	v_and_b32_e32 v7, 0xffff0000, v16                          // 000000001A04: 260E20FF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v16                              // 000000001A0C: 240C2090
	v_and_b32_e32 v11, 0xffff0000, v17                         // 000000001A10: 261622FF FFFF0000
	v_lshlrev_b32_e32 v10, 16, v17                             // 000000001A18: 24142290
	v_pk_mul_f32 v[24:25], v[2:3], v[18:19]                    // 000000001A1C: D3B14018 18022502
	v_pk_mul_f32 v[18:19], v[2:3], v[14:15]                    // 000000001A24: D3B14012 18021D02
	v_pk_mul_f32 v[16:17], v[2:3], v[32:33]                    // 000000001A2C: D3B14010 18024102
	v_pk_mul_f32 v[26:27], v[2:3], v[12:13]                    // 000000001A34: D3B1401A 18021902
	v_pk_mul_f32 v[12:13], v[18:19], v[0:1]                    // 000000001A3C: D3B1400C 18020112
	v_pk_mul_f32 v[14:15], v[16:17], v[8:9]                    // 000000001A44: D3B1400E 18021110
	v_pk_fma_f32 v[32:33], v[26:27], v[20:21], v[12:13]        // 000000001A4C: D3B04020 1C32291A
	v_pk_fma_f32 v[34:35], v[24:25], v[22:23], v[14:15]        // 000000001A54: D3B04022 1C3A2D18
	v_pk_mul_f32 v[14:15], v[2:3], v[28:29]                    // 000000001A5C: D3B1400E 18023902
	v_pk_mul_f32 v[12:13], v[2:3], v[30:31]                    // 000000001A64: D3B1400C 18023D02
	v_pk_fma_f32 v[28:29], v[14:15], v[6:7], v[32:33]          // 000000001A6C: D3B0401C 1C820D0E
	v_pk_fma_f32 v[30:31], v[12:13], v[10:11], v[34:35]        // 000000001A74: D3B0401E 1C8A150C
	v_add_f32_e32 v28, v28, v29                                // 000000001A7C: 02383B1C
	v_add_f32_e32 v28, v30, v28                                // 000000001A80: 0238391E
	v_add_f32_e32 v28, v31, v28                                // 000000001A84: 0238391F
	s_nop 1                                                    // 000000001A88: BF800001
	v_add_f32_dpp v28, v28, v28 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A8C: 023838FA FF09181C
	s_nop 1                                                    // 000000001A94: BF800001
	v_add_f32_dpp v28, v28, v28 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A98: 023838FA FF09141C
	s_nop 1                                                    // 000000001AA0: BF800001
	v_add_f32_dpp v28, v28, v28 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001AA4: 023838FA FF09121C
	s_nop 1                                                    // 000000001AAC: BF800001
	v_add_f32_dpp v28, v28, v28 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001AB0: 023838FA FF09111C
	v_mov_b32_e32 v29, v28                                     // 000000001AB8: 7E3A031C
	s_nop 1                                                    // 000000001ABC: BF800001
	v_mov_b32_dpp v29, v29 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1// 000000001AC0: 7E3A02FA AF09421D
	v_add_f32_e32 v28, v29, v28                                // 000000001AC8: 0238391D
	s_nop 1                                                    // 000000001ACC: BF800001
	v_add_f32_dpp v28, v28, v28 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001AD0: 023838FA FF09431C
	v_readlane_b32 s0, v28, 63                                 // 000000001AD8: D2890000 00017F1C
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001AE0: BE82206A
	s_cbranch_execz 167                                        // 000000001AE4: BF8800A7 <c662_rmsnorm_bwd+0x684>
	s_load_dword s1, s[4:5], 0x2c                              // 000000001AE8: C0020042 0000002C
	v_mov_b32_e32 v30, s13                                     // 000000001AF0: 7E3C020D
	s_waitcnt lgkmcnt(0)                                       // 000000001AF4: BF8CC07F
	v_mul_lo_u32 v28, s1, v4                                   // 000000001AF8: D285001C 00020801
	v_ashrrev_i32_e32 v29, 31, v28                             // 000000001B00: 223A389F
	v_lshlrev_b64 v[28:29], 1, v[28:29]                        // 000000001B04: D28F001C 00023881
	v_add_co_u32_e32 v4, vcc, s12, v28                         // 000000001B0C: 3208380C
	v_addc_co_u32_e32 v28, vcc, v30, v29, vcc                  // 000000001B10: 38383B1E
	s_mov_b32 s1, 0x44400000                                   // 000000001B14: BE8100FF 44400000
	v_mov_b32_e32 v29, s0                                      // 000000001B1C: 7E3A0200
	v_div_scale_f32 v30, s[2:3], s1, s1, v29                   // 000000001B20: D1E0021E 04740201
	v_rcp_f32_e32 v31, v30                                     // 000000001B28: 7E3E451E
	v_add_co_u32_e32 v4, vcc, v4, v5                           // 000000001B2C: 32080B04
	v_addc_co_u32_e32 v5, vcc, 0, v28, vcc                     // 000000001B30: 380A3880
	v_fma_f32 v28, -v30, v31, 1.0                              // 000000001B34: D1CB001C 23CA3F1E
	v_fmac_f32_e32 v31, v28, v31                               // 000000001B3C: 763E3F1C
	v_mov_b32_e32 v28, 0x44400000                              // 000000001B40: 7E3802FF 44400000
	v_div_scale_f32 v28, vcc, s0, v28, s0                      // 000000001B48: D1E06A1C 00023800
	v_mul_f32_e32 v32, v28, v31                                // 000000001B50: 0A403F1C
	v_fma_f32 v33, -v30, v32, v28                              // 000000001B54: D1CB0021 2472411E
	v_fmac_f32_e32 v32, v33, v31                               // 000000001B5C: 76403F21
	v_fma_f32 v28, -v30, v32, v28                              // 000000001B60: D1CB001C 2472411E
	v_div_fmas_f32 v28, v28, v31, v32                          // 000000001B68: D1E2001C 04823F1C
	v_div_fixup_f32 v28, v28, s1, v29                          // 000000001B70: D1DE001C 0474031C
	v_pk_fma_f32 v[22:23], v[24:25], v[28:29], v[22:23] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]// 000000001B78: D3B04116 2C5A3918
	v_pk_mul_f32 v[22:23], v[2:3], v[22:23]                    // 000000001B80: D3B14016 18022D02
	v_pk_fma_f32 v[20:21], v[26:27], v[28:29], v[20:21] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]// 000000001B88: D3B04114 2C52391A
	v_bfe_u32 v24, v22, 16, 1                                  // 000000001B90: D1C80018 02052116
	s_movk_i32 s0, 0x7fff                                      // 000000001B98: B0007FFF
	v_pk_mul_f32 v[20:21], v[2:3], v[20:21]                    // 000000001B9C: D3B14014 18022902
	v_bfe_u32 v25, v23, 16, 1                                  // 000000001BA4: D1C80019 02052117
	v_add3_u32 v24, v22, v24, s0                               // 000000001BAC: D1FF0018 00023116
	v_mov_b32_e32 v29, 0x7fff                                  // 000000001BB4: 7E3A02FF 00007FFF
	v_bfe_u32 v26, v20, 16, 1                                  // 000000001BBC: D1C8001A 02052114
	v_add3_u32 v25, v23, v25, s0                               // 000000001BC4: D1FF0019 00023317
	v_lshrrev_b32_e32 v24, 16, v24                             // 000000001BCC: 20303090
	v_cmp_o_f32_e32 vcc, v22, v22                              // 000000001BD0: 7C8E2D16
	v_pk_fma_f32 v[8:9], v[16:17], v[28:29], v[8:9] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]// 000000001BD4: D3B04108 2C223910
	v_bfe_u32 v27, v21, 16, 1                                  // 000000001BDC: D1C8001B 02052115
	v_add3_u32 v26, v20, v26, s0                               // 000000001BE4: D1FF001A 00023514
	v_lshrrev_b32_e32 v25, 16, v25                             // 000000001BEC: 20323290
	v_cndmask_b32_e32 v22, v29, v24, vcc                       // 000000001BF0: 002C311D
	v_cmp_o_f32_e32 vcc, v23, v23                              // 000000001BF4: 7C8E2F17
	v_pk_mul_f32 v[8:9], v[2:3], v[8:9]                        // 000000001BF8: D3B14008 18021102
	v_add3_u32 v27, v21, v27, s0                               // 000000001C00: D1FF001B 00023715
	v_lshrrev_b32_e32 v26, 16, v26                             // 000000001C08: 20343490
	v_cndmask_b32_e32 v23, v29, v25, vcc                       // 000000001C0C: 002E331D
	v_cmp_o_f32_e32 vcc, v20, v20                              // 000000001C10: 7C8E2914
	v_pk_fma_f32 v[0:1], v[18:19], v[28:29], v[0:1] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]// 000000001C14: D3B04100 2C023912
	v_bfe_u32 v16, v8, 16, 1                                   // 000000001C1C: D1C80010 02052108
	v_lshrrev_b32_e32 v27, 16, v27                             // 000000001C24: 20363690
	v_cndmask_b32_e32 v20, v29, v26, vcc                       // 000000001C28: 0028351D
	v_cmp_o_f32_e32 vcc, v21, v21                              // 000000001C2C: 7C8E2B15
	v_pk_mul_f32 v[0:1], v[2:3], v[0:1]                        // 000000001C30: D3B14000 18020102
	v_bfe_u32 v17, v9, 16, 1                                   // 000000001C38: D1C80011 02052109
	v_add3_u32 v16, v8, v16, s0                                // 000000001C40: D1FF0010 00022108
	v_cndmask_b32_e32 v21, v29, v27, vcc                       // 000000001C48: 002A371D
	v_bfe_u32 v18, v0, 16, 1                                   // 000000001C4C: D1C80012 02052100
	v_add3_u32 v17, v9, v17, s0                                // 000000001C54: D1FF0011 00022309
	v_lshrrev_b32_e32 v16, 16, v16                             // 000000001C5C: 20202090
	v_cmp_o_f32_e32 vcc, v8, v8                                // 000000001C60: 7C8E1108
	v_bfe_u32 v19, v1, 16, 1                                   // 000000001C64: D1C80013 02052101
	v_add3_u32 v18, v0, v18, s0                                // 000000001C6C: D1FF0012 00022500
	v_lshrrev_b32_e32 v17, 16, v17                             // 000000001C74: 20222290
	v_cndmask_b32_e32 v8, v29, v16, vcc                        // 000000001C78: 0010211D
	v_cmp_o_f32_e32 vcc, v9, v9                                // 000000001C7C: 7C8E1309
	v_add3_u32 v19, v1, v19, s0                                // 000000001C80: D1FF0013 00022701
	v_lshrrev_b32_e32 v18, 16, v18                             // 000000001C88: 20242490
	v_cndmask_b32_e32 v9, v29, v17, vcc                        // 000000001C8C: 0012231D
	v_cmp_o_f32_e32 vcc, v0, v0                                // 000000001C90: 7C8E0100
	v_lshrrev_b32_e32 v19, 16, v19                             // 000000001C94: 20262690
	v_cndmask_b32_e32 v0, v29, v18, vcc                        // 000000001C98: 0000251D
	v_cmp_o_f32_e32 vcc, v1, v1                                // 000000001C9C: 7C8E0301
	s_mov_b32 s1, 0x5040100                                    // 000000001CA0: BE8100FF 05040100
	v_cndmask_b32_e32 v1, v29, v19, vcc                        // 000000001CA8: 0002271D
	v_perm_b32 v9, v9, v8, s1                                  // 000000001CAC: D1ED0009 00061109
	v_perm_b32 v8, v1, v0, s1                                  // 000000001CB4: D1ED0008 00060101
	v_pk_fma_f32 v[0:1], v[14:15], v[28:29], v[6:7] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]// 000000001CBC: D3B04100 2C1A390E
	v_pk_fma_f32 v[6:7], v[12:13], v[28:29], v[10:11] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]// 000000001CC4: D3B04106 2C2A390C
	v_pk_mul_f32 v[0:1], v[2:3], v[0:1]                        // 000000001CCC: D3B14000 18020102
	v_pk_mul_f32 v[2:3], v[2:3], v[6:7]                        // 000000001CD4: D3B14002 18020D02
	v_bfe_u32 v6, v2, 16, 1                                    // 000000001CDC: D1C80006 02052102
	v_bfe_u32 v7, v3, 16, 1                                    // 000000001CE4: D1C80007 02052103
	v_add3_u32 v6, v2, v6, s0                                  // 000000001CEC: D1FF0006 00020D02
	v_cmp_o_f32_e32 vcc, v2, v2                                // 000000001CF4: 7C8E0502
	global_store_dwordx2 v[4:5], v[8:9], off offset:512        // 000000001CF8: DC748200 007F0804
	v_bfe_u32 v8, v0, 16, 1                                    // 000000001D00: D1C80008 02052100
	v_add3_u32 v7, v3, v7, s0                                  // 000000001D08: D1FF0007 00020F03
	v_cndmask_b32_sdwa v2, v29, v6, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D10: 00040CF9 0506061D
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000001D18: 7C8E0703
	v_bfe_u32 v9, v1, 16, 1                                    // 000000001D1C: D1C80009 02052101
	v_add3_u32 v8, v0, v8, s0                                  // 000000001D24: D1FF0008 00021100
	v_cndmask_b32_sdwa v3, v29, v7, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D2C: 00060EF9 0506061D
	v_cmp_o_f32_e32 vcc, v0, v0                                // 000000001D34: 7C8E0100
	v_add3_u32 v9, v1, v9, s0                                  // 000000001D38: D1FF0009 00021301
	v_cndmask_b32_sdwa v0, v29, v8, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D40: 000010F9 0506061D
	v_cmp_o_f32_e32 vcc, v1, v1                                // 000000001D48: 7C8E0301
	v_cndmask_b32_sdwa v1, v29, v9, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D4C: 000212F9 0506061D
	v_perm_b32 v23, v23, v22, s1                               // 000000001D54: D1ED0017 00062D17
	v_perm_b32 v22, v21, v20, s1                               // 000000001D5C: D1ED0016 00062915
	v_perm_b32 v3, v3, v2, s1                                  // 000000001D64: D1ED0003 00060503
	v_perm_b32 v2, v1, v0, s1                                  // 000000001D6C: D1ED0002 00060101
	global_store_dwordx2 v[4:5], v[22:23], off                 // 000000001D74: DC748000 007F1604
	global_store_dwordx2 v[4:5], v[2:3], off offset:1024       // 000000001D7C: DC748400 007F0204
	s_endpgm                                                   // 000000001D84: BF810000
	s_nop 0                                                    // 000000001D88: BF800000
	s_nop 0                                                    // 000000001D8C: BF800000
	s_nop 0                                                    // 000000001D90: BF800000
	s_nop 0                                                    // 000000001D94: BF800000
	s_nop 0                                                    // 000000001D98: BF800000
	s_nop 0                                                    // 000000001D9C: BF800000
	s_nop 0                                                    // 000000001DA0: BF800000
	s_nop 0                                                    // 000000001DA4: BF800000
	s_nop 0                                                    // 000000001DA8: BF800000
	s_nop 0                                                    // 000000001DAC: BF800000
	s_nop 0                                                    // 000000001DB0: BF800000
	s_nop 0                                                    // 000000001DB4: BF800000
	s_nop 0                                                    // 000000001DB8: BF800000
	s_nop 0                                                    // 000000001DBC: BF800000
	s_nop 0                                                    // 000000001DC0: BF800000
	s_nop 0                                                    // 000000001DC4: BF800000
	s_nop 0                                                    // 000000001DC8: BF800000
	s_nop 0                                                    // 000000001DCC: BF800000
	s_nop 0                                                    // 000000001DD0: BF800000
	s_nop 0                                                    // 000000001DD4: BF800000
	s_nop 0                                                    // 000000001DD8: BF800000
	s_nop 0                                                    // 000000001DDC: BF800000
	s_nop 0                                                    // 000000001DE0: BF800000
	s_nop 0                                                    // 000000001DE4: BF800000
	s_nop 0                                                    // 000000001DE8: BF800000
	s_nop 0                                                    // 000000001DEC: BF800000
	s_nop 0                                                    // 000000001DF0: BF800000
	s_nop 0                                                    // 000000001DF4: BF800000
	s_nop 0                                                    // 000000001DF8: BF800000
	s_nop 0                                                    // 000000001DFC: BF800000
	s_nop 0                                                    // 000000001E00: BF800000
	s_nop 0                                                    // 000000001E04: BF800000
	s_nop 0                                                    // 000000001E08: BF800000
	s_nop 0                                                    // 000000001E0C: BF800000
	s_nop 0                                                    // 000000001E10: BF800000
	s_nop 0                                                    // 000000001E14: BF800000
	s_nop 0                                                    // 000000001E18: BF800000
	s_nop 0                                                    // 000000001E1C: BF800000
	s_nop 0                                                    // 000000001E20: BF800000
	s_nop 0                                                    // 000000001E24: BF800000
	s_nop 0                                                    // 000000001E28: BF800000
	s_nop 0                                                    // 000000001E2C: BF800000
	s_nop 0                                                    // 000000001E30: BF800000
	s_nop 0                                                    // 000000001E34: BF800000
	s_nop 0                                                    // 000000001E38: BF800000
	s_nop 0                                                    // 000000001E3C: BF800000
	s_nop 0                                                    // 000000001E40: BF800000
	s_nop 0                                                    // 000000001E44: BF800000
	s_nop 0                                                    // 000000001E48: BF800000
	s_nop 0                                                    // 000000001E4C: BF800000
	s_nop 0                                                    // 000000001E50: BF800000
	s_nop 0                                                    // 000000001E54: BF800000
	s_nop 0                                                    // 000000001E58: BF800000
	s_nop 0                                                    // 000000001E5C: BF800000
	s_nop 0                                                    // 000000001E60: BF800000
	s_nop 0                                                    // 000000001E64: BF800000
	s_nop 0                                                    // 000000001E68: BF800000
	s_nop 0                                                    // 000000001E6C: BF800000
	s_nop 0                                                    // 000000001E70: BF800000
	s_nop 0                                                    // 000000001E74: BF800000
	s_nop 0                                                    // 000000001E78: BF800000
	s_nop 0                                                    // 000000001E7C: BF800000
	s_nop 0                                                    // 000000001E80: BF800000
	s_nop 0                                                    // 000000001E84: BF800000
	s_nop 0                                                    // 000000001E88: BF800000
	s_nop 0                                                    // 000000001E8C: BF800000
	s_nop 0                                                    // 000000001E90: BF800000
	s_nop 0                                                    // 000000001E94: BF800000
	s_nop 0                                                    // 000000001E98: BF800000
	s_nop 0                                                    // 000000001E9C: BF800000
	s_nop 0                                                    // 000000001EA0: BF800000
	s_nop 0                                                    // 000000001EA4: BF800000
	s_nop 0                                                    // 000000001EA8: BF800000
	s_nop 0                                                    // 000000001EAC: BF800000
	s_nop 0                                                    // 000000001EB0: BF800000
	s_nop 0                                                    // 000000001EB4: BF800000
	s_nop 0                                                    // 000000001EB8: BF800000
	s_nop 0                                                    // 000000001EBC: BF800000
	s_nop 0                                                    // 000000001EC0: BF800000
	s_nop 0                                                    // 000000001EC4: BF800000
	s_nop 0                                                    // 000000001EC8: BF800000
	s_nop 0                                                    // 000000001ECC: BF800000
	s_nop 0                                                    // 000000001ED0: BF800000
	s_nop 0                                                    // 000000001ED4: BF800000
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
