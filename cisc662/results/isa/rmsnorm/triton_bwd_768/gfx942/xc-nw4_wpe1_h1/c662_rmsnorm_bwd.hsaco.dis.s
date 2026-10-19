
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_bwd_768/gfx942/xc-nw4_wpe1_h1/c662_rmsnorm_bwd.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <c662_rmsnorm_bwd>:
	s_load_dwordx2 s[2:3], s[0:1], 0x0                         // 000000001700: C0060080 00000000
	s_load_dwordx8 s[4:11], s[0:1], 0x8                        // 000000001708: C00E0100 00000008
	s_load_dwordx4 s[12:15], s[0:1], 0x28                      // 000000001710: C00A0300 00000028
	s_waitcnt lgkmcnt(0)                                       // 000000001718: BF8CC07F
	s_branch 56                                                // 00000000171C: BF820038 <c662_rmsnorm_bwd+0x100>
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
	v_lshrrev_b32_e32 v1, 6, v0                                // 000000001800: 20020086
	v_lshl_or_b32 v4, s16, 2, v1                               // 000000001804: D2000004 04050410
	v_mov_b32_e32 v1, 0                                        // 00000000180C: 7E020280
	v_cmp_gt_i32_e32 vcc, s10, v4                              // 000000001810: 7D88080A
	v_mov_b32_e32 v2, 0                                        // 000000001814: 7E040280
	v_mov_b32_e32 v3, v1                                       // 000000001818: 7E060301
	s_and_saveexec_b64 s[0:1], vcc                             // 00000000181C: BE80206A
	s_cbranch_execz 7                                          // 000000001820: BF880007 <c662_rmsnorm_bwd+0x140>
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000001824: 220A089F
	v_lshl_add_u64 v[2:3], v[4:5], 2, s[6:7]                   // 000000001828: D2080002 00190504
	global_load_dword v2, v[2:3], off                          // 000000001830: DC508000 027F0002
	s_waitcnt vmcnt(0)                                         // 000000001838: BF8C0F70
	v_mov_b32_e32 v3, v2                                       // 00000000183C: 7E060302
	s_or_b64 exec, exec, s[0:1]                                // 000000001840: 87FE007E
	v_mul_lo_u32 v6, s12, v4                                   // 000000001844: D2850006 0002080C
	v_lshlrev_b32_e32 v0, 2, v0                                // 00000000184C: 24000082
	v_ashrrev_i32_e32 v7, 31, v6                               // 000000001850: 220E0C9F
	v_and_b32_e32 v0, 0xfc, v0                                 // 000000001854: 260000FF 000000FC
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[4:5]                   // 00000000185C: D2080006 00110306
	v_lshlrev_b32_e32 v0, 1, v0                                // 000000001864: 24000081
	v_lshl_add_u64 v[8:9], v[6:7], 0, v[0:1]                   // 000000001868: D2080008 04010106
	v_mov_b32_e32 v7, 0                                        // 000000001870: 7E0E0280
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001874: BE80206A
	s_cbranch_execz 4                                          // 000000001878: BF880004 <c662_rmsnorm_bwd+0x18c>
	global_load_dwordx2 v[6:7], v[8:9], off                    // 00000000187C: DC548000 067F0008
	s_waitcnt vmcnt(0)                                         // 000000001884: BF8C0F70
	v_mov_b32_e32 v1, v6                                       // 000000001888: 7E020306
	s_or_b64 exec, exec, s[0:1]                                // 00000000188C: 87FE007E
	v_mul_lo_u32 v10, s11, v4                                  // 000000001890: D285000A 0002080B
	v_ashrrev_i32_e32 v11, 31, v10                             // 000000001898: 2216149F
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[2:3]               // 00000000189C: D208000A 0009030A
	v_mov_b32_e32 v15, 0                                       // 0000000018A4: 7E1E0280
	v_mov_b32_e32 v14, v0                                      // 0000000018A8: 7E1C0300
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[14:15]             // 0000000018AC: D208000A 0439010A
	v_mov_b32_e32 v12, 0                                       // 0000000018B4: 7E180280
	v_mov_b32_e32 v13, 0                                       // 0000000018B8: 7E1A0280
	s_and_saveexec_b64 s[0:1], vcc                             // 0000000018BC: BE80206A
	s_cbranch_execz 2                                          // 0000000018C0: BF880002 <c662_rmsnorm_bwd+0x1cc>
	global_load_dwordx2 v[12:13], v[10:11], off                // 0000000018C4: DC548000 0C7F000A
	s_or_b64 exec, exec, s[0:1]                                // 0000000018CC: 87FE007E
	v_mov_b32_e32 v17, 0                                       // 0000000018D0: 7E220280
	s_and_saveexec_b64 s[0:1], vcc                             // 0000000018D4: BE80206A
	s_cbranch_execz 4                                          // 0000000018D8: BF880004 <c662_rmsnorm_bwd+0x1ec>
	global_load_dwordx2 v[16:17], v[8:9], off offset:512       // 0000000018DC: DC548200 107F0008
	s_waitcnt vmcnt(0)                                         // 0000000018E4: BF8C0F70
	v_mov_b32_e32 v15, v16                                     // 0000000018E8: 7E1E0310
	s_or_b64 exec, exec, s[0:1]                                // 0000000018EC: 87FE007E
	v_mov_b32_e32 v22, 0                                       // 0000000018F0: 7E2C0280
	v_mov_b32_e32 v18, 0                                       // 0000000018F4: 7E240280
	v_mov_b32_e32 v19, 0                                       // 0000000018F8: 7E260280
	s_and_saveexec_b64 s[0:1], vcc                             // 0000000018FC: BE80206A
	s_cbranch_execz 2                                          // 000000001900: BF880002 <c662_rmsnorm_bwd+0x20c>
	global_load_dwordx2 v[18:19], v[10:11], off offset:512     // 000000001904: DC548200 127F000A
	s_or_b64 exec, exec, s[0:1]                                // 00000000190C: 87FE007E
	v_mov_b32_e32 v23, 0                                       // 000000001910: 7E2E0280
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001914: BE80206A
	s_cbranch_execz 2                                          // 000000001918: BF880002 <c662_rmsnorm_bwd+0x224>
	global_load_dwordx2 v[22:23], v[8:9], off offset:1024      // 00000000191C: DC548400 167F0008
	s_or_b64 exec, exec, s[0:1]                                // 000000001924: 87FE007E
	v_mov_b32_e32 v20, 0                                       // 000000001928: 7E280280
	v_mov_b32_e32 v21, 0                                       // 00000000192C: 7E2A0280
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001930: BE80206A
	s_cbranch_execz 2                                          // 000000001934: BF880002 <c662_rmsnorm_bwd+0x240>
	global_load_dwordx2 v[20:21], v[10:11], off offset:1024    // 000000001938: DC548400 147F000A
	s_or_b64 exec, exec, s[0:1]                                // 000000001940: 87FE007E
	s_waitcnt vmcnt(0)                                         // 000000001944: BF8C0F70
	v_and_b32_e32 v9, 0xffff0000, v18                          // 000000001948: 261224FF FFFF0000
	v_lshlrev_b32_e32 v8, 16, v18                              // 000000001950: 24102490
	v_and_b32_e32 v11, 0xffff0000, v19                         // 000000001954: 261626FF FFFF0000
	v_lshlrev_b32_e32 v10, 16, v19                             // 00000000195C: 24142690
	v_and_b32_e32 v19, 0xffff0000, v15                         // 000000001960: 26261EFF FFFF0000
	v_lshlrev_b32_e32 v18, 16, v15                             // 000000001968: 24241E90
	v_and_b32_e32 v15, 0xffff0000, v17                         // 00000000196C: 261E22FF FFFF0000
	v_lshlrev_b32_e32 v14, 16, v17                             // 000000001974: 241C2290
	v_and_b32_e32 v31, 0xffff0000, v22                         // 000000001978: 263E2CFF FFFF0000
	v_lshlrev_b32_e32 v30, 16, v22                             // 000000001980: 243C2C90
	v_and_b32_e32 v33, 0xffff0000, v23                         // 000000001984: 26422EFF FFFF0000
	v_lshlrev_b32_e32 v32, 16, v23                             // 00000000198C: 24402E90
	v_and_b32_e32 v23, 0xffff0000, v12                         // 000000001990: 262E18FF FFFF0000
	v_lshlrev_b32_e32 v22, 16, v12                             // 000000001998: 242C1890
	v_and_b32_e32 v25, 0xffff0000, v13                         // 00000000199C: 26321AFF FFFF0000
	v_lshlrev_b32_e32 v24, 16, v13                             // 0000000019A4: 24301A90
	v_and_b32_e32 v17, 0xffff0000, v1                          // 0000000019A8: 262202FF FFFF0000
	v_lshlrev_b32_e32 v16, 16, v1                              // 0000000019B0: 24200290
	v_and_b32_e32 v27, 0xffff0000, v7                          // 0000000019B4: 26360EFF FFFF0000
	v_lshlrev_b32_e32 v26, 16, v7                              // 0000000019BC: 24340E90
	v_and_b32_e32 v7, 0xffff0000, v20                          // 0000000019C0: 260E28FF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v20                              // 0000000019C8: 240C2890
	v_and_b32_e32 v13, 0xffff0000, v21                         // 0000000019CC: 261A2AFF FFFF0000
	v_lshlrev_b32_e32 v12, 16, v21                             // 0000000019D4: 24182A90
	v_pk_mul_f32 v[20:21], v[2:3], v[18:19]                    // 0000000019D8: D3B14014 18022502
	v_pk_mul_f32 v[18:19], v[2:3], v[14:15]                    // 0000000019E0: D3B14012 18021D02
	v_pk_mul_f32 v[28:29], v[2:3], v[16:17]                    // 0000000019E8: D3B1401C 18022102
	v_pk_mul_f32 v[26:27], v[2:3], v[26:27]                    // 0000000019F0: D3B1401A 18023502
	v_pk_mul_f32 v[14:15], v[20:21], v[8:9]                    // 0000000019F8: D3B1400E 18021114
	v_pk_mul_f32 v[16:17], v[18:19], v[10:11]                  // 000000001A00: D3B14010 18021512
	v_pk_fma_f32 v[34:35], v[28:29], v[22:23], v[14:15]        // 000000001A08: D3B04022 1C3A2D1C
	v_pk_fma_f32 v[36:37], v[26:27], v[24:25], v[16:17]        // 000000001A10: D3B04024 1C42311A
	v_pk_mul_f32 v[16:17], v[2:3], v[30:31]                    // 000000001A18: D3B14010 18023D02
	v_pk_mul_f32 v[14:15], v[2:3], v[32:33]                    // 000000001A20: D3B1400E 18024102
	v_pk_fma_f32 v[30:31], v[16:17], v[6:7], v[34:35]          // 000000001A28: D3B0401E 1C8A0D10
	v_pk_fma_f32 v[32:33], v[14:15], v[12:13], v[36:37]        // 000000001A30: D3B04020 1C92190E
	v_add_f32_e32 v1, v30, v31                                 // 000000001A38: 02023F1E
	v_add_f32_e32 v1, v32, v1                                  // 000000001A3C: 02020320
	v_add_f32_e32 v1, v33, v1                                  // 000000001A40: 02020321
	s_nop 1                                                    // 000000001A44: BF800001
	v_add_f32_dpp v1, v1, v1 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A48: 020202FA FF091801
	s_nop 1                                                    // 000000001A50: BF800001
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A54: 020202FA FF091401
	s_nop 1                                                    // 000000001A5C: BF800001
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A60: 020202FA FF091201
	s_nop 1                                                    // 000000001A68: BF800001
	v_add_f32_dpp v1, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A6C: 020202FA FF091101
	v_mov_b32_e32 v5, v1                                       // 000000001A74: 7E0A0301
	s_nop 1                                                    // 000000001A78: BF800001
	v_mov_b32_dpp v5, v5 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1// 000000001A7C: 7E0A02FA AF094205
	v_add_f32_e32 v1, v5, v1                                   // 000000001A84: 02020305
	s_nop 1                                                    // 000000001A88: BF800001
	v_add_f32_dpp v1, v1, v1 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001A8C: 020202FA FF094301
	s_nop 0                                                    // 000000001A94: BF800000
	v_readlane_b32 s0, v1, 63                                  // 000000001A98: D2890000 00017F01
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001AA0: BE82206A
	s_cbranch_execz 163                                        // 000000001AA4: BF8800A3 <c662_rmsnorm_bwd+0x634>
	s_mov_b32 s1, 0x44400000                                   // 000000001AA8: BE8100FF 44400000
	v_mov_b32_e32 v30, s0                                      // 000000001AB0: 7E3C0200
	v_div_scale_f32 v31, s[2:3], s1, s1, v30                   // 000000001AB4: D1E0021F 04780201
	v_rcp_f32_e32 v32, v31                                     // 000000001ABC: 7E40451F
	v_mul_lo_u32 v4, s13, v4                                   // 000000001AC0: D2850004 0002080D
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000001AC8: 220A089F
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[8:9]                   // 000000001ACC: D2080004 00210304
	v_mov_b32_e32 v1, 0                                        // 000000001AD4: 7E020280
	v_lshl_add_u64 v[0:1], v[4:5], 0, v[0:1]                   // 000000001AD8: D2080000 04010104
	v_fma_f32 v4, -v31, v32, 1.0                               // 000000001AE0: D1CB0004 23CA411F
	v_fmac_f32_e32 v32, v4, v32                                // 000000001AE8: 76404104
	v_mov_b32_e32 v4, 0x44400000                               // 000000001AEC: 7E0802FF 44400000
	v_div_scale_f32 v4, vcc, s0, v4, s0                        // 000000001AF4: D1E06A04 00020800
	v_mul_f32_e32 v5, v4, v32                                  // 000000001AFC: 0A0A4104
	v_fma_f32 v33, -v31, v5, v4                                // 000000001B00: D1CB0021 24120B1F
	v_fmac_f32_e32 v5, v33, v32                                // 000000001B08: 760A4121
	v_fma_f32 v4, -v31, v5, v4                                 // 000000001B0C: D1CB0004 24120B1F
	v_div_fmas_f32 v4, v4, v32, v5                             // 000000001B14: D1E20004 04164104
	v_div_fixup_f32 v4, v4, s1, v30                            // 000000001B1C: D1DE0004 04780304
	v_pk_fma_f32 v[24:25], v[26:27], v[4:5], v[24:25] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]// 000000001B24: D3B04118 2C62091A
	v_pk_fma_f32 v[22:23], v[28:29], v[4:5], v[22:23] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]// 000000001B2C: D3B04116 2C5A091C
	v_pk_mul_f32 v[24:25], v[2:3], v[24:25]                    // 000000001B34: D3B14018 18023102
	s_movk_i32 s0, 0x7fff                                      // 000000001B3C: B0007FFF
	v_bfe_u32 v5, v24, 16, 1                                   // 000000001B40: D1C80005 02052118
	v_pk_mul_f32 v[22:23], v[2:3], v[22:23]                    // 000000001B48: D3B14016 18022D02
	v_bfe_u32 v26, v25, 16, 1                                  // 000000001B50: D1C8001A 02052119
	v_add3_u32 v5, v24, v5, s0                                 // 000000001B58: D1FF0005 00020B18
	v_bfe_u32 v27, v22, 16, 1                                  // 000000001B60: D1C8001B 02052116
	v_add3_u32 v26, v25, v26, s0                               // 000000001B68: D1FF001A 00023519
	v_lshrrev_b32_e32 v5, 16, v5                               // 000000001B70: 200A0A90
	v_mov_b32_e32 v29, 0x7fff                                  // 000000001B74: 7E3A02FF 00007FFF
	v_cmp_o_f32_e32 vcc, v24, v24                              // 000000001B7C: 7C8E3118
	v_add3_u32 v27, v22, v27, s0                               // 000000001B80: D1FF001B 00023716
	v_lshrrev_b32_e32 v26, 16, v26                             // 000000001B88: 20343490
	v_cndmask_b32_e32 v5, v29, v5, vcc                         // 000000001B8C: 000A0B1D
	v_cmp_o_f32_e32 vcc, v25, v25                              // 000000001B90: 7C8E3319
	v_bfe_u32 v28, v23, 16, 1                                  // 000000001B94: D1C8001C 02052117
	v_lshrrev_b32_e32 v27, 16, v27                             // 000000001B9C: 20363690
	v_cndmask_b32_e32 v24, v29, v26, vcc                       // 000000001BA0: 0030351D
	s_mov_b32 s1, 0x5040100                                    // 000000001BA4: BE8100FF 05040100
	v_cmp_o_f32_e32 vcc, v22, v22                              // 000000001BAC: 7C8E2D16
	v_add3_u32 v28, v23, v28, s0                               // 000000001BB0: D1FF001C 00023917
	v_perm_b32 v25, v24, v5, s1                                // 000000001BB8: D1ED0019 00060B18
	v_cndmask_b32_e32 v5, v29, v27, vcc                        // 000000001BC0: 000A371D
	v_lshrrev_b32_e32 v28, 16, v28                             // 000000001BC4: 20383890
	v_cmp_o_f32_e32 vcc, v23, v23                              // 000000001BC8: 7C8E2F17
	v_pk_fma_f32 v[10:11], v[18:19], v[4:5], v[10:11] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]// 000000001BCC: D3B0410A 2C2A0912
	v_pk_fma_f32 v[8:9], v[20:21], v[4:5], v[8:9] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]// 000000001BD4: D3B04108 2C220914
	v_cndmask_b32_e32 v22, v29, v28, vcc                       // 000000001BDC: 002C391D
	v_pk_mul_f32 v[10:11], v[2:3], v[10:11]                    // 000000001BE0: D3B1400A 18021502
	v_perm_b32 v24, v22, v5, s1                                // 000000001BE8: D1ED0018 00060B16
	v_bfe_u32 v5, v10, 16, 1                                   // 000000001BF0: D1C80005 0205210A
	v_pk_mul_f32 v[8:9], v[2:3], v[8:9]                        // 000000001BF8: D3B14008 18021102
	v_bfe_u32 v18, v11, 16, 1                                  // 000000001C00: D1C80012 0205210B
	v_add3_u32 v5, v10, v5, s0                                 // 000000001C08: D1FF0005 00020B0A
	v_bfe_u32 v19, v8, 16, 1                                   // 000000001C10: D1C80013 02052108
	v_add3_u32 v18, v11, v18, s0                               // 000000001C18: D1FF0012 0002250B
	v_lshrrev_b32_e32 v5, 16, v5                               // 000000001C20: 200A0A90
	v_cmp_o_f32_e32 vcc, v10, v10                              // 000000001C24: 7C8E150A
	v_bfe_u32 v20, v9, 16, 1                                   // 000000001C28: D1C80014 02052109
	v_add3_u32 v19, v8, v19, s0                                // 000000001C30: D1FF0013 00022708
	v_lshrrev_b32_e32 v18, 16, v18                             // 000000001C38: 20242490
	v_cndmask_b32_e32 v5, v29, v5, vcc                         // 000000001C3C: 000A0B1D
	v_cmp_o_f32_e32 vcc, v11, v11                              // 000000001C40: 7C8E170B
	v_add3_u32 v20, v9, v20, s0                                // 000000001C44: D1FF0014 00022909
	v_lshrrev_b32_e32 v19, 16, v19                             // 000000001C4C: 20262690
	v_cndmask_b32_e32 v10, v29, v18, vcc                       // 000000001C50: 0014251D
	v_cmp_o_f32_e32 vcc, v8, v8                                // 000000001C54: 7C8E1108
	v_lshrrev_b32_e32 v20, 16, v20                             // 000000001C58: 20282890
	v_perm_b32 v11, v10, v5, s1                                // 000000001C5C: D1ED000B 00060B0A
	v_cndmask_b32_e32 v5, v29, v19, vcc                        // 000000001C64: 000A271D
	v_cmp_o_f32_e32 vcc, v9, v9                                // 000000001C68: 7C8E1309
	v_pk_fma_f32 v[6:7], v[16:17], v[4:5], v[6:7] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]// 000000001C6C: D3B04106 2C1A0910
	global_store_dwordx2 v[0:1], v[24:25], off                 // 000000001C74: DC748000 007F1800
	v_cndmask_b32_e32 v8, v29, v20, vcc                        // 000000001C7C: 0010291D
	v_perm_b32 v10, v8, v5, s1                                 // 000000001C80: D1ED000A 00060B08
	v_pk_fma_f32 v[4:5], v[14:15], v[4:5], v[12:13] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[1,0,0]// 000000001C88: D3B04104 2C32090E
	v_pk_mul_f32 v[6:7], v[2:3], v[6:7]                        // 000000001C90: D3B14006 18020D02
	v_pk_mul_f32 v[2:3], v[2:3], v[4:5]                        // 000000001C98: D3B14002 18020902
	v_bfe_u32 v8, v6, 16, 1                                    // 000000001CA0: D1C80008 02052106
	v_bfe_u32 v4, v2, 16, 1                                    // 000000001CA8: D1C80004 02052102
	v_bfe_u32 v5, v3, 16, 1                                    // 000000001CB0: D1C80005 02052103
	v_add3_u32 v4, v2, v4, s0                                  // 000000001CB8: D1FF0004 00020902
	v_cmp_o_f32_e32 vcc, v2, v2                                // 000000001CC0: 7C8E0502
	v_add3_u32 v5, v3, v5, s0                                  // 000000001CC4: D1FF0005 00020B03
	v_bfe_u32 v9, v7, 16, 1                                    // 000000001CCC: D1C80009 02052107
	v_cndmask_b32_sdwa v2, v29, v4, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001CD4: 000408F9 0506061D
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000001CDC: 7C8E0703
	v_add3_u32 v8, v6, v8, s0                                  // 000000001CE0: D1FF0008 00021106
	v_add3_u32 v9, v7, v9, s0                                  // 000000001CE8: D1FF0009 00021307
	v_cndmask_b32_sdwa v3, v29, v5, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001CF0: 00060AF9 0506061D
	v_cmp_o_f32_e32 vcc, v6, v6                                // 000000001CF8: 7C8E0D06
	v_perm_b32 v3, v3, v2, s1                                  // 000000001CFC: D1ED0003 00060503
	global_store_dwordx2 v[0:1], v[10:11], off offset:512      // 000000001D04: DC748200 007F0A00
	v_cndmask_b32_sdwa v2, v29, v8, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D0C: 000410F9 0506061D
	v_cmp_o_f32_e32 vcc, v7, v7                                // 000000001D14: 7C8E0F07
	s_nop 1                                                    // 000000001D18: BF800001
	v_cndmask_b32_sdwa v4, v29, v9, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D1C: 000812F9 0506061D
	v_perm_b32 v2, v4, v2, s1                                  // 000000001D24: D1ED0002 00060504
	global_store_dwordx2 v[0:1], v[2:3], off offset:1024       // 000000001D2C: DC748400 007F0200
	s_endpgm                                                   // 000000001D34: BF810000
	s_nop 0                                                    // 000000001D38: BF800000
	s_nop 0                                                    // 000000001D3C: BF800000
	s_nop 0                                                    // 000000001D40: BF800000
	s_nop 0                                                    // 000000001D44: BF800000
	s_nop 0                                                    // 000000001D48: BF800000
	s_nop 0                                                    // 000000001D4C: BF800000
	s_nop 0                                                    // 000000001D50: BF800000
	s_nop 0                                                    // 000000001D54: BF800000
	s_nop 0                                                    // 000000001D58: BF800000
	s_nop 0                                                    // 000000001D5C: BF800000
	s_nop 0                                                    // 000000001D60: BF800000
	s_nop 0                                                    // 000000001D64: BF800000
	s_nop 0                                                    // 000000001D68: BF800000
	s_nop 0                                                    // 000000001D6C: BF800000
	s_nop 0                                                    // 000000001D70: BF800000
	s_nop 0                                                    // 000000001D74: BF800000
	s_nop 0                                                    // 000000001D78: BF800000
	s_nop 0                                                    // 000000001D7C: BF800000
	s_nop 0                                                    // 000000001D80: BF800000
	s_nop 0                                                    // 000000001D84: BF800000
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
