
/u/tmahmood1/project/nanochat/cisc662/results/isa/qkrope/triton_v1/gfx908/xc-nw4_wpe1_h1/c662_qkrope_bwd.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001900 <c662_qkrope_bwd>:
	s_load_dword s0, s[4:5], 0x58                              // 000000001900: C0020002 00000058
	s_load_dwordx2 s[8:9], s[4:5], 0x64                        // 000000001908: C0060202 00000064
	v_lshrrev_b32_e32 v7, 4, v0                                // 000000001910: 200E0084
	v_lshlrev_b32_e32 v6, 2, v0                                // 000000001914: 240C0082
	s_waitcnt lgkmcnt(0)                                       // 000000001918: BF8CC07F
	s_cmp_ge_i32 s6, s0                                        // 00000000191C: BF030006
	s_cbranch_scc0 99                                          // 000000001920: BF840063 <c662_qkrope_bwd+0x1b0>
	s_load_dword s1, s[4:5], 0x60                              // 000000001924: C0020042 00000060
	s_load_dword s7, s[4:5], 0x54                              // 00000000192C: C00201C2 00000054
	s_abs_i32 s10, s8                                          // 000000001934: BE8A3008
	v_cvt_f32_u32_e32 v1, s10                                  // 000000001938: 7E020C0A
	s_sub_i32 s0, s6, s0                                       // 00000000193C: 81800006
	s_waitcnt lgkmcnt(0)                                       // 000000001940: BF8CC07F
	s_abs_i32 s2, s1                                           // 000000001944: BE823001
	v_cvt_f32_u32_e32 v0, s2                                   // 000000001948: 7E000C02
	s_sub_i32 s12, 0, s2                                       // 00000000194C: 818C0280
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000001950: 7E024701
	s_lshl_b32 s3, s0, 4                                       // 000000001954: 8E038400
	v_rcp_iflag_f32_e32 v0, v0                                 // 000000001958: 7E004700
	s_bfe_i32 s0, s0, 0x1001b                                  // 00000000195C: 9300FF00 0001001B
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001964: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v1, v1                                   // 00000000196C: 7E020F01
	v_mul_f32_e32 v0, 0x4f7ffffe, v0                           // 000000001970: 0A0000FF 4F7FFFFE
	v_cvt_u32_f32_e32 v2, v0                                   // 000000001978: 7E040F00
	v_or_b32_e32 v0, s3, v7                                    // 00000000197C: 28000E03
	v_add_u32_e32 v4, s0, v0                                   // 000000001980: 68080000
	s_sub_i32 s11, 0, s10                                      // 000000001984: 818B0A80
	v_mul_lo_u32 v3, s12, v2                                   // 000000001988: D2850003 0002040C
	v_xor_b32_e32 v4, s0, v4                                   // 000000001990: 2A080800
	v_mul_lo_u32 v5, s11, v1                                   // 000000001994: D2850005 0002020B
	s_ashr_i32 s1, s1, 31                                      // 00000000199C: 90019F01
	v_mul_hi_u32 v3, v2, v3                                    // 0000000019A0: D2860003 00020702
	s_xor_b32 s0, s0, s1                                       // 0000000019A8: 88000100
	v_add_u32_e32 v2, v2, v3                                   // 0000000019AC: 68040702
	v_mul_hi_u32 v2, v4, v2                                    // 0000000019B0: D2860002 00020504
	v_mul_hi_u32 v3, v1, v5                                    // 0000000019B8: D2860003 00020B01
	v_mul_lo_u32 v5, v2, s2                                    // 0000000019C0: D2850005 00000502
	v_add_u32_e32 v1, v1, v3                                   // 0000000019C8: 68020701
	v_add_u32_e32 v3, 1, v2                                    // 0000000019CC: 68060481
	v_sub_u32_e32 v4, v4, v5                                   // 0000000019D0: 6A080B04
	v_cmp_le_u32_e32 vcc, s2, v4                               // 0000000019D4: 7D960802
	v_cndmask_b32_e32 v2, v2, v3, vcc                          // 0000000019D8: 00040702
	v_subrev_u32_e32 v3, s2, v4                                // 0000000019DC: 6C060802
	v_cndmask_b32_e32 v3, v4, v3, vcc                          // 0000000019E0: 00060704
	v_add_u32_e32 v4, 1, v2                                    // 0000000019E4: 68080481
	v_cmp_le_u32_e32 vcc, s2, v3                               // 0000000019E8: 7D960602
	v_cndmask_b32_e32 v2, v2, v4, vcc                          // 0000000019EC: 00040902
	v_xor_b32_e32 v2, s0, v2                                   // 0000000019F0: 2A040400
	v_subrev_u32_e32 v3, s0, v2                                // 0000000019F4: 6C060400
	v_sub_u32_e32 v2, 0, v3                                    // 0000000019F8: 6A040680
	v_max_i32_e32 v4, v3, v2                                   // 0000000019FC: 1A080503
	v_mul_hi_u32 v1, v4, v1                                    // 000000001A00: D2860001 00020304
	v_ashrrev_i32_e32 v3, 31, v3                               // 000000001A08: 2206069F
	s_load_dwordx2 s[2:3], s[4:5], 0x18                        // 000000001A0C: C0060082 00000018
	v_and_b32_e32 v2, 60, v6                                   // 000000001A14: 26040CBC
	v_mul_lo_u32 v5, v1, s10                                   // 000000001A18: D2850005 00001501
	v_lshl_or_b32 v1, v0, 7, v2                                // 000000001A20: D2000001 04090F00
	v_cmp_gt_i32_e32 vcc, s7, v0                               // 000000001A28: 7D880007
	v_sub_u32_e32 v4, v4, v5                                   // 000000001A2C: 6A080B04
	v_subrev_u32_e32 v5, s10, v4                               // 000000001A30: 6C0A080A
	v_cmp_le_u32_e64 s[0:1], s10, v4                           // 000000001A34: D0CB0000 0002080A
	v_cndmask_b32_e64 v4, v4, v5, s[0:1]                       // 000000001A3C: D1000004 00020B04
	v_subrev_u32_e32 v5, s10, v4                               // 000000001A44: 6C0A080A
	v_cmp_le_u32_e64 s[0:1], s10, v4                           // 000000001A48: D0CB0000 0002080A
	v_cndmask_b32_e64 v4, v4, v5, s[0:1]                       // 000000001A50: D1000004 00020B04
	v_xor_b32_e32 v4, v4, v3                                   // 000000001A58: 2A080704
	v_sub_u32_e32 v3, v4, v3                                   // 000000001A5C: 6A060704
	v_mad_u64_u32 v[4:5], s[0:1], v3, s9, v[2:3]               // 000000001A60: D1E80004 04081303
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001A68: 2204029F
	v_lshlrev_b64 v[2:3], 1, v[1:2]                            // 000000001A6C: D28F0002 00020281
	s_waitcnt lgkmcnt(0)                                       // 000000001A74: BF8CC07F
	v_mov_b32_e32 v5, s3                                       // 000000001A78: 7E0A0203
	v_add_co_u32_e64 v2, s[0:1], s2, v2                        // 000000001A7C: D1190002 00020402
	v_addc_co_u32_e64 v3, s[0:1], v5, v3, s[0:1]               // 000000001A84: D11C0003 00020705
	s_and_b64 s[10:11], vcc, exec                              // 000000001A8C: 868A7E6A
	s_cbranch_execz 8                                          // 000000001A90: BF880008 <c662_qkrope_bwd+0x1b4>
	s_load_dwordx2 s[12:13], s[4:5], 0x8                       // 000000001A94: C0060302 00000008
	s_load_dwordx2 s[6:7], s[4:5], 0x28                        // 000000001A9C: C0060182 00000028
	s_load_dwordx2 s[2:3], s[4:5], 0x38                        // 000000001AA4: C0060082 00000038
	s_branch 99                                                // 000000001AAC: BF820063 <c662_qkrope_bwd+0x33c>
	s_mov_b64 s[10:11], 0                                      // 000000001AB0: BE8A0180
	s_load_dword s0, s[4:5], 0x5c                              // 000000001AB4: C0020002 0000005C
	s_load_dword s1, s[4:5], 0x50                              // 000000001ABC: C0020042 00000050
	s_abs_i32 s8, s8                                           // 000000001AC4: BE883008
	v_cvt_f32_u32_e32 v1, s8                                   // 000000001AC8: 7E020C08
	s_lshl_b32 s3, s6, 4                                       // 000000001ACC: 8E038406
	s_waitcnt lgkmcnt(0)                                       // 000000001AD0: BF8CC07F
	s_abs_i32 s2, s0                                           // 000000001AD4: BE823000
	v_cvt_f32_u32_e32 v0, s2                                   // 000000001AD8: 7E000C02
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000001ADC: 7E024701
	s_bfe_i32 s6, s6, 0x1001b                                  // 000000001AE0: 9306FF06 0001001B
	s_ashr_i32 s0, s0, 31                                      // 000000001AE8: 90009F00
	v_rcp_iflag_f32_e32 v0, v0                                 // 000000001AEC: 7E004700
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001AF0: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v1, v1                                   // 000000001AF8: 7E020F01
	s_xor_b32 s0, s6, s0                                       // 000000001AFC: 88000006
	v_mul_f32_e32 v0, 0x4f7ffffe, v0                           // 000000001B00: 0A0000FF 4F7FFFFE
	v_cvt_u32_f32_e32 v2, v0                                   // 000000001B08: 7E040F00
	v_or_b32_e32 v0, s3, v7                                    // 000000001B0C: 28000E03
	s_sub_i32 s3, 0, s2                                        // 000000001B10: 81830280
	v_add_u32_e32 v4, s6, v0                                   // 000000001B14: 68080006
	v_mul_lo_u32 v3, s3, v2                                    // 000000001B18: D2850003 00020403
	s_sub_i32 s3, 0, s8                                        // 000000001B20: 81830880
	v_xor_b32_e32 v4, s6, v4                                   // 000000001B24: 2A080806
	v_mul_lo_u32 v5, s3, v1                                    // 000000001B28: D2850005 00020203
	v_mul_hi_u32 v3, v2, v3                                    // 000000001B30: D2860003 00020702
	v_add_u32_e32 v2, v2, v3                                   // 000000001B38: 68040702
	v_mul_hi_u32 v2, v4, v2                                    // 000000001B3C: D2860002 00020504
	v_mul_hi_u32 v3, v1, v5                                    // 000000001B44: D2860003 00020B01
	v_mul_lo_u32 v5, v2, s2                                    // 000000001B4C: D2850005 00000502
	v_add_u32_e32 v1, v1, v3                                   // 000000001B54: 68020701
	v_add_u32_e32 v3, 1, v2                                    // 000000001B58: 68060481
	v_sub_u32_e32 v4, v4, v5                                   // 000000001B5C: 6A080B04
	v_cmp_le_u32_e32 vcc, s2, v4                               // 000000001B60: 7D960802
	v_cndmask_b32_e32 v2, v2, v3, vcc                          // 000000001B64: 00040702
	v_subrev_u32_e32 v3, s2, v4                                // 000000001B68: 6C060802
	v_cndmask_b32_e32 v3, v4, v3, vcc                          // 000000001B6C: 00060704
	v_add_u32_e32 v4, 1, v2                                    // 000000001B70: 68080481
	v_cmp_le_u32_e32 vcc, s2, v3                               // 000000001B74: 7D960602
	v_cndmask_b32_e32 v2, v2, v4, vcc                          // 000000001B78: 00040902
	v_xor_b32_e32 v2, s0, v2                                   // 000000001B7C: 2A040400
	v_subrev_u32_e32 v2, s0, v2                                // 000000001B80: 6C040400
	v_sub_u32_e32 v3, 0, v2                                    // 000000001B84: 6A060480
	v_max_i32_e32 v3, v2, v3                                   // 000000001B88: 1A060702
	v_mul_hi_u32 v1, v3, v1                                    // 000000001B8C: D2860001 00020303
	v_cmp_gt_i32_e32 vcc, s1, v0                               // 000000001B94: 7D880001
	v_ashrrev_i32_e32 v2, 31, v2                               // 000000001B98: 2204049F
	s_load_dwordx2 s[12:13], s[4:5], 0x0                       // 000000001B9C: C0060302 00000000
	s_load_dwordx2 s[14:15], s[4:5], 0x10                      // 000000001BA4: C0060382 00000010
	s_load_dwordx2 s[6:7], s[4:5], 0x20                        // 000000001BAC: C0060182 00000020
	s_load_dwordx2 s[2:3], s[4:5], 0x30                        // 000000001BB4: C0060082 00000030
	v_mul_lo_u32 v1, v1, s8                                    // 000000001BBC: D2850001 00001101
	v_sub_u32_e32 v1, v3, v1                                   // 000000001BC4: 6A020303
	v_subrev_u32_e32 v3, s8, v1                                // 000000001BC8: 6C060208
	v_cmp_le_u32_e64 s[0:1], s8, v1                            // 000000001BCC: D0CB0000 00020208
	v_cndmask_b32_e64 v1, v1, v3, s[0:1]                       // 000000001BD4: D1000001 00020701
	v_subrev_u32_e32 v3, s8, v1                                // 000000001BDC: 6C060208
	v_cmp_le_u32_e64 s[0:1], s8, v1                            // 000000001BE0: D0CB0000 00020208
	v_cndmask_b32_e64 v1, v1, v3, s[0:1]                       // 000000001BE8: D1000001 00020701
	v_xor_b32_e32 v1, v1, v2                                   // 000000001BF0: 2A020501
	v_sub_u32_e32 v3, v1, v2                                   // 000000001BF4: 6A060501
	v_and_b32_e32 v2, 60, v6                                   // 000000001BF8: 26040CBC
	v_lshl_or_b32 v1, v0, 7, v2                                // 000000001BFC: D2000001 04090F00
	v_mad_u64_u32 v[4:5], s[0:1], v3, s9, v[2:3]               // 000000001C04: D1E80004 04081303
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001C0C: 2204029F
	v_lshlrev_b64 v[2:3], 1, v[1:2]                            // 000000001C10: D28F0002 00020281
	s_waitcnt lgkmcnt(0)                                       // 000000001C18: BF8CC07F
	v_mov_b32_e32 v5, s15                                      // 000000001C1C: 7E0A020F
	v_add_co_u32_e64 v2, s[0:1], s14, v2                       // 000000001C20: D1190002 0002040E
	v_addc_co_u32_e64 v3, s[0:1], v5, v3, s[0:1]               // 000000001C28: D11C0003 00020705
	s_andn2_b64 s[0:1], s[10:11], exec                         // 000000001C30: 89807E0A
	s_and_b64 s[8:9], vcc, exec                                // 000000001C34: 86887E6A
	s_or_b64 s[10:11], s[0:1], s[8:9]                          // 000000001C38: 878A0800
	v_mov_b32_e32 v6, 0                                        // 000000001C3C: 7E0C0280
	s_mov_b64 s[0:1], 0                                        // 000000001C40: BE800180
	v_mov_b32_e32 v8, 0                                        // 000000001C44: 7E100280
	v_mov_b32_e32 v9, 0                                        // 000000001C48: 7E120280
	s_and_saveexec_b64 s[8:9], s[10:11]                        // 000000001C4C: BE88200A
	s_cbranch_execz 3                                          // 000000001C50: BF880003 <c662_qkrope_bwd+0x360>
	global_load_dwordx2 v[8:9], v[2:3], off                    // 000000001C54: DC548000 087F0002
	s_mov_b64 s[0:1], exec                                     // 000000001C5C: BE80017E
	s_or_b64 exec, exec, s[8:9]                                // 000000001C60: 87FE087E
	v_mov_b32_e32 v7, 0                                        // 000000001C64: 7E0E0280
	s_and_saveexec_b64 s[8:9], s[0:1]                          // 000000001C68: BE882000
	s_cbranch_execz 2                                          // 000000001C6C: BF880002 <c662_qkrope_bwd+0x378>
	global_load_dwordx2 v[6:7], v[2:3], off offset:128         // 000000001C70: DC548080 067F0002
	s_or_b64 exec, exec, s[8:9]                                // 000000001C78: 87FE087E
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001C7C: 2204029F
	v_lshlrev_b64 v[2:3], 1, v[1:2]                            // 000000001C80: D28F0002 00020281
	s_waitcnt lgkmcnt(0)                                       // 000000001C88: BF8CC07F
	v_mov_b32_e32 v1, s13                                      // 000000001C8C: 7E02020D
	v_add_co_u32_e32 v14, vcc, s12, v2                         // 000000001C90: 321C040C
	v_addc_co_u32_e32 v15, vcc, v1, v3, vcc                    // 000000001C94: 381E0701
	v_mov_b32_e32 v10, 0                                       // 000000001C98: 7E140280
	v_mov_b32_e32 v12, 0                                       // 000000001C9C: 7E180280
	v_mov_b32_e32 v13, 0                                       // 000000001CA0: 7E1A0280
	s_and_saveexec_b64 s[8:9], s[0:1]                          // 000000001CA4: BE882000
	s_cbranch_execz 2                                          // 000000001CA8: BF880002 <c662_qkrope_bwd+0x3b4>
	global_load_dwordx2 v[12:13], v[14:15], off                // 000000001CAC: DC548000 0C7F000E
	s_or_b64 exec, exec, s[8:9]                                // 000000001CB4: 87FE087E
	v_mov_b32_e32 v11, 0                                       // 000000001CB8: 7E160280
	s_and_saveexec_b64 s[8:9], s[0:1]                          // 000000001CBC: BE882000
	s_cbranch_execz 2                                          // 000000001CC0: BF880002 <c662_qkrope_bwd+0x3cc>
	global_load_dwordx2 v[10:11], v[14:15], off offset:128     // 000000001CC4: DC548080 0A7F000E
	s_or_b64 exec, exec, s[8:9]                                // 000000001CCC: 87FE087E
	v_mov_b32_e32 v14, 0                                       // 000000001CD0: 7E1C0280
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000001CD4: 220A089F
	v_mov_b32_e32 v16, 0                                       // 000000001CD8: 7E200280
	v_mov_b32_e32 v17, 0                                       // 000000001CDC: 7E220280
	s_and_saveexec_b64 s[8:9], s[0:1]                          // 000000001CE0: BE882000
	s_cbranch_execz 10                                         // 000000001CE4: BF88000A <c662_qkrope_bwd+0x410>
	s_load_dwordx2 s[10:11], s[4:5], 0x40                      // 000000001CE8: C0060282 00000040
	v_lshlrev_b64 v[15:16], 1, v[4:5]                          // 000000001CF0: D28F000F 00020881
	s_waitcnt lgkmcnt(0)                                       // 000000001CF8: BF8CC07F
	v_mov_b32_e32 v1, s11                                      // 000000001CFC: 7E02020B
	v_add_co_u32_e32 v15, vcc, s10, v15                        // 000000001D00: 321E1E0A
	v_addc_co_u32_e32 v16, vcc, v1, v16, vcc                   // 000000001D04: 38202101
	global_load_dwordx2 v[16:17], v[15:16], off                // 000000001D08: DC548000 107F000F
	s_or_b64 exec, exec, s[8:9]                                // 000000001D10: 87FE087E
	v_mov_b32_e32 v15, 0                                       // 000000001D14: 7E1E0280
	s_and_saveexec_b64 s[8:9], s[0:1]                          // 000000001D18: BE882000
	s_cbranch_execz 10                                         // 000000001D1C: BF88000A <c662_qkrope_bwd+0x448>
	s_load_dwordx2 s[10:11], s[4:5], 0x48                      // 000000001D20: C0060282 00000048
	v_lshlrev_b64 v[4:5], 1, v[4:5]                            // 000000001D28: D28F0004 00020881
	s_waitcnt lgkmcnt(0)                                       // 000000001D30: BF8CC07F
	v_mov_b32_e32 v1, s11                                      // 000000001D34: 7E02020B
	v_add_co_u32_e32 v4, vcc, s10, v4                          // 000000001D38: 3208080A
	v_addc_co_u32_e32 v5, vcc, v1, v5, vcc                     // 000000001D3C: 380A0B01
	global_load_dwordx2 v[14:15], v[4:5], off                  // 000000001D40: DC548000 0E7F0004
	s_or_b64 exec, exec, s[8:9]                                // 000000001D48: 87FE087E
	s_load_dword s8, s[4:5], 0x6c                              // 000000001D4C: C0020202 0000006C
	v_mov_b32_e32 v1, 0                                        // 000000001D54: 7E020280
	s_and_saveexec_b64 s[4:5], s[0:1]                          // 000000001D58: BE842000
	s_cbranch_execz 8                                          // 000000001D5C: BF880008 <c662_qkrope_bwd+0x480>
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000001D60: 2202009F
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 000000001D64: D28F0000 00020082
	v_mov_b32_e32 v4, s7                                       // 000000001D6C: 7E080207
	v_add_co_u32_e32 v0, vcc, s6, v0                           // 000000001D70: 32000006
	v_addc_co_u32_e32 v1, vcc, v4, v1, vcc                     // 000000001D74: 38020304
	global_load_dword v1, v[0:1], off                          // 000000001D78: DC508000 017F0000
	s_or_b64 exec, exec, s[4:5]                                // 000000001D80: 87FE047E
	s_waitcnt vmcnt(0)                                         // 000000001D84: BF8C0F70
	v_and_b32_e32 v20, 0xffff0000, v9                          // 000000001D88: 262812FF FFFF0000
	v_lshlrev_b32_e32 v21, 16, v9                              // 000000001D90: 242A1290
	v_and_b32_e32 v22, 0xffff0000, v6                          // 000000001D94: 262C0CFF FFFF0000
	v_and_b32_e32 v9, 0xffff0000, v14                          // 000000001D9C: 26121CFF FFFF0000
	v_and_b32_e32 v18, 0xffff0000, v8                          // 000000001DA4: 262410FF FFFF0000
	v_lshlrev_b32_e32 v19, 16, v8                              // 000000001DAC: 24261090
	v_lshlrev_b32_e32 v23, 16, v6                              // 000000001DB0: 242E0C90
	v_and_b32_e32 v24, 0xffff0000, v7                          // 000000001DB4: 26300EFF FFFF0000
	v_lshlrev_b32_e32 v25, 16, v7                              // 000000001DBC: 24320E90
	v_and_b32_e32 v7, 0xffff0000, v16                          // 000000001DC0: 260E20FF FFFF0000
	v_lshlrev_b32_e32 v14, 16, v14                             // 000000001DC8: 241C1C90
	v_and_b32_e32 v5, 0xffff0000, v15                          // 000000001DCC: 260A1EFF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v15                              // 000000001DD4: 240C1E90
	v_mul_f32_e32 v15, v22, v9                                 // 000000001DD8: 0A1E1316
	v_lshlrev_b32_e32 v8, 16, v16                              // 000000001DDC: 24102090
	v_and_b32_e32 v0, 0xffff0000, v17                          // 000000001DE0: 260022FF FFFF0000
	v_fmac_f32_e32 v15, v18, v7                                // 000000001DE8: 761E0F12
	v_mul_f32_e32 v16, v18, v9                                 // 000000001DEC: 0A201312
	v_mul_f32_e32 v18, v19, v14                                // 000000001DF0: 0A241D13
	v_and_b32_e32 v27, 0xffff0000, v10                         // 000000001DF4: 263614FF FFFF0000
	v_lshlrev_b32_e32 v28, 16, v10                             // 000000001DFC: 24381490
	v_mul_f32_e32 v10, v20, v5                                 // 000000001E00: 0A140B14
	v_lshlrev_b32_e32 v4, 16, v17                              // 000000001E04: 24082290
	v_fma_f32 v16, v22, v7, -v16                               // 000000001E08: D1CB0010 84420F16
	v_mul_f32_e32 v17, v23, v14                                // 000000001E10: 0A221D17
	v_fma_f32 v22, v23, v8, -v18                               // 000000001E14: D1CB0016 844A1117
	v_and_b32_e32 v18, 0xffff0000, v12                         // 000000001E1C: 262418FF FFFF0000
	v_mul_f32_e32 v31, v24, v5                                 // 000000001E24: 0A3E0B18
	v_fma_f32 v24, v24, v0, -v10                               // 000000001E28: D1CB0018 842A0118
	v_mul_f32_e32 v10, v21, v6                                 // 000000001E30: 0A140D15
	v_fmac_f32_e32 v17, v19, v8                                // 000000001E34: 76221113
	v_lshlrev_b32_e32 v19, 16, v12                             // 000000001E38: 24261890
	v_and_b32_e32 v29, 0xffff0000, v11                         // 000000001E3C: 263A16FF FFFF0000
	v_lshlrev_b32_e32 v30, 16, v11                             // 000000001E44: 243C1690
	v_fmac_f32_e32 v31, v20, v0                                // 000000001E48: 763E0114
	v_mul_f32_e32 v20, v25, v6                                 // 000000001E4C: 0A280D19
	v_fma_f32 v25, v25, v4, -v10                               // 000000001E50: D1CB0019 842A0919
	s_waitcnt lgkmcnt(0)                                       // 000000001E58: BF8CC07F
	v_mul_f32_e32 v12, s8, v18                                 // 000000001E5C: 0A182408
	v_mul_f32_e32 v10, s8, v27                                 // 000000001E60: 0A143608
	v_mul_f32_e32 v11, s8, v28                                 // 000000001E64: 0A163808
	v_mul_f32_e32 v18, v15, v1                                 // 000000001E68: 0A24030F
	v_mul_f32_e32 v15, v16, v1                                 // 000000001E6C: 0A1E0310
	v_mul_f32_e32 v16, v22, v1                                 // 000000001E70: 0A200316
	v_and_b32_e32 v23, 0xffff0000, v13                         // 000000001E74: 262E1AFF FFFF0000
	v_lshlrev_b32_e32 v26, 16, v13                             // 000000001E7C: 24341A90
	v_fmac_f32_e32 v20, v21, v4                                // 000000001E80: 76280915
	v_mul_f32_e32 v13, s8, v19                                 // 000000001E84: 0A1A2608
	v_mul_f32_e32 v19, v17, v1                                 // 000000001E88: 0A260311
	v_mul_f32_e32 v17, v11, v16                                // 000000001E8C: 0A22210B
	v_mul_f32_e32 v21, v10, v15                                // 000000001E90: 0A2A1F0A
	v_fmac_f32_e32 v17, v13, v19                               // 000000001E94: 7622270D
	v_fmac_f32_e32 v21, v12, v18                               // 000000001E98: 762A250C
	v_add_f32_e32 v27, v17, v21                                // 000000001E9C: 02362B11
	v_mul_f32_e32 v17, s8, v30                                 // 000000001EA0: 0A223C08
	v_mul_f32_e32 v22, v20, v1                                 // 000000001EA4: 0A2C0314
	v_mul_f32_e32 v20, v25, v1                                 // 000000001EA8: 0A280319
	v_mul_f32_e32 v21, s8, v26                                 // 000000001EAC: 0A2A3408
	v_mul_f32_e32 v25, v17, v20                                // 000000001EB0: 0A322911
	v_fmac_f32_e32 v25, v21, v22                               // 000000001EB4: 76322D15
	v_add_f32_e32 v27, v25, v27                                // 000000001EB8: 02363719
	v_mul_f32_e32 v25, s8, v23                                 // 000000001EBC: 0A322E08
	v_mul_f32_e32 v23, s8, v29                                 // 000000001EC0: 0A2E3A08
	v_mul_f32_e32 v24, v24, v1                                 // 000000001EC4: 0A300318
	v_mul_f32_e32 v26, v31, v1                                 // 000000001EC8: 0A34031F
	v_mul_f32_e32 v28, v23, v24                                // 000000001ECC: 0A383117
	v_fmac_f32_e32 v28, v25, v26                               // 000000001ED0: 76383519
	v_add_f32_e32 v27, v28, v27                                // 000000001ED4: 0236371C
	ds_swizzle_b32 v28, v27 offset:swizzle(SWAP,8)             // 000000001ED8: D87A201F 1C00001B
	s_waitcnt lgkmcnt(0)                                       // 000000001EE0: BF8CC07F
	v_add_f32_e32 v27, v27, v28                                // 000000001EE4: 0236391B
	ds_swizzle_b32 v28, v27 offset:swizzle(SWAP,4)             // 000000001EE8: D87A101F 1C00001B
	s_waitcnt lgkmcnt(0)                                       // 000000001EF0: BF8CC07F
	v_add_f32_e32 v27, v27, v28                                // 000000001EF4: 0236391B
	ds_swizzle_b32 v28, v27 offset:swizzle(SWAP,2)             // 000000001EF8: D87A081F 1C00001B
	s_waitcnt lgkmcnt(0)                                       // 000000001F00: BF8CC07F
	v_add_f32_e32 v27, v27, v28                                // 000000001F04: 0236391B
	ds_swizzle_b32 v28, v27 offset:swizzle(SWAP,1)             // 000000001F08: D87A041F 1C00001B
	s_and_saveexec_b64 s[4:5], s[0:1]                          // 000000001F10: BE842000
	s_cbranch_execz 124                                        // 000000001F14: BF88007C <c662_qkrope_bwd+0x808>
	s_waitcnt lgkmcnt(0)                                       // 000000001F18: BF8CC07F
	v_add_f32_e32 v27, v27, v28                                // 000000001F1C: 0236391B
	v_mul_f32_e32 v27, 0x3c000000, v27                         // 000000001F20: 0A3636FF 3C000000
	v_fma_f32 v10, -v15, v27, v10                              // 000000001F28: D1CB000A 242A370F
	v_fma_f32 v12, -v18, v27, v12                              // 000000001F30: D1CB000C 24323712
	v_mul_f32_e32 v10, v1, v10                                 // 000000001F38: 0A141501
	v_mul_f32_e32 v12, v1, v12                                 // 000000001F3C: 0A181901
	v_fma_f32 v11, -v16, v27, v11                              // 000000001F40: D1CB000B 242E3710
	v_mul_f32_e32 v15, v10, v7                                 // 000000001F48: 0A1E0F0A
	v_fma_f32 v13, -v19, v27, v13                              // 000000001F4C: D1CB000D 24363713
	v_mul_f32_e32 v11, v1, v11                                 // 000000001F54: 0A161701
	v_fmac_f32_e32 v15, v12, v9                                // 000000001F58: 761E130C
	v_mul_f32_e32 v9, v10, v9                                  // 000000001F5C: 0A12130A
	v_mul_f32_e32 v13, v1, v13                                 // 000000001F60: 0A1A1B01
	v_mul_f32_e32 v16, v11, v8                                 // 000000001F64: 0A20110B
	v_mul_f32_e32 v11, v11, v14                                // 000000001F68: 0A161D0B
	v_fma_f32 v7, v12, v7, -v9                                 // 000000001F6C: D1CB0007 84260F0C
	v_fma_f32 v12, -v20, v27, v17                              // 000000001F74: D1CB000C 24463714
	v_fma_f32 v8, v13, v8, -v11                                // 000000001F7C: D1CB0008 842E110D
	v_fma_f32 v9, -v26, v27, v25                               // 000000001F84: D1CB0009 2466371A
	v_fma_f32 v10, -v22, v27, v21                              // 000000001F8C: D1CB000A 24563716
	v_fma_f32 v11, -v24, v27, v23                              // 000000001F94: D1CB000B 245E3718
	v_mul_f32_e32 v12, v1, v12                                 // 000000001F9C: 0A181901
	v_fmac_f32_e32 v16, v13, v14                               // 000000001FA0: 76201D0D
	v_mul_f32_e32 v10, v1, v10                                 // 000000001FA4: 0A141501
	v_mul_f32_e32 v9, v1, v9                                   // 000000001FA8: 0A121301
	v_mul_f32_e32 v1, v1, v11                                  // 000000001FAC: 0A021701
	v_mul_f32_e32 v13, v12, v4                                 // 000000001FB0: 0A1A090C
	v_mul_f32_e32 v11, v1, v0                                  // 000000001FB4: 0A160101
	v_fmac_f32_e32 v13, v10, v6                                // 000000001FB8: 761A0D0A
	v_mul_f32_e32 v6, v12, v6                                  // 000000001FBC: 0A0C0D0C
	v_mul_f32_e32 v1, v1, v5                                   // 000000001FC0: 0A020B01
	v_mov_b32_e32 v29, s3                                      // 000000001FC4: 7E3A0203
	v_add_co_u32_e32 v2, vcc, s2, v2                           // 000000001FC8: 32040402
	v_fma_f32 v0, v9, v0, -v1                                  // 000000001FCC: D1CB0000 84060109
	v_fma_f32 v1, v10, v4, -v6                                 // 000000001FD4: D1CB0001 841A090A
	v_addc_co_u32_e32 v3, vcc, v29, v3, vcc                    // 000000001FDC: 3806071D
	v_bfe_u32 v4, v1, 16, 1                                    // 000000001FE0: D1C80004 02052101
	s_movk_i32 s0, 0x7fff                                      // 000000001FE8: B0007FFF
	v_fmac_f32_e32 v11, v9, v5                                 // 000000001FEC: 76160B09
	v_bfe_u32 v5, v0, 16, 1                                    // 000000001FF0: D1C80005 02052100
	v_add3_u32 v4, v1, v4, s0                                  // 000000001FF8: D1FF0004 00020901
	v_cmp_o_f32_e32 vcc, v1, v1                                // 000000002000: 7C8E0301
	v_mov_b32_e32 v10, 0x7fff                                  // 000000002004: 7E1402FF 00007FFF
	v_bfe_u32 v6, v8, 16, 1                                    // 00000000200C: D1C80006 02052108
	v_add3_u32 v5, v0, v5, s0                                  // 000000002014: D1FF0005 00020B00
	v_cndmask_b32_sdwa v1, v10, v4, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000201C: 000208F9 0506060A
	v_cmp_o_f32_e32 vcc, v0, v0                                // 000000002024: 7C8E0100
	v_bfe_u32 v9, v7, 16, 1                                    // 000000002028: D1C80009 02052107
	v_add3_u32 v6, v8, v6, s0                                  // 000000002030: D1FF0006 00020D08
	v_cndmask_b32_sdwa v0, v10, v5, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002038: 00000AF9 0506060A
	s_mov_b32 s1, 0x5040100                                    // 000000002040: BE8100FF 05040100
	v_cmp_o_f32_e32 vcc, v8, v8                                // 000000002048: 7C8E1108
	v_add3_u32 v9, v7, v9, s0                                  // 00000000204C: D1FF0009 00021307
	v_perm_b32 v1, v0, v1, s1                                  // 000000002054: D1ED0001 00060300
	v_cndmask_b32_sdwa v0, v10, v6, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000205C: 00000CF9 0506060A
	v_cmp_o_f32_e32 vcc, v7, v7                                // 000000002064: 7C8E0F07
	v_cndmask_b32_sdwa v4, v10, v9, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002068: 000812F9 0506060A
	v_perm_b32 v0, v4, v0, s1                                  // 000000002070: D1ED0000 00060104
	global_store_dwordx2 v[2:3], v[0:1], off                   // 000000002078: DC748000 007F0002
	v_bfe_u32 v0, v16, 16, 1                                   // 000000002080: D1C80000 02052110
	v_add3_u32 v0, v16, v0, s0                                 // 000000002088: D1FF0000 00020110
	v_cmp_o_f32_e32 vcc, v16, v16                              // 000000002090: 7C8E2110
	v_bfe_u32 v1, v15, 16, 1                                   // 000000002094: D1C80001 0205210F
	v_cndmask_b32_sdwa v0, v10, v0, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000209C: 000000F9 0506060A
	v_add3_u32 v1, v15, v1, s0                                 // 0000000020A4: D1FF0001 0002030F
	v_cmp_o_f32_e32 vcc, v15, v15                              // 0000000020AC: 7C8E1F0F
	v_cndmask_b32_sdwa v4, v10, v1, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000020B0: 000802F9 0506060A
	v_bfe_u32 v1, v13, 16, 1                                   // 0000000020B8: D1C80001 0205210D
	v_add3_u32 v1, v13, v1, s0                                 // 0000000020C0: D1FF0001 0002030D
	v_cmp_o_f32_e32 vcc, v13, v13                              // 0000000020C8: 7C8E1B0D
	v_bfe_u32 v5, v11, 16, 1                                   // 0000000020CC: D1C80005 0205210B
	v_cndmask_b32_sdwa v1, v10, v1, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000020D4: 000202F9 0506060A
	v_add3_u32 v5, v11, v5, s0                                 // 0000000020DC: D1FF0005 00020B0B
	v_cmp_o_f32_e32 vcc, v11, v11                              // 0000000020E4: 7C8E170B
	v_cndmask_b32_sdwa v5, v10, v5, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000020E8: 000A0AF9 0506060A
	v_perm_b32 v1, v5, v1, s1                                  // 0000000020F0: D1ED0001 00060305
	v_perm_b32 v0, v4, v0, s1                                  // 0000000020F8: D1ED0000 00060104
	global_store_dwordx2 v[2:3], v[0:1], off offset:128        // 000000002100: DC748080 007F0002
	s_endpgm                                                   // 000000002108: BF810000
