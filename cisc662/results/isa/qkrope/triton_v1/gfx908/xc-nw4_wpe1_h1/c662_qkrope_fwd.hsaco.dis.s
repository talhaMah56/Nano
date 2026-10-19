
/u/tmahmood1/project/nanochat/cisc662/results/isa/qkrope/triton_v1/gfx908/xc-nw4_wpe1_h1/c662_qkrope_fwd.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001900 <c662_qkrope_fwd>:
	s_load_dword s0, s[4:5], 0x48                              // 000000001900: C0020002 00000048
	s_load_dwordx2 s[10:11], s[4:5], 0x54                      // 000000001908: C0060282 00000054
	v_lshrrev_b32_e32 v9, 6, v0                                // 000000001910: 20120086
	v_lshrrev_b32_e32 v11, 4, v0                               // 000000001914: 20160084
	v_and_b32_e32 v10, 15, v0                                  // 000000001918: 2614008F
	s_waitcnt lgkmcnt(0)                                       // 00000000191C: BF8CC07F
	s_cmp_ge_i32 s6, s0                                        // 000000001920: BF030006
	s_cbranch_scc0 103                                         // 000000001924: BF840067 <c662_qkrope_fwd+0x1c4>
	s_load_dword s1, s[4:5], 0x50                              // 000000001928: C0020042 00000050
	s_load_dword s7, s[4:5], 0x44                              // 000000001930: C00201C2 00000044
	s_abs_i32 s12, s10                                         // 000000001938: BE8C300A
	v_cvt_f32_u32_e32 v2, s12                                  // 00000000193C: 7E040C0C
	s_sub_i32 s0, s6, s0                                       // 000000001940: 81800006
	s_waitcnt lgkmcnt(0)                                       // 000000001944: BF8CC07F
	s_abs_i32 s8, s1                                           // 000000001948: BE883001
	v_cvt_f32_u32_e32 v1, s8                                   // 00000000194C: 7E020C08
	s_sub_i32 s3, 0, s8                                        // 000000001950: 81830880
	v_rcp_iflag_f32_e32 v2, v2                                 // 000000001954: 7E044702
	s_lshl_b32 s9, s0, 4                                       // 000000001958: 8E098400
	v_rcp_iflag_f32_e32 v1, v1                                 // 00000000195C: 7E024701
	s_bfe_i32 s0, s0, 0x1001b                                  // 000000001960: 9300FF00 0001001B
	v_mul_f32_e32 v2, 0x4f7ffffe, v2                           // 000000001968: 0A0404FF 4F7FFFFE
	v_cvt_u32_f32_e32 v2, v2                                   // 000000001970: 7E040F02
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001974: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v1, v1                                   // 00000000197C: 7E020F01
	v_or_b32_e32 v4, s9, v11                                   // 000000001980: 28081609
	v_add_u32_e32 v5, s0, v4                                   // 000000001984: 680A0800
	s_sub_i32 s2, 0, s12                                       // 000000001988: 81820C80
	v_mul_lo_u32 v3, s3, v1                                    // 00000000198C: D2850003 00020203
	v_xor_b32_e32 v5, s0, v5                                   // 000000001994: 2A0A0A00
	v_mul_lo_u32 v6, s2, v2                                    // 000000001998: D2850006 00020402
	s_ashr_i32 s1, s1, 31                                      // 0000000019A0: 90019F01
	v_mul_hi_u32 v3, v1, v3                                    // 0000000019A4: D2860003 00020701
	s_xor_b32 s0, s0, s1                                       // 0000000019AC: 88000100
	s_load_dwordx2 s[2:3], s[4:5], 0x8                         // 0000000019B0: C0060082 00000008
	v_lshrrev_b32_e32 v17, 6, v0                               // 0000000019B8: 20220086
	v_add_u32_e32 v1, v1, v3                                   // 0000000019BC: 68020701
	v_mul_hi_u32 v1, v5, v1                                    // 0000000019C0: D2860001 00020305
	v_mul_hi_u32 v3, v2, v6                                    // 0000000019C8: D2860003 00020D02
	v_mul_lo_u32 v6, v1, s8                                    // 0000000019D0: D2850006 00001101
	v_add_u32_e32 v2, v2, v3                                   // 0000000019D8: 68040702
	v_add_u32_e32 v3, 1, v1                                    // 0000000019DC: 68060281
	v_sub_u32_e32 v5, v5, v6                                   // 0000000019E0: 6A0A0D05
	v_cmp_le_u32_e32 vcc, s8, v5                               // 0000000019E4: 7D960A08
	v_cndmask_b32_e32 v1, v1, v3, vcc                          // 0000000019E8: 00020701
	v_subrev_u32_e32 v3, s8, v5                                // 0000000019EC: 6C060A08
	v_cndmask_b32_e32 v3, v5, v3, vcc                          // 0000000019F0: 00060705
	v_add_u32_e32 v5, 1, v1                                    // 0000000019F4: 680A0281
	v_cmp_le_u32_e32 vcc, s8, v3                               // 0000000019F8: 7D960608
	v_cndmask_b32_e32 v1, v1, v5, vcc                          // 0000000019FC: 00020B01
	v_xor_b32_e32 v1, s0, v1                                   // 000000001A00: 2A020200
	v_subrev_u32_e32 v3, s0, v1                                // 000000001A04: 6C060200
	v_sub_u32_e32 v1, 0, v3                                    // 000000001A08: 6A020680
	v_max_i32_e32 v5, v3, v1                                   // 000000001A0C: 1A0A0303
	v_mul_hi_u32 v6, v5, v2                                    // 000000001A10: D2860006 00020505
	v_ashrrev_i32_e32 v3, 31, v3                               // 000000001A18: 2206069F
	v_cmp_gt_i32_e32 vcc, s7, v4                               // 000000001A1C: 7D880807
	v_or_b32_e32 v1, s9, v10                                   // 000000001A20: 28021409
	v_mul_lo_u32 v6, v6, s12                                   // 000000001A24: D2850006 00001906
	v_and_b32_e32 v2, 0xf0, v0                                 // 000000001A2C: 260400FF 000000F0
	v_cmp_gt_i32_e64 s[8:9], s7, v1                            // 000000001A34: D0C40008 00020207
	s_and_b64 s[14:15], vcc, exec                              // 000000001A3C: 868E7E6A
	v_sub_u32_e32 v5, v5, v6                                   // 000000001A40: 6A0A0D05
	v_subrev_u32_e32 v6, s12, v5                               // 000000001A44: 6C0C0A0C
	v_cmp_le_u32_e64 s[0:1], s12, v5                           // 000000001A48: D0CB0000 00020A0C
	v_cndmask_b32_e64 v5, v5, v6, s[0:1]                       // 000000001A50: D1000005 00020D05
	v_subrev_u32_e32 v6, s12, v5                               // 000000001A58: 6C0C0A0C
	v_cmp_le_u32_e64 s[0:1], s12, v5                           // 000000001A5C: D0CB0000 00020A0C
	v_cndmask_b32_e64 v5, v5, v6, s[0:1]                       // 000000001A64: D1000005 00020D05
	v_xor_b32_e32 v5, v5, v3                                   // 000000001A6C: 2A0A0705
	v_sub_u32_e32 v6, v5, v3                                   // 000000001A70: 6A0C0705
	v_lshlrev_b32_e32 v5, 2, v10                               // 000000001A74: 240A1482
	v_lshl_or_b32 v3, v4, 7, v5                                // 000000001A78: D2000003 04150F04
	v_mad_u64_u32 v[6:7], s[0:1], v6, s11, v[5:6]              // 000000001A80: D1E80006 04141706
	v_ashrrev_i32_e32 v4, 31, v3                               // 000000001A88: 2208069F
	v_lshlrev_b64 v[7:8], 1, v[3:4]                            // 000000001A8C: D28F0007 00020681
	s_waitcnt lgkmcnt(0)                                       // 000000001A94: BF8CC07F
	v_mov_b32_e32 v4, s3                                       // 000000001A98: 7E080203
	v_add_co_u32_e64 v7, s[0:1], s2, v7                        // 000000001A9C: D1190007 00020E02
	v_addc_co_u32_e64 v8, s[0:1], v4, v8, s[0:1]               // 000000001AA4: D11C0008 00021104
	s_cbranch_execz 6                                          // 000000001AAC: BF880006 <c662_qkrope_fwd+0x1c8>
	s_load_dwordx2 s[6:7], s[4:5], 0x18                        // 000000001AB0: C0060182 00000018
	s_load_dwordx2 s[12:13], s[4:5], 0x28                      // 000000001AB8: C0060302 00000028
	s_branch 106                                               // 000000001AC0: BF82006A <c662_qkrope_fwd+0x36c>
	s_mov_b64 s[14:15], 0                                      // 000000001AC4: BE8E0180
	s_load_dword s0, s[4:5], 0x4c                              // 000000001AC8: C0020002 0000004C
	s_abs_i32 s10, s10                                         // 000000001AD0: BE8A300A
	v_cvt_f32_u32_e32 v2, s10                                  // 000000001AD4: 7E040C0A
	s_lshl_b32 s2, s6, 4                                       // 000000001AD8: 8E028406
	s_bfe_i32 s3, s6, 0x1001b                                  // 000000001ADC: 9303FF06 0001001B
	s_waitcnt lgkmcnt(0)                                       // 000000001AE4: BF8CC07F
	s_abs_i32 s1, s0                                           // 000000001AE8: BE813000
	v_cvt_f32_u32_e32 v1, s1                                   // 000000001AEC: 7E020C01
	s_sub_i32 s6, 0, s1                                        // 000000001AF0: 81860180
	v_rcp_iflag_f32_e32 v2, v2                                 // 000000001AF4: 7E044702
	v_or_b32_e32 v3, s2, v11                                   // 000000001AF8: 28061602
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000001AFC: 7E024701
	v_add_u32_e32 v5, s3, v3                                   // 000000001B00: 680A0603
	v_mul_f32_e32 v2, 0x4f7ffffe, v2                           // 000000001B04: 0A0404FF 4F7FFFFE
	v_cvt_u32_f32_e32 v2, v2                                   // 000000001B0C: 7E040F02
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001B10: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v1, v1                                   // 000000001B18: 7E020F01
	v_xor_b32_e32 v5, s3, v5                                   // 000000001B1C: 2A0A0A03
	s_ashr_i32 s0, s0, 31                                      // 000000001B20: 90009F00
	s_xor_b32 s0, s3, s0                                       // 000000001B24: 88000003
	v_mul_lo_u32 v4, s6, v1                                    // 000000001B28: D2850004 00020206
	s_sub_i32 s6, 0, s10                                       // 000000001B30: 81860A80
	v_mul_lo_u32 v6, s6, v2                                    // 000000001B34: D2850006 00020406
	v_mov_b32_e32 v17, v9                                      // 000000001B3C: 7E220309
	v_mul_hi_u32 v4, v1, v4                                    // 000000001B40: D2860004 00020901
	v_add_u32_e32 v1, v1, v4                                   // 000000001B48: 68020901
	v_mul_hi_u32 v1, v5, v1                                    // 000000001B4C: D2860001 00020305
	v_mul_hi_u32 v4, v2, v6                                    // 000000001B54: D2860004 00020D02
	v_mul_lo_u32 v6, v1, s1                                    // 000000001B5C: D2850006 00000301
	v_add_u32_e32 v2, v2, v4                                   // 000000001B64: 68040902
	v_add_u32_e32 v4, 1, v1                                    // 000000001B68: 68080281
	v_sub_u32_e32 v5, v5, v6                                   // 000000001B6C: 6A0A0D05
	v_cmp_le_u32_e32 vcc, s1, v5                               // 000000001B70: 7D960A01
	v_cndmask_b32_e32 v1, v1, v4, vcc                          // 000000001B74: 00020901
	v_subrev_u32_e32 v4, s1, v5                                // 000000001B78: 6C080A01
	v_cndmask_b32_e32 v4, v5, v4, vcc                          // 000000001B7C: 00080905
	v_add_u32_e32 v5, 1, v1                                    // 000000001B80: 680A0281
	v_cmp_le_u32_e32 vcc, s1, v4                               // 000000001B84: 7D960801
	v_cndmask_b32_e32 v1, v1, v5, vcc                          // 000000001B88: 00020B01
	v_xor_b32_e32 v1, s0, v1                                   // 000000001B8C: 2A020200
	v_subrev_u32_e32 v4, s0, v1                                // 000000001B90: 6C080200
	v_sub_u32_e32 v1, 0, v4                                    // 000000001B94: 6A020880
	v_max_i32_e32 v5, v4, v1                                   // 000000001B98: 1A0A0304
	v_mul_hi_u32 v6, v5, v2                                    // 000000001B9C: D2860006 00020505
	v_or_b32_e32 v1, s2, v10                                   // 000000001BA4: 28021402
	s_load_dwordx2 s[16:17], s[4:5], 0x0                       // 000000001BA8: C0060402 00000000
	s_load_dwordx2 s[6:7], s[4:5], 0x10                        // 000000001BB0: C0060182 00000010
	s_load_dword s0, s[4:5], 0x40                              // 000000001BB8: C0020002 00000040
	s_load_dwordx2 s[12:13], s[4:5], 0x20                      // 000000001BC0: C0060302 00000020
	v_ashrrev_i32_e32 v4, 31, v4                               // 000000001BC8: 2208089F
	v_mul_lo_u32 v6, v6, s10                                   // 000000001BCC: D2850006 00001506
	v_and_b32_e32 v2, 0xf0, v0                                 // 000000001BD4: 260400FF 000000F0
	s_waitcnt lgkmcnt(0)                                       // 000000001BDC: BF8CC07F
	v_cmp_gt_i32_e32 vcc, s0, v3                               // 000000001BE0: 7D880600
	v_cmp_gt_i32_e64 s[0:1], s0, v1                            // 000000001BE4: D0C40000 00020200
	v_sub_u32_e32 v5, v5, v6                                   // 000000001BEC: 6A0A0D05
	v_subrev_u32_e32 v6, s10, v5                               // 000000001BF0: 6C0C0A0A
	v_cmp_le_u32_e64 s[2:3], s10, v5                           // 000000001BF4: D0CB0002 00020A0A
	v_cndmask_b32_e64 v5, v5, v6, s[2:3]                       // 000000001BFC: D1000005 000A0D05
	v_subrev_u32_e32 v6, s10, v5                               // 000000001C04: 6C0C0A0A
	v_cmp_le_u32_e64 s[2:3], s10, v5                           // 000000001C08: D0CB0002 00020A0A
	v_cndmask_b32_e64 v5, v5, v6, s[2:3]                       // 000000001C10: D1000005 000A0D05
	v_xor_b32_e32 v5, v5, v4                                   // 000000001C18: 2A0A0905
	v_sub_u32_e32 v4, v5, v4                                   // 000000001C1C: 6A080905
	v_lshlrev_b32_e32 v5, 2, v10                               // 000000001C20: 240A1482
	v_lshl_or_b32 v3, v3, 7, v5                                // 000000001C24: D2000003 04150F03
	v_mad_u64_u32 v[6:7], s[2:3], v4, s11, v[5:6]              // 000000001C2C: D1E80206 04141704
	v_ashrrev_i32_e32 v4, 31, v3                               // 000000001C34: 2208069F
	v_lshlrev_b64 v[7:8], 1, v[3:4]                            // 000000001C38: D28F0007 00020681
	v_mov_b32_e32 v4, s17                                      // 000000001C40: 7E080211
	v_add_co_u32_e64 v7, s[2:3], s16, v7                       // 000000001C44: D1190207 00020E10
	v_addc_co_u32_e64 v8, s[2:3], v4, v8, s[2:3]               // 000000001C4C: D11C0208 000A1104
	s_andn2_b64 s[2:3], s[8:9], exec                           // 000000001C54: 89827E08
	s_and_b64 s[0:1], s[0:1], exec                             // 000000001C58: 86807E00
	s_or_b64 s[8:9], s[2:3], s[0:1]                            // 000000001C5C: 87880002
	s_andn2_b64 s[0:1], s[14:15], exec                         // 000000001C60: 89807E0E
	s_and_b64 s[2:3], vcc, exec                                // 000000001C64: 86827E6A
	s_or_b64 s[14:15], s[0:1], s[2:3]                          // 000000001C68: 878E0200
	v_mov_b32_e32 v9, 0                                        // 000000001C6C: 7E120280
	s_mov_b64 s[0:1], 0                                        // 000000001C70: BE800180
	v_mov_b32_e32 v11, 0                                       // 000000001C74: 7E160280
	v_mov_b32_e32 v12, 0                                       // 000000001C78: 7E180280
	s_and_saveexec_b64 s[2:3], s[14:15]                        // 000000001C7C: BE82200E
	s_cbranch_execz 3                                          // 000000001C80: BF880003 <c662_qkrope_fwd+0x390>
	global_load_dwordx2 v[11:12], v[7:8], off                  // 000000001C84: DC548000 0B7F0007
	s_mov_b64 s[0:1], exec                                     // 000000001C8C: BE80017E
	s_or_b64 exec, exec, s[2:3]                                // 000000001C90: 87FE027E
	v_mov_b32_e32 v10, 0                                       // 000000001C94: 7E140280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001C98: BE822000
	s_cbranch_execz 2                                          // 000000001C9C: BF880002 <c662_qkrope_fwd+0x3a8>
	global_load_dwordx2 v[9:10], v[7:8], off offset:128        // 000000001CA0: DC548080 097F0007
	s_or_b64 exec, exec, s[2:3]                                // 000000001CA8: 87FE027E
	s_load_dword s10, s[4:5], 0x60                             // 000000001CAC: C0020282 00000060
	v_mov_b32_e32 v13, 0                                       // 000000001CB4: 7E1A0280
	v_ashrrev_i32_e32 v7, 31, v6                               // 000000001CB8: 220E0C9F
	v_mov_b32_e32 v15, 0                                       // 000000001CBC: 7E1E0280
	v_mov_b32_e32 v16, 0                                       // 000000001CC0: 7E200280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001CC4: BE822000
	s_cbranch_execz 10                                         // 000000001CC8: BF88000A <c662_qkrope_fwd+0x3f4>
	s_load_dwordx2 s[14:15], s[4:5], 0x30                      // 000000001CCC: C0060382 00000030
	v_lshlrev_b64 v[14:15], 1, v[6:7]                          // 000000001CD4: D28F000E 00020C81
	s_waitcnt lgkmcnt(0)                                       // 000000001CDC: BF8CC07F
	v_mov_b32_e32 v4, s15                                      // 000000001CE0: 7E08020F
	v_add_co_u32_e32 v14, vcc, s14, v14                        // 000000001CE4: 321C1C0E
	v_addc_co_u32_e32 v15, vcc, v4, v15, vcc                   // 000000001CE8: 381E1F04
	global_load_dwordx2 v[15:16], v[14:15], off                // 000000001CEC: DC548000 0F7F000E
	s_or_b64 exec, exec, s[2:3]                                // 000000001CF4: 87FE027E
	v_mov_b32_e32 v14, 0                                       // 000000001CF8: 7E1C0280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001CFC: BE822000
	s_cbranch_execz 10                                         // 000000001D00: BF88000A <c662_qkrope_fwd+0x42c>
	s_load_dwordx2 s[14:15], s[4:5], 0x38                      // 000000001D04: C0060382 00000038
	v_lshlrev_b64 v[6:7], 1, v[6:7]                            // 000000001D0C: D28F0006 00020C81
	s_waitcnt lgkmcnt(0)                                       // 000000001D14: BF8CC07F
	v_mov_b32_e32 v4, s15                                      // 000000001D18: 7E08020F
	v_add_co_u32_e32 v6, vcc, s14, v6                          // 000000001D1C: 320C0C0E
	v_addc_co_u32_e32 v7, vcc, v4, v7, vcc                     // 000000001D20: 380E0F04
	global_load_dwordx2 v[13:14], v[6:7], off                  // 000000001D24: DC548000 0D7F0006
	s_or_b64 exec, exec, s[2:3]                                // 000000001D2C: 87FE027E
	s_waitcnt vmcnt(0)                                         // 000000001D30: BF8C0F70
	v_and_b32_e32 v4, 0xffff0000, v11                          // 000000001D34: 260816FF FFFF0000
	v_lshlrev_b32_e32 v7, 16, v11                              // 000000001D3C: 240E1690
	v_and_b32_e32 v6, 0xffff0000, v9                           // 000000001D40: 260C12FF FFFF0000
	v_and_b32_e32 v11, 0xffff0000, v13                         // 000000001D48: 26161AFF FFFF0000
	v_lshlrev_b32_e32 v8, 16, v9                               // 000000001D50: 24101290
	v_and_b32_e32 v9, 0xffff0000, v15                          // 000000001D54: 26121EFF FFFF0000
	v_and_b32_e32 v21, 0xffff0000, v10                         // 000000001D5C: 262A14FF FFFF0000
	v_lshlrev_b32_e32 v22, 16, v10                             // 000000001D64: 242C1490
	v_mul_f32_e32 v10, v6, v11                                 // 000000001D68: 0A141706
	v_lshlrev_b32_e32 v18, 16, v13                             // 000000001D6C: 24241A90
	v_fmac_f32_e32 v10, v4, v9                                 // 000000001D70: 76141304
	v_mul_f32_e32 v4, v4, v11                                  // 000000001D74: 0A081704
	v_lshlrev_b32_e32 v15, 16, v15                             // 000000001D78: 241E1E90
	v_and_b32_e32 v19, 0xffff0000, v12                         // 000000001D7C: 262618FF FFFF0000
	v_lshlrev_b32_e32 v20, 16, v12                             // 000000001D84: 24281890
	v_and_b32_e32 v24, 0xffff0000, v14                         // 000000001D88: 26301CFF FFFF0000
	v_lshlrev_b32_e32 v14, 16, v14                             // 000000001D90: 241C1C90
	v_mul_f32_e32 v11, v8, v18                                 // 000000001D94: 0A162508
	v_fma_f32 v6, v6, v9, -v4                                  // 000000001D98: D1CB0006 84121306
	v_mul_f32_e32 v4, v7, v18                                  // 000000001DA0: 0A082507
	v_and_b32_e32 v23, 0xffff0000, v16                         // 000000001DA4: 262E20FF FFFF0000
	v_lshlrev_b32_e32 v16, 16, v16                             // 000000001DAC: 24202090
	v_fmac_f32_e32 v11, v7, v15                                // 000000001DB0: 76161F07
	v_fma_f32 v7, v8, v15, -v4                                 // 000000001DB4: D1CB0007 84121F08
	v_mul_f32_e32 v4, v19, v24                                 // 000000001DBC: 0A083113
	v_mul_f32_e32 v9, v20, v14                                 // 000000001DC0: 0A121D14
	v_mul_f32_e32 v13, v22, v14                                // 000000001DC4: 0A1A1D16
	v_fma_f32 v8, v21, v23, -v4                                // 000000001DC8: D1CB0008 84122F15
	v_fma_f32 v9, v22, v16, -v9                                // 000000001DD0: D1CB0009 84262116
	v_mul_f32_e32 v4, v7, v7                                   // 000000001DD8: 0A080F07
	v_mul_f32_e32 v14, v6, v6                                  // 000000001DDC: 0A1C0D06
	v_mul_f32_e32 v12, v21, v24                                // 000000001DE0: 0A183115
	v_fmac_f32_e32 v13, v20, v16                               // 000000001DE4: 761A2114
	v_mul_f32_e32 v15, v9, v9                                  // 000000001DE8: 0A1E1309
	v_fmac_f32_e32 v4, v11, v11                                // 000000001DEC: 7608170B
	v_fmac_f32_e32 v14, v10, v10                               // 000000001DF0: 761C150A
	v_fmac_f32_e32 v12, v19, v23                               // 000000001DF4: 76182F13
	v_mul_f32_e32 v16, v8, v8                                  // 000000001DF8: 0A201108
	v_fmac_f32_e32 v15, v13, v13                               // 000000001DFC: 761E1B0D
	v_add_f32_e32 v4, v4, v14                                  // 000000001E00: 02081D04
	v_fmac_f32_e32 v16, v12, v12                               // 000000001E04: 7620190C
	v_add_f32_e32 v4, v15, v4                                  // 000000001E08: 0208090F
	v_add_f32_e32 v4, v16, v4                                  // 000000001E0C: 02080910
	ds_swizzle_b32 v14, v4 offset:swizzle(SWAP,8)              // 000000001E10: D87A201F 0E000004
	v_and_b32_e32 v0, 48, v0                                   // 000000001E18: 260000B0
	v_and_or_b32 v0, v17, 3, v0                                // 000000001E1C: D2010000 04010711
	v_lshrrev_b32_e32 v2, 2, v2                                // 000000001E24: 20040482
	v_cmp_eq_u32_e32 vcc, 0, v0                                // 000000001E28: 7D940080
	s_waitcnt lgkmcnt(0)                                       // 000000001E2C: BF8CC07F
	v_add_f32_e32 v4, v4, v14                                  // 000000001E30: 02081D04
	ds_swizzle_b32 v14, v4 offset:swizzle(SWAP,4)              // 000000001E34: D87A101F 0E000004
	v_add_u32_e32 v2, 0, v2                                    // 000000001E3C: 68040480
	s_and_b64 s[8:9], s[8:9], vcc                              // 000000001E40: 86886A08
	s_waitcnt lgkmcnt(0)                                       // 000000001E44: BF8CC07F
	v_add_f32_e32 v4, v4, v14                                  // 000000001E48: 02081D04
	ds_swizzle_b32 v14, v4 offset:swizzle(SWAP,2)              // 000000001E4C: D87A081F 0E000004
	s_waitcnt lgkmcnt(0)                                       // 000000001E54: BF8CC07F
	v_add_f32_e32 v4, v4, v14                                  // 000000001E58: 02081D04
	ds_swizzle_b32 v14, v4 offset:swizzle(SWAP,1)              // 000000001E5C: D87A041F 0E000004
	s_waitcnt lgkmcnt(0)                                       // 000000001E64: BF8CC07F
	v_add_f32_e32 v4, v4, v14                                  // 000000001E68: 02081D04
	v_mov_b32_e32 v14, s10                                     // 000000001E6C: 7E1C020A
	v_fmac_f32_e32 v14, 0x3c000000, v4                         // 000000001E70: 761C08FF 3C000000
	v_rsq_f32_e32 v14, v14                                     // 000000001E78: 7E1C490E
	ds_write_b32 v2, v14                                       // 000000001E7C: D81A0000 00000E02
	s_waitcnt lgkmcnt(0)                                       // 000000001E84: BF8CC07F
	s_barrier                                                  // 000000001E88: BF8A0000
	s_and_saveexec_b64 s[2:3], s[8:9]                          // 000000001E8C: BE822008
	s_cbranch_execnz 4                                         // 000000001E90: BF890004 <c662_qkrope_fwd+0x5a4>
	s_or_b64 exec, exec, s[2:3]                                // 000000001E94: 87FE027E
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001E98: BE822000
	s_cbranch_execnz 16                                        // 000000001E9C: BF890010 <c662_qkrope_fwd+0x5e0>
	s_endpgm                                                   // 000000001EA0: BF810000
	v_add_u32_e32 v4, 0, v5                                    // 000000001EA4: 68080A80
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001EA8: 2204029F
	ds_read_b32 v4, v4                                         // 000000001EAC: D86C0000 04000004
	v_lshlrev_b64 v[0:1], 2, v[1:2]                            // 000000001EB4: D28F0000 00020282
	v_mov_b32_e32 v2, s13                                      // 000000001EBC: 7E04020D
	v_add_co_u32_e32 v0, vcc, s12, v0                          // 000000001EC0: 3200000C
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc                     // 000000001EC4: 38020302
	s_waitcnt lgkmcnt(0)                                       // 000000001EC8: BF8CC07F
	global_store_dword v[0:1], v4, off                         // 000000001ECC: DC708000 007F0400
	s_or_b64 exec, exec, s[2:3]                                // 000000001ED4: 87FE027E
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001ED8: BE822000
	s_cbranch_execz 65520                                      // 000000001EDC: BF88FFF0 <c662_qkrope_fwd+0x5a0>
	s_load_dword s0, s[4:5], 0x5c                              // 000000001EE0: C0020002 0000005C
	v_ashrrev_i32_e32 v4, 31, v3                               // 000000001EE8: 2208069F
	v_lshlrev_b64 v[0:1], 1, v[3:4]                            // 000000001EEC: D28F0000 00020681
	v_mov_b32_e32 v2, s7                                       // 000000001EF4: 7E040207
	v_add_co_u32_e32 v0, vcc, s6, v0                           // 000000001EF8: 32000006
	s_waitcnt lgkmcnt(0)                                       // 000000001EFC: BF8CC07F
	v_mul_f32_e32 v4, s0, v14                                  // 000000001F00: 0A081C00
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc                     // 000000001F04: 38020302
	v_mul_f32_e32 v2, v10, v4                                  // 000000001F08: 0A04090A
	v_mul_f32_e32 v10, v13, v4                                 // 000000001F0C: 0A14090D
	v_mul_f32_e32 v5, v11, v4                                  // 000000001F10: 0A0A090B
	v_mul_f32_e32 v3, v12, v4                                  // 000000001F14: 0A06090C
	v_bfe_u32 v11, v10, 16, 1                                  // 000000001F18: D1C8000B 0205210A
	s_movk_i32 s0, 0x7fff                                      // 000000001F20: B0007FFF
	v_bfe_u32 v12, v3, 16, 1                                   // 000000001F24: D1C8000C 02052103
	v_add3_u32 v11, v10, v11, s0                               // 000000001F2C: D1FF000B 0002170A
	v_bfe_u32 v13, v5, 16, 1                                   // 000000001F34: D1C8000D 02052105
	v_add3_u32 v12, v3, v12, s0                                // 000000001F3C: D1FF000C 00021903
	v_lshrrev_b32_e32 v11, 16, v11                             // 000000001F44: 20161690
	v_mov_b32_e32 v15, 0x7fff                                  // 000000001F48: 7E1E02FF 00007FFF
	v_cmp_o_f32_e32 vcc, v10, v10                              // 000000001F50: 7C8E150A
	v_bfe_u32 v14, v2, 16, 1                                   // 000000001F54: D1C8000E 02052102
	v_add3_u32 v13, v5, v13, s0                                // 000000001F5C: D1FF000D 00021B05
	v_lshrrev_b32_e32 v12, 16, v12                             // 000000001F64: 20181890
	v_cndmask_b32_e32 v10, v15, v11, vcc                       // 000000001F68: 0014170F
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000001F6C: 7C8E0703
	v_add3_u32 v14, v2, v14, s0                                // 000000001F70: D1FF000E 00021D02
	v_lshrrev_b32_e32 v13, 16, v13                             // 000000001F78: 201A1A90
	v_cndmask_b32_e32 v3, v15, v12, vcc                        // 000000001F7C: 0006190F
	v_cmp_o_f32_e32 vcc, v5, v5                                // 000000001F80: 7C8E0B05
	v_lshrrev_b32_e32 v14, 16, v14                             // 000000001F84: 201C1C90
	v_cndmask_b32_e32 v5, v15, v13, vcc                        // 000000001F88: 000A1B0F
	v_cmp_o_f32_e32 vcc, v2, v2                                // 000000001F8C: 7C8E0502
	s_mov_b32 s1, 0x5040100                                    // 000000001F90: BE8100FF 05040100
	v_cndmask_b32_e32 v2, v15, v14, vcc                        // 000000001F98: 00041D0F
	v_perm_b32 v3, v3, v10, s1                                 // 000000001F9C: D1ED0003 00061503
	v_perm_b32 v2, v2, v5, s1                                  // 000000001FA4: D1ED0002 00060B02
	global_store_dwordx2 v[0:1], v[2:3], off                   // 000000001FAC: DC748000 007F0200
	v_mul_f32_e32 v2, v6, v4                                   // 000000001FB4: 0A040906
	v_mul_f32_e32 v5, v7, v4                                   // 000000001FB8: 0A0A0907
	v_mul_f32_e32 v3, v8, v4                                   // 000000001FBC: 0A060908
	v_mul_f32_e32 v4, v9, v4                                   // 000000001FC0: 0A080909
	v_bfe_u32 v6, v4, 16, 1                                    // 000000001FC4: D1C80006 02052104
	v_bfe_u32 v7, v3, 16, 1                                    // 000000001FCC: D1C80007 02052103
	v_add3_u32 v6, v4, v6, s0                                  // 000000001FD4: D1FF0006 00020D04
	v_cmp_o_f32_e32 vcc, v4, v4                                // 000000001FDC: 7C8E0904
	v_bfe_u32 v8, v5, 16, 1                                    // 000000001FE0: D1C80008 02052105
	v_add3_u32 v7, v3, v7, s0                                  // 000000001FE8: D1FF0007 00020F03
	v_cndmask_b32_sdwa v4, v15, v6, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001FF0: 00080CF9 0506060F
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000001FF8: 7C8E0703
	v_bfe_u32 v9, v2, 16, 1                                    // 000000001FFC: D1C80009 02052102
	v_add3_u32 v8, v5, v8, s0                                  // 000000002004: D1FF0008 00021105
	v_cndmask_b32_sdwa v3, v15, v7, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000200C: 00060EF9 0506060F
	v_cmp_o_f32_e32 vcc, v5, v5                                // 000000002014: 7C8E0B05
	v_add3_u32 v9, v2, v9, s0                                  // 000000002018: D1FF0009 00021302
	v_perm_b32 v3, v3, v4, s1                                  // 000000002020: D1ED0003 00060903
	v_cndmask_b32_sdwa v4, v15, v8, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002028: 000810F9 0506060F
	v_cmp_o_f32_e32 vcc, v2, v2                                // 000000002030: 7C8E0502
	v_cndmask_b32_sdwa v2, v15, v9, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002034: 000412F9 0506060F
	v_perm_b32 v2, v2, v4, s1                                  // 00000000203C: D1ED0002 00060902
	global_store_dwordx2 v[0:1], v[2:3], off offset:128        // 000000002044: DC748080 007F0200
	s_endpgm                                                   // 00000000204C: BF810000
