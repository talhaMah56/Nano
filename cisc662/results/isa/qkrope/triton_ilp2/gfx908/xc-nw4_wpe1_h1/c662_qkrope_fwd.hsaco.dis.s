
/u/tmahmood1/project/nanochat/cisc662/results/isa/qkrope/triton_ilp2/gfx908/xc-nw4_wpe1_h1/c662_qkrope_fwd.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001900 <c662_qkrope_fwd>:
	s_load_dword s0, s[4:5], 0x48                              // 000000001900: C0020002 00000048
	s_load_dwordx2 s[12:13], s[4:5], 0x54                      // 000000001908: C0060302 00000054
	v_lshrrev_b32_e32 v1, 6, v0                                // 000000001910: 20020086
	v_lshrrev_b32_e32 v4, 3, v0                                // 000000001914: 20080083
	v_and_b32_e32 v2, 31, v0                                   // 000000001918: 2604009F
	s_waitcnt lgkmcnt(0)                                       // 00000000191C: BF8CC07F
	s_cmp_ge_i32 s6, s0                                        // 000000001920: BF030006
	v_lshlrev_b32_e32 v3, 3, v0                                // 000000001924: 24060083
	s_cbranch_scc0 104                                         // 000000001928: BF840068 <c662_qkrope_fwd+0x1cc>
	s_load_dword s1, s[4:5], 0x50                              // 00000000192C: C0020042 00000050
	s_load_dword s7, s[4:5], 0x44                              // 000000001934: C00201C2 00000044
	s_abs_i32 s10, s12                                         // 00000000193C: BE8A300C
	s_sub_i32 s0, s6, s0                                       // 000000001940: 81800006
	v_cvt_f32_u32_e32 v6, s10                                  // 000000001944: 7E0C0C0A
	s_waitcnt lgkmcnt(0)                                       // 000000001948: BF8CC07F
	s_abs_i32 s8, s1                                           // 00000000194C: BE883001
	v_cvt_f32_u32_e32 v5, s8                                   // 000000001950: 7E0A0C08
	s_sub_i32 s11, 0, s8                                       // 000000001954: 818B0880
	s_lshl_b32 s9, s0, 5                                       // 000000001958: 8E098500
	s_bfe_i32 s0, s0, 0x1001a                                  // 00000000195C: 9300FF00 0001001A
	v_rcp_iflag_f32_e32 v5, v5                                 // 000000001964: 7E0A4705
	v_or_b32_e32 v8, s9, v4                                    // 000000001968: 28100809
	v_add_u32_e32 v9, s0, v8                                   // 00000000196C: 68121000
	v_rcp_iflag_f32_e32 v6, v6                                 // 000000001970: 7E0C4706
	v_mul_f32_e32 v5, 0x4f7ffffe, v5                           // 000000001974: 0A0A0AFF 4F7FFFFE
	v_cvt_u32_f32_e32 v5, v5                                   // 00000000197C: 7E0A0F05
	v_xor_b32_e32 v9, s0, v9                                   // 000000001980: 2A121200
	v_mul_f32_e32 v6, 0x4f7ffffe, v6                           // 000000001984: 0A0C0CFF 4F7FFFFE
	v_cvt_u32_f32_e32 v6, v6                                   // 00000000198C: 7E0C0F06
	v_mul_lo_u32 v7, s11, v5                                   // 000000001990: D2850007 00020A0B
	s_sub_i32 s11, 0, s10                                      // 000000001998: 818B0A80
	s_ashr_i32 s1, s1, 31                                      // 00000000199C: 90019F01
	s_xor_b32 s0, s0, s1                                       // 0000000019A0: 88000100
	v_mul_hi_u32 v7, v5, v7                                    // 0000000019A4: D2860007 00020F05
	s_load_dwordx2 s[2:3], s[4:5], 0x8                         // 0000000019AC: C0060082 00000008
	v_and_b32_e32 v24, 31, v0                                  // 0000000019B4: 2630009F
	v_or_b32_e32 v17, s9, v24                                  // 0000000019B8: 28223009
	v_add_u32_e32 v5, v5, v7                                   // 0000000019BC: 680A0F05
	v_mul_hi_u32 v5, v9, v5                                    // 0000000019C0: D2860005 00020B09
	v_mul_lo_u32 v7, s11, v6                                   // 0000000019C8: D2850007 00020C0B
	v_lshrrev_b32_e32 v18, 6, v0                               // 0000000019D0: 20240086
	v_and_b32_e32 v23, 0xf8, v0                                // 0000000019D4: 262E00FF 000000F8
	v_mul_lo_u32 v10, v5, s8                                   // 0000000019DC: D285000A 00001105
	v_add_u32_e32 v11, 1, v5                                   // 0000000019E4: 68160A81
	v_mul_hi_u32 v7, v6, v7                                    // 0000000019E8: D2860007 00020F06
	v_sub_u32_e32 v9, v9, v10                                  // 0000000019F0: 6A121509
	v_cmp_le_u32_e32 vcc, s8, v9                               // 0000000019F4: 7D961208
	v_subrev_u32_e32 v10, s8, v9                               // 0000000019F8: 6C141208
	v_cndmask_b32_e32 v5, v5, v11, vcc                         // 0000000019FC: 000A1705
	v_cndmask_b32_e32 v9, v9, v10, vcc                         // 000000001A00: 00121509
	v_add_u32_e32 v10, 1, v5                                   // 000000001A04: 68140A81
	v_cmp_le_u32_e32 vcc, s8, v9                               // 000000001A08: 7D961208
	v_cndmask_b32_e32 v5, v5, v10, vcc                         // 000000001A0C: 000A1505
	v_xor_b32_e32 v5, s0, v5                                   // 000000001A10: 2A0A0A00
	v_subrev_u32_e32 v5, s0, v5                                // 000000001A14: 6C0A0A00
	v_sub_u32_e32 v9, 0, v5                                    // 000000001A18: 6A120A80
	v_max_i32_e32 v9, v5, v9                                   // 000000001A1C: 1A121305
	v_add_u32_e32 v6, v6, v7                                   // 000000001A20: 680C0F06
	v_mul_hi_u32 v6, v9, v6                                    // 000000001A24: D2860006 00020D09
	v_ashrrev_i32_e32 v5, 31, v5                               // 000000001A2C: 220A0A9F
	v_cmp_gt_i32_e32 vcc, s7, v8                               // 000000001A30: 7D881007
	v_cmp_gt_i32_e64 s[8:9], s7, v17                           // 000000001A34: D0C40008 00022207
	v_mul_lo_u32 v6, v6, s10                                   // 000000001A3C: D2850006 00001506
	s_and_b64 s[14:15], vcc, exec                              // 000000001A44: 868E7E6A
	v_sub_u32_e32 v6, v9, v6                                   // 000000001A48: 6A0C0D09
	v_subrev_u32_e32 v7, s10, v6                               // 000000001A4C: 6C0E0C0A
	v_cmp_le_u32_e64 s[0:1], s10, v6                           // 000000001A50: D0CB0000 00020C0A
	v_cndmask_b32_e64 v6, v6, v7, s[0:1]                       // 000000001A58: D1000006 00020F06
	v_subrev_u32_e32 v7, s10, v6                               // 000000001A60: 6C0E0C0A
	v_cmp_le_u32_e64 s[0:1], s10, v6                           // 000000001A64: D0CB0000 00020C0A
	v_cndmask_b32_e64 v6, v6, v7, s[0:1]                       // 000000001A6C: D1000006 00020F06
	v_xor_b32_e32 v6, v6, v5                                   // 000000001A74: 2A0C0B06
	v_sub_u32_e32 v6, v6, v5                                   // 000000001A78: 6A0C0B06
	v_and_b32_e32 v5, 56, v3                                   // 000000001A7C: 260A06B8
	v_lshl_or_b32 v19, v8, 7, v5                               // 000000001A80: D2000013 04150F08
	v_ashrrev_i32_e32 v20, 31, v19                             // 000000001A88: 2228269F
	v_mad_u64_u32 v[21:22], s[0:1], v6, s13, v[5:6]            // 000000001A8C: D1E80015 04141B06
	v_lshlrev_b64 v[5:6], 1, v[19:20]                          // 000000001A94: D28F0005 00022681
	s_waitcnt lgkmcnt(0)                                       // 000000001A9C: BF8CC07F
	v_mov_b32_e32 v7, s3                                       // 000000001AA0: 7E0E0203
	v_add_co_u32_e64 v9, s[0:1], s2, v5                        // 000000001AA4: D1190009 00020A02
	v_addc_co_u32_e64 v10, s[0:1], v7, v6, s[0:1]              // 000000001AAC: D11C000A 00020D07
	s_cbranch_execz 6                                          // 000000001AB4: BF880006 <c662_qkrope_fwd+0x1d0>
	s_load_dwordx2 s[6:7], s[4:5], 0x18                        // 000000001AB8: C0060182 00000018
	s_load_dwordx2 s[10:11], s[4:5], 0x28                      // 000000001AC0: C0060282 00000028
	s_branch 107                                               // 000000001AC8: BF82006B <c662_qkrope_fwd+0x378>
	s_mov_b64 s[14:15], 0                                      // 000000001ACC: BE8E0180
	s_load_dword s0, s[4:5], 0x4c                              // 000000001AD0: C0020002 0000004C
	s_abs_i32 s12, s12                                         // 000000001AD8: BE8C300C
	v_cvt_f32_u32_e32 v6, s12                                  // 000000001ADC: 7E0C0C0C
	s_lshl_b32 s2, s6, 5                                       // 000000001AE0: 8E028506
	s_bfe_i32 s3, s6, 0x1001a                                  // 000000001AE4: 9303FF06 0001001A
	s_waitcnt lgkmcnt(0)                                       // 000000001AEC: BF8CC07F
	s_abs_i32 s1, s0                                           // 000000001AF0: BE813000
	v_cvt_f32_u32_e32 v5, s1                                   // 000000001AF4: 7E0A0C01
	s_sub_i32 s6, 0, s1                                        // 000000001AF8: 81860180
	v_rcp_iflag_f32_e32 v6, v6                                 // 000000001AFC: 7E0C4706
	v_or_b32_e32 v4, s2, v4                                    // 000000001B00: 28080802
	v_rcp_iflag_f32_e32 v5, v5                                 // 000000001B04: 7E0A4705
	v_add_u32_e32 v8, s3, v4                                   // 000000001B08: 68100803
	v_mul_f32_e32 v6, 0x4f7ffffe, v6                           // 000000001B0C: 0A0C0CFF 4F7FFFFE
	v_cvt_u32_f32_e32 v6, v6                                   // 000000001B14: 7E0C0F06
	v_mul_f32_e32 v5, 0x4f7ffffe, v5                           // 000000001B18: 0A0A0AFF 4F7FFFFE
	v_cvt_u32_f32_e32 v5, v5                                   // 000000001B20: 7E0A0F05
	v_xor_b32_e32 v8, s3, v8                                   // 000000001B24: 2A101003
	s_ashr_i32 s0, s0, 31                                      // 000000001B28: 90009F00
	s_xor_b32 s0, s3, s0                                       // 000000001B2C: 88000003
	v_mul_lo_u32 v7, s6, v5                                    // 000000001B30: D2850007 00020A06
	s_sub_i32 s6, 0, s12                                       // 000000001B38: 81860C80
	v_mul_lo_u32 v9, s6, v6                                    // 000000001B3C: D2850009 00020C06
	v_or_b32_e32 v17, s2, v2                                   // 000000001B44: 28220402
	v_mul_hi_u32 v7, v5, v7                                    // 000000001B48: D2860007 00020F05
	v_and_b32_e32 v3, 56, v3                                   // 000000001B50: 260606B8
	v_lshl_or_b32 v19, v4, 7, v3                               // 000000001B54: D2000013 040D0F04
	v_ashrrev_i32_e32 v20, 31, v19                             // 000000001B5C: 2228269F
	v_add_u32_e32 v5, v5, v7                                   // 000000001B60: 680A0F05
	v_mul_hi_u32 v5, v8, v5                                    // 000000001B64: D2860005 00020B08
	v_mul_hi_u32 v7, v6, v9                                    // 000000001B6C: D2860007 00021306
	v_and_b32_e32 v23, 0xf8, v0                                // 000000001B74: 262E00FF 000000F8
	v_mov_b32_e32 v18, v1                                      // 000000001B7C: 7E240301
	v_mul_lo_u32 v9, v5, s1                                    // 000000001B80: D2850009 00000305
	v_add_u32_e32 v6, v6, v7                                   // 000000001B88: 680C0F06
	v_add_u32_e32 v7, 1, v5                                    // 000000001B8C: 680E0A81
	v_mov_b32_e32 v24, v2                                      // 000000001B90: 7E300302
	v_sub_u32_e32 v8, v8, v9                                   // 000000001B94: 6A101308
	v_cmp_le_u32_e32 vcc, s1, v8                               // 000000001B98: 7D961001
	v_cndmask_b32_e32 v5, v5, v7, vcc                          // 000000001B9C: 000A0F05
	v_subrev_u32_e32 v7, s1, v8                                // 000000001BA0: 6C0E1001
	v_cndmask_b32_e32 v7, v8, v7, vcc                          // 000000001BA4: 000E0F08
	v_add_u32_e32 v8, 1, v5                                    // 000000001BA8: 68100A81
	v_cmp_le_u32_e32 vcc, s1, v7                               // 000000001BAC: 7D960E01
	v_cndmask_b32_e32 v5, v5, v8, vcc                          // 000000001BB0: 000A1105
	v_xor_b32_e32 v5, s0, v5                                   // 000000001BB4: 2A0A0A00
	v_subrev_u32_e32 v5, s0, v5                                // 000000001BB8: 6C0A0A00
	v_sub_u32_e32 v7, 0, v5                                    // 000000001BBC: 6A0E0A80
	v_max_i32_e32 v7, v5, v7                                   // 000000001BC0: 1A0E0F05
	v_mul_hi_u32 v6, v7, v6                                    // 000000001BC4: D2860006 00020D07
	s_load_dwordx2 s[16:17], s[4:5], 0x0                       // 000000001BCC: C0060402 00000000
	s_load_dwordx2 s[6:7], s[4:5], 0x10                        // 000000001BD4: C0060182 00000010
	s_load_dword s0, s[4:5], 0x40                              // 000000001BDC: C0020002 00000040
	s_load_dwordx2 s[10:11], s[4:5], 0x20                      // 000000001BE4: C0060282 00000020
	v_ashrrev_i32_e32 v5, 31, v5                               // 000000001BEC: 220A0A9F
	v_mul_lo_u32 v6, v6, s12                                   // 000000001BF0: D2850006 00001906
	s_waitcnt lgkmcnt(0)                                       // 000000001BF8: BF8CC07F
	v_cmp_gt_i32_e32 vcc, s0, v4                               // 000000001BFC: 7D880800
	v_cmp_gt_i32_e64 s[0:1], s0, v17                           // 000000001C00: D0C40000 00022200
	s_and_b64 s[0:1], s[0:1], exec                             // 000000001C08: 86807E00
	v_sub_u32_e32 v6, v7, v6                                   // 000000001C0C: 6A0C0D07
	v_subrev_u32_e32 v7, s12, v6                               // 000000001C10: 6C0E0C0C
	v_cmp_le_u32_e64 s[2:3], s12, v6                           // 000000001C14: D0CB0002 00020C0C
	v_cndmask_b32_e64 v6, v6, v7, s[2:3]                       // 000000001C1C: D1000006 000A0F06
	v_subrev_u32_e32 v7, s12, v6                               // 000000001C24: 6C0E0C0C
	v_cmp_le_u32_e64 s[2:3], s12, v6                           // 000000001C28: D0CB0002 00020C0C
	v_cndmask_b32_e64 v6, v6, v7, s[2:3]                       // 000000001C30: D1000006 000A0F06
	v_xor_b32_e32 v6, v6, v5                                   // 000000001C38: 2A0C0B06
	v_sub_u32_e32 v5, v6, v5                                   // 000000001C3C: 6A0A0B06
	v_mad_u64_u32 v[21:22], s[2:3], v5, s13, v[3:4]            // 000000001C40: D1E80215 040C1B05
	v_lshlrev_b64 v[3:4], 1, v[19:20]                          // 000000001C48: D28F0003 00022681
	v_mov_b32_e32 v5, s17                                      // 000000001C50: 7E0A0211
	v_add_co_u32_e64 v9, s[2:3], s16, v3                       // 000000001C54: D1190209 00020610
	v_addc_co_u32_e64 v10, s[2:3], v5, v4, s[2:3]              // 000000001C5C: D11C020A 000A0905
	s_andn2_b64 s[2:3], s[8:9], exec                           // 000000001C64: 89827E08
	s_or_b64 s[8:9], s[2:3], s[0:1]                            // 000000001C68: 87880002
	s_andn2_b64 s[0:1], s[14:15], exec                         // 000000001C6C: 89807E0E
	s_and_b64 s[2:3], vcc, exec                                // 000000001C70: 86827E6A
	s_or_b64 s[14:15], s[0:1], s[2:3]                          // 000000001C74: 878E0200
	v_mov_b32_e32 v1, 0                                        // 000000001C78: 7E020280
	s_mov_b64 s[0:1], 0                                        // 000000001C7C: BE800180
	v_mov_b32_e32 v5, 0                                        // 000000001C80: 7E0A0280
	v_mov_b32_e32 v6, 0                                        // 000000001C84: 7E0C0280
	v_mov_b32_e32 v7, 0                                        // 000000001C88: 7E0E0280
	v_mov_b32_e32 v8, 0                                        // 000000001C8C: 7E100280
	s_and_saveexec_b64 s[2:3], s[14:15]                        // 000000001C90: BE82200E
	s_cbranch_execz 3                                          // 000000001C94: BF880003 <c662_qkrope_fwd+0x3a4>
	global_load_dwordx4 v[5:8], v[9:10], off                   // 000000001C98: DC5C8000 057F0009
	s_mov_b64 s[0:1], exec                                     // 000000001CA0: BE80017E
	s_or_b64 exec, exec, s[2:3]                                // 000000001CA4: 87FE027E
	v_mov_b32_e32 v2, 0                                        // 000000001CA8: 7E040280
	v_mov_b32_e32 v3, 0                                        // 000000001CAC: 7E060280
	v_mov_b32_e32 v4, 0                                        // 000000001CB0: 7E080280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001CB4: BE822000
	s_cbranch_execz 2                                          // 000000001CB8: BF880002 <c662_qkrope_fwd+0x3c4>
	global_load_dwordx4 v[1:4], v[9:10], off offset:128        // 000000001CBC: DC5C8080 017F0009
	s_or_b64 exec, exec, s[2:3]                                // 000000001CC4: 87FE027E
	s_load_dword s12, s[4:5], 0x60                             // 000000001CC8: C0020302 00000060
	v_mov_b32_e32 v9, 0                                        // 000000001CD0: 7E120280
	v_ashrrev_i32_e32 v22, 31, v21                             // 000000001CD4: 222C2A9F
	v_mov_b32_e32 v13, 0                                       // 000000001CD8: 7E1A0280
	v_mov_b32_e32 v14, 0                                       // 000000001CDC: 7E1C0280
	v_mov_b32_e32 v15, 0                                       // 000000001CE0: 7E1E0280
	v_mov_b32_e32 v16, 0                                       // 000000001CE4: 7E200280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001CE8: BE822000
	s_cbranch_execz 10                                         // 000000001CEC: BF88000A <c662_qkrope_fwd+0x418>
	s_load_dwordx2 s[14:15], s[4:5], 0x30                      // 000000001CF0: C0060382 00000030
	v_lshlrev_b64 v[10:11], 1, v[21:22]                        // 000000001CF8: D28F000A 00022A81
	s_waitcnt lgkmcnt(0)                                       // 000000001D00: BF8CC07F
	v_mov_b32_e32 v12, s15                                     // 000000001D04: 7E18020F
	v_add_co_u32_e32 v10, vcc, s14, v10                        // 000000001D08: 3214140E
	v_addc_co_u32_e32 v11, vcc, v12, v11, vcc                  // 000000001D0C: 3816170C
	global_load_dwordx4 v[13:16], v[10:11], off                // 000000001D10: DC5C8000 0D7F000A
	s_or_b64 exec, exec, s[2:3]                                // 000000001D18: 87FE027E
	v_mov_b32_e32 v10, 0                                       // 000000001D1C: 7E140280
	v_mov_b32_e32 v11, 0                                       // 000000001D20: 7E160280
	v_mov_b32_e32 v12, 0                                       // 000000001D24: 7E180280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001D28: BE822000
	s_cbranch_execz 10                                         // 000000001D2C: BF88000A <c662_qkrope_fwd+0x458>
	s_load_dwordx2 s[14:15], s[4:5], 0x38                      // 000000001D30: C0060382 00000038
	v_lshlrev_b64 v[9:10], 1, v[21:22]                         // 000000001D38: D28F0009 00022A81
	s_waitcnt lgkmcnt(0)                                       // 000000001D40: BF8CC07F
	v_mov_b32_e32 v11, s15                                     // 000000001D44: 7E16020F
	v_add_co_u32_e32 v9, vcc, s14, v9                          // 000000001D48: 3212120E
	v_addc_co_u32_e32 v10, vcc, v11, v10, vcc                  // 000000001D4C: 3814150B
	global_load_dwordx4 v[9:12], v[9:10], off                  // 000000001D50: DC5C8000 097F0009
	s_or_b64 exec, exec, s[2:3]                                // 000000001D58: 87FE027E
	s_waitcnt vmcnt(0)                                         // 000000001D5C: BF8C0F70
	v_and_b32_e32 v20, 0xffff0000, v5                          // 000000001D60: 26280AFF FFFF0000
	v_and_b32_e32 v26, 0xffff0000, v1                          // 000000001D68: 263402FF FFFF0000
	v_lshlrev_b32_e32 v27, 16, v1                              // 000000001D70: 24360290
	v_and_b32_e32 v28, 0xffff0000, v2                          // 000000001D74: 263804FF FFFF0000
	v_lshlrev_b32_e32 v29, 16, v2                              // 000000001D7C: 243A0490
	v_and_b32_e32 v1, 0xffff0000, v13                          // 000000001D80: 26021AFF FFFF0000
	v_lshlrev_b32_e32 v2, 16, v13                              // 000000001D88: 24041A90
	v_and_b32_e32 v13, 0xffff0000, v9                          // 000000001D8C: 261A12FF FFFF0000
	v_and_b32_e32 v34, 0xffff0000, v16                         // 000000001D94: 264420FF FFFF0000
	v_lshlrev_b32_e32 v35, 16, v16                             // 000000001D9C: 24462090
	v_lshlrev_b32_e32 v16, 16, v9                              // 000000001DA0: 24201290
	v_and_b32_e32 v36, 0xffff0000, v10                         // 000000001DA4: 264814FF FFFF0000
	v_lshlrev_b32_e32 v37, 16, v10                             // 000000001DAC: 244A1490
	v_mul_f32_e32 v9, v26, v13                                 // 000000001DB0: 0A121B1A
	v_mul_f32_e32 v10, v20, v13                                // 000000001DB4: 0A141B14
	v_lshlrev_b32_e32 v5, 16, v5                               // 000000001DB8: 240A0A90
	v_fmac_f32_e32 v9, v20, v1                                 // 000000001DBC: 76120314
	v_fma_f32 v1, v26, v1, -v10                                // 000000001DC0: D1CB0001 842A031A
	v_mul_f32_e32 v10, v27, v16                                // 000000001DC8: 0A14211B
	v_and_b32_e32 v21, 0xffff0000, v6                          // 000000001DCC: 262A0CFF FFFF0000
	v_fmac_f32_e32 v10, v5, v2                                 // 000000001DD4: 76140505
	v_mul_f32_e32 v5, v5, v16                                  // 000000001DD8: 0A0A2105
	v_lshlrev_b32_e32 v6, 16, v6                               // 000000001DDC: 240C0C90
	v_and_b32_e32 v30, 0xffff0000, v3                          // 000000001DE0: 263C06FF FFFF0000
	v_lshlrev_b32_e32 v31, 16, v3                              // 000000001DE8: 243E0690
	v_and_b32_e32 v3, 0xffff0000, v14                          // 000000001DEC: 26061CFF FFFF0000
	v_and_b32_e32 v38, 0xffff0000, v11                         // 000000001DF4: 264C16FF FFFF0000
	v_lshlrev_b32_e32 v39, 16, v11                             // 000000001DFC: 244E1690
	v_fma_f32 v2, v27, v2, -v5                                 // 000000001E00: D1CB0002 8416051B
	v_mul_f32_e32 v11, v28, v36                                // 000000001E08: 0A16491C
	v_mul_f32_e32 v5, v21, v36                                 // 000000001E0C: 0A0A4915
	v_and_b32_e32 v32, 0xffff0000, v4                          // 000000001E10: 264008FF FFFF0000
	v_lshlrev_b32_e32 v33, 16, v4                              // 000000001E18: 24420890
	v_lshlrev_b32_e32 v4, 16, v14                              // 000000001E1C: 24081C90
	v_and_b32_e32 v40, 0xffff0000, v12                         // 000000001E20: 265018FF FFFF0000
	v_lshlrev_b32_e32 v41, 16, v12                             // 000000001E28: 24521890
	v_fmac_f32_e32 v11, v21, v3                                // 000000001E2C: 76160715
	v_fma_f32 v3, v28, v3, -v5                                 // 000000001E30: D1CB0003 8416071C
	v_mul_f32_e32 v12, v29, v37                                // 000000001E38: 0A184B1D
	v_mul_f32_e32 v5, v6, v37                                  // 000000001E3C: 0A0A4B06
	v_mul_f32_e32 v20, v2, v2                                  // 000000001E40: 0A280502
	v_mul_f32_e32 v21, v1, v1                                  // 000000001E44: 0A2A0301
	v_fmac_f32_e32 v12, v6, v4                                 // 000000001E48: 76180906
	v_fma_f32 v4, v29, v4, -v5                                 // 000000001E4C: D1CB0004 8416091D
	v_fmac_f32_e32 v20, v10, v10                               // 000000001E54: 7628150A
	v_fmac_f32_e32 v21, v9, v9                                 // 000000001E58: 762A1309
	v_add_f32_e32 v20, v20, v21                                // 000000001E5C: 02282B14
	v_mul_f32_e32 v21, v4, v4                                  // 000000001E60: 0A2A0904
	v_and_b32_e32 v22, 0xffff0000, v7                          // 000000001E64: 262C0EFF FFFF0000
	v_lshlrev_b32_e32 v7, 16, v7                               // 000000001E6C: 240E0E90
	v_fmac_f32_e32 v21, v12, v12                               // 000000001E70: 762A190C
	v_and_b32_e32 v14, 0xffff0000, v15                         // 000000001E74: 261C1EFF FFFF0000
	v_lshlrev_b32_e32 v15, 16, v15                             // 000000001E7C: 241E1E90
	v_mul_f32_e32 v13, v30, v38                                // 000000001E80: 0A1A4D1E
	v_mul_f32_e32 v5, v22, v38                                 // 000000001E84: 0A0A4D16
	v_mul_f32_e32 v6, v7, v39                                  // 000000001E88: 0A0C4F07
	v_add_f32_e32 v20, v21, v20                                // 000000001E8C: 02282915
	v_mul_f32_e32 v21, v3, v3                                  // 000000001E90: 0A2A0703
	v_fmac_f32_e32 v13, v22, v14                               // 000000001E94: 761A1D16
	v_fma_f32 v5, v30, v14, -v5                                // 000000001E98: D1CB0005 84161D1E
	v_mul_f32_e32 v14, v31, v39                                // 000000001EA0: 0A1C4F1F
	v_fma_f32 v6, v31, v15, -v6                                // 000000001EA4: D1CB0006 841A1F1F
	v_fmac_f32_e32 v21, v11, v11                               // 000000001EAC: 762A170B
	v_fmac_f32_e32 v14, v7, v15                                // 000000001EB0: 761C1F07
	v_add_f32_e32 v20, v21, v20                                // 000000001EB4: 02282915
	v_mul_f32_e32 v21, v6, v6                                  // 000000001EB8: 0A2A0D06
	v_and_b32_e32 v25, 0xffff0000, v8                          // 000000001EBC: 263210FF FFFF0000
	v_lshlrev_b32_e32 v8, 16, v8                               // 000000001EC4: 24101090
	v_mul_f32_e32 v16, v33, v41                                // 000000001EC8: 0A205321
	v_fmac_f32_e32 v21, v14, v14                               // 000000001ECC: 762A1D0E
	v_fmac_f32_e32 v16, v8, v35                                // 000000001ED0: 76204708
	v_mul_f32_e32 v8, v8, v41                                  // 000000001ED4: 0A105308
	v_add_f32_e32 v20, v21, v20                                // 000000001ED8: 02282915
	v_mul_f32_e32 v21, v5, v5                                  // 000000001EDC: 0A2A0B05
	v_fma_f32 v8, v33, v35, -v8                                // 000000001EE0: D1CB0008 84224721
	v_fmac_f32_e32 v21, v13, v13                               // 000000001EE8: 762A1B0D
	v_mul_f32_e32 v7, v25, v40                                 // 000000001EEC: 0A0E5119
	v_add_f32_e32 v20, v21, v20                                // 000000001EF0: 02282915
	v_mul_f32_e32 v21, v8, v8                                  // 000000001EF4: 0A2A1108
	v_mul_f32_e32 v15, v32, v40                                // 000000001EF8: 0A1E5120
	v_fma_f32 v7, v32, v34, -v7                                // 000000001EFC: D1CB0007 841E4520
	v_fmac_f32_e32 v21, v16, v16                               // 000000001F04: 762A2110
	v_fmac_f32_e32 v15, v25, v34                               // 000000001F08: 761E4519
	v_add_f32_e32 v20, v21, v20                                // 000000001F0C: 02282915
	v_mul_f32_e32 v21, v7, v7                                  // 000000001F10: 0A2A0F07
	v_fmac_f32_e32 v21, v15, v15                               // 000000001F14: 762A1F0F
	v_add_f32_e32 v20, v21, v20                                // 000000001F18: 02282915
	ds_swizzle_b32 v21, v20 offset:swizzle(SWAP,4)             // 000000001F1C: D87A101F 15000014
	v_and_b32_e32 v0, 32, v0                                   // 000000001F24: 260000A0
	v_and_or_b32 v0, v18, 3, v0                                // 000000001F28: D2010000 04010712
	v_cmp_eq_u32_e32 vcc, 0, v0                                // 000000001F30: 7D940080
	s_and_b64 s[8:9], s[8:9], vcc                              // 000000001F34: 86886A08
	s_waitcnt lgkmcnt(0)                                       // 000000001F38: BF8CC07F
	v_add_f32_e32 v20, v20, v21                                // 000000001F3C: 02282B14
	ds_swizzle_b32 v21, v20 offset:swizzle(SWAP,2)             // 000000001F40: D87A081F 15000014
	s_waitcnt lgkmcnt(0)                                       // 000000001F48: BF8CC07F
	v_add_f32_e32 v20, v20, v21                                // 000000001F4C: 02282B14
	ds_swizzle_b32 v21, v20 offset:swizzle(SWAP,1)             // 000000001F50: D87A041F 15000014
	s_waitcnt lgkmcnt(0)                                       // 000000001F58: BF8CC07F
	v_add_f32_e32 v20, v20, v21                                // 000000001F5C: 02282B14
	v_mov_b32_e32 v21, s12                                     // 000000001F60: 7E2A020C
	v_fmac_f32_e32 v21, 0x3c000000, v20                        // 000000001F64: 762A28FF 3C000000
	v_rsq_f32_e32 v21, v21                                     // 000000001F6C: 7E2A4915
	v_lshrrev_b32_e32 v20, 1, v23                              // 000000001F70: 20282E81
	v_add_u32_e32 v20, 0, v20                                  // 000000001F74: 68282880
	ds_write_b32 v20, v21                                      // 000000001F78: D81A0000 00001514
	s_waitcnt lgkmcnt(0)                                       // 000000001F80: BF8CC07F
	s_barrier                                                  // 000000001F84: BF8A0000
	s_and_saveexec_b64 s[2:3], s[8:9]                          // 000000001F88: BE822008
	s_cbranch_execnz 4                                         // 000000001F8C: BF890004 <c662_qkrope_fwd+0x6a0>
	s_or_b64 exec, exec, s[2:3]                                // 000000001F90: 87FE027E
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001F94: BE822000
	s_cbranch_execnz 17                                        // 000000001F98: BF890011 <c662_qkrope_fwd+0x6e0>
	s_endpgm                                                   // 000000001F9C: BF810000
	v_lshl_add_u32 v20, v24, 2, 0                              // 000000001FA0: D1FD0014 02010518
	v_ashrrev_i32_e32 v18, 31, v17                             // 000000001FA8: 2224229F
	ds_read_b32 v20, v20                                       // 000000001FAC: D86C0000 14000014
	v_lshlrev_b64 v[17:18], 2, v[17:18]                        // 000000001FB4: D28F0011 00022282
	v_mov_b32_e32 v0, s11                                      // 000000001FBC: 7E00020B
	v_add_co_u32_e32 v17, vcc, s10, v17                        // 000000001FC0: 3222220A
	v_addc_co_u32_e32 v18, vcc, v0, v18, vcc                   // 000000001FC4: 38242500
	s_waitcnt lgkmcnt(0)                                       // 000000001FC8: BF8CC07F
	global_store_dword v[17:18], v20, off                      // 000000001FCC: DC708000 007F1411
	s_or_b64 exec, exec, s[2:3]                                // 000000001FD4: 87FE027E
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001FD8: BE822000
	s_cbranch_execz 65519                                      // 000000001FDC: BF88FFEF <c662_qkrope_fwd+0x69c>
	s_load_dword s0, s[4:5], 0x5c                              // 000000001FE0: C0020002 0000005C
	v_ashrrev_i32_e32 v20, 31, v19                             // 000000001FE8: 2228269F
	v_lshlrev_b64 v[17:18], 1, v[19:20]                        // 000000001FEC: D28F0011 00022681
	v_mov_b32_e32 v0, s7                                       // 000000001FF4: 7E000207
	v_add_co_u32_e32 v17, vcc, s6, v17                         // 000000001FF8: 32222206
	s_waitcnt lgkmcnt(0)                                       // 000000001FFC: BF8CC07F
	v_mul_f32_e32 v19, s0, v21                                 // 000000002000: 0A262A00
	v_addc_co_u32_e32 v18, vcc, v0, v18, vcc                   // 000000002004: 38242500
	v_mul_f32_e32 v0, v9, v19                                  // 000000002008: 0A002709
	v_mul_f32_e32 v9, v10, v19                                 // 00000000200C: 0A12270A
	v_mul_f32_e32 v10, v11, v19                                // 000000002010: 0A14270B
	v_mul_f32_e32 v11, v13, v19                                // 000000002014: 0A16270D
	v_mul_f32_e32 v13, v14, v19                                // 000000002018: 0A1A270E
	v_mul_f32_e32 v14, v16, v19                                // 00000000201C: 0A1C2710
	v_mul_f32_e32 v20, v12, v19                                // 000000002020: 0A28270C
	v_mul_f32_e32 v12, v15, v19                                // 000000002024: 0A18270F
	v_bfe_u32 v15, v14, 16, 1                                  // 000000002028: D1C8000F 0205210E
	s_movk_i32 s0, 0x7fff                                      // 000000002030: B0007FFF
	v_bfe_u32 v16, v12, 16, 1                                  // 000000002034: D1C80010 0205210C
	v_add3_u32 v15, v14, v15, s0                               // 00000000203C: D1FF000F 00021F0E
	v_bfe_u32 v21, v13, 16, 1                                  // 000000002044: D1C80015 0205210D
	v_add3_u32 v16, v12, v16, s0                               // 00000000204C: D1FF0010 0002210C
	v_lshrrev_b32_e32 v15, 16, v15                             // 000000002054: 201E1E90
	v_mov_b32_e32 v27, 0x7fff                                  // 000000002058: 7E3602FF 00007FFF
	v_cmp_o_f32_e32 vcc, v14, v14                              // 000000002060: 7C8E1D0E
	v_bfe_u32 v22, v11, 16, 1                                  // 000000002064: D1C80016 0205210B
	v_add3_u32 v21, v13, v21, s0                               // 00000000206C: D1FF0015 00022B0D
	v_lshrrev_b32_e32 v16, 16, v16                             // 000000002074: 20202090
	v_cndmask_b32_e32 v14, v27, v15, vcc                       // 000000002078: 001C1F1B
	v_cmp_o_f32_e32 vcc, v12, v12                              // 00000000207C: 7C8E190C
	v_bfe_u32 v23, v20, 16, 1                                  // 000000002080: D1C80017 02052114
	v_add3_u32 v22, v11, v22, s0                               // 000000002088: D1FF0016 00022D0B
	v_lshrrev_b32_e32 v21, 16, v21                             // 000000002090: 202A2A90
	v_cndmask_b32_e32 v12, v27, v16, vcc                       // 000000002094: 0018211B
	v_cmp_o_f32_e32 vcc, v13, v13                              // 000000002098: 7C8E1B0D
	v_bfe_u32 v24, v10, 16, 1                                  // 00000000209C: D1C80018 0205210A
	v_add3_u32 v23, v20, v23, s0                               // 0000000020A4: D1FF0017 00022F14
	v_lshrrev_b32_e32 v22, 16, v22                             // 0000000020AC: 202C2C90
	v_cndmask_b32_e32 v13, v27, v21, vcc                       // 0000000020B0: 001A2B1B
	v_cmp_o_f32_e32 vcc, v11, v11                              // 0000000020B4: 7C8E170B
	v_bfe_u32 v25, v9, 16, 1                                   // 0000000020B8: D1C80019 02052109
	v_add3_u32 v24, v10, v24, s0                               // 0000000020C0: D1FF0018 0002310A
	v_lshrrev_b32_e32 v23, 16, v23                             // 0000000020C8: 202E2E90
	s_mov_b32 s1, 0x5040100                                    // 0000000020CC: BE8100FF 05040100
	v_cndmask_b32_e32 v11, v27, v22, vcc                       // 0000000020D4: 00162D1B
	v_cmp_o_f32_e32 vcc, v20, v20                              // 0000000020D8: 7C8E2914
	v_bfe_u32 v26, v0, 16, 1                                   // 0000000020DC: D1C8001A 02052100
	v_add3_u32 v25, v9, v25, s0                                // 0000000020E4: D1FF0019 00023309
	v_lshrrev_b32_e32 v24, 16, v24                             // 0000000020EC: 20303090
	v_perm_b32 v11, v11, v13, s1                               // 0000000020F0: D1ED000B 00061B0B
	v_cndmask_b32_e32 v13, v27, v23, vcc                       // 0000000020F8: 001A2F1B
	v_cmp_o_f32_e32 vcc, v10, v10                              // 0000000020FC: 7C8E150A
	v_add3_u32 v26, v0, v26, s0                                // 000000002100: D1FF001A 00023500
	v_lshrrev_b32_e32 v25, 16, v25                             // 000000002108: 20323290
	v_cndmask_b32_e32 v10, v27, v24, vcc                       // 00000000210C: 0014311B
	v_cmp_o_f32_e32 vcc, v9, v9                                // 000000002110: 7C8E1309
	v_lshrrev_b32_e32 v26, 16, v26                             // 000000002114: 20343490
	v_cndmask_b32_e32 v9, v27, v25, vcc                        // 000000002118: 0012331B
	v_cmp_o_f32_e32 vcc, v0, v0                                // 00000000211C: 7C8E0100
	v_cndmask_b32_e32 v0, v27, v26, vcc                        // 000000002120: 0000351B
	v_perm_b32 v12, v12, v14, s1                               // 000000002124: D1ED000C 00061D0C
	v_perm_b32 v10, v10, v13, s1                               // 00000000212C: D1ED000A 00061B0A
	v_perm_b32 v9, v0, v9, s1                                  // 000000002134: D1ED0009 00061300
	global_store_dwordx4 v[17:18], v[9:12], off                // 00000000213C: DC7C8000 007F0911
	v_mul_f32_e32 v0, v1, v19                                  // 000000002144: 0A002701
	v_mul_f32_e32 v9, v2, v19                                  // 000000002148: 0A122702
	v_mul_f32_e32 v2, v5, v19                                  // 00000000214C: 0A042705
	v_mul_f32_e32 v5, v6, v19                                  // 000000002150: 0A0A2706
	v_mul_f32_e32 v6, v8, v19                                  // 000000002154: 0A0C2708
	v_mul_f32_e32 v1, v3, v19                                  // 000000002158: 0A022703
	v_mul_f32_e32 v3, v7, v19                                  // 00000000215C: 0A062707
	v_bfe_u32 v7, v6, 16, 1                                    // 000000002160: D1C80007 02052106
	v_bfe_u32 v8, v3, 16, 1                                    // 000000002168: D1C80008 02052103
	v_add3_u32 v7, v6, v7, s0                                  // 000000002170: D1FF0007 00020F06
	v_cmp_o_f32_e32 vcc, v6, v6                                // 000000002178: 7C8E0D06
	v_bfe_u32 v10, v5, 16, 1                                   // 00000000217C: D1C8000A 02052105
	v_add3_u32 v8, v3, v8, s0                                  // 000000002184: D1FF0008 00021103
	v_cndmask_b32_sdwa v6, v27, v7, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000218C: 000C0EF9 0506061B
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000002194: 7C8E0703
	v_mul_f32_e32 v4, v4, v19                                  // 000000002198: 0A082704
	v_bfe_u32 v11, v2, 16, 1                                   // 00000000219C: D1C8000B 02052102
	v_add3_u32 v10, v5, v10, s0                                // 0000000021A4: D1FF000A 00021505
	v_cndmask_b32_sdwa v3, v27, v8, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000021AC: 000610F9 0506061B
	v_cmp_o_f32_e32 vcc, v5, v5                                // 0000000021B4: 7C8E0B05
	v_bfe_u32 v12, v4, 16, 1                                   // 0000000021B8: D1C8000C 02052104
	v_add3_u32 v11, v2, v11, s0                                // 0000000021C0: D1FF000B 00021702
	v_cndmask_b32_sdwa v5, v27, v10, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000021C8: 000A14F9 0506061B
	v_cmp_o_f32_e32 vcc, v2, v2                                // 0000000021D0: 7C8E0502
	v_bfe_u32 v13, v1, 16, 1                                   // 0000000021D4: D1C8000D 02052101
	v_add3_u32 v12, v4, v12, s0                                // 0000000021DC: D1FF000C 00021904
	v_cndmask_b32_sdwa v2, v27, v11, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000021E4: 000416F9 0506061B
	v_cmp_o_f32_e32 vcc, v4, v4                                // 0000000021EC: 7C8E0904
	v_bfe_u32 v14, v9, 16, 1                                   // 0000000021F0: D1C8000E 02052109
	v_add3_u32 v13, v1, v13, s0                                // 0000000021F8: D1FF000D 00021B01
	v_cndmask_b32_sdwa v4, v27, v12, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002200: 000818F9 0506061B
	v_cmp_o_f32_e32 vcc, v1, v1                                // 000000002208: 7C8E0301
	v_bfe_u32 v15, v0, 16, 1                                   // 00000000220C: D1C8000F 02052100
	v_add3_u32 v14, v9, v14, s0                                // 000000002214: D1FF000E 00021D09
	v_cndmask_b32_sdwa v1, v27, v13, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000221C: 00021AF9 0506061B
	v_cmp_o_f32_e32 vcc, v9, v9                                // 000000002224: 7C8E1309
	v_add3_u32 v15, v0, v15, s0                                // 000000002228: D1FF000F 00021F00
	v_perm_b32 v1, v1, v4, s1                                  // 000000002230: D1ED0001 00060901
	v_cndmask_b32_sdwa v4, v27, v14, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002238: 00081CF9 0506061B
	v_cmp_o_f32_e32 vcc, v0, v0                                // 000000002240: 7C8E0100
	v_cndmask_b32_sdwa v0, v27, v15, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002244: 00001EF9 0506061B
	v_perm_b32 v3, v3, v6, s1                                  // 00000000224C: D1ED0003 00060D03
	v_perm_b32 v2, v2, v5, s1                                  // 000000002254: D1ED0002 00060B02
	v_perm_b32 v0, v0, v4, s1                                  // 00000000225C: D1ED0000 00060900
	global_store_dwordx4 v[17:18], v[0:3], off offset:128      // 000000002264: DC7C8080 007F0011
	s_endpgm                                                   // 00000000226C: BF810000
