
/u/tmahmood1/project/nanochat/cisc662/results/isa/qkrope/triton_v1/gfx90a/xc-nw4_wpe1_h1/c662_qkrope_fwd.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001900 <c662_qkrope_fwd>:
	s_load_dwordx2 s[6:7], s[4:5], 0x0                         // 000000001900: C0060182 00000000
	s_load_dwordx8 s[8:15], s[4:5], 0x8                        // 000000001908: C00E0202 00000008
	s_waitcnt lgkmcnt(0)                                       // 000000001910: BF8CC07F
	s_branch 58                                                // 000000001914: BF82003A <c662_qkrope_fwd+0x100>
	s_nop 0                                                    // 000000001918: BF800000
	s_nop 0                                                    // 00000000191C: BF800000
	s_nop 0                                                    // 000000001920: BF800000
	s_nop 0                                                    // 000000001924: BF800000
	s_nop 0                                                    // 000000001928: BF800000
	s_nop 0                                                    // 00000000192C: BF800000
	s_nop 0                                                    // 000000001930: BF800000
	s_nop 0                                                    // 000000001934: BF800000
	s_nop 0                                                    // 000000001938: BF800000
	s_nop 0                                                    // 00000000193C: BF800000
	s_nop 0                                                    // 000000001940: BF800000
	s_nop 0                                                    // 000000001944: BF800000
	s_nop 0                                                    // 000000001948: BF800000
	s_nop 0                                                    // 00000000194C: BF800000
	s_nop 0                                                    // 000000001950: BF800000
	s_nop 0                                                    // 000000001954: BF800000
	s_nop 0                                                    // 000000001958: BF800000
	s_nop 0                                                    // 00000000195C: BF800000
	s_nop 0                                                    // 000000001960: BF800000
	s_nop 0                                                    // 000000001964: BF800000
	s_nop 0                                                    // 000000001968: BF800000
	s_nop 0                                                    // 00000000196C: BF800000
	s_nop 0                                                    // 000000001970: BF800000
	s_nop 0                                                    // 000000001974: BF800000
	s_nop 0                                                    // 000000001978: BF800000
	s_nop 0                                                    // 00000000197C: BF800000
	s_nop 0                                                    // 000000001980: BF800000
	s_nop 0                                                    // 000000001984: BF800000
	s_nop 0                                                    // 000000001988: BF800000
	s_nop 0                                                    // 00000000198C: BF800000
	s_nop 0                                                    // 000000001990: BF800000
	s_nop 0                                                    // 000000001994: BF800000
	s_nop 0                                                    // 000000001998: BF800000
	s_nop 0                                                    // 00000000199C: BF800000
	s_nop 0                                                    // 0000000019A0: BF800000
	s_nop 0                                                    // 0000000019A4: BF800000
	s_nop 0                                                    // 0000000019A8: BF800000
	s_nop 0                                                    // 0000000019AC: BF800000
	s_nop 0                                                    // 0000000019B0: BF800000
	s_nop 0                                                    // 0000000019B4: BF800000
	s_nop 0                                                    // 0000000019B8: BF800000
	s_nop 0                                                    // 0000000019BC: BF800000
	s_nop 0                                                    // 0000000019C0: BF800000
	s_nop 0                                                    // 0000000019C4: BF800000
	s_nop 0                                                    // 0000000019C8: BF800000
	s_nop 0                                                    // 0000000019CC: BF800000
	s_nop 0                                                    // 0000000019D0: BF800000
	s_nop 0                                                    // 0000000019D4: BF800000
	s_nop 0                                                    // 0000000019D8: BF800000
	s_nop 0                                                    // 0000000019DC: BF800000
	s_nop 0                                                    // 0000000019E0: BF800000
	s_nop 0                                                    // 0000000019E4: BF800000
	s_nop 0                                                    // 0000000019E8: BF800000
	s_nop 0                                                    // 0000000019EC: BF800000
	s_nop 0                                                    // 0000000019F0: BF800000
	s_nop 0                                                    // 0000000019F4: BF800000
	s_nop 0                                                    // 0000000019F8: BF800000
	s_nop 0                                                    // 0000000019FC: BF800000
	s_load_dword s0, s[4:5], 0x48                              // 000000001A00: C0020002 00000048
	s_load_dwordx2 s[20:21], s[4:5], 0x54                      // 000000001A08: C0060502 00000054
	v_lshrrev_b32_e32 v7, 6, v0                                // 000000001A10: 200E0086
	v_lshrrev_b32_e32 v11, 4, v0                               // 000000001A14: 20160084
	v_and_b32_e32 v10, 15, v0                                  // 000000001A18: 2614008F
	s_waitcnt lgkmcnt(0)                                       // 000000001A1C: BF8CC07F
	s_cmp_ge_i32 s16, s0                                       // 000000001A20: BF030010
	s_cbranch_scc0 103                                         // 000000001A24: BF840067 <c662_qkrope_fwd+0x2c4>
	s_load_dword s1, s[4:5], 0x44                              // 000000001A28: C0020042 00000044
	s_load_dword s2, s[4:5], 0x50                              // 000000001A30: C0020082 00000050
	s_sub_i32 s0, s16, s0                                      // 000000001A38: 81800010
	s_lshl_b32 s3, s0, 4                                       // 000000001A3C: 8E038400
	v_or_b32_e32 v4, s3, v11                                   // 000000001A40: 28081603
	v_or_b32_e32 v2, s3, v10                                   // 000000001A44: 28041403
	s_waitcnt lgkmcnt(0)                                       // 000000001A48: BF8CC07F
	s_abs_i32 s17, s2                                          // 000000001A4C: BE913002
	v_cvt_f32_u32_e32 v5, s17                                  // 000000001A50: 7E0A0C11
	s_bfe_i32 s0, s0, 0x1001b                                  // 000000001A54: 9300FF00 0001001B
	v_cmp_gt_i32_e32 vcc, s1, v4                               // 000000001A5C: 7D880801
	v_cmp_gt_i32_e64 s[18:19], s1, v2                          // 000000001A60: D0C40012 00020401
	v_rcp_iflag_f32_e32 v5, v5                                 // 000000001A68: 7E0A4705
	s_ashr_i32 s1, s2, 31                                      // 000000001A6C: 90019F02
	v_add_u32_e32 v6, s0, v4                                   // 000000001A70: 680C0800
	s_xor_b32 s2, s0, s1                                       // 000000001A74: 88020100
	v_mul_f32_e32 v5, 0x4f7ffffe, v5                           // 000000001A78: 0A0A0AFF 4F7FFFFE
	v_cvt_u32_f32_e32 v5, v5                                   // 000000001A80: 7E0A0F05
	v_xor_b32_e32 v6, s0, v6                                   // 000000001A84: 2A0C0C00
	s_sub_i32 s0, 0, s17                                       // 000000001A88: 81801180
	s_abs_i32 s3, s20                                          // 000000001A8C: BE833014
	v_mul_lo_u32 v8, s0, v5                                    // 000000001A90: D2850008 00020A00
	v_mul_hi_u32 v8, v5, v8                                    // 000000001A98: D2860008 00021105
	v_add_u32_e32 v5, v5, v8                                   // 000000001AA0: 680A1105
	v_mul_hi_u32 v5, v6, v5                                    // 000000001AA4: D2860005 00020B06
	v_mul_lo_u32 v8, v5, s17                                   // 000000001AAC: D2850008 00002305
	v_sub_u32_e32 v6, v6, v8                                   // 000000001AB4: 6A0C1106
	v_add_u32_e32 v8, 1, v5                                    // 000000001AB8: 68100A81
	v_cmp_le_u32_e64 s[0:1], s17, v6                           // 000000001ABC: D0CB0000 00020C11
	v_cndmask_b32_e64 v5, v5, v8, s[0:1]                       // 000000001AC4: D1000005 00021105
	v_subrev_u32_e32 v8, s17, v6                               // 000000001ACC: 6C100C11
	v_cndmask_b32_e64 v6, v6, v8, s[0:1]                       // 000000001AD0: D1000006 00021106
	v_cvt_f32_u32_e32 v8, s3                                   // 000000001AD8: 7E100C03
	v_cmp_le_u32_e64 s[0:1], s17, v6                           // 000000001ADC: D0CB0000 00020C11
	v_add_u32_e32 v9, 1, v5                                    // 000000001AE4: 68120A81
	v_cndmask_b32_e64 v5, v5, v9, s[0:1]                       // 000000001AE8: D1000005 00021305
	v_rcp_iflag_f32_e32 v6, v8                                 // 000000001AF0: 7E0C4708
	v_xor_b32_e32 v5, s2, v5                                   // 000000001AF4: 2A0A0A02
	v_subrev_u32_e32 v5, s2, v5                                // 000000001AF8: 6C0A0A02
	v_sub_u32_e32 v9, 0, v5                                    // 000000001AFC: 6A120A80
	v_mul_f32_e32 v6, 0x4f7ffffe, v6                           // 000000001B00: 0A0C0CFF 4F7FFFFE
	v_cvt_u32_f32_e32 v6, v6                                   // 000000001B08: 7E0C0F06
	s_sub_i32 s0, 0, s3                                        // 000000001B0C: 81800380
	v_ashrrev_i32_e32 v8, 31, v5                               // 000000001B10: 22100A9F
	v_max_i32_e32 v5, v5, v9                                   // 000000001B14: 1A0A1305
	v_mul_lo_u32 v9, s0, v6                                    // 000000001B18: D2850009 00020C00
	v_mul_hi_u32 v9, v6, v9                                    // 000000001B20: D2860009 00021306
	v_add_u32_e32 v6, v6, v9                                   // 000000001B28: 680C1306
	v_mul_hi_u32 v6, v5, v6                                    // 000000001B2C: D2860006 00020D05
	v_mul_lo_u32 v6, v6, s3                                    // 000000001B34: D2850006 00000706
	v_sub_u32_e32 v5, v5, v6                                   // 000000001B3C: 6A0A0D05
	v_subrev_u32_e32 v6, s3, v5                                // 000000001B40: 6C0C0A03
	v_cmp_le_u32_e64 s[0:1], s3, v5                            // 000000001B44: D0CB0000 00020A03
	v_cndmask_b32_e64 v5, v5, v6, s[0:1]                       // 000000001B4C: D1000005 00020D05
	v_subrev_u32_e32 v6, s3, v5                                // 000000001B54: 6C0C0A03
	v_cmp_le_u32_e64 s[0:1], s3, v5                            // 000000001B58: D0CB0000 00020A03
	v_cndmask_b32_e64 v5, v5, v6, s[0:1]                       // 000000001B60: D1000005 00020D05
	v_xor_b32_e32 v5, v5, v8                                   // 000000001B68: 2A0A1105
	v_lshlrev_b32_e32 v6, 2, v10                               // 000000001B6C: 240C1482
	v_sub_u32_e32 v5, v5, v8                                   // 000000001B70: 6A0A1105
	v_lshl_or_b32 v4, v4, 7, v6                                // 000000001B74: D2000004 04190F04
	v_mad_u64_u32 v[8:9], s[0:1], v5, s21, v[6:7]              // 000000001B7C: D1E80008 04182B05
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000001B84: 220A089F
	v_lshlrev_b64 v[12:13], 1, v[4:5]                          // 000000001B88: D28F000C 00020881
	v_mov_b32_e32 v5, s9                                       // 000000001B90: 7E0A0209
	v_add_co_u32_e64 v12, s[0:1], s8, v12                      // 000000001B94: D119000C 00021808
	v_addc_co_u32_e64 v13, s[0:1], v5, v13, s[0:1]             // 000000001B9C: D11C000D 00021B05
	v_lshrrev_b32_e32 v3, 6, v0                                // 000000001BA4: 20060086
	v_and_b32_e32 v1, 0xf0, v0                                 // 000000001BA8: 260200FF 000000F0
	s_and_b64 s[8:9], vcc, exec                                // 000000001BB0: 86887E6A
	s_cbranch_execz 4                                          // 000000001BB4: BF880004 <c662_qkrope_fwd+0x2c8>
	s_load_dwordx2 s[14:15], s[4:5], 0x28                      // 000000001BB8: C0060382 00000028
	s_branch 105                                               // 000000001BC0: BF820069 <c662_qkrope_fwd+0x468>
	s_mov_b64 s[8:9], 0                                        // 000000001BC4: BE880180
	s_load_dword s2, s[4:5], 0x4c                              // 000000001BC8: C0020082 0000004C
	s_load_dword s0, s[4:5], 0x40                              // 000000001BD0: C0020002 00000040
	s_bfe_i32 s3, s16, 0x1001b                                 // 000000001BD8: 9303FF10 0001001B
	s_lshl_b32 s1, s16, 4                                      // 000000001BE0: 8E018410
	v_or_b32_e32 v4, s1, v11                                   // 000000001BE4: 28081601
	s_waitcnt lgkmcnt(0)                                       // 000000001BE8: BF8CC07F
	s_abs_i32 s12, s2                                          // 000000001BEC: BE8C3002
	v_cvt_f32_u32_e32 v3, s12                                  // 000000001BF0: 7E060C0C
	s_ashr_i32 s2, s2, 31                                      // 000000001BF4: 90029F02
	s_xor_b32 s13, s3, s2                                      // 000000001BF8: 880D0203
	s_sub_i32 s2, 0, s12                                       // 000000001BFC: 81820C80
	v_rcp_iflag_f32_e32 v3, v3                                 // 000000001C00: 7E064703
	v_add_u32_e32 v5, s3, v4                                   // 000000001C04: 680A0803
	v_xor_b32_e32 v5, s3, v5                                   // 000000001C08: 2A0A0A03
	s_abs_i32 s16, s20                                         // 000000001C0C: BE903014
	v_mul_f32_e32 v3, 0x4f7ffffe, v3                           // 000000001C10: 0A0606FF 4F7FFFFE
	v_cvt_u32_f32_e32 v3, v3                                   // 000000001C18: 7E060F03
	v_cmp_gt_i32_e32 vcc, s0, v4                               // 000000001C1C: 7D880800
	v_or_b32_e32 v2, s1, v10                                   // 000000001C20: 28041401
	v_cmp_gt_i32_e64 s[0:1], s0, v2                            // 000000001C24: D0C40000 00020400
	v_mul_lo_u32 v6, s2, v3                                    // 000000001C2C: D2850006 00020602
	v_mul_hi_u32 v6, v3, v6                                    // 000000001C34: D2860006 00020D03
	v_add_u32_e32 v3, v3, v6                                   // 000000001C3C: 68060D03
	v_mul_hi_u32 v3, v5, v3                                    // 000000001C40: D2860003 00020705
	v_mul_lo_u32 v6, v3, s12                                   // 000000001C48: D2850006 00001903
	v_sub_u32_e32 v5, v5, v6                                   // 000000001C50: 6A0A0D05
	v_add_u32_e32 v6, 1, v3                                    // 000000001C54: 680C0681
	v_cmp_le_u32_e64 s[2:3], s12, v5                           // 000000001C58: D0CB0002 00020A0C
	v_cndmask_b32_e64 v3, v3, v6, s[2:3]                       // 000000001C60: D1000003 000A0D03
	v_subrev_u32_e32 v6, s12, v5                               // 000000001C68: 6C0C0A0C
	v_cndmask_b32_e64 v5, v5, v6, s[2:3]                       // 000000001C6C: D1000005 000A0D05
	v_cvt_f32_u32_e32 v6, s16                                  // 000000001C74: 7E0C0C10
	v_cmp_le_u32_e64 s[2:3], s12, v5                           // 000000001C78: D0CB0002 00020A0C
	v_add_u32_e32 v8, 1, v3                                    // 000000001C80: 68100681
	v_cndmask_b32_e64 v3, v3, v8, s[2:3]                       // 000000001C84: D1000003 000A1103
	v_rcp_iflag_f32_e32 v5, v6                                 // 000000001C8C: 7E0A4706
	v_xor_b32_e32 v3, s13, v3                                  // 000000001C90: 2A06060D
	v_subrev_u32_e32 v3, s13, v3                               // 000000001C94: 6C06060D
	v_sub_u32_e32 v8, 0, v3                                    // 000000001C98: 6A100680
	v_mul_f32_e32 v5, 0x4f7ffffe, v5                           // 000000001C9C: 0A0A0AFF 4F7FFFFE
	v_cvt_u32_f32_e32 v5, v5                                   // 000000001CA4: 7E0A0F05
	s_sub_i32 s2, 0, s16                                       // 000000001CA8: 81821080
	v_ashrrev_i32_e32 v6, 31, v3                               // 000000001CAC: 220C069F
	v_max_i32_e32 v3, v3, v8                                   // 000000001CB0: 1A061103
	v_mul_lo_u32 v8, s2, v5                                    // 000000001CB4: D2850008 00020A02
	v_mul_hi_u32 v8, v5, v8                                    // 000000001CBC: D2860008 00021105
	v_add_u32_e32 v5, v5, v8                                   // 000000001CC4: 680A1105
	v_mul_hi_u32 v5, v3, v5                                    // 000000001CC8: D2860005 00020B03
	v_mul_lo_u32 v5, v5, s16                                   // 000000001CD0: D2850005 00002105
	v_sub_u32_e32 v3, v3, v5                                   // 000000001CD8: 6A060B03
	v_subrev_u32_e32 v5, s16, v3                               // 000000001CDC: 6C0A0610
	v_cmp_le_u32_e64 s[2:3], s16, v3                           // 000000001CE0: D0CB0002 00020610
	v_cndmask_b32_e64 v3, v3, v5, s[2:3]                       // 000000001CE8: D1000003 000A0B03
	v_subrev_u32_e32 v5, s16, v3                               // 000000001CF0: 6C0A0610
	v_cmp_le_u32_e64 s[2:3], s16, v3                           // 000000001CF4: D0CB0002 00020610
	v_cndmask_b32_e64 v3, v3, v5, s[2:3]                       // 000000001CFC: D1000003 000A0B03
	v_xor_b32_e32 v3, v3, v6                                   // 000000001D04: 2A060D03
	v_sub_u32_e32 v3, v3, v6                                   // 000000001D08: 6A060D03
	v_lshlrev_b32_e32 v6, 2, v10                               // 000000001D0C: 240C1482
	v_lshl_or_b32 v4, v4, 7, v6                                // 000000001D10: D2000004 04190F04
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000001D18: 220A089F
	v_mad_u64_u32 v[8:9], s[2:3], v3, s21, v[6:7]              // 000000001D1C: D1E80208 04182B03
	v_lshlrev_b64 v[10:11], 1, v[4:5]                          // 000000001D24: D28F000A 00020881
	v_mov_b32_e32 v3, s7                                       // 000000001D2C: 7E060207
	v_add_co_u32_e64 v12, s[2:3], s6, v10                      // 000000001D30: D119020C 00021406
	v_addc_co_u32_e64 v13, s[2:3], v3, v11, s[2:3]             // 000000001D38: D11C020D 000A1703
	s_andn2_b64 s[2:3], s[18:19], exec                         // 000000001D40: 89827E12
	s_and_b64 s[0:1], s[0:1], exec                             // 000000001D44: 86807E00
	s_or_b64 s[18:19], s[2:3], s[0:1]                          // 000000001D48: 87920002
	s_andn2_b64 s[0:1], s[8:9], exec                           // 000000001D4C: 89807E08
	s_and_b64 s[2:3], vcc, exec                                // 000000001D50: 86827E6A
	v_and_b32_e32 v1, 0xf0, v0                                 // 000000001D54: 260200FF 000000F0
	s_or_b64 s[8:9], s[0:1], s[2:3]                            // 000000001D5C: 87880200
	s_mov_b64 s[12:13], s[10:11]                               // 000000001D60: BE8C010A
	v_mov_b32_e32 v3, v7                                       // 000000001D64: 7E060307
	v_mov_b32_e32 v10, 0                                       // 000000001D68: 7E140280
	s_mov_b64 s[0:1], 0                                        // 000000001D6C: BE800180
	v_mov_b32_e32 v14, 0                                       // 000000001D70: 7E1C0280
	v_mov_b32_e32 v15, 0                                       // 000000001D74: 7E1E0280
	s_and_saveexec_b64 s[2:3], s[8:9]                          // 000000001D78: BE822008
	s_cbranch_execz 3                                          // 000000001D7C: BF880003 <c662_qkrope_fwd+0x48c>
	global_load_dwordx2 v[14:15], v[12:13], off                // 000000001D80: DC548000 0E7F000C
	s_mov_b64 s[0:1], exec                                     // 000000001D88: BE80017E
	s_or_b64 exec, exec, s[2:3]                                // 000000001D8C: 87FE027E
	v_mov_b32_e32 v11, 0                                       // 000000001D90: 7E160280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001D94: BE822000
	s_cbranch_execz 2                                          // 000000001D98: BF880002 <c662_qkrope_fwd+0x4a4>
	global_load_dwordx2 v[10:11], v[12:13], off offset:128     // 000000001D9C: DC548080 0A7F000C
	s_or_b64 exec, exec, s[2:3]                                // 000000001DA4: 87FE027E
	s_load_dword s6, s[4:5], 0x60                              // 000000001DA8: C0020182 00000060
	v_mov_b32_e32 v16, 0                                       // 000000001DB0: 7E200280
	v_ashrrev_i32_e32 v9, 31, v8                               // 000000001DB4: 2212109F
	v_mov_b32_e32 v18, 0                                       // 000000001DB8: 7E240280
	v_mov_b32_e32 v19, 0                                       // 000000001DBC: 7E260280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001DC0: BE822000
	s_cbranch_execz 10                                         // 000000001DC4: BF88000A <c662_qkrope_fwd+0x4f0>
	s_load_dwordx2 s[8:9], s[4:5], 0x30                        // 000000001DC8: C0060202 00000030
	v_lshlrev_b64 v[12:13], 1, v[8:9]                          // 000000001DD0: D28F000C 00021081
	s_waitcnt lgkmcnt(0)                                       // 000000001DD8: BF8CC07F
	v_mov_b32_e32 v5, s9                                       // 000000001DDC: 7E0A0209
	v_add_co_u32_e32 v12, vcc, s8, v12                         // 000000001DE0: 32181808
	v_addc_co_u32_e32 v13, vcc, v5, v13, vcc                   // 000000001DE4: 381A1B05
	global_load_dwordx2 v[18:19], v[12:13], off                // 000000001DE8: DC548000 127F000C
	s_or_b64 exec, exec, s[2:3]                                // 000000001DF0: 87FE027E
	v_mov_b32_e32 v17, 0                                       // 000000001DF4: 7E220280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001DF8: BE822000
	s_cbranch_execz 10                                         // 000000001DFC: BF88000A <c662_qkrope_fwd+0x528>
	s_load_dwordx2 s[8:9], s[4:5], 0x38                        // 000000001E00: C0060202 00000038
	v_lshlrev_b64 v[8:9], 1, v[8:9]                            // 000000001E08: D28F0008 00021081
	s_waitcnt lgkmcnt(0)                                       // 000000001E10: BF8CC07F
	v_mov_b32_e32 v5, s9                                       // 000000001E14: 7E0A0209
	v_add_co_u32_e32 v8, vcc, s8, v8                           // 000000001E18: 32101008
	v_addc_co_u32_e32 v9, vcc, v5, v9, vcc                     // 000000001E1C: 38121305
	global_load_dwordx2 v[16:17], v[8:9], off                  // 000000001E20: DC548000 107F0008
	s_or_b64 exec, exec, s[2:3]                                // 000000001E28: 87FE027E
	s_waitcnt vmcnt(0)                                         // 000000001E2C: BF8C0F70
	v_and_b32_e32 v21, 0xffff0000, v10                         // 000000001E30: 262A14FF FFFF0000
	v_lshlrev_b32_e32 v20, 16, v10                             // 000000001E38: 24281490
	v_and_b32_e32 v25, 0xffff0000, v16                         // 000000001E3C: 263220FF FFFF0000
	v_lshlrev_b32_e32 v24, 16, v16                             // 000000001E44: 24302090
	v_and_b32_e32 v9, 0xffff0000, v14                          // 000000001E48: 26121CFF FFFF0000
	v_lshlrev_b32_e32 v8, 16, v14                              // 000000001E50: 24101C90
	v_and_b32_e32 v23, 0xffff0000, v18                         // 000000001E54: 262E24FF FFFF0000
	v_lshlrev_b32_e32 v22, 16, v18                             // 000000001E5C: 242C2490
	v_pk_mul_f32 v[12:13], v[20:21], v[24:25]                  // 000000001E60: D3B1400C 18023114
	v_pk_fma_f32 v[12:13], v[8:9], v[22:23], v[12:13]          // 000000001E68: D3B0400C 1C322D08
	v_and_b32_e32 v27, 0xffff0000, v15                         // 000000001E70: 26361EFF FFFF0000
	v_lshlrev_b32_e32 v26, 16, v15                             // 000000001E78: 24341E90
	v_and_b32_e32 v29, 0xffff0000, v11                         // 000000001E7C: 263A16FF FFFF0000
	v_lshlrev_b32_e32 v28, 16, v11                             // 000000001E84: 24381690
	v_and_b32_e32 v11, 0xffff0000, v19                         // 000000001E88: 261626FF FFFF0000
	v_lshlrev_b32_e32 v10, 16, v19                             // 000000001E90: 24142690
	v_and_b32_e32 v19, 0xffff0000, v17                         // 000000001E94: 262622FF FFFF0000
	v_lshlrev_b32_e32 v18, 16, v17                             // 000000001E9C: 24242290
	v_pk_mul_f32 v[8:9], v[8:9], v[24:25]                      // 000000001EA0: D3B14008 18023108
	v_pk_mul_f32 v[14:15], v[28:29], v[18:19]                  // 000000001EA8: D3B1400E 1802251C
	v_pk_fma_f32 v[8:9], v[20:21], v[22:23], v[8:9] neg_lo:[0,0,1] neg_hi:[0,0,1]// 000000001EB0: D3B04408 9C222D14
	v_pk_mul_f32 v[16:17], v[26:27], v[18:19]                  // 000000001EB8: D3B14010 1802251A
	v_pk_fma_f32 v[14:15], v[26:27], v[10:11], v[14:15]        // 000000001EC0: D3B0400E 1C3A151A
	v_pk_fma_f32 v[10:11], v[28:29], v[10:11], v[16:17] neg_lo:[0,0,1] neg_hi:[0,0,1]// 000000001EC8: D3B0440A 9C42151C
	v_pk_mul_f32 v[16:17], v[8:9], v[8:9]                      // 000000001ED0: D3B14010 18021108
	v_pk_fma_f32 v[16:17], v[12:13], v[12:13], v[16:17]        // 000000001ED8: D3B04010 1C42190C
	v_pk_mul_f32 v[18:19], v[10:11], v[10:11]                  // 000000001EE0: D3B14012 1802150A
	v_pk_fma_f32 v[18:19], v[14:15], v[14:15], v[18:19]        // 000000001EE8: D3B04012 1C4A1D0E
	v_add_f32_e32 v5, v16, v17                                 // 000000001EF0: 020A2310
	v_add_f32_e32 v5, v18, v5                                  // 000000001EF4: 020A0B12
	v_add_f32_e32 v5, v19, v5                                  // 000000001EF8: 020A0B13
	v_mov_b32_e32 v7, v5                                       // 000000001EFC: 7E0E0305
	v_and_b32_e32 v0, 48, v0                                   // 000000001F00: 260000B0
	v_and_or_b32 v0, v3, 3, v0                                 // 000000001F04: D2010000 04010703
	v_mov_b32_dpp v7, v7 row_shr:8 row_mask:0xf bank_mask:0xc  // 000000001F0C: 7E0E02FA FC011807
	v_lshrrev_b32_e32 v1, 2, v1                                // 000000001F14: 20020282
	v_cmp_eq_u32_e32 vcc, 0, v0                                // 000000001F18: 7D940080
	v_mov_b32_dpp v7, v5 row_shl:8 row_mask:0xf bank_mask:0x3  // 000000001F1C: 7E0E02FA F3010805
	v_add_f32_e32 v5, v5, v7                                   // 000000001F24: 020A0F05
	v_mov_b32_e32 v7, v5                                       // 000000001F28: 7E0E0305
	v_add_u32_e32 v1, 0, v1                                    // 000000001F2C: 68020280
	s_nop 0                                                    // 000000001F30: BF800000
	v_mov_b32_dpp v7, v7 row_shr:4 row_mask:0xf bank_mask:0xa  // 000000001F34: 7E0E02FA FA011407
	s_nop 1                                                    // 000000001F3C: BF800001
	v_mov_b32_dpp v7, v5 row_shl:4 row_mask:0xf bank_mask:0x5  // 000000001F40: 7E0E02FA F5010405
	v_add_f32_e32 v5, v5, v7                                   // 000000001F48: 020A0F05
	v_mov_b32_e32 v7, v5                                       // 000000001F4C: 7E0E0305
	s_nop 1                                                    // 000000001F50: BF800001
	v_mov_b32_dpp v7, v7 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000001F54: 7E0E02FA FF004E07
	v_add_f32_e32 v5, v5, v7                                   // 000000001F5C: 020A0F05
	v_mov_b32_e32 v7, v5                                       // 000000001F60: 7E0E0305
	s_nop 1                                                    // 000000001F64: BF800001
	v_mov_b32_dpp v7, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 000000001F68: 7E0E02FA FF00B107
	v_add_f32_e32 v5, v5, v7                                   // 000000001F70: 020A0F05
	s_waitcnt lgkmcnt(0)                                       // 000000001F74: BF8CC07F
	v_mov_b32_e32 v7, s6                                       // 000000001F78: 7E0E0206
	v_fmac_f32_e32 v7, 0x3c000000, v5                          // 000000001F7C: 760E0AFF 3C000000
	v_rsq_f32_e32 v7, v7                                       // 000000001F84: 7E0E4907
	s_and_b64 s[6:7], s[18:19], vcc                            // 000000001F88: 86866A12
	ds_write_b32 v1, v7                                        // 000000001F8C: D81A0000 00000701
	s_waitcnt lgkmcnt(0)                                       // 000000001F94: BF8CC07F
	s_barrier                                                  // 000000001F98: BF8A0000
	s_and_saveexec_b64 s[2:3], s[6:7]                          // 000000001F9C: BE822006
	s_cbranch_execnz 4                                         // 000000001FA0: BF890004 <c662_qkrope_fwd+0x6b4>
	s_or_b64 exec, exec, s[2:3]                                // 000000001FA4: 87FE027E
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001FA8: BE822000
	s_cbranch_execnz 16                                        // 000000001FAC: BF890010 <c662_qkrope_fwd+0x6f0>
	s_endpgm                                                   // 000000001FB0: BF810000
	v_ashrrev_i32_e32 v3, 31, v2                               // 000000001FB4: 2206049F
	v_lshlrev_b64 v[0:1], 2, v[2:3]                            // 000000001FB8: D28F0000 00020482
	v_add_u32_e32 v3, 0, v6                                    // 000000001FC0: 68060C80
	ds_read_b32 v3, v3                                         // 000000001FC4: D86C0000 03000003
	v_mov_b32_e32 v2, s15                                      // 000000001FCC: 7E04020F
	v_add_co_u32_e32 v0, vcc, s14, v0                          // 000000001FD0: 3200000E
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc                     // 000000001FD4: 38020302
	s_waitcnt lgkmcnt(0)                                       // 000000001FD8: BF8CC07F
	global_store_dword v[0:1], v3, off                         // 000000001FDC: DC708000 007F0300
	s_or_b64 exec, exec, s[2:3]                                // 000000001FE4: 87FE027E
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001FE8: BE822000
	s_cbranch_execz 65520                                      // 000000001FEC: BF88FFF0 <c662_qkrope_fwd+0x6b0>
	s_load_dword s0, s[4:5], 0x5c                              // 000000001FF0: C0020002 0000005C
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000001FF8: 220A089F
	v_mov_b32_e32 v3, s13                                      // 000000001FFC: 7E06020D
	v_lshlrev_b64 v[0:1], 1, v[4:5]                            // 000000002000: D28F0000 00020881
	v_add_co_u32_e32 v0, vcc, s12, v0                          // 000000002008: 3200000C
	s_waitcnt lgkmcnt(0)                                       // 00000000200C: BF8CC07F
	v_mul_f32_e32 v2, s0, v7                                   // 000000002010: 0A040E00
	v_pk_mul_f32 v[6:7], v[14:15], v[2:3] op_sel_hi:[1,0]      // 000000002014: D3B14006 0802050E
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc                     // 00000000201C: 38020303
	v_pk_mul_f32 v[4:5], v[12:13], v[2:3] op_sel_hi:[1,0]      // 000000002020: D3B14004 0802050C
	v_bfe_u32 v3, v6, 16, 1                                    // 000000002028: D1C80003 02052106
	s_movk_i32 s0, 0x7fff                                      // 000000002030: B0007FFF
	v_bfe_u32 v12, v7, 16, 1                                   // 000000002034: D1C8000C 02052107
	v_add3_u32 v3, v6, v3, s0                                  // 00000000203C: D1FF0003 00020706
	v_bfe_u32 v13, v4, 16, 1                                   // 000000002044: D1C8000D 02052104
	v_add3_u32 v12, v7, v12, s0                                // 00000000204C: D1FF000C 00021907
	v_lshrrev_b32_e32 v3, 16, v3                               // 000000002054: 20060690
	v_mov_b32_e32 v15, 0x7fff                                  // 000000002058: 7E1E02FF 00007FFF
	v_cmp_o_f32_e32 vcc, v6, v6                                // 000000002060: 7C8E0D06
	v_bfe_u32 v14, v5, 16, 1                                   // 000000002064: D1C8000E 02052105
	v_add3_u32 v13, v4, v13, s0                                // 00000000206C: D1FF000D 00021B04
	v_lshrrev_b32_e32 v12, 16, v12                             // 000000002074: 20181890
	v_cndmask_b32_e32 v3, v15, v3, vcc                         // 000000002078: 0006070F
	v_cmp_o_f32_e32 vcc, v7, v7                                // 00000000207C: 7C8E0F07
	v_add3_u32 v14, v5, v14, s0                                // 000000002080: D1FF000E 00021D05
	v_lshrrev_b32_e32 v13, 16, v13                             // 000000002088: 201A1A90
	v_cndmask_b32_e32 v6, v15, v12, vcc                        // 00000000208C: 000C190F
	s_mov_b32 s1, 0x5040100                                    // 000000002090: BE8100FF 05040100
	v_cmp_o_f32_e32 vcc, v4, v4                                // 000000002098: 7C8E0904
	v_lshrrev_b32_e32 v14, 16, v14                             // 00000000209C: 201C1C90
	v_perm_b32 v7, v6, v3, s1                                  // 0000000020A0: D1ED0007 00060706
	v_cndmask_b32_e32 v3, v15, v13, vcc                        // 0000000020A8: 00061B0F
	v_cmp_o_f32_e32 vcc, v5, v5                                // 0000000020AC: 7C8E0B05
	v_cndmask_b32_e32 v4, v15, v14, vcc                        // 0000000020B0: 00081D0F
	v_perm_b32 v6, v4, v3, s1                                  // 0000000020B4: D1ED0006 00060704
	v_pk_mul_f32 v[4:5], v[8:9], v[2:3] op_sel_hi:[1,0]        // 0000000020BC: D3B14004 08020508
	v_pk_mul_f32 v[2:3], v[10:11], v[2:3] op_sel_hi:[1,0]      // 0000000020C4: D3B14002 0802050A
	global_store_dwordx2 v[0:1], v[6:7], off                   // 0000000020CC: DC748000 007F0600
	v_bfe_u32 v6, v2, 16, 1                                    // 0000000020D4: D1C80006 02052102
	v_bfe_u32 v7, v3, 16, 1                                    // 0000000020DC: D1C80007 02052103
	v_add3_u32 v6, v2, v6, s0                                  // 0000000020E4: D1FF0006 00020D02
	v_cmp_o_f32_e32 vcc, v2, v2                                // 0000000020EC: 7C8E0502
	v_bfe_u32 v8, v4, 16, 1                                    // 0000000020F0: D1C80008 02052104
	v_add3_u32 v7, v3, v7, s0                                  // 0000000020F8: D1FF0007 00020F03
	v_cndmask_b32_sdwa v2, v15, v6, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002100: 00040CF9 0506060F
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000002108: 7C8E0703
	v_bfe_u32 v9, v5, 16, 1                                    // 00000000210C: D1C80009 02052105
	v_add3_u32 v8, v4, v8, s0                                  // 000000002114: D1FF0008 00021104
	v_cndmask_b32_sdwa v3, v15, v7, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000211C: 00060EF9 0506060F
	v_cmp_o_f32_e32 vcc, v4, v4                                // 000000002124: 7C8E0904
	v_add3_u32 v9, v5, v9, s0                                  // 000000002128: D1FF0009 00021305
	v_perm_b32 v3, v3, v2, s1                                  // 000000002130: D1ED0003 00060503
	v_cndmask_b32_sdwa v2, v15, v8, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002138: 000410F9 0506060F
	v_cmp_o_f32_e32 vcc, v5, v5                                // 000000002140: 7C8E0B05
	v_cndmask_b32_sdwa v4, v15, v9, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002144: 000812F9 0506060F
	v_perm_b32 v2, v4, v2, s1                                  // 00000000214C: D1ED0002 00060504
	global_store_dwordx2 v[0:1], v[2:3], off offset:128        // 000000002154: DC748080 007F0200
	s_endpgm                                                   // 00000000215C: BF810000
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
	s_nop 0                                                    // 000000002300: BF800000
	s_nop 0                                                    // 000000002304: BF800000
	s_nop 0                                                    // 000000002308: BF800000
	s_nop 0                                                    // 00000000230C: BF800000
	s_nop 0                                                    // 000000002310: BF800000
	s_nop 0                                                    // 000000002314: BF800000
	s_nop 0                                                    // 000000002318: BF800000
	s_nop 0                                                    // 00000000231C: BF800000
	s_nop 0                                                    // 000000002320: BF800000
	s_nop 0                                                    // 000000002324: BF800000
	s_nop 0                                                    // 000000002328: BF800000
	s_nop 0                                                    // 00000000232C: BF800000
	s_nop 0                                                    // 000000002330: BF800000
	s_nop 0                                                    // 000000002334: BF800000
	s_nop 0                                                    // 000000002338: BF800000
	s_nop 0                                                    // 00000000233C: BF800000
	s_nop 0                                                    // 000000002340: BF800000
	s_nop 0                                                    // 000000002344: BF800000
	s_nop 0                                                    // 000000002348: BF800000
	s_nop 0                                                    // 00000000234C: BF800000
	s_nop 0                                                    // 000000002350: BF800000
	s_nop 0                                                    // 000000002354: BF800000
	s_nop 0                                                    // 000000002358: BF800000
	s_nop 0                                                    // 00000000235C: BF800000
	s_nop 0                                                    // 000000002360: BF800000
	s_nop 0                                                    // 000000002364: BF800000
	s_nop 0                                                    // 000000002368: BF800000
	s_nop 0                                                    // 00000000236C: BF800000
	s_nop 0                                                    // 000000002370: BF800000
	s_nop 0                                                    // 000000002374: BF800000
	s_nop 0                                                    // 000000002378: BF800000
	s_nop 0                                                    // 00000000237C: BF800000
	s_nop 0                                                    // 000000002380: BF800000
	s_nop 0                                                    // 000000002384: BF800000
	s_nop 0                                                    // 000000002388: BF800000
	s_nop 0                                                    // 00000000238C: BF800000
	s_nop 0                                                    // 000000002390: BF800000
	s_nop 0                                                    // 000000002394: BF800000
	s_nop 0                                                    // 000000002398: BF800000
	s_nop 0                                                    // 00000000239C: BF800000
	s_nop 0                                                    // 0000000023A0: BF800000
	s_nop 0                                                    // 0000000023A4: BF800000
	s_nop 0                                                    // 0000000023A8: BF800000
	s_nop 0                                                    // 0000000023AC: BF800000
	s_nop 0                                                    // 0000000023B0: BF800000
	s_nop 0                                                    // 0000000023B4: BF800000
	s_nop 0                                                    // 0000000023B8: BF800000
	s_nop 0                                                    // 0000000023BC: BF800000
	s_nop 0                                                    // 0000000023C0: BF800000
	s_nop 0                                                    // 0000000023C4: BF800000
	s_nop 0                                                    // 0000000023C8: BF800000
	s_nop 0                                                    // 0000000023CC: BF800000
	s_nop 0                                                    // 0000000023D0: BF800000
	s_nop 0                                                    // 0000000023D4: BF800000
	s_nop 0                                                    // 0000000023D8: BF800000
	s_nop 0                                                    // 0000000023DC: BF800000
	s_nop 0                                                    // 0000000023E0: BF800000
	s_nop 0                                                    // 0000000023E4: BF800000
	s_nop 0                                                    // 0000000023E8: BF800000
	s_nop 0                                                    // 0000000023EC: BF800000
	s_nop 0                                                    // 0000000023F0: BF800000
	s_nop 0                                                    // 0000000023F4: BF800000
	s_nop 0                                                    // 0000000023F8: BF800000
	s_nop 0                                                    // 0000000023FC: BF800000
	s_nop 0                                                    // 000000002400: BF800000
	s_nop 0                                                    // 000000002404: BF800000
	s_nop 0                                                    // 000000002408: BF800000
	s_nop 0                                                    // 00000000240C: BF800000
	s_nop 0                                                    // 000000002410: BF800000
	s_nop 0                                                    // 000000002414: BF800000
	s_nop 0                                                    // 000000002418: BF800000
	s_nop 0                                                    // 00000000241C: BF800000
	s_nop 0                                                    // 000000002420: BF800000
	s_nop 0                                                    // 000000002424: BF800000
	s_nop 0                                                    // 000000002428: BF800000
	s_nop 0                                                    // 00000000242C: BF800000
	s_nop 0                                                    // 000000002430: BF800000
	s_nop 0                                                    // 000000002434: BF800000
	s_nop 0                                                    // 000000002438: BF800000
	s_nop 0                                                    // 00000000243C: BF800000
	s_nop 0                                                    // 000000002440: BF800000
	s_nop 0                                                    // 000000002444: BF800000
	s_nop 0                                                    // 000000002448: BF800000
	s_nop 0                                                    // 00000000244C: BF800000
	s_nop 0                                                    // 000000002450: BF800000
	s_nop 0                                                    // 000000002454: BF800000
	s_nop 0                                                    // 000000002458: BF800000
	s_nop 0                                                    // 00000000245C: BF800000
	s_nop 0                                                    // 000000002460: BF800000
	s_nop 0                                                    // 000000002464: BF800000
	s_nop 0                                                    // 000000002468: BF800000
	s_nop 0                                                    // 00000000246C: BF800000
	s_nop 0                                                    // 000000002470: BF800000
	s_nop 0                                                    // 000000002474: BF800000
	s_nop 0                                                    // 000000002478: BF800000
	s_nop 0                                                    // 00000000247C: BF800000
	s_nop 0                                                    // 000000002480: BF800000
	s_nop 0                                                    // 000000002484: BF800000
	s_nop 0                                                    // 000000002488: BF800000
	s_nop 0                                                    // 00000000248C: BF800000
	s_nop 0                                                    // 000000002490: BF800000
	s_nop 0                                                    // 000000002494: BF800000
	s_nop 0                                                    // 000000002498: BF800000
	s_nop 0                                                    // 00000000249C: BF800000
	s_nop 0                                                    // 0000000024A0: BF800000
	s_nop 0                                                    // 0000000024A4: BF800000
	s_nop 0                                                    // 0000000024A8: BF800000
	s_nop 0                                                    // 0000000024AC: BF800000
	s_nop 0                                                    // 0000000024B0: BF800000
	s_nop 0                                                    // 0000000024B4: BF800000
	s_nop 0                                                    // 0000000024B8: BF800000
	s_nop 0                                                    // 0000000024BC: BF800000
	s_nop 0                                                    // 0000000024C0: BF800000
	s_nop 0                                                    // 0000000024C4: BF800000
	s_nop 0                                                    // 0000000024C8: BF800000
	s_nop 0                                                    // 0000000024CC: BF800000
	s_nop 0                                                    // 0000000024D0: BF800000
	s_nop 0                                                    // 0000000024D4: BF800000
	s_nop 0                                                    // 0000000024D8: BF800000
	s_nop 0                                                    // 0000000024DC: BF800000
	s_nop 0                                                    // 0000000024E0: BF800000
	s_nop 0                                                    // 0000000024E4: BF800000
	s_nop 0                                                    // 0000000024E8: BF800000
	s_nop 0                                                    // 0000000024EC: BF800000
	s_nop 0                                                    // 0000000024F0: BF800000
	s_nop 0                                                    // 0000000024F4: BF800000
	s_nop 0                                                    // 0000000024F8: BF800000
	s_nop 0                                                    // 0000000024FC: BF800000
	s_nop 0                                                    // 000000002500: BF800000
	s_nop 0                                                    // 000000002504: BF800000
	s_nop 0                                                    // 000000002508: BF800000
	s_nop 0                                                    // 00000000250C: BF800000
	s_nop 0                                                    // 000000002510: BF800000
	s_nop 0                                                    // 000000002514: BF800000
	s_nop 0                                                    // 000000002518: BF800000
	s_nop 0                                                    // 00000000251C: BF800000
	s_nop 0                                                    // 000000002520: BF800000
	s_nop 0                                                    // 000000002524: BF800000
	s_nop 0                                                    // 000000002528: BF800000
	s_nop 0                                                    // 00000000252C: BF800000
	s_nop 0                                                    // 000000002530: BF800000
	s_nop 0                                                    // 000000002534: BF800000
	s_nop 0                                                    // 000000002538: BF800000
	s_nop 0                                                    // 00000000253C: BF800000
	s_nop 0                                                    // 000000002540: BF800000
	s_nop 0                                                    // 000000002544: BF800000
	s_nop 0                                                    // 000000002548: BF800000
	s_nop 0                                                    // 00000000254C: BF800000
	s_nop 0                                                    // 000000002550: BF800000
	s_nop 0                                                    // 000000002554: BF800000
	s_nop 0                                                    // 000000002558: BF800000
	s_nop 0                                                    // 00000000255C: BF800000
	s_nop 0                                                    // 000000002560: BF800000
	s_nop 0                                                    // 000000002564: BF800000
	s_nop 0                                                    // 000000002568: BF800000
	s_nop 0                                                    // 00000000256C: BF800000
	s_nop 0                                                    // 000000002570: BF800000
	s_nop 0                                                    // 000000002574: BF800000
	s_nop 0                                                    // 000000002578: BF800000
	s_nop 0                                                    // 00000000257C: BF800000
