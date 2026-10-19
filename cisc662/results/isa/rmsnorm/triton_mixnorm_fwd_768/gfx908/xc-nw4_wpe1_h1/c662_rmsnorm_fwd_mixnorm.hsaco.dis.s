
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_mixnorm_fwd_768/gfx908/xc-nw4_wpe1_h1/c662_rmsnorm_fwd_mixnorm.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001800 <c662_rmsnorm_fwd_mixnorm>:
	s_load_dwordx2 s[2:3], s[4:5], 0x38                        // 000000001800: C0060082 00000038
	s_lshl_b32 s14, s6, 2                                      // 000000001808: 8E0E8206
	v_lshrrev_b32_e32 v1, 6, v0                                // 00000000180C: 20020086
	v_or_b32_e32 v17, s14, v1                                  // 000000001810: 2822020E
	v_lshlrev_b32_e32 v19, 2, v0                               // 000000001814: 24260082
	s_waitcnt lgkmcnt(0)                                       // 000000001818: BF8CC07F
	v_mul_lo_u32 v1, s3, v17                                   // 00000000181C: D2850001 00022203
	s_load_dwordx4 s[8:11], s[4:5], 0x0                        // 000000001824: C00A0202 00000000
	s_load_dword s3, s[4:5], 0x40                              // 00000000182C: C00200C2 00000040
	v_cmp_gt_i32_e64 s[0:1], s2, v17                           // 000000001834: D0C40000 00022202
	v_mov_b32_e32 v7, 0                                        // 00000000183C: 7E0E0280
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001840: 2204029F
	v_lshlrev_b64 v[1:2], 1, v[1:2]                            // 000000001844: D28F0001 00020281
	s_waitcnt lgkmcnt(0)                                       // 00000000184C: BF8CC07F
	v_mov_b32_e32 v3, s9                                       // 000000001850: 7E060209
	v_add_co_u32_e32 v1, vcc, s8, v1                           // 000000001854: 32020208
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 000000001858: 38040503
	v_and_b32_e32 v3, 0xfc, v19                                // 00000000185C: 260626FF 000000FC
	v_lshlrev_b32_e32 v18, 1, v3                               // 000000001864: 24240681
	v_add_co_u32_e32 v13, vcc, v1, v18                         // 000000001868: 321A2501
	v_addc_co_u32_e32 v14, vcc, 0, v2, vcc                     // 00000000186C: 381C0480
	v_mov_b32_e32 v11, 0                                       // 000000001870: 7E160280
	v_mov_b32_e32 v12, 0                                       // 000000001874: 7E180280
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 000000001878: BE862000
	s_cbranch_execz 2                                          // 00000000187C: BF880002 <c662_rmsnorm_fwd_mixnorm+0x88>
	global_load_dwordx2 v[11:12], v[13:14], off                // 000000001880: DC548000 0B7F000D
	s_or_b64 exec, exec, s[6:7]                                // 000000001888: 87FE067E
	v_mul_lo_u32 v1, s3, v17                                   // 00000000188C: D2850001 00022203
	v_mov_b32_e32 v3, s11                                      // 000000001894: 7E06020B
	v_mov_b32_e32 v8, 0                                        // 000000001898: 7E100280
	v_ashrrev_i32_e32 v2, 31, v1                               // 00000000189C: 2204029F
	v_lshlrev_b64 v[1:2], 1, v[1:2]                            // 0000000018A0: D28F0001 00020281
	v_add_co_u32_e32 v1, vcc, s10, v1                          // 0000000018A8: 3202020A
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 0000000018AC: 38040503
	v_add_co_u32_e32 v15, vcc, v1, v18                         // 0000000018B0: 321E2501
	v_addc_co_u32_e32 v16, vcc, 0, v2, vcc                     // 0000000018B4: 38200480
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 0000000018B8: BE862000
	s_cbranch_execz 2                                          // 0000000018BC: BF880002 <c662_rmsnorm_fwd_mixnorm+0xc8>
	global_load_dwordx2 v[7:8], v[15:16], off                  // 0000000018C0: DC548000 077F000F
	s_or_b64 exec, exec, s[6:7]                                // 0000000018C8: 87FE067E
	v_mov_b32_e32 v5, 0                                        // 0000000018CC: 7E0A0280
	v_mov_b32_e32 v9, 0                                        // 0000000018D0: 7E120280
	v_mov_b32_e32 v10, 0                                       // 0000000018D4: 7E140280
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 0000000018D8: BE862000
	s_cbranch_execz 2                                          // 0000000018DC: BF880002 <c662_rmsnorm_fwd_mixnorm+0xe8>
	global_load_dwordx2 v[9:10], v[13:14], off offset:512      // 0000000018E0: DC548200 097F000D
	s_or_b64 exec, exec, s[6:7]                                // 0000000018E8: 87FE067E
	v_mov_b32_e32 v6, 0                                        // 0000000018EC: 7E0C0280
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 0000000018F0: BE862000
	s_cbranch_execz 2                                          // 0000000018F4: BF880002 <c662_rmsnorm_fwd_mixnorm+0x100>
	global_load_dwordx2 v[5:6], v[15:16], off offset:512       // 0000000018F8: DC548200 057F000F
	s_or_b64 exec, exec, s[6:7]                                // 000000001900: 87FE067E
	s_load_dwordx4 s[8:11], s[4:5], 0x10                       // 000000001904: C00A0202 00000010
	s_load_dwordx2 s[6:7], s[4:5], 0x20                        // 00000000190C: C0060182 00000020
	v_mov_b32_e32 v1, 0                                        // 000000001914: 7E020280
	v_mov_b32_e32 v3, 0                                        // 000000001918: 7E060280
	v_mov_b32_e32 v4, 0                                        // 00000000191C: 7E080280
	s_and_saveexec_b64 s[12:13], s[0:1]                        // 000000001920: BE8C2000
	s_cbranch_execz 2                                          // 000000001924: BF880002 <c662_rmsnorm_fwd_mixnorm+0x130>
	global_load_dwordx2 v[3:4], v[13:14], off offset:1024      // 000000001928: DC548400 037F000D
	s_or_b64 exec, exec, s[12:13]                              // 000000001930: 87FE0C7E
	s_load_dword s13, s[4:5], 0x44                             // 000000001934: C0020342 00000044
	s_waitcnt lgkmcnt(0)                                       // 00000000193C: BF8CC07F
	s_load_dword s3, s[8:9], 0x0                               // 000000001940: C00200C4 00000000
	s_load_dword s12, s[10:11], 0x0                            // 000000001948: C0020305 00000000
	v_mov_b32_e32 v2, 0                                        // 000000001950: 7E040280
	s_and_saveexec_b64 s[8:9], s[0:1]                          // 000000001954: BE882000
	s_cbranch_execz 2                                          // 000000001958: BF880002 <c662_rmsnorm_fwd_mixnorm+0x164>
	global_load_dwordx2 v[1:2], v[15:16], off offset:1024      // 00000000195C: DC548400 017F000F
	s_or_b64 exec, exec, s[8:9]                                // 000000001964: 87FE087E
	s_waitcnt vmcnt(0)                                         // 000000001968: BF8C0F70
	v_and_b32_e32 v15, 0xffff0000, v7                          // 00000000196C: 261E0EFF FFFF0000
	v_lshlrev_b32_e32 v7, 16, v7                               // 000000001974: 240E0E90
	v_and_b32_e32 v13, 0xffff0000, v11                         // 000000001978: 261A16FF FFFF0000
	v_lshlrev_b32_e32 v11, 16, v11                             // 000000001980: 24161690
	s_waitcnt lgkmcnt(0)                                       // 000000001984: BF8CC07F
	v_mul_f32_e32 v7, s12, v7                                  // 000000001988: 0A0E0E0C
	v_mul_f32_e32 v15, s12, v15                                // 00000000198C: 0A1E1E0C
	v_fmac_f32_e32 v15, s3, v13                                // 000000001990: 761E1A03
	v_fmac_f32_e32 v7, s3, v11                                 // 000000001994: 760E1603
	v_bfe_u32 v11, v15, 16, 1                                  // 000000001998: D1C8000B 0205210F
	v_bfe_u32 v13, v7, 16, 1                                   // 0000000019A0: D1C8000D 02052107
	s_movk_i32 s10, 0x7fff                                     // 0000000019A8: B00A7FFF
	v_add3_u32 v11, v15, v11, s10                              // 0000000019AC: D1FF000B 002A170F
	v_add3_u32 v13, v7, v13, s10                               // 0000000019B4: D1FF000D 002A1B07
	v_lshrrev_b32_e32 v11, 16, v11                             // 0000000019BC: 20161690
	v_lshrrev_b32_e32 v20, 16, v13                             // 0000000019C0: 20281A90
	v_mov_b32_e32 v13, 0x7fff                                  // 0000000019C4: 7E1A02FF 00007FFF
	v_cmp_o_f32_e32 vcc, v15, v15                              // 0000000019CC: 7C8E1F0F
	v_cndmask_b32_e32 v11, v13, v11, vcc                       // 0000000019D0: 0016170D
	v_cmp_o_f32_e32 vcc, v7, v7                                // 0000000019D4: 7C8E0F07
	v_and_b32_e32 v16, 0xffff0000, v8                          // 0000000019D8: 262010FF FFFF0000
	v_cndmask_b32_e32 v7, v13, v20, vcc                        // 0000000019E0: 000E290D
	s_mov_b32 s11, 0x5040100                                   // 0000000019E4: BE8B00FF 05040100
	v_and_b32_e32 v14, 0xffff0000, v12                         // 0000000019EC: 261C18FF FFFF0000
	v_lshlrev_b32_e32 v8, 16, v8                               // 0000000019F4: 24101090
	v_perm_b32 v7, v11, v7, s11                                // 0000000019F8: D1ED0007 002E0F0B
	v_mul_f32_e32 v11, s12, v16                                // 000000001A00: 0A16200C
	v_lshlrev_b32_e32 v12, 16, v12                             // 000000001A04: 24181890
	v_mul_f32_e32 v8, s12, v8                                  // 000000001A08: 0A10100C
	v_fmac_f32_e32 v11, s3, v14                                // 000000001A0C: 76161C03
	v_fmac_f32_e32 v8, s3, v12                                 // 000000001A10: 76101803
	v_bfe_u32 v12, v11, 16, 1                                  // 000000001A14: D1C8000C 0205210B
	v_bfe_u32 v14, v8, 16, 1                                   // 000000001A1C: D1C8000E 02052108
	v_add3_u32 v12, v11, v12, s10                              // 000000001A24: D1FF000C 002A190B
	v_cmp_o_f32_e32 vcc, v11, v11                              // 000000001A2C: 7C8E170B
	v_mul_lo_u32 v11, s13, v17                                 // 000000001A30: D285000B 0002220D
	v_add3_u32 v14, v8, v14, s10                               // 000000001A38: D1FF000E 002A1D08
	v_lshrrev_b32_e32 v12, 16, v12                             // 000000001A40: 20181890
	v_lshrrev_b32_e32 v14, 16, v14                             // 000000001A44: 201C1C90
	v_cndmask_b32_e32 v12, v13, v12, vcc                       // 000000001A48: 0018190D
	v_cmp_o_f32_e32 vcc, v8, v8                                // 000000001A4C: 7C8E1108
	v_cndmask_b32_e32 v8, v13, v14, vcc                        // 000000001A50: 00101D0D
	v_perm_b32 v8, v12, v8, s11                                // 000000001A54: D1ED0008 002E110C
	v_ashrrev_i32_e32 v12, 31, v11                             // 000000001A5C: 2218169F
	v_lshlrev_b64 v[11:12], 1, v[11:12]                        // 000000001A60: D28F000B 00021681
	v_mov_b32_e32 v14, s7                                      // 000000001A68: 7E1C0207
	v_add_co_u32_e32 v11, vcc, s6, v11                         // 000000001A6C: 32161606
	v_addc_co_u32_e32 v12, vcc, v14, v12, vcc                  // 000000001A70: 3818190E
	v_add_co_u32_e32 v11, vcc, v11, v18                        // 000000001A74: 3216250B
	s_mov_b32 s8, s12                                          // 000000001A78: BE88000C
	s_mov_b32 s9, s3                                           // 000000001A7C: BE890003
	v_addc_co_u32_e32 v12, vcc, 0, v12, vcc                    // 000000001A80: 38181880
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 000000001A84: BE862000
	s_cbranch_execz 2                                          // 000000001A88: BF880002 <c662_rmsnorm_fwd_mixnorm+0x294>
	global_store_dwordx2 v[11:12], v[7:8], off                 // 000000001A8C: DC748000 007F070B
	s_or_b64 exec, exec, s[6:7]                                // 000000001A94: 87FE067E
	v_and_b32_e32 v16, 0xffff0000, v5                          // 000000001A98: 26200AFF FFFF0000
	v_and_b32_e32 v14, 0xffff0000, v9                          // 000000001AA0: 261C12FF FFFF0000
	v_lshlrev_b32_e32 v5, 16, v5                               // 000000001AA8: 240A0A90
	v_mul_f32_e32 v16, s8, v16                                 // 000000001AAC: 0A202008
	v_lshlrev_b32_e32 v9, 16, v9                               // 000000001AB0: 24121290
	v_mul_f32_e32 v5, s12, v5                                  // 000000001AB4: 0A0A0A0C
	v_fmac_f32_e32 v16, s9, v14                                // 000000001AB8: 76201C09
	v_fmac_f32_e32 v5, s3, v9                                  // 000000001ABC: 760A1203
	v_bfe_u32 v9, v16, 16, 1                                   // 000000001AC0: D1C80009 02052110
	v_bfe_u32 v14, v5, 16, 1                                   // 000000001AC8: D1C8000E 02052105
	v_add3_u32 v9, v16, v9, s10                                // 000000001AD0: D1FF0009 002A1310
	v_add3_u32 v14, v5, v14, s10                               // 000000001AD8: D1FF000E 002A1D05
	v_lshrrev_b32_e32 v9, 16, v9                               // 000000001AE0: 20121290
	v_cmp_o_f32_e32 vcc, v16, v16                              // 000000001AE4: 7C8E2110
	v_lshrrev_b32_e32 v14, 16, v14                             // 000000001AE8: 201C1C90
	v_cndmask_b32_e32 v9, v13, v9, vcc                         // 000000001AEC: 0012130D
	v_cmp_o_f32_e32 vcc, v5, v5                                // 000000001AF0: 7C8E0B05
	v_and_b32_e32 v20, 0xffff0000, v6                          // 000000001AF4: 26280CFF FFFF0000
	v_cndmask_b32_e32 v5, v13, v14, vcc                        // 000000001AFC: 000A1D0D
	v_and_b32_e32 v15, 0xffff0000, v10                         // 000000001B00: 261E14FF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v6                               // 000000001B08: 240C0C90
	v_perm_b32 v5, v9, v5, s11                                 // 000000001B0C: D1ED0005 002E0B09
	v_mul_f32_e32 v9, s8, v20                                  // 000000001B14: 0A122808
	v_lshlrev_b32_e32 v10, 16, v10                             // 000000001B18: 24141490
	v_mul_f32_e32 v6, s12, v6                                  // 000000001B1C: 0A0C0C0C
	v_fmac_f32_e32 v9, s9, v15                                 // 000000001B20: 76121E09
	v_fmac_f32_e32 v6, s3, v10                                 // 000000001B24: 760C1403
	v_bfe_u32 v10, v9, 16, 1                                   // 000000001B28: D1C8000A 02052109
	v_bfe_u32 v14, v6, 16, 1                                   // 000000001B30: D1C8000E 02052106
	v_add3_u32 v10, v9, v10, s10                               // 000000001B38: D1FF000A 002A1509
	v_cmp_o_f32_e32 vcc, v9, v9                                // 000000001B40: 7C8E1309
	v_add3_u32 v14, v6, v14, s10                               // 000000001B44: D1FF000E 002A1D06
	v_cndmask_b32_sdwa v9, v13, v10, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001B4C: 001214F9 0506060D
	v_cmp_o_f32_e32 vcc, v6, v6                                // 000000001B54: 7C8E0D06
	v_cndmask_b32_sdwa v6, v13, v14, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001B58: 000C1CF9 0506060D
	v_perm_b32 v6, v9, v6, s11                                 // 000000001B60: D1ED0006 002E0D09
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 000000001B68: BE862000
	s_cbranch_execz 2                                          // 000000001B6C: BF880002 <c662_rmsnorm_fwd_mixnorm+0x378>
	global_store_dwordx2 v[11:12], v[5:6], off offset:512      // 000000001B70: DC748200 007F050B
	s_or_b64 exec, exec, s[6:7]                                // 000000001B78: 87FE067E
	v_and_b32_e32 v13, 0xffff0000, v1                          // 000000001B7C: 261A02FF FFFF0000
	v_and_b32_e32 v9, 0xffff0000, v3                           // 000000001B84: 261206FF FFFF0000
	v_lshlrev_b32_e32 v1, 16, v1                               // 000000001B8C: 24020290
	v_mul_f32_e32 v13, s8, v13                                 // 000000001B90: 0A1A1A08
	v_lshlrev_b32_e32 v3, 16, v3                               // 000000001B94: 24060690
	v_mul_f32_e32 v1, s12, v1                                  // 000000001B98: 0A02020C
	v_fmac_f32_e32 v13, s9, v9                                 // 000000001B9C: 761A1209
	v_fmac_f32_e32 v1, s3, v3                                  // 000000001BA0: 76020603
	v_bfe_u32 v3, v13, 16, 1                                   // 000000001BA4: D1C80003 0205210D
	v_bfe_u32 v9, v1, 16, 1                                    // 000000001BAC: D1C80009 02052101
	v_add3_u32 v3, v13, v3, s10                                // 000000001BB4: D1FF0003 002A070D
	v_add3_u32 v9, v1, v9, s10                                 // 000000001BBC: D1FF0009 002A1301
	v_lshrrev_b32_e32 v3, 16, v3                               // 000000001BC4: 20060690
	v_mov_b32_e32 v15, 0x7fff                                  // 000000001BC8: 7E1E02FF 00007FFF
	v_cmp_o_f32_e32 vcc, v13, v13                              // 000000001BD0: 7C8E1B0D
	v_lshrrev_b32_e32 v9, 16, v9                               // 000000001BD4: 20121290
	v_cndmask_b32_e32 v3, v15, v3, vcc                         // 000000001BD8: 0006070F
	v_cmp_o_f32_e32 vcc, v1, v1                                // 000000001BDC: 7C8E0301
	v_and_b32_e32 v14, 0xffff0000, v2                          // 000000001BE0: 261C04FF FFFF0000
	v_cndmask_b32_e32 v1, v15, v9, vcc                         // 000000001BE8: 0002130F
	s_mov_b32 s6, 0x5040100                                    // 000000001BEC: BE8600FF 05040100
	v_and_b32_e32 v10, 0xffff0000, v4                          // 000000001BF4: 261408FF FFFF0000
	v_lshlrev_b32_e32 v2, 16, v2                               // 000000001BFC: 24040490
	v_perm_b32 v1, v3, v1, s6                                  // 000000001C00: D1ED0001 001A0303
	v_mul_f32_e32 v3, s8, v14                                  // 000000001C08: 0A061C08
	v_lshlrev_b32_e32 v4, 16, v4                               // 000000001C0C: 24080890
	v_mul_f32_e32 v2, s12, v2                                  // 000000001C10: 0A04040C
	v_fmac_f32_e32 v3, s9, v10                                 // 000000001C14: 76061409
	s_load_dword s11, s[4:5], 0x4c                             // 000000001C18: C00202C2 0000004C
	v_fmac_f32_e32 v2, s3, v4                                  // 000000001C20: 76040803
	v_bfe_u32 v4, v3, 16, 1                                    // 000000001C24: D1C80004 02052103
	v_bfe_u32 v9, v2, 16, 1                                    // 000000001C2C: D1C80009 02052102
	v_add3_u32 v4, v3, v4, s10                                 // 000000001C34: D1FF0004 002A0903
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000001C3C: 7C8E0703
	v_add3_u32 v9, v2, v9, s10                                 // 000000001C40: D1FF0009 002A1302
	v_cndmask_b32_sdwa v3, v15, v4, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001C48: 000608F9 0506060F
	v_cmp_o_f32_e32 vcc, v2, v2                                // 000000001C50: 7C8E0502
	v_cndmask_b32_sdwa v2, v15, v9, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001C54: 000412F9 0506060F
	v_perm_b32 v2, v3, v2, s6                                  // 000000001C5C: D1ED0002 001A0503
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 000000001C64: BE862000
	s_cbranch_execz 2                                          // 000000001C68: BF880002 <c662_rmsnorm_fwd_mixnorm+0x474>
	global_store_dwordx2 v[11:12], v[1:2], off offset:1024     // 000000001C6C: DC748400 007F010B
	s_or_b64 exec, exec, s[6:7]                                // 000000001C74: 87FE067E
	v_and_b32_e32 v11, 0xffff0000, v7                          // 000000001C78: 26160EFF FFFF0000
	v_lshlrev_b32_e32 v12, 16, v7                              // 000000001C80: 24180E90
	v_and_b32_e32 v13, 0xffff0000, v8                          // 000000001C84: 261A10FF FFFF0000
	v_lshlrev_b32_e32 v14, 16, v8                              // 000000001C8C: 241C1090
	v_and_b32_e32 v7, 0xffff0000, v5                           // 000000001C90: 260E0AFF FFFF0000
	v_lshlrev_b32_e32 v8, 16, v5                               // 000000001C98: 24100A90
	v_and_b32_e32 v9, 0xffff0000, v6                           // 000000001C9C: 26120CFF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v6                               // 000000001CA4: 240C0C90
	v_mul_f32_e32 v5, v8, v8                                   // 000000001CA8: 0A0A1108
	v_mul_f32_e32 v10, v7, v7                                  // 000000001CAC: 0A140F07
	v_mul_f32_e32 v15, v6, v6                                  // 000000001CB0: 0A1E0D06
	v_fmac_f32_e32 v5, v12, v12                                // 000000001CB4: 760A190C
	v_fmac_f32_e32 v10, v11, v11                               // 000000001CB8: 7614170B
	v_and_b32_e32 v3, 0xffff0000, v1                           // 000000001CBC: 260602FF FFFF0000
	v_lshlrev_b32_e32 v1, 16, v1                               // 000000001CC4: 24020290
	v_mul_f32_e32 v16, v9, v9                                  // 000000001CC8: 0A201309
	v_fmac_f32_e32 v15, v14, v14                               // 000000001CCC: 761E1D0E
	v_and_b32_e32 v4, 0xffff0000, v2                           // 000000001CD0: 260804FF FFFF0000
	v_lshlrev_b32_e32 v2, 16, v2                               // 000000001CD8: 24040490
	v_fmac_f32_e32 v5, v1, v1                                  // 000000001CDC: 760A0301
	v_fmac_f32_e32 v10, v3, v3                                 // 000000001CE0: 76140703
	v_fmac_f32_e32 v16, v13, v13                               // 000000001CE4: 76201B0D
	v_fmac_f32_e32 v15, v2, v2                                 // 000000001CE8: 761E0502
	v_add_f32_e32 v5, v5, v10                                  // 000000001CEC: 020A1505
	v_fmac_f32_e32 v16, v4, v4                                 // 000000001CF0: 76200904
	v_add_f32_e32 v5, v15, v5                                  // 000000001CF4: 020A0B0F
	v_add_f32_e32 v5, v16, v5                                  // 000000001CF8: 020A0B10
	v_xor_b32_e32 v10, 0x80, v19                               // 000000001CFC: 2A1426FF 00000080
	ds_bpermute_b32 v10, v10, v5                               // 000000001D04: D87E0000 0A00050A
	s_mov_b32 s3, 0x44400000                                   // 000000001D0C: BE8300FF 44400000
	s_waitcnt lgkmcnt(0)                                       // 000000001D14: BF8CC07F
	v_add_f32_e32 v5, v5, v10                                  // 000000001D18: 020A1505
	ds_swizzle_b32 v10, v5 offset:swizzle(SWAP,16)             // 000000001D1C: D87A401F 0A000005
	s_waitcnt lgkmcnt(0)                                       // 000000001D24: BF8CC07F
	v_add_f32_e32 v5, v5, v10                                  // 000000001D28: 020A1505
	ds_swizzle_b32 v10, v5 offset:swizzle(SWAP,8)              // 000000001D2C: D87A201F 0A000005
	s_waitcnt lgkmcnt(0)                                       // 000000001D34: BF8CC07F
	v_add_f32_e32 v5, v5, v10                                  // 000000001D38: 020A1505
	ds_swizzle_b32 v10, v5 offset:swizzle(SWAP,4)              // 000000001D3C: D87A101F 0A000005
	s_waitcnt lgkmcnt(0)                                       // 000000001D44: BF8CC07F
	v_add_f32_e32 v5, v5, v10                                  // 000000001D48: 020A1505
	ds_swizzle_b32 v10, v5 offset:swizzle(SWAP,2)              // 000000001D4C: D87A081F 0A000005
	s_waitcnt lgkmcnt(0)                                       // 000000001D54: BF8CC07F
	v_add_f32_e32 v5, v5, v10                                  // 000000001D58: 020A1505
	ds_swizzle_b32 v10, v5 offset:swizzle(SWAP,1)              // 000000001D5C: D87A041F 0A000005
	s_waitcnt lgkmcnt(0)                                       // 000000001D64: BF8CC07F
	v_add_f32_e32 v5, v5, v10                                  // 000000001D68: 020A1505
	v_div_scale_f32 v10, s[6:7], s3, s3, v5                    // 000000001D6C: D1E0060A 04140603
	v_div_scale_f32 v15, vcc, v5, s3, v5                       // 000000001D74: D1E06A0F 04140705
	v_rcp_f32_e32 v16, v10                                     // 000000001D7C: 7E20450A
	v_fma_f32 v19, -v10, v16, 1.0                              // 000000001D80: D1CB0013 23CA210A
	v_fmac_f32_e32 v16, v19, v16                               // 000000001D88: 76202113
	v_mul_f32_e32 v19, v15, v16                                // 000000001D8C: 0A26210F
	v_fma_f32 v20, -v10, v19, v15                              // 000000001D90: D1CB0014 243E270A
	v_fmac_f32_e32 v19, v20, v16                               // 000000001D98: 76262114
	v_fma_f32 v10, -v10, v19, v15                              // 000000001D9C: D1CB000A 243E270A
	v_div_fmas_f32 v10, v10, v16, v19                          // 000000001DA4: D1E2000A 044E210A
	v_div_fixup_f32 v5, v10, s3, v5                            // 000000001DAC: D1DE0005 0414070A
	v_add_f32_e32 v5, s11, v5                                  // 000000001DB4: 020A0A0B
	v_rsq_f32_e32 v5, v5                                       // 000000001DB8: 7E0A4905
	v_and_b32_e32 v10, 0xc0, v0                                // 000000001DBC: 261400FF 000000C0
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 000000001DC4: BE862000
	s_cbranch_execz 134                                        // 000000001DC8: BF880086 <c662_rmsnorm_fwd_mixnorm+0x7e4>
	s_load_dword s3, s[4:5], 0x48                              // 000000001DCC: C00200C2 00000048
	s_load_dwordx2 s[0:1], s[4:5], 0x28                        // 000000001DD4: C0060002 00000028
	v_mul_f32_e32 v11, v5, v11                                 // 000000001DDC: 0A161705
	v_mov_b32_e32 v21, 0x7fff                                  // 000000001DE0: 7E2A02FF 00007FFF
	v_bfe_u32 v20, v11, 16, 1                                  // 000000001DE8: D1C80014 0205210B
	s_waitcnt lgkmcnt(0)                                       // 000000001DF0: BF8CC07F
	v_mul_lo_u32 v15, s3, v17                                  // 000000001DF4: D285000F 00022203
	v_mov_b32_e32 v17, s1                                      // 000000001DFC: 7E220201
	s_mov_b32 s1, 0x5040100                                    // 000000001E00: BE8100FF 05040100
	v_mul_f32_e32 v6, v5, v6                                   // 000000001E08: 0A0C0D05
	v_ashrrev_i32_e32 v16, 31, v15                             // 000000001E0C: 22201E9F
	v_lshlrev_b64 v[15:16], 1, v[15:16]                        // 000000001E10: D28F000F 00021E81
	v_mul_f32_e32 v8, v5, v8                                   // 000000001E18: 0A101105
	v_add_co_u32_e32 v15, vcc, s0, v15                         // 000000001E1C: 321E1E00
	v_addc_co_u32_e32 v16, vcc, v17, v16, vcc                  // 000000001E20: 38202111
	v_mul_f32_e32 v17, v5, v12                                 // 000000001E24: 0A221905
	v_mul_f32_e32 v12, v5, v13                                 // 000000001E28: 0A181B05
	v_mul_f32_e32 v13, v5, v14                                 // 000000001E2C: 0A1A1D05
	v_add_co_u32_e32 v15, vcc, v15, v18                        // 000000001E30: 321E250F
	v_bfe_u32 v14, v13, 16, 1                                  // 000000001E34: D1C8000E 0205210D
	s_movk_i32 s0, 0x7fff                                      // 000000001E3C: B0007FFF
	v_addc_co_u32_e32 v16, vcc, 0, v16, vcc                    // 000000001E40: 38202080
	v_bfe_u32 v18, v12, 16, 1                                  // 000000001E44: D1C80012 0205210C
	v_add3_u32 v14, v13, v14, s0                               // 000000001E4C: D1FF000E 00021D0D
	v_bfe_u32 v19, v17, 16, 1                                  // 000000001E54: D1C80013 02052111
	v_add3_u32 v18, v12, v18, s0                               // 000000001E5C: D1FF0012 0002250C
	v_lshrrev_b32_e32 v14, 16, v14                             // 000000001E64: 201C1C90
	v_cmp_o_f32_e32 vcc, v13, v13                              // 000000001E68: 7C8E1B0D
	v_add3_u32 v19, v17, v19, s0                               // 000000001E6C: D1FF0013 00022711
	v_lshrrev_b32_e32 v18, 16, v18                             // 000000001E74: 20242490
	v_cndmask_b32_e32 v13, v21, v14, vcc                       // 000000001E78: 001A1D15
	v_cmp_o_f32_e32 vcc, v12, v12                              // 000000001E7C: 7C8E190C
	v_add3_u32 v20, v11, v20, s0                               // 000000001E80: D1FF0014 0002290B
	v_lshrrev_b32_e32 v19, 16, v19                             // 000000001E88: 20262690
	v_cndmask_b32_e32 v12, v21, v18, vcc                       // 000000001E8C: 00182515
	v_cmp_o_f32_e32 vcc, v17, v17                              // 000000001E90: 7C8E2311
	v_lshrrev_b32_e32 v20, 16, v20                             // 000000001E94: 20282890
	v_perm_b32 v12, v12, v13, s1                               // 000000001E98: D1ED000C 00061B0C
	v_cndmask_b32_e32 v13, v21, v19, vcc                       // 000000001EA0: 001A2715
	v_cmp_o_f32_e32 vcc, v11, v11                              // 000000001EA4: 7C8E170B
	v_cndmask_b32_e32 v11, v21, v20, vcc                       // 000000001EA8: 00162915
	v_perm_b32 v11, v11, v13, s1                               // 000000001EAC: D1ED000B 00061B0B
	global_store_dwordx2 v[15:16], v[11:12], off               // 000000001EB4: DC748000 007F0B0F
	v_mul_f32_e32 v11, v5, v7                                  // 000000001EBC: 0A160F05
	v_mul_f32_e32 v7, v5, v9                                   // 000000001EC0: 0A0E1305
	v_bfe_u32 v9, v6, 16, 1                                    // 000000001EC4: D1C80009 02052106
	v_bfe_u32 v12, v7, 16, 1                                   // 000000001ECC: D1C8000C 02052107
	v_add3_u32 v9, v6, v9, s0                                  // 000000001ED4: D1FF0009 00021306
	v_bfe_u32 v13, v8, 16, 1                                   // 000000001EDC: D1C8000D 02052108
	v_add3_u32 v12, v7, v12, s0                                // 000000001EE4: D1FF000C 00021907
	v_lshrrev_b32_e32 v9, 16, v9                               // 000000001EEC: 20121290
	v_cmp_o_f32_e32 vcc, v6, v6                                // 000000001EF0: 7C8E0D06
	v_bfe_u32 v14, v11, 16, 1                                  // 000000001EF4: D1C8000E 0205210B
	v_add3_u32 v13, v8, v13, s0                                // 000000001EFC: D1FF000D 00021B08
	v_lshrrev_b32_e32 v12, 16, v12                             // 000000001F04: 20181890
	v_cndmask_b32_e32 v6, v21, v9, vcc                         // 000000001F08: 000C1315
	v_cmp_o_f32_e32 vcc, v7, v7                                // 000000001F0C: 7C8E0F07
	v_add3_u32 v14, v11, v14, s0                               // 000000001F10: D1FF000E 00021D0B
	v_lshrrev_b32_e32 v13, 16, v13                             // 000000001F18: 201A1A90
	v_cndmask_b32_e32 v7, v21, v12, vcc                        // 000000001F1C: 000E1915
	v_cmp_o_f32_e32 vcc, v8, v8                                // 000000001F20: 7C8E1108
	v_lshrrev_b32_e32 v14, 16, v14                             // 000000001F24: 201C1C90
	v_perm_b32 v7, v7, v6, s1                                  // 000000001F28: D1ED0007 00060D07
	v_cndmask_b32_e32 v6, v21, v13, vcc                        // 000000001F30: 000C1B15
	v_cmp_o_f32_e32 vcc, v11, v11                              // 000000001F34: 7C8E170B
	v_cndmask_b32_e32 v8, v21, v14, vcc                        // 000000001F38: 00101D15
	v_perm_b32 v6, v8, v6, s1                                  // 000000001F3C: D1ED0006 00060D08
	v_mul_f32_e32 v2, v5, v2                                   // 000000001F44: 0A040505
	global_store_dwordx2 v[15:16], v[6:7], off offset:512      // 000000001F48: DC748200 007F060F
	v_mul_f32_e32 v4, v5, v4                                   // 000000001F50: 0A080905
	v_bfe_u32 v6, v2, 16, 1                                    // 000000001F54: D1C80006 02052102
	v_mul_f32_e32 v1, v5, v1                                   // 000000001F5C: 0A020305
	v_bfe_u32 v7, v4, 16, 1                                    // 000000001F60: D1C80007 02052104
	v_add3_u32 v6, v2, v6, s0                                  // 000000001F68: D1FF0006 00020D02
	v_cmp_o_f32_e32 vcc, v2, v2                                // 000000001F70: 7C8E0502
	v_mul_f32_e32 v3, v5, v3                                   // 000000001F74: 0A060705
	v_bfe_u32 v8, v1, 16, 1                                    // 000000001F78: D1C80008 02052101
	v_add3_u32 v7, v4, v7, s0                                  // 000000001F80: D1FF0007 00020F04
	v_cndmask_b32_sdwa v2, v21, v6, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001F88: 00040CF9 05060615
	v_cmp_o_f32_e32 vcc, v4, v4                                // 000000001F90: 7C8E0904
	v_bfe_u32 v9, v3, 16, 1                                    // 000000001F94: D1C80009 02052103
	v_add3_u32 v8, v1, v8, s0                                  // 000000001F9C: D1FF0008 00021101
	v_cndmask_b32_sdwa v4, v21, v7, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001FA4: 00080EF9 05060615
	v_cmp_o_f32_e32 vcc, v1, v1                                // 000000001FAC: 7C8E0301
	v_add3_u32 v9, v3, v9, s0                                  // 000000001FB0: D1FF0009 00021303
	v_cndmask_b32_sdwa v1, v21, v8, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001FB8: 000210F9 05060615
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000001FC0: 7C8E0703
	v_cndmask_b32_sdwa v3, v21, v9, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001FC4: 000612F9 05060615
	v_perm_b32 v2, v4, v2, s1                                  // 000000001FCC: D1ED0002 00060504
	v_perm_b32 v1, v3, v1, s1                                  // 000000001FD4: D1ED0001 00060303
	global_store_dwordx2 v[15:16], v[1:2], off offset:1024     // 000000001FDC: DC748400 007F010F
	s_or_b64 exec, exec, s[6:7]                                // 000000001FE4: 87FE067E
	v_and_b32_e32 v3, 3, v0                                    // 000000001FE8: 26060083
	v_or_b32_e32 v1, s14, v3                                   // 000000001FEC: 2802060E
	v_and_b32_e32 v0, 0xfc, v0                                 // 000000001FF0: 260000FF 000000FC
	v_cmp_gt_i32_e32 vcc, s2, v1                               // 000000001FF8: 7D880202
	v_lshrrev_b32_e32 v2, 4, v10                               // 000000001FFC: 20041484
	v_cmp_eq_u32_e64 s[0:1], 0, v0                             // 000000002000: D0CA0000 00020080
	v_add_u32_e32 v2, 0, v2                                    // 000000002008: 68040480
	s_and_b64 s[0:1], s[0:1], vcc                              // 00000000200C: 86806A00
	ds_write_b32 v2, v5                                        // 000000002010: D81A0000 00000502
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000002018: BF8C0070
	s_barrier                                                  // 00000000201C: BF8A0000
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000002020: BE822000
	s_cbranch_execz 15                                         // 000000002024: BF88000F <c662_rmsnorm_fwd_mixnorm+0x864>
	s_load_dwordx2 s[0:1], s[4:5], 0x30                        // 000000002028: C0060002 00000030
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000002030: 2204029F
	v_lshl_add_u32 v3, v3, 2, 0                                // 000000002034: D1FD0003 02010503
	v_lshlrev_b64 v[0:1], 2, v[1:2]                            // 00000000203C: D28F0000 00020282
	ds_read_b32 v2, v3                                         // 000000002044: D86C0000 02000003
	s_waitcnt lgkmcnt(0)                                       // 00000000204C: BF8CC07F
	v_mov_b32_e32 v3, s1                                       // 000000002050: 7E060201
	v_add_co_u32_e32 v0, vcc, s0, v0                           // 000000002054: 32000000
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc                     // 000000002058: 38020303
	global_store_dword v[0:1], v2, off                         // 00000000205C: DC708000 007F0200
	s_endpgm                                                   // 000000002064: BF810000
