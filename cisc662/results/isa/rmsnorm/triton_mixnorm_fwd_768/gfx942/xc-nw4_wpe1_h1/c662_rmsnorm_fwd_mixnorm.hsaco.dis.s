
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_mixnorm_fwd_768/gfx942/xc-nw4_wpe1_h1/c662_rmsnorm_fwd_mixnorm.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001800 <c662_rmsnorm_fwd_mixnorm>:
	s_load_dwordx2 s[2:3], s[0:1], 0x0                         // 000000001800: C0060080 00000000
	s_load_dwordx8 s[4:11], s[0:1], 0x8                        // 000000001808: C00E0100 00000008
	s_load_dwordx4 s[12:15], s[0:1], 0x28                      // 000000001810: C00A0300 00000028
	s_waitcnt lgkmcnt(0)                                       // 000000001818: BF8CC07F
	s_branch 56                                                // 00000000181C: BF820038 <c662_rmsnorm_fwd_mixnorm+0x100>
	s_nop 0                                                    // 000000001820: BF800000
	s_nop 0                                                    // 000000001824: BF800000
	s_nop 0                                                    // 000000001828: BF800000
	s_nop 0                                                    // 00000000182C: BF800000
	s_nop 0                                                    // 000000001830: BF800000
	s_nop 0                                                    // 000000001834: BF800000
	s_nop 0                                                    // 000000001838: BF800000
	s_nop 0                                                    // 00000000183C: BF800000
	s_nop 0                                                    // 000000001840: BF800000
	s_nop 0                                                    // 000000001844: BF800000
	s_nop 0                                                    // 000000001848: BF800000
	s_nop 0                                                    // 00000000184C: BF800000
	s_nop 0                                                    // 000000001850: BF800000
	s_nop 0                                                    // 000000001854: BF800000
	s_nop 0                                                    // 000000001858: BF800000
	s_nop 0                                                    // 00000000185C: BF800000
	s_nop 0                                                    // 000000001860: BF800000
	s_nop 0                                                    // 000000001864: BF800000
	s_nop 0                                                    // 000000001868: BF800000
	s_nop 0                                                    // 00000000186C: BF800000
	s_nop 0                                                    // 000000001870: BF800000
	s_nop 0                                                    // 000000001874: BF800000
	s_nop 0                                                    // 000000001878: BF800000
	s_nop 0                                                    // 00000000187C: BF800000
	s_nop 0                                                    // 000000001880: BF800000
	s_nop 0                                                    // 000000001884: BF800000
	s_nop 0                                                    // 000000001888: BF800000
	s_nop 0                                                    // 00000000188C: BF800000
	s_nop 0                                                    // 000000001890: BF800000
	s_nop 0                                                    // 000000001894: BF800000
	s_nop 0                                                    // 000000001898: BF800000
	s_nop 0                                                    // 00000000189C: BF800000
	s_nop 0                                                    // 0000000018A0: BF800000
	s_nop 0                                                    // 0000000018A4: BF800000
	s_nop 0                                                    // 0000000018A8: BF800000
	s_nop 0                                                    // 0000000018AC: BF800000
	s_nop 0                                                    // 0000000018B0: BF800000
	s_nop 0                                                    // 0000000018B4: BF800000
	s_nop 0                                                    // 0000000018B8: BF800000
	s_nop 0                                                    // 0000000018BC: BF800000
	s_nop 0                                                    // 0000000018C0: BF800000
	s_nop 0                                                    // 0000000018C4: BF800000
	s_nop 0                                                    // 0000000018C8: BF800000
	s_nop 0                                                    // 0000000018CC: BF800000
	s_nop 0                                                    // 0000000018D0: BF800000
	s_nop 0                                                    // 0000000018D4: BF800000
	s_nop 0                                                    // 0000000018D8: BF800000
	s_nop 0                                                    // 0000000018DC: BF800000
	s_nop 0                                                    // 0000000018E0: BF800000
	s_nop 0                                                    // 0000000018E4: BF800000
	s_nop 0                                                    // 0000000018E8: BF800000
	s_nop 0                                                    // 0000000018EC: BF800000
	s_nop 0                                                    // 0000000018F0: BF800000
	s_nop 0                                                    // 0000000018F4: BF800000
	s_nop 0                                                    // 0000000018F8: BF800000
	s_nop 0                                                    // 0000000018FC: BF800000
	s_load_dwordx2 s[18:19], s[0:1], 0x38                      // 000000001900: C0060480 00000038
	s_load_dword s21, s[0:1], 0x40                             // 000000001908: C0020540 00000040
	s_lshl_b32 s20, s16, 2                                     // 000000001910: 8E148210
	v_lshrrev_b32_e32 v1, 6, v0                                // 000000001914: 20020086
	v_or_b32_e32 v1, s20, v1                                   // 000000001918: 28020214
	s_waitcnt lgkmcnt(0)                                       // 00000000191C: BF8CC07F
	v_mul_lo_u32 v2, s19, v1                                   // 000000001920: D2850002 00020213
	v_ashrrev_i32_e32 v3, 31, v2                               // 000000001928: 2206049F
	v_lshl_add_u64 v[4:5], v[2:3], 1, s[2:3]                   // 00000000192C: D2080004 00090302
	v_lshlrev_b32_e32 v2, 2, v0                                // 000000001934: 24040082
	v_and_b32_e32 v2, 0xfc, v2                                 // 000000001938: 260404FF 000000FC
	v_mov_b32_e32 v3, 0                                        // 000000001940: 7E060280
	v_lshlrev_b32_e32 v2, 1, v2                                // 000000001944: 24040481
	v_cmp_gt_i32_e64 s[16:17], s18, v1                         // 000000001948: D0C40010 00020212
	v_lshl_add_u64 v[16:17], v[4:5], 0, v[2:3]                 // 000000001950: D2080010 04090104
	v_mov_b32_e32 v10, v3                                      // 000000001958: 7E140303
	v_mov_b32_e32 v11, v3                                      // 00000000195C: 7E160303
	s_and_saveexec_b64 s[2:3], s[16:17]                        // 000000001960: BE822010
	s_cbranch_execz 2                                          // 000000001964: BF880002 <c662_rmsnorm_fwd_mixnorm+0x170>
	global_load_dwordx2 v[10:11], v[16:17], off                // 000000001968: DC548000 0A7F0010
	s_or_b64 exec, exec, s[2:3]                                // 000000001970: 87FE027E
	v_mul_lo_u32 v4, s21, v1                                   // 000000001974: D2850004 00020215
	v_ashrrev_i32_e32 v5, 31, v4                               // 00000000197C: 220A089F
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]                   // 000000001980: D2080004 00110304
	v_lshl_add_u64 v[18:19], v[4:5], 0, v[2:3]                 // 000000001988: D2080012 04090104
	v_mov_b32_e32 v15, 0                                       // 000000001990: 7E1E0280
	s_and_saveexec_b64 s[2:3], s[16:17]                        // 000000001994: BE822010
	s_cbranch_execz 4                                          // 000000001998: BF880004 <c662_rmsnorm_fwd_mixnorm+0x1ac>
	global_load_dwordx2 v[14:15], v[18:19], off                // 00000000199C: DC548000 0E7F0012
	s_waitcnt vmcnt(0)                                         // 0000000019A4: BF8C0F70
	v_mov_b32_e32 v3, v14                                      // 0000000019A8: 7E06030E
	s_or_b64 exec, exec, s[2:3]                                // 0000000019AC: 87FE027E
	v_mov_b32_e32 v8, 0                                        // 0000000019B0: 7E100280
	v_mov_b32_e32 v12, 0                                       // 0000000019B4: 7E180280
	v_mov_b32_e32 v13, 0                                       // 0000000019B8: 7E1A0280
	s_and_saveexec_b64 s[2:3], s[16:17]                        // 0000000019BC: BE822010
	s_cbranch_execz 2                                          // 0000000019C0: BF880002 <c662_rmsnorm_fwd_mixnorm+0x1cc>
	global_load_dwordx2 v[12:13], v[16:17], off offset:512     // 0000000019C4: DC548200 0C7F0010
	s_or_b64 exec, exec, s[2:3]                                // 0000000019CC: 87FE027E
	v_mov_b32_e32 v9, 0                                        // 0000000019D0: 7E120280
	s_and_saveexec_b64 s[2:3], s[16:17]                        // 0000000019D4: BE822010
	s_cbranch_execz 2                                          // 0000000019D8: BF880002 <c662_rmsnorm_fwd_mixnorm+0x1e4>
	global_load_dwordx2 v[8:9], v[18:19], off offset:512       // 0000000019DC: DC548200 087F0012
	s_or_b64 exec, exec, s[2:3]                                // 0000000019E4: 87FE027E
	v_mov_b32_e32 v4, 0                                        // 0000000019E8: 7E080280
	v_mov_b32_e32 v6, 0                                        // 0000000019EC: 7E0C0280
	v_mov_b32_e32 v7, 0                                        // 0000000019F0: 7E0E0280
	s_and_saveexec_b64 s[2:3], s[16:17]                        // 0000000019F4: BE822010
	s_cbranch_execz 2                                          // 0000000019F8: BF880002 <c662_rmsnorm_fwd_mixnorm+0x204>
	global_load_dwordx2 v[6:7], v[16:17], off offset:1024      // 0000000019FC: DC548400 067F0010
	s_or_b64 exec, exec, s[2:3]                                // 000000001A04: 87FE027E
	s_load_dword s19, s[0:1], 0x44                             // 000000001A08: C00204C0 00000044
	s_load_dword s2, s[6:7], 0x0                               // 000000001A10: C0020083 00000000
	s_load_dword s4, s[8:9], 0x0                               // 000000001A18: C0020104 00000000
	v_mov_b32_e32 v5, 0                                        // 000000001A20: 7E0A0280
	s_and_saveexec_b64 s[6:7], s[16:17]                        // 000000001A24: BE862010
	s_cbranch_execz 2                                          // 000000001A28: BF880002 <c662_rmsnorm_fwd_mixnorm+0x234>
	global_load_dwordx2 v[4:5], v[18:19], off offset:1024      // 000000001A2C: DC548400 047F0012
	s_or_b64 exec, exec, s[6:7]                                // 000000001A34: 87FE067E
	s_waitcnt lgkmcnt(0)                                       // 000000001A38: BF8CC07F
	s_mov_b32 s5, s4                                           // 000000001A3C: BE850004
	s_waitcnt vmcnt(0)                                         // 000000001A40: BF8C0F70
	v_and_b32_e32 v17, 0xffff0000, v10                         // 000000001A44: 262214FF FFFF0000
	v_lshlrev_b32_e32 v16, 16, v10                             // 000000001A4C: 24201490
	v_and_b32_e32 v19, 0xffff0000, v11                         // 000000001A50: 262616FF FFFF0000
	v_lshlrev_b32_e32 v18, 16, v11                             // 000000001A58: 24241690
	v_and_b32_e32 v11, 0xffff0000, v3                          // 000000001A5C: 261606FF FFFF0000
	v_lshlrev_b32_e32 v10, 16, v3                              // 000000001A64: 24140690
	s_mov_b32 s3, s2                                           // 000000001A68: BE830002
	v_pk_mul_f32 v[10:11], s[4:5], v[10:11] op_sel_hi:[0,1]    // 000000001A6C: D3B1400A 10021404
	v_pk_fma_f32 v[10:11], s[2:3], v[16:17], v[10:11] op_sel_hi:[0,1,1]// 000000001A74: D3B0400A 142A2002
	v_bfe_u32 v3, v10, 16, 1                                   // 000000001A7C: D1C80003 0205210A
	s_movk_i32 s8, 0x7fff                                      // 000000001A84: B0087FFF
	v_bfe_u32 v14, v11, 16, 1                                  // 000000001A88: D1C8000E 0205210B
	v_add3_u32 v3, v10, v3, s8                                 // 000000001A90: D1FF0003 0022070A
	v_add3_u32 v14, v11, v14, s8                               // 000000001A98: D1FF000E 00221D0B
	v_lshrrev_b32_e32 v3, 16, v3                               // 000000001AA0: 20060690
	v_mov_b32_e32 v16, 0x7fff                                  // 000000001AA4: 7E2002FF 00007FFF
	v_cmp_o_f32_e32 vcc, v10, v10                              // 000000001AAC: 7C8E150A
	v_and_b32_e32 v21, 0xffff0000, v15                         // 000000001AB0: 262A1EFF FFFF0000
	v_lshlrev_b32_e32 v20, 16, v15                             // 000000001AB8: 24281E90
	v_lshrrev_b32_e32 v14, 16, v14                             // 000000001ABC: 201C1C90
	v_cndmask_b32_e32 v3, v16, v3, vcc                         // 000000001AC0: 00060710
	v_cmp_o_f32_e32 vcc, v11, v11                              // 000000001AC4: 7C8E170B
	s_mov_b32 s9, 0x5040100                                    // 000000001AC8: BE8900FF 05040100
	s_nop 0                                                    // 000000001AD0: BF800000
	v_cndmask_b32_e32 v10, v16, v14, vcc                       // 000000001AD4: 00141D10
	v_pk_mul_f32 v[14:15], s[4:5], v[20:21] op_sel_hi:[0,1]    // 000000001AD8: D3B1400E 10022804
	v_pk_fma_f32 v[14:15], s[2:3], v[18:19], v[14:15] op_sel_hi:[0,1,1]// 000000001AE0: D3B0400E 143A2402
	v_perm_b32 v10, v10, v3, s9                                // 000000001AE8: D1ED000A 0026070A
	v_bfe_u32 v3, v14, 16, 1                                   // 000000001AF0: D1C80003 0205210E
	v_bfe_u32 v11, v15, 16, 1                                  // 000000001AF8: D1C8000B 0205210F
	v_add3_u32 v3, v14, v3, s8                                 // 000000001B00: D1FF0003 0022070E
	v_add3_u32 v11, v15, v11, s8                               // 000000001B08: D1FF000B 0022170F
	v_lshrrev_b32_e32 v3, 16, v3                               // 000000001B10: 20060690
	v_cmp_o_f32_e32 vcc, v14, v14                              // 000000001B14: 7C8E1D0E
	v_lshrrev_b32_e32 v11, 16, v11                             // 000000001B18: 20161690
	v_mul_lo_u32 v14, s19, v1                                  // 000000001B1C: D285000E 00020213
	v_cndmask_b32_e32 v3, v16, v3, vcc                         // 000000001B24: 00060710
	v_cmp_o_f32_e32 vcc, v15, v15                              // 000000001B28: 7C8E1F0F
	v_ashrrev_i32_e32 v15, 31, v14                             // 000000001B2C: 221E1C9F
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[10:11]             // 000000001B30: D208000E 0029030E
	v_cndmask_b32_e32 v11, v16, v11, vcc                       // 000000001B38: 00161710
	v_perm_b32 v11, v11, v3, s9                                // 000000001B3C: D1ED000B 0026070B
	v_mov_b32_e32 v3, 0                                        // 000000001B44: 7E060280
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[2:3]               // 000000001B48: D208000E 0409010E
	s_and_saveexec_b64 s[6:7], s[16:17]                        // 000000001B50: BE862010
	s_cbranch_execz 2                                          // 000000001B54: BF880002 <c662_rmsnorm_fwd_mixnorm+0x360>
	global_store_dwordx2 v[14:15], v[10:11], off               // 000000001B58: DC748000 007F0A0E
	s_or_b64 exec, exec, s[6:7]                                // 000000001B60: 87FE067E
	v_and_b32_e32 v19, 0xffff0000, v12                         // 000000001B64: 262618FF FFFF0000
	v_lshlrev_b32_e32 v18, 16, v12                             // 000000001B6C: 24241890
	v_and_b32_e32 v21, 0xffff0000, v13                         // 000000001B70: 262A1AFF FFFF0000
	v_lshlrev_b32_e32 v20, 16, v13                             // 000000001B78: 24281A90
	v_and_b32_e32 v13, 0xffff0000, v8                          // 000000001B7C: 261A10FF FFFF0000
	v_lshlrev_b32_e32 v12, 16, v8                              // 000000001B84: 24181090
	v_and_b32_e32 v23, 0xffff0000, v9                          // 000000001B88: 262E12FF FFFF0000
	v_lshlrev_b32_e32 v22, 16, v9                              // 000000001B90: 242C1290
	v_pk_mul_f32 v[8:9], s[4:5], v[12:13]                      // 000000001B94: D3B14008 18021804
	s_nop 0                                                    // 000000001B9C: BF800000
	v_pk_fma_f32 v[8:9], s[2:3], v[18:19], v[8:9]              // 000000001BA0: D3B04008 1C222402
	s_nop 0                                                    // 000000001BA8: BF800000
	v_bfe_u32 v3, v8, 16, 1                                    // 000000001BAC: D1C80003 02052108
	v_bfe_u32 v12, v9, 16, 1                                   // 000000001BB4: D1C8000C 02052109
	v_add3_u32 v3, v8, v3, s8                                  // 000000001BBC: D1FF0003 00220708
	v_add3_u32 v12, v9, v12, s8                                // 000000001BC4: D1FF000C 00221909
	v_lshrrev_b32_e32 v3, 16, v3                               // 000000001BCC: 20060690
	v_cmp_o_f32_e32 vcc, v8, v8                                // 000000001BD0: 7C8E1108
	v_lshrrev_b32_e32 v12, 16, v12                             // 000000001BD4: 20181890
	s_nop 0                                                    // 000000001BD8: BF800000
	v_cndmask_b32_e32 v3, v16, v3, vcc                         // 000000001BDC: 00060710
	v_cmp_o_f32_e32 vcc, v9, v9                                // 000000001BE0: 7C8E1309
	s_nop 1                                                    // 000000001BE4: BF800001
	v_cndmask_b32_e32 v8, v16, v12, vcc                        // 000000001BE8: 00101910
	v_pk_mul_f32 v[12:13], s[4:5], v[22:23]                    // 000000001BEC: D3B1400C 18022C04
	v_perm_b32 v8, v8, v3, s9                                  // 000000001BF4: D1ED0008 00260708
	v_pk_fma_f32 v[12:13], s[2:3], v[20:21], v[12:13]          // 000000001BFC: D3B0400C 1C322802
	s_nop 0                                                    // 000000001C04: BF800000
	v_bfe_u32 v3, v12, 16, 1                                   // 000000001C08: D1C80003 0205210C
	v_bfe_u32 v9, v13, 16, 1                                   // 000000001C10: D1C80009 0205210D
	v_add3_u32 v3, v12, v3, s8                                 // 000000001C18: D1FF0003 0022070C
	v_cmp_o_f32_e32 vcc, v12, v12                              // 000000001C20: 7C8E190C
	v_add3_u32 v9, v13, v9, s8                                 // 000000001C24: D1FF0009 0022130D
	s_nop 0                                                    // 000000001C2C: BF800000
	v_cndmask_b32_sdwa v3, v16, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001C30: 000606F9 05060610
	v_cmp_o_f32_e32 vcc, v13, v13                              // 000000001C38: 7C8E1B0D
	s_nop 1                                                    // 000000001C3C: BF800001
	v_cndmask_b32_sdwa v9, v16, v9, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001C40: 001212F9 05060610
	v_perm_b32 v9, v9, v3, s9                                  // 000000001C48: D1ED0009 00260709
	s_and_saveexec_b64 s[6:7], s[16:17]                        // 000000001C50: BE862010
	s_cbranch_execz 2                                          // 000000001C54: BF880002 <c662_rmsnorm_fwd_mixnorm+0x460>
	global_store_dwordx2 v[14:15], v[8:9], off offset:512      // 000000001C58: DC748200 007F080E
	s_or_b64 exec, exec, s[6:7]                                // 000000001C60: 87FE067E
	v_and_b32_e32 v13, 0xffff0000, v6                          // 000000001C64: 261A0CFF FFFF0000
	v_lshlrev_b32_e32 v12, 16, v6                              // 000000001C6C: 24180C90
	v_and_b32_e32 v17, 0xffff0000, v7                          // 000000001C70: 26220EFF FFFF0000
	v_lshlrev_b32_e32 v16, 16, v7                              // 000000001C78: 24200E90
	v_and_b32_e32 v7, 0xffff0000, v4                           // 000000001C7C: 260E08FF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v4                               // 000000001C84: 240C0890
	v_and_b32_e32 v21, 0xffff0000, v5                          // 000000001C88: 262A0AFF FFFF0000
	v_lshlrev_b32_e32 v20, 16, v5                              // 000000001C90: 24280A90
	v_pk_mul_f32 v[4:5], s[4:5], v[6:7]                        // 000000001C94: D3B14004 18020C04
	v_mov_b32_e32 v7, 0x7fff                                   // 000000001C9C: 7E0E02FF 00007FFF
	v_pk_fma_f32 v[4:5], s[2:3], v[12:13], v[4:5]              // 000000001CA4: D3B04004 1C121802
	s_mov_b32 s7, 0x5040100                                    // 000000001CAC: BE8700FF 05040100
	v_bfe_u32 v3, v4, 16, 1                                    // 000000001CB4: D1C80003 02052104
	v_bfe_u32 v6, v5, 16, 1                                    // 000000001CBC: D1C80006 02052105
	v_add3_u32 v3, v4, v3, s8                                  // 000000001CC4: D1FF0003 00220704
	v_add3_u32 v6, v5, v6, s8                                  // 000000001CCC: D1FF0006 00220D05
	v_lshrrev_b32_e32 v3, 16, v3                               // 000000001CD4: 20060690
	v_cmp_o_f32_e32 vcc, v4, v4                                // 000000001CD8: 7C8E0904
	v_lshrrev_b32_e32 v6, 16, v6                               // 000000001CDC: 200C0C90
	s_load_dword s6, s[0:1], 0x4c                              // 000000001CE0: C0020180 0000004C
	v_cndmask_b32_e32 v3, v7, v3, vcc                          // 000000001CE8: 00060707
	v_cmp_o_f32_e32 vcc, v5, v5                                // 000000001CEC: 7C8E0B05
	s_nop 1                                                    // 000000001CF0: BF800001
	v_cndmask_b32_e32 v4, v7, v6, vcc                          // 000000001CF4: 00080D07
	v_perm_b32 v18, v4, v3, s7                                 // 000000001CF8: D1ED0012 001E0704
	v_pk_mul_f32 v[4:5], s[4:5], v[20:21]                      // 000000001D00: D3B14004 18022804
	s_nop 0                                                    // 000000001D08: BF800000
	v_pk_fma_f32 v[4:5], s[2:3], v[16:17], v[4:5]              // 000000001D0C: D3B04004 1C122002
	s_nop 0                                                    // 000000001D14: BF800000
	v_bfe_u32 v3, v4, 16, 1                                    // 000000001D18: D1C80003 02052104
	v_bfe_u32 v6, v5, 16, 1                                    // 000000001D20: D1C80006 02052105
	v_add3_u32 v3, v4, v3, s8                                  // 000000001D28: D1FF0003 00220704
	v_cmp_o_f32_e32 vcc, v4, v4                                // 000000001D30: 7C8E0904
	v_add3_u32 v6, v5, v6, s8                                  // 000000001D34: D1FF0006 00220D05
	s_nop 0                                                    // 000000001D3C: BF800000
	v_cndmask_b32_sdwa v3, v7, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D40: 000606F9 05060607
	v_cmp_o_f32_e32 vcc, v5, v5                                // 000000001D48: 7C8E0B05
	s_nop 1                                                    // 000000001D4C: BF800001
	v_cndmask_b32_sdwa v4, v7, v6, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001D50: 00080CF9 05060607
	v_perm_b32 v19, v4, v3, s7                                 // 000000001D58: D1ED0013 001E0704
	s_and_saveexec_b64 s[2:3], s[16:17]                        // 000000001D60: BE822010
	s_cbranch_execz 2                                          // 000000001D64: BF880002 <c662_rmsnorm_fwd_mixnorm+0x570>
	global_store_dwordx2 v[14:15], v[18:19], off offset:1024   // 000000001D68: DC748400 007F120E
	s_or_b64 exec, exec, s[2:3]                                // 000000001D70: 87FE027E
	v_and_b32_e32 v15, 0xffff0000, v10                         // 000000001D74: 261E14FF FFFF0000
	v_lshlrev_b32_e32 v14, 16, v10                             // 000000001D7C: 241C1490
	v_and_b32_e32 v17, 0xffff0000, v11                         // 000000001D80: 262216FF FFFF0000
	v_lshlrev_b32_e32 v16, 16, v11                             // 000000001D88: 24201690
	v_and_b32_e32 v11, 0xffff0000, v8                          // 000000001D8C: 261610FF FFFF0000
	v_lshlrev_b32_e32 v10, 16, v8                              // 000000001D94: 24141090
	v_and_b32_e32 v13, 0xffff0000, v9                          // 000000001D98: 261A12FF FFFF0000
	v_lshlrev_b32_e32 v12, 16, v9                              // 000000001DA0: 24181290
	v_pk_mul_f32 v[4:5], v[10:11], v[10:11]                    // 000000001DA4: D3B14004 1802150A
	v_pk_mul_f32 v[6:7], v[12:13], v[12:13]                    // 000000001DAC: D3B14006 1802190C
	v_pk_fma_f32 v[8:9], v[14:15], v[14:15], v[4:5]            // 000000001DB4: D3B04008 1C121D0E
	v_and_b32_e32 v5, 0xffff0000, v18                          // 000000001DBC: 260A24FF FFFF0000
	v_lshlrev_b32_e32 v4, 16, v18                              // 000000001DC4: 24082490
	v_pk_fma_f32 v[20:21], v[16:17], v[16:17], v[6:7]          // 000000001DC8: D3B04014 1C1A2110
	v_and_b32_e32 v7, 0xffff0000, v19                          // 000000001DD0: 260E26FF FFFF0000
	v_lshlrev_b32_e32 v6, 16, v19                              // 000000001DD8: 240C2690
	v_pk_fma_f32 v[8:9], v[4:5], v[4:5], v[8:9]                // 000000001DDC: D3B04008 1C220904
	v_pk_fma_f32 v[18:19], v[6:7], v[6:7], v[20:21]            // 000000001DE4: D3B04012 1C520D06
	v_add_f32_e32 v3, v8, v9                                   // 000000001DEC: 02061308
	v_add_f32_e32 v3, v18, v3                                  // 000000001DF0: 02060712
	v_add_f32_e32 v3, v19, v3                                  // 000000001DF4: 02060713
	s_mov_b32 s5, 0x44400000                                   // 000000001DF8: BE8500FF 44400000
	s_nop 0                                                    // 000000001E00: BF800000
	v_add_f32_dpp v3, v3, v3 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001E04: 020606FA FF091803
	s_nop 1                                                    // 000000001E0C: BF800001
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001E10: 020606FA FF091403
	s_nop 1                                                    // 000000001E18: BF800001
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001E1C: 020606FA FF091203
	s_nop 1                                                    // 000000001E24: BF800001
	v_add_f32_dpp v3, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001E28: 020606FA FF091103
	v_mov_b32_e32 v8, v3                                       // 000000001E30: 7E100303
	s_nop 1                                                    // 000000001E34: BF800001
	v_mov_b32_dpp v8, v8 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1// 000000001E38: 7E1002FA AF094208
	v_add_f32_e32 v3, v3, v8                                   // 000000001E40: 02061103
	s_nop 1                                                    // 000000001E44: BF800001
	v_add_f32_dpp v3, v3, v3 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001E48: 020606FA FF094303
	s_nop 0                                                    // 000000001E50: BF800000
	v_readlane_b32 s4, v3, 63                                  // 000000001E54: D2890004 00017F03
	s_nop 1                                                    // 000000001E5C: BF800001
	v_mov_b32_e32 v3, s4                                       // 000000001E60: 7E060204
	v_div_scale_f32 v8, s[2:3], s5, s5, v3                     // 000000001E64: D1E00208 040C0A05
	v_rcp_f32_e32 v9, v8                                       // 000000001E6C: 7E124508
	s_nop 0                                                    // 000000001E70: BF800000
	v_fma_f32 v18, -v8, v9, 1.0                                // 000000001E74: D1CB0012 23CA1308
	v_fmac_f32_e32 v9, v18, v9                                 // 000000001E7C: 76121312
	v_mov_b32_e32 v18, 0x44400000                              // 000000001E80: 7E2402FF 44400000
	v_div_scale_f32 v18, vcc, s4, v18, s4                      // 000000001E88: D1E06A12 00122404
	v_mul_f32_e32 v19, v18, v9                                 // 000000001E90: 0A261312
	v_fma_f32 v20, -v8, v19, v18                               // 000000001E94: D1CB0014 244A2708
	v_fmac_f32_e32 v19, v20, v9                                // 000000001E9C: 76261314
	v_fma_f32 v8, -v8, v19, v18                                // 000000001EA0: D1CB0008 244A2708
	v_div_fmas_f32 v8, v8, v9, v19                             // 000000001EA8: D1E20008 044E1308
	v_div_fixup_f32 v3, v8, s5, v3                             // 000000001EB0: D1DE0003 040C0B08
	s_waitcnt lgkmcnt(0)                                       // 000000001EB8: BF8CC07F
	v_add_f32_e32 v3, s6, v3                                   // 000000001EBC: 02060606
	v_rsq_f32_e32 v8, v3                                       // 000000001EC0: 7E104903
	v_and_b32_e32 v9, 0xc0, v0                                 // 000000001EC4: 261200FF 000000C0
	s_and_saveexec_b64 s[2:3], s[16:17]                        // 000000001ECC: BE822010
	s_cbranch_execz 131                                        // 000000001ED0: BF880083 <c662_rmsnorm_fwd_mixnorm+0x8e0>
	s_load_dword s0, s[0:1], 0x48                              // 000000001ED4: C0020000 00000048
	v_pk_mul_f32 v[16:17], v[8:9], v[16:17] op_sel_hi:[0,1]    // 000000001EDC: D3B14010 10022108
	v_bfe_u32 v20, v16, 16, 1                                  // 000000001EE4: D1C80014 02052110
	v_mov_b32_e32 v3, 0                                        // 000000001EEC: 7E060280
	v_pk_mul_f32 v[14:15], v[8:9], v[14:15] op_sel_hi:[0,1]    // 000000001EF0: D3B1400E 10021D08
	s_waitcnt lgkmcnt(0)                                       // 000000001EF8: BF8CC07F
	v_mul_lo_u32 v18, s0, v1                                   // 000000001EFC: D2850012 00020200
	v_ashrrev_i32_e32 v19, 31, v18                             // 000000001F04: 2226249F
	s_movk_i32 s0, 0x7fff                                      // 000000001F08: B0007FFF
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[12:13]             // 000000001F0C: D2080012 00310312
	v_bfe_u32 v1, v17, 16, 1                                   // 000000001F14: D1C80001 02052111
	v_add3_u32 v20, v16, v20, s0                               // 000000001F1C: D1FF0014 00022910
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[2:3]                 // 000000001F24: D2080002 04090112
	v_bfe_u32 v18, v14, 16, 1                                  // 000000001F2C: D1C80012 0205210E
	v_add3_u32 v1, v17, v1, s0                                 // 000000001F34: D1FF0001 00020311
	v_lshrrev_b32_e32 v20, 16, v20                             // 000000001F3C: 20282890
	v_mov_b32_e32 v21, 0x7fff                                  // 000000001F40: 7E2A02FF 00007FFF
	v_cmp_o_f32_e32 vcc, v16, v16                              // 000000001F48: 7C8E2110
	v_bfe_u32 v19, v15, 16, 1                                  // 000000001F4C: D1C80013 0205210F
	v_add3_u32 v18, v14, v18, s0                               // 000000001F54: D1FF0012 0002250E
	v_lshrrev_b32_e32 v1, 16, v1                               // 000000001F5C: 20020290
	v_cndmask_b32_e32 v16, v21, v20, vcc                       // 000000001F60: 00202915
	v_cmp_o_f32_e32 vcc, v17, v17                              // 000000001F64: 7C8E2311
	v_add3_u32 v19, v15, v19, s0                               // 000000001F68: D1FF0013 0002270F
	v_lshrrev_b32_e32 v18, 16, v18                             // 000000001F70: 20242490
	v_cndmask_b32_e32 v1, v21, v1, vcc                         // 000000001F74: 00020315
	s_mov_b32 s1, 0x5040100                                    // 000000001F78: BE8100FF 05040100
	v_cmp_o_f32_e32 vcc, v14, v14                              // 000000001F80: 7C8E1D0E
	v_lshrrev_b32_e32 v19, 16, v19                             // 000000001F84: 20262690
	v_perm_b32 v17, v1, v16, s1                                // 000000001F88: D1ED0011 00062101
	v_cndmask_b32_e32 v1, v21, v18, vcc                        // 000000001F90: 00022515
	v_cmp_o_f32_e32 vcc, v15, v15                              // 000000001F94: 7C8E1F0F
	v_pk_mul_f32 v[12:13], v[8:9], v[12:13] op_sel_hi:[0,1]    // 000000001F98: D3B1400C 10021908
	v_pk_mul_f32 v[10:11], v[8:9], v[10:11] op_sel_hi:[0,1]    // 000000001FA0: D3B1400A 10021508
	v_cndmask_b32_e32 v14, v21, v19, vcc                       // 000000001FA8: 001C2715
	v_perm_b32 v16, v14, v1, s1                                // 000000001FAC: D1ED0010 0006030E
	v_bfe_u32 v1, v12, 16, 1                                   // 000000001FB4: D1C80001 0205210C
	v_bfe_u32 v14, v13, 16, 1                                  // 000000001FBC: D1C8000E 0205210D
	v_add3_u32 v1, v12, v1, s0                                 // 000000001FC4: D1FF0001 0002030C
	v_bfe_u32 v15, v10, 16, 1                                  // 000000001FCC: D1C8000F 0205210A
	v_add3_u32 v14, v13, v14, s0                               // 000000001FD4: D1FF000E 00021D0D
	v_lshrrev_b32_e32 v1, 16, v1                               // 000000001FDC: 20020290
	v_cmp_o_f32_e32 vcc, v12, v12                              // 000000001FE0: 7C8E190C
	global_store_dwordx2 v[2:3], v[16:17], off                 // 000000001FE4: DC748000 007F1002
	v_bfe_u32 v16, v11, 16, 1                                  // 000000001FEC: D1C80010 0205210B
	v_add3_u32 v15, v10, v15, s0                               // 000000001FF4: D1FF000F 00021F0A
	v_lshrrev_b32_e32 v14, 16, v14                             // 000000001FFC: 201C1C90
	v_cndmask_b32_e32 v1, v21, v1, vcc                         // 000000002000: 00020315
	v_cmp_o_f32_e32 vcc, v13, v13                              // 000000002004: 7C8E1B0D
	v_add3_u32 v16, v11, v16, s0                               // 000000002008: D1FF0010 0002210B
	v_lshrrev_b32_e32 v15, 16, v15                             // 000000002010: 201E1E90
	v_cndmask_b32_e32 v12, v21, v14, vcc                       // 000000002014: 00181D15
	v_cmp_o_f32_e32 vcc, v10, v10                              // 000000002018: 7C8E150A
	v_lshrrev_b32_e32 v16, 16, v16                             // 00000000201C: 20202090
	v_perm_b32 v13, v12, v1, s1                                // 000000002020: D1ED000D 0006030C
	v_cndmask_b32_e32 v1, v21, v15, vcc                        // 000000002028: 00021F15
	v_cmp_o_f32_e32 vcc, v11, v11                              // 00000000202C: 7C8E170B
	v_pk_mul_f32 v[6:7], v[8:9], v[6:7] op_sel_hi:[0,1]        // 000000002030: D3B14006 10020D08
	v_pk_mul_f32 v[4:5], v[8:9], v[4:5] op_sel_hi:[0,1]        // 000000002038: D3B14004 10020908
	v_cndmask_b32_e32 v10, v21, v16, vcc                       // 000000002040: 00142115
	v_perm_b32 v12, v10, v1, s1                                // 000000002044: D1ED000C 0006030A
	v_bfe_u32 v1, v6, 16, 1                                    // 00000000204C: D1C80001 02052106
	v_bfe_u32 v10, v7, 16, 1                                   // 000000002054: D1C8000A 02052107
	v_add3_u32 v1, v6, v1, s0                                  // 00000000205C: D1FF0001 00020306
	v_cmp_o_f32_e32 vcc, v6, v6                                // 000000002064: 7C8E0D06
	v_bfe_u32 v11, v4, 16, 1                                   // 000000002068: D1C8000B 02052104
	v_add3_u32 v10, v7, v10, s0                                // 000000002070: D1FF000A 00021507
	v_cndmask_b32_sdwa v1, v21, v1, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002078: 000202F9 05060615
	v_cmp_o_f32_e32 vcc, v7, v7                                // 000000002080: 7C8E0F07
	global_store_dwordx2 v[2:3], v[12:13], off offset:512      // 000000002084: DC748200 007F0C02
	v_bfe_u32 v12, v5, 16, 1                                   // 00000000208C: D1C8000C 02052105
	v_add3_u32 v11, v4, v11, s0                                // 000000002094: D1FF000B 00021704
	v_cndmask_b32_sdwa v6, v21, v10, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000209C: 000C14F9 05060615
	v_cmp_o_f32_e32 vcc, v4, v4                                // 0000000020A4: 7C8E0904
	v_add3_u32 v12, v5, v12, s0                                // 0000000020A8: D1FF000C 00021905
	v_perm_b32 v7, v6, v1, s1                                  // 0000000020B0: D1ED0007 00060306
	v_cndmask_b32_sdwa v1, v21, v11, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000020B8: 000216F9 05060615
	v_cmp_o_f32_e32 vcc, v5, v5                                // 0000000020C0: 7C8E0B05
	s_nop 1                                                    // 0000000020C4: BF800001
	v_cndmask_b32_sdwa v4, v21, v12, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000020C8: 000818F9 05060615
	v_perm_b32 v6, v4, v1, s1                                  // 0000000020D0: D1ED0006 00060304
	global_store_dwordx2 v[2:3], v[6:7], off offset:1024       // 0000000020D8: DC748400 007F0602
	s_or_b64 exec, exec, s[2:3]                                // 0000000020E0: 87FE027E
	v_and_b32_e32 v1, 3, v0                                    // 0000000020E4: 26020083
	v_or_b32_e32 v2, s20, v1                                   // 0000000020E8: 28040214
	v_and_b32_e32 v0, 0xfc, v0                                 // 0000000020EC: 260000FF 000000FC
	v_cmp_gt_i32_e32 vcc, s18, v2                              // 0000000020F4: 7D880412
	v_lshrrev_b32_e32 v3, 4, v9                                // 0000000020F8: 20061284
	v_cmp_eq_u32_e64 s[0:1], 0, v0                             // 0000000020FC: D0CA0000 00020080
	v_add_u32_e32 v3, 0, v3                                    // 000000002104: 68060680
	s_and_b64 s[0:1], s[0:1], vcc                              // 000000002108: 86806A00
	ds_write_b32 v3, v8                                        // 00000000210C: D81A0000 00000803
	s_waitcnt lgkmcnt(0)                                       // 000000002114: BF8CC07F
	s_barrier                                                  // 000000002118: BF8A0000
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 00000000211C: BE822000
	s_cbranch_execz 10                                         // 000000002120: BF88000A <c662_rmsnorm_fwd_mixnorm+0x94c>
	v_lshl_add_u32 v0, v1, 2, 0                                // 000000002124: D1FD0000 02010501
	ds_read_b32 v4, v0                                         // 00000000212C: D86C0000 04000000
	v_ashrrev_i32_e32 v3, 31, v2                               // 000000002134: 2206049F
	v_lshl_add_u64 v[0:1], v[2:3], 2, s[14:15]                 // 000000002138: D2080000 00390502
	s_waitcnt lgkmcnt(0)                                       // 000000002140: BF8CC07F
	global_store_dword v[0:1], v4, off                         // 000000002144: DC708000 007F0400
	s_endpgm                                                   // 00000000214C: BF810000
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
