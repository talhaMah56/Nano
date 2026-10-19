
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_v1_768/gfx908/xc-nw4_wpe1_h1/c662_rmsnorm_fwd_v1.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <c662_rmsnorm_fwd_v1>:
	v_lshlrev_b32_e32 v6, 2, v0                                // 000000001700: 240C0082
	s_movk_i32 s0, 0x300                                       // 000000001704: B0000300
	v_cmp_gt_u32_e64 s[0:1], s0, v6                            // 000000001708: D0CC0000 00020C00
	v_mov_b32_e32 v1, 0                                        // 000000001710: 7E020280
	v_lshlrev_b32_e32 v3, 1, v6                                // 000000001714: 24060C81
	v_mov_b32_e32 v2, 0                                        // 000000001718: 7E040280
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 00000000171C: BE822000
	s_cbranch_execz 12                                         // 000000001720: BF88000C <c662_rmsnorm_fwd_v1+0x54>
	s_load_dword s7, s[4:5], 0x18                              // 000000001724: C00201C2 00000018
	s_load_dwordx2 s[8:9], s[4:5], 0x0                         // 00000000172C: C0060202 00000000
	s_waitcnt lgkmcnt(0)                                       // 000000001734: BF8CC07F
	s_mul_i32 s10, s7, s6                                      // 000000001738: 920A0607
	s_ashr_i32 s11, s10, 31                                    // 00000000173C: 900B9F0A
	s_lshl_b64 s[10:11], s[10:11], 1                           // 000000001740: 8E8A810A
	s_add_u32 s8, s8, s10                                      // 000000001744: 80080A08
	s_addc_u32 s9, s9, s11                                     // 000000001748: 82090B09
	global_load_dwordx2 v[1:2], v3, s[8:9]                     // 00000000174C: DC548000 01080003
	s_or_b64 exec, exec, s[2:3]                                // 000000001754: 87FE027E
	s_waitcnt vmcnt(0)                                         // 000000001758: BF8C0F70
	v_and_b32_e32 v4, 0xffff0000, v1                           // 00000000175C: 260802FF FFFF0000
	v_lshlrev_b32_e32 v1, 16, v1                               // 000000001764: 24020290
	v_mul_f32_e32 v7, v4, v4                                   // 000000001768: 0A0E0904
	v_and_b32_e32 v5, 0xffff0000, v2                           // 00000000176C: 260A04FF FFFF0000
	v_lshlrev_b32_e32 v2, 16, v2                               // 000000001774: 24040490
	v_fmac_f32_e32 v7, v1, v1                                  // 000000001778: 760E0301
	v_fmac_f32_e32 v7, v2, v2                                  // 00000000177C: 760E0502
	v_fmac_f32_e32 v7, v5, v5                                  // 000000001780: 760E0B05
	v_xor_b32_e32 v6, 0x80, v6                                 // 000000001784: 2A0C0CFF 00000080
	ds_bpermute_b32 v6, v6, v7                                 // 00000000178C: D87E0000 06000706
	v_and_b32_e32 v8, 63, v0                                   // 000000001794: 261000BF
	v_cmp_eq_u32_e32 vcc, 0, v8                                // 000000001798: 7D941080
	s_waitcnt lgkmcnt(0)                                       // 00000000179C: BF8CC07F
	v_add_f32_e32 v6, v7, v6                                   // 0000000017A0: 020C0D07
	ds_swizzle_b32 v7, v6 offset:swizzle(SWAP,16)              // 0000000017A4: D87A401F 07000006
	s_waitcnt lgkmcnt(0)                                       // 0000000017AC: BF8CC07F
	v_add_f32_e32 v6, v6, v7                                   // 0000000017B0: 020C0F06
	ds_swizzle_b32 v7, v6 offset:swizzle(SWAP,8)               // 0000000017B4: D87A201F 07000006
	s_waitcnt lgkmcnt(0)                                       // 0000000017BC: BF8CC07F
	v_add_f32_e32 v6, v6, v7                                   // 0000000017C0: 020C0F06
	ds_swizzle_b32 v7, v6 offset:swizzle(SWAP,4)               // 0000000017C4: D87A101F 07000006
	s_waitcnt lgkmcnt(0)                                       // 0000000017CC: BF8CC07F
	v_add_f32_e32 v6, v6, v7                                   // 0000000017D0: 020C0F06
	ds_swizzle_b32 v7, v6 offset:swizzle(SWAP,2)               // 0000000017D4: D87A081F 07000006
	s_waitcnt lgkmcnt(0)                                       // 0000000017DC: BF8CC07F
	v_add_f32_e32 v6, v6, v7                                   // 0000000017E0: 020C0F06
	ds_swizzle_b32 v7, v6 offset:swizzle(SWAP,1)               // 0000000017E4: D87A041F 07000006
	s_and_saveexec_b64 s[2:3], vcc                             // 0000000017EC: BE82206A
	s_cbranch_execz 7                                          // 0000000017F0: BF880007 <c662_rmsnorm_fwd_v1+0x110>
	s_waitcnt lgkmcnt(0)                                       // 0000000017F4: BF8CC07F
	v_add_f32_e32 v6, v6, v7                                   // 0000000017F8: 020C0F06
	v_lshrrev_b32_e32 v7, 4, v0                                // 0000000017FC: 200E0084
	v_and_b32_e32 v7, 12, v7                                   // 000000001800: 260E0E8C
	v_add_u32_e32 v7, 0, v7                                    // 000000001804: 680E0E80
	ds_write_b32 v7, v6                                        // 000000001808: D81A0000 00000607
	s_or_b64 exec, exec, s[2:3]                                // 000000001810: 87FE027E
	v_cmp_gt_u32_e32 vcc, 4, v0                                // 000000001814: 7D980084
	v_lshl_add_u32 v6, v0, 2, 0                                // 000000001818: D1FD0006 02010500
	s_waitcnt lgkmcnt(0)                                       // 000000001820: BF8CC07F
	v_mov_b32_e32 v7, 0                                        // 000000001824: 7E0E0280
	s_barrier                                                  // 000000001828: BF8A0000
	s_and_saveexec_b64 s[2:3], vcc                             // 00000000182C: BE82206A
	ds_read_b32 v7, v6                                         // 000000001830: D86C0000 07000006
	s_or_b64 exec, exec, s[2:3]                                // 000000001838: 87FE027E
	s_waitcnt lgkmcnt(0)                                       // 00000000183C: BF8CC07F
	ds_swizzle_b32 v8, v7 offset:swizzle(SWAP,2)               // 000000001840: D87A081F 08000007
	s_load_dword s7, s[4:5], 0x20                              // 000000001848: C00201C2 00000020
	v_cmp_eq_u32_e64 s[2:3], 0, v0                             // 000000001850: D0CA0002 00020080
	s_waitcnt lgkmcnt(0)                                       // 000000001858: BF8CC07F
	v_add_f32_e32 v7, v7, v8                                   // 00000000185C: 020E1107
	ds_swizzle_b32 v8, v7 offset:swizzle(SWAP,1)               // 000000001860: D87A041F 08000007
	s_and_saveexec_b64 s[8:9], s[2:3]                          // 000000001868: BE882002
	s_cbranch_execz 4                                          // 00000000186C: BF880004 <c662_rmsnorm_fwd_v1+0x180>
	s_waitcnt lgkmcnt(0)                                       // 000000001870: BF8CC07F
	v_add_f32_e32 v0, v7, v8                                   // 000000001874: 02001107
	ds_write_b32 v6, v0                                        // 000000001878: D81A0000 00000006
	s_or_b64 exec, exec, s[8:9]                                // 000000001880: 87FE087E
	v_mov_b32_e32 v0, 0                                        // 000000001884: 7E000280
	s_waitcnt lgkmcnt(0)                                       // 000000001888: BF8CC07F
	s_barrier                                                  // 00000000188C: BF8A0000
	ds_read_b32 v0, v0                                         // 000000001890: D86C0000 00000000
	s_mov_b32 s10, 0x44400000                                  // 000000001898: BE8A00FF 44400000
	s_waitcnt lgkmcnt(0)                                       // 0000000018A0: BF8CC07F
	v_div_scale_f32 v6, s[8:9], s10, s10, v0                   // 0000000018A4: D1E00806 0400140A
	v_div_scale_f32 v7, vcc, v0, s10, v0                       // 0000000018AC: D1E06A07 04001500
	v_rcp_f32_e32 v8, v6                                       // 0000000018B4: 7E104506
	v_fma_f32 v9, -v6, v8, 1.0                                 // 0000000018B8: D1CB0009 23CA1106
	v_fmac_f32_e32 v8, v9, v8                                  // 0000000018C0: 76101109
	v_mul_f32_e32 v9, v7, v8                                   // 0000000018C4: 0A121107
	v_fma_f32 v10, -v6, v9, v7                                 // 0000000018C8: D1CB000A 241E1306
	v_fmac_f32_e32 v9, v10, v8                                 // 0000000018D0: 7612110A
	v_fma_f32 v6, -v6, v9, v7                                  // 0000000018D4: D1CB0006 241E1306
	v_div_fmas_f32 v6, v6, v8, v9                              // 0000000018DC: D1E20006 04261106
	v_div_fixup_f32 v0, v6, s10, v0                            // 0000000018E4: D1DE0000 04001506
	v_add_f32_e32 v0, s7, v0                                   // 0000000018EC: 02000007
	v_rsq_f32_e32 v0, v0                                       // 0000000018F0: 7E004900
	s_and_saveexec_b64 s[8:9], s[0:1]                          // 0000000018F4: BE882000
	s_cbranch_execnz 4                                         // 0000000018F8: BF890004 <c662_rmsnorm_fwd_v1+0x20c>
	s_or_b64 exec, exec, s[8:9]                                // 0000000018FC: 87FE087E
	s_and_saveexec_b64 s[0:1], s[2:3]                          // 000000001900: BE802002
	s_cbranch_execnz 57                                        // 000000001904: BF890039 <c662_rmsnorm_fwd_v1+0x2ec>
	s_endpgm                                                   // 000000001908: BF810000
	v_mul_f32_e32 v2, v0, v2                                   // 00000000190C: 0A040500
	v_mul_f32_e32 v5, v0, v5                                   // 000000001910: 0A0A0B00
	v_bfe_u32 v6, v2, 16, 1                                    // 000000001914: D1C80006 02052102
	s_movk_i32 s10, 0x7fff                                     // 00000000191C: B00A7FFF
	v_mul_f32_e32 v1, v0, v1                                   // 000000001920: 0A020300
	v_bfe_u32 v7, v5, 16, 1                                    // 000000001924: D1C80007 02052105
	v_add3_u32 v6, v2, v6, s10                                 // 00000000192C: D1FF0006 002A0D02
	s_load_dwordx2 s[0:1], s[4:5], 0x8                         // 000000001934: C0060002 00000008
	s_load_dword s7, s[4:5], 0x1c                              // 00000000193C: C00201C2 0000001C
	v_mul_f32_e32 v4, v0, v4                                   // 000000001944: 0A080900
	v_bfe_u32 v8, v1, 16, 1                                    // 000000001948: D1C80008 02052101
	v_add3_u32 v7, v5, v7, s10                                 // 000000001950: D1FF0007 002A0F05
	v_lshrrev_b32_e32 v6, 16, v6                               // 000000001958: 200C0C90
	v_mov_b32_e32 v10, 0x7fff                                  // 00000000195C: 7E1402FF 00007FFF
	v_cmp_o_f32_e32 vcc, v2, v2                                // 000000001964: 7C8E0502
	v_bfe_u32 v9, v4, 16, 1                                    // 000000001968: D1C80009 02052104
	v_add3_u32 v8, v1, v8, s10                                 // 000000001970: D1FF0008 002A1101
	v_lshrrev_b32_e32 v7, 16, v7                               // 000000001978: 200E0E90
	v_cndmask_b32_e32 v2, v10, v6, vcc                         // 00000000197C: 00040D0A
	v_cmp_o_f32_e32 vcc, v5, v5                                // 000000001980: 7C8E0B05
	v_add3_u32 v9, v4, v9, s10                                 // 000000001984: D1FF0009 002A1304
	v_lshrrev_b32_e32 v8, 16, v8                               // 00000000198C: 20101090
	v_cndmask_b32_e32 v5, v10, v7, vcc                         // 000000001990: 000A0F0A
	v_cmp_o_f32_e32 vcc, v1, v1                                // 000000001994: 7C8E0301
	v_lshrrev_b32_e32 v9, 16, v9                               // 000000001998: 20121290
	v_cndmask_b32_e32 v1, v10, v8, vcc                         // 00000000199C: 0002110A
	v_cmp_o_f32_e32 vcc, v4, v4                                // 0000000019A0: 7C8E0904
	s_mov_b32 s10, 0x5040100                                   // 0000000019A4: BE8A00FF 05040100
	v_cndmask_b32_e32 v4, v10, v9, vcc                         // 0000000019AC: 0008130A
	v_perm_b32 v2, v5, v2, s10                                 // 0000000019B0: D1ED0002 002A0505
	v_perm_b32 v1, v4, v1, s10                                 // 0000000019B8: D1ED0001 002A0304
	s_waitcnt lgkmcnt(0)                                       // 0000000019C0: BF8CC07F
	s_mul_i32 s10, s7, s6                                      // 0000000019C4: 920A0607
	s_ashr_i32 s11, s10, 31                                    // 0000000019C8: 900B9F0A
	s_lshl_b64 s[10:11], s[10:11], 1                           // 0000000019CC: 8E8A810A
	s_add_u32 s0, s0, s10                                      // 0000000019D0: 80000A00
	s_addc_u32 s1, s1, s11                                     // 0000000019D4: 82010B01
	global_store_dwordx2 v3, v[1:2], s[0:1]                    // 0000000019D8: DC748000 00000103
	s_or_b64 exec, exec, s[8:9]                                // 0000000019E0: 87FE087E
	s_and_saveexec_b64 s[0:1], s[2:3]                          // 0000000019E4: BE802002
	s_cbranch_execz 65479                                      // 0000000019E8: BF88FFC7 <c662_rmsnorm_fwd_v1+0x208>
	s_load_dwordx2 s[0:1], s[4:5], 0x10                        // 0000000019EC: C0060002 00000010
	s_ashr_i32 s7, s6, 31                                      // 0000000019F4: 90079F06
	s_lshl_b64 s[2:3], s[6:7], 2                               // 0000000019F8: 8E828206
	v_mov_b32_e32 v1, 0                                        // 0000000019FC: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000001A00: BF8CC07F
	s_add_u32 s0, s0, s2                                       // 000000001A04: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001A08: 82010301
	global_store_dword v1, v0, s[0:1]                          // 000000001A0C: DC708000 00000001
	s_endpgm                                                   // 000000001A14: BF810000
