
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_mixnorm_bwd_finish_768/gfx908/xc-nw8_st2_wpe1_h1/c662_rmsnorm_bwd_mixnorm_finish.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <c662_rmsnorm_bwd_mixnorm_finish>:
	s_load_dword s7, s[4:5], 0x18                              // 000000001700: C00201C2 00000018
	s_load_dwordx4 s[0:3], s[4:5], 0x8                         // 000000001708: C00A0002 00000008
	v_lshlrev_b32_e32 v15, 2, v0                               // 000000001710: 241E0082
	s_waitcnt lgkmcnt(0)                                       // 000000001714: BF8CC07F
	s_cmp_lt_i32 s7, 1                                         // 000000001718: BF048107
	s_cbranch_scc1 61                                          // 00000000171C: BF85003D <c662_rmsnorm_bwd_mixnorm_finish+0x114>
	s_load_dwordx2 s[4:5], s[4:5], 0x0                         // 000000001720: C0060102 00000000
	s_mul_i32 s8, s7, s6                                       // 000000001728: 92080607
	s_ashr_i32 s9, s8, 31                                      // 00000000172C: 90099F08
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001730: 8E888208
	v_mov_b32_e32 v5, 0                                        // 000000001734: 7E0A0280
	s_waitcnt lgkmcnt(0)                                       // 000000001738: BF8CC07F
	s_add_u32 s8, s4, s8                                       // 00000000173C: 80080804
	v_mov_b32_e32 v7, 0                                        // 000000001740: 7E0E0280
	v_mov_b32_e32 v9, 0                                        // 000000001744: 7E120280
	v_mov_b32_e32 v11, 0                                       // 000000001748: 7E160280
	v_and_b32_e32 v16, 0x7fc, v15                              // 00000000174C: 26201EFF 000007FC
	s_addc_u32 s9, s5, s9                                      // 000000001754: 82090905
	v_mov_b32_e32 v6, 0                                        // 000000001758: 7E0C0280
	v_mov_b32_e32 v8, 0                                        // 00000000175C: 7E100280
	v_mov_b32_e32 v10, 0                                       // 000000001760: 7E140280
	v_mov_b32_e32 v12, 0                                       // 000000001764: 7E180280
	s_mov_b32 s10, 0                                           // 000000001768: BE8A0080
	s_branch 17                                                // 00000000176C: BF820011 <c662_rmsnorm_bwd_mixnorm_finish+0xb4>
	s_or_b64 exec, exec, s[4:5]                                // 000000001770: 87FE047E
	s_waitcnt vmcnt(0)                                         // 000000001774: BF8C0F70
	v_cvt_f64_f32_e32 v[13:14], v1                             // 000000001778: 7E1A2101
	v_cvt_f64_f32_e32 v[1:2], v2                               // 00000000177C: 7E022102
	v_cvt_f64_f32_e32 v[17:18], v3                             // 000000001780: 7E222103
	v_cvt_f64_f32_e32 v[3:4], v4                               // 000000001784: 7E062104
	v_add_f64 v[5:6], v[5:6], v[13:14]                         // 000000001788: D2800005 00021B05
	v_add_f64 v[7:8], v[7:8], v[1:2]                           // 000000001790: D2800007 00020307
	v_add_f64 v[9:10], v[9:10], v[17:18]                       // 000000001798: D2800009 00022309
	v_add_f64 v[11:12], v[11:12], v[3:4]                       // 0000000017A0: D280000B 0002070B
	s_addk_i32 s10, 0x800                                      // 0000000017A8: B70A0800
	s_cmp_lt_i32 s10, s7                                       // 0000000017AC: BF04070A
	s_cbranch_scc0 17                                          // 0000000017B0: BF840011 <c662_rmsnorm_bwd_mixnorm_finish+0xf8>
	v_add_u32_e32 v13, s10, v16                                // 0000000017B4: 681A200A
	v_cmp_gt_i32_e32 vcc, s7, v13                              // 0000000017B8: 7D881A07
	v_mov_b32_e32 v1, 0                                        // 0000000017BC: 7E020280
	v_mov_b32_e32 v2, 0                                        // 0000000017C0: 7E040280
	v_mov_b32_e32 v3, 0                                        // 0000000017C4: 7E060280
	v_mov_b32_e32 v4, 0                                        // 0000000017C8: 7E080280
	s_and_saveexec_b64 s[4:5], vcc                             // 0000000017CC: BE84206A
	s_cbranch_execz 65511                                      // 0000000017D0: BF88FFE7 <c662_rmsnorm_bwd_mixnorm_finish+0x70>
	v_ashrrev_i32_e32 v14, 31, v13                             // 0000000017D4: 221C1A9F
	v_lshlrev_b64 v[1:2], 2, v[13:14]                          // 0000000017D8: D28F0001 00021A82
	v_mov_b32_e32 v3, s9                                       // 0000000017E0: 7E060209
	v_add_co_u32_e32 v1, vcc, s8, v1                           // 0000000017E4: 32020208
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc                     // 0000000017E8: 38040503
	global_load_dwordx4 v[1:4], v[1:2], off                    // 0000000017EC: DC5C8000 017F0001
	s_branch 65502                                             // 0000000017F4: BF82FFDE <c662_rmsnorm_bwd_mixnorm_finish+0x70>
	v_add_f64 v[1:2], v[5:6], v[7:8]                           // 0000000017F8: D2800001 00020F05
	v_add_f64 v[1:2], v[9:10], v[1:2]                          // 000000001800: D2800001 00020309
	v_add_f64 v[1:2], v[11:12], v[1:2]                         // 000000001808: D2800001 0002030B
	s_branch 2                                                 // 000000001810: BF820002 <c662_rmsnorm_bwd_mixnorm_finish+0x11c>
	v_mov_b32_e32 v1, 0                                        // 000000001814: 7E020280
	v_mov_b32_e32 v2, 0                                        // 000000001818: 7E040280
	v_xor_b32_e32 v4, 0x80, v15                                // 00000000181C: 2A081EFF 00000080
	ds_bpermute_b32 v3, v4, v1                                 // 000000001824: D87E0000 03000104
	ds_bpermute_b32 v4, v4, v2                                 // 00000000182C: D87E0000 04000204
	v_and_b32_e32 v5, 63, v0                                   // 000000001834: 260A00BF
	v_cmp_eq_u32_e32 vcc, 0, v5                                // 000000001838: 7D940A80
	s_waitcnt lgkmcnt(0)                                       // 00000000183C: BF8CC07F
	v_add_f64 v[1:2], v[1:2], v[3:4]                           // 000000001840: D2800001 00020701
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,16)              // 000000001848: D87A401F 03000001
	ds_swizzle_b32 v4, v2 offset:swizzle(SWAP,16)              // 000000001850: D87A401F 04000002
	s_waitcnt lgkmcnt(0)                                       // 000000001858: BF8CC07F
	v_add_f64 v[1:2], v[1:2], v[3:4]                           // 00000000185C: D2800001 00020701
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,8)               // 000000001864: D87A201F 03000001
	ds_swizzle_b32 v4, v2 offset:swizzle(SWAP,8)               // 00000000186C: D87A201F 04000002
	s_waitcnt lgkmcnt(0)                                       // 000000001874: BF8CC07F
	v_add_f64 v[1:2], v[1:2], v[3:4]                           // 000000001878: D2800001 00020701
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,4)               // 000000001880: D87A101F 03000001
	ds_swizzle_b32 v4, v2 offset:swizzle(SWAP,4)               // 000000001888: D87A101F 04000002
	s_waitcnt lgkmcnt(0)                                       // 000000001890: BF8CC07F
	v_add_f64 v[1:2], v[1:2], v[3:4]                           // 000000001894: D2800001 00020701
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,2)               // 00000000189C: D87A081F 03000001
	ds_swizzle_b32 v4, v2 offset:swizzle(SWAP,2)               // 0000000018A4: D87A081F 04000002
	s_waitcnt lgkmcnt(0)                                       // 0000000018AC: BF8CC07F
	v_add_f64 v[1:2], v[1:2], v[3:4]                           // 0000000018B0: D2800001 00020701
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,1)               // 0000000018B8: D87A041F 03000001
	ds_swizzle_b32 v4, v2 offset:swizzle(SWAP,1)               // 0000000018C0: D87A041F 04000002
	s_and_saveexec_b64 s[4:5], vcc                             // 0000000018C8: BE84206A
	s_cbranch_execz 8                                          // 0000000018CC: BF880008 <c662_rmsnorm_bwd_mixnorm_finish+0x1f0>
	s_waitcnt lgkmcnt(0)                                       // 0000000018D0: BF8CC07F
	v_add_f64 v[1:2], v[1:2], v[3:4]                           // 0000000018D4: D2800001 00020701
	v_lshrrev_b32_e32 v3, 3, v0                                // 0000000018DC: 20060083
	v_and_b32_e32 v3, 56, v3                                   // 0000000018E0: 260606B8
	v_add_u32_e32 v3, 0, v3                                    // 0000000018E4: 68060680
	ds_write_b64 v3, v[1:2]                                    // 0000000018E8: D89A0000 00000103
	s_or_b64 exec, exec, s[4:5]                                // 0000000018F0: 87FE047E
	v_mov_b32_e32 v1, 0                                        // 0000000018F4: 7E020280
	v_cmp_gt_u32_e32 vcc, 8, v0                                // 0000000018F8: 7D980088
	v_mov_b32_e32 v2, 0                                        // 0000000018FC: 7E040280
	v_lshl_add_u32 v5, v0, 3, 0                                // 000000001900: D1FD0005 02010700
	s_waitcnt lgkmcnt(0)                                       // 000000001908: BF8CC07F
	s_barrier                                                  // 00000000190C: BF8A0000
	s_and_saveexec_b64 s[4:5], vcc                             // 000000001910: BE84206A
	ds_read_b64 v[1:2], v5                                     // 000000001914: D8EC0000 01000005
	s_or_b64 exec, exec, s[4:5]                                // 00000000191C: 87FE047E
	s_waitcnt lgkmcnt(0)                                       // 000000001920: BF8CC07F
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,4)               // 000000001924: D87A101F 03000001
	ds_swizzle_b32 v4, v2 offset:swizzle(SWAP,4)               // 00000000192C: D87A101F 04000002
	v_cmp_eq_u32_e32 vcc, 0, v0                                // 000000001934: 7D940080
	s_waitcnt lgkmcnt(0)                                       // 000000001938: BF8CC07F
	v_add_f64 v[1:2], v[1:2], v[3:4]                           // 00000000193C: D2800001 00020701
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,2)               // 000000001944: D87A081F 03000001
	ds_swizzle_b32 v4, v2 offset:swizzle(SWAP,2)               // 00000000194C: D87A081F 04000002
	s_waitcnt lgkmcnt(0)                                       // 000000001954: BF8CC07F
	v_add_f64 v[1:2], v[1:2], v[3:4]                           // 000000001958: D2800001 00020701
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,1)               // 000000001960: D87A041F 03000001
	ds_swizzle_b32 v4, v2 offset:swizzle(SWAP,1)               // 000000001968: D87A041F 04000002
	s_and_saveexec_b64 s[4:5], vcc                             // 000000001970: BE84206A
	s_cbranch_execz 5                                          // 000000001974: BF880005 <c662_rmsnorm_bwd_mixnorm_finish+0x28c>
	s_waitcnt lgkmcnt(0)                                       // 000000001978: BF8CC07F
	v_add_f64 v[0:1], v[1:2], v[3:4]                           // 00000000197C: D2800000 00020701
	ds_write_b64 v5, v[0:1]                                    // 000000001984: D89A0000 00000005
	s_or_b64 exec, exec, s[4:5]                                // 00000000198C: 87FE047E
	s_waitcnt lgkmcnt(0)                                       // 000000001990: BF8CC07F
	s_barrier                                                  // 000000001994: BF8A0000
	s_and_saveexec_b64 s[4:5], vcc                             // 000000001998: BE84206A
	s_cbranch_execz 10                                         // 00000000199C: BF88000A <c662_rmsnorm_bwd_mixnorm_finish+0x2c8>
	v_mov_b32_e32 v2, 0                                        // 0000000019A0: 7E040280
	ds_read_b64 v[0:1], v2                                     // 0000000019A4: D8EC0000 00000002
	s_cmp_eq_u32 s6, 0                                         // 0000000019AC: BF068006
	s_cselect_b32 s1, s1, s3                                   // 0000000019B0: 85010301
	s_cselect_b32 s0, s0, s2                                   // 0000000019B4: 85000200
	s_waitcnt lgkmcnt(0)                                       // 0000000019B8: BF8CC07F
	v_cvt_f32_f64_e32 v0, v[0:1]                               // 0000000019BC: 7E001F00
	global_store_dword v2, v0, s[0:1]                          // 0000000019C0: DC708000 00000002
	s_endpgm                                                   // 0000000019C8: BF810000
