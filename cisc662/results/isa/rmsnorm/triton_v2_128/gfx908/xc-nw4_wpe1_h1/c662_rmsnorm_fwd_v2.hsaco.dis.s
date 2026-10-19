
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_v2_128/gfx908/xc-nw4_wpe1_h1/c662_rmsnorm_fwd_v2.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <c662_rmsnorm_fwd_v2>:
	s_load_dword s7, s[4:5], 0x18                              // 000000001700: C00201C2 00000018
	s_load_dword s8, s[4:5], 0x24                              // 000000001708: C0020202 00000024
	s_lshl_b32 s6, s6, 4                                       // 000000001710: 8E068406
	v_lshrrev_b32_e32 v1, 4, v0                                // 000000001714: 20020084
	v_and_b32_e32 v5, 15, v0                                   // 000000001718: 260A008F
	v_or_b32_e32 v7, s6, v1                                    // 00000000171C: 280E0206
	v_lshlrev_b32_e32 v2, 3, v5                                // 000000001720: 24040A83
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF8CC07F
	v_cmp_gt_i32_e32 vcc, s7, v7                               // 000000001728: 7D880E07
	v_mov_b32_e32 v1, 0                                        // 00000000172C: 7E020280
	v_lshlrev_b32_e32 v6, 1, v2                                // 000000001730: 240C0481
	v_mov_b32_e32 v2, 0                                        // 000000001734: 7E040280
	v_mov_b32_e32 v3, 0                                        // 000000001738: 7E060280
	v_mov_b32_e32 v4, 0                                        // 00000000173C: 7E080280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001740: BE82206A
	s_cbranch_execz 21                                         // 000000001744: BF880015 <c662_rmsnorm_fwd_v2+0x9c>
	s_load_dword s9, s[4:5], 0x1c                              // 000000001748: C0020242 0000001C
	s_load_dwordx2 s[0:1], s[4:5], 0x0                         // 000000001750: C0060002 00000000
	s_waitcnt lgkmcnt(0)                                       // 000000001758: BF8CC07F
	v_mul_lo_u32 v1, s9, v7                                    // 00000000175C: D2850001 00020E09
	v_mov_b32_e32 v3, s1                                       // 000000001764: 7E060201
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001768: 2204029F
	v_lshlrev_b64 v[1:2], 1, v[1:2]                            // 00000000176C: D28F0001 00020281
	v_add_co_u32_e64 v1, s[0:1], s0, v1                        // 000000001774: D1190001 00020200
	v_addc_co_u32_e64 v2, s[0:1], v3, v2, s[0:1]               // 00000000177C: D11C0002 00020503
	v_add_co_u32_e64 v1, s[0:1], v1, v6                        // 000000001784: D1190001 00020D01
	v_addc_co_u32_e64 v2, s[0:1], 0, v2, s[0:1]                // 00000000178C: D11C0002 00020480
	global_load_dwordx4 v[1:4], v[1:2], off                    // 000000001794: DC5C8000 017F0001
	s_or_b64 exec, exec, s[2:3]                                // 00000000179C: 87FE027E
	s_waitcnt vmcnt(0)                                         // 0000000017A0: BF8C0F70
	v_and_b32_e32 v8, 0xffff0000, v1                           // 0000000017A4: 261002FF FFFF0000
	v_lshlrev_b32_e32 v9, 16, v1                               // 0000000017AC: 24120290
	v_mul_f32_e32 v1, v8, v8                                   // 0000000017B0: 0A021108
	v_lshlrev_b32_e32 v11, 16, v2                              // 0000000017B4: 24160490
	v_fmac_f32_e32 v1, v9, v9                                  // 0000000017B8: 76021309
	v_and_b32_e32 v10, 0xffff0000, v2                          // 0000000017BC: 261404FF FFFF0000
	v_fmac_f32_e32 v1, v11, v11                                // 0000000017C4: 7602170B
	v_and_b32_e32 v12, 0xffff0000, v3                          // 0000000017C8: 261806FF FFFF0000
	v_lshlrev_b32_e32 v3, 16, v3                               // 0000000017D0: 24060690
	v_fmac_f32_e32 v1, v10, v10                                // 0000000017D4: 7602150A
	v_fmac_f32_e32 v1, v3, v3                                  // 0000000017D8: 76020703
	v_and_b32_e32 v13, 0xffff0000, v4                          // 0000000017DC: 261A08FF FFFF0000
	v_lshlrev_b32_e32 v4, 16, v4                               // 0000000017E4: 24080890
	v_fmac_f32_e32 v1, v12, v12                                // 0000000017E8: 7602190C
	v_fmac_f32_e32 v1, v4, v4                                  // 0000000017EC: 76020904
	v_fmac_f32_e32 v1, v13, v13                                // 0000000017F0: 76021B0D
	ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,8)               // 0000000017F4: D87A201F 02000001
	s_waitcnt lgkmcnt(0)                                       // 0000000017FC: BF8CC07F
	v_add_f32_e32 v1, v1, v2                                   // 000000001800: 02020501
	ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,4)               // 000000001804: D87A101F 02000001
	s_waitcnt lgkmcnt(0)                                       // 00000000180C: BF8CC07F
	v_add_f32_e32 v1, v1, v2                                   // 000000001810: 02020501
	ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,2)               // 000000001814: D87A081F 02000001
	s_waitcnt lgkmcnt(0)                                       // 00000000181C: BF8CC07F
	v_add_f32_e32 v1, v1, v2                                   // 000000001820: 02020501
	ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,1)               // 000000001824: D87A041F 02000001
	s_waitcnt lgkmcnt(0)                                       // 00000000182C: BF8CC07F
	v_add_f32_e32 v1, v1, v2                                   // 000000001830: 02020501
	v_mov_b32_e32 v2, s8                                       // 000000001834: 7E040208
	v_fmac_f32_e32 v2, 0x3c000000, v1                          // 000000001838: 760402FF 3C000000
	v_rsq_f32_e32 v2, v2                                       // 000000001840: 7E044902
	v_and_b32_e32 v1, 0xf0, v0                                 // 000000001844: 260200FF 000000F0
	s_and_saveexec_b64 s[0:1], vcc                             // 00000000184C: BE80206A
	s_cbranch_execz 94                                         // 000000001850: BF88005E <c662_rmsnorm_fwd_v2+0x2cc>
	s_load_dword s8, s[4:5], 0x20                              // 000000001854: C0020202 00000020
	s_load_dwordx2 s[2:3], s[4:5], 0x8                         // 00000000185C: C0060082 00000008
	v_mul_f32_e32 v4, v2, v4                                   // 000000001864: 0A080902
	v_mul_f32_e32 v3, v2, v3                                   // 000000001868: 0A060702
	s_waitcnt lgkmcnt(0)                                       // 00000000186C: BF8CC07F
	v_mul_lo_u32 v14, s8, v7                                   // 000000001870: D285000E 00020E08
	v_mov_b32_e32 v0, s3                                       // 000000001878: 7E000203
	v_ashrrev_i32_e32 v15, 31, v14                             // 00000000187C: 221E1C9F
	v_lshlrev_b64 v[14:15], 1, v[14:15]                        // 000000001880: D28F000E 00021C81
	v_add_co_u32_e32 v7, vcc, s2, v14                          // 000000001888: 320E1C02
	v_addc_co_u32_e32 v0, vcc, v0, v15, vcc                    // 00000000188C: 38001F00
	v_add_co_u32_e32 v14, vcc, v7, v6                          // 000000001890: 321C0D07
	v_addc_co_u32_e32 v15, vcc, 0, v0, vcc                     // 000000001894: 381E0080
	v_mul_f32_e32 v6, v2, v9                                   // 000000001898: 0A0C1302
	v_mul_f32_e32 v7, v2, v10                                  // 00000000189C: 0A0E1502
	v_mul_f32_e32 v10, v2, v11                                 // 0000000018A0: 0A141702
	v_mul_f32_e32 v9, v2, v13                                  // 0000000018A4: 0A121B02
	v_bfe_u32 v11, v4, 16, 1                                   // 0000000018A8: D1C8000B 02052104
	s_movk_i32 s2, 0x7fff                                      // 0000000018B0: B0027FFF
	v_mul_f32_e32 v0, v2, v8                                   // 0000000018B4: 0A001102
	v_mul_f32_e32 v8, v2, v12                                  // 0000000018B8: 0A101902
	v_bfe_u32 v12, v9, 16, 1                                   // 0000000018BC: D1C8000C 02052109
	v_add3_u32 v11, v4, v11, s2                                // 0000000018C4: D1FF000B 000A1704
	v_cmp_o_f32_e32 vcc, v4, v4                                // 0000000018CC: 7C8E0904
	v_mov_b32_e32 v4, 0x7fff                                   // 0000000018D0: 7E0802FF 00007FFF
	v_bfe_u32 v13, v3, 16, 1                                   // 0000000018D8: D1C8000D 02052103
	v_add3_u32 v12, v9, v12, s2                                // 0000000018E0: D1FF000C 000A1909
	v_cndmask_b32_sdwa v11, v4, v11, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000018E8: 001616F9 05060604
	v_cmp_o_f32_e32 vcc, v9, v9                                // 0000000018F0: 7C8E1309
	v_bfe_u32 v16, v8, 16, 1                                   // 0000000018F4: D1C80010 02052108
	v_add3_u32 v13, v3, v13, s2                                // 0000000018FC: D1FF000D 000A1B03
	v_cndmask_b32_sdwa v9, v4, v12, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001904: 001218F9 05060604
	v_cmp_o_f32_e32 vcc, v3, v3                                // 00000000190C: 7C8E0703
	v_bfe_u32 v17, v10, 16, 1                                  // 000000001910: D1C80011 0205210A
	v_bfe_u32 v18, v7, 16, 1                                   // 000000001918: D1C80012 02052107
	v_bfe_u32 v19, v6, 16, 1                                   // 000000001920: D1C80013 02052106
	v_bfe_u32 v20, v0, 16, 1                                   // 000000001928: D1C80014 02052100
	v_add3_u32 v16, v8, v16, s2                                // 000000001930: D1FF0010 000A2108
	v_cndmask_b32_sdwa v3, v4, v13, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001938: 00061AF9 05060604
	v_cmp_o_f32_e32 vcc, v8, v8                                // 000000001940: 7C8E1108
	v_add3_u32 v20, v0, v20, s2                                // 000000001944: D1FF0014 000A2900
	v_add3_u32 v19, v6, v19, s2                                // 00000000194C: D1FF0013 000A2706
	v_add3_u32 v18, v7, v18, s2                                // 000000001954: D1FF0012 000A2507
	v_add3_u32 v17, v10, v17, s2                               // 00000000195C: D1FF0011 000A230A
	s_mov_b32 s2, 0x5040100                                    // 000000001964: BE8200FF 05040100
	v_cndmask_b32_sdwa v8, v4, v16, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000196C: 001020F9 05060604
	v_cmp_o_f32_e32 vcc, v10, v10                              // 000000001974: 7C8E150A
	v_perm_b32 v8, v8, v3, s2                                  // 000000001978: D1ED0008 000A0708
	v_cndmask_b32_sdwa v3, v4, v17, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000001980: 000622F9 05060604
	v_cmp_o_f32_e32 vcc, v7, v7                                // 000000001988: 7C8E0F07
	v_cndmask_b32_sdwa v7, v4, v18, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000198C: 000E24F9 05060604
	v_cmp_o_f32_e32 vcc, v6, v6                                // 000000001994: 7C8E0D06
	v_perm_b32 v7, v7, v3, s2                                  // 000000001998: D1ED0007 000A0707
	v_cndmask_b32_sdwa v3, v4, v19, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000019A0: 000626F9 05060604
	v_cmp_o_f32_e32 vcc, v0, v0                                // 0000000019A8: 7C8E0100
	v_cndmask_b32_sdwa v0, v4, v20, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000019AC: 000028F9 05060604
	v_perm_b32 v9, v9, v11, s2                                 // 0000000019B4: D1ED0009 000A1709
	v_perm_b32 v6, v0, v3, s2                                  // 0000000019BC: D1ED0006 000A0700
	global_store_dwordx4 v[14:15], v[6:9], off                 // 0000000019C4: DC7C8000 007F060E
	s_or_b64 exec, exec, s[0:1]                                // 0000000019CC: 87FE007E
	v_or_b32_e32 v0, s6, v5                                    // 0000000019D0: 28000A06
	v_cmp_gt_i32_e32 vcc, s7, v0                               // 0000000019D4: 7D880007
	v_lshrrev_b32_e32 v3, 2, v1                                // 0000000019D8: 20060282
	v_cmp_eq_u32_e64 s[0:1], 0, v1                             // 0000000019DC: D0CA0000 00020280
	v_add_u32_e32 v3, 0, v3                                    // 0000000019E4: 68060680
	s_and_b64 s[0:1], s[0:1], vcc                              // 0000000019E8: 86806A00
	ds_write_b32 v3, v2                                        // 0000000019EC: D81A0000 00000203
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 0000000019F4: BF8C0070
	s_barrier                                                  // 0000000019F8: BF8A0000
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 0000000019FC: BE822000
	s_cbranch_execz 15                                         // 000000001A00: BF88000F <c662_rmsnorm_fwd_v2+0x340>
	s_load_dwordx2 s[0:1], s[4:5], 0x10                        // 000000001A04: C0060002 00000010
	v_lshl_add_u32 v2, v5, 2, 0                                // 000000001A0C: D1FD0002 02010505
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000001A14: 2202009F
	ds_read_b32 v2, v2                                         // 000000001A18: D86C0000 02000002
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 000000001A20: D28F0000 00020082
	s_waitcnt lgkmcnt(0)                                       // 000000001A28: BF8CC07F
	v_mov_b32_e32 v3, s1                                       // 000000001A2C: 7E060201
	v_add_co_u32_e32 v0, vcc, s0, v0                           // 000000001A30: 32000000
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc                     // 000000001A34: 38020303
	global_store_dword v[0:1], v2, off                         // 000000001A38: DC708000 007F0200
	s_endpgm                                                   // 000000001A40: BF810000
