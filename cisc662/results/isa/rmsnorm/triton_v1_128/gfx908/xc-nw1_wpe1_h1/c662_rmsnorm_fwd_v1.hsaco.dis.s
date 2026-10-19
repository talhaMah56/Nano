
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_v1_128/gfx908/xc-nw1_wpe1_h1/c662_rmsnorm_fwd_v1.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <c662_rmsnorm_fwd_v1>:
	s_load_dwordx2 s[8:9], s[4:5], 0x18                        // 000000001700: C0060202 00000018
	s_load_dwordx4 s[0:3], s[4:5], 0x0                         // 000000001708: C00A0002 00000000
	v_lshlrev_b32_e32 v2, 2, v0                                // 000000001710: 24040082
	v_xor_b32_e32 v3, 0x80, v2                                 // 000000001714: 2A0604FF 00000080
	s_movk_i32 s7, 0x7fff                                      // 00000000171C: B0077FFF
	s_waitcnt lgkmcnt(0)                                       // 000000001720: BF8CC07F
	s_mul_i32 s10, s8, s6                                      // 000000001724: 920A0608
	s_ashr_i32 s11, s10, 31                                    // 000000001728: 900B9F0A
	s_lshl_b64 s[10:11], s[10:11], 1                           // 00000000172C: 8E8A810A
	s_add_u32 s0, s0, s10                                      // 000000001730: 80000A00
	s_addc_u32 s1, s1, s11                                     // 000000001734: 82010B01
	global_load_dword v1, v2, s[0:1]                           // 000000001738: DC508000 01000002
	s_load_dword s0, s[4:5], 0x20                              // 000000001740: C0020002 00000020
	v_mov_b32_e32 v6, 0x7fff                                   // 000000001748: 7E0C02FF 00007FFF
	s_mov_b32 s8, 0x5040100                                    // 000000001750: BE8800FF 05040100
	s_waitcnt lgkmcnt(0)                                       // 000000001758: BF8CC07F
	v_mov_b32_e32 v7, s0                                       // 00000000175C: 7E0E0200
	s_mul_i32 s0, s9, s6                                       // 000000001760: 92000609
	s_ashr_i32 s1, s0, 31                                      // 000000001764: 90019F00
	s_lshl_b64 s[0:1], s[0:1], 1                               // 000000001768: 8E808100
	s_add_u32 s0, s2, s0                                       // 00000000176C: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001770: 82010103
	s_waitcnt vmcnt(0)                                         // 000000001774: BF8C0F70
	v_and_b32_e32 v5, 0xffff0000, v1                           // 000000001778: 260A02FF FFFF0000
	v_lshlrev_b32_e32 v4, 16, v1                               // 000000001780: 24080290
	v_mul_f32_e32 v1, v5, v5                                   // 000000001784: 0A020B05
	v_fmac_f32_e32 v1, v4, v4                                  // 000000001788: 76020904
	ds_bpermute_b32 v3, v3, v1                                 // 00000000178C: D87E0000 03000103
	s_waitcnt lgkmcnt(0)                                       // 000000001794: BF8CC07F
	v_add_f32_e32 v1, v1, v3                                   // 000000001798: 02020701
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,16)              // 00000000179C: D87A401F 03000001
	s_waitcnt lgkmcnt(0)                                       // 0000000017A4: BF8CC07F
	v_add_f32_e32 v1, v1, v3                                   // 0000000017A8: 02020701
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,8)               // 0000000017AC: D87A201F 03000001
	s_waitcnt lgkmcnt(0)                                       // 0000000017B4: BF8CC07F
	v_add_f32_e32 v1, v1, v3                                   // 0000000017B8: 02020701
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,4)               // 0000000017BC: D87A101F 03000001
	s_waitcnt lgkmcnt(0)                                       // 0000000017C4: BF8CC07F
	v_add_f32_e32 v1, v1, v3                                   // 0000000017C8: 02020701
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,2)               // 0000000017CC: D87A081F 03000001
	s_waitcnt lgkmcnt(0)                                       // 0000000017D4: BF8CC07F
	v_add_f32_e32 v1, v1, v3                                   // 0000000017D8: 02020701
	ds_swizzle_b32 v3, v1 offset:swizzle(SWAP,1)               // 0000000017DC: D87A041F 03000001
	s_waitcnt lgkmcnt(0)                                       // 0000000017E4: BF8CC07F
	v_add_f32_e32 v1, v1, v3                                   // 0000000017E8: 02020701
	v_fmac_f32_e32 v7, 0x3c000000, v1                          // 0000000017EC: 760E02FF 3C000000
	v_rsq_f32_e32 v1, v7                                       // 0000000017F4: 7E024907
	v_mul_f32_e32 v4, v1, v4                                   // 0000000017F8: 0A080901
	v_mul_f32_e32 v3, v1, v5                                   // 0000000017FC: 0A060B01
	v_bfe_u32 v5, v4, 16, 1                                    // 000000001800: D1C80005 02052104
	v_bfe_u32 v7, v3, 16, 1                                    // 000000001808: D1C80007 02052103
	v_add3_u32 v5, v4, v5, s7                                  // 000000001810: D1FF0005 001E0B04
	v_add3_u32 v7, v3, v7, s7                                  // 000000001818: D1FF0007 001E0F03
	v_lshrrev_b32_e32 v5, 16, v5                               // 000000001820: 200A0A90
	v_cmp_o_f32_e32 vcc, v4, v4                                // 000000001824: 7C8E0904
	v_lshrrev_b32_e32 v7, 16, v7                               // 000000001828: 200E0E90
	v_cndmask_b32_e32 v4, v6, v5, vcc                          // 00000000182C: 00080B06
	v_cmp_o_f32_e32 vcc, v3, v3                                // 000000001830: 7C8E0703
	v_cndmask_b32_e32 v3, v6, v7, vcc                          // 000000001834: 00060F06
	v_perm_b32 v3, v3, v4, s8                                  // 000000001838: D1ED0003 00220903
	v_cmp_eq_u32_e32 vcc, 0, v0                                // 000000001840: 7D940080
	global_store_dword v2, v3, s[0:1]                          // 000000001844: DC708000 00000302
	s_and_saveexec_b64 s[0:1], vcc                             // 00000000184C: BE80206A
	s_cbranch_execz 10                                         // 000000001850: BF88000A <c662_rmsnorm_fwd_v1+0x17c>
	s_load_dwordx2 s[0:1], s[4:5], 0x10                        // 000000001854: C0060002 00000010
	s_ashr_i32 s7, s6, 31                                      // 00000000185C: 90079F06
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001860: 8E828206
	v_mov_b32_e32 v0, 0                                        // 000000001864: 7E000280
	s_waitcnt lgkmcnt(0)                                       // 000000001868: BF8CC07F
	s_add_u32 s0, s0, s2                                       // 00000000186C: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001870: 82010301
	global_store_dword v0, v1, s[0:1]                          // 000000001874: DC708000 00000100
	s_endpgm                                                   // 00000000187C: BF810000
