
/u/tmahmood1/project/nanochat/cisc662/results/isa/rmsnorm/triton_mixnorm_bwd_768/gfx908/xc-nw4_wpe1_h1/c662_rmsnorm_bwd_mixnorm.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001a00 <c662_rmsnorm_bwd_mixnorm>:
	s_load_dword s12, s[4:5], 0x64                             // 000000001A00: C0020302 00000064
	s_load_dword s16, s[4:5], 0x58                             // 000000001A08: C0020402 00000058
	s_load_dwordx4 s[8:11], s[4:5], 0x10                       // 000000001A10: C00A0202 00000010
	s_lshl_b32 s17, s6, 2                                      // 000000001A18: 8E118206
	v_lshrrev_b32_e32 v1, 6, v0                                // 000000001A1C: 20020086
	v_or_b32_e32 v1, s17, v1                                   // 000000001A20: 28020211
	s_waitcnt lgkmcnt(0)                                       // 000000001A24: BF8CC07F
	v_cmp_le_i32_e64 s[0:1], s16, v1                           // 000000001A28: D0C30000 00020210
	v_cmp_gt_i32_e32 vcc, s16, v1                              // 000000001A30: 7D880210
	v_mov_b32_e32 v8, 0                                        // 000000001A34: 7E100280
	v_mov_b32_e32 v32, 0                                       // 000000001A38: 7E400280
	v_mov_b32_e32 v33, 0                                       // 000000001A3C: 7E420280
	s_and_saveexec_b64 s[6:7], vcc                             // 000000001A40: BE86206A
	s_cbranch_execz 15                                         // 000000001A44: BF88000F <c662_rmsnorm_bwd_mixnorm+0x84>
	s_load_dwordx2 s[2:3], s[4:5], 0x30                        // 000000001A48: C0060082 00000030
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001A50: 2204029F
	v_lshlrev_b64 v[2:3], 2, v[1:2]                            // 000000001A54: D28F0002 00020282
	s_waitcnt lgkmcnt(0)                                       // 000000001A5C: BF8CC07F
	v_mov_b32_e32 v4, s3                                       // 000000001A60: 7E080203
	v_add_co_u32_e64 v2, s[2:3], s2, v2                        // 000000001A64: D1190202 00020402
	v_addc_co_u32_e64 v3, s[2:3], v4, v3, s[2:3]               // 000000001A6C: D11C0203 000A0704
	global_load_dword v32, v[2:3], off                         // 000000001A74: DC508000 207F0002
	s_waitcnt vmcnt(0)                                         // 000000001A7C: BF8C0F70
	v_mov_b32_e32 v33, v32                                     // 000000001A80: 7E420320
	s_or_b64 exec, exec, s[6:7]                                // 000000001A84: 87FE067E
	v_mul_lo_u32 v2, s12, v1                                   // 000000001A88: D2850002 0002020C
	s_load_dwordx2 s[6:7], s[4:5], 0x8                         // 000000001A90: C0060182 00000008
	s_load_dword s12, s[4:5], 0x68                             // 000000001A98: C0020302 00000068
	v_lshlrev_b32_e32 v41, 2, v0                               // 000000001AA0: 24520082
	v_and_b32_e32 v4, 0xfc, v41                                // 000000001AA4: 260852FF 000000FC
	v_ashrrev_i32_e32 v3, 31, v2                               // 000000001AAC: 2206049F
	v_lshlrev_b64 v[2:3], 1, v[2:3]                            // 000000001AB0: D28F0002 00020481
	v_mov_b32_e32 v5, s9                                       // 000000001AB8: 7E0A0209
	v_add_co_u32_e64 v2, s[2:3], s8, v2                        // 000000001ABC: D1190202 00020408
	v_addc_co_u32_e64 v3, s[2:3], v5, v3, s[2:3]               // 000000001AC4: D11C0203 000A0705
	v_lshlrev_b32_e32 v36, 1, v4                               // 000000001ACC: 24480881
	v_add_co_u32_e64 v2, s[2:3], v2, v36                       // 000000001AD0: D1190202 00024902
	v_addc_co_u32_e64 v3, s[2:3], 0, v3, s[2:3]                // 000000001AD8: D11C0203 000A0680
	v_mov_b32_e32 v9, 0                                        // 000000001AE0: 7E120280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001AE4: BE82206A
	s_cbranch_execz 2                                          // 000000001AE8: BF880002 <c662_rmsnorm_bwd_mixnorm+0xf4>
	global_load_dwordx2 v[8:9], v[2:3], off                    // 000000001AEC: DC548000 087F0002
	s_or_b64 exec, exec, s[2:3]                                // 000000001AF4: 87FE027E
	s_waitcnt lgkmcnt(0)                                       // 000000001AF8: BF8CC07F
	v_mul_lo_u32 v4, s12, v1                                   // 000000001AFC: D2850004 0002020C
	s_load_dwordx2 s[8:9], s[4:5], 0x0                         // 000000001B04: C0060202 00000000
	s_load_dword s12, s[4:5], 0x60                             // 000000001B0C: C0020302 00000060
	v_mov_b32_e32 v6, s11                                      // 000000001B14: 7E0C020B
	v_mov_b32_e32 v10, 0                                       // 000000001B18: 7E140280
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000001B1C: 220A089F
	v_lshlrev_b64 v[4:5], 1, v[4:5]                            // 000000001B20: D28F0004 00020881
	v_mov_b32_e32 v12, 0                                       // 000000001B28: 7E180280
	v_add_co_u32_e64 v4, s[2:3], s10, v4                       // 000000001B2C: D1190204 0002080A
	v_addc_co_u32_e64 v5, s[2:3], v6, v5, s[2:3]               // 000000001B34: D11C0205 000A0B06
	v_add_co_u32_e64 v20, s[2:3], v4, v36                      // 000000001B3C: D1190214 00024904
	v_addc_co_u32_e64 v21, s[2:3], 0, v5, s[2:3]               // 000000001B44: D11C0215 000A0A80
	v_mov_b32_e32 v13, 0                                       // 000000001B4C: 7E1A0280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001B50: BE82206A
	s_cbranch_execz 2                                          // 000000001B54: BF880002 <c662_rmsnorm_bwd_mixnorm+0x160>
	global_load_dwordx2 v[12:13], v[20:21], off                // 000000001B58: DC548000 0C7F0014
	s_or_b64 exec, exec, s[2:3]                                // 000000001B60: 87FE027E
	s_waitcnt lgkmcnt(0)                                       // 000000001B64: BF8CC07F
	v_mul_lo_u32 v4, s12, v1                                   // 000000001B68: D2850004 0002020C
	s_load_dword s10, s[4:5], 0x5c                             // 000000001B70: C0020282 0000005C
	v_mov_b32_e32 v6, s7                                       // 000000001B78: 7E0C0207
	v_mov_b32_e32 v11, 0                                       // 000000001B7C: 7E160280
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000001B80: 220A089F
	v_lshlrev_b64 v[4:5], 1, v[4:5]                            // 000000001B84: D28F0004 00020881
	v_add_co_u32_e64 v4, s[2:3], s6, v4                        // 000000001B8C: D1190204 00020806
	v_addc_co_u32_e64 v5, s[2:3], v6, v5, s[2:3]               // 000000001B94: D11C0205 000A0B06
	v_add_co_u32_e64 v22, s[2:3], v4, v36                      // 000000001B9C: D1190216 00024904
	v_addc_co_u32_e64 v23, s[2:3], 0, v5, s[2:3]               // 000000001BA4: D11C0217 000A0A80
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001BAC: BE82206A
	s_cbranch_execz 2                                          // 000000001BB0: BF880002 <c662_rmsnorm_bwd_mixnorm+0x1bc>
	global_load_dwordx2 v[10:11], v[22:23], off                // 000000001BB4: DC548000 0A7F0016
	s_or_b64 exec, exec, s[2:3]                                // 000000001BBC: 87FE027E
	s_waitcnt lgkmcnt(0)                                       // 000000001BC0: BF8CC07F
	v_mul_lo_u32 v4, s10, v1                                   // 000000001BC4: D2850004 0002020A
	v_mov_b32_e32 v6, s9                                       // 000000001BCC: 7E0C0209
	v_mov_b32_e32 v16, 0                                       // 000000001BD0: 7E200280
	v_mov_b32_e32 v7, 0                                        // 000000001BD4: 7E0E0280
	v_ashrrev_i32_e32 v5, 31, v4                               // 000000001BD8: 220A089F
	v_lshlrev_b64 v[4:5], 1, v[4:5]                            // 000000001BDC: D28F0004 00020881
	v_add_co_u32_e64 v4, s[2:3], s8, v4                        // 000000001BE4: D1190204 00020808
	v_addc_co_u32_e64 v5, s[2:3], v6, v5, s[2:3]               // 000000001BEC: D11C0205 000A0B06
	v_add_co_u32_e64 v24, s[2:3], v4, v36                      // 000000001BF4: D1190218 00024904
	v_addc_co_u32_e64 v25, s[2:3], 0, v5, s[2:3]               // 000000001BFC: D11C0219 000A0A80
	v_mov_b32_e32 v6, 0                                        // 000000001C04: 7E0C0280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001C08: BE82206A
	s_cbranch_execz 2                                          // 000000001C0C: BF880002 <c662_rmsnorm_bwd_mixnorm+0x218>
	global_load_dwordx2 v[6:7], v[24:25], off                  // 000000001C10: DC548000 067F0018
	s_or_b64 exec, exec, s[2:3]                                // 000000001C18: 87FE027E
	v_mov_b32_e32 v17, 0                                       // 000000001C1C: 7E220280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001C20: BE82206A
	s_cbranch_execz 2                                          // 000000001C24: BF880002 <c662_rmsnorm_bwd_mixnorm+0x230>
	global_load_dwordx2 v[16:17], v[2:3], off offset:512       // 000000001C28: DC548200 107F0002
	s_or_b64 exec, exec, s[2:3]                                // 000000001C30: 87FE027E
	v_mov_b32_e32 v14, 0                                       // 000000001C34: 7E1C0280
	v_mov_b32_e32 v18, 0                                       // 000000001C38: 7E240280
	v_mov_b32_e32 v19, 0                                       // 000000001C3C: 7E260280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001C40: BE82206A
	s_cbranch_execz 2                                          // 000000001C44: BF880002 <c662_rmsnorm_bwd_mixnorm+0x250>
	global_load_dwordx2 v[18:19], v[20:21], off offset:512     // 000000001C48: DC548200 127F0014
	s_or_b64 exec, exec, s[2:3]                                // 000000001C50: 87FE027E
	v_mov_b32_e32 v15, 0                                       // 000000001C54: 7E1E0280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001C58: BE82206A
	s_cbranch_execz 2                                          // 000000001C5C: BF880002 <c662_rmsnorm_bwd_mixnorm+0x268>
	global_load_dwordx2 v[14:15], v[22:23], off offset:512     // 000000001C60: DC548200 0E7F0016
	s_or_b64 exec, exec, s[2:3]                                // 000000001C68: 87FE027E
	v_mov_b32_e32 v26, 0                                       // 000000001C6C: 7E340280
	v_mov_b32_e32 v4, 0                                        // 000000001C70: 7E080280
	v_mov_b32_e32 v5, 0                                        // 000000001C74: 7E0A0280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001C78: BE82206A
	s_cbranch_execz 2                                          // 000000001C7C: BF880002 <c662_rmsnorm_bwd_mixnorm+0x288>
	global_load_dwordx2 v[4:5], v[24:25], off offset:512       // 000000001C80: DC548200 047F0018
	s_or_b64 exec, exec, s[2:3]                                // 000000001C88: 87FE027E
	v_mov_b32_e32 v27, 0                                       // 000000001C8C: 7E360280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001C90: BE82206A
	s_cbranch_execz 2                                          // 000000001C94: BF880002 <c662_rmsnorm_bwd_mixnorm+0x2a0>
	global_load_dwordx2 v[26:27], v[2:3], off offset:1024      // 000000001C98: DC548400 1A7F0002
	s_or_b64 exec, exec, s[2:3]                                // 000000001CA0: 87FE027E
	v_mov_b32_e32 v30, 0                                       // 000000001CA4: 7E3C0280
	v_mov_b32_e32 v28, 0                                       // 000000001CA8: 7E380280
	v_mov_b32_e32 v29, 0                                       // 000000001CAC: 7E3A0280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001CB0: BE82206A
	s_cbranch_execz 2                                          // 000000001CB4: BF880002 <c662_rmsnorm_bwd_mixnorm+0x2c0>
	global_load_dwordx2 v[28:29], v[20:21], off offset:1024    // 000000001CB8: DC548400 1C7F0014
	s_or_b64 exec, exec, s[2:3]                                // 000000001CC0: 87FE027E
	s_load_dwordx4 s[12:15], s[4:5], 0x20                      // 000000001CC4: C00A0302 00000020
	s_load_dwordx4 s[8:11], s[4:5], 0x38                       // 000000001CCC: C00A0202 00000038
	v_mov_b32_e32 v31, 0                                       // 000000001CD4: 7E3E0280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001CD8: BE82206A
	s_cbranch_execz 2                                          // 000000001CDC: BF880002 <c662_rmsnorm_bwd_mixnorm+0x2e8>
	global_load_dwordx2 v[30:31], v[22:23], off offset:1024    // 000000001CE0: DC548400 1E7F0016
	s_or_b64 exec, exec, s[2:3]                                // 000000001CE8: 87FE027E
	s_waitcnt lgkmcnt(0)                                       // 000000001CEC: BF8CC07F
	s_load_dword s18, s[14:15], 0x0                            // 000000001CF0: C0020487 00000000
	s_load_dwordx2 s[6:7], s[4:5], 0x6c                        // 000000001CF8: C0060182 0000006C
	s_load_dword s19, s[12:13], 0x0                            // 000000001D00: C00204C6 00000000
	v_mov_b32_e32 v2, 0                                        // 000000001D08: 7E040280
	v_mov_b32_e32 v3, 0                                        // 000000001D0C: 7E060280
	s_and_saveexec_b64 s[2:3], vcc                             // 000000001D10: BE82206A
	s_cbranch_execz 2                                          // 000000001D14: BF880002 <c662_rmsnorm_bwd_mixnorm+0x320>
	global_load_dwordx2 v[2:3], v[24:25], off offset:1024      // 000000001D18: DC548400 027F0018
	s_or_b64 exec, exec, s[2:3]                                // 000000001D20: 87FE027E
	s_waitcnt vmcnt(0)                                         // 000000001D24: BF8C0F70
	v_lshlrev_b32_e32 v22, 16, v28                             // 000000001D28: 242C3890
	v_and_b32_e32 v37, 0xffff0000, v30                         // 000000001D2C: 264A3CFF FFFF0000
	v_lshlrev_b32_e32 v39, 16, v30                             // 000000001D34: 244E3C90
	v_lshlrev_b32_e32 v30, 16, v26                             // 000000001D38: 243C3490
	v_and_b32_e32 v20, 0xffff0000, v28                         // 000000001D3C: 262838FF FFFF0000
	s_waitcnt lgkmcnt(0)                                       // 000000001D44: BF8CC07F
	v_mul_f32_e32 v44, s18, v22                                // 000000001D48: 0A582C12
	v_and_b32_e32 v25, 0xffff0000, v26                         // 000000001D4C: 263234FF FFFF0000
	v_mul_f32_e32 v45, s18, v20                                // 000000001D54: 0A5A2812
	v_lshlrev_b32_e32 v23, 16, v29                             // 000000001D58: 242E3A90
	v_fmac_f32_e32 v44, s19, v30                               // 000000001D5C: 76583C13
	v_and_b32_e32 v24, 0xffff0000, v27                         // 000000001D60: 263036FF FFFF0000
	v_lshlrev_b32_e32 v26, 16, v27                             // 000000001D68: 24343690
	v_and_b32_e32 v21, 0xffff0000, v29                         // 000000001D6C: 262A3AFF FFFF0000
	v_mul_f32_e32 v50, s18, v23                                // 000000001D74: 0A642E12
	v_fmac_f32_e32 v45, s19, v25                               // 000000001D78: 765A3213
	v_bfe_u32 v27, v44, 16, 1                                  // 000000001D7C: D1C8001B 0205212C
	s_movk_i32 s2, 0x7fff                                      // 000000001D84: B0027FFF
	v_mul_f32_e32 v51, s18, v21                                // 000000001D88: 0A662A12
	v_add3_u32 v52, v44, v27, s2                               // 000000001D8C: D1FF0034 000A372C
	v_bfe_u32 v27, v45, 16, 1                                  // 000000001D94: D1C8001B 0205212D
	v_fmac_f32_e32 v50, s19, v26                               // 000000001D9C: 76643413
	v_add3_u32 v53, v45, v27, s2                               // 000000001DA0: D1FF0035 000A372D
	v_fmac_f32_e32 v51, s19, v24                               // 000000001DA8: 76663013
	v_bfe_u32 v27, v50, 16, 1                                  // 000000001DAC: D1C8001B 02052132
	v_and_b32_e32 v38, 0xffff0000, v31                         // 000000001DB4: 264C3EFF FFFF0000
	v_lshlrev_b32_e32 v40, 16, v31                             // 000000001DBC: 24503E90
	v_add3_u32 v54, v50, v27, s2                               // 000000001DC0: D1FF0036 000A3732
	v_bfe_u32 v27, v51, 16, 1                                  // 000000001DC8: D1C8001B 02052133
	v_and_b32_e32 v28, 0xffff0000, v17                         // 000000001DD0: 263822FF FFFF0000
	v_lshlrev_b32_e32 v31, 16, v17                             // 000000001DD8: 243E2290
	v_lshlrev_b32_e32 v17, 16, v18                             // 000000001DDC: 24222490
	v_add3_u32 v55, v51, v27, s2                               // 000000001DE0: D1FF0037 000A3733
	v_and_b32_e32 v27, 0xffff0000, v16                         // 000000001DE8: 263620FF FFFF0000
	v_lshlrev_b32_e32 v29, 16, v16                             // 000000001DF0: 243A2090
	v_and_b32_e32 v16, 0xffff0000, v18                         // 000000001DF4: 262024FF FFFF0000
	v_mul_f32_e32 v56, s18, v17                                // 000000001DFC: 0A702212
	v_mul_f32_e32 v57, s18, v16                                // 000000001E00: 0A722012
	v_and_b32_e32 v18, 0xffff0000, v19                         // 000000001E04: 262426FF FFFF0000
	v_lshlrev_b32_e32 v19, 16, v19                             // 000000001E0C: 24262690
	v_fmac_f32_e32 v56, s19, v29                               // 000000001E10: 76703A13
	v_mul_f32_e32 v58, s18, v19                                // 000000001E14: 0A742612
	v_fmac_f32_e32 v57, s19, v27                               // 000000001E18: 76723613
	v_bfe_u32 v34, v56, 16, 1                                  // 000000001E1C: D1C80022 02052138
	v_and_b32_e32 v53, 0xffff0000, v53                         // 000000001E24: 266A6AFF FFFF0000
	v_mov_b32_e32 v70, 0x7fff0000                              // 000000001E2C: 7E8C02FF 7FFF0000
	v_cmp_o_f32_e32 vcc, v45, v45                              // 000000001E34: 7C8E5B2D
	v_mul_f32_e32 v59, s18, v18                                // 000000001E38: 0A762412
	v_add3_u32 v60, v56, v34, s2                               // 000000001E3C: D1FF003C 000A4538
	v_bfe_u32 v34, v57, 16, 1                                  // 000000001E44: D1C80022 02052139
	v_fmac_f32_e32 v58, s19, v31                               // 000000001E4C: 76743E13
	v_and_b32_e32 v52, 0xffff0000, v52                         // 000000001E50: 266868FF FFFF0000
	v_cndmask_b32_e32 v45, v70, v53, vcc                       // 000000001E58: 005A6B46
	v_cmp_o_f32_e32 vcc, v44, v44                              // 000000001E5C: 7C8E592C
	v_add3_u32 v61, v57, v34, s2                               // 000000001E60: D1FF003D 000A4539
	v_fmac_f32_e32 v59, s19, v28                               // 000000001E68: 76763813
	v_bfe_u32 v34, v58, 16, 1                                  // 000000001E6C: D1C80022 0205213A
	v_cndmask_b32_e32 v44, v70, v52, vcc                       // 000000001E74: 00586946
	v_and_b32_e32 v52, 0xffff0000, v55                         // 000000001E78: 26686EFF FFFF0000
	v_cmp_o_f32_e32 vcc, v51, v51                              // 000000001E80: 7C8E6733
	v_add3_u32 v62, v58, v34, s2                               // 000000001E84: D1FF003E 000A453A
	v_bfe_u32 v34, v59, 16, 1                                  // 000000001E8C: D1C80022 0205213B
	v_and_b32_e32 v46, 0xffff0000, v14                         // 000000001E94: 265C1CFF FFFF0000
	v_lshlrev_b32_e32 v48, 16, v14                             // 000000001E9C: 24601C90
	v_and_b32_e32 v14, 0xffff0000, v12                         // 000000001EA0: 261C18FF FFFF0000
	v_and_b32_e32 v53, 0xffff0000, v54                         // 000000001EA8: 266A6CFF FFFF0000
	v_cndmask_b32_e32 v52, v70, v52, vcc                       // 000000001EB0: 00686946
	v_cmp_o_f32_e32 vcc, v50, v50                              // 000000001EB4: 7C8E6532
	v_add3_u32 v63, v59, v34, s2                               // 000000001EB8: D1FF003F 000A453B
	v_and_b32_e32 v47, 0xffff0000, v15                         // 000000001EC0: 265E1EFF FFFF0000
	v_lshlrev_b32_e32 v49, 16, v15                             // 000000001EC8: 24621E90
	v_and_b32_e32 v34, 0xffff0000, v8                          // 000000001ECC: 264410FF FFFF0000
	v_lshlrev_b32_e32 v35, 16, v8                              // 000000001ED4: 24461090
	v_lshlrev_b32_e32 v15, 16, v12                             // 000000001ED8: 241E1890
	v_mul_f32_e32 v8, s18, v14                                 // 000000001EDC: 0A101C12
	v_cndmask_b32_e32 v53, v70, v53, vcc                       // 000000001EE0: 006A6B46
	v_and_b32_e32 v50, 0xffff0000, v61                         // 000000001EE4: 26647AFF FFFF0000
	v_cmp_o_f32_e32 vcc, v57, v57                              // 000000001EEC: 7C8E7339
	v_and_b32_e32 v12, 0xffff0000, v13                         // 000000001EF0: 26181AFF FFFF0000
	v_fmac_f32_e32 v8, s19, v34                                // 000000001EF8: 76104413
	v_mul_f32_e32 v64, s18, v15                                // 000000001EFC: 0A801E12
	v_and_b32_e32 v51, 0xffff0000, v60                         // 000000001F00: 266678FF FFFF0000
	v_cndmask_b32_e32 v50, v70, v50, vcc                       // 000000001F08: 00646546
	v_cmp_o_f32_e32 vcc, v56, v56                              // 000000001F0C: 7C8E7138
	v_and_b32_e32 v42, 0xffff0000, v9                          // 000000001F10: 265412FF FFFF0000
	v_lshlrev_b32_e32 v43, 16, v9                              // 000000001F18: 24561290
	v_lshlrev_b32_e32 v13, 16, v13                             // 000000001F1C: 241A1A90
	v_bfe_u32 v9, v8, 16, 1                                    // 000000001F20: D1C80009 02052108
	v_fmac_f32_e32 v64, s19, v35                               // 000000001F28: 76804613
	v_mul_f32_e32 v66, s18, v12                                // 000000001F2C: 0A841812
	v_cndmask_b32_e32 v51, v70, v51, vcc                       // 000000001F30: 00666746
	v_and_b32_e32 v54, 0xffff0000, v63                         // 000000001F34: 266C7EFF FFFF0000
	v_cmp_o_f32_e32 vcc, v59, v59                              // 000000001F3C: 7C8E773B
	v_add3_u32 v9, v8, v9, s2                                  // 000000001F40: D1FF0009 000A1308
	v_bfe_u32 v65, v64, 16, 1                                  // 000000001F48: D1C80041 02052140
	v_fmac_f32_e32 v66, s19, v42                               // 000000001F50: 76845413
	v_mul_f32_e32 v68, s18, v13                                // 000000001F54: 0A881A12
	v_and_b32_e32 v55, 0xffff0000, v62                         // 000000001F58: 266E7CFF FFFF0000
	v_cndmask_b32_e32 v54, v70, v54, vcc                       // 000000001F60: 006C6D46
	v_cmp_o_f32_e32 vcc, v58, v58                              // 000000001F64: 7C8E753A
	v_add3_u32 v65, v64, v65, s2                               // 000000001F68: D1FF0041 000A8340
	v_bfe_u32 v67, v66, 16, 1                                  // 000000001F70: D1C80043 02052142
	v_fmac_f32_e32 v68, s19, v43                               // 000000001F78: 76885613
	v_cndmask_b32_e32 v57, v70, v55, vcc                       // 000000001F7C: 00726F46
	v_and_b32_e32 v9, 0xffff0000, v9                           // 000000001F80: 261212FF FFFF0000
	v_cmp_o_f32_e32 vcc, v8, v8                                // 000000001F88: 7C8E1108
	v_add3_u32 v67, v66, v67, s2                               // 000000001F8C: D1FF0043 000A8742
	v_bfe_u32 v69, v68, 16, 1                                  // 000000001F94: D1C80045 02052144
	v_and_b32_e32 v55, 0xffff0000, v65                         // 000000001F9C: 266E82FF FFFF0000
	v_cndmask_b32_e32 v8, v70, v9, vcc                         // 000000001FA4: 00101346
	v_cmp_o_f32_e32 vcc, v64, v64                              // 000000001FA8: 7C8E8140
	v_add3_u32 v69, v68, v69, s2                               // 000000001FAC: D1FF0045 000A8B44
	v_cndmask_b32_e32 v9, v70, v55, vcc                        // 000000001FB4: 00126F46
	v_and_b32_e32 v55, 0xffff0000, v67                         // 000000001FB8: 266E86FF FFFF0000
	v_cmp_o_f32_e32 vcc, v66, v66                              // 000000001FC0: 7C8E8542
	v_and_b32_e32 v56, 0xffff0000, v69                         // 000000001FC4: 26708AFF FFFF0000
	v_cndmask_b32_e32 v59, v70, v55, vcc                       // 000000001FCC: 00766F46
	v_cmp_o_f32_e32 vcc, v68, v68                              // 000000001FD0: 7C8E8944
	v_cndmask_b32_e32 v58, v70, v56, vcc                       // 000000001FD4: 00747146
	v_mul_f32_e32 v55, v33, v50                                // 000000001FD8: 0A6E6521
	v_mul_f32_e32 v56, v32, v51                                // 000000001FDC: 0A706720
	v_and_b32_e32 v60, 0xffff0000, v10                         // 000000001FE0: 267814FF FFFF0000
	v_lshlrev_b32_e32 v61, 16, v10                             // 000000001FE8: 247A1490
	v_mul_f32_e32 v64, v33, v8                                 // 000000001FEC: 0A801121
	v_mul_f32_e32 v65, v32, v9                                 // 000000001FF0: 0A821320
	v_mul_f32_e32 v8, v56, v48                                 // 000000001FF4: 0A106138
	v_mul_f32_e32 v9, v55, v46                                 // 000000001FF8: 0A125D37
	v_mul_f32_e32 v50, v33, v45                                // 000000001FFC: 0A645B21
	v_mul_f32_e32 v51, v32, v44                                // 000000002000: 0A665920
	v_fmac_f32_e32 v8, v65, v61                                // 000000002004: 76107B41
	v_fmac_f32_e32 v9, v64, v60                                // 000000002008: 76127940
	v_fmac_f32_e32 v8, v51, v39                                // 00000000200C: 76104F33
	v_fmac_f32_e32 v9, v50, v37                                // 000000002010: 76124B32
	v_mul_f32_e32 v57, v32, v57                                // 000000002014: 0A727320
	v_lshlrev_b32_e32 v63, 16, v11                             // 000000002018: 247E1690
	v_add_f32_e32 v8, v8, v9                                   // 00000000201C: 02101308
	v_mul_f32_e32 v66, v32, v58                                // 000000002020: 0A847520
	v_mul_f32_e32 v9, v57, v49                                 // 000000002024: 0A126339
	v_mul_f32_e32 v53, v32, v53                                // 000000002028: 0A6A6B20
	v_fmac_f32_e32 v9, v66, v63                                // 00000000202C: 76127F42
	v_fmac_f32_e32 v9, v53, v40                                // 000000002030: 76125135
	v_mul_f32_e32 v58, v33, v54                                // 000000002034: 0A746D21
	v_and_b32_e32 v62, 0xffff0000, v11                         // 000000002038: 267C16FF FFFF0000
	v_add_f32_e32 v8, v9, v8                                   // 000000002040: 02101109
	v_mul_f32_e32 v59, v33, v59                                // 000000002044: 0A767721
	v_mul_f32_e32 v9, v58, v47                                 // 000000002048: 0A125F3A
	v_mul_f32_e32 v54, v33, v52                                // 00000000204C: 0A6C6921
	v_fmac_f32_e32 v9, v59, v62                                // 000000002050: 76127D3B
	v_fmac_f32_e32 v9, v54, v38                                // 000000002054: 76124D36
	v_add_f32_e32 v8, v9, v8                                   // 000000002058: 02101109
	v_xor_b32_e32 v10, 0x80, v41                               // 00000000205C: 2A1452FF 00000080
	ds_bpermute_b32 v9, v10, v8                                // 000000002064: D87E0000 0900080A
	s_mov_b32 s12, 0x44400000                                  // 00000000206C: BE8C00FF 44400000
	v_mov_b32_e32 v44, s9                                      // 000000002074: 7E580209
	s_waitcnt lgkmcnt(0)                                       // 000000002078: BF8CC07F
	v_add_f32_e32 v8, v8, v9                                   // 00000000207C: 02101308
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,16)              // 000000002080: D87A401F 09000008
	s_waitcnt lgkmcnt(0)                                       // 000000002088: BF8CC07F
	v_add_f32_e32 v8, v8, v9                                   // 00000000208C: 02101308
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,8)               // 000000002090: D87A201F 09000008
	s_waitcnt lgkmcnt(0)                                       // 000000002098: BF8CC07F
	v_add_f32_e32 v8, v8, v9                                   // 00000000209C: 02101308
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,4)               // 0000000020A0: D87A101F 09000008
	s_waitcnt lgkmcnt(0)                                       // 0000000020A8: BF8CC07F
	v_add_f32_e32 v8, v8, v9                                   // 0000000020AC: 02101308
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,2)               // 0000000020B0: D87A081F 09000008
	s_waitcnt lgkmcnt(0)                                       // 0000000020B8: BF8CC07F
	v_add_f32_e32 v8, v8, v9                                   // 0000000020BC: 02101308
	ds_swizzle_b32 v9, v8 offset:swizzle(SWAP,1)               // 0000000020C0: D87A041F 09000008
	s_waitcnt lgkmcnt(0)                                       // 0000000020C8: BF8CC07F
	v_add_f32_e32 v52, v8, v9                                  // 0000000020CC: 02681308
	v_div_scale_f32 v11, s[2:3], s12, s12, v52                 // 0000000020D0: D1E0020B 04D0180C
	v_div_scale_f32 v41, vcc, v52, s12, v52                    // 0000000020D8: D1E06A29 04D01934
	v_mul_lo_u32 v8, s6, v1                                    // 0000000020E0: D2850008 00020206
	s_mov_b32 s6, s19                                          // 0000000020E8: BE860013
	v_ashrrev_i32_e32 v9, 31, v8                               // 0000000020EC: 2212109F
	v_lshlrev_b64 v[8:9], 1, v[8:9]                            // 0000000020F0: D28F0008 00021081
	v_add_co_u32_e64 v67, s[2:3], s8, v8                       // 0000000020F8: D1190243 00021008
	v_addc_co_u32_e64 v68, s[2:3], v44, v9, s[2:3]             // 000000002100: D11C0244 000A132C
	v_lshlrev_b32_e32 v44, 16, v7                              // 000000002108: 24580E90
	v_rcp_f32_e32 v45, v11                                     // 00000000210C: 7E5A450B
	v_fma_f32 v8, -v11, v45, 1.0                               // 000000002110: D1CB0008 23CA5B0B
	v_fmac_f32_e32 v45, v8, v45                                // 000000002118: 765A5B08
	v_mul_f32_e32 v9, v41, v45                                 // 00000000211C: 0A125B29
	v_fma_f32 v8, -v11, v9, v41                                // 000000002120: D1CB0008 24A6130B
	v_fmac_f32_e32 v9, v8, v45                                 // 000000002128: 76125B08
	v_mul_lo_u32 v8, s7, v1                                    // 00000000212C: D2850008 00020207
	v_fma_f32 v1, -v11, v9, v41                                // 000000002134: D1CB0001 24A6130B
	v_div_fmas_f32 v1, v1, v45, v9                             // 00000000213C: D1E20001 04265B01
	v_mov_b32_e32 v11, s11                                     // 000000002144: 7E16020B
	v_ashrrev_i32_e32 v9, 31, v8                               // 000000002148: 2212109F
	v_lshlrev_b64 v[8:9], 1, v[8:9]                            // 00000000214C: D28F0008 00021081
	v_and_b32_e32 v41, 0xffff0000, v6                          // 000000002154: 26520CFF FFFF0000
	v_add_co_u32_e32 v69, vcc, s10, v8                         // 00000000215C: 328A100A
	v_addc_co_u32_e32 v70, vcc, v11, v9, vcc                   // 000000002160: 388C130B
	v_add_co_u32_e32 v8, vcc, v67, v36                         // 000000002164: 32104943
	v_addc_co_u32_e32 v9, vcc, 0, v68, vcc                     // 000000002168: 38128880
	v_lshlrev_b32_e32 v45, 16, v6                              // 00000000216C: 245A0C90
	v_and_b32_e32 v11, 0xffff0000, v7                          // 000000002170: 26160EFF FFFF0000
	v_add_co_u32_e32 v6, vcc, v69, v36                         // 000000002178: 320C4945
	v_div_fixup_f32 v1, v1, s12, v52                           // 00000000217C: D1DE0001 04D01901
	v_fma_f32 v36, -v65, v1, v61                               // 000000002184: D1CB0024 24F60341
	v_fma_f32 v52, -v64, v1, v60                               // 00000000218C: D1CB0034 24F20340
	v_fma_f32 v60, -v66, v1, v63                               // 000000002194: D1CB003C 24FE0342
	v_fma_f32 v59, -v59, v1, v62                               // 00000000219C: D1CB003B 24FA033B
	v_addc_co_u32_e32 v7, vcc, 0, v70, vcc                     // 0000000021A4: 380E8C80
	v_fmac_f32_e32 v41, v33, v52                               // 0000000021A8: 76526921
	v_fmac_f32_e32 v45, v32, v36                               // 0000000021AC: 765A4920
	v_fmac_f32_e32 v11, v33, v59                               // 0000000021B0: 76167721
	v_fmac_f32_e32 v44, v32, v60                               // 0000000021B4: 76587920
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 0000000021B8: BE822000
	s_xor_b64 s[2:3], exec, s[2:3]                             // 0000000021BC: 8882027E
	s_andn2_saveexec_b64 s[2:3], s[2:3]                        // 0000000021C0: BE822302
	s_cbranch_execz 81                                         // 0000000021C4: BF880051 <c662_rmsnorm_bwd_mixnorm+0x90c>
	v_mul_f32_e32 v60, s19, v44                                // 0000000021C8: 0A785813
	v_mul_f32_e32 v59, s6, v11                                 // 0000000021CC: 0A761606
	v_bfe_u32 v61, v60, 16, 1                                  // 0000000021D0: D1C8003D 0205213C
	s_movk_i32 s7, 0x7fff                                      // 0000000021D8: B0077FFF
	v_mul_f32_e32 v52, s19, v45                                // 0000000021DC: 0A685A13
	v_bfe_u32 v62, v59, 16, 1                                  // 0000000021E0: D1C8003E 0205213B
	v_add3_u32 v61, v60, v61, s7                               // 0000000021E8: D1FF003D 001E7B3C
	v_cmp_o_f32_e32 vcc, v60, v60                              // 0000000021F0: 7C8E793C
	v_mov_b32_e32 v65, 0x7fff                                  // 0000000021F4: 7E8202FF 00007FFF
	v_mul_f32_e32 v36, s6, v41                                 // 0000000021FC: 0A485206
	v_bfe_u32 v63, v52, 16, 1                                  // 000000002200: D1C8003F 02052134
	v_add3_u32 v62, v59, v62, s7                               // 000000002208: D1FF003E 001E7D3B
	v_cndmask_b32_sdwa v60, v65, v61, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002210: 00787AF9 05060641
	v_cmp_o_f32_e32 vcc, v59, v59                              // 000000002218: 7C8E773B
	v_bfe_u32 v64, v36, 16, 1                                  // 00000000221C: D1C80040 02052124
	v_add3_u32 v63, v52, v63, s7                               // 000000002224: D1FF003F 001E7F34
	v_cndmask_b32_sdwa v59, v65, v62, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000222C: 00767CF9 05060641
	v_cmp_o_f32_e32 vcc, v52, v52                              // 000000002234: 7C8E6934
	v_add3_u32 v64, v36, v64, s7                               // 000000002238: D1FF0040 001E8124
	v_cndmask_b32_sdwa v52, v65, v63, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002240: 00687EF9 05060641
	v_cmp_o_f32_e32 vcc, v36, v36                              // 000000002248: 7C8E4924
	s_mov_b32 s8, 0x5040100                                    // 00000000224C: BE8800FF 05040100
	v_cndmask_b32_sdwa v36, v65, v64, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002254: 004880F9 05060641
	v_perm_b32 v60, v59, v60, s8                               // 00000000225C: D1ED003C 0022793B
	v_perm_b32 v59, v36, v52, s8                               // 000000002264: D1ED003B 00226924
	v_mul_f32_e32 v36, s18, v45                                // 00000000226C: 0A485A12
	v_bfe_u32 v52, v36, 16, 1                                  // 000000002270: D1C80034 02052124
	v_add3_u32 v52, v36, v52, s7                               // 000000002278: D1FF0034 001E6924
	v_cmp_o_f32_e32 vcc, v36, v36                              // 000000002280: 7C8E4924
	v_cndmask_b32_sdwa v36, v65, v52, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002284: 004868F9 05060641
	v_mul_f32_e32 v52, s18, v41                                // 00000000228C: 0A685212
	global_store_dwordx2 v[8:9], v[59:60], off                 // 000000002290: DC748000 007F3B08
	v_bfe_u32 v59, v52, 16, 1                                  // 000000002298: D1C8003B 02052134
	v_add3_u32 v59, v52, v59, s7                               // 0000000022A0: D1FF003B 001E7734
	v_cmp_o_f32_e32 vcc, v52, v52                              // 0000000022A8: 7C8E6934
	v_cndmask_b32_sdwa v52, v65, v59, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000022AC: 006876F9 05060641
	v_mul_f32_e32 v59, s18, v44                                // 0000000022B4: 0A765812
	v_bfe_u32 v60, v59, 16, 1                                  // 0000000022B8: D1C8003C 0205213B
	v_add3_u32 v60, v59, v60, s7                               // 0000000022C0: D1FF003C 001E793B
	v_cmp_o_f32_e32 vcc, v59, v59                              // 0000000022C8: 7C8E773B
	v_cndmask_b32_sdwa v59, v65, v60, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000022CC: 007678F9 05060641
	v_mul_f32_e32 v60, s18, v11                                // 0000000022D4: 0A781612
	v_bfe_u32 v61, v60, 16, 1                                  // 0000000022D8: D1C8003D 0205213C
	v_add3_u32 v61, v60, v61, s7                               // 0000000022E0: D1FF003D 001E7B3C
	v_cmp_o_f32_e32 vcc, v60, v60                              // 0000000022E8: 7C8E793C
	v_cndmask_b32_sdwa v60, v65, v61, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000022EC: 00787AF9 05060641
	v_perm_b32 v60, v60, v59, s8                               // 0000000022F4: D1ED003C 0022773C
	v_perm_b32 v59, v52, v36, s8                               // 0000000022FC: D1ED003B 00224934
	global_store_dwordx2 v[6:7], v[59:60], off                 // 000000002304: DC748000 007F3B06
	s_or_b64 exec, exec, s[2:3]                                // 00000000230C: 87FE027E
	v_and_b32_e32 v36, 0xffff0000, v4                          // 000000002310: 264808FF FFFF0000
	v_lshlrev_b32_e32 v52, 16, v4                              // 000000002318: 24680890
	v_and_b32_e32 v4, 0xffff0000, v5                           // 00000000231C: 26080AFF FFFF0000
	v_lshlrev_b32_e32 v5, 16, v5                               // 000000002324: 240A0A90
	v_fma_f32 v48, -v56, v1, v48                               // 000000002328: D1CB0030 24C20338
	v_fma_f32 v46, -v55, v1, v46                               // 000000002330: D1CB002E 24BA0337
	v_fma_f32 v49, -v57, v1, v49                               // 000000002338: D1CB0031 24C60339
	v_fma_f32 v47, -v58, v1, v47                               // 000000002340: D1CB002F 24BE033A
	v_fmac_f32_e32 v36, v33, v46                               // 000000002348: 76485D21
	v_fmac_f32_e32 v52, v32, v48                               // 00000000234C: 76686120
	v_fmac_f32_e32 v4, v33, v47                                // 000000002350: 76085F21
	v_fmac_f32_e32 v5, v32, v49                                // 000000002354: 760A6320
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000002358: BE822000
	s_xor_b64 s[2:3], exec, s[2:3]                             // 00000000235C: 8882027E
	s_andn2_saveexec_b64 s[2:3], s[2:3]                        // 000000002360: BE822302
	s_cbranch_execz 81                                         // 000000002364: BF880051 <c662_rmsnorm_bwd_mixnorm+0xaac>
	v_mul_f32_e32 v49, s19, v5                                 // 000000002368: 0A620A13
	v_mul_f32_e32 v47, s6, v4                                  // 00000000236C: 0A5E0806
	v_bfe_u32 v55, v49, 16, 1                                  // 000000002370: D1C80037 02052131
	s_movk_i32 s7, 0x7fff                                      // 000000002378: B0077FFF
	v_mul_f32_e32 v48, s19, v52                                // 00000000237C: 0A606813
	v_bfe_u32 v56, v47, 16, 1                                  // 000000002380: D1C80038 0205212F
	v_add3_u32 v55, v49, v55, s7                               // 000000002388: D1FF0037 001E6F31
	v_cmp_o_f32_e32 vcc, v49, v49                              // 000000002390: 7C8E6331
	v_mov_b32_e32 v49, 0x7fff                                  // 000000002394: 7E6202FF 00007FFF
	v_mul_f32_e32 v46, s6, v36                                 // 00000000239C: 0A5C4806
	v_bfe_u32 v57, v48, 16, 1                                  // 0000000023A0: D1C80039 02052130
	v_add3_u32 v56, v47, v56, s7                               // 0000000023A8: D1FF0038 001E712F
	v_cndmask_b32_sdwa v55, v49, v55, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000023B0: 006E6EF9 05060631
	v_cmp_o_f32_e32 vcc, v47, v47                              // 0000000023B8: 7C8E5F2F
	v_bfe_u32 v58, v46, 16, 1                                  // 0000000023BC: D1C8003A 0205212E
	v_add3_u32 v57, v48, v57, s7                               // 0000000023C4: D1FF0039 001E7330
	v_cndmask_b32_sdwa v47, v49, v56, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000023CC: 005E70F9 05060631
	v_cmp_o_f32_e32 vcc, v48, v48                              // 0000000023D4: 7C8E6130
	v_add3_u32 v58, v46, v58, s7                               // 0000000023D8: D1FF003A 001E752E
	v_cndmask_b32_sdwa v48, v49, v57, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000023E0: 006072F9 05060631
	v_cmp_o_f32_e32 vcc, v46, v46                              // 0000000023E8: 7C8E5D2E
	s_mov_b32 s8, 0x5040100                                    // 0000000023EC: BE8800FF 05040100
	v_cndmask_b32_sdwa v46, v49, v58, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000023F4: 005C74F9 05060631
	v_perm_b32 v47, v47, v55, s8                               // 0000000023FC: D1ED002F 00226F2F
	v_perm_b32 v46, v46, v48, s8                               // 000000002404: D1ED002E 0022612E
	global_store_dwordx2 v[8:9], v[46:47], off offset:512      // 00000000240C: DC748200 007F2E08
	v_mul_f32_e32 v46, s18, v52                                // 000000002414: 0A5C6812
	v_bfe_u32 v47, v46, 16, 1                                  // 000000002418: D1C8002F 0205212E
	v_add3_u32 v47, v46, v47, s7                               // 000000002420: D1FF002F 001E5F2E
	v_cmp_o_f32_e32 vcc, v46, v46                              // 000000002428: 7C8E5D2E
	v_cndmask_b32_sdwa v46, v49, v47, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000242C: 005C5EF9 05060631
	v_mul_f32_e32 v47, s18, v36                                // 000000002434: 0A5E4812
	v_bfe_u32 v48, v47, 16, 1                                  // 000000002438: D1C80030 0205212F
	v_add3_u32 v48, v47, v48, s7                               // 000000002440: D1FF0030 001E612F
	v_cmp_o_f32_e32 vcc, v47, v47                              // 000000002448: 7C8E5F2F
	v_mul_f32_e32 v47, s18, v5                                 // 00000000244C: 0A5E0A12
	v_bfe_u32 v55, v47, 16, 1                                  // 000000002450: D1C80037 0205212F
	v_cndmask_b32_sdwa v48, v49, v48, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002458: 006060F9 05060631
	v_add3_u32 v55, v47, v55, s7                               // 000000002460: D1FF0037 001E6F2F
	v_cmp_o_f32_e32 vcc, v47, v47                              // 000000002468: 7C8E5F2F
	v_cndmask_b32_sdwa v47, v49, v55, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000246C: 005E6EF9 05060631
	v_mul_f32_e32 v55, s18, v4                                 // 000000002474: 0A6E0812
	v_bfe_u32 v56, v55, 16, 1                                  // 000000002478: D1C80038 02052137
	v_add3_u32 v56, v55, v56, s7                               // 000000002480: D1FF0038 001E7137
	v_cmp_o_f32_e32 vcc, v55, v55                              // 000000002488: 7C8E6F37
	v_cndmask_b32_sdwa v49, v49, v56, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000248C: 006270F9 05060631
	v_perm_b32 v47, v49, v47, s8                               // 000000002494: D1ED002F 00225F31
	v_perm_b32 v46, v48, v46, s8                               // 00000000249C: D1ED002E 00225D30
	global_store_dwordx2 v[6:7], v[46:47], off offset:512      // 0000000024A4: DC748200 007F2E06
	s_or_b64 exec, exec, s[2:3]                                // 0000000024AC: 87FE027E
	v_and_b32_e32 v46, 0xffff0000, v2                          // 0000000024B0: 265C04FF FFFF0000
	v_lshlrev_b32_e32 v47, 16, v2                              // 0000000024B8: 245E0490
	v_and_b32_e32 v2, 0xffff0000, v3                           // 0000000024BC: 260406FF FFFF0000
	v_lshlrev_b32_e32 v3, 16, v3                               // 0000000024C4: 24060690
	v_fma_f32 v39, -v51, v1, v39                               // 0000000024C8: D1CB0027 249E0333
	v_fma_f32 v37, -v50, v1, v37                               // 0000000024D0: D1CB0025 24960332
	v_fma_f32 v40, -v53, v1, v40                               // 0000000024D8: D1CB0028 24A20335
	v_fma_f32 v1, -v54, v1, v38                                // 0000000024E0: D1CB0001 249A0336
	v_fmac_f32_e32 v46, v33, v37                               // 0000000024E8: 765C4B21
	v_fmac_f32_e32 v47, v32, v39                               // 0000000024EC: 765E4F20
	v_fmac_f32_e32 v2, v33, v1                                 // 0000000024F0: 76040321
	v_fmac_f32_e32 v3, v32, v40                                // 0000000024F4: 76065120
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 0000000024F8: BE822000
	s_xor_b64 s[0:1], exec, s[2:3]                             // 0000000024FC: 8880027E
	s_or_saveexec_b64 s[0:1], s[0:1]                           // 000000002500: BE802100
	v_and_b32_e32 v1, 0xc0, v0                                 // 000000002504: 260200FF 000000C0
	s_xor_b64 exec, exec, s[0:1]                               // 00000000250C: 88FE007E
	s_cbranch_execz 81                                         // 000000002510: BF880051 <c662_rmsnorm_bwd_mixnorm+0xc58>
	v_mul_f32_e32 v38, s19, v3                                 // 000000002514: 0A4C0613
	v_mul_f32_e32 v33, s6, v2                                  // 000000002518: 0A420406
	v_bfe_u32 v39, v38, 16, 1                                  // 00000000251C: D1C80027 02052126
	s_movk_i32 s2, 0x7fff                                      // 000000002524: B0027FFF
	v_mul_f32_e32 v37, s19, v47                                // 000000002528: 0A4A5E13
	v_bfe_u32 v40, v33, 16, 1                                  // 00000000252C: D1C80028 02052121
	v_add3_u32 v39, v38, v39, s2                               // 000000002534: D1FF0027 000A4F26
	v_cmp_o_f32_e32 vcc, v38, v38                              // 00000000253C: 7C8E4D26
	v_mov_b32_e32 v38, 0x7fff                                  // 000000002540: 7E4C02FF 00007FFF
	v_mul_f32_e32 v32, s6, v46                                 // 000000002548: 0A405C06
	v_bfe_u32 v48, v37, 16, 1                                  // 00000000254C: D1C80030 02052125
	v_add3_u32 v40, v33, v40, s2                               // 000000002554: D1FF0028 000A5121
	v_cndmask_b32_sdwa v39, v38, v39, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000255C: 004E4EF9 05060626
	v_cmp_o_f32_e32 vcc, v33, v33                              // 000000002564: 7C8E4321
	v_bfe_u32 v49, v32, 16, 1                                  // 000000002568: D1C80031 02052120
	v_add3_u32 v48, v37, v48, s2                               // 000000002570: D1FF0030 000A6125
	v_cndmask_b32_sdwa v33, v38, v40, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002578: 004250F9 05060626
	v_cmp_o_f32_e32 vcc, v37, v37                              // 000000002580: 7C8E4B25
	v_add3_u32 v49, v32, v49, s2                               // 000000002584: D1FF0031 000A6320
	v_cndmask_b32_sdwa v37, v38, v48, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 00000000258C: 004A60F9 05060626
	v_cmp_o_f32_e32 vcc, v32, v32                              // 000000002594: 7C8E4120
	s_mov_b32 s3, 0x5040100                                    // 000000002598: BE8300FF 05040100
	v_cndmask_b32_sdwa v32, v38, v49, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000025A0: 004062F9 05060626
	v_perm_b32 v33, v33, v39, s3                               // 0000000025A8: D1ED0021 000E4F21
	v_perm_b32 v32, v32, v37, s3                               // 0000000025B0: D1ED0020 000E4B20
	global_store_dwordx2 v[8:9], v[32:33], off offset:1024     // 0000000025B8: DC748400 007F2008
	v_mul_f32_e32 v8, s18, v47                                 // 0000000025C0: 0A105E12
	v_bfe_u32 v9, v8, 16, 1                                    // 0000000025C4: D1C80009 02052108
	v_add3_u32 v9, v8, v9, s2                                  // 0000000025CC: D1FF0009 000A1308
	v_cmp_o_f32_e32 vcc, v8, v8                                // 0000000025D4: 7C8E1108
	v_cndmask_b32_sdwa v8, v38, v9, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 0000000025D8: 001012F9 05060626
	v_mul_f32_e32 v9, s18, v46                                 // 0000000025E0: 0A125C12
	v_bfe_u32 v32, v9, 16, 1                                   // 0000000025E4: D1C80020 02052109
	v_add3_u32 v32, v9, v32, s2                                // 0000000025EC: D1FF0020 000A4109
	v_cmp_o_f32_e32 vcc, v9, v9                                // 0000000025F4: 7C8E1309
	v_mul_f32_e32 v9, s18, v3                                  // 0000000025F8: 0A120612
	v_bfe_u32 v33, v9, 16, 1                                   // 0000000025FC: D1C80021 02052109
	v_cndmask_b32_sdwa v32, v38, v32, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002604: 004040F9 05060626
	v_add3_u32 v33, v9, v33, s2                                // 00000000260C: D1FF0021 000A4309
	v_cmp_o_f32_e32 vcc, v9, v9                                // 000000002614: 7C8E1309
	v_cndmask_b32_sdwa v9, v38, v33, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002618: 001242F9 05060626
	v_mul_f32_e32 v33, s18, v2                                 // 000000002620: 0A420412
	v_bfe_u32 v37, v33, 16, 1                                  // 000000002624: D1C80025 02052121
	v_add3_u32 v37, v33, v37, s2                               // 00000000262C: D1FF0025 000A4B21
	v_cmp_o_f32_e32 vcc, v33, v33                              // 000000002634: 7C8E4321
	v_cndmask_b32_sdwa v33, v38, v37, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1// 000000002638: 00424AF9 05060626
	v_perm_b32 v9, v33, v9, s3                                 // 000000002640: D1ED0009 000E1321
	v_perm_b32 v8, v32, v8, s3                                 // 000000002648: D1ED0008 000E1120
	global_store_dwordx2 v[6:7], v[8:9], off offset:1024       // 000000002650: DC748400 007F0806
	s_or_b64 exec, exec, s[0:1]                                // 000000002658: 87FE007E
	v_mul_f32_e32 v6, v52, v29                                 // 00000000265C: 0A0C3B34
	v_mul_f32_e32 v7, v36, v27                                 // 000000002660: 0A0E3724
	v_mul_f32_e32 v8, v5, v31                                  // 000000002664: 0A103F05
	v_fmac_f32_e32 v6, v45, v35                                // 000000002668: 760C472D
	v_fmac_f32_e32 v7, v41, v34                                // 00000000266C: 760E4529
	v_mul_f32_e32 v9, v4, v28                                  // 000000002670: 0A123904
	v_fmac_f32_e32 v8, v44, v43                                // 000000002674: 7610572C
	v_fmac_f32_e32 v6, v47, v30                                // 000000002678: 760C3D2F
	v_fmac_f32_e32 v7, v46, v25                                // 00000000267C: 760E332E
	v_fmac_f32_e32 v9, v11, v42                                // 000000002680: 7612550B
	v_fmac_f32_e32 v8, v3, v26                                 // 000000002684: 76103503
	v_add_f32_e32 v6, v6, v7                                   // 000000002688: 020C0F06
	v_fmac_f32_e32 v9, v2, v24                                 // 00000000268C: 76123102
	v_add_f32_e32 v6, v8, v6                                   // 000000002690: 020C0D08
	v_add_f32_e32 v6, v9, v6                                   // 000000002694: 020C0D09
	ds_bpermute_b32 v7, v10, v6                                // 000000002698: D87E0000 0700060A
	v_and_b32_e32 v8, 3, v0                                    // 0000000026A0: 26100083
	v_lshrrev_b32_e32 v1, 4, v1                                // 0000000026A4: 20020284
	v_and_b32_e32 v24, 0xfc, v0                                // 0000000026A8: 263000FF 000000FC
	v_or_b32_e32 v0, s17, v8                                   // 0000000026B0: 28001011
	s_waitcnt lgkmcnt(0)                                       // 0000000026B4: BF8CC07F
	v_add_f32_e32 v6, v6, v7                                   // 0000000026B8: 020C0F06
	ds_swizzle_b32 v7, v6 offset:swizzle(SWAP,16)              // 0000000026BC: D87A401F 07000006
	v_lshlrev_b32_e32 v8, 2, v8                                // 0000000026C4: 24101082
	v_cmp_eq_u32_e32 vcc, 0, v24                               // 0000000026C8: 7D943080
	v_cmp_gt_i32_e64 s[0:1], s16, v0                           // 0000000026CC: D0C40000 00020010
	s_and_b64 s[0:1], vcc, s[0:1]                              // 0000000026D4: 8680006A
	s_waitcnt lgkmcnt(0)                                       // 0000000026D8: BF8CC07F
	v_add_f32_e32 v6, v6, v7                                   // 0000000026DC: 020C0F06
	ds_swizzle_b32 v7, v6 offset:swizzle(SWAP,8)               // 0000000026E0: D87A201F 07000006
	s_waitcnt lgkmcnt(0)                                       // 0000000026E8: BF8CC07F
	v_add_f32_e32 v6, v6, v7                                   // 0000000026EC: 020C0F06
	ds_swizzle_b32 v7, v6 offset:swizzle(SWAP,4)               // 0000000026F0: D87A101F 07000006
	s_waitcnt lgkmcnt(0)                                       // 0000000026F8: BF8CC07F
	v_add_f32_e32 v6, v6, v7                                   // 0000000026FC: 020C0F06
	ds_swizzle_b32 v9, v6 offset:swizzle(SWAP,2)               // 000000002700: D87A081F 09000006
	v_add_u32_e32 v7, 0, v1                                    // 000000002708: 680E0280
	v_ashrrev_i32_e32 v1, 31, v0                               // 00000000270C: 2202009F
	s_waitcnt lgkmcnt(0)                                       // 000000002710: BF8CC07F
	v_add_f32_e32 v6, v6, v9                                   // 000000002714: 020C1306
	ds_swizzle_b32 v9, v6 offset:swizzle(SWAP,1)               // 000000002718: D87A041F 09000006
	s_waitcnt lgkmcnt(0)                                       // 000000002720: BF8CC07F
	v_add_f32_e32 v6, v6, v9                                   // 000000002724: 020C1306
	ds_write_b32 v7, v6                                        // 000000002728: D81A0000 00000607
	v_add_u32_e32 v6, 0, v8                                    // 000000002730: 680C1080
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000002734: BF8C0070
	s_barrier                                                  // 000000002738: BF8A0000
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 00000000273C: BE822000
	s_cbranch_execz 12                                         // 000000002740: BF88000C <c662_rmsnorm_bwd_mixnorm+0xd74>
	s_load_dwordx2 s[6:7], s[4:5], 0x48                        // 000000002744: C0060182 00000048
	ds_read_b32 v24, v6                                        // 00000000274C: D86C0000 18000006
	v_lshlrev_b64 v[8:9], 2, v[0:1]                            // 000000002754: D28F0008 00020082
	s_waitcnt lgkmcnt(0)                                       // 00000000275C: BF8CC07F
	v_mov_b32_e32 v25, s7                                      // 000000002760: 7E320207
	v_add_co_u32_e32 v8, vcc, s6, v8                           // 000000002764: 32101006
	v_addc_co_u32_e32 v9, vcc, v25, v9, vcc                    // 000000002768: 38121319
	global_store_dword v[8:9], v24, off                        // 00000000276C: DC708000 007F1808
	s_or_b64 exec, exec, s[2:3]                                // 000000002774: 87FE027E
	v_mul_f32_e32 v8, v52, v17                                 // 000000002778: 0A102334
	v_mul_f32_e32 v9, v36, v16                                 // 00000000277C: 0A122124
	v_mul_f32_e32 v5, v5, v19                                  // 000000002780: 0A0A2705
	v_mul_f32_e32 v4, v4, v18                                  // 000000002784: 0A082504
	v_fmac_f32_e32 v8, v45, v15                                // 000000002788: 76101F2D
	v_fmac_f32_e32 v9, v41, v14                                // 00000000278C: 76121D29
	v_fmac_f32_e32 v5, v44, v13                                // 000000002790: 760A1B2C
	v_fmac_f32_e32 v4, v11, v12                                // 000000002794: 7608190B
	v_fmac_f32_e32 v8, v47, v22                                // 000000002798: 76102D2F
	v_fmac_f32_e32 v9, v46, v20                                // 00000000279C: 7612292E
	v_fmac_f32_e32 v5, v3, v23                                 // 0000000027A0: 760A2F03
	v_fmac_f32_e32 v4, v2, v21                                 // 0000000027A4: 76082B02
	v_add_f32_e32 v2, v8, v9                                   // 0000000027A8: 02041308
	v_add_f32_e32 v2, v5, v2                                   // 0000000027AC: 02040505
	v_add_f32_e32 v2, v4, v2                                   // 0000000027B0: 02040504
	ds_bpermute_b32 v3, v10, v2                                // 0000000027B4: D87E0000 0300020A
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 0000000027BC: BF8C0070
	s_barrier                                                  // 0000000027C0: BF8A0000
	v_add_f32_e32 v2, v2, v3                                   // 0000000027C4: 02040702
	ds_swizzle_b32 v3, v2 offset:swizzle(SWAP,16)              // 0000000027C8: D87A401F 03000002
	s_waitcnt lgkmcnt(0)                                       // 0000000027D0: BF8CC07F
	v_add_f32_e32 v2, v2, v3                                   // 0000000027D4: 02040702
	ds_swizzle_b32 v3, v2 offset:swizzle(SWAP,8)               // 0000000027D8: D87A201F 03000002
	s_waitcnt lgkmcnt(0)                                       // 0000000027E0: BF8CC07F
	v_add_f32_e32 v2, v2, v3                                   // 0000000027E4: 02040702
	ds_swizzle_b32 v3, v2 offset:swizzle(SWAP,4)               // 0000000027E8: D87A101F 03000002
	s_waitcnt lgkmcnt(0)                                       // 0000000027F0: BF8CC07F
	v_add_f32_e32 v2, v2, v3                                   // 0000000027F4: 02040702
	ds_swizzle_b32 v3, v2 offset:swizzle(SWAP,2)               // 0000000027F8: D87A081F 03000002
	s_waitcnt lgkmcnt(0)                                       // 000000002800: BF8CC07F
	v_add_f32_e32 v2, v2, v3                                   // 000000002804: 02040702
	ds_swizzle_b32 v3, v2 offset:swizzle(SWAP,1)               // 000000002808: D87A041F 03000002
	s_waitcnt lgkmcnt(0)                                       // 000000002810: BF8CC07F
	v_add_f32_e32 v2, v2, v3                                   // 000000002814: 02040702
	ds_write_b32 v7, v2                                        // 000000002818: D81A0000 00000207
	s_waitcnt lgkmcnt(0)                                       // 000000002820: BF8CC07F
	s_barrier                                                  // 000000002824: BF8A0000
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000002828: BE822000
	s_cbranch_execz 12                                         // 00000000282C: BF88000C <c662_rmsnorm_bwd_mixnorm+0xe60>
	s_load_dwordx2 s[0:1], s[4:5], 0x50                        // 000000002830: C0060002 00000050
	ds_read_b32 v2, v6                                         // 000000002838: D86C0000 02000006
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 000000002840: D28F0000 00020082
	s_waitcnt lgkmcnt(0)                                       // 000000002848: BF8CC07F
	v_mov_b32_e32 v3, s1                                       // 00000000284C: 7E060201
	v_add_co_u32_e32 v0, vcc, s0, v0                           // 000000002850: 32000000
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc                     // 000000002854: 38020303
	global_store_dword v[0:1], v2, off                         // 000000002858: DC708000 007F0200
	s_endpgm                                                   // 000000002860: BF810000
