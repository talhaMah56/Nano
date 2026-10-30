# BOARD.md — task board for the agent loop (protocol: docs/LOOP.md)

NEEDS-USER: (none)

Roles in use: (none)

Tiers: **0** = must ship (critical path). **1** = restored, should ship. **2** = stretch, only if claimed by 11-06.
**3** = stretch, only after Tier 1 is DONE. Every non-Tier-0 card must be in REVIEW by **11-18** or it is DROPPED.
Card details: `docs/TASKS.md` (this board overrides its session numbers, dates and scope).

| ID | Tier | Title | Depends on | Due (REVIEW by) | Status | GPU column |
|---|---|---|---|---|---|---|
| AMD-2 | 0 | HIP RMSNorm via hipify, static gfx908/90a/942 + nvcc/clang control | AMD-1 | 11-03 | CLAIMED(main-session, 10-29): `kernels/cuda/rmsnorm.cu` v0 compiles (sm_90a: fwd768 32 regs, bwd768 62, 0 spills) | none (CPU only) |
| ANA-1 | 0 | Roofline v1 with ncu bytes (BAS-6 job 3417340) | BAS-6 | 11-04 | READY | none |
| BAS-5 | 0 | CPU per-op references + table | INF-2 | 11-05 | CLAIMED(main-session, 10-29): job 3484920 queued; then write `results/cpu_reference_table.md` | queued 3484920 |
| KC-1 | 0 | CUDA RMSNorm: torch wrapper, gates, timing, ablations | AMD-2 | 11-06 | BLOCKED(AMD-2) | |
| INT-1 | 0 | Integration shim + graph-break gate G3 | KT-1..3 | 11-10 | READY | |
| KC-2 | 0 | CUDA QK-norm + RoPE | KC-1 | 11-12 | BLOCKED(KC-1) | |
| AMD-4 | 0 | HIP ports of qkrope and ce; complete static AMD table | AMD-2, KC-2, KC-3 | 11-19 | BLOCKED(KC-2, KC-3) | none |
| KC-3 | 0 | CUDA softcap + cross-entropy | KC-1 | 11-17 | BLOCKED(KC-1) | |
| INT-2 | 0 | Loss-curve gates G4 per integrated variant | INT-1 | 11-17 | BLOCKED(INT-1) | |
| TUNE-1 | 0 | Fix the top ncu finding per Tier-A kernel | KC-1..3 | 11-19 | BLOCKED(KC-*) | |
| INT-3 | 0 | End-to-end timing, first pass | INT-2 | 11-20 | BLOCKED(INT-2) | |
| INT-5 | 0 | D5 per-kernel go/no-go (operator) | INT-3 | 11-20 | BLOCKED(INT-3) | none |
| INT-4 | 0 | Data freeze job + before/after nsys | INT-5 | 11-24 | BLOCKED(INT-5) | |
| INT-6 | 0 | Final measurement campaign | INT-4 | 11-26 | BLOCKED(INT-4) | |
| AMD-5 | 0 | Cross-generation static comparison (sm_70/80/90a) | KC-1, KC-2 | 11-19 | BLOCKED(KC-2) | none |
| KT-4 | 1 | Chunked lm_head + softcap + CE (variant A only) | KT-3 | 11-12 | READY | |
| KT-5 | 1 | Triton GEMM study at nanochat shapes (sweep version: tutorial matmul + configs vs cuBLAS, ISA) | KT-3 | 11-12 | READY | |
| AMD-3 | 1 | FA3 cubin dissection (full) + Triton attention cross-compile if KT-6 exists | AMD-1 | 11-12 | READY | none |
| DEC-1 | 2 | bf16 weight cache + corrected MBU + decode nsys | INF-1 | 11-16 | READY | |
| DEC-2 | 2 | Sync removal + CUDA graphs for the T=1 step | DEC-1 | 11-18 | BLOCKED(DEC-1) | |
| KT-7 | 2 | relu² in the c_fc GEMM epilogue + derivative in c_proj dX | KT-5 | 11-18 | BLOCKED(KT-5) | |
| KT-6 | 3 | Triton FA2-style attention forward vs FA3 vs SDPA | Tier 1 DONE | 11-18 | BLOCKED(Tier 1) | |
| KT-8 | 3 | Muon polar-express epilogues | Tier 1 DONE | 11-18 | BLOCKED(Tier 1) | |
| KC-4 | 3 | CUDA GEMM teaching ladder | Tier 1 DONE | 11-18 | BLOCKED(Tier 1) | |
| ANA-2/3/7 | 0 | Final figures (after INT-6) | INT-6 | 11-28 | BLOCKED(INT-6) | none |
| ANA-8 | 0 | Final deck (11-30 presentation; Hamza presents, speaker notes required) | ANA-7 | 11-29 | BLOCKED | none |
| ANA-10 | 0 | Final report | ANA-8 | 12-07 | BLOCKED | none |

## Review findings
(reviewers append per card: `### <ID> review <n> — FAIL|PASS by <reviewer>, <date>` + numbered findings)
