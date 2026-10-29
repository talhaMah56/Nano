# ADM-1 — hardware-access requests (drafts, ready to send)

Prepared 2026-10-18 by Claude for Talha. **Nothing here has been sent** — every route needs a human (email, web form, or a faculty PI).
Fill the `<…>` placeholders, send, then log the date in `STATUS.md` → "Access status". Follow-ups: **10-27**, decision **D2 on 11-03**; late access is still accepted until 12-01.
Facts used below come from `docs/research/research_amd-path.json` and S1/S2 measurements; nothing is invented.

Checked by Claude (read-only, 2026-10-18): `sacctmgr show assoc account=bffp-dtai-gh` lists 23 users and **Hamza has no DeltaAI account** in this allocation (only `tmahmood1`).

| # | Route | Who acts | Expected wait | What it unlocks |
|---|---|---|---|---|
| 1 | Instructor email (scoping + allocation manager) | Talha | days | static-AMD acceptance, DARWIN V100/MI100, who can move DeltaAI → Delta units |
| 2 | NCSA help desk: Delta GPU (gpuMI100x8, gpuA100x4) for `bffp` | Talha (cc instructor) | unknown | measured MI100/MI210 + A100 rows (AMD-6, AMD-5) |
| 3 | `ssh login.delta.ncsa.illinois.edu` + `accounts` | Talha (needs Duo; Claude cannot do it) | minutes | tells whether a Delta login already exists |
| 4 | Chameleon CHI@TACC `gpu_mi100` | advisor/instructor as PI | ≈ 1 business day after PI approval | bare-metal MI100 with a ROCm image |
| 5 | AMD AI Developer Program (MI300X cloud credit) | Talha | sign-up only; **do not redeem** until the AMD-6 smoke pack is ready | MI300X (gfx942) rows |
| 6 | ACCESS: own projects, or an Explore request with the advisor as co-PI | Talha + advisor letter | "most approved within 2 business days" (how-to page), up to 2 weeks (policy page) | Delta GPU SUs independent of the class allocation |

---

## 1. Email to the course instructor

**To:** `<instructor email>`  **Subject:** CISC662 project (Talha & Hamza Mahmood): platform change, AMD scope and Delta usage

Dear Professor `<name>`,

Our project ("Optimizing nanochat down to the assembly level: NVPTX and AMDGCN") is under way on NCSA DeltaAI. Three points need your input:

1. **Platform change.** The proposal named DARWIN V100 as the primary GPU; we will instead work entirely on NCSA (DeltaAI GH200, H100-class, plus NCSA Delta for AMD), and we are not asking for DARWIN. Baselines are measured (e.g. 8.05 s per 100 training steps with torch.compile + FlashAttention-3; eager 15.08 s).
2. **AMD half.** DeltaAI has no AMD GPUs and ROCm cannot be installed on its aarch64 nodes. We can already compile our kernels to AMDGCN (gfx908/gfx90a/gfx942) and analyse the ISA statically, and we have Delta access for a small set of measured AMD runs (point 3), pending that node coming out of maintenance. **Would a static-only AMD comparison, clearly labelled as such, be acceptable if no AMD hardware arrives by early November?**
3. **Delta access.** We already have the Delta accounts `bffp-delta-gpu` and `bffp-delta-cpu` and will run the AMD measurements on Delta's `gpuMI100x8` node within a small budget (at most about 6 Delta GPU SU, well inside the class pool). Please tell us if you would like us to stay under a different limit.

So far we have used 0.6 of the GH200 hours for this project and plan at most 25 over the semester.

Thank you,
Talha Mahmood (`tmahmood1`), on behalf of Talha and Hamza Mahmood (Talha does the coding; Hamza presents)

## 2. Email to the NCSA help desk

**To:** help@ncsa.illinois.edu  **Cc:** `<instructor email>`  **Subject:** Adding NCSA Delta GPU (gpuMI100x8, gpuA100x4) to project bffp for a course project

Hello,

I am a student on the ACCESS class project `bffp` (DeltaAI account `bffp-dtai-gh`, user `tmahmood1`). For a course project comparing GPU kernels on NVIDIA and AMD we need a small amount of time on **NCSA Delta GPU**: the `gpuMI100x8` partition (MI100 gfx908 and the MI210 gfx90a node) and, if possible, `gpuA100x4`.

- Could NCSA Delta GPU be added to project `bffp` by an ACCESS exchange from DeltaAI (we are asking our instructor, who manages the allocation, to initiate it)? We estimate **20–40 Delta GPU SU** in total, short batch jobs only.
- Only `tmahmood1` (Talha Mahmood) needs access; Hamza does not run jobs.
- The Unix groups `delta_active_users` and `delta_bffp` already exist on my account — does that mean a Delta login is already possible once units exist?

Thank you,
Talha Mahmood

## 3. Delta login check (2 minutes, from a laptop with Duo)
```bash
ssh tmahmood1@login.delta.ncsa.illinois.edu    # Duo prompt
accounts                                       # which allocations appear? note them in STATUS.md
```

## 4. Chameleon Cloud (needs a faculty PI)
Ask the advisor or instructor to request **PI eligibility** at chameleoncloud.org (faculty/research staff only), create a project, and add both students. Hardware: CHI@TACC node type `gpu_mi100` (8 bare-metal nodes, 2× MI100 each), image `CC-Ubuntu24.04-ROCm`. Leases are short: work inside tmux on the node and rsync results back after every run.

## 5. AMD AI Developer Program
Sign up with the student email. Note which credit terms are shown (sources disagree: "$100 for 30 days" vs "25 h for 10 days") and whether a credit card is required. **Do not redeem** the credit until AMD-6's smoke pack (scripts + expected outputs) is ready, because the VM is destroyed on expiry.

## 6. ACCESS allocations
1. Log in at allocations.access-ci.org and list every project where Talha or Hamza is PI, co-PI or user (the proposal mentions "an NSF ACCESS allocation").
2. If one of them can exchange to **NCSA Delta GPU**, use the exchange calculator to read the DeltaAI → Delta GPU rate, record it, and request ≈ 20–40 Delta GPU SU.
3. Otherwise submit an **ACCESS Explore** request, Talha as PI with the advisor as co-PI (advisor letter required), resource NCSA Delta GPU, justification: "≈ 40 GPU-hours on gpuMI100x8 and gpuA100x4 to measure hand-written HIP/Triton kernels against their CUDA versions for a course project; all development and NVIDIA work is on DeltaAI." Record the request number.
