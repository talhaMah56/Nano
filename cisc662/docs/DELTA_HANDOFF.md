# DELTA_HANDOFF — run the measured-AMD experiments on NCSA Delta (MI100 / MI210)

Written 2026-10-20 for an agent that will work on **Delta** (not DeltaAI) and has not seen this project. Read this file, then `docs/AGENT_GUIDE.md` (rules still apply, except where this file overrides them), then the AMD-6 card in `docs/TASKS.md`. This file is the task card for **AMD-6** plus the environment facts needed to do it. Every measurement from this work is a *measured AMD row*; everything else on the AMD side so far is STATIC (compiled, never run).

## 0. What you are doing and why
Project: CISC662 "Optimizing nanochat down to the assembly level: NVPTX and AMDGCN". DeltaAI (GH200) holds the NVIDIA results. DeltaAI has no AMD GPUs, so the AMD half is currently static (ISA analysis only). Delta has one AMD node, `gpuMI100x8` (8 × MI100 gfx908 + 1 × MI210 gfx90a, 2 TB RAM, AMD Milan x86_64). Your job: take the three Tier-A Triton kernels already written and tested on NVIDIA, run them on MI100/MI210, and produce **measured** timings and profiler data so the static AMD numbers can be validated.

**Goal (in priority order; stop when the budget says stop):**
1. Smoke test: ROCm PyTorch sees the GPU (`rocminfo`, `torch.version.hip`, `torch.cuda.get_device_properties(0).gcnArchName`).
2. Correctness: gates G1/G2 (tests in `cisc662/tests/`) pass on the AMD device for rmsnorm, qkrope, ce. Report failures honestly; a failure on AMD is a finding, not something to hide.
3. Timing: kbench microbenchmarks of the three kernels vs eager PyTorch (and vs `torch.compile` if it works on ROCm) at the R12 shapes.
4. Profiling: `rocprofv3 --kernel-trace --stats` for the same runs; compare the kernels' VGPR/AGPR/LDS/occupancy from the trace with our static numbers (`cisc662/results/static_isa.csv`).
5. Optional: `rocprof-compute profile ... --roof-only` on 2–3 kernels; static tuning pairs re-timed (e.g. `waves_per_eu`, `num_warps`); R12 training smoke on MI210 (32 GB) with `--device-batch-size=8`.

**Acceptance (from the task card):** at least one measured kernel time per Tier-A kernel on at least one AMD architecture, with rocprofv3 timings, and the static VGPR/LDS numbers validated against `kernel_trace.csv`.

## 1. Hard limits — do not exceed
- **Budget: at most 6 Delta GPU SU for this whole task.** The partition's charge factor is 0.25 per GPU-hour (the AMD-6 card says ≤ 24 MI100/MI210 GPU-hours). Before any job, run `sacctmgr`/`accounts` and the project's usage command to see the real balance and which **Delta account** you charge — it is NOT `bffp-dtai-gh` (that is the DeltaAI class pool and must not be used here). If you cannot identify an account that is authorised for Delta GPUs, **stop and tell the user**; do not guess an account name and do not try the DeltaAI account.
- Re-plan checkpoint: after 2 SU spent, stop and write down what is done and what the rest would cost.
- One GPU per job, **walltime ≤ 30 min** per job (short jobs start fast), ≤ 2 jobs of this project queued at once. Never ask for the full node (9 GPUs).
- Request GPUs **by type**, always: `--gpus=mi100:1` or `--gpus=mi210:1` (with `--partition=gpuMI100x8`). Plain `--gpus=1` returns an unpredictable mix and invalidates results. Record the `gcnArchName` printed inside every job.
- Login nodes: any process is killed after 30 CPU-minutes or 7 days of wall-clock time. No benchmarks, no compiles of big things, no test suites on login nodes; use `nice -n 19` for anything that runs more than a minute; use short jobs for the rest. No idle background daemons. If you use tmux, close it at the end of the session and check for stray processes (`ps -u $USER`) before leaving.
- Do not `sleep`-poll in the foreground. Use `squeue -u $USER` and read logs after the job ends; if your harness supports a bounded monitor, use that (see `cisc662/slurm/monitor.sh`, which is bounded to 6 h).
- No `git commit`/`push` without the user's explicit OK. Branch is `cisc662`. Remote is `nano` (https://github.com/talhaMah56/Nano.git); the agent has no credentials, the user pushes.
- Files: `/u` is shared Harbor storage mounted on both DeltaAI and Delta (100 GB / 750k inode quota). Put large things (venv, containers, caches, profiles) under `/work/nvme/<your Delta project>/<user>/cisc662/` — `/work` is also shared with DeltaAI. No backups on `/work`.

## 2. Environment facts (what is the same, what differs)
| Item | DeltaAI (where results so far come from) | Delta (where you work) |
|---|---|---|
| CPU arch | Grace aarch64 | **AMD Milan x86_64** |
| Python venv | uv venv with torch 2.9.1+cu128, triton 3.5.1 | **cannot be reused** (aarch64 CUDA wheels). Build a new x86_64 ROCm venv |
| GPU | GH200 sm_90a, warp 32 | MI100 gfx908 / MI210 gfx90a, **wavefront 64**, 32 GB HBM2 |
| Matrix unit | tensor cores (`HGMMA`) | MFMA (`v_mfma_*`) |
| LDS / shared | up to 227 KB | **64 KiB per CU** |
| Repo | `/u/tmahmood1/project/nanochat`, branch `cisc662` | same files (shared `/u`) — verify with `git -C … status` before you start |

Software plan, in order of preference (record whichever you use and its exact versions in every result row):
1. Pre-staged container: `/sw/external/MI100/pytorch-2.10.0-amdgpu.sif` (check it exists; note the torch version gap vs our 2.9.1). Run with `apptainer exec --rocm …`.
2. Else an x86_64 venv with `torch==2.9.1+rocm6.4` (about 4.5 GB, put on `/work/nvme`). Triton ships inside that torch wheel's dependency (ROCm Triton, version may differ from 3.5.1 — record it). Install with `uv` or `pip` in a **job or with `nice -n 19`**, not a long login-node process.
Set `HF_HUB_OFFLINE=1`, and use `NANOCHAT_ATTN_IMPL=sdpa` (FA3 is NVIDIA-only). `torch.cuda` APIs work on ROCm builds (device string stays `"cuda"`).

## 3. The code you will run (all already in the repo)
- Kernels: `cisc662/kernels/triton/rmsnorm.py`, `qkrope.py`, `ce.py`.
- Gates: `cisc662/tests/` (`test_rmsnorm*.py`, `test_qkrope.py`, `test_ce.py`, shared tolerances in `cisc662/tests/gates.py`). Gate semantics: G1 = value error vs an fp64 reference on the same inputs; G2 = gradient error and ≤ 2× the stock PyTorch error. Tolerance amendment A1 in `docs/PLAN.md`. Do not loosen tolerances to make AMD pass.
- Timing: `cisc662/bench/kbench.py` (CUDA events — also works as HIP events through torch; L2 flush; 3 process trials; bootstrap CIs) and suites `cisc662/bench/suite_*.py` (e.g. `suite_ce.py`; check for rmsnorm/qkrope suites with `ls cisc662/bench`). Result rows go to `cisc662/results/db/<kernel>/*.json`; use a **separate subdirectory or filename prefix `amd_`** so DeltaAI rows are never overwritten.
- Slurm templates: `cisc662/slurm/_common.sh` (run_step, SU ledger — the ledger is written for the DeltaAI formula; make a Delta copy `slurm/_common_delta.sh` with the Delta account and 0.25 factor, do not edit the DeltaAI one), `s4_ce.sbatch` as a model job script.
- Static numbers to compare against: `cisc662/results/static_isa.csv`, ISA dumps under `cisc662/results/isa/<kernel>/triton_<variant>/gfx908|gfx90a|gfx942/`, and the launch options in `cisc662/analysis/kernel_registry.py` (num_warps etc. were tuned for GH200; these are the options the static AMD numbers used).

## 4. Known things that will break on AMD (read before running)
1. **`ce.py` variant `rec_approx`** (the tanh.approx ablation) uses PTX inline asm (`tanh.approx.f32`). It only exists on NVIDIA. Skip it on AMD (filter the variant or catch the compile error and record "N/A: NVPTX-only").
2. Triton kernels written with `num_warps` tuned for warp-32: on AMD the same `num_warps=8` means 8 wavefronts of 64 lanes = 512 threads. Block shapes that assumed 32-lane reductions still work (Triton handles it) but the achieved occupancy and register use differ — that is exactly what we want to measure; record it, do not "fix" it silently.
3. **tanh/exp/log**: `ce.py` has its own `_softcap`, `_exp`, `_log1p` helpers with an interpreter fallback. On ROCm Triton, `libdevice` is the AMD one; confirm the kernel compiles and G1 holds (precision of `exp` may differ slightly). If G1 fails only for the saturated-target row, report the measured relative error rather than adjusting the kernel.
4. bf16 on gfx908 (MI100) has no native bf16 arithmetic instructions beyond MFMA; conversions are emulated. If a bf16 kernel is very slow or fails on gfx908 but works on gfx90a (MI210), say so — it is a result.
5. `torch.compile` (Inductor) may not fully work on this ROCm stack. If it fails, record the error and benchmark eager vs ours only; do not spend more than 1 SU debugging it.
6. Any code that hardcodes `sm_90`, `cuda:90`, `NANOCHAT_ATTN_IMPL=fa3`, `cuobjdump`, `ncu` or `nsys` is NVIDIA-only. Use `grep -rn "sm_90\|cuobjdump\|ncu\|nsys\|fa3" cisc662/bench cisc662/tests` and keep those paths out of your runs.
7. Memory: MI100/MI210 have 32 GB. R12 batch 16 at seq 2048 needs more than that for eager; use `--device-batch-size=8` (this is the "R12-A2" config in `docs/PLAN.md`) for any training smoke test, and say so.

## 5. Step-by-step
1. **Access check (human step first):** `ssh login.delta.ncsa.illinois.edu` needs the user's Duo approval. If you are already on a Delta login node, run `accounts` and `sinfo -p gpuMI100x8`. Write the account name and remaining balance in `docs/STATUS.md` → "Access status". If there is no usable account, stop and report.
2. Create `cisc662/slurm/delta_env.sh` (a copy of the DeltaAI `env.sh` with the Delta paths/account, ROCm env, `HF_HUB_OFFLINE=1`). Source it from every job.
3. Build the environment (section 2) in a ≤ 30-min job on `--gpus=mi210:1` (or CPU partition if the build needs no GPU). Verify with a smoke job: `rocminfo | grep -m3 gfx`, `python -c "import torch,triton; print(torch.__version__, torch.version.hip, triton.__version__, torch.cuda.get_device_properties(0).gcnArchName)"`.
4. Run the gates on **MI210 first** (gfx90a is the closer match to our static `gfx90a` numbers and has 32 GB, native packed bf16 conversions), then on MI100 (gfx908). One job per architecture. Collect JUnit XML under `cisc662/results/runs/amd_<jobid>/`.
5. Run kbench for rmsnorm (fwd, bwd), qkrope (fwd, bwd), ce (rec, rec_2k fwd and fwd+bwd) at the R12 shapes, **eager vs ours** (add `compile` if it works), 3 process trials. Record rows only for kernels whose gates passed (the repo's `cisc662.tests.gates` helper refuses to record otherwise — do not bypass it).
6. Re-run the same cases under `rocprofv3 --kernel-trace --stats --output-format csv -d <dir> -- python -m …` (check `rocprofv3 --help` for the exact flags of the installed version). Extract per kernel: duration, VGPR/AGPR/SGPR counts, LDS bytes, workgroup size. Compare with `static_isa.csv`; write the comparison table to `cisc662/results/amd_static_vs_measured.md` (columns: kernel, arch, static VGPR/LDS/occupancy, measured VGPR/LDS, duration, achieved GB/s, percent of that GPU's HBM peak, note). MI100 HBM peak ≈ 1.2 TB/s, MI210 ≈ 1.6 TB/s (datasheet — label as datasheet, and measure a plain copy kernel in the same job to get the achievable number, as we did on GH200: 3.65 TB/s of 4.0).
7. Optional steps from section 0, item 5, only if budget remains.
8. Update `docs/STATUS.md` (what ran, job IDs, SU used, versions), `results/RUNLOG.md`, and add an ISA/finding card `docs/findings.md#amd-measured`. Leave changes uncommitted and tell the user.

## 6. What to report back (end of your session)
- Account used, jobs (IDs, partition, GPU type, wall time, SU), total SU spent vs the 6 SU cap.
- Software versions (container or venv; torch, triton, ROCm, `gcnArchName`).
- Gate results per kernel × architecture (pass/fail with the numbers).
- Timing table (median µs, CI, speedup vs eager, achieved GB/s, percent of measured copy bandwidth).
- Static-vs-measured table (step 6) and any disagreement (a disagreement is a good finding: explain it, e.g. the compiler's printed occupancy vs the recomputed one).
- Everything that failed and why, with the error text. Do not report an unmeasured number as measured; label anything projected or static.
- Anything the user must do (accounts, Duo, allocation exchange).

## 7. Contacts / references in the repo
- Plan and decision D2: `docs/PLAN.md` §7–8; access drafts: `docs/access_requests.md`; static AMD pipeline: `docs/TASKS.md` (AMD-1..AMD-4); ISA cards: `docs/findings.md`; running state: `docs/STATUS.md`.
- Delta documentation facts used here (from NCSA's system-architecture page): the MI100 node is one node, 8 MI100 + 1 MI210, 32 GB per GPU, 2 TB RAM; select GPUs by type with `--gpus=mi100:N` / `--gpus=mi210:1`; login nodes have no GPUs; `/u` and `/sw` are Harbor, `/work` is Lustre, `/projects` is Taiga.
