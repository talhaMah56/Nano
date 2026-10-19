# AGENT_GUIDE — how to execute a task from TASKS.md on DeltaAI

This guide is for any agent (or person) picking up a task card. Follow it exactly; it encodes the cluster's billing rules and the project's measurement protocol. The plan is [PLAN.md](PLAN.md); live state is [STATUS.md](STATUS.md).

## 1. Start of every task (5 minutes, no exceptions)
```bash
ssh gh-login01.delta.ncsa.illinois.edu            # tmux session 'cisc662' lives on THIS node only
tmux attach -t cisc662 || tmux new -s cisc662
source /u/tmahmood1/project/nanochat/cisc662/env/env.sh   # venv, caches, CC=gcc, cwd = repo
sed -n 1,80p cisc662/docs/STATUS.md               # what is done, what is in flight, gotchas
accounts | grep bffp; tail -3 $CISC662_WORK/logs/job_ledger.csv; quota | grep -E "nvme|/u/"
squeue -u $USER                                   # never start a GPU job while an unrelated one of ours is queued, unless the card says so
```
**Login-node process rule (NCSA enforces it with an auto-killer and an email to the user):** no process outside the batch queues may use more than **30 minutes of CPU time** *or* live longer than **7 days of wall-clock time**, idle or not (on 2026-10-19 two idle `gvfsd` daemons with 0 CPU-seconds, alive since 10-12, were killed on gh-login02/03 for the 7-day limit). Wall-clock consequences: every login-node helper must be bounded (`slurm/queue.sh` exits after ≤ 12 h, `slurm/monitor.sh` after ≤ 6 h); a tmux session is opened for a working session and closed (`tmux kill-session -t cisc662`) at its end — never left for days; at the start of every session check *all four* login nodes for leftovers with `for h in gh-login0{1,2,3,4}; do ssh -o BatchMode=yes $h 'ps -u $USER -o pid,etime,time,comm --sort=-etimes | head -6'; done` and kill our own stale ones (tmux servers, monitors, idle shells older than ~3 days); `gvfs-daemon`/`gvfs-metadata` are masked for this user (`~/.config/systemd/user/*.service → /dev/null`, undo with `systemctl --user unmask …`) because nothing here uses them; long-lived VS Code Remote sessions keep `systemd --user`/`dbus-daemon` alive for days, so close the remote connection (or "Kill VS Code Server on Host") at least weekly.
CPU-time consequences: On 2026-10-17 a filesystem-wide `bfs … / -type d -name claude_skills` search spawned by the Claude Code harness was killed at 36 CPU-minutes, and the tmux server died the same night. Therefore: never run `find`, `bfs`, `grep -r`, `du` or any search from `/`, `/work`, `/u` or `$HOME` — search only inside `/u/tmahmood1/project/nanochat` and `$CISC662_WORK`; split cross-compile and analysis batches so no single process passes ~20 CPU-minutes; do not run perpetual monitor loops (use `squeue -u $USER` on demand, or `slurm/monitor.sh`, which now exits after 6 h); a long Claude Code session accumulates CPU time in its own process — check `ps -o etime,time -p <claude pid>` and restart the session (`/resume` keeps the conversation) before it nears 20 CPU-minutes.

Read the task card, its `Depends on` cards' deliverables (they must exist), and the listed `Inputs` (research JSONs under `docs/research/`, scratch scripts on `/work/nvme`). Do not re-derive facts that are in `docs/research/` or `PLAN.md` §2–§4.

## 2. Where things go
- Code: `/u/tmahmood1/project/nanochat/cisc662/…` (git branch `cisc662`). Never edit files outside `cisc662/` except the three documented opt-in patches (`nanochat/flash_attention.py`, `scripts/base_train.py`, `nanochat/common.py`); new behaviour goes behind env vars with upstream defaults unchanged. Kernels enter the model through the integration shim (`cisc662/integrate.py`, loaded only by `cisc662/bench/train_wrap.py`), which monkeypatches `nanochat.gpt` at run time — that is not an upstream edit.
- Heavy data: `$CISC662_WORK = /work/nvme/bffp/tmahmood1/cisc662` (checkpoints under `nanochat_base/`, profiles under `profiles/`, logs under `logs/`, scratch under `scratch/<task-id>/`). Keep the project < 50 GB.
- **Never write to `/projects/bffp` or `/work/hdd/bffp`** (over quota, writes fail or hurt the whole class).
- Results that figures use: small JSON/CSV/MD under `cisc662/results/`; raw `.nsys-rep`/`.ncu-rep` stay on `/work/nvme` and are referenced by path.

## 3. Running GPU work — the only allowed shape
Every GPU job is an `sbatch` script in `cisc662/slurm/` that starts with:
```bash
#SBATCH --job-name=c662-<task-id>
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4                 # 1x billing. ghx4-interactive is 2x and allows one job; test/full are staff-only
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16               # 16–32; never > 72 per GPU
#SBATCH --mem=64g                        # always explicit; never > 110g per GPU
#SBATCH --time=00:25:00                  # ≤ 30 min backfills in minutes; > 45 min needs a reason in STATUS.md
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out
# profiling jobs add:  #SBATCH --constraint=nvperf   (free, node-shared, policy-compliant)
source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh   # env, HF_HUB_OFFLINE=1, ledger, run_step, use_local_base_dir
```
Rules:
1. **Never** `--exclusive`, `--gpus=4`, `--constraint=perf`, `--constraint=xpmem` (whole node = 4 SU/h). Never more than 2 project jobs queued at once.
2. Run `bash -n script && sbatch --test-only script` before the first real submission of any new script. Syntax-check Python with `python -m py_compile` on the login node first — a queue wait is the expensive part.
3. Put many short experiments into one job (`run_step` helper) rather than many jobs. Timing comparisons must run on the same GPU in the same job, interleaved ABAB.
4. Throwaway checkpoints from timing runs: call `use_local_base_dir` (node-local `/tmp`, purged after the job). Real checkpoints: `NANOCHAT_BASE_DIR=$CISC662_WORK/nanochat_base`.
5. FA3-labelled runs must abort if `nanochat.flash_attention.HAS_FA3` is False (silent SDPA fallback otherwise). `kbench.py` and the integration shim check this; do the same in new harnesses.
6. Start `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv -lms 500 > $OUT/gpu_monitor.csv &` in every timing job (see `s1_train_timing.sbatch`).
7. After the job: read the whole `.out`, confirm `FAILED STEPS: none`, run the task's analysis script, copy summaries into `cisc662/results/`, and record elapsed SU (ledger line) in STATUS.md.
8. Interactive GPU debugging (rare): `srun -A bffp-dtai-gh -p ghx4 --gpus=1 --cpus-per-task=16 --mem=64G --time=00:20:00 --pty bash` inside tmux; exit the moment you are done — idle allocations are billed.
9. CPU-only work (CPU reference, data prep): `--cpus-per-task=2 --mem=16g` without `--gpus-per-node` (≈ 0.144 SU/h; a 72-core job bills the full GPU rate, 1 SU/h — CPU-only jobs still draw SU). Cross-compiles, `nsys stats`, plotting: login node, `nice -n 19`, ≤ 8 threads (`MAX_JOBS=8`, `TORCHINDUCTOR_COMPILE_THREADS=8` are set in env.sh), each process under 20 CPU-minutes. No torch.compile, no benchmarks, no training on the login node.
10. Spend accounting: `/sw/user/scripts/jobcharge -u tmahmood1 -d 30 --detail` is the authoritative number; the ledger line `_common.sh` appends (SU computed from the job's Slurm billing weight) is the per-job cross-check; `srun` probes that bypass `_common.sh` must be logged by hand (job ID + SU) in STATUS.md.

Budget: the card's `GPU budget` is a cap. If a job fails, fix the cause on the login node before resubmitting; two failed submissions of the same job → stop and write the problem into STATUS.md.

## 4. Measurement protocol (short form; PLAN.md §7 is binding)
- Kernel times: CUDA events, L2 flush, warm-up, 100 reps × 3 trials (use `cisc662/bench/kbench.py`). Never Python wall clock for kernels.
- Every speedup names its baseline (eager or compile), the torch/Triton versions (2.9.1 / 3.5.1), clock mode, and L2 state. "≥ 1.3×" only with bootstrap-CI lower bound ≥ 1.3.
- No timing in any table before gates G1–G2 pass (G3–G4 for integrated kernels). Write `gates_passed` into the result JSON. Tests: in a GPU job `pytest cisc662/tests -q`; on the login node only `pytest cisc662/tests -q -m "not real_shape and not gpu"` (tiny shapes; the real-shape fp64 CE reference alone is 8.6 GB per tensor and would hit the login-node memory and CPU-time limits).
- End-to-end numbers always come with the nsys per-step GPU kernel time next to the wall-clock s/100 steps (PLAN.md §7.1): the wall clock includes ≈ 6 ms of host time per step.
- nsys = attribution; ncu = counters (base clocks by default; `--clock-control none` if a time is shown); always kernel-filtered.

## 5. Kernel development conventions
- One file per kernel under `kernels/triton|cuda|hip/`, containing: the kernel, a `reference()` in fp32/fp64 PyTorch, a `torch.autograd.Function` or `torch.library` wrapper, and `if __name__ == "__main__":` self-test at a small shape.
- Triton: always pass divisibility hints (contiguous last dim multiple of 16) — the S1 cross-compiles showed scalar 16-bit loads without them. Record `kernel.n_regs`, `kernel.n_spills`, and dump `ptx`/`sass` via `kernel.asm` for the ISA table.
- CUDA: `torch.utils.cpp_extension.load(..., extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "--ptxas-options=-v"])`, `CC=gcc CXX=g++` (env.sh). Keep the ptxas `-v` output (registers, spills) in the result JSON. Provenance matters for the ISA tables: nvcc/ptxas 12.9 and cuobjdump/nvdisasm come from the **system** HPC SDK 25.5 module, Triton uses its **bundled** ptxas 12.8, HIP uses the system CCE clang 20 (LLVM 20.1.6), Triton's AMD path uses its bundled LLVM 22 — record `nvcc --version` / tool versions in every ISA row (PLAN.md R10). Only torch/Triton/FA3 live in our own venv, which is a stated deviation from the proposal's "own toolchains" line.
- HIP: `HIPIFY=$CISC662_WORK/scratch/amd-rocm-planning/dl_hipify/hipify-perl` (not on PATH); `FR=$CISC662_WORK/scratch/amd-rocm-planning/hipify_test/fake_rocm` (HIP 6.2 headers from the Triton wheel); `CB=/opt/cray/pe/cce/20.0.0/cce-clang/aarch64/bin` (clang, llvm-objdump, llvm-readelf — not on PATH). `perl $HIPIFY k.cu > k.hip`; fix wave64 (64-bit masks, `warpSize` assumptions, LDS ≤ 64 KiB, DPP for hot reductions); then `env -u CPATH $CB/clang -x hip --offload-arch=gfx90a --cuda-device-only --rocm-path=$FR -DHIP_ENABLE_WARP_SYNC_BUILTINS -O3 -S k.hip -o k.gfx90a.s` and `… -c --no-gpu-bundle-output -o k.gfx90a.hsaco`; `$CB/llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx90a k.gfx90a.hsaco`; `$CB/llvm-readelf --notes k.gfx90a.hsaco`. The `-nogpulib -nogpuinc` + `mini_hip.h` form (`$CISC662_WORK/scratch/toolchain-feasibility/hip_clang/`) works only for shim-based kernels that do not include `<hip/hip_runtime.h>`. Labels: every AMD number is **STATIC** unless measured on AMD hardware.
- Numerics: fused kernels accumulate in fp32; bf16 only at the store. Tolerances come from `tests/tolerances.py`, never ad hoc.

## 6. Finishing a task (the handoff is the deliverable)
1. Deliverables exist at the paths the card names; the acceptance criteria are checked, not assumed (run the tests; paste the key numbers).
2. Update `docs/STATUS.md`: Done (with result paths and job IDs), Gates status, Decisions, Budget (session and cumulative SU), Known gotchas, Next actions. Keep it under ~150 lines; move history into `docs/findings.md`.
3. Add a short ISA-card entry to `docs/findings.md` for every kernel variant measured (template in TASKS.md, ANA-5).
4. Do not `git commit` or `git push` unless the task card or the user says so; leave the tree clean of stray files (`git status`).
5. If something is blocked (access, quota, queue), say so in STATUS.md with the exact error and what was tried. Never report a number that was not produced by a job you can name.

## 7. Things that look reasonable but are forbidden
- Using `ghx4-interactive` for anything longer than 10 minutes, or leaving an `srun --pty` shell idle.
- Requesting a whole node "to be safe", or > 110 GB memory per GPU (bills extra).
- Running `uv sync` without `--frozen --inexact` (it rewrites `uv.lock` / removes project extras); installing anything into the shared venv for a one-off experiment (use a separate venv under `$CISC662_WORK/venvs/<name>`).
- Re-downloading FA3 or datasets inside a GPU job (pre-fetch on the login node; jobs run `HF_HUB_OFFLINE=1`).
- Claiming speedups from ncu durations, from warm-L2 microbenchmarks, or against a baseline that silently fell back to SDPA.
- Writing results into the deck by hand: every figure is generated by a script in `analysis/` from files in `results/` and stamped with its job ID.
- Filesystem-wide searches (`find /`, `bfs /`, `grep -r /work …`), perpetual `watch`/`while true` loops, or any login-node process that could pass 30 CPU-minutes or 7 days of wall-clock time (NCSA kills it; the user gets a policy email). Leaving tmux sessions, monitors or shells running between working sessions counts.
- Relying on tmux for anything that must survive: it is a convenience, the server can be killed; all long work is `sbatch`, all state is in git/STATUS.md.
