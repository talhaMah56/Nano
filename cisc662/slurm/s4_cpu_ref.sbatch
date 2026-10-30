#!/bin/bash
# Session 4, task BAS-5: per-op CPU references (goal i) on the Grace host: 1 pinned thread, then all 72 cores.
# ghx4 only schedules with a GPU attached, so the GPU idles; billing = 1 GPU-equivalent (72 cores ~ 1000/h, same as a GPU).
# The serial step is single-threaded and needs only a few minutes; time limit 25 min, expected use < 15 min (~0.25 SU).
#SBATCH --job-name=c662-s4-cpuref
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110g
#SBATCH --time=00:25:00
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out

source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh
JOBID=${SLURM_JOB_ID:-local}
OUT=$CISC662_REPO/cisc662/results/runs/s4_cpuref_${JOBID}
DB=$CISC662_REPO/cisc662/results/db/cpu
mkdir -p "$OUT" "$DB"
lscpu > "$OUT/lscpu.txt" || true
export CUDA_VISIBLE_DEVICES=
# pin to the first CPU Slurm gave us, memory local to it (job 3484920: --membind=0 was rejected inside the job cpuset)
CPU0=$(python -c "import os; print(min(os.sched_getaffinity(0)))")
run_step cpu_t1 "$OUT/cpu_t1.log" timeout -k 15 600 numactl --physcpubind=$CPU0 --localalloc env OMP_NUM_THREADS=1 \
    python -m cisc662.bench.cpu_ref --threads 1 --reps 10 --out "$DB/cpu_ref_t1_${JOBID}.json"
run_step cpu_t72 "$OUT/cpu_t72.log" timeout -k 15 480 env OMP_NUM_THREADS=72 \
    python -m cisc662.bench.cpu_ref --threads 72 --reps 20 --out "$DB/cpu_ref_t72_${JOBID}.json"
tail -n 12 "$OUT/cpu_t1.log" "$OUT/cpu_t72.log" || true
echo "FAILED STEPS:${FAILED_STEPS:- none}"
