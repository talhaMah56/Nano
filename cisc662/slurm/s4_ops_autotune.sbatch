#!/bin/bash
# Session 4, task BAS-4: the remaining per-op baseline rows -- torch.compile(mode="max-autotune-no-cudagraphs)
# for the 9 GEMM shapes (forward K=768/3072 and the dW shapes with K=32768), timed in the SAME job as the eager
# cuBLAS rows of those shapes (same-GPU rule, PLAN 7.1). The S2 eager/compile rows of every op (job 3406329)
# stay as they are. Autotune benchmarking happens in kbench's untimed first call; Inductor's autotune results
# are cached on /work/nvme (TORCHINDUCTOR_CACHE_DIR), so trials 2-3 reuse them. Time: <= 22 min.
#SBATCH --job-name=c662-s4-bas4
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=00:25:00
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out

source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh
JOBID=${SLURM_JOB_ID:-local}
OUT=$CISC662_REPO/cisc662/results/runs/s4_bas4_${JOBID}
mkdir -p "$OUT"
DEADLINE=$(( JOB_T0 + 23 * 60 ))
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
    --format=csv -lms 500 > "$OUT/gpu_monitor.csv" &
MON_PID=$!
export TORCHINDUCTOR_COMPILE_THREADS=16
TB=$(( DEADLINE - $(date +%s) - 90 ))
run_step bench "$OUT/bench.log" timeout -k 15 $(( TB + 75 )) \
    python -m cisc662.bench.ops_at_shapes --trials 3 --forms eager,autotune --cases '^(eager|autotune)_gemm' \
        --out-dir "$CISC662_REPO/cisc662/results/db/ops_baseline" --job-id "$JOBID" --time-budget-s "$TB"
grep -E '^SUMMARY|autotune_gemm' "$OUT/bench.log" | tail -n 20 || true
kill $MON_PID 2>/dev/null || true
echo "FAILED STEPS:${FAILED_STEPS:- none}"
