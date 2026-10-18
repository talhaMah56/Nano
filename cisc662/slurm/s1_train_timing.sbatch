#!/bin/bash
# Session 1, job B2: training timing baseline "seconds per 100 steps" (steps 10..109) of config R12,
# 3 reps x {compile, eager} x {fa3, sdpa}, interleaved, all on the SAME GPU in one job.
#SBATCH --job-name=c662-s1-timing
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
OUT=$CISC662_REPO/cisc662/results/runs/s1_timing_${SLURM_JOB_ID}
mkdir -p "$OUT"
use_local_base_dir   # throwaway checkpoints -> node-local /tmp
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used \
    --format=csv -lms 500 > "$OUT/gpu_monitor.csv" &
MON_PID=$!
R12="--depth=12 --max-seq-len=2048 --device-batch-size=16 --total-batch-size=32768 --window-pattern=L --core-metric-every=-1 --sample-every=-1"

for rep in 1 2 3; do
    for mode in compile eager; do
        for attn in fa3 sdpa; do
            dis=0; [ "$mode" = eager ] && dis=1
            run_step "train_${mode}_${attn}_r${rep}" "$OUT/train_${mode}_${attn}_r${rep}.log" \
                env TORCH_COMPILE_DISABLE=$dis NANOCHAT_ATTN_IMPL=$attn \
                python -m scripts.base_train $R12 --num-iterations=110 --eval-every=-1 --model-tag=t_${mode}_${attn}_${rep}
        done
    done
done

kill $MON_PID 2>/dev/null || true
python -m cisc662.analysis.summarize_s1 "$OUT" || true
echo "FAILED STEPS:${FAILED_STEPS:- none}"
