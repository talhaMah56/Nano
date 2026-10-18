#!/bin/bash
# Session 1, job D: serial CPU reference (proposal goal i) on one full Grace CPU (72 Neoverse V2 cores).
# Same R12 model at batch 1 x 2048 tokens (identical work per step on CPU and GPU), fp32 eager on CPU:
#   GPU (compile+fa3, eager+fa3) vs CPU 72 threads vs CPU 1 thread; then decode latency at batch 1.
# Billing note: 72 cores == 1 GPU-equivalent on DeltaAI (CPU=13.89/core, GPU=1000, MAX_TRES), so no extra charge.
#SBATCH --job-name=c662-s1-cpuref
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110g
#SBATCH --time=00:30:00
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out

source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh
OUT=$CISC662_REPO/cisc662/results/runs/s1_cpuref_${SLURM_JOB_ID}
mkdir -p "$OUT"
lscpu > "$OUT/lscpu.txt" || true
SHARED_BASE=$CISC662_WORK/nanochat_base
use_local_base_dir

B1="--depth=12 --max-seq-len=2048 --device-batch-size=1 --total-batch-size=2048 --window-pattern=L --core-metric-every=-1 --sample-every=-1 --eval-every=-1"

# GPU at batch 1 (same work per step as the CPU runs)
run_step gpu_b1_compile "$OUT/train_b1-compile_fa3_r1.log" env NANOCHAT_ATTN_IMPL=fa3 \
    python -m scripts.base_train $B1 --num-iterations=110 --model-tag=c_gpu_compile
run_step gpu_b1_eager "$OUT/train_b1-eager_fa3_r1.log" env TORCH_COMPILE_DISABLE=1 NANOCHAT_ATTN_IMPL=fa3 \
    python -m scripts.base_train $B1 --num-iterations=110 --model-tag=c_gpu_eager

# CPU, fp32 eager, GPU hidden (so nanochat picks fp32 + SDPA and never touches CUDA)
CPU_ENV="env CUDA_VISIBLE_DEVICES= TORCH_COMPILE_DISABLE=1 NANOCHAT_DTYPE=float32"
run_step cpu_t72 "$OUT/cpu_t72.log" $CPU_ENV OMP_NUM_THREADS=72 \
    python -m scripts.base_train $B1 --device-type=cpu --num-iterations=6 --model-tag=c_cpu72
run_step cpu_t1 "$OUT/cpu_t1.log" $CPU_ENV OMP_NUM_THREADS=1 \
    python -m scripts.base_train $B1 --device-type=cpu --num-iterations=4 --model-tag=c_cpu1

# decode latency at batch 1 (needs checkpoint r12_conv from job B)
export NANOCHAT_BASE_DIR=$SHARED_BASE
run_step decode_gpu "$OUT/decode_gpu.log" env NANOCHAT_ATTN_IMPL=fa3 \
    python -m cisc662.bench.decode_bench --device cuda -g r12_conv --prompt-tokens 512 --decode-tokens 32 --batch-sizes 1
run_step decode_cpu_t72 "$OUT/decode_cpu_t72.log" $CPU_ENV OMP_NUM_THREADS=72 \
    python -m cisc662.bench.decode_bench --device cpu -g r12_conv --prompt-tokens 512 --decode-tokens 32 --batch-sizes 1
run_step decode_cpu_t1 "$OUT/decode_cpu_t1.log" $CPU_ENV OMP_NUM_THREADS=1 \
    python -m cisc662.bench.decode_bench --device cpu -g r12_conv --prompt-tokens 512 --decode-tokens 16 --batch-sizes 1

echo "FAILED STEPS:${FAILED_STEPS:- none}"
