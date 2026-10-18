#!/bin/bash
# Session 1, job B1: convergence run (loss-curve baseline + checkpoint r12_conv) and the inference baselines.
# Reduced config R12: depth 12 (768d, 6x128 heads), seq 2048, 16x2048 = 32768 tok/step, no grad accum, window L, bf16.
#SBATCH --job-name=c662-s1-convinf
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
OUT=$CISC662_REPO/cisc662/results/runs/s1_convinf_${SLURM_JOB_ID}
P=$CISC662_WORK/profiles
mkdir -p "$OUT" "$P"
export NANOCHAT_BASE_DIR=$CISC662_WORK/nanochat_base
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used \
    --format=csv -lms 500 > "$OUT/gpu_monitor.csv" &
MON_PID=$!
R12="--depth=12 --max-seq-len=2048 --device-batch-size=16 --total-batch-size=32768 --window-pattern=L --core-metric-every=-1 --sample-every=-1"

# 1) convergence / loss-curve baseline (stock: torch.compile + FA3); checkpoint kept on /work/nvme
run_step conv "$OUT/conv_compile_fa3.log" env NANOCHAT_ATTN_IMPL=fa3 \
    python -m scripts.base_train $R12 --num-iterations=2000 --eval-every=250 --eval-tokens=2097152 --model-tag=r12_conv

# 2) inference benchmark, fixed workload: prompt 512, decode 256, batch 1/8/32/128
for rep in 1 2; do
    for attn in fa3 sdpa; do
        run_step "infer_${attn}_r${rep}" "$OUT/infer_${attn}_r${rep}.log" env NANOCHAT_ATTN_IMPL=$attn \
            python -m scripts.infer_bench -i base -g r12_conv --prompt-tokens 512 --decode-tokens 256 --batch-sizes 1,8,32,128
    done
done

# 3) nsys capture of the decode workload (64 decode tokens, batch 1 and 128)
run_step nsys_infer "$OUT/nsys_infer_fa3.log" env NANOCHAT_ATTN_IMPL=fa3 \
    nsys profile -t cuda,nvtx --force-overwrite true -o "$P/s1_infer_fa3" \
    python -m scripts.infer_bench -i base -g r12_conv --prompt-tokens 512 --decode-tokens 64 --batch-sizes 1,128
nsys stats --report cuda_gpu_kern_sum --format csv --output "$OUT/nsys_infer_fa3" "$P/s1_infer_fa3.nsys-rep" > /dev/null 2>&1 || true

kill $MON_PID 2>/dev/null || true
python -m cisc662.analysis.summarize_s1 "$OUT" || true
echo "FAILED STEPS:${FAILED_STEPS:- none}"
