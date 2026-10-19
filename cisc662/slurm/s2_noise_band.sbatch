#!/bin/bash
# Session 2, task BAS-2: loss-curve noise envelope for gate G4.
# Two stock R12-L runs (1000 steps, same seed and data order) -> the run-to-run spread that every integrated
# kernel's loss curve is compared against.
# Analysis (login node, after the job):
#   python -m cisc662.analysis.losscurve <OUT>/noise_a.log <OUT>/noise_b.log --make-envelope
#SBATCH --job-name=c662-s2-noise
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:15:00
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out

source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh
OUT=$CISC662_REPO/cisc662/results/runs/s2_noise_${SLURM_JOB_ID}
mkdir -p "$OUT"
use_local_base_dir   # throwaway checkpoints -> node-local /tmp
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
    --format=csv -lms 500 > "$OUT/gpu_monitor.csv" &
MON_PID=$!
python -c "import nanochat.flash_attention as f; assert f.HAS_FA3, 'FA3 not available -> abort (would silently fall back to SDPA)'; print('HAS_FA3', f.HAS_FA3)"
# R12-L (PLAN.md section 3): R12 + 1000 iterations, eval every 250 steps on 2,097,152 tokens
R12L="--depth=12 --max-seq-len=2048 --device-batch-size=16 --total-batch-size=32768 --window-pattern=L --core-metric-every=-1 --sample-every=-1 --num-iterations=1000 --eval-every=250 --eval-tokens=2097152"

for tag in noise_a noise_b; do
    run_step "$tag" "$OUT/$tag.log" timeout 360 python -m scripts.base_train $R12L --model-tag=$tag
done

kill $MON_PID 2>/dev/null || true
grep -h "Validation bpb" "$OUT"/noise_*.log || true
echo "FAILED STEPS:${FAILED_STEPS:- none}"
