#!/bin/bash
# Session 1, job C: Nsight Systems captures of 5 training steps (steps 20..24) of the R12 config, per variant.
# Gives the per-kernel time breakdown (which of the 5 target kernel classes dominate) for eager vs compile, FA3 vs SDPA.
# eager_fa3_ops additionally emits one NVTX range per autograd op (with shapes) to attribute eager kernels to ops.
#SBATCH --job-name=c662-s1-nsys
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out

source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh
OUT=$CISC662_REPO/cisc662/results/runs/s1_nsys_${SLURM_JOB_ID}
P=$CISC662_WORK/profiles/s1_nsys_${SLURM_JOB_ID}
mkdir -p "$OUT" "$P"
use_local_base_dir

R12="--depth=12 --max-seq-len=2048 --device-batch-size=16 --total-batch-size=32768 --window-pattern=L --core-metric-every=-1 --sample-every=-1 --eval-every=-1 --num-iterations=30"
NSYS="nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true"

capture() {  # capture <name> <compile|eager> <fa3|sdpa> [extra env...]
    local name=$1 mode=$2 attn=$3; shift 3
    local dis=0; [ "$mode" = eager ] && dis=1
    run_step "nsys_${name}" "$OUT/${name}.log" env TORCH_COMPILE_DISABLE=$dis NANOCHAT_ATTN_IMPL=$attn \
        NANOCHAT_PROFILE_STEPS=20:25 "$@" $NSYS -o "$P/${name}" python -m scripts.base_train $R12 --model-tag=p_${name}
    for rep in cuda_gpu_kern_sum nvtx_sum cuda_gpu_trace; do
        nsys stats --report $rep --format csv --output "$OUT/${name}" "$P/${name}.nsys-rep" > /dev/null 2>&1 || echo "stats $rep failed for $name"
    done
}

capture compile_fa3  compile fa3
capture eager_fa3    eager   fa3
capture compile_sdpa compile sdpa
capture eager_sdpa   eager   sdpa
capture eager_fa3_ops eager  fa3 NANOCHAT_EMIT_NVTX=1
nsys stats --report nvtx_kern_sum --format csv --output "$OUT/eager_fa3_ops" "$P/eager_fa3_ops.nsys-rep" > /dev/null 2>&1 || true

ls -la "$OUT" "$P"
echo "FAILED STEPS:${FAILED_STEPS:- none}"
