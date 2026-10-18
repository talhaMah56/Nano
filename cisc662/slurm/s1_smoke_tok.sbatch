#!/bin/bash
# Session 1, job A: toolchain smoke test + roofline ceilings + profiler permission checks, then tokenizer training.
# One GH200 for ~15 min. The tokenizer step is CPU-only but shares the allocation to avoid a second queue wait.
#SBATCH --job-name=c662-smoke-tok
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=00:30:00
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out

source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh
R=$CISC662_REPO/cisc662/results
P=$CISC662_WORK/profiles
mkdir -p "$R" "$P"

echo "### outbound internet from compute node?"
curl -sS -m 10 -o /dev/null -w "huggingface.co HTTP %{http_code}\n" https://huggingface.co || echo "no outbound internet"

echo "### 1) smoke test + machine characterization"
python -m cisc662.bench.smoke --out "$R/smoke_gh200.json"

echo "### 2a) Nsight Systems check"
nsys --version
nsys profile -t cuda,nvtx --force-overwrite true -o "$P/smoke_nsys" python -m cisc662.bench.tiny_target \
    && nsys stats --report cuda_gpu_kern_sum --format csv "$P/smoke_nsys.nsys-rep" | head -15 \
    || echo "NSYS CHECK FAILED"

echo "### 2b) Nsight Compute check (needs GPU performance-counter permission)"
ncu --version
ncu --kernel-name regex:_add_kernel --launch-count 1 \
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    python -m cisc662.bench.tiny_target 2>&1 | tail -20 || echo "NCU CHECK FAILED"

echo "### 3) tokenizer training (rustbpe, CPU) + eval"
python -m scripts.tok_train
python -m scripts.tok_eval || echo "tok_eval failed (non-critical)"
ls -la "$NANOCHAT_BASE_DIR/tokenizer"
