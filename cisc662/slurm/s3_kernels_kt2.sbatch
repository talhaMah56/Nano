#!/bin/bash
# Session 3, task KT-2: fused QK-norm + RoPE + 1.2 scale (fwd+bwd)
#   q and k in one launch; fp32 rotation, rstd from the rotated row (KT-2 design decision)
# S3 is split into one job per kernel (s3_kernels_kt1 / _kt2) instead of one s3_kernels.sbatch: the builders'
# estimates (KT-1 10-18 min, KT-2 ~10 min) would overrun one 30-min job, and two jobs fail independently.
# Steps and deadline logic: slurm/s3_kernel_body.sh. Time table (min): selftest 0.5 · gates 3-5 · ISA 0.5 · bench rest (<= 23).
#SBATCH --job-name=c662-s3-kt2
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:25:00
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out

source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh
KID=qkrope
TEST_FILE=cisc662/tests/test_qkrope.py
EXTRA_TESTS=""
SUITE=cisc662.bench.suite_qkrope
source "$CISC662_REPO/cisc662/slurm/s3_kernel_body.sh"
