# Source this in every shell / Slurm job:  source /u/tmahmood1/project/nanochat/cisc662/env/env.sh
# Keeps all heavy I/O (venv, caches, data, checkpoints, profiles) on /work/nvme, code in $HOME.
# /projects/bffp and /work/hdd/bffp are over their soft quotas -- do not write there.

export CISC662_REPO=/u/tmahmood1/project/nanochat
export CISC662_WORK=/work/nvme/bffp/tmahmood1/cisc662

# Slurm account for DeltaAI (shared class allocation -- be frugal, see cisc662/docs/PLAN.md)
export CISC662_ACCOUNT=bffp-dtai-gh

# uv-managed Python env lives on /work/nvme (fast, keeps $HOME light)
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=$CISC662_WORK/venv
export UV_CACHE_DIR=$CISC662_WORK/cache/uv
export UV_PYTHON_INSTALL_DIR=$CISC662_WORK/cache/uv-python
export PIP_CACHE_DIR=$CISC662_WORK/cache/pip

# nanochat data / tokenizer / checkpoints
export NANOCHAT_BASE_DIR=$CISC662_WORK/nanochat_base

# Compiler caches persist across jobs (saves GPU time re-compiling)
export TRITON_CACHE_DIR=$CISC662_WORK/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$CISC662_WORK/cache/inductor
export HF_HOME=$CISC662_WORK/cache/hf
export TIKTOKEN_CACHE_DIR=$CISC662_WORK/cache/tiktoken
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0   # no xet chunk cache (ADM-2): the hub blobs are the only copy we need

# Same as nanochat's speedrun: torch threads are set explicitly where it matters
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# CUDA toolchain from the NVIDIA HPC SDK already on PATH (nvcc 12.9, nsys, ncu, cuobjdump, nvdisasm)
export CUDA_HOME=${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9}
# Login shells export the Cray wrappers (CC=cc, CXX=CC); keep them out of Triton launcher / Inductor C++ builds.
export CC=gcc CXX=g++
# Login node cgroup: 16 cores / 20 GB / 1000 tasks per user -- cap parallel builds.
export MAX_JOBS=8 TORCHINDUCTOR_COMPILE_THREADS=8

if [ -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]; then
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
fi
cd "$CISC662_REPO"
