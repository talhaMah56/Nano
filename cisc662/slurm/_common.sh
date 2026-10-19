# Sourced at the top of every cisc662 sbatch script (after the #SBATCH lines).
# - loads the project environment
# - prints job/GPU/git metadata into the job log
# - appends one line per job to $CISC662_WORK/logs/job_ledger.csv (GPU-hour budget tracking)
source /u/tmahmood1/project/nanochat/cisc662/env/env.sh
set -eo pipefail
# Jobs never touch the network: FA3 and other hub files were pre-fetched into $HF_HOME on the login node.
export HF_HUB_OFFLINE=1

JOB_T0=$(date +%s)
echo "=== job ${SLURM_JOB_NAME:-local} id=${SLURM_JOB_ID:-na} host=$(hostname) start=$(date -Is)"
echo "=== gpus=${SLURM_GPUS_ON_NODE:-0} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}MB"
echo "=== git $(git -C "$CISC662_REPO" rev-parse --short HEAD) on $(git -C "$CISC662_REPO" rev-parse --abbrev-ref HEAD), $(git -C "$CISC662_REPO" status --porcelain | wc -l) uncommitted paths"
nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv || true

_cisc662_on_exit() {
    local rc=$?
    local el=$(( $(date +%s) - JOB_T0 ))
    local ngpu=${SLURM_GPUS_ON_NODE:-0}
    # SU from Slurm's billing weight (MAX_TRES: max(GPU*1000, cores*13.889, GiB*9.017)); fallback recomputes it.
    local bill
    bill=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'billing=[0-9]*' | head -1 | cut -d= -f2)
    if [ -z "$bill" ]; then
        local cpus=${SLURM_CPUS_PER_TASK:-1} memmb=${SLURM_MEM_PER_NODE:-0}
        bill=$(awk "BEGIN{g=${ngpu}*1000; c=${cpus}*13.888888; m=${memmb}/1024*9.01726; b=g; if(c>b)b=c; if(m>b)b=m; printf \"%d\", b}")
    fi
    local su
    su=$(awk "BEGIN{printf \"%.3f\", ${bill}*${el}/3600000}")
    # columns: jobid,name,end,elapsed_s,ngpu,rc,su
    echo "${SLURM_JOB_ID:-na},${SLURM_JOB_NAME:-local},$(date -Is),${el},${ngpu},${rc},${su}" >> "$CISC662_WORK/logs/job_ledger.csv"
    echo "=== end rc=${rc} elapsed=${el}s billing=${bill} su=${su}"
}
trap _cisc662_on_exit EXIT

# run_step <name> <logfile> <cmd...>: runs one experiment step, logs its wall time, and keeps the job
# going if it fails (a failed step is recorded in the job log and in $FAILED_STEPS).
FAILED_STEPS=""
run_step() {
    local name=$1 log=$2; shift 2
    local t0=$(date +%s)
    echo ">>> [$(date +%H:%M:%S)] start ${name} -> ${log}"
    if "$@" > "$log" 2>&1; then
        echo "<<< [$(date +%H:%M:%S)] ok    ${name} ($(( $(date +%s) - t0 ))s)"
    else
        local rc=$?   # must be read first: the $(date ...) below would reset $?
        echo "!!! [$(date +%H:%M:%S)] FAIL  ${name} rc=${rc} ($(( $(date +%s) - t0 ))s); tail of log:"
        tail -n 15 "$log" | sed 's/^/    /'
        FAILED_STEPS="${FAILED_STEPS} ${name}"
    fi
}

# Node-local scratch for throwaway outputs (e.g. checkpoints written by timing runs); purged by Slurm after the job.
# Usage: use_local_base_dir  -> NANOCHAT_BASE_DIR on /tmp with symlinks to the shared data + tokenizer.
use_local_base_dir() {
    local shared=$CISC662_WORK/nanochat_base
    export NANOCHAT_BASE_DIR=/tmp/nanochat_${SLURM_JOB_ID:-$$}
    mkdir -p "$NANOCHAT_BASE_DIR"
    for d in base_data_climbmix tokenizer; do
        [ -e "$shared/$d" ] && ln -sfn "$shared/$d" "$NANOCHAT_BASE_DIR/$d"
    done
    echo "=== NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR (node-local; data/tokenizer symlinked from $shared)"
}
