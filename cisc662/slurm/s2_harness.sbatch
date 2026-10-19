#!/bin/bash
# Session 2: one GPU job that validates the measurement harness and records the platform facts.
#   INF-1 kbench smoke (3 trials)  ·  INF-2 pytest cisc662/tests (full, GPU)  ·  BAS-3 clock / gpu-freq / latency /
#   decode re-measurement  ·  INF-3 ops_at_shapes (eager + compile rows, dry run then 3 trials).
# Steps run most-important-first; every step is a run_step wrapped in `timeout`, clipped to a shared deadline
# (job start + 27 min - tail reserve), and skipped when too little time is left, so the job always ends
# COMPLETED before the 30-min walltime (slurm/queue.sh halts on any other end state).
#
# Time table (expected / cap, minutes; caps are clipped to the deadline):
#   0  preamble: FA3 assert, nvidia-smi dumps, monitor        0.5
#   1  kbench smoke, 3 trials -> results/db/smoke             1.5 / 4
#   2  pytest cisc662/tests -q (full) + junit xml             3   / 7
#   3  BAS-3 clock: 60-s bf16 8192^3 GEMM loop                1.5 / 2.5
#   4  BAS-3 gpu-freq step: srun --gpu-freq=1755 (smoke x1 + 20-s GEMM loop, sampled every 500 ms)
#                                                             1.5 / 4
#   5  BAS-3 latency: pointer chase 1 GiB (+64K/4M/32M/256M)  1   / 3
#   6  BAS-3 decode b=1, 256 tokens: CPU 1 thread x3 and GPU x3, interleaved
#                                                             3   / 6 x 2.5
#   7  ops_at_shapes --trials 1 (dry run: warms the Inductor cache on /work/nvme; results stay in $OUT)
#                                                             6   / budget 6 + slack 1.5
#   8  ops_at_shapes --trials 3 -> results/db/ops_baseline, --time-budget-s = what is left - 1.5 min
#                                                             9   (rest)
#   9  merge (platform json) + decode aggregate               0.3
#   total expected ~27 (step 8 absorbs the slack); trap + ledger after.  1 GPU x 0.5 h = 0.5 SU cap, ~0.47 expected.
#SBATCH --job-name=c662-s2-harness
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out

source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh
JOBID=${SLURM_JOB_ID:-local}
OUT=$CISC662_REPO/cisc662/results/runs/s2_harness_${JOBID}
DB=$CISC662_REPO/cisc662/results/db
mkdir -p "$OUT/decode" "$DB/platform" "$DB/decode"
touch "$OUT/.t0"
export OUT
SHARED_BASE=$CISC662_WORK/nanochat_base          # r12_conv checkpoint, read-only use (decode)

# ---------------------------------------------------------------- deadline / step helpers
DEADLINE=$(( JOB_T0 + 27 * 60 ))
TAIL_RESERVE=60                                  # merge + aggregate after the last step
SKIPPED_STEPS=""
left_s() { echo $(( DEADLINE - TAIL_RESERVE - $(date +%s) )); }

# step <name> <log> <cap_s> <min_s> <cmd...>
#   run_step with a timeout of min(cap, time left); skipped (not failed) if less than min_s is left.
#   Shell functions cannot run under `timeout`; they get the timeout in $STEP_T and apply it themselves.
STEP_T=0
step() {
    local name=$1 log=$2 cap=$3 min=$4; shift 4
    local t; t=$(left_s)
    if [ "$t" -gt "$cap" ]; then t=$cap; fi
    if [ "$t" -lt "$min" ]; then
        echo "--- [$(date +%H:%M:%S)] skip  ${name}: ${t}s left < ${min}s"
        SKIPPED_STEPS="${SKIPPED_STEPS} ${name}"
        return 0
    fi
    STEP_T=$t
    if declare -F "$1" > /dev/null; then
        run_step "$name" "$log" "$@"
    else
        run_step "$name" "$log" timeout -k 15 "$t" "$@"
    fi
}

# kb_step <name> <log> <budget_cap_s> <min_budget_s> <cmd...>: kbench-based suite with --time-budget-s.
#   Children skip cases that would start after the budget; a case already running finishes (seconds with a
#   warm cache). timeout = budget + KB_SLACK <= time left: if it fires, only the running trial is lost, since
#   kbench writes the aggregate after every completed trial (its own 180-s grace kill comes later than ours).
KB_SLACK=90
kb_step() {
    local name=$1 log=$2 cap=$3 min=$4; shift 4
    local tb; tb=$(( $(left_s) - KB_SLACK ))
    if [ "$tb" -gt "$cap" ]; then tb=$cap; fi
    if [ "$tb" -lt "$min" ]; then
        echo "--- [$(date +%H:%M:%S)] skip  ${name}: time budget ${tb}s < ${min}s"
        SKIPPED_STEPS="${SKIPPED_STEPS} ${name}"
        return 0
    fi
    run_step "$name" "$log" timeout -k 15 $(( tb + KB_SLACK )) "$@" --time-budget-s "$tb"
}

# nvidia-smi name of the clock-event bitmask (new drivers: clocks_event_reasons, old: clocks_throttle_reasons)
SMI_REASONS=""
for f in clocks_event_reasons.active clocks_throttle_reasons.active; do
    if timeout 30 nvidia-smi --query-gpu=$f --format=csv,noheader > /dev/null 2>&1; then SMI_REASONS=$f; break; fi
done
export SMI_REASONS
echo "=== clock-reason field: ${SMI_REASONS:-none accepted}"

# with_smi <csv> <cmd...>: per-rep nvidia-smi sampler (200 ms) around one command, timeout $STEP_T
with_smi() {
    local csv=$1; shift
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu --format=csv -lms 200 \
        > "$csv" 2> /dev/null &
    local smi=$! rc=0
    timeout -k 15 "$STEP_T" "$@" || rc=$?
    kill "$smi" 2> /dev/null || true
    return $rc
}

# gpufreq_step: BAS-3 step 2, a job STEP with --gpu-freq=1755. Slurm sets the clock for the step only and
# resets it afterwards. Inside: 500-ms clock sampler + kbench smoke (1 trial) + a 20-s GEMM loop (the smoke
# suite alone keeps the GPU busy for well under a second, too short to judge the clock). stderr kept.
gpufreq_step() {
    local rc=0
    timeout -k 20 "$STEP_T" srun --overlap --ntasks=1 --nodes=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
        --gpu-freq=1755 bash -c '
            nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,power.draw${SMI_REASONS:+,$SMI_REASONS} \
                --format=csv -lms 500 > "$OUT/gpufreq_probe.csv" &
            SMI=$!
            nvidia-smi --query-gpu=clocks.applications.graphics,clocks.sm,clocks.max.sm --format=csv || true
            python -m cisc662.bench.kbench --suite smoke --trials 1 --out-dir "$OUT/gpufreq_smoke"; rc1=$?
            python -m cisc662.bench.platform_probe clock --seconds 20 --interval-s 1 --out "$OUT/clock_pinned.json"; rc2=$?
            kill $SMI 2> /dev/null
            exit $(( rc1 != 0 || rc2 != 0 ))' \
        2> "$OUT/gpufreq_step.stderr" || rc=$?
    echo "$rc" > "$OUT/gpufreq_step.rc"
    return $rc
}

# ---------------------------------------------------------------- 0 preamble
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used${SMI_REASONS:+,$SMI_REASONS} \
    --format=csv -lms 500 > "$OUT/gpu_monitor.csv" &
MON_PID=$!
# (unguarded steps get a timeout too, so nothing before step 1 can hang the job; a failed assert aborts it)
timeout 180 python -c "import nanochat.flash_attention as f; assert f.HAS_FA3, 'FA3 not available -> abort (would silently fall back to SDPA)'; print('HAS_FA3', f.HAS_FA3)"
timeout 30 nvidia-smi -q > "$OUT/nvidia_smi_q_start.txt" 2>&1 || true
timeout 30 nvidia-smi -q -d SUPPORTED_CLOCKS > "$OUT/supported_clocks.txt" 2>&1 || true
NODE=$(hostname -s)

# ---------------------------------------------------------------- 1 INF-1 kbench smoke
step kbench_smoke "$OUT/kbench_smoke.log" 240 60 \
    python -m cisc662.bench.kbench --suite smoke --trials 3 --out-dir "$DB/smoke" --job-id "$JOBID"
grep -h "^ACCEPTANCE" "$OUT/kbench_smoke.log" 2> /dev/null || true

# ---------------------------------------------------------------- 2 INF-2 tests (full, incl. real_shape + gpu)
step pytest_full "$OUT/pytest_full.log" 420 60 \
    python -m pytest cisc662/tests -q -p no:cacheprovider --junitxml="$OUT/pytest_junit.xml"
tail -n 3 "$OUT/pytest_full.log" 2> /dev/null || true

# ---------------------------------------------------------------- 3 BAS-3 sustained clock
step clock "$OUT/clock.log" 150 90 \
    python -m cisc662.bench.platform_probe clock --seconds 60 --out "$OUT/clock.json"

# ---------------------------------------------------------------- 4 BAS-3 gpu-freq job step
step gpufreq "$OUT/gpufreq_step.log" 240 90 gpufreq_step
timeout 30 nvidia-smi --query-gpu=clocks.applications.graphics,clocks.sm --format=csv,noheader 2>&1 \
    | sed 's/^/=== after gpu-freq step (app clock, sm clock): /' || true

# ---------------------------------------------------------------- 5 BAS-3 latency
step latency "$OUT/latency.log" 180 60 \
    python -m cisc662.bench.platform_probe latency --bytes 1073741824 \
        --extra-bytes 65536,4194304,33554432,268435456 --out "$OUT/latency.json"

# ---------------------------------------------------------------- 6 BAS-3 decode re-measurement (b=1, 256 tokens)
# CPU: 1 thread pinned to the first core of this job's cpuset (numactl as PLAN.md 7.5; taskset fallback);
# fp32 eager, GPU hidden -- the same env as S1 (slurm/s1_cpu_ref.sbatch). GPU: FA3, stock decode path.
CPU1=$(python -c "import os; print(min(os.sched_getaffinity(0)))" 2> /dev/null || echo 0)
if numactl --physcpubind="$CPU1" --localalloc true > /dev/null 2>&1; then
    CPU_BIND="numactl --physcpubind=$CPU1 --localalloc"
elif taskset -c "$CPU1" true > /dev/null 2>&1; then
    CPU_BIND="taskset -c $CPU1"
else
    CPU_BIND=""
fi
echo "=== decode CPU binding: ${CPU_BIND:-none}"
CPU_ENV="env CUDA_VISIBLE_DEVICES= TORCH_COMPILE_DISABLE=1 NANOCHAT_DTYPE=float32 OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR=$SHARED_BASE"
DEC="-m cisc662.bench.decode_bench -g r12_conv --prompt-tokens 512 --decode-tokens 256 --batch-sizes 1"
for rep in 1 2 3; do
    step "decode_cpu_r${rep}" "$OUT/decode/cpu_r${rep}.log" 150 60 \
        $CPU_ENV $CPU_BIND python $DEC --device cpu
    step "decode_gpu_r${rep}" "$OUT/decode/gpu_r${rep}.log" 120 45 \
        with_smi "$OUT/decode/gpu_r${rep}.smi.csv" \
        env NANOCHAT_ATTN_IMPL=fa3 NANOCHAT_BASE_DIR="$SHARED_BASE" python $DEC --device cuda
done

# ---------------------------------------------------------------- 7 INF-3 dry run (1 trial; warms the cache)
kb_step ops_dryrun "$OUT/ops_dryrun.log" 360 120 \
    python -m cisc662.bench.ops_at_shapes --trials 1 --forms eager,compile --out-dir "$OUT/ops_dryrun" --job-id "$JOBID"

# ---------------------------------------------------------------- 8 INF-3 3 trials -> results/db/ops_baseline
kb_step ops_trials3 "$OUT/ops_trials3.log" 1200 120 \
    python -m cisc662.bench.ops_at_shapes --trials 3 --forms eager,compile --out-dir "$DB/ops_baseline" --job-id "$JOBID"

kill $MON_PID 2> /dev/null || true

# ---------------------------------------------------------------- 9 merge / aggregate (CPU only, seconds)
GF_RC=$(cat "$OUT/gpufreq_step.rc" 2> /dev/null || echo 255)
run_step merge_platform "$OUT/merge_platform.log" timeout 60 \
    python -m cisc662.bench.platform_probe merge --job-id "$JOBID" \
        --clock "$OUT/clock.json" --latency "$OUT/latency.json" \
        --gpufreq-csv "$OUT/gpufreq_probe.csv" --gpufreq-stderr "$OUT/gpufreq_step.stderr" --gpufreq-rc "$GF_RC" \
        --pinned-clock "$OUT/clock_pinned.json" \
        --smoke-default-dir "$DB/smoke/by_job/$JOBID" --smoke-pinned-dir "$OUT/gpufreq_smoke/by_job/$JOBID" \
        --out "$DB/platform/clocks_${JOBID}.json"
cat "$OUT/merge_platform.log" 2> /dev/null || true
shopt -s nullglob
run_step decode_aggregate "$OUT/decode_aggregate.log" timeout 60 \
    python -m cisc662.analysis.decode_remeasure --job-id "$JOBID" --node "$NODE" --cpu-bind "${CPU_BIND:-none}" \
        --cpu-logs "$OUT"/decode/cpu_r*.log --gpu-logs "$OUT"/decode/gpu_r*.log \
        --gpu-clock-csvs "$OUT"/decode/gpu_r*.smi.csv \
        --platform-json "$DB/platform/clocks_${JOBID}.json" --out "$DB/decode/s2_remeasure.json"
shopt -u nullglob
cat "$OUT/decode_aggregate.log" 2> /dev/null || true
# decode_remeasure also writes s2_remeasure_<jobid>.json in results/db/decode; keep a copy next to the run too
cp "$DB/decode/s2_remeasure.json" "$OUT/decode/s2_remeasure_${JOBID}.json" 2> /dev/null || true

echo "=== outputs of this job:"
find "$OUT" "$DB" -newer "$OUT/.t0" -type f 2> /dev/null | sed "s|^$CISC662_REPO/||" | sort | head -200 || true
rm -f "$OUT/.t0"
echo "SKIPPED STEPS:${SKIPPED_STEPS:- none}"
echo "FAILED STEPS:${FAILED_STEPS:- none}"
