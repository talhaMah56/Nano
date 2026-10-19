#!/bin/bash
# Session 2 follow-up to job 3406282 (s2_harness): re-measure with the fixed kbench and tests.
#   - kbench now queues a GPU spin before every start event, so host overhead (dynamo guards, AOTAutograd,
#     autograd.grad) no longer leaks into the CUDA-event times (3406282: compiled RMSNorm fwd+bwd 349 us vs
#     122 us eager; warm-L2 RMSNorm slower than cold) -> INF-1 smoke (now with an L2-resident flush proof)
#     and INF-3 ops_baseline (3 trials) again
#   - test_qkrope_g1[r12] now xfails (stock bf16 itself misses the rtol/atol rule) -> pytest full again, with
#     -rxs so every xfail/skip reason is in the log
# Time table (minutes, expected / cap): smoke 0.5/4 · pytest 0.5/5 · ops 3 trials (warm Inductor cache) 3/9.
#SBATCH --job-name=c662-s2-rerun
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
JOBID=${SLURM_JOB_ID:-local}
OUT=$CISC662_REPO/cisc662/results/runs/s2_rerun_${JOBID}
DB=$CISC662_REPO/cisc662/results/db
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active \
    --format=csv -lms 500 > "$OUT/gpu_monitor.csv" &
MON_PID=$!
timeout 180 python -c "import nanochat.flash_attention as f; assert f.HAS_FA3, 'FA3 not available -> abort (would silently fall back to SDPA)'; print('HAS_FA3', f.HAS_FA3)"

run_step kbench_smoke "$OUT/kbench_smoke.log" timeout -k 15 240 \
    python -m cisc662.bench.kbench --suite smoke --trials 3 --out-dir "$DB/smoke" --job-id "$JOBID"
grep -h "^ACCEPTANCE" "$OUT/kbench_smoke.log" || true

run_step pytest_full "$OUT/pytest_full.log" timeout -k 15 300 \
    python -m pytest cisc662/tests -q -rxs -p no:cacheprovider --junitxml="$OUT/pytest_junit.xml"
tail -n 3 "$OUT/pytest_full.log" || true

# ops: children start no case after minute 12.5 (TB); hard timeout at minute 13.5 (+15 s kill), walltime 15
TB=$(( JOB_T0 + 810 - $(date +%s) - 60 ))
if [ "$TB" -gt 120 ]; then
    run_step ops_trials3 "$OUT/ops_trials3.log" timeout -k 15 $(( TB + 60 )) \
        python -m cisc662.bench.ops_at_shapes --trials 3 --forms eager,compile --out-dir "$DB/ops_baseline" \
            --job-id "$JOBID" --time-budget-s "$TB"
else
    echo "skip ops_trials3: only ${TB}s of budget left"; FAILED_STEPS="${FAILED_STEPS} ops_trials3_skipped"
fi

kill $MON_PID 2>/dev/null || true
echo "FAILED STEPS:${FAILED_STEPS:- none}"
