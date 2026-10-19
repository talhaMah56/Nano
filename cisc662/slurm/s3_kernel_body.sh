# Shared body of the S3 kernel jobs (sourced by s3_kernels_kt1.sbatch / s3_kernels_kt2.sbatch after _common.sh).
# Expects: KID (kernel_id: rmsnorm | qkrope), TEST_FILE, SUITE (python module), EXTRA_TESTS (optional pytest paths).
# Order: self-test -> gates (pytest) -> ISA dump -> 3-trial benchmark (gates recorded only if pytest passed).
# Every step is clipped to a shared deadline (job start + 23 min) so the job ends COMPLETED inside its 25-min walltime.
JOBID=${SLURM_JOB_ID:-local}
OUT=$CISC662_REPO/cisc662/results/runs/s3_${KID}_${JOBID}
mkdir -p "$OUT"
DEADLINE=$(( JOB_T0 + 23 * 60 ))
left_s() { echo $(( DEADLINE - $(date +%s) )); }
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active \
    --format=csv -lms 500 > "$OUT/gpu_monitor.csv" &
MON_PID=$!
timeout 180 python -c "import nanochat.flash_attention as f; assert f.HAS_FA3, 'FA3 not available -> abort (would silently fall back to SDPA)'; print('HAS_FA3', f.HAS_FA3)"

run_step selftest "$OUT/selftest.log" timeout -k 15 120 python -m cisc662.kernels.triton.$KID
run_step gates "$OUT/gates.log" timeout -k 15 600 \
    python -m pytest "$TEST_FILE" $EXTRA_TESTS -q -rxs -s -p no:cacheprovider --junitxml="$OUT/gates_junit.xml"
tail -n 3 "$OUT/gates.log" || true
GATES_OK=0; [[ " $FAILED_STEPS " != *" gates "* ]] && GATES_OK=1
echo "=== gates (G1/G2) passed: $GATES_OK"
run_step isa "$OUT/isa.log" timeout -k 15 300 \
    python -m cisc662.kernels.triton.$KID --dump-isa "$CISC662_REPO/cisc662/results/isa/$KID"

TB=$(( $(left_s) - 90 ))
if [ "$TB" -gt 180 ]; then
    run_step bench "$OUT/bench.log" timeout -k 15 $(( TB + 75 )) \
        python -m $SUITE --trials 3 --out-dir "$CISC662_REPO/cisc662/results/db/$KID" --job-id "$JOBID" --time-budget-s "$TB"
else
    echo "skip bench: only ${TB}s left"; FAILED_STEPS="${FAILED_STEPS} bench_skipped"
fi
# G1/G2 are recorded on this job's "ours" rows only if the gate tests passed (PLAN 7.3: no timing row without gates)
if [ "$GATES_OK" = 1 ]; then
    for f in "$CISC662_REPO"/cisc662/results/db/$KID/by_job/$JOBID/*.json "$CISC662_REPO"/cisc662/results/db/$KID/*.json; do
        [ -f "$f" ] || continue
        python - "$f" "$JOBID" <<'EOF' && python -m cisc662.tests.gates "$f" G1 G2 > /dev/null || true
import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(0 if str(r.get("job_id")) == sys.argv[2] and r.get("form") not in ("eager", "compile") else 1)
EOF
    done
    echo "=== recorded G1,G2 on this job's kernel rows"
fi
kill $MON_PID 2>/dev/null || true
echo "FAILED STEPS:${FAILED_STEPS:- none}"
