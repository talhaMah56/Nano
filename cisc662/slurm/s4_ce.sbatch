#!/bin/bash
# Session 4, task KT-3: fused softcap + cross-entropy (kernel_id ce) -- gates, ISA, peak memory, benchmark.
# Same structure as the S3 kernel jobs (slurm/s3_kernel_body.sh): self-test -> gates (pytest) -> ISA dump ->
# peak memory -> 3-trial benchmark, each step clipped to one deadline (job start + 23 min).
# Differences: gates are recorded on the ours_* rows only, never on the ablation_* rows (tanh.approx fails G1
# by design); an extra peak-memory step. Time table (min): selftest 0.5 · gates 5-8 (fp64 references at R12)
# · ISA 1 · peak-mem 1.5 · bench rest (>= 9; the suite drops cases it cannot finish, see --time-budget-s).
#SBATCH --job-name=c662-s4-kt3
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=00:25:00
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out

source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh
KID=ce
JOBID=${SLURM_JOB_ID:-local}
OUT=$CISC662_REPO/cisc662/results/runs/s4_${KID}_${JOBID}
DB=$CISC662_REPO/cisc662/results/db/$KID
mkdir -p "$OUT" "$DB"
DEADLINE=$(( JOB_T0 + 23 * 60 ))
left_s() { echo $(( DEADLINE - $(date +%s) )); }
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active \
    --format=csv -lms 500 > "$OUT/gpu_monitor.csv" &
MON_PID=$!

run_step selftest "$OUT/selftest.log" timeout -k 15 180 python -m cisc662.kernels.triton.ce
run_step gates "$OUT/gates.log" timeout -k 15 600 \
    python -m pytest cisc662/tests/test_ce.py -q -rxs -s -p no:cacheprovider --junitxml="$OUT/gates_junit.xml"
tail -n 3 "$OUT/gates.log" || true
GATES_OK=0; [[ " $FAILED_STEPS " != *" gates "* ]] && GATES_OK=1
echo "=== gates (G1/G2) passed: $GATES_OK"
run_step isa "$OUT/isa.log" timeout -k 15 240 \
    python -m cisc662.kernels.triton.ce --dump-isa "$CISC662_REPO/cisc662/results/isa/$KID"
run_step peakmem "$OUT/peakmem.log" timeout -k 15 240 \
    python -m cisc662.bench.suite_ce --peak-mem --out-dir "$DB" --job-id "$JOBID"

TB=$(( $(left_s) - 90 ))
if [ "$TB" -gt 180 ]; then
    run_step bench "$OUT/bench.log" timeout -k 15 $(( TB + 75 )) \
        python -m cisc662.bench.suite_ce --trials 3 --out-dir "$DB" --job-id "$JOBID" --time-budget-s "$TB"
    grep '^SUMMARY' "$OUT/bench.log" || true
else
    echo "skip bench: only ${TB}s left"; FAILED_STEPS="${FAILED_STEPS} bench_skipped"
fi
# G1/G2 go on this job's ours_* rows only if the gate tests passed (PLAN 7.3); never on eager/compile/ablation rows
if [ "$GATES_OK" = 1 ]; then
    for f in "$DB"/by_job/$JOBID/*.json "$DB"/*.json; do
        [ -f "$f" ] || continue
        python - "$f" "$JOBID" <<'EOF' && python -m cisc662.tests.gates "$f" G1 G2 > /dev/null || true
import json, sys
r = json.load(open(sys.argv[1]))
form = str(r.get("form", ""))
sys.exit(0 if str(r.get("job_id")) == sys.argv[2] and form not in ("eager", "compile")
         and not form.startswith("ablation") else 1)
EOF
    done
    echo "=== recorded G1,G2 on this job's ours_* rows (not on ablation rows)"
fi
kill $MON_PID 2>/dev/null || true
echo "FAILED STEPS:${FAILED_STEPS:- none}"
