#!/bin/bash
# Bounded view of this project's Slurm jobs and GPU-hour spend:  bash cisc662/slurm/monitor.sh [interval_s] [max_hours]
# Polls every 300 s by default and EXITS after 6 h, so it can never approach NCSA's 30-CPU-minute login-node limit.
# Jobs are sbatch jobs and do not depend on this script; `squeue -u $USER` on demand is just as good.
INTERVAL=${1:-300}
MAX_HOURS=${2:-6}
LEDGER=/work/nvme/bffp/tmahmood1/cisc662/logs/job_ledger.csv
END=$(( $(date +%s) + MAX_HOURS * 3600 ))
while [ "$(date +%s)" -lt "$END" ]; do
    clear
    date
    echo "--- my jobs ---"
    squeue -u "$USER" -o "%.10i %.22j %.9P %.8T %.10M %.10l %R"
    echo "--- project SU so far (job ledger; jobcharge -u $USER -d 30 is authoritative) ---"
    if [ -f "$LEDGER" ]; then
        awk -F, '{ if (NF >= 7) su += $7; else su += $4*$5/3600; n++ } END {printf "%d jobs, %.2f SU\n", n, su}' "$LEDGER"
        tail -n 5 "$LEDGER"
    fi
    echo "--- allocation balance ---"
    accounts 2>/dev/null | grep -E "bffp|Account"
    echo "(refresh every ${INTERVAL}s; exits automatically after ${MAX_HOURS} h)"
    sleep "$INTERVAL"
done
echo "monitor.sh: reached ${MAX_HOURS} h, exiting (restart if still needed)"
