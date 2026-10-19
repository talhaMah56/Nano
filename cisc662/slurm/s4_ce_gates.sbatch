#!/bin/bash
# Session 4, task KT-3 (gates re-run): job 3417307 timed the ce kernels and passed all 50 gated tests, but its
# pytest run exited 1 on one non-gate test (the tanh.approx ablation expectation, which was wrong -- fixed in
# test_ce.py), so it correctly recorded no G1/G2. The kernel code is unchanged since 3417307 (only the module
# docstring differs; checked against the queue's archive of that job). This job re-runs test_ce.py and, only if
# it passes completely, records G1/G2 on 3417307's ours_* rows (never on eager/compile/ablation rows), adding
# a gates_evidence note that names this job. Time: < 3 min of GPU.
#SBATCH --job-name=c662-s4-kt3gates
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=00:10:00
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out

source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh
TIMED_JOB=3417307
JOBID=${SLURM_JOB_ID:-local}
OUT=$CISC662_REPO/cisc662/results/runs/s4_ce_gates_${JOBID}
DB=$CISC662_REPO/cisc662/results/db/ce
mkdir -p "$OUT"
md5sum cisc662/kernels/triton/ce.py cisc662/tests/test_ce.py | tee "$OUT/sources.md5"
run_step gates "$OUT/gates.log" timeout -k 15 480 \
    python -m pytest cisc662/tests/test_ce.py -q -rxs -s -p no:cacheprovider --junitxml="$OUT/gates_junit.xml"
tail -n 3 "$OUT/gates.log" || true
if [[ " $FAILED_STEPS " != *" gates "* ]]; then
    n=0
    for f in "$DB"/*.json; do
        python - "$f" "$TIMED_JOB" "$JOBID" <<'EOF' || continue
import json, os, sys, tempfile
path, timed, gates_job = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(path))
form = str(r.get("form", ""))
if str(r.get("job_id")) != timed or not form.startswith("ours"):
    sys.exit(1)
ev = r.get("gates_evidence") or {}
ev.update({g: f"test_ce.py passed in job {gates_job} (kernel code identical to timed job {timed})" for g in ("G1", "G2")})
r["gates_evidence"] = ev
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".json")
with os.fdopen(fd, "w") as fh:
    json.dump(r, fh, indent=2)
    fh.write("\n")
os.replace(tmp, path)
EOF
        python -m cisc662.tests.gates "$f" G1 G2 && n=$((n + 1))
    done
    echo "=== recorded G1,G2 on $n ours_* rows of job $TIMED_JOB"
    [ "$n" -gt 0 ] || { echo "no rows of job $TIMED_JOB found"; FAILED_STEPS="${FAILED_STEPS} no_rows"; }
fi
echo "FAILED STEPS:${FAILED_STEPS:- none}"
