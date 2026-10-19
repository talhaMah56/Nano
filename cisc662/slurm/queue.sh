#!/bin/bash
# Budget-guarded, bounded job queue for this project. Run it inside tmux on gh-login01, e.g.
#   tmux new-window -t cisc662 -n queue "bash cisc662/slurm/queue.sh cisc662/slurm/queue_s2.txt 1.5"
# Queue file: one entry per line, processed strictly in order:
#   <sbatch path relative to the repo> [TASK-IDS]      e.g.  cisc662/slurm/s2_noise_band.sbatch BAS-2
#   WAIT <reason>      blocks the queue at that point; release it by editing the line to '#WAIT ...'
#   # ...              comment
# Only APPEND entries or comment them out in place -- progress is tracked by line number in <queue>.state.
# Guarantees (AGENT_GUIDE.md section 3, PLAN.md section 6):
#   - at most MAX_INFLIGHT=2 project jobs (names c662-*) pending/running at once
#   - before each submission: session SU (finished jobs: charged SU; unfinished: their full walltime) + the new
#     job's full walltime must be <= SESSION_CAP, and the project total (ledger + in-flight walltime + new job)
#     must stay below PROJECT_STOP=8 SU (the pre-P1 re-plan checkpoint, which needs the user)
#   - bash -n and sbatch --test-only before every real submission
#   - halts (no further submissions) as soon as a job it submitted ends in any state other than COMPLETED
#   - every submission/completion goes to cisc662/results/runlog.csv + cisc662/results/RUNLOG.md; the exact
#     script, the uncommitted git diff and the untracked cisc662 sources are archived in $CISC662_WORK/logs/submitted/
#   - exits when the queue is drained and its jobs have finished, or after MAX_HOURS (bounded: NCSA kills
#     login-node processes beyond 30 CPU-minutes; this loop uses a few CPU-seconds per hour)
set -uo pipefail
source /u/tmahmood1/project/nanochat/cisc662/env/env.sh > /dev/null
QFILE=$(readlink -f "$1")
SESSION_CAP=${2:-1.5}
MAX_HOURS=${3:-12}
POLL=${POLL:-120}
MAX_INFLIGHT=2
PROJECT_STOP=8.0
STATE=$QFILE.state            # "<lineno> <jobid> <script>" per submitted entry
HALT=$QFILE.halt              # written when the queue stops on an error; delete it to allow a restart
RUNLOG_CSV=$CISC662_REPO/cisc662/results/runlog.csv
RUNLOG_MD=$CISC662_REPO/cisc662/results/RUNLOG.md
LEDGER=$CISC662_WORK/logs/job_ledger.csv
ARCH=$CISC662_WORK/logs/submitted
mkdir -p "$ARCH"; touch "$STATE"
[ -f "$RUNLOG_CSV" ] || echo "time,event,jobid,job_name,tasks,script,git_sha,git_dirty,state,elapsed_s,su,log,archive" > "$RUNLOG_CSV"
END=$(( $(date +%s) + MAX_HOURS * 3600 ))

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# billing weight (Slurm MAX_TRES) and walltime (s) declared by an sbatch script
script_billing() {
    awk '/^#SBATCH/ {
            if (match($0, /--gpus-per-node=[0-9]+/)) g = substr($0, RSTART+16, RLENGTH-16)
            if (match($0, /--cpus-per-task=[0-9]+/)) c = substr($0, RSTART+16, RLENGTH-16)
            if (match($0, /--mem=[0-9]+[gG]/))        m = substr($0, RSTART+6, RLENGTH-7)
         }
         END { b = g*1000; if (c*13.889 > b) b = c*13.889; if (m*9.017 > b) b = m*9.017; printf "%d", b }' "$1"
}
script_walltime_s() {
    local t; t=$(grep -oE '^#SBATCH --time=[0-9:]+' "$1" | head -1 | cut -d= -f2)
    awk -v t="$t" 'BEGIN { n = split(t, a, ":"); s = 0; for (i = 1; i <= n; i++) s = s*60 + a[i];
                           if (n == 2) s *= 60; if (n == 1) s *= 60; print s }'
}

# "<state> <su>" for one job: charged SU if finished, full-walltime SU (upper bound) if pending/running
job_cost() {
    local line st el tl tres bill
    line=$(sacct -X -n -P -j "$1" -o State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES 2>/dev/null | head -1)
    st=$(echo "$line" | cut -d'|' -f1 | awk '{print $1}'); el=$(echo "$line" | cut -d'|' -f2)
    tl=$(echo "$line" | cut -d'|' -f3); tres=$(echo "$line" | cut -d'|' -f4,5)
    bill=$(echo "$tres" | grep -oE 'billing=[0-9]+' | head -1 | cut -d= -f2); bill=${bill:-1000}
    [ -z "$st" ] && st=PENDING
    case "$st" in
        PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED)
            awk -v b="$bill" -v t="${tl:-30}" -v s="$st" 'BEGIN { printf "%s %.3f\n", s, b*t*60/3600000 }' ;;
        *)  awk -v b="$bill" -v e="${el:-0}" -v s="$st" 'BEGIN { printf "%s %.3f\n", s, b*e/3600000 }' ;;
    esac
}

ledger_su() {   # project SU recorded by _common.sh (S1 lines predate the SU column: elapsed * ngpu)
    [ -f "$LEDGER" ] || { echo 0; return; }
    awk -F, '{ if (NF >= 7) su += $7; else su += $4*$5/3600 } END { printf "%.3f", su }' "$LEDGER"
}

record_end() {  # jobid script state su
    local jobid=$1 script=$2 st=$3 su=$4 name el log fs
    name=$(sacct -X -n -P -j "$jobid" -o JobName | head -1); el=$(sacct -X -n -P -j "$jobid" -o ElapsedRaw | head -1)
    log=$CISC662_WORK/logs/${name}-${jobid}.out
    fs=$(grep -h "^FAILED STEPS:" "$log" 2>/dev/null | tail -1); fs=${fs:-"FAILED STEPS: (line missing)"}
    echo "$(date -Is),end,$jobid,$name,,$script,,,$st,$el,$su,$log," >> "$RUNLOG_CSV"
    echo "- $(date '+%Y-%m-%d %H:%M') **END** \`$jobid\` $name -- **$st**, ${el}s, $su SU, ${fs} -- log \`$log\`" >> "$RUNLOG_MD"
    say "job $jobid ($name) ended: $st, ${el}s, $su SU; $fs"
}

submit() {  # lineno script tasks
    local lineno=$1 script=$2 tasks=$3 jobid sha dirty base tag
    bash -n "$script" || { say "bash -n failed for $script"; return 1; }
    sbatch --test-only "$script" > /dev/null 2>&1 || { say "sbatch --test-only failed for $script"; sbatch --test-only "$script"; return 1; }
    jobid=$(sbatch --parsable "$script") || { say "sbatch failed for $script"; return 1; }
    jobid=${jobid%%;*}
    echo "$lineno $jobid $script" >> "$STATE"
    sha=$(git -C "$CISC662_REPO" rev-parse --short HEAD); dirty=$(git -C "$CISC662_REPO" status --porcelain | wc -l)
    base=$(basename "$script"); tag=${jobid}_${base%.sbatch}
    cp "$script" "$ARCH/${tag}.sbatch"
    git -C "$CISC662_REPO" diff HEAD > "$ARCH/${tag}_git.diff"
    (cd "$CISC662_REPO" && git ls-files --others --exclude-standard -- cisc662 | grep -E '\.(py|sh|sbatch|c|cu|hip|md|txt)$' \
        | tar -czf "$ARCH/${tag}_untracked.tgz" -T - 2>/dev/null)
    local name; name=$(grep -oE '^#SBATCH --job-name=[^ ]+' "$script" | cut -d= -f2)
    echo "$(date -Is),submit,$jobid,$name,$tasks,$script,$sha,$dirty,,,,$CISC662_WORK/logs/${name}-${jobid}.out,$ARCH/${tag}.sbatch" >> "$RUNLOG_CSV"
    echo "- $(date '+%Y-%m-%d %H:%M') **SUBMIT** \`$jobid\` $name (${tasks:-?}) -- script \`$script\` (archived \`$ARCH/${tag}.sbatch\`), git $sha + $dirty uncommitted paths (diff + untracked sources archived)" >> "$RUNLOG_MD"
    say "submitted $jobid <- $script ($tasks)"
}

say "queue $QFILE: session cap ${SESSION_CAP} SU, project stop ${PROJECT_STOP} SU, <= ${MAX_INFLIGHT} in flight, poll ${POLL}s, exits after ${MAX_HOURS} h"
declare -A ENDED=()
while [ "$(date +%s)" -lt "$END" ]; do
    [ -f "$HALT" ] && { say "HALT file present: $(cat "$HALT") -- not submitting; exiting"; exit 1; }
    # 1. account for the jobs this queue submitted
    session_su=0; inflight_su=0; unfinished=0
    while read -r ln jid scr; do
        [ -z "${jid:-}" ] && continue
        read -r st su < <(job_cost "$jid")
        session_su=$(awk -v a="$session_su" -v b="$su" 'BEGIN { printf "%.3f", a+b }')
        case "$st" in
            PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED)
                unfinished=$((unfinished + 1)); inflight_su=$(awk -v a="$inflight_su" -v b="$su" 'BEGIN { printf "%.3f", a+b }') ;;
            *)  if [ -z "${ENDED[$jid]:-}" ] && ! grep -q ",end,$jid," "$RUNLOG_CSV"; then record_end "$jid" "$scr" "$st" "$su"; fi
                ENDED[$jid]=1
                if [ "$st" != COMPLETED ]; then
                    echo "job $jid ($scr) ended $st at $(date -Is)" > "$HALT"
                    echo "- $(date '+%Y-%m-%d %H:%M') **QUEUE HALTED**: job \`$jid\` ended $st -- fix the cause, delete \`$HALT\`, restart the queue" >> "$RUNLOG_MD"
                    say "job $jid ended $st -> HALT (see $HALT)"; exit 1
                fi ;;
        esac
    done < "$STATE"
    # 2. next entry in order
    next_ln=""; next_scr=""; next_tasks=""; blocked=""
    ln=0
    while IFS= read -r entry || [ -n "$entry" ]; do
        ln=$((ln + 1))
        entry=$(echo "$entry" | sed 's/[[:space:]]*$//')
        [ -z "$entry" ] && continue
        case "$entry" in \#*) continue ;; WAIT*) blocked=$entry; break ;; esac
        if ! awk -v l="$ln" '$1 == l { f = 1 } END { exit !f }' "$STATE"; then
            next_ln=$ln; next_scr=$(echo "$entry" | awk '{print $1}'); next_tasks=$(echo "$entry" | awk '{print $2}'); break
        fi
    done < "$QFILE"
    mine=$(squeue -u "$USER" -h -o %j 2>/dev/null | grep -c '^c662-')
    [ "$unfinished" -gt "$mine" ] && mine=$unfinished   # a just-submitted job may not be in squeue yet
    if [ -z "$next_ln" ]; then
        if [ -n "$blocked" ]; then say "blocked at '$blocked' ($mine in flight, session ${session_su} SU)"
        elif [ "$unfinished" -eq 0 ]; then say "queue drained, all jobs finished (session ${session_su} SU). exiting"; exit 0
        else say "queue drained, waiting for $unfinished job(s)"; fi
    elif [ "$mine" -ge "$MAX_INFLIGHT" ]; then
        say "next: $next_scr -- waiting ($mine c662 jobs in flight)"
    else
        scr_abs=$CISC662_REPO/$next_scr; [ -f "$next_scr" ] && scr_abs=$(readlink -f "$next_scr")
        if [ ! -f "$scr_abs" ]; then echo "missing script $next_scr" > "$HALT"; say "missing script $next_scr -> HALT"; exit 1; fi
        new_su=$(awk -v b="$(script_billing "$scr_abs")" -v t="$(script_walltime_s "$scr_abs")" 'BEGIN { printf "%.3f", b*t/3600000 }')
        proj_su=$(awk -v l="$(ledger_su)" -v i="$inflight_su" -v n="$new_su" 'BEGIN { printf "%.3f", l+i+n }')
        if awk -v s="$session_su" -v n="$new_su" -v c="$SESSION_CAP" 'BEGIN { exit !(s+n > c) }'; then
            echo "session cap: ${session_su} + ${new_su} > ${SESSION_CAP} SU before $next_scr" > "$HALT"
            echo "- $(date '+%Y-%m-%d %H:%M') **QUEUE HALTED (budget)**: $(cat "$HALT")" >> "$RUNLOG_MD"; say "$(cat "$HALT") -> HALT"; exit 1
        fi
        if awk -v p="$proj_su" -v c="$PROJECT_STOP" 'BEGIN { exit !(p >= c) }'; then
            echo "project total would reach ${proj_su} SU (>= ${PROJECT_STOP}: pre-P1 re-plan checkpoint, ask the user) before $next_scr" > "$HALT"
            echo "- $(date '+%Y-%m-%d %H:%M') **QUEUE HALTED (budget)**: $(cat "$HALT")" >> "$RUNLOG_MD"; say "$(cat "$HALT") -> HALT"; exit 1
        fi
        say "submitting $next_scr: worst case ${new_su} SU (session ${session_su} -> cap ${SESSION_CAP}; project -> ${proj_su})"
        if ! submit "$next_ln" "$scr_abs" "$next_tasks"; then
            echo "submission failed for $next_scr at $(date -Is)" > "$HALT"; say "submission failed -> HALT"; exit 1
        fi
        continue   # re-evaluate immediately (a second job may fit)
    fi
    sleep "$POLL"
done
say "reached ${MAX_HOURS} h, exiting (restart to continue; state is in $STATE)"
