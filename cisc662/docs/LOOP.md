# LOOP.md — self-running worker/reviewer loop (start here if you are an agent)

Written 2026-10-29. This file is self-contained: an agent that reads **this file, then `docs/BOARD.md`, then
`docs/AGENT_GUIDE.md`** has everything needed to pick a task, do it, get it reviewed and land it, with no human in
the loop. Where this file and AGENT_GUIDE.md disagree, this file wins. Deadlines: code freeze **11-24**, final
presentation **11-30**, report **12-07** (PLAN.md section 6-REV).

## 1. Roles
| Role | How many | Does | Never does |
|---|---|---|---|
| **Worker** | 2–3 | claims one READY card, implements it in its own worktree, runs CPU checks, queues GPU jobs, hands off for review | reviews its own card; edits another card's files; merges into `cisc662` |
| **Reviewer** | 1–2 | reviews cards in REVIEW against section 5; PASS → merges; FAIL → writes findings and sends back | implements the card it reviews; loosens a gate to make it pass |
| **Operator** | exactly 1 (may also be a reviewer) | owns the GPU queue runner (section 4), the SU ledger, the board's GPU column, the end-of-day STATUS.md update and login-node hygiene | writes kernel code |

An agent picks its role at start from `docs/BOARD.md` → "Roles in use" (first free role, in the order operator,
reviewer, worker) and writes its name there (`worker-1`, `reviewer-1`, `operator`). It removes the entry when it stops.

## 2. The board (`docs/BOARD.md`) is the only coordination channel
- Status flow: `READY → CLAIMED(<agent>, <date>) → REVIEW → (PASS → DONE) | (FAIL → CLAIMED again by the same worker)`.
  `BLOCKED(<reason>)` when a dependency is not DONE. `DROPPED(<reason>)` only by the operator, on the dates in section 7.
- **Claiming:** edit the card's status line to `CLAIMED(<you>, <date>)` and commit that one-line change on `cisc662`
  immediately (`git commit -m "board: claim <ID>"`). If the commit conflicts, someone else won: pull, pick another.
  Take the highest-priority READY card whose dependencies are DONE (Tier 0 before Tier 1 before Tier 2 before Tier 3).
- A claim older than 48 h with no commits on its branch is stale: the operator resets it to READY.
- Every status change is one small commit on `cisc662` touching only BOARD.md.

## 3. Worker procedure (one card at a time)
1. Read the card in `docs/TASKS.md` (steps, deliverables, acceptance) and its BOARD.md line (tier, deadline, any
   scope change written there — BOARD.md overrides TASKS.md on scope and dates).
2. `git worktree add /u/tmahmood1/project/wt-<ID> -b task/<ID> cisc662` and work only there. Only touch the files the card names plus
   new files under `cisc662/`. Upstream nanochat files are touched only through the existing opt-in hooks.
3. CPU checks on the login node (rules in section 6): unit tests under `TRITON_INTERPRET=1` where the code supports
   it, `nvcc -c ... --ptxas-options=-v` compile checks, `python -m py_compile`, `bash -n`, `sbatch --test-only`.
4. GPU work: write ONE sbatch per card, `cisc662/slurm/l_<ID>.sbatch`, modelled on `s4_ce.sbatch` (gates first, then
   kbench/ncu; `run_step`; walltime ≤ 30 min, one GPU, `--account=bffp-dtai-gh --partition=ghx4`). **Its first line
   after the `#SBATCH` block must be `export CISC662_REPO_OVERRIDE=/u/tmahmood1/project/wt-<ID>`, before
   `source .../_common.sh`**, so the job runs your branch's code and writes results into your worktree (without it
   the job silently runs the main tree). Commit it on your branch, then append the ABSOLUTE path
   `/u/tmahmood1/project/wt-<ID>/cisc662/slurm/l_<ID>.sbatch <ID>` to `cisc662/slurm/queue_loop.txt` on `cisc662`
   (one-line commit). The operator's runner submits it; you read the result in `results/runs/` when the board's GPU
   column says `ENDED <jobid>`. Never call `sbatch` yourself.
5. Results: gated rows only (the `cisc662.tests.gates` helper refuses otherwise — never bypass it). Write the ISA card
   section in `docs/findings.md` if the card asks for one. Numbers you did not measure are labelled STATIC,
   PROJECTED or SCALED; **nothing AMD is "measured" unless a Delta job produced it**.
6. Hand-off: commit on `task/<ID>` with a message listing what changed, job ids, and the acceptance items met/unmet.
   Set the card to `REVIEW` and write in BOARD.md: branch, commit, job ids, a 3-line summary.
7. After FAIL: fix only what the review lists, re-queue GPU work if needed, back to REVIEW. Two FAILs on the same
   card → the operator decides: narrow the scope (written on the board) or DROP.

## 4. GPU queue (operator only)
- One runner, in tmux session `cisc662` on the login node the operator uses:
  `tmux new-session -d -s cisc662 -n queue "bash cisc662/slurm/queue.sh cisc662/slurm/queue_loop.txt <cap> 30"`
  The runner already enforces: ≤ 2 project jobs in flight, `bash -n` + `sbatch --test-only` before submitting,
  per-session cap, project stop at 8 SU, halt on any non-COMPLETED job, archival of the exact script, RUNLOG rows.
- Caps: **session cap 1.5 SU per day**; project total **may not pass 8 SU without the user**. The ledger is
  `$CISC662_WORK/logs/job_ledger.csv` (cumulative before the loop: about 0.85 + the BAS-5 job).
- If the runner halts (`queue_loop.txt.halt`): read the failed job's log, write the cause on the card, set the card
  back to CLAIMED for its worker, delete the halt file, restart the runner.
- After each job ends: write `ENDED <jobid> <state> <SU>` in the card's GPU column.
- The runner exits after 30 h by design (login-node 7-day and CPU-time rules); restart it daily. Close tmux and
  check `ps -u $USER` on **all four login nodes** at the end of each day.

## 5. Review checklist (reviewer; every item must hold for PASS)
1. **Scope:** only the card's files changed (`git diff --stat cisc662...task/<ID>`); no upstream nanochat edits outside
   the opt-in hooks; no secrets, no large binaries (`.ncu-rep`, `.nsys-rep`, checkpoints stay on `/work`).
2. **Correctness:** the gate tests exist, ran on the GPU in the job the card cites (read its log, not the summary),
   and passed with the A1 tolerances unchanged. Re-run the CPU-side tests yourself.
3. **Numbers:** every number in findings.md / the hand-off note appears in a results file from a cited job id;
   speedups are against eager and compile **from the same job**; no unlabeled STATIC/SCALED/PROJECTED value;
   AMD numbers are STATIC unless from a Delta job.
4. **Code:** reads like the surrounding code; no dead code or commented-out experiments; error paths don't hide
   failures (no bare `except: pass`); one-kernel-per-direction / no-graph-break tests where the card asks.
5. **Acceptance:** each acceptance bullet of the card is met, or the unmet ones are listed with a reason and the
   operator agreed to narrow the scope on the board.
6. **Budget:** the card's jobs stayed within its GPU budget (TASKS.md) or the overrun is justified on the board.
- **PASS:** `git checkout cisc662 && git merge --no-ff task/<ID> -m "<ID>: <title> (reviewed by <you>)"`, run
  `python -m pytest cisc662/tests -q -x -k "not gpu"` (CPU subset) on the merged tree, set DONE, remove the worktree.
  If the merge conflicts: FAIL with "rebase on cisc662".
- **FAIL:** write numbered findings under the card in BOARD.md (file:line, what is wrong, what would fix it).

## 6. Hard rules (anyone)
- DeltaAI only: account `bffp-dtai-gh`, partition `ghx4`. Never `bffp-delta-gpu`, never submit to Delta from here.
- Login node: any process is killed at 30 CPU-minutes or 7 days. Use `nice -n 19`, ≤ 8 threads, split compile
  sweeps into < 20-CPU-minute processes; no benchmarks or GPU work on the login node; no idle daemons.
  No foreground `sleep` polling.
- **Git:** commits are allowed on `task/<ID>` branches, one-line BOARD.md/queue commits on `cisc662`, and reviewer
  merges into `cisc662`. **Never push to GitHub (`nano`)** — the user pushes. Pushing `cisc662` to the shared bare
  repo `shared` (on `/work/nvme`) is allowed for the operator at end of day. Never rewrite history on `cisc662`.
  End commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never edit another agent's worktree. Never delete results files; supersede them with a newer job.
- Stop and write `NEEDS-USER: <question>` at the top of BOARD.md (then work on something else) for: anything that
  would pass 8 SU, a change to a gate tolerance, a scope change to a Tier 0 card, touching Delta, or anything
  irreversible. The user reads that line.

## 7. Dates the operator enforces
- **11-12 checkpoint:** if INT-1 and KC-2 are not DONE, Tier 2/3 cards not yet CLAIMED become DROPPED, and KC-3 is
  narrowed to a compile-check + static ISA.
- **11-18:** every non-Tier-0 card not in REVIEW is DROPPED (report it as "attempted, partial" if work exists).
- **11-20:** D5 go/no-go per kernel (operator writes the decision table in STATUS.md).
- **11-24 code freeze:** after this only INT-6 (final measurement job), figures, deck and report change.
- End of each day: operator updates `docs/STATUS.md` (done today, SU used, blockers) and pushes to `shared`.
