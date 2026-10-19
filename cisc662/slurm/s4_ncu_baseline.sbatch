#!/bin/bash
# Session 4, task BAS-6: Nsight Compute sections for the compiled step's top kernels + our kernels' measured bytes.
#   step 1  one compiled R12 step (step 21, warm caches) inside the NANOCHAT_PROFILE_STEPS window, kernels of the
#           top-15 families by S1 GPU time (78% of the step): cuBLAS nvjet GEMMs, FA3 fwd/bwd, Inductor's CE tail,
#           relu^2 and fused RoPE/QK-norm kernels -- about 450 launches, 5 sections each
#   step 2  our Triton kernels (c662_*: rmsnorm v2, qkrope, ce rec/gif) at R12 shape via bench/ncu_targets.py,
#           3 launches each; the summary takes the last (warm) launch per kernel
#   step 3  profiling/ncu_summary.py -> results/db/ncu/{baseline_compile,ours}_<job>.json (feeds ANA-1 roofline v1)
# ncu runs with --profile-from-start off (honours the cudaProfilerStart/Stop window; PLAN 7.1) and
# --clock-control none (times comparable to nsys/events). Time table (min): step 1 <= 15, step 2 <= 5, step 3 < 1.
#SBATCH --job-name=c662-s4-bas6
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=00:25:00
#SBATCH --constraint=nvperf
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out

source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh
JOBID=${SLURM_JOB_ID:-local}
OUT=$CISC662_REPO/cisc662/results/runs/s4_bas6_${JOBID}
P=$CISC662_WORK/profiles/s4_bas6_${JOBID}
DB=$CISC662_REPO/cisc662/results/db/ncu
mkdir -p "$OUT" "$P" "$DB"
use_local_base_dir
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
    --format=csv -lms 500 > "$OUT/gpu_monitor.csv" &
MON_PID=$!

SECTIONS="--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section WarpStateStats"
EXTRA="--metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"
R12="--depth=12 --max-seq-len=2048 --device-batch-size=16 --total-batch-size=32768 --window-pattern=L --core-metric-every=-1 --sample-every=-1 --eval-every=-1"
TOP="regex:nvjet_tst|FlashAttn(Fwd|Bwd)Sm90|triton_red_fused__log_softmax|triton_\w+_fused\w*relu|triton_per_fused__to_copy__unsafe_view_add_cat_mean"

run_step ncu_step "$OUT/ncu_step.log" timeout -k 30 900 env NANOCHAT_ATTN_IMPL=fa3 NANOCHAT_PROFILE_STEPS=21:22 \
    ncu --profile-from-start off --target-processes application-only --nvtx --kernel-name-base demangled \
        -k "$TOP" --launch-skip 0 --launch-count 1200 $SECTIONS $EXTRA --clock-control none \
        --force-overwrite -o "$P/ncu_step" \
    python -m scripts.base_train $R12 --num-iterations=23 --model-tag=p_s4_ncu
if [ -f "$P/ncu_step.ncu-rep" ]; then
    ncu --import "$P/ncu_step.ncu-rep" --csv --page raw --print-units base > "$OUT/ncu_step_raw.csv" || echo "import failed"
    python -m cisc662.profiling.ncu_summary "$OUT/ncu_step_raw.csv" --json "$DB/baseline_compile_${JOBID}.json" \
        > "$OUT/ncu_step_summary.txt" 2>&1 || echo "summary failed"
    head -n 20 "$OUT/ncu_step_summary.txt" | cut -c1-200
fi

run_step ncu_ours "$OUT/ncu_ours.log" timeout -k 30 420 \
    ncu --target-processes application-only -k "regex:^c662_" $SECTIONS $EXTRA --clock-control none \
        --force-overwrite -o "$P/ncu_ours" \
    python -m cisc662.bench.ncu_targets --iters 3
if [ -f "$P/ncu_ours.ncu-rep" ]; then
    ncu --import "$P/ncu_ours.ncu-rep" --csv --page raw --print-units base > "$OUT/ncu_ours_raw.csv" || echo "import failed"
    python -m cisc662.profiling.ncu_summary "$OUT/ncu_ours_raw.csv" --json "$DB/ours_${JOBID}.json" \
        > "$OUT/ncu_ours_summary.txt" 2>&1 || echo "summary failed"
    cat "$OUT/ncu_ours_summary.txt" | cut -c1-200
fi
# keep only the CSV/JSON in git; the .ncu-rep files stay on /work/nvme (see .gitignore)
kill $MON_PID 2>/dev/null || true
echo "FAILED STEPS:${FAILED_STEPS:- none}"
