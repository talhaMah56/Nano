#!/bin/bash
# Session 2, task BAS-1 (decision D1): what does torch.compile already fuse in R12, and how many DRAM bytes
# does each fused region move?
#   step 1  R12 for 3 steps with TORCH_COMPILE_DEBUG -> Inductor output_code.py per graph (caches disabled so
#           codegen actually runs; TORCH_LOGS=output_code is written at the same time as a fallback)
#   step 2  ncu byte survey of ONE whole compiled step (step 20) inside the NANOCHAT_PROFILE_STEPS window.
#           --profile-from-start off is mandatory (otherwise the window is ignored, see PLAN.md 7.1).
#           Deviation from the card: no -k filter (all ~915 launches of the step, not only the 772 matching
#           'triton_|nvjet|flash'), so the table also covers non-Triton kernels; the window bounds the count.
#   step 3  fallback only if step 2 failed: Tier-A Triton kernels only, 300 launches.
# Analysis (login node, after the job): python -m cisc662.analysis.inductor_fusion <OUT>
#SBATCH --job-name=c662-s2-inductor
#SBATCH --account=bffp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:25:00
#SBATCH --constraint=nvperf
#SBATCH --output=/work/nvme/bffp/tmahmood1/cisc662/logs/%x-%j.out

source /u/tmahmood1/project/nanochat/cisc662/slurm/_common.sh
OUT=$CISC662_REPO/cisc662/results/runs/s2_inductor_${SLURM_JOB_ID}
P=$CISC662_WORK/profiles/s2_inductor_${SLURM_JOB_ID}
mkdir -p "$OUT" "$P"
use_local_base_dir   # throwaway checkpoints -> node-local /tmp
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
    --format=csv -lms 500 > "$OUT/gpu_monitor.csv" &
MON_PID=$!
R12="--depth=12 --max-seq-len=2048 --device-batch-size=16 --total-batch-size=32768 --window-pattern=L --core-metric-every=-1 --sample-every=-1 --eval-every=-1"
python -c "import nanochat.flash_attention as f; assert f.HAS_FA3, 'FA3 not available -> abort (would silently fall back to SDPA)'; print('HAS_FA3', f.HAS_FA3)"

# step 1: Inductor generated code (fresh codegen, no cache reads/writes)
# the run log holds all generated code (TORCH_LOGS also writes to stderr) -> keep it on /work/nvme, tail into $OUT
run_step inductor_dump "$P/inductor_dump.log" timeout 480 \
    env TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR="$P/compile_debug" \
        TORCH_LOGS=output_code TORCH_LOGS_OUT="$P/output_code.log" \
    python -m scripts.base_train $R12 --num-iterations=3 --model-tag=p_s2_dump
tail -n 200 "$P/inductor_dump.log" > "$OUT/inductor_dump_tail.log" || true
# keep only the generated kernels + fx graph summaries in git-tracked results (the full debug dir stays on /work/nvme)
mkdir -p "$OUT/output_code"
n=0
while IFS= read -r f; do
    n=$((n + 1)); cp "$f" "$OUT/output_code/graph_$(printf %03d $n)_output_code.py"
    d=$(dirname "$f"); for g in fx_graph_readable.py fx_graph_runnable.py ir_post_fusion.txt; do [ -f "$d/$g" ] && cp "$d/$g" "$OUT/output_code/graph_$(printf %03d $n)_$g"; done
    echo "graph_$(printf %03d $n) <- $f" >> "$OUT/output_code/INDEX.txt"
done < <(find "$P/compile_debug" -name output_code.py 2>/dev/null | sort)
# kernel -> post-grad node provenance (TORCH_COMPILE_DEBUG=1 enables provenance tracking level 1)
m=0
while IFS= read -r f; do
    m=$((m + 1)); cp "$f" "$OUT/output_code/provenance_$(printf %03d $m).json"; echo "provenance_$(printf %03d $m).json <- $f" >> "$OUT/output_code/INDEX.txt"
done < <(find "$P/compile_debug" -name "inductor_provenance_tracking_node_mappings*.json" 2>/dev/null | sort)
echo "=== copied $n output_code.py files"
[ "$n" -eq 0 ] && [ -s "$P/output_code.log" ] && cp "$P/output_code.log" "$OUT/output_code/output_code.log"

# step 2: ncu byte survey of the whole step 20 (warm caches from the S1 compile of the same code)
NCU_METRICS=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size
run_step ncu_step20 "$OUT/ncu_step20.log" timeout 720 \
    env NANOCHAT_PROFILE_STEPS=20:21 \
    ncu --profile-from-start off --target-processes application-only --nvtx --launch-skip 0 --launch-count 2000 --metrics "$NCU_METRICS" \
        --clock-control none --force-overwrite -o "$P/ncu_step20" \
    python -m scripts.base_train $R12 --num-iterations=22 --model-tag=p_s2_ncu
if [ -f "$P/ncu_step20.ncu-rep" ]; then
    ncu --import "$P/ncu_step20.ncu-rep" --csv --page raw --print-units base > "$OUT/ncu_step20_raw.csv" || echo "ncu import failed"
elif [ $(( $(date +%s) - JOB_T0 )) -lt 1000 ]; then
    # step 3: fallback, Tier-A Triton kernels only (only if >= ~8 min of walltime remain)
    run_step ncu_step20_tierA "$OUT/ncu_step20_tierA.log" timeout 360 \
        env NANOCHAT_PROFILE_STEPS=20:21 \
        ncu --profile-from-start off --target-processes application-only --nvtx -k regex:'triton_' --launch-skip 0 --launch-count 300 --metrics "$NCU_METRICS" \
            --clock-control none --force-overwrite -o "$P/ncu_step20_tierA" \
        python -m scripts.base_train $R12 --num-iterations=22 --model-tag=p_s2_ncu2
    [ -f "$P/ncu_step20_tierA.ncu-rep" ] && ncu --import "$P/ncu_step20_tierA.ncu-rep" --csv --page raw --print-units base > "$OUT/ncu_step20_tierA_raw.csv" || echo "ncu import (tierA) failed"
else
    echo "skip ncu_step20_tierA: not enough walltime left"; FAILED_STEPS="${FAILED_STEPS} ncu_step20_tierA_skipped"
fi

kill $MON_PID 2>/dev/null || true
du -sh "$P" "$OUT"; ls -la "$OUT"
echo "FAILED STEPS:${FAILED_STEPS:- none}"
