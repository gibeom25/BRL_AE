#!/usr/bin/env bash
# Phase 5 rollouts in experiment1_plan.md §3.2 order. Skips runs whose merged result JSON already exists.
#
#   bash experiment1/run/phase5_rollouts.sh            # everything, in priority order
#   EPISODES=2 ONLY="1" bash experiment1/run/phase5_rollouts.sh   # quick partial run (writes to the same dir!)
#
# Env overrides:
#   EPISODES     episodes per task (default 20 -> n=200)
#   NUM_WORKERS  parallel env processes, tasks split across them (default 5: 32 CPU cores, 4 threads each,
#                leaves cores for a concurrent training job; each process holds the VLM, ~2 GB GPU)
#   AUX          auxiliary perturbation for step 6: light | noise (default light: changes front AND wrist,
#                contrast to camera/noise which change only the front camera — see env_check/perturb_matrix.json)
#   ONLY         space-separated subset of steps (1..6) to run
#   ROLLOUT_DIR  default outputs/experiment1/rollouts
set -u
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
EPISODES=${EPISODES:-20}
NUM_WORKERS=${NUM_WORKERS:-5}
AUX=${AUX:-light}
ONLY=${ONLY:-"1 2 3 4 5 6"}
ROLLOUT_DIR=${ROLLOUT_DIR:-outputs/experiment1/rollouts}
CKPT_ROOT=outputs/experiment1/ckpt
LOGDIR=$ROLLOUT_DIR/run_logs
mkdir -p "$LOGDIR"
PY="conda run --no-capture-output -n brl_ae python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/phase5.log"; }

# "trained" = final train_log.json written (has elapsed_min) — mid-training step*.pt files do not count
has_ckpt() { [ -f "$CKPT_ROOT/$1/train_log.json" ] && grep -q '"elapsed_min"' "$CKPT_ROOT/$1/train_log.json"; }

run_one() {  # model perturb h_mode
  local model=$1 perturb=$2 hmode=$3 tag="$1__$2__$3"
  if [ -f "$ROLLOUT_DIR/$tag.json" ]; then log "SKIP $tag (result exists)"; return 0; fi
  if [ "$model" = "M5_verify" ]; then log "SKIP $tag (M5_verify rollout not implemented yet -> N/A)"; return 0; fi
  if ! has_ckpt "$model"; then log "SKIP $tag (training of $model not finished -> N/A for now; rerun later)"; return 0; fi
  log "START $tag (episodes/task=$EPISODES, workers=$NUM_WORKERS)"
  local pids=() w
  for ((w = 0; w < NUM_WORKERS; w++)); do
    local ids=()
    for ((t = w; t < 10; t += NUM_WORKERS)); do ids+=("$t"); done
    [ ${#ids[@]} -eq 0 ] && continue
    $PY experiment1/scripts/06_rollout.py --model "$model" --perturb "$perturb" --h_mode "$hmode" \
      --episodes_per_task "$EPISODES" --task_ids "${ids[@]}" --save_video -o rollout_dir="$ROLLOUT_DIR" \
      > "$LOGDIR/${tag}__w$w.log" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  if [ $fail -ne 0 ]; then log "FAIL $tag (see $LOGDIR/${tag}__w*.log); partial JSONs kept, not merged"; return 1; fi
  $PY experiment1/scripts/06b_merge_rollouts.py --tag "$tag" --rollout_dir "$ROLLOUT_DIR" 2>&1 | tee -a "$LOGDIR/phase5.log"
  log "DONE $tag"
}

step() { [[ " $ONLY " == *" $1 "* ]]; }

log "phase5 begin: EPISODES=$EPISODES NUM_WORKERS=$NUM_WORKERS AUX=$AUX ONLY='$ONLY'"
# 1) M1, M2, M3 clean   (M2_O promoted to required)
step 1 && { run_one M1_h none normal; run_one M3_hO none normal; run_one M2_O none normal; }
# 2) M1, M2, M3 Camera
step 2 && { run_one M1_h camera normal; run_one M3_hO camera normal; run_one M2_O camera normal; }
# 3) M3 clean + online h_swap
step 3 && run_one M3_hO none swap
# 4) M4 clean, Camera
step 4 && { run_one M4_moe none normal; run_one M4_moe camera normal; }
# 5) M5 clean, Camera
step 5 && { run_one M5_verify none normal; run_one M5_verify camera normal; }
# 6) all models, auxiliary perturbation
step 6 && for m in M1_h M2_O M3_hO M4_moe M5_verify; do run_one "$m" "$AUX" normal; done
log "phase5 end"
