#!/usr/bin/env bash
# Phase 4 chain (GPU training jobs <= 2 at any time; rollouts/offline eval may overlap training):
#   wait M1_h & M3_hO finished
#   -> start M2_O + M4_moe training (30k, concurrent)          [2 GPU training jobs]
#   -> offline eval M1_h, M3_hO -> phase5 rollouts steps 1-3 (M2 entries skipped until M2 is done)
#   -> wait M2_O & M4_moe finished -> offline eval M2_O, M4_moe -> phase5 steps 1-4 again (fills M2/M4; finished runs skipped)
#   verifier (04_train_verifier.py) comes after this: not implemented yet.
# Log: outputs/setup_logs/phase4_chain.log
set -u
cd "$(dirname "$0")/../.."
L=outputs/setup_logs
CK=outputs/experiment1/ckpt
PY="conda run --no-capture-output -n brl_ae python"
log() { echo "[$(date '+%F %T')] $*" | tee -a $L/phase4_chain.log; }
trained() { [ -f "$CK/$1/train_log.json" ] && grep -q '"elapsed_min"' "$CK/$1/train_log.json"; }
wait_trained() { while ! trained "$1"; do sleep 20; done; log "$1 training finished"; }

log "chain start"
wait_trained M1_h
wait_trained M3_hO

for m in m2_O m4_moe; do
  if trained "$( [ $m = m2_O ] && echo M2_O || echo M4_moe )"; then log "$m already trained, skip"; continue; fi
  nohup $PY experiment1/scripts/02_train_ae.py --config experiment1/configs/$m.yaml > $L/p6_train_$m.log 2>&1 &
  log "started training $m (pid $!) -> $L/p6_train_$m.log"
done

for m in M1_h M3_hO; do
  [ -f outputs/experiment1/offline/$m.json ] && { log "offline $m exists, skip"; continue; }
  $PY experiment1/scripts/05_offline_eval.py --model $m > $L/offline_$m.log 2>&1 && log "offline eval $m done" || log "offline eval $m FAILED ($L/offline_$m.log)"
done

ONLY="1 2 3" bash experiment1/run/phase5_rollouts.sh > $L/phase5_pass1.log 2>&1
log "phase5 pass1 (steps 1-3) finished"

wait_trained M2_O
wait_trained M4_moe
for m in M2_O M4_moe; do
  [ -f outputs/experiment1/offline/$m.json ] && { log "offline $m exists, skip"; continue; }
  $PY experiment1/scripts/05_offline_eval.py --model $m > $L/offline_$m.log 2>&1 && log "offline eval $m done" || log "offline eval $m FAILED ($L/offline_$m.log)"
done
ONLY="1 2 3 4" bash experiment1/run/phase5_rollouts.sh > $L/phase5_pass2.log 2>&1
log "phase5 pass2 (steps 1-4) finished; next: verifier (not implemented)"
log "chain end"
