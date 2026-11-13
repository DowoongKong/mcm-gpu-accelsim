#!/bin/bash
# Hypothesis-1 reference-run queue (docs/h1_llama70b_decode_batch_sweep.ko.md section 9): for every batch size in order, wait
# until the capture chain has finished that trace (S6 TRACES PASS), build its ref_mcm arm (h1_make_ref_arm.sh) and launch
# sim/launch_ref.sh inside mcm-accelsim -- never with more than MAX simulations of this family running at once
# (user instruction 2026-11-09: "16 thread로 2개보다 많이 돌리지는 마"; each ref is 16 threads, so MAX=2 keeps 32 threads).
# Running sims are counted from the accel-sim.out command lines that read a trace under llm_hypothesis1/.
# Usage: h1_ref_queue.sh <model dir name> <MAX concurrent> <B> [<B> ...]
set -u
MODEL=$1; MAX=$2; shift 2
H=/home/user/accel-sim-docker/llm_exp
ROOT=$H/llm_hypothesis1/$MODEL
CROOT=/accel-sim/llm_exp/llm_hypothesis1/$MODEL
LOG=$ROOT/ref_queue.log
log(){ echo "$(date '+%F %T') [h1_ref_queue] $*" | tee -a $LOG; }
running(){ pgrep -fc 'accel-sim\.out -config \./gpgpusim\.config -config \./trace\.config -trace /accel-sim/llm_exp/llm_hypothesis1/' || true; }
exec {LK}>$ROOT/.ref_queue.lock
flock -n $LK || { echo "ref queue already running"; exit 1; }
log "queue start: model=$MODEL max_concurrent=$MAX batches=$*; currently running $(running)"
for B in "$@"; do
  CFG=s256_b$(printf '%04d' $B)_decode_layer0
  P=$ROOT/$CFG
  if [ -e $P/sim/ref_mcm/run.log ]; then log "$CFG: ref_mcm already has run.log -- skipping"; continue; fi
  if ! ls $P/h1_check_traces_*.log >/dev/null 2>&1 || ! grep -q '^S6 TRACES' $P/h1_check_traces_*.log 2>/dev/null; then
    log "$CFG: waiting for the capture chain (trace check not finished)"
    until ls $P/h1_check_traces_*.log >/dev/null 2>&1 && grep -q '^S6 TRACES' $P/h1_check_traces_*.log 2>/dev/null; do sleep 60; done
  fi
  if ! grep -q '^S6 TRACES PASS' $P/h1_check_traces_*.log; then log "$CFG: trace check did not PASS -- skipping"; continue; fi
  out=$(bash $H/scripts/h1_make_ref_arm.sh $MODEL $CFG 2>&1) || { log "$CFG: arm build failed: $out"; continue; }
  log "$CFG: arm built: $out"
  if [ "$(running)" -ge "$MAX" ]; then
    log "$CFG: $(running) sims running (max $MAX) -- waiting for a slot"
    until [ "$(running)" -lt "$MAX" ]; do sleep 60; done
  fi
  docker exec -d -u 1035:1035 mcm-accelsim bash $CROOT/$CFG/sim/launch_ref.sh
  until [ -e $P/sim/ref_mcm/run.log ] || grep -q 'ABORT' $P/sim/launch.log 2>/dev/null; do sleep 10; done
  if [ -e $P/sim/ref_mcm/run.log ]; then log "$CFG: launched ($(running) running now)"; else log "$CFG: launcher aborted: $(grep ABORT $P/sim/launch.log | tail -1)"; fi
done
log "queue finished: all batches launched or skipped"
