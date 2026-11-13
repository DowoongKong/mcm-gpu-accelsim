#!/bin/bash
# Hypothesis-1 simulation job queue (docs/h1_llama70b_decode_batch_sweep.ko.md section 10): runs arms in the given order,
# never more than MAX accel-sim.out processes of the llm_hypothesis1 family at once (user instruction 2026-11-09: 16-thread
# arms, at most two). A job is "<model dir>|<config dir>|<arm>" with arm = ref_mcm | base_mcm | ptwhop0_mcm.
#   ref_mcm      : waits for the capture chain's S6 TRACES PASS, builds the arm with h1_make_ref_arm.sh, launches sim/launch_ref.sh
#   base/ptwhop0 : waits for ref_mcm's EXIT=0 + placement map, builds with h1_make_arm.sh, launches sim/launch_<arm>.sh
# The queue itself waits for every prerequisite before taking a slot, so a launched arm starts its simulator immediately.
# Usage: h1_job_queue.sh <queue name> <MAX> <job> [<job> ...]
set -u
QNAME=$1; MAX=$2; shift 2
H=/home/user/accel-sim-docker/llm_exp
LOGDIR=$H/llm_hypothesis1
LOG=$LOGDIR/queue_$QNAME.log
log(){ echo "$(date '+%F %T') [queue:$QNAME] $*" | tee -a $LOG; }
running(){ ps -eo stat=,comm= | awk '$2=="accel-sim.out" && $1 !~ /^Z/' | wc -l; }   # live simulator processes on the host (zombies and gdb wrappers excluded): the cap is global
RAM_MIN_GB=${RAM_MIN_GB:-30}   # 2026-11-09: a batch-1024 decode-layer sim reaches 24 GB RSS and the host swap is full; launch only with this much MemAvailable
avail_gb(){ awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
exec {LK}>$LOGDIR/.queue_$QNAME.lock
flock -n $LK || { echo "queue $QNAME already running"; exit 1; }
log "start: max_concurrent=$MAX jobs=$#; running now $(running)"
for JOB in "$@"; do
  IFS='|' read -r MODEL CFG ARM <<< "$JOB"
  P=$LOGDIR/$MODEL/$CFG; CP=/accel-sim/llm_exp/llm_hypothesis1/$MODEL/$CFG
  if [ -e $P/sim/$ARM/run.log ]; then log "$MODEL/$CFG/$ARM: run.log exists -- skipping"; continue; fi
  if [ "$ARM" = debug_gdb ]; then
    [ -x $P/sim/debug_gdb/run_gdb.sh ] || { log "$MODEL/$CFG/$ARM: run_gdb.sh missing -- skipping"; continue; }
    out="gdb reproduction (sim/debug_gdb)"; LAUNCH=$CP/sim/debug_gdb/run_gdb.sh
  elif [ "$ARM" = ref_mcm ]; then
    if ! ls $P/h1_check_traces_*.log >/dev/null 2>&1 || ! grep -q '^S6 TRACES' $P/h1_check_traces_*.log 2>/dev/null; then
      log "$MODEL/$CFG/$ARM: waiting for the capture chain (trace check)"
      until ls $P/h1_check_traces_*.log >/dev/null 2>&1 && grep -q '^S6 TRACES' $P/h1_check_traces_*.log 2>/dev/null; do sleep 120; done
    fi
    grep -q '^S6 TRACES PASS' $P/h1_check_traces_*.log || { log "$MODEL/$CFG/$ARM: trace check not PASS -- skipping"; continue; }
    out=$(bash $H/scripts/h1_make_ref_arm.sh $MODEL $CFG 2>&1) || { log "$MODEL/$CFG/$ARM: arm build failed: $out"; continue; }
    LAUNCH=$CP/sim/launch_ref.sh
  else
    if ! grep -q '^EXIT=0 ' $P/sim/ref_mcm/exit_code.txt 2>/dev/null; then
      log "$MODEL/$CFG/$ARM: waiting for ref_mcm to finish"
      until [ -s $P/sim/ref_mcm/exit_code.txt ]; do sleep 120; done
      grep -q '^EXIT=0 ' $P/sim/ref_mcm/exit_code.txt || { log "$MODEL/$CFG/$ARM: ref_mcm failed ($(cat $P/sim/ref_mcm/exit_code.txt)) -- skipping"; continue; }
    fi
    [ -s $P/sim/ref_mcm/placement_mcm.map ] || { log "$MODEL/$CFG/$ARM: ref placement map missing -- skipping"; continue; }
    out=$(bash $H/scripts/h1_make_arm.sh $MODEL $CFG $ARM 2>&1) || { log "$MODEL/$CFG/$ARM: arm build failed: $out"; continue; }
    LAUNCH=$CP/sim/launch_$ARM.sh
  fi
  log "$MODEL/$CFG/$ARM: arm ready: $out"
  if [ "$(running)" -ge "$MAX" ] || [ "$(avail_gb)" -lt "$RAM_MIN_GB" ]; then
    log "$MODEL/$CFG/$ARM: $(running) sims running (max $MAX), MemAvailable $(avail_gb) GB (min $RAM_MIN_GB) -- waiting"
    until [ "$(running)" -lt "$MAX" ] && [ "$(avail_gb)" -ge "$RAM_MIN_GB" ]; do sleep 60; done
  fi
  docker exec -d -u 1035:1035 mcm-accelsim bash $LAUNCH
  LLOG=$P/sim/launch.log; [ "$ARM" = ref_mcm ] || LLOG=$P/sim/launch_$ARM.log; [ "$ARM" = debug_gdb ] && LLOG=$P/sim/debug_gdb/run.log
  until [ -e $P/sim/$ARM/run.log ] || grep -q 'ABORT' $LLOG 2>/dev/null; do sleep 10; done
  if [ -e $P/sim/$ARM/run.log ]; then log "$MODEL/$CFG/$ARM: launched ($(running) running now)"; else log "$MODEL/$CFG/$ARM: launcher aborted: $(grep ABORT $LLOG | tail -1)"; fi
  sleep 5
done
log "finished: every job launched or skipped"
