#!/bin/bash
# steps_prepare_chain.sh -- host side, step 9 (docs/s9_b8s256_steps.ko.md section 5). One step-piece capture from
# start to simulation-ready: capture (llmtrace, tracer attached) -> weight-copy check -> post-processing -> per-window
# trace checks -> piece lists -> L1seq list -> experiment dirs (L1seq + pieces, llm_sim_tree_fast) with DRY checks ->
# optional post hook (e.g. kit build). No simulation is started.
# Usage: steps_prepare_chain.sh <model key> <exp dir name> <trace dir name> <batch> <S> <max tokens> <traced layers 1|2>
#                               <host model dir> <log tag> [<wait-for file>] [<post hook script>]
set -u
MODEL_KEY=$1; EXP=$2; TDIR=$3; BATCH=$4; S=$5; MAXTOK=$6; LAYERS=$7; HF_DIR=$8; TAG=$9
WAIT_FILE=${10:-}; POST_HOOK=${11:-}
H=/home/user/accel-sim-docker/llm_exp
X=$H/experiments/$EXP
T=$H/traces/$TDIR
CT=/accel-sim/llm_exp/traces/$TDIR                      # the same directory inside the containers
LOG=$X/chain.log
log(){ echo "$(date '+%F %T') [${TAG}_chain] $*" | tee -a $LOG; }
exec {LK}>$X/.chain.lock
flock -n $LK || { echo "chain already running"; exit 1; }
mkdir -p $T
[ -e $T/exit_code.txt ] && { log "$T already holds a capture (exit_code.txt) -- refusing to overwrite"; exit 1; }

if [ -n "$WAIT_FILE" ]; then
  log "waiting for $WAIT_FILE (GPU handover)"
  until [ -f "$WAIT_FILE" ]; do sleep 60; done
  log "handover file present: $(cat $WAIT_FILE)"
fi
free_gb=$(df -BG --output=avail $H | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge 20 ] || { log "disk free ${free_gb} GB < 20 GB -- stop"; exit 1; }
log "capture start ($MODEL_KEY, batch $BATCH, S $S, $MAXTOK tokens, traced layers $LAYERS; disk free ${free_gb} GB)"
docker exec llmtrace bash -c "
cd /accel-sim/llm_exp/scripts
mkdir -p $CT
env CUDA_INJECTION64_PATH=/accel-sim/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so \
    ACTIVE_FROM_START=0 USER_DEFINED_FOLDERS=1 TRACES_FOLDER=$CT NOBANNER=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  /venv/bin/python steps_capture.py $CT $MODEL_KEY $BATCH $S $MAXTOK $LAYERS \
  > $CT/run.log 2>&1; echo \"EXIT=\$?\" > $CT/exit_code.txt"
grep -q '^EXIT=0' $T/exit_code.txt && grep -q '^STEPS CAPTURE PASS' $T/run.log \
  || { log "capture failed ($(cat $T/exit_code.txt 2>/dev/null); $(grep '^STEPS CAPTURE' $T/run.log)) -- stop"; exit 1; }
log "capture passed: $(grep '^STEPS CAPTURE' $T/run.log); $(grep '^trace bytes' $T/run.log)"
grep -A12 '^pieces:' $T/run.log | grep -v '^pieces:' | while read -r ln; do log "  $ln"; done

python3 $H/scripts/check_weight_copies.py $HF_DIR $T/traces > $T/${TAG}_check_weight_copies.log 2>&1
grep -q '^WEIGHT COPIES PASS' $T/${TAG}_check_weight_copies.log || { log "G9-5 weight-copy check failed -- stop"; exit 1; }
log "G9-5 weight copies PASS"

log "post-processing start"
docker exec llmtrace bash -c "
cd $CT/traces
t0=\$SECONDS
nice -n 10 /accel-sim/accel-sim-framework/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing \
  ./kernelslist_ctx_* > ../${TAG}_postprocess.log 2>&1
echo \"EXIT=\$? SECONDS=\$((SECONDS-t0))\" > ../${TAG}_postprocess_exit.txt"
grep -q '^EXIT=0' $T/${TAG}_postprocess_exit.txt || { log "post-processing failed ($(cat $T/${TAG}_postprocess_exit.txt)) -- stop"; exit 1; }
log "post-processing done: $(cat $T/${TAG}_postprocess_exit.txt)"

last_piece=layer0; [ "$LAYERS" = 2 ] && last_piece=layer1
for step in $(awk -F'\t' 'NR>1 {print $1}' $T/regions.tsv | uniq); do
  a=$(awk -F'\t' -v s=$step '$1==s && $2=="prep" {print $3}' $T/regions.tsv)
  b=$(awk -F'\t' -v s=$step -v p=$last_piece '$1==s && $2==p {print $4}' $T/regions.tsv)
  c=$(awk -F'\t' -v s=$step '$1==s && $2=="tail" {print $3}' $T/regions.tsv)
  d=$(awk -F'\t' -v s=$step '$1==s && $2=="tail" {print $4}' $T/regions.tsv)
  for w in "$a $b" "$c $d"; do
    set -- $w
    nice -n 10 python3 $H/scripts/s6_check_traces.py $T/traces $H/llm_sim_tree_fast/gpu-simulator/ISA_Def/ampere_opcode.h \
      $T/${TAG}_postprocess.log $T/${TAG}_postprocess_exit.txt $1 $2 > $T/${TAG}_check_traces_$1_$2.log 2>&1
    grep -q '^S6 TRACES PASS' $T/${TAG}_check_traces_$1_$2.log || { log "G9-7 trace checks failed for $step window $1..$2 -- stop"; exit 1; }
    log "G9-7 trace checks passed: $step window $1..$2"
  done
done

python3 $H/scripts/s8_make_piece_lists.py $T/traces $T/regions.tsv > $T/${TAG}_piece_lists.log 2>&1
grep -q '^S8 PIECE LISTS PASS' $T/${TAG}_piece_lists.log || { log "piece lists failed -- stop"; exit 1; }
log "piece lists built"
python3 $H/scripts/make_l1seq_list.py $T/traces $T/regions.tsv > $T/${TAG}_l1seq_list.log 2>&1 \
  || { log "L1seq list failed -- stop"; exit 1; }
L1_N=$(grep '^L1SEQ LIST' $T/${TAG}_l1seq_list.log | sed 's/.*kernels=\([0-9]*\).*/\1/')
L1_C=$(grep '^L1SEQ LIST' $T/${TAG}_l1seq_list.log | sed 's/.*copies=\([0-9]*\).*/\1/')
L1_MD5=$(grep '^L1SEQ LIST' $T/${TAG}_l1seq_list.log | sed 's/.*md5=\([0-9a-f]*\).*/\1/')
want_n=$(awk -F'\t' 'NR>1 && ($2=="prep"||$2=="layer0"||$2=="tail") {s+=$5} END{print s}' $T/regions.tsv)
want_c=$(grep -c '^MemcpyHtoD,' $T/traces/kernelslist.g)
[ "$L1_N" = "$want_n" ] && [ "$L1_C" = "$want_c" ] \
  || { log "G9-8 L1seq list mismatch: kernels $L1_N (want $want_n) copies $L1_C (want $want_c) -- stop"; exit 1; }
log "L1seq list built: $L1_N kernels, $L1_C copies, md5 $L1_MD5; ranges: $(grep -v '^L1SEQ' $T/${TAG}_l1seq_list.log | tr '\t' ' ' | tr '\n' ';')"

cap_exception(){ # $1 dir
  cat > $1/CAP_EXCEPTION.txt <<EOT
Instruction cap exception (-gpgpu_max_insn 0) for $EXP/$(basename $1), recorded $(date '+%F') by Claude on the user's instruction
of 2026-09-19 ("b8s256으로 Qwen이랑 deepseek 모두 만들어줘": batch reduced from 32 to 8 so that whole prefill layers can be
simulated; same rule as the b32 L1seq / P-L2 and the step-7 exceptions). A batch-8 prefill layer piece is ~1.8e10 thread
instructions (> the standing 1e10 cap); the cap would truncate the prefill layer at ~57%, so this piece must run uncapped.
If this exception is not intended, delete this file before launching (the launcher then refuses to start).
EOT
}
make_dir(){ # $1 piece, $2 n_kernels, $3 cap, $4 trace md5
  mkdir -p $X/$1
  sed -e "s|@DIR@|$1|g" -e "s|@EXP@|$EXP|g" -e "s|@TRACE_DIR@|$TDIR|g" \
      -e "s|@N_KERNELS@|$2|g" -e "s|@MAX_INSN@|$3|g" -e "s|@MEASURE@|1|g" -e "s|@TRACE_MD5@|$4|g" \
      -e "s|@REF_TH@|16|g" -e "s|@ARM_TH@|8|g" -e "s|@PARALLEL@|1|g" \
      $H/scripts/steps_launch_template_fast.sh > $X/$1/launch.sh
  chmod +x $X/$1/launch.sh
  [ "$3" = 0 ] && cap_exception $X/$1
  docker exec -u 1035:1035 -e DRY=1 mcm-accelsim bash /accel-sim/llm_exp/experiments/$EXP/$1/launch.sh > /dev/null 2>&1
  if grep -q 'DRY run finished' $X/$1/launch.log; then log "dir $1: $2 kernels, cap $3, DRY OK"; return 0
  else log "dir $1: DRY FAILED ($(tail -1 $X/$1/launch.log))"; return 1; fi
}
dry_bad=0
make_dir L1seq $L1_N 0 $L1_MD5 || dry_bad=1
sed -i "2i # L1seq ($MODEL_KEY, batch $BATCH x $S, layer 0 only): prefill -> decode1 -> decode2 in original kernel order = P-O + P-L1 + D1-O + D1-L1 + D2-O + D2-L1\n# ($L1_N kernels, all $L1_C copies). Per-piece numbers = per-kernel cumulative stats over the kernel index ranges\n# $(grep -v '^L1SEQ' $T/${TAG}_l1seq_list.log | tr '\t' ' ' | tr '\n' ';') (llm_exp/scripts/s8_extract_l1seq.py --traces <trace dir>). List built by scripts/make_l1seq_list.py." $X/L1seq/launch.sh
while IFS=$'\t' read -r piece nk warp md5 ok path; do
  [ "$ok" = OK ] || continue
  case $piece in P-L1|P-L2) cap=0 ;; *) cap=10000000000 ;; esac
  make_dir $piece $nk $cap $md5 || dry_bad=1
  log "  piece $piece stats warp $warp"
done < <(grep -P '^\S+-\S+\t' $T/${TAG}_piece_lists.log)
[ $dry_bad = 0 ] || { log "some DRY check failed -- stop before the post hook"; exit 1; }
log "preparation finished -- no simulation started"
if [ -n "$POST_HOOK" ]; then
  log "post hook start: $POST_HOOK"
  bash $POST_HOOK >> $LOG 2>&1 && log "post hook finished OK" || log "post hook FAILED (see above)"
fi
