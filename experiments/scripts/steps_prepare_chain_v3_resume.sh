#!/bin/bash
# steps_prepare_chain_v3_resume.sh -- step 17: resume chain v3 AFTER a finished capture + copy check + post-processing
# (the host-side chain was killed with the session on 2026-11-03 during the window checks). Same arguments as v3;
# requires <trace dir>/exit_code.txt EXIT=0 and <tag>_postprocess_exit.txt EXIT=0, then runs: per-window trace checks,
# piece lists, L1seq list, experiment dirs with DRY checks, optional post hook. No capture, no simulation.
# steps_prepare_chain_v3.sh -- host side, step 17 (v3 = v2 with the latest-tree launch template and the H2D copy count passed to the DRY check; v2 kept intact) (docs/s9_b8s256_steps.ko.md sections 5 and 10). v2 = v1 + the instruction
# cap is decided per piece from the stats warp count: pieces whose thread instructions cannot exceed 1e10 (warp x 32 < 1e10)
# keep the standard cap 10000000000; only pieces that must exceed it get cap 0 + CAP_EXCEPTION.txt. One step-piece capture from
# start to simulation-ready: capture (llmtrace, tracer attached) -> weight-copy check -> post-processing -> per-window
# trace checks -> piece lists -> L1seq list -> experiment dirs (L1seq + pieces, llm_sim_tree_fast) with DRY checks ->
# optional post hook (e.g. kit build). No simulation is started.
# Usage: steps_prepare_chain_v2.sh <model key> <exp dir name> <trace dir name> <batch> <S> <max tokens> <traced layers 1|2>
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
grep -q "^EXIT=0" $T/exit_code.txt && grep -q "^STEPS CAPTURE PASS" $T/run.log || { log "no finished capture in $T -- stop"; exit 1; }
grep -q "^EXIT=0" $T/${TAG}_postprocess_exit.txt || { log "post-processing not finished in $T -- stop"; exit 1; }
grep -q "^WEIGHT COPIES PASS" $T/${TAG}_check_weight_copies.log || { log "weight-copy check not passed -- stop"; exit 1; }
log "RESUME after capture/copy-check/post-processing: $(cat $T/${TAG}_postprocess_exit.txt)"
last_piece=layer0; [ "$LAYERS" = 2 ] && last_piece=layer1
for step in $(awk -F'\t' 'NR>1 {print $1}' $T/regions.tsv | uniq); do
  a=$(awk -F'\t' -v s=$step '$1==s && $2=="prep" {print $3}' $T/regions.tsv)
  b=$(awk -F'\t' -v s=$step -v p=$last_piece '$1==s && $2==p {print $4}' $T/regions.tsv)
  c=$(awk -F'\t' -v s=$step '$1==s && $2=="tail" {print $3}' $T/regions.tsv)
  d=$(awk -F'\t' -v s=$step '$1==s && $2=="tail" {print $4}' $T/regions.tsv)
  for w in "$a $b" "$c $d"; do
    set -- $w
    nice -n 10 python3 $H/scripts/s6_check_traces.py $T/traces $H/llm_sim_tree_pwcstat_l1fwd/gpu-simulator/ISA_Def/ampere_opcode.h \
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
of 2026-09-19 ("b8s256으로 Qwen이랑 deepseek 모두 만들어줘" / "b4s256으로도 트레이스 뽑아줘": batch reduced from 32 so that whole prefill layers can be
simulated; same rule as the b32 L1seq / P-L2 and the step-7 exceptions). This piece's stats warp count x 32 exceeds the standing 1e10 cap
(thread instructions), so the cap would truncate the prefill layer and the piece must run uncapped (v2 chain: pieces that
fit under 1e10 keep the standard cap).
If this exception is not intended, delete this file before launching (the launcher then refuses to start).
EOT
}
cap_for(){ # $1 stats warp count -> standard cap when warp x 32 (upper bound of thread instructions) stays below 1e10, else 0
  if [ $(( $1 * 32 )) -lt 10000000000 ]; then echo 10000000000; else echo 0; fi; }
make_dir(){ # $1 piece, $2 n_kernels, $3 cap, $4 trace md5
  mkdir -p $X/$1
  sed -e "s|@DIR@|$1|g" -e "s|@EXP@|$EXP|g" -e "s|@TRACE_DIR@|$TDIR|g" \
      -e "s|@N_KERNELS@|$2|g" -e "s|@MAX_INSN@|$3|g" -e "s|@MEASURE@|1|g" -e "s|@TRACE_MD5@|$4|g" -e "s|@MIN_COPIES@|$want_c|g" \
      -e "s|@REF_TH@|16|g" -e "s|@ARM_TH@|8|g" -e "s|@PARALLEL@|1|g" \
      $H/scripts/steps_launch_template_latest.sh > $X/$1/launch.sh
  chmod +x $X/$1/launch.sh
  [ "$3" = 0 ] && cap_exception $X/$1
  docker exec -u 1035:1035 -e DRY=1 mcm-accelsim bash /accel-sim/llm_exp/experiments/$EXP/$1/launch.sh > /dev/null 2>&1
  if grep -q 'DRY run finished' $X/$1/launch.log; then log "dir $1: $2 kernels, cap $3, DRY OK"; return 0
  else log "dir $1: DRY FAILED ($(tail -1 $X/$1/launch.log))"; return 1; fi
}
dry_bad=0
L1_WARP=$(awk -F'\t' '$1 ~ /^(P|D[0-9]+)-(O|L1)$/ {s+=$3} END{print s+0}' $T/${TAG}_piece_lists.log)
L1_CAP=$(cap_for $L1_WARP)
log "L1seq stats warp $L1_WARP (x32 = $((L1_WARP*32)) thread upper bound) -> cap $L1_CAP"
make_dir L1seq $L1_N $L1_CAP $L1_MD5 || dry_bad=1
sed -i "2i # L1seq ($MODEL_KEY, batch $BATCH x $S, layer 0 only): prefill -> decode1 -> decode2 in original kernel order = P-O + P-L1 + D1-O + D1-L1 + D2-O + D2-L1\n# ($L1_N kernels, all $L1_C copies). Per-piece numbers = per-kernel cumulative stats over the kernel index ranges\n# $(grep -v '^L1SEQ' $T/${TAG}_l1seq_list.log | tr '\t' ' ' | tr '\n' ';') (llm_exp/scripts/s8_extract_l1seq.py --traces <trace dir>). List built by scripts/make_l1seq_list.py." $X/L1seq/launch.sh
while IFS=$'\t' read -r piece nk warp md5 ok path; do
  [ "$ok" = OK ] || continue
  cap=$(cap_for $warp)
  make_dir $piece $nk $cap $md5 || dry_bad=1
  log "  piece $piece stats warp $warp (x32 = $((warp*32))) -> cap $cap"
done < <(grep -P '^\S+-\S+\t' $T/${TAG}_piece_lists.log)
[ $dry_bad = 0 ] || { log "some DRY check failed -- stop before the post hook"; exit 1; }
log "preparation finished -- no simulation started"
if [ -n "$POST_HOOK" ]; then
  log "post hook start: $POST_HOOK"
  bash $POST_HOOK >> $LOG 2>&1 && log "post hook finished OK" || log "post hook FAILED (see above)"
fi

