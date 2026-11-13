#!/bin/bash
# Hypothesis-1 capture chain v4 (= v3 + GPU-phase pipelining, docs/s42_decode_gen_sweep.ko.md section 8, 2026-11-13): identical steps and
# log lines, but after the capture and the weight-copy check the post-processing, the S6 trace checks and the SUMMARY row run in a
# background subshell and the chain returns at once ("GPU phase done"), so a queue can start the next capture while the CPU-only
# phases finish. The background subshell inherits the chain lock, so the same model/name cannot be re-captured before it ends.
# v3 header:
# Hypothesis-1 capture chain v3 (= v2 + capture container from $CAPCTR (default llmtrace; llmtrace0 = GPU 0) + S6 in 4 parallel sub-ranges, 2026-11-12 02:55)
# Hypothesis-1 capture chain v2 (docs/s25b_h1_capture_v2_init_window.ko.md): like h1_decode_batch_chain.sh but runs
# scripts/h1_capture_v2.py (tensor address map; initialization windows W1/W2 for the FIRST batch of the list only),
# SASS_DUMP=1 (tracer patch 30), output directories s<S>_b<BBBB>_decode_layer0_v2, and the S6 trace checks per window.
# No simulation is started. Usage:
#   [INIT_FIRST=0] [PREFILL_WINDOW=1 NAME=prefill_decode_layer0] h1_capture_v2_chain_c_v4.sh <model key> <model dir name> <S> <max_tokens> <max_batched_tokens> <kv bytes> <B> [<B> ...]
set -u
MODEL_KEY=$1; MODEL_NAME=$2; S=$3; MAXTOK=$4; MAXB=$5; KV=$6; shift 6
H=/home/user/accel-sim-docker/llm_exp
ROOT=$H/llm_hypothesis1/$MODEL_NAME
CROOT=/accel-sim/llm_exp/llm_hypothesis1/$MODEL_NAME
case $MODEL_KEY in
  llama70b) HF_DIR=/home/user/hf_cache/Llama-3.3-70B-Instruct-L3 ;;
  deepseek) HF_DIR=/home/user/hf_cache/DeepSeek-R1-Distill-Llama-8B ;;
  qwen)     HF_DIR=/home/user/hf_cache/Qwen2.5-7B ;;
  mixtral)  HF_DIR=/home/user/hf_cache/Mixtral-8x7B-v0.1-L3 ;;
  qwen3moe)   HF_DIR=/home/user/hf_cache/Qwen3-30B-A3B-L3 ;;
  qwen3dense) HF_DIR=/home/user/hf_cache/Qwen3-32B-L3 ;;
  *) echo "unknown model key $MODEL_KEY"; exit 1 ;;
esac
OPCODES=$H/llm_sim_tree_pwcstat_l1fwd/gpu-simulator/ISA_Def/ampere_opcode.h
TRACER=/accel-sim/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so
PP=/accel-sim/accel-sim-framework/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing
mkdir -p $ROOT
LOG=$ROOT/chain_v2.log
SUM=$ROOT/SUMMARY_v2.tsv
log(){ echo "$(date '+%F %T') [h1_chain_v2] $*" | tee -a $LOG; }
exec {LK}>$ROOT/.chain_v2_${NAME:-decode_layer0}.lock
flock -n $LK || { echo "chain v2 already running for ${NAME:-decode_layer0}"; exit 1; }
[ -s $SUM ] || printf 'config\tbatch\tinit_windows\tcapture_step\ttotal_steps\tseq_min\tseq_max\twindows\tn_decode_kernels\tdecode_warp_insts\tinit_warp_insts\tkv_warp_insts\ttrace_bytes\ttraceg_bytes\tcapture_s\tpostprocess_s\tchecks\n' > $SUM
log "sweep v2 start: model=$MODEL_KEY S=$S max_tokens=$MAXTOK max_batched=$MAXB kv=$KV batches=$* init_first=${INIT_FIRST:-1} prefill_window=${PREFILL_WINDOW:-0} name=${NAME:-decode_layer0} (chain v4: post-processing in background)"

# CPU-only phases of one capture (v3 lines 72-107), run in the background by the loop below.
post_phase(){ local T=$1 CT=$2 CFG=$3 B=$4 cap_s=$5 wc_ok=$6 pp_line pp_s tr_all W wname first last ok
  log "$CFG: post-processing start"
  docker exec ${CAPCTR:-llmtrace} bash -c "
cd $CT/traces
t0=\$SECONDS
nice -n 10 $PP ./kernelslist_ctx_* > ../h1_postprocess.log 2>&1
echo \"EXIT=\$? SECONDS=\$((SECONDS-t0))\" > ../h1_postprocess_exit.txt"
  pp_line=$(cat $T/h1_postprocess_exit.txt)
  pp_s=$(echo "$pp_line" | sed 's/.*SECONDS=//')
  if ! grep -q '^EXIT=0' $T/h1_postprocess_exit.txt; then
    log "$CFG: post-processing FAILED ($pp_line) -- next batch"
    printf '%s\t%s\tPP_FAILED\n' "$CFG" "$B" >> $SUM
    return 1
  fi
  log "$CFG: post-processing done ($pp_line)"
  tr_all=PASS
  for W in $(python3 -c "import json;w=json.load(open('$T/capture_info.json'))['windows'];print(' '.join(f'{k}:{v[0]}:{v[1]}' for k,v in w.items() if v))"); do
    IFS=: read -r wname first last <<< "$W"
    OPCODES=$OPCODES bash $H/scripts/s6_parallel.sh $T $wname $first $last ${S6_PAR:-4} > /dev/null 2>&1   # v3: 4 parallel sub-ranges, merged log
    ok=$(grep -q '^S6 TRACES PASS' $T/h1_check_traces_${wname}_${first}_${last}.log && echo PASS || echo FAIL)
    [ "$ok" = PASS ] || tr_all=FAIL
    log "$CFG: trace checks [$wname $first-$last] $ok ($(grep '^S6 TRACES' $T/h1_check_traces_${wname}_${first}_${last}.log))"
  done
  python3 - "$T" "$CFG" "$B" "$cap_s" "$pp_s" "$wc_ok" "$tr_all" >> $SUM <<'EOF'
import glob, json, os, sys
t, cfg, b, cap_s, pp_s, wc, tr = sys.argv[1:8]
j = json.load(open(os.path.join(t, "capture_info.json")))
traceg = sum(os.path.getsize(p) for p in glob.glob(os.path.join(t, "traces", "*.traceg.xz")))
w = {k: v for k, v in j["windows"].items() if v}; wi = j["window_warp_insts_by_window"]
print("\t".join(str(x) for x in [cfg, b, j["init_windows"], j["capture_step"], j["total_steps"], j["seq_lens_min"], j["seq_lens_max"],
      ";".join(f"{k}={v[0]}-{v[1]}" for k, v in w.items()), j["n_window_kernels"], wi.get("decode_layer0", 0),
      wi.get("init_layer0", 0), wi.get("kv_zero_fill", 0), j["trace_bytes"], traceg, cap_s, pp_s,
      f"capture=PASS weights={wc} traces={tr}"]))
EOF
  log "$CFG: done"
}

# INIT_FIRST=0 in the environment skips the W1/W2 initialization windows for every batch (they do not depend on the
# batch size; the b1 capture of 2026-11-10 already holds them).
FIRST=${INIT_FIRST:-1}
for B in "$@"; do
  CFG=s${S}_b$(printf '%04d' $B)_${NAME:-decode_layer0}_v2
  T=$ROOT/$CFG; CT=$CROOT/$CFG
  if [ -e $T/exit_code.txt ]; then log "$CFG: already captured -- skipping"; FIRST=0; continue; fi
  mkdir -p $T
  free_gb=$(df -BG --output=avail $H | tail -1 | tr -dc '0-9')
  [ "$free_gb" -ge ${DISK_MIN:-20} ] || { log "$CFG: disk free ${free_gb} GB < ${DISK_MIN:-20} GB -- stop"; exit 1; }
  IW=""; [ "$FIRST" = 1 ] && IW="--init-windows"
  [ "${PREFILL_WINDOW:-0}" = 1 ] && IW="$IW --prefill-window"
  [ "${PREFILL_WINDOW:-0}" = layer0 ] && IW="$IW --prefill-window --prefill-layer0"
  [ -n "${FORCE_ROUTING:-}" ] && IW="$IW --force-routing $FORCE_ROUTING"
  [ -n "${MIN_CTX:-}" ] && IW="$IW --min-context $MIN_CTX"   # s41: capture a later decode step (context >= MIN_CTX)
  log "$CFG: capture start (disk free ${free_gb} GB, $IW)"
  t0=$SECONDS
  docker exec ${CAPCTR:-llmtrace} bash -c "
cd /accel-sim/llm_exp/scripts
mkdir -p $CT
env CUDA_INJECTION64_PATH=$TRACER ACTIVE_FROM_START=0 USER_DEFINED_FOLDERS=1 TRACES_FOLDER=$CT NOBANNER=1 SASS_DUMP=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  /venv/bin/python h1_capture_v2.py $CT $MODEL_KEY $B $S $MAXTOK $MAXB $KV $IW > $CT/run.log 2>&1
echo \"EXIT=\$?\" > $CT/exit_code.txt"
  cap_s=$((SECONDS - t0))
  FIRST=0
  if ! grep -q '^EXIT=0' $T/exit_code.txt || ! grep -q '^H1 CAPTURE PASS' $T/run.log; then
    log "$CFG: capture FAILED ($(cat $T/exit_code.txt 2>/dev/null); $(grep -m1 '^H1 CAPTURE' $T/run.log); $(grep -m1 -i 'out of memory\|Error' $T/run.log | cut -c1-160)) -- next batch"
    printf '%s\t%s\tFAILED\n' "$CFG" "$B" >> $SUM
    continue
  fi
  log "$CFG: capture PASS in ${cap_s}s: $(grep '^windows' $T/run.log | cut -c1-200)"
  grep '^CHECK' $T/run.log | sed 's/^/    /' | while read -r ln; do log "$ln"; done

  python3 $H/scripts/check_weight_copies.py $HF_DIR $T/traces > $T/h1_check_weight_copies.log 2>&1
  wc_ok=$(grep -q '^WEIGHT COPIES PASS' $T/h1_check_weight_copies.log && echo PASS || echo FAIL)
  log "$CFG: weight copies $wc_ok"
  post_phase $T $CT $CFG $B $cap_s $wc_ok > /dev/null 2>&1 < /dev/null &
  log "$CFG: GPU phase done -- post-processing + S6 continue in the background (pid $!)"
done
log "sweep v2 finished -- no simulation started (chain v4: background post-processing may still run)"
