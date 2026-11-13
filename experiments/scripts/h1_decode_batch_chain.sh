#!/bin/bash
# Hypothesis-1 decode batch sweep chain (docs/h1_llama70b_decode_batch_sweep.ko.md): for every batch size, capture the
# layer-0 kernels of the first all-decode step (llmtrace, tracer attached) -> weight-copy check -> post-processing ->
# opcode / LDGSTS trace check -> one summary row. Batches are independent: a failure is logged and the next one runs.
# No simulation is started. Output layout: llm_hypothesis1/<model name>/s<S>_b<BBBB>_decode_layer0/{traces/,run.log,...}
# Usage: h1_decode_batch_chain.sh <model key> <model dir name> <S> <max_tokens> <max_batched_tokens> <kv bytes> <B> [<B> ...]
set -u
MODEL_KEY=$1; MODEL_NAME=$2; S=$3; MAXTOK=$4; MAXB=$5; KV=$6; shift 6
H=/home/user/accel-sim-docker/llm_exp
ROOT=$H/llm_hypothesis1/$MODEL_NAME
CROOT=/accel-sim/llm_exp/llm_hypothesis1/$MODEL_NAME          # the same directory inside the containers
case $MODEL_KEY in
  llama70b) HF_DIR=/home/user/hf_cache/Llama-3.3-70B-Instruct-L3 ;;
  deepseek) HF_DIR=/home/user/hf_cache/DeepSeek-R1-Distill-Llama-8B ;;
  qwen)     HF_DIR=/home/user/hf_cache/Qwen2.5-7B ;;
  mixtral)  HF_DIR=/home/user/hf_cache/Mixtral-8x7B-v0.1-L3 ;;
  *) echo "unknown model key $MODEL_KEY"; exit 1 ;;
esac
OPCODES=$H/llm_sim_tree_pwcstat_l1fwd/gpu-simulator/ISA_Def/ampere_opcode.h
TRACER=/accel-sim/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so
PP=/accel-sim/accel-sim-framework/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing
mkdir -p $ROOT
LOG=$ROOT/chain.log
SUM=$ROOT/SUMMARY.tsv
log(){ echo "$(date '+%F %T') [h1_chain] $*" | tee -a $LOG; }
exec {LK}>$ROOT/.chain.lock
flock -n $LK || { echo "chain already running"; exit 1; }
[ -s $SUM ] || printf 'config\tbatch\tcapture_step\ttotal_steps\tseq_min\tseq_max\twindow\tn_kernels\twarp_insts\ttrace_bytes\ttraceg_bytes\tcapture_s\tpostprocess_s\tchecks\n' > $SUM
log "sweep start: model=$MODEL_KEY S=$S max_tokens=$MAXTOK max_batched=$MAXB kv=$KV batches=$*"

for B in "$@"; do
  CFG=s${S}_b$(printf '%04d' $B)_decode_layer0
  T=$ROOT/$CFG; CT=$CROOT/$CFG
  if [ -e $T/exit_code.txt ]; then log "$CFG: already captured (exit_code.txt present) -- skipping"; continue; fi
  mkdir -p $T
  free_gb=$(df -BG --output=avail $H | tail -1 | tr -dc '0-9')
  [ "$free_gb" -ge 20 ] || { log "$CFG: disk free ${free_gb} GB < 20 GB -- stop"; exit 1; }
  log "$CFG: capture start (disk free ${free_gb} GB)"
  t0=$SECONDS
  docker exec llmtrace bash -c "
cd /accel-sim/llm_exp/scripts
mkdir -p $CT
env CUDA_INJECTION64_PATH=$TRACER ACTIVE_FROM_START=0 USER_DEFINED_FOLDERS=1 TRACES_FOLDER=$CT NOBANNER=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  /venv/bin/python h1_decode_layer_capture.py $CT $MODEL_KEY $B $S $MAXTOK $MAXB $KV > $CT/run.log 2>&1
echo \"EXIT=\$?\" > $CT/exit_code.txt"
  cap_s=$((SECONDS - t0))
  if ! grep -q '^EXIT=0' $T/exit_code.txt || ! grep -q '^H1 CAPTURE PASS' $T/run.log; then
    log "$CFG: capture FAILED ($(cat $T/exit_code.txt 2>/dev/null); $(grep -m1 '^H1 CAPTURE' $T/run.log); $(grep -m1 -i 'out of memory\|Error' $T/run.log | cut -c1-160)) -- next batch"
    printf '%s\t%s\tFAILED\n' "$CFG" "$B" >> $SUM
    continue
  fi
  log "$CFG: capture PASS in ${cap_s}s: $(grep '^window warp insts' $T/run.log)"
  grep '^CHECK' $T/run.log | sed 's/^/    /' | while read -r ln; do log "$ln"; done

  python3 $H/scripts/check_weight_copies.py $HF_DIR $T/traces > $T/h1_check_weight_copies.log 2>&1
  wc_ok=$(grep -q '^WEIGHT COPIES PASS' $T/h1_check_weight_copies.log && echo PASS || echo FAIL)
  log "$CFG: weight copies $wc_ok"

  log "$CFG: post-processing start"
  docker exec llmtrace bash -c "
cd $CT/traces
t0=\$SECONDS
nice -n 10 $PP ./kernelslist_ctx_* > ../h1_postprocess.log 2>&1
echo \"EXIT=\$? SECONDS=\$((SECONDS-t0))\" > ../h1_postprocess_exit.txt"
  pp_line=$(cat $T/h1_postprocess_exit.txt)
  pp_s=$(echo "$pp_line" | sed 's/.*SECONDS=//')
  if ! grep -q '^EXIT=0' $T/h1_postprocess_exit.txt; then
    log "$CFG: post-processing FAILED ($pp_line) -- next batch"
    printf '%s\t%s\tPP_FAILED\n' "$CFG" "$B" >> $SUM
    continue
  fi
  log "$CFG: post-processing done ($pp_line)"

  first=$(python3 -c "import json;print(json.load(open('$T/capture_info.json'))['window'][0])")
  last=$(python3 -c "import json;print(json.load(open('$T/capture_info.json'))['window'][1])")
  nice -n 10 python3 $H/scripts/s6_check_traces.py $T/traces $OPCODES $T/h1_postprocess.log $T/h1_postprocess_exit.txt $first $last \
    > $T/h1_check_traces_${first}_${last}.log 2>&1
  tr_ok=$(grep -q '^S6 TRACES PASS' $T/h1_check_traces_${first}_${last}.log && echo PASS || echo FAIL)
  log "$CFG: trace checks $tr_ok ($(grep '^S6 TRACES' $T/h1_check_traces_${first}_${last}.log))"

  python3 - "$T" "$CFG" "$B" "$cap_s" "$pp_s" "$wc_ok" "$tr_ok" >> $SUM <<'EOF'
import glob, json, os, sys
t, cfg, b, cap_s, pp_s, wc, tr = sys.argv[1:8]
j = json.load(open(os.path.join(t, "capture_info.json")))
traceg = sum(os.path.getsize(p) for p in glob.glob(os.path.join(t, "traces", "*.traceg.xz")))
print("\t".join(str(x) for x in [cfg, b, j["capture_step"], j["total_steps"], j["seq_lens_min"], j["seq_lens_max"],
      f"{j['window'][0]}-{j['window'][1]}", j["n_window_kernels"], j["window_warp_insts"], j["trace_bytes"], traceg,
      cap_s, pp_s, f"capture=PASS weights={wc} traces={tr}"]))
EOF
  log "$CFG: done"
done
log "sweep finished -- no simulation started"
