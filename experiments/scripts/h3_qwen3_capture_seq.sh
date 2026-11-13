#!/bin/bash
# Step 27 section 4.6 (docs/s27_prefill_capture_and_decode_plan.ko.md): Qwen3 MoE then dense capture on the one GPU that
# llmtrace sees. For each model: smoke without the tracer (s17_smoke_70b.py, 3 layers, b4 S=256 3 tokens), then the
# capture chain h1_capture_v2_chain_b.sh with the Mixtral settings (b4 x S1024, max_tokens 32, max_batched 4096, KV 4 GiB,
# prefill_layer0 + decode_layer0 windows, INIT_FIRST=0, DISK_MIN=10). The dense model starts as soon as the MoE chain
# reports its capture step finished (GPU free); the MoE chain keeps post-processing / S6 in parallel (per-model lock).
# The chain file is only executed, never edited. Usage: h3_qwen3_capture_seq.sh   (run on the host, detached)
set -u
H=/home/user/accel-sim-docker/llm_exp
LOG=$H/llm_hypothesis1/qwen3_capture_seq.log
log(){ echo "$(date '+%F %T') [h3_qwen3_seq] $*" | tee -a $LOG; }

smoke(){  # <model key> <model dir name> <container model dir>
  local root=$H/llm_hypothesis1/$2 croot=/accel-sim/llm_exp/llm_hypothesis1/$2
  mkdir -p $root
  log "$1: smoke (no tracer) b4 S=256 3 tokens"
  docker exec llmtrace bash -c "cd /accel-sim/llm_exp/scripts && VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
    /venv/bin/python s17_smoke_70b.py $3 4 256 3 3 none > $croot/smoke_b4.log 2>&1; echo EXIT=\$? >> $croot/smoke_b4.log"
  if ! grep -q '^S17 SMOKE PASS' $root/smoke_b4.log; then
    log "$1: smoke FAILED: $(grep -m1 'S17 SMOKE\|Error\|EXIT' $root/smoke_b4.log | cut -c1-200)"; return 1; fi
  log "$1: $(grep '^S17 SMOKE' $root/smoke_b4.log); $(grep '^decoder layers' $root/smoke_b4.log)"
}

capture(){  # <model key> <model dir name>; returns once the chain has passed its capture step (chain keeps running)
  local clog=$H/llm_hypothesis1/$2/chain_v2.log n0
  touch $clog; n0=$(wc -l < $clog)
  log "$1: capture chain start"
  INIT_FIRST=0 PREFILL_WINDOW=layer0 NAME=prefill_l0_decode_layer0 DISK_MIN=10 \
    setsid nohup bash $H/scripts/h1_capture_v2_chain_b.sh $1 $2 1024 32 4096 4294967296 4 \
    > $H/llm_hypothesis1/$2/chain_v2_launch_20261111_prefill_s1024_b4.out 2>&1 < /dev/null &
  until tail -n +$((n0 + 1)) $clog | grep -qE 'capture (PASS|FAILED)|-- stop|already running'; do sleep 60; done
  log "$1: $(tail -n +$((n0 + 1)) $clog | grep -E 'capture (PASS|FAILED)|-- stop|already running' | head -1 | cut -c1-260)"
  tail -n +$((n0 + 1)) $clog | grep -q 'capture PASS'
}

log "start (disk free $(df -BG --output=avail $H | tail -1 | tr -dc '0-9') GB)"
smoke qwen3moe Qwen3-30B-A3B /hf_cache/Qwen3-30B-A3B-L3 && capture qwen3moe Qwen3-30B-A3B \
  || log "qwen3moe did not reach capture PASS -- continuing with the dense model"
smoke qwen3dense Qwen3-32B /hf_cache/Qwen3-32B-L3 && capture qwen3dense Qwen3-32B \
  || log "qwen3dense did not reach capture PASS"
log "GPU part done; the chains continue with post-processing and S6 (see each model's chain_v2.log)"
