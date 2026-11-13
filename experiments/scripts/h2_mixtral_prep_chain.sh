#!/bin/bash
# H2 (docs/h2_mixtral8x7b_prep.ko.md): prepare Mixtral-8x7B-v0.1 for the decode layer-0 batch sweep and run it after the
# Llama-70B sweep releases the GPU. Stages (each logged to llm_hypothesis1/Mixtral-8x7B-v0.1/prep.log, stop on failure):
#   1 download the shards that hold layers 0..2 + embed/norm/lm_head (1, 2, 19 of 19) + config/tokenizer files -> temp dir
#   2 repackage into a 3-layer checkpoint (scripts/s17_make_truncated_ckpt.py) -> ~/hf_cache/Mixtral-8x7B-v0.1-L3
#   3 tokenizer check inside llmtrace (no GPU): BOS <s> == 1, prompt text long enough for 1024 distinct windows
#   4 wait until the Llama-70B H1 chain logs "sweep finished" (GPU hand-over)
#   5 smoke without the tracer (scripts/s17_smoke_70b.py, model-generic): 3 layers, b4 S=256 3 tokens, GPU peak
#   6 the H1 batch sweep: scripts/h1_decode_batch_chain.sh mixtral Mixtral-8x7B-v0.1 256 32 16384 4 GiB 1 4 16 64 256 1024
# The temp download is NOT deleted (deletions need the user's instruction). No simulation is started.
set -u
H=/home/user/accel-sim-docker/llm_exp
HF=/home/user/hfvenv/bin/hf
export HF_HOME=/home/user/hf_cache
REPO=mistralai/Mixtral-8x7B-v0.1
TMP=/home/user/hf_cache/_tmp_mixtral8x7b
OUT=/home/user/hf_cache/Mixtral-8x7B-v0.1-L3
CONT_OUT=/hf_cache/Mixtral-8x7B-v0.1-L3
ROOT=$H/llm_hypothesis1/Mixtral-8x7B-v0.1
CROOT=/accel-sim/llm_exp/llm_hypothesis1/Mixtral-8x7B-v0.1
LLAMA_LOG=$H/llm_hypothesis1/Llama-3.3-70B-Instruct/chain.log
mkdir -p $ROOT
LOG=$ROOT/prep.log
log(){ echo "$(date '+%F %T') [h2_prep] $*" | tee -a $LOG; }
exec {LK}>$ROOT/.prep.lock
flock -n $LK || { echo "prep chain already running"; exit 1; }

# ---------- 1 download ----------
if [ -s $OUT/model.safetensors.index.json ] && grep -q '^TRUNCATE PASS' $ROOT/truncate.log 2>/dev/null; then
  log "stage 1-2 skipped: $OUT already holds a verified truncated checkpoint"
else
  free_gb=$(df -BG --output=avail /home/user | tail -1 | tr -dc '0-9')
  [ "$free_gb" -ge 30 ] || { log "disk free ${free_gb} GB < 30 GB -- stop before download"; exit 1; }
  log "stage 1: download $REPO shards 1,2,19 + small files -> $TMP (disk free ${free_gb} GB)"
  t0=$SECONDS
  # File names are positional arguments: `hf download REPO FILE...`. (First attempt used `--include FILE...`; the CLI took
  # only the first name as the include pattern, then ignored it because the rest were explicit file names -> config.json
  # was not fetched. 2026-11-09)
  $HF download $REPO config.json generation_config.json tokenizer.json tokenizer.model tokenizer_config.json \
      special_tokens_map.json model.safetensors.index.json model-00001-of-00019.safetensors model-00002-of-00019.safetensors \
      model-00019-of-00019.safetensors --local-dir $TMP > $ROOT/download.log 2>&1
  rc=$?
  log "stage 1: hf download EXIT=$rc in $((SECONDS-t0))s; $(du -sh $TMP 2>/dev/null | cut -f1) on disk"
  [ $rc = 0 ] || { log "download failed -- stop"; exit 1; }
  for f in config.json tokenizer.json model.safetensors.index.json model-00001-of-00019.safetensors model-00002-of-00019.safetensors model-00019-of-00019.safetensors; do
    [ -s $TMP/$f ] || { log "missing $TMP/$f -- stop"; exit 1; }; done
  ls -l $TMP/*.safetensors | awk '{print "    " $5, $9}' | while read -r ln; do log "$ln"; done

  # ---------- 2 repackage ----------
  log "stage 2: truncate to layers 0..2 -> $OUT"
  python3 $H/scripts/s17_make_truncated_ckpt.py $TMP $OUT --layers 3 > $ROOT/truncate.log 2>&1
  grep -q '^TRUNCATE PASS' $ROOT/truncate.log || { log "truncate FAILED: $(grep -m1 'FAIL\|Error' $ROOT/truncate.log)"; exit 1; }
  log "stage 2: $(grep '^TRUNCATE' $ROOT/truncate.log); $(du -sh $OUT | cut -f1)"
  grep '^CHECK\|^wrote' $ROOT/truncate.log | while read -r ln; do log "    $ln"; done
fi

# ---------- 3 tokenizer check (container, no GPU) ----------
log "stage 3: tokenizer check in llmtrace"
docker exec -i llmtrace /venv/bin/python - > $ROOT/tokenizer_check.log 2>&1 <<EOF
import json
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast.from_pretrained("$CONT_OUT")
bos = tok.convert_tokens_to_ids("<s>")
ids = tok(open("/hf_cache/DeepSeek-R1-Distill-Llama-8B/README.md", encoding="utf-8").read(), add_special_tokens=False).input_ids
cfg = json.load(open("$CONT_OUT/config.json"))
print("bos_id", bos, "config bos", cfg["bos_token_id"], "num_hidden_layers", cfg["num_hidden_layers"], "readme_tokens", len(ids))
ok = bos == 1 == cfg["bos_token_id"] and cfg["num_hidden_layers"] == 3 and len(ids) >= 255 + 1023
print("TOKENIZER CHECK", "PASS" if ok else "FAIL")
EOF
grep -q '^TOKENIZER CHECK PASS' $ROOT/tokenizer_check.log || { log "tokenizer check FAILED: $(cat $ROOT/tokenizer_check.log | tail -3)"; exit 1; }
log "stage 3: $(grep '^bos_id' $ROOT/tokenizer_check.log)"

# ---------- 4 GPU hand-over ----------
log "stage 4: waiting for the Llama-70B sweep to finish ($LLAMA_LOG)"
until grep -q 'sweep finished' $LLAMA_LOG 2>/dev/null; do sleep 60; done
log "stage 4: Llama-70B sweep finished -- GPU free"

# ---------- 5 smoke without tracer ----------
log "stage 5: smoke (no tracer) b4 S=256 3 tokens"
docker exec llmtrace bash -c "cd /accel-sim/llm_exp/scripts && VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  /venv/bin/python s17_smoke_70b.py $CONT_OUT 4 256 3 3 1 > $CROOT/smoke_b4.log 2>&1; echo EXIT=\$? >> $CROOT/smoke_b4.log"
grep -q '^S17 SMOKE PASS' $ROOT/smoke_b4.log || { log "smoke FAILED: $(grep -m1 'S17 SMOKE\|Error\|EXIT' $ROOT/smoke_b4.log)"; exit 1; }
log "stage 5: $(grep '^S17 SMOKE' $ROOT/smoke_b4.log); $(grep '^decoder layers' $ROOT/smoke_b4.log)"

# ---------- 6 batch sweep ----------
log "stage 6: H1 batch sweep start (mixtral)"
bash $H/scripts/h1_decode_batch_chain.sh mixtral Mixtral-8x7B-v0.1 256 32 16384 4294967296 1 4 16 64 256 1024
log "stage 6: H1 chain returned $?"
log "prep chain finished -- temp download $TMP kept ($(du -sh $TMP 2>/dev/null | cut -f1)); no simulation started"
