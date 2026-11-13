#!/bin/bash
# H3 (docs/s27_prefill_capture_and_decode_plan.ko.md section 4.3): prepare a 3-layer truncated Qwen3 checkpoint for tracing.
#   h3_qwen3_prep_chain.sh moe    -> Qwen/Qwen3-30B-A3B  shards 1,2,16 of 16 (layers 0..2 + embed, norm, lm_head) -> ~/hf_cache/Qwen3-30B-A3B-L3
#   h3_qwen3_prep_chain.sh dense  -> Qwen/Qwen3-32B      shards 1,2,17 of 17                                       -> ~/hf_cache/Qwen3-32B-L3
# Stages (logged to llm_hypothesis1/<model>/prep.log, stop on failure): 1 download the needed shards + small files to a temp
# dir, 2 repackage with scripts/s17_make_truncated_ckpt.py (--layers 3), 3 tokenizer check inside llmtrace (no GPU).
# Same recipe as h2_mixtral_prep_chain.sh. The temp download is NOT deleted (deletions need the user's instruction).
set -u
KIND=${1:?moe|dense}
H=/home/user/accel-sim-docker/llm_exp
HF=/home/user/hfvenv/bin/hf
export HF_HOME=/home/user/hf_cache
case $KIND in
  moe)   REPO=Qwen/Qwen3-30B-A3B; NAME=Qwen3-30B-A3B; SH="model-00001-of-00016.safetensors model-00002-of-00016.safetensors model-00016-of-00016.safetensors" ;;
  dense) REPO=Qwen/Qwen3-32B;     NAME=Qwen3-32B;     SH="model-00001-of-00017.safetensors model-00002-of-00017.safetensors model-00017-of-00017.safetensors" ;;
  *) echo "unknown kind $KIND"; exit 1 ;;
esac
TMP=/home/user/hf_cache/_tmp_$(echo $NAME | tr 'A-Z-' 'a-z_')
OUT=/home/user/hf_cache/$NAME-L3
CONT_OUT=/hf_cache/$NAME-L3
ROOT=$H/llm_hypothesis1/$NAME
mkdir -p $ROOT
LOG=$ROOT/prep.log
log(){ echo "$(date '+%F %T') [h3_prep $KIND] $*" | tee -a $LOG; }
exec {LK}>$ROOT/.prep.lock
flock -n $LK || { echo "prep chain already running"; exit 1; }

if [ -s $OUT/model.safetensors.index.json ] && grep -q '^TRUNCATE PASS' $ROOT/truncate.log 2>/dev/null; then
  log "stage 1-2 skipped: $OUT already holds a verified truncated checkpoint"
else
  free_gb=$(df -BG --output=avail /home/user | tail -1 | tr -dc '0-9')
  [ "$free_gb" -ge ${DISK_MIN:-30} ] || { log "disk free ${free_gb} GB < ${DISK_MIN:-30} GB -- stop before download"; exit 1; }
  log "stage 1: download $REPO shards [$SH] + small files -> $TMP (disk free ${free_gb} GB)"
  t0=$SECONDS
  $HF download $REPO config.json generation_config.json tokenizer.json tokenizer_config.json merges.txt vocab.json \
      model.safetensors.index.json $SH --local-dir $TMP > $ROOT/download.log 2>&1
  rc=$?
  log "stage 1: hf download EXIT=$rc in $((SECONDS-t0))s; $(du -sh $TMP 2>/dev/null | cut -f1) on disk"
  [ $rc = 0 ] || { log "download failed -- stop: $(tail -n 2 $ROOT/download.log)"; exit 1; }
  for f in config.json tokenizer.json model.safetensors.index.json $SH; do [ -s $TMP/$f ] || { log "missing $TMP/$f -- stop"; exit 1; }; done
  ls -l $TMP/*.safetensors | awk '{print "    " $5, $9}' | while read -r ln; do log "$ln"; done
  log "stage 2: truncate to layers 0..2 -> $OUT"
  python3 $H/scripts/s17_make_truncated_ckpt.py $TMP $OUT --layers 3 > $ROOT/truncate.log 2>&1
  grep -q '^TRUNCATE PASS' $ROOT/truncate.log || { log "truncate FAILED: $(grep -m1 'FAIL\|Error' $ROOT/truncate.log)"; exit 1; }
  log "stage 2: $(grep '^TRUNCATE' $ROOT/truncate.log); $(du -sh $OUT | cut -f1)"
  grep '^CHECK\|^wrote' $ROOT/truncate.log | while read -r ln; do log "    $ln"; done
fi

log "stage 3: tokenizer check in llmtrace (Qwen3: no BOS is prepended; eos/pad <|endoftext|> = 151643)"
docker exec -i llmtrace /venv/bin/python - > $ROOT/tokenizer_check.log 2>&1 <<PYEOF
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$CONT_OUT")
eot = tok.convert_tokens_to_ids("<|endoftext|>")
ids = tok(open("/hf_cache/DeepSeek-R1-Distill-Llama-8B/README.md", encoding="utf-8").read(), add_special_tokens=False).input_ids
cfg = json.load(open("$CONT_OUT/config.json"))
print("endoftext_id", eot, "config bos", cfg.get("bos_token_id"), "num_hidden_layers", cfg["num_hidden_layers"], "readme_tokens", len(ids), "arch", cfg["architectures"])
ok = eot == 151643 and cfg["num_hidden_layers"] == 3 and len(ids) >= 1023 + 3 * 150
print("TOKENIZER CHECK", "PASS" if ok else "FAIL")
PYEOF
grep -q '^TOKENIZER CHECK PASS' $ROOT/tokenizer_check.log || { log "tokenizer check FAILED: $(tail -n 3 $ROOT/tokenizer_check.log)"; exit 1; }
log "stage 3: $(grep '^endoftext_id' $ROOT/tokenizer_check.log)"
log "prep done: $OUT ($(du -sh $OUT | cut -f1)); temp shards kept in $TMP"
