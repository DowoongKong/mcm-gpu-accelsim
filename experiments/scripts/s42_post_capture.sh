#!/bin/bash
# s42_post_capture.sh <model dir name> <tag: ctx12k|ctx16k> <min ctx> <expected num_blocks>
# After a decode-sweep capture chain (docs/s42_decode_gen_sweep.ko.md section 7): chain PASS check -> sweep gate against every earlier
# point -> prefill-extrapolated + KV-replay placement map (no decode-own ref) -> kernelslist_decode.g -> hash injection kit.
# Stops at the first failing step. Log: analysis/s42_sweep/post_capture.log (+ per-model build logs in the map dir).
set -u
M=$1; TAG=$2; MINCTX=$3; NB=$4
H=/home/user/accel-sim-docker/llm_exp; R=$H/llm_hypothesis1/$M; A=$H/analysis/s34_prefill_placement; L=$H/analysis/s42_sweep/post_capture.log
CFGN=s4096_b0032_decode_${TAG}_layer0_v2; CAP=$R/$CFGN; MAPD=$A/$M/$CFGN; REF4=$R/s4096_b0032_decode_layer0_v2/sim_noprefill/ref_mcm/gpgpusim.config
mkdir -p $(dirname $L); log(){ echo "$(date '+%F %T') [$M $TAG] $*" >> $L; }
fail(){ log "STOP: $*"; exit 1; }
# 1. chain result
grep -q "$CFGN: trace checks.*PASS" $R/chain_v2.log && grep -q "$CFGN: done" $R/chain_v2.log || fail "chain not PASS/done"
grep -q '^H1 CAPTURE PASS' $CAP/run.log && grep -q '^WEIGHT COPIES PASS' $CAP/h1_check_weight_copies.log || fail "capture/weight-copy PASS line missing"
# 2. sweep gate against all earlier points of the sweep
OLDS=$R/s4096_b0032_decode_layer0_v2
for t in ctx8k ctx12k; do [ "$t" = "$TAG" ] && break; OLDS="$OLDS $R/s4096_b0032_decode_${t}_layer0_v2"; done
# s44: SWEEP_OLDS = space-separated tags of the earlier points to compare with ("base" = s4096_b0032_decode_layer0_v2), e.g. forced-routing
# captures compare with the natural captures of the same or an earlier capture step.
[ -n "${SWEEP_OLDS:-}" ] && { OLDS=""; for t in $SWEEP_OLDS; do [ $t = base ] && OLDS="$OLDS $R/s4096_b0032_decode_layer0_v2" || OLDS="$OLDS $R/s4096_b0032_decode_${t}_layer0_v2"; done; }
python3 $H/scripts/s42_sweep_gate.py $CAP $NB $MINCTX $OLDS > /dev/null 2>&1; log "sweep gate: $(tail -n1 $CAP/SWEEP_GATE.txt 2>/dev/null) ($(grep -c '^PASS' $CAP/SWEEP_GATE.txt) PASS lines)"
grep -q '^SWEEP GATE: PASS' $CAP/SWEEP_GATE.txt || fail "sweep gate FAIL: $(grep '^FAIL' $CAP/SWEEP_GATE.txt | cut -c1-200 | tr '\n' ' ')"
# 3. placement map (reuses the 4K build's prefill prediction and split-K overrides)
P=$(grep -m1 -o "predict: [^ ]*" $A/$M/s4096_b0032_decode_layer0_v2/build.log | sed 's#predict: ##; s#/traces/kernelslist.g##')
[ -d "$P" ] || fail "prefill capture dir not found from the 4K build log"
mkdir -p $MAPD && cp -r $A/$M/s4096_b0032_decode_layer0_v2/pred_prefill $MAPD/ && { [ -d $A/$M/s4096_b0032_decode_layer0_v2/vb_override ] && cp -r $A/$M/s4096_b0032_decode_layer0_v2/vb_override $MAPD/ || true; }
log "map build start (PRE $P)"
CFG_FROM=$REF4 nice -n 5 bash $H/scripts/s41_build_map_noref.sh $CAP $P $MAPD 2 > $MAPD/build_driver.log 2>&1
grep -q '^GATES: PASS' $MAPD/GATES.txt 2>/dev/null || fail "map gates not PASS: $(cat $MAPD/GATES.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
log "map: $(grep -c '^D ' $MAPD/placement_prefill_chunked.map) D lines, $(tail -n1 $MAPD/GATES.txt); $(grep -m1 'kv replay' $MAPD/build.log | grep -o 'steps [0-9]*')"
# 4. decode-window kernel list (all HtoD copy lines + the window kernels, as h1_make_decode_arms_s31.sh)
read WA WB <<< "$(python3 -c "import json;w=json.load(open('$CAP/capture_info.json'))['windows']['decode_layer0'];print(w[0],w[1])")"
grep -E "MemcpyHtoD" $CAP/traces/kernelslist.g > $CAP/traces/kernelslist_decode.g
grep -vE "MemcpyHtoD" $CAP/traces/kernelslist.g | awk -v a=$WA -v b=$WB '{match($0,/kernel-([0-9]+)-/,m); if (m[1]+0>=a && m[1]+0<=b) print}' >> $CAP/traces/kernelslist_decode.g
NK=$(grep -vc MemcpyHtoD $CAP/traces/kernelslist_decode.g); [ $NK = $((WB-WA+1)) ] || fail "kernelslist_decode.g holds $NK kernels, window $WA-$WB"
# 5. hash injection kit
OUT=$(CFG_FROM=$REF4 MEM_GB=256 bash $H/scripts/make_inject_hash_kit_s31.sh $CAP $MAPD/placement_prefill_chunked.map 2>&1 | tail -n1)
echo "$OUT" | grep -q '^inject kit:' || fail "kit build: $OUT"
log "DONE $OUT"
