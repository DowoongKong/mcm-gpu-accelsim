#!/bin/bash
# s34_build_prefill_map.sh <decode capture dir> <prefill capture dir (b1 x S, prefill_layer0 window)> <out dir> [P=2]
# Builds <out>/placement_prefill_chunked.map for the decode capture without simulating prefill (docs/s34 section 0):
#   1. s22 predictor (--config = decode ref config) on the prefill window kernels of the prefill capture (skips existing json)
#   2. translate those candidates (weights only) into the decode capture's VA space
#   3. KV replay candidates from the decode capture's recorded schedule (one per step before the decode window)
#   4. decode-own candidate from the decode capture's ref dump (sim/ref_mcm/placement_mcm.map; required)
#   5. s26_make_placement_map.py rule C: prefill weights -> kv steps -> decode-own, --select decode_own, --ref ref map
set -u
DEC=$1; PRE=$2; OUT=$3; P=${4:-2}; H=/home/user/accel-sim-docker/llm_exp; S=$H/scripts
SIMD=$DEC/sim_noprefill; [ -d $SIMD ] || SIMD=$DEC/sim; CFG=$SIMD/ref_mcm/gpgpusim.config; REF=$SIMD/ref_mcm/placement_mcm.map
mkdir -p $OUT; L=$OUT/build.log; log(){ echo "$(date '+%F %T') $*" | tee -a $L; }
[ -s $CFG ] || { log "no decode ref config $CFG"; exit 1; }; [ -s $REF ] || { log "no decode ref map $REF (ref arm not finished)"; exit 1; }
grep -q '^S6 TRACES PASS' $PRE/h1_check_traces_prefill_layer0_*.log 2>/dev/null || { log "prefill window S6 PASS missing in $PRE"; exit 1; }
read A B <<< "$(python3 -c "import json;w=json.load(open('$PRE/capture_info.json'))['windows']['prefill_layer0'];print(w[0],w[1])")"
KL=$PRE/traces/kernelslist.g; [ -s $KL ] || KL=$(ls $PRE/traces/kernelslist*.g | head -1)   # the FULL list (locale collation put kernelslist_decode.g first)
POS=$(grep 'kernel-' $KL | awk -v a=$A -v b=$B '{match($0,/kernel-([0-9]+)-/,m); if (m[1]+0>=a && m[1]+0<=b) print NR}' | tr '\n' ' ')
log "prefill window kernels $A-$B -> positions [$POS] in $KL; config $CFG"
TODO=""; for k in $POS; do [ -s $OUT/pred_prefill/k$k.json ] || TODO="$TODO $k"; done
if [ -n "$TODO" ]; then log "predictor on$TODO (P=$P)"; bash $S/s26_run_predict.sh $PRE $CFG $OUT/pred_prefill $P $TODO >> $L 2>&1; fi
for k in $POS; do [ -s $OUT/pred_prefill/k$k.json ] || { log "predictor output k$k.json missing"; exit 1; }; done
# split-K weight GEMMs among the prefill kernels (grid z > 1 or splitK in the name; FlashAttention split-KV is attention, not a weight
# kernel): their first touch has same-wave ties the predictor cannot decide (s26 V-a: 84 %), so each one is reference-simulated alone
# and its translated dump overrides the prediction (user rule 11-12: accuracy first). Record -> $OUT/splitk_overrides.txt
: > $OUT/splitk_overrides.txt; OVR=""
for k in $POS; do python3 - $OUT/pred_prefill/k$k.json <<'PY' || continue
import json, sys; J = json.load(open(sys.argv[1])); g = J.get('grid') or ''; n = J.get('name') or ''
z = int(g.strip('()').split(',')[2]) if g.count(',') == 2 else 1
gemm = any(t in n for t in ('gemm', 'Gemm', 'cutlass', 'Kernel2', 'xmma', 'wmma', 'fused_moe')) and 'flash' not in n.lower()
sys.exit(0 if gemm and (z > 1 or 'splitk' in n.lower()) else 1)
PY
  log "split-K weight GEMM at k$k: $(python3 -c "import json;J=json.load(open('$OUT/pred_prefill/k$k.json'));print(J.get('grid'),(J.get('name') or '')[:70])") -> single-kernel ref override"
  bash $S/s34_vb_kernel_override.sh $PRE $k $OUT/vb_override/k$k $P >> $L 2>&1 || { log "override sim k$k FAILED"; echo "k$k NONE" >> $OUT/splitk_overrides.txt; continue; }
  python3 $S/s34_translate_dump.py $PRE $DEC $OUT/vb_override/k$k/placement_mcm.map $OUT/override_k$k.map | tee -a $L
  echo "k$k $OUT/override_k$k.map" >> $OUT/splitk_overrides.txt; OVR="$OVR${OVR:+,}$OUT/override_k$k.map"; done
rm -rf $OUT/cand; mkdir -p $OUT/cand
FILES=$(for k in $POS; do echo $OUT/pred_prefill/k$k.json; done | tr '\n' ' ')
python3 $S/s34_translate_candidates.py $PRE $DEC $OUT/cand 1 $FILES | tee -a $L
NA=$(ls $OUT/cand/k*.json | wc -l)
python3 $S/s34_kv_replay_candidates.py $DEC $OUT/cand $((NA+1)) | tee -a $L
NK=$(ls $OUT/cand/k*.json | wc -l); NB=$((NK-NA))
python3 $S/s34_refmap_candidate.py $REF $OUT/cand/k$((NK+1)).json $((NK+1)) | tee -a $L
python3 $S/s26_make_placement_map.py $OUT/cand $OUT/placement_prefill_chunked.map --windows prefill_w:1-$NA,kv_replay:$((NA+1))-$NK,decode_own:$((NK+1))-$((NK+1)) --select prefill_w,kv_replay --ref $REF --stats $OUT/stats.json ${OVR:+--override $OVR} | tee -a $L
python3 - $OUT/placement_prefill_chunked.map $DEC <<'PY' | tee -a $L
import json, sys, collections
m, d = sys.argv[1], sys.argv[2]; own = {int(l.split()[1],16): int(l.split()[2]) for l in open(m) if l.startswith('D ')}
ci = json.load(open(d+'/capture_info.json')); vm = json.load(open(d+'/va_map.json')); kv = vm['kv_cache_layer0']; ptr, st, bs = kv['ptr'], kv['stride'], kv['shape'][2]; vb = ptr + st[0]*2; bt = vm['block_table']
chips = collections.Counter()
for r in range(ci['batch']):
    ps = set()
    for t in range(ci['S']):
        blk = bt[r][t//bs]; off = (blk*st[1] + (t%bs)*st[2])*2
        for p in ((ptr+off)>>16, (vb+off)>>16):
            if p in own: ps.add(own[p])
    chips[len(ps)] += 1
print('final map: chiplets per request KV (count of requests):', dict(sorted(chips.items())))
PY
python3 $S/s34_gates.py $OUT $DEC $NA | tee -a $L
log "done: $OUT/placement_prefill_chunked.map ($(tail -n1 $OUT/GATES.txt))"
