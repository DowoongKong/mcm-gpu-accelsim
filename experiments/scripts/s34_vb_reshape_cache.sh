#!/bin/bash
# s34_vb_reshape_cache.sh <prefill capture dir (b1 x S, prefill_layer0 window)> [TH=8]  -- V-b for the KV replay rule (docs/s34 section 3):
# simulate ONLY the prefill step's reshape_and_cache kernel (1 CTA per token, grid = 4096) as a ref run (s31 ref template, standard cap)
# and compare the dumped KV page owners with the chunked-replay prediction for step 1 of the same capture (its own VA space).
# A few minutes of simulation; waits until fewer than 2 accel-sim.out are alive. Output: analysis/s34_prefill_placement/vb/<model>/rc_ref/,
# result line "V-b ... agreement N/M" appended to analysis/s34_prefill_placement/vb/<model>/VB.txt.
set -u
PRE=$1; TH=${2:-8}; H=/home/user/accel-sim-docker/llm_exp; S=$H/scripts; T=$H/llm_hypothesis1/_templates/mem128_s31_20261111/sim/ref_mcm
MODEL=$(basename $(dirname $PRE)); O=$H/analysis/s34_prefill_placement/vb/$MODEL; R=$O/rc_ref; mkdir -p $R; L=$O/VB.txt
grep -q '^S6 TRACES PASS' $PRE/h1_check_traces_prefill_layer0_*.log 2>/dev/null || { echo "prefill S6 PASS missing"; exit 1; }
read A B <<< "$(python3 -c "import json;w=json.load(open('$PRE/capture_info.json'))['windows']['prefill_layer0'];print(w[0],w[1])")"
KL=$PRE/traces/kernelslist.g; RC=""
for f in $(grep 'kernel-' $KL | awk -v a=$A -v b=$B '{match($0,/kernel-([0-9]+)-/,m); if (m[1]+0>=a && m[1]+0<=b) print}'); do
  n=$(xz -dc $PRE/traces/$f | head -3 | grep -o 'kernel name = .*' | cut -c15-); case "$n" in *reshape_and_cache*) RC=$f;; esac; [ -n "$RC" ] && break; done
[ -n "$RC" ] || { echo "no reshape_and_cache kernel in the prefill window"; exit 1; }
{ grep MemcpyHtoD $KL; echo $RC; } > $PRE/traces/kernelslist_vb_rc.g
cp -p $T/gpgpusim.config $T/trace.config $T/config_ampere_islip.icnt $R/
CP=/accel-sim/llm_exp/${PRE#$H/}; CR=/accel-sim/llm_exp/${R#$H/}
cat > $R/run.sh <<EOT
#!/bin/bash
TREE=/accel-sim/llm_exp/llm_sim_tree_s31; TH=\${TH:-$TH}; TRACE=$CP/traces/kernelslist_vb_rc.g
cd $CR; source \$TREE/gpu-simulator/setup_environment.sh release >/dev/null 2>&1; export ACCELSIM_NO_PIN=1
{ echo "s34 V-b reshape_and_cache ref tree=\$TREE threads=\$TH trace=\$TRACE kernel=$RC"; date; md5sum \$TREE/gpu-simulator/bin/release/accel-sim.out gpgpusim.config; } > provenance.txt
t0=\$SECONDS; env OMP_NUM_THREADS=\$TH ACCELSIM_ALLOW_UNVERIFIED_THREADS=1 nice -n 10 \$TREE/gpu-simulator/bin/release/accel-sim.out -config ./gpgpusim.config -config ./trace.config -trace \$TRACE > run.log 2>&1
echo "EXIT=\$? SECONDS=\$((SECONDS-t0))" > exit_code.txt
EOT
chmod +x $R/run.sh
while [ "$(ps -o stat= -C accel-sim.out | grep -vc '^Z')" -ge 2 ]; do sleep 60; done
echo "$(date '+%F %T') V-b launch (kernel $RC, TH $TH)" | tee -a $L; docker exec -u 1035:1035 -e TH=$TH mcm-accelsim bash $CR/run.sh
echo "$(date '+%F %T') V-b run: $(cat $R/exit_code.txt)" | tee -a $L; grep -q '^EXIT=0' $R/exit_code.txt || exit 1
mkdir -p $O/replay_pre; python3 $S/s34_kv_replay_candidates.py $PRE $O/replay_pre 1 | tee -a $L
python3 - $R/placement_mcm.map $O/replay_pre/k1.json $PRE <<'EOF_C' | tee -a $L
import json, sys, collections
own = {int(l.split()[1],16): int(l.split()[2]) for l in open(sys.argv[1]) if l.startswith('D ')}
J = json.load(open(sys.argv[2])); pred = {}
for p, d in J['pages'].items():
    c = sorted((tuple(v[:3]), int(k)) for k, v in d.items())[0]; pred[int(p)] = (c[1], len(d) > 1 and sorted(tuple(v[:3]) for v in d.values())[0] == sorted(tuple(v[:3]) for v in d.values())[1] if len(d) > 1 else False)
common = [p for p in pred if p in own]; ok = sum(own[p] == pred[p][0] for p in common); ties = sum(1 for p in common if pred[p][1])
print(f"V-b reshape_and_cache (prefill step 1, {len(pred):,} KV pages predicted; dump D {len(own):,}): common {len(common):,}, owner agreement {ok}/{len(common)} = {100*ok/max(1,len(common)):.1f} % (tied-in-wave pages among common: {ties}); dump-only pages (non-KV) {len(set(own)-set(pred)):,}; owner dist dump {dict(collections.Counter(own[p] for p in common))} pred {dict(collections.Counter(pred[p][0] for p in common))}")
EOF_C
