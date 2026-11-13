#!/bin/bash
# make_inject_hash_kit_s31.sh <decode capture dir> <placement_prefill_chunked.map> -- same kit as make_inject_kit_addon_s31.sh, but both arm
# configs add the s39 DRAM channel hashing (-gpgpu_memory_partition_indexing 7 = IPOLY inside each chiplet, -gpgpu_mem_hash_chiplets 4;
# config-only, binary unchanged, docs/s39 section 6.1). Original header follows.
# package the prefill-extrapolated decode experiment
# (docs/s34) for llm_sim_tree_s31: decode-window traces + kernelslist_decode.g, the predicted map (+ GATES/stats), and
# experiments/<cfg>_inject/{base_prefill (owner-only load of the map, dumps placement_end.map), ptwhop0_prefill (exact load of
# ../base_prefill/placement_end.map, dumps placement_check.map)}, queue_arms.sh (base -> ptwhop0 -> X-3). Memory template follows the
# capture's decode ref config (128 GB or 256 GB). Output: ~/kits/outbound/llm_addons_s31/<model>_<cfg>_inject_s31_<date>{,.tar,.tar.md5}
set -u
CAP=$(cd ${1:?capture dir} && pwd); MAP=${2:?map}; H=/home/user/accel-sim-docker/llm_exp; OUT=/home/user/kits/outbound/llm_addons_s31
CFG=$(basename $CAP); MODEL=$(basename $(dirname $CAP)); DATE=${ADDON_DATE:-$(date +%Y%m%d)}; NAME=${MODEL}_${CFG}_inject_hash${MEM_GB:+_mem$MEM_GB}_s31_${DATE}   # s42: MEM_GB forces the memory template and tags the kit name
die(){ echo "ABORT: $*" >&2; exit 1; }
[ -s $MAP ] || die "map missing"; O=$(dirname $MAP); grep -q '^GATES: PASS' $O/GATES.txt 2>/dev/null || die "GATES not PASS for $MAP"
S6=$(ls $CAP/h1_check_traces_decode_layer0_*.log | head -1); grep -q '^S6 TRACES PASS' $S6 || die "decode S6 PASS missing"
RC=$CAP/sim_noprefill/ref_mcm/gpgpusim.config; [ -s $RC ] || RC=$CAP/sim/ref_mcm/gpgpusim.config
# s41: a capture with no decode-own ref run (long-context L = 8K) takes the memory template from CFG_FROM (the same model's 4K ref config);
# never fall back silently to 128 GB (Llama-70B needs 256 GB).
NOREF=0; [ -s $RC ] || { RC=${CFG_FROM:-}; NOREF=1; [ -s "$RC" ] || die "no decode ref config for the memory template (set CFG_FROM)"; }
T=$H/llm_hypothesis1/_templates/mem128_s31_20261111/sim; MEMTXT="128 GB"; grep -q '^-gpu_mem_size_mb 262144' $RC 2>/dev/null && { T=$H/llm_hypothesis1/_templates/mem256_s31_20261112/sim; MEMTXT="256 GB (Llama-70B)"; }
case ${MEM_GB:-} in   # s42 section 12: one memory size for every point of the decode sweep
  256) T=$H/llm_hypothesis1/_templates/mem256_s31_20261112/sim; MEMTXT="256 GB (s42 sweep: all models 256 GiB)" ;;
  128) T=$H/llm_hypothesis1/_templates/mem128_s31_20261111/sim; MEMTXT="128 GB" ;;
  "") ;;
  *) die "MEM_GB must be 128 or 256" ;;
esac
read A B WARP BATCH S <<< "$(python3 -c "import json;ci=json.load(open('$CAP/capture_info.json'));w=ci['windows']['decode_layer0'];print(w[0],w[1],ci['window_warp_insts_by_window']['decode_layer0'],ci['batch'],ci['S'])")"
THREAD=$((WARP*32)); CAP0=0; [ $THREAD -gt 10000000000 ] && CAP0=1
D=$OUT/$NAME; rm -rf $D; mkdir -p $D/traces/$CFG $D/capture $D/experiments/${CFG}_inject; E=$D/experiments/${CFG}_inject
for f in $(grep -v MemcpyHtoD $CAP/traces/kernelslist_decode.g); do cp -p $CAP/traces/$f $D/traces/$CFG/ || die "trace $f missing"; done; cp -p $CAP/traces/kernelslist_decode.g $D/traces/$CFG/
cp -p $CAP/capture_info.json $CAP/va_map.json $CAP/steps.tsv $S6 $D/capture/; cp -p $MAP $E/placement_prefill_chunked.map; cp -p $O/GATES.txt $E/MAP_GATES.txt; cp -p $O/stats.json $E/map_stats.json 2>/dev/null || true
mkdir -p $E/base_prefill $E/ptwhop0_prefill
sed -e 's#^-gmmu_placement_load ../ref_mcm/placement_mcm.map#-gmmu_placement_load ../placement_prefill_chunked.map#' -e 's#^-gmmu_placement_exact 1#-gmmu_placement_exact 0#' -e 's#^-gmmu_placement_dump ./placement_check.map#-gmmu_placement_dump ./placement_end.map#' $T/base_mcm/gpgpusim.config > $E/base_prefill/gpgpusim.config
sed -e 's#^-gmmu_placement_load ../ref_mcm/placement_mcm.map#-gmmu_placement_load ../base_prefill/placement_end.map#' $T/ptwhop0_mcm/gpgpusim.config > $E/ptwhop0_prefill/gpgpusim.config
for a in base_prefill ptwhop0_prefill; do grep -q "^-gpgpu_memory_partition_indexing 0$" $E/$a/gpgpusim.config || die "indexing line not found"; sed -i "s/^-gpgpu_memory_partition_indexing 0$/-gpgpu_memory_partition_indexing 7\n-gpgpu_mem_hash_chiplets 4/" $E/$a/gpgpusim.config; done
for a in base_prefill ptwhop0_prefill; do cp -p $T/base_mcm/trace.config $T/base_mcm/config_ampere_islip.icnt $E/$a/; [ $CAP0 = 1 ] && sed -i 's/^-gpgpu_max_insn 10000000000$/-gpgpu_max_insn 0/' $E/$a/gpgpusim.config; done
echo "cee287166ea343c67ccc86d33628affa  gpu-simulator/bin/release/accel-sim.out" > $D/TREE_MD5.txt
cat > $E/run_arm.sh <<'EOT'
#!/bin/bash
A=${1:?arm}; E=$(cd $(dirname $0) && pwd); ADDON=$(cd $E/../.. && pwd); CFG=$(basename $E); CFG=${CFG%_inject}
TREE=${TREE:-$(cat $ADDON/TREE 2>/dev/null)}; [ -n "$TREE" ] && [ -x $TREE/gpu-simulator/bin/release/accel-sim.out ] || { echo "set TREE"; exit 1; }
TH=${TH:-16}; TRACE=$ADDON/traces/$CFG/kernelslist_decode.g; cd $E/$A || exit 1
source $TREE/gpu-simulator/setup_environment.sh release >/dev/null 2>&1
{ echo "s34 inject ${CFG} arm=$A tree=$TREE threads=$TH trace=$TRACE"; date; hostname; md5sum $TREE/gpu-simulator/bin/release/accel-sim.out gpgpusim.config $TRACE; } > provenance.txt
export ACCELSIM_NO_PIN=1; t0=$SECONDS
env OMP_NUM_THREADS=$TH ACCELSIM_ALLOW_UNVERIFIED_THREADS=1 nice -n 10 $TREE/gpu-simulator/bin/release/accel-sim.out -config ./gpgpusim.config -config ./trace.config -trace $TRACE > run.log 2>&1
echo "EXIT=$? SECONDS=$((SECONDS-t0)) load=$(cut -d' ' -f1 /proc/loadavg)" > exit_code.txt
EOT
cat > $E/queue_arms.sh <<'EOT'
#!/bin/bash
# queue_arms.sh -- base_prefill (owner-only load of the predicted map; dumps placement_end.map) -> ptwhop0_prefill (exact replay of that
# end state) -> X-3 cmp. Detach: TREE=... setsid nohup bash queue_arms.sh > queue.log 2>&1 &
E=$(cd $(dirname $0) && pwd); wait_arm(){ while ! grep -q '^EXIT=' $E/$1/exit_code.txt 2>/dev/null; do sleep 60; done; echo "$(date '+%F %T') $1: $(cat $E/$1/exit_code.txt)"; }
TREE=$TREE setsid nohup bash $E/run_arm.sh base_prefill > /dev/null 2>&1 < /dev/null & wait_arm base_prefill
grep -q '^EXIT=0' $E/base_prefill/exit_code.txt || { echo "base failed"; exit 1; }
grep -q '^# gpgpu-sim placement snapshot v3' $E/base_prefill/placement_end.map && [ "$(grep -c ffffffffffffffff $E/base_prefill/placement_end.map)" = 0 ] || { echo "base end map not v3 / has PFN -1"; exit 1; }
TREE=$TREE setsid nohup bash $E/run_arm.sh ptwhop0_prefill > /dev/null 2>&1 < /dev/null & wait_arm ptwhop0_prefill
cmp -s $E/ptwhop0_prefill/placement_check.map $E/base_prefill/placement_end.map && echo "$(date '+%F %T') X-3 PASS" || echo "$(date '+%F %T') X-3 FAIL" | tee $E/X3_CHECK.txt; echo "$(date '+%F %T') all arms done"
EOT
chmod +x $E/run_arm.sh $E/queue_arms.sh; cp -p $H/scripts/h1_slice_four_metrics.py $D/
cat > $D/README.ko.md <<EOT
# $NAME — prefill 외삽 배치를 주입한 decode layer-0 실험, **DRAM 채널 해싱 켬** (llm_sim_tree_s31 전용, 메모리 $MEMTXT)
- 해싱: 두 arm config의 \`-gpgpu_memory_partition_indexing 0\` → \`7\`(칩렛 안 IPOLY, 칩렛 비트 유지) + \`-gpgpu_mem_hash_chiplets 4\`. 바이너리는 기존 s31 그대로(cee28716). 같은 실험의 해싱 없는 결과는 로컬에 있음(비교용). 근거: 로컬 MoE decode-own에서 해싱만으로 총 cycle ×1.38, walk 평균 지연 −54 %(docs/s39 §6.1).
- decode 창: $MODEL / $CFG, 커널 $A–$B ($((B-A+1))개), warp $WARP$( [ $CAP0 = 1 ] && echo ", cap 0 예외" ), batch $BATCH S $S. 트레이스 \`traces/$CFG/\`.
- 주입 맵 \`experiments/${CFG}_inject/placement_prefill_chunked.map\`: prefill을 시뮬하지 않고 만든 배치(가중치 = prefill step-1 커널 예측기, KV = chunked step 스케줄 재생; docs/s34 §0, 게이트 MAP_GATES.txt).
- arm: \`base_prefill\`(맵 owner 적재 → decode 중 나머지 first touch → 끝 상태 v3 저장) → \`ptwhop0_prefill\`(그 끝 상태를 PFN까지 exact 적재, 홉 지연 0) → X-3(두 맵 byte 동일). base_prefill = 1로 정규화, decode-own ref(로컬 결과)는 비교용.
\`\`\`bash
export TREE=/path/to/llm_sim_tree_s31; cd experiments/${CFG}_inject
TREE=\$TREE setsid nohup bash queue_arms.sh > queue.log 2>&1 &
\`\`\`
$( [ $NOREF = 1 ] && echo "- 이 캡처(긴 생성 문맥, docs/s41)는 decode-own ref·해싱 없는 결과가 로컬에 없음: 비교 기준은 같은 모델 L = 4K 해싱 킷(\`*_s4096_b0032_decode_layer0_v2_inject_hash_s31_20261113\`)의 결과." )
- 끝나면 두 arm의 run.log·exit_code.txt·placement_end.map·X3_CHECK.txt를 보내 주면 4지표(\`h1_slice_four_metrics.py\`)를 뽑습니다.
EOT
( cd $D && find . -type f ! -name MANIFEST.md5 -print0 | sort -z | xargs -0 md5sum > MANIFEST.md5 ); ( cd $OUT && tar -cf $NAME.tar $NAME && md5sum $NAME.tar > $NAME.tar.md5 )
echo "inject kit: $OUT/$NAME.tar ($(du -sh $OUT/$NAME.tar | cut -f1)) md5 $(cut -c1-8 $OUT/$NAME.tar.md5)"
