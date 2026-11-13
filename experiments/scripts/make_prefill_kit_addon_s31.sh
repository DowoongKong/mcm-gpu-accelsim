#!/bin/bash
# make_prefill_kit_addon_s31.sh <prefill capture dir (b1 x S, prefill_layer0 window)> [mem128|mem256]
# Package the PREFILL step-1 layer-0 window of a b1 x S4096 capture as an add-on for llm_sim_tree_s31 (docs/s30 section 9, s34):
# traces of the prefill window + kernelslist_prefill.g (MemcpyHtoD lines + window kernels), capture metadata, and three arms under
# experiments/<cfg>_prefill/: ref_mcm (first touch, dumps v3 map), ptwhop0_mcm (loads ../ref_mcm/placement_mcm.map exactly -> protocol),
# ptwhop0_ft (first touch, no map dependency -> may run in PARALLEL with ref when time is short; queue_arms_parallel.sh then compares the
# two dumped maps). Cap: -gpgpu_max_insn 0 with CAP_EXCEPTION when warp insts x 32 > 1e10 (prefill layers always). Memory template:
# mem128 (Qwen, Mixtral) or mem256 (Llama-70B). Output: ~/kits/outbound/llm_addons_s31/<model>_<cfg>_prefill_s31_<date>{,.tar,.tar.md5}
set -u
CAP=${1:?capture dir}; CAP=$(cd $CAP && pwd); MEM=${2:-mem128}
H=/home/user/accel-sim-docker/llm_exp; OUT=/home/user/kits/outbound/llm_addons_s31
case $MEM in mem128) TPL=$H/llm_hypothesis1/_templates/mem128_s31_20261111/sim; MEMTXT="128 GB (4 x 32 GB, 37-bit mapping)";; mem256) TPL=$H/llm_hypothesis1/_templates/mem256_s31_20261112/sim; MEMTXT="256 GB (4 x 64 GB, 38-bit mapping, Llama-70B)";; *) echo "mem128|mem256"; exit 1;; esac
BIN_MD5=cee287166ea343c67ccc86d33628affa; SO_MD5=8f0bffefff5176210632181b57c3f0eb
CFG=$(basename $CAP); MODEL=$(basename $(dirname $CAP)); DATE=${ADDON_DATE:-$(date +%Y%m%d)}; NAME=${MODEL}_${CFG}_prefill_s31_${DATE}
die(){ echo "ABORT: $*" >&2; exit 1; }
grep -q '^EXIT=0' $CAP/exit_code.txt || die "capture exit_code is not 0"
grep -q '^H1 CAPTURE PASS' $CAP/run.log || die "no 'H1 CAPTURE PASS' in run.log"
grep -q '^EXIT=0' $CAP/h1_postprocess_exit.txt 2>/dev/null || die "post-processing not finished"
S6=$(ls $CAP/h1_check_traces_prefill_layer0_*.log 2>/dev/null | head -1); [ -n "$S6" ] && grep -q '^S6 TRACES PASS' $S6 || die "prefill-window S6 PASS log missing"
[ "$(md5sum $H/llm_sim_tree_s31/gpu-simulator/bin/release/accel-sim.out | cut -c1-32)" = $BIN_MD5 ] || die "llm_sim_tree_s31 binary md5 changed"
read A B WARP BATCH S KEY <<< "$(python3 - $CAP/capture_info.json <<'PY'
import json, sys
ci = json.load(open(sys.argv[1])); w = ci["windows"]["prefill_layer0"]; wi = ci["window_warp_insts_by_window"]["prefill_layer0"]
print(w[0], w[1], wi, ci["batch"], ci["S"], ci["model"])
PY
)"
THREAD_INSN=$((WARP * 32)); CAP0=0; [ $THREAD_INSN -gt 10000000000 ] && CAP0=1
echo "== $NAME: prefill window $A-$B ($((B-A+1)) kernels), warp insts $WARP (x32 = $THREAD_INSN -> cap0=$CAP0), batch $BATCH S $S, memory $MEM"
# kernel list for the prefill window
grep -E "MemcpyHtoD" $CAP/traces/kernelslist.g > $CAP/traces/kernelslist_prefill.g
grep -vE "MemcpyHtoD" $CAP/traces/kernelslist.g | awk -v a=$A -v b=$B '{match($0,/kernel-([0-9]+)-/,m); if (m[1]+0>=a && m[1]+0<=b) print}' >> $CAP/traces/kernelslist_prefill.g
NK=$(grep -vc MemcpyHtoD $CAP/traces/kernelslist_prefill.g); [ "$NK" -eq $((B-A+1)) ] || die "prefill list holds $NK kernels, window has $((B-A+1))"
# add-on tree
D=$OUT/$NAME; rm -rf $D; mkdir -p $D/traces/$CFG $D/capture $D/experiments/${CFG}_prefill
for f in $(grep -v MemcpyHtoD $CAP/traces/kernelslist_prefill.g); do cp -p $CAP/traces/$f $D/traces/$CFG/ || die "trace file $f missing"; done
cp -p $CAP/traces/kernelslist_prefill.g $CAP/traces/kernelslist.g $D/traces/$CFG/
cp -p $CAP/capture_info.json $CAP/va_map.json $CAP/steps.tsv $CAP/run.log $S6 $CAP/exit_code.txt $D/capture/
grep -E "$CFG" $H/llm_hypothesis1/$MODEL/chain_v2.log > $D/capture/chain_v2_lines.log || true
E=$D/experiments/${CFG}_prefill
for arm in ref_mcm ptwhop0_mcm ptwhop0_ft; do src=$arm; [ $arm = ptwhop0_ft ] && src=ptwhop0_mcm; mkdir -p $E/$arm; cp -p $TPL/$src/trace.config $TPL/$src/config_ampere_islip.icnt $E/$arm/
  if [ $arm = ptwhop0_ft ]; then   # first-touch ptwhop0: no map dependency (parallel option); dumps its own map for the post-hoc comparison
    sed -e '/^-gmmu_placement_load /d' -e '/^-gmmu_placement_exact 1$/d' -e 's#^-gmmu_placement_dump ./placement_check.map#-gmmu_placement_dump ./placement_mcm.map#' $TPL/$src/gpgpusim.config > $E/$arm/gpgpusim.config
  else cp -p $TPL/$src/gpgpusim.config $E/$arm/gpgpusim.config; fi
  if [ $CAP0 = 1 ]; then sed -i 's/^-gpgpu_max_insn 10000000000$/-gpgpu_max_insn 0/' $E/$arm/gpgpusim.config; grep -q '^-gpgpu_max_insn 0$' $E/$arm/gpgpusim.config || die "cap patch failed"; fi; done
[ $CAP0 = 1 ] && cat > $E/CAP_EXCEPTION.txt <<EOT
-gpgpu_max_insn 0 (no cap) in all arms of ${CFG}_prefill.
Reason: the prefill layer-0 window (step 1, $S tokens) holds $WARP warp instructions = about $THREAD_INSN thread instructions, above the
standard cap 1e10, which would cut the layer short. Same exception as llm_sim_kit_20261111/experiments/h1_s1024_b4_prefill. $(date '+%F %T')
EOT
echo "$BIN_MD5  gpu-simulator/bin/release/accel-sim.out
$SO_MD5  gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-12020/release/libcudart.so" > $D/TREE_MD5.txt
cat > $E/run_arm.sh <<'EOT'
#!/bin/bash
# run_arm.sh <ref_mcm|ptwhop0_mcm|ptwhop0_ft> -- run one prefill arm in the foreground (launch_addon.sh detaches it). TREE = llm_sim_tree_s31
# root (dir holding gpu-simulator/), from $TREE or ../../TREE. Threads: TH (default 16). Output: run.log, exit_code.txt, provenance.txt.
A=${1:?arm}; E=$(cd $(dirname $0) && pwd); ADDON=$(cd $E/../.. && pwd); CFG=$(basename $E); CFG=${CFG%_prefill}
TREE=${TREE:-$(cat $ADDON/TREE 2>/dev/null)}; [ -n "$TREE" ] && [ -x $TREE/gpu-simulator/bin/release/accel-sim.out ] || { echo "set TREE=<dir holding gpu-simulator/>"; exit 1; }
TH=${TH:-16}; TRACE=$ADDON/traces/$CFG/kernelslist_prefill.g
cd $E/$A || exit 1
source $TREE/gpu-simulator/setup_environment.sh release >/dev/null 2>&1
{ echo "h1 ${CFG}_prefill arm=$A tree=$TREE threads=$TH trace=$TRACE max_insn=$(grep '^-gpgpu_max_insn ' gpgpusim.config | awk '{print $2}')"; date; hostname
  md5sum $TREE/gpu-simulator/bin/release/accel-sim.out $GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG/libcudart.so gpgpusim.config trace.config config_ampere_islip.icnt $TRACE
  ldd $TREE/gpu-simulator/bin/release/accel-sim.out | grep cudart
} > provenance.txt
export ACCELSIM_NO_PIN=1
t0=$SECONDS
env OMP_NUM_THREADS=$TH ACCELSIM_ALLOW_UNVERIFIED_THREADS=1 nice -n 10 \
  $TREE/gpu-simulator/bin/release/accel-sim.out -config ./gpgpusim.config -config ./trace.config -trace $TRACE > run.log 2>&1
echo "EXIT=$? SECONDS=$((SECONDS-t0)) load=$(cut -d' ' -f1 /proc/loadavg)" > exit_code.txt
EOT
cat > $E/launch_addon.sh <<'EOT'
#!/bin/bash
# launch_addon.sh <ref_mcm|ptwhop0_mcm|ptwhop0_ft> -- preflight, then start run_arm.sh detached (setsid). TH=16 by default.
set -u
A=$1; E=$(cd $(dirname $0) && pwd); ADDON=$(cd $E/../.. && pwd); CFG=$(basename $E); CFG=${CFG%_prefill}
die(){ echo "ABORT: $*" | tee -a $E/launch.log; exit 1; }
TREE=${TREE:-$(cat $ADDON/TREE 2>/dev/null)}; [ -n "$TREE" ] && [ -x $TREE/gpu-simulator/bin/release/accel-sim.out ] || die "TREE not set or binary missing"
BIN=$(md5sum $TREE/gpu-simulator/bin/release/accel-sim.out | cut -c1-32); grep -q "^$BIN " $ADDON/TREE_MD5.txt || echo "WARNING: binary md5 $BIN differs from TREE_MD5.txt (same-source rebuild is fine)" | tee -a $E/launch.log
grep -q '^S6 TRACES PASS' $ADDON/capture/h1_check_traces_prefill_layer0_*.log || die "S6 PASS log missing"
for f in $(grep -v MemcpyHtoD $ADDON/traces/$CFG/kernelslist_prefill.g); do [ -s $ADDON/traces/$CFG/$f ] || die "trace file $f missing"; done
grep -qx -- '-gpgpu_perf_sim_memcpy 0' $E/$A/gpgpusim.config || die "not an s31 config"
if [ $A = ptwhop0_mcm ]; then grep -q '^EXIT=0' $E/ref_mcm/exit_code.txt 2>/dev/null || die "ref_mcm not finished"
  grep -q '^# gpgpu-sim placement snapshot v3' $E/ref_mcm/placement_mcm.map 2>/dev/null && grep -q '^R 0x' $E/ref_mcm/placement_mcm.map || die "ref map missing or not v3"
  [ "$(grep -c 'ffffffffffffffff' $E/ref_mcm/placement_mcm.map)" = 0 ] || die "ref map has PFN -1 entries"; fi
[ -e $E/$A/exit_code.txt ] && die "$A already ran (remove exit_code.txt/run.log to rerun)"
for pid in $(pgrep -x accel-sim.out); do [ "$(readlink /proc/$pid/cwd 2>/dev/null)" = "$E/$A" ] && die "$A is already running (pid $pid)"; done
echo "$(date '+%F %T') launch $A (TH=${TH:-16}, TREE=$TREE)" | tee -a $E/launch.log
TREE=$TREE TH=${TH:-16} setsid nohup bash $E/run_arm.sh $A > /dev/null 2>&1 < /dev/null &
EOT
cat > $E/queue_arms.sh <<'EOT'
#!/bin/bash
# queue_arms.sh -- PROTOCOL order: ref_mcm (16t) then ptwhop0_mcm (loads ref's v3 map exactly), then the X-3 cmp. Detach:
#   TREE=... setsid nohup bash queue_arms.sh > queue.log 2>&1 &
E=$(cd $(dirname $0) && pwd)
wait_arm(){ while ! grep -q '^EXIT=' $E/$1/exit_code.txt 2>/dev/null; do sleep 120; done; echo "$(date '+%F %T') $1: $(cat $E/$1/exit_code.txt)"; }
bash $E/launch_addon.sh ref_mcm || exit 1; wait_arm ref_mcm
grep -q '^EXIT=0' $E/ref_mcm/exit_code.txt || { echo "ref failed; ptwhop0 not started"; exit 1; }
bash $E/launch_addon.sh ptwhop0_mcm; wait_arm ptwhop0_mcm
if cmp -s $E/ptwhop0_mcm/placement_check.map $E/ref_mcm/placement_mcm.map; then echo "$(date '+%F %T') X-3 PASS"; else echo "$(date '+%F %T') X-3 FAIL"; fi | tee $E/X3_CHECK.txt
echo "$(date '+%F %T') all arms done"
EOT
cat > $E/queue_arms_parallel.sh <<'EOT'
#!/bin/bash
# queue_arms_parallel.sh -- TIME-SAVING order (needs the user's OK): ref_mcm and ptwhop0_ft run TOGETHER (2 x 16t), both first-touch. The
# weight first touch of a prefill GEMM is decided by CTA order (s26 V-a: 100 % for non-split-K GEMMs), so the two dumped maps are expected
# to be equal; the owner agreement of the two maps is reported at the end (PLACEMENT_CHECK.txt). Detach:
#   TREE=... setsid nohup bash queue_arms_parallel.sh > queue_parallel.log 2>&1 &
E=$(cd $(dirname $0) && pwd)
wait_arm(){ while ! grep -q '^EXIT=' $E/$1/exit_code.txt 2>/dev/null; do sleep 120; done; echo "$(date '+%F %T') $1: $(cat $E/$1/exit_code.txt)"; }
bash $E/launch_addon.sh ref_mcm || exit 1; sleep 5; bash $E/launch_addon.sh ptwhop0_ft || exit 1
wait_arm ref_mcm; wait_arm ptwhop0_ft
python3 - $E/ref_mcm/placement_mcm.map $E/ptwhop0_ft/placement_mcm.map <<'PY' | tee $E/PLACEMENT_CHECK.txt
import sys
def rd(p): return {l.split()[1]: l.split()[2] for l in open(p) if l.startswith('D ')}, {(l.split()[1], l.split()[2]): l.split()[3] for l in open(p) if l.startswith('N ')}
a, na = rd(sys.argv[1]); b, nb = rd(sys.argv[2]); common = [k for k in a if k in b]; ok = sum(a[k] == b[k] for k in common)
nc = [k for k in na if k in nb]; nok = sum(na[k] == nb[k] for k in nc)
print(f"PLACEMENT_CHECK ref vs ptwhop0_ft: data pages ref {len(a):,} ft {len(b):,} common {len(common):,} owner-equal {ok:,} ({100*ok/max(1,len(common)):.2f} %); PT nodes common {len(nc)} equal {nok}")
print("PLACEMENT_EQUAL" if ok == len(common) == len(a) == len(b) and nok == len(nc) else "PLACEMENT_DIFFERS (report the % above; the two arms then differ by placement, not only by the hop latency)")
PY
echo "$(date '+%F %T') all arms done"
EOT
chmod +x $E/run_arm.sh $E/launch_addon.sh $E/queue_arms.sh $E/queue_arms_parallel.sh
cp -p $H/scripts/h1_slice_four_metrics.py $D/ 2>/dev/null || true
EST_S=$(( WARP / 33000 )); EST_H=$(python3 -c "print(round($EST_S/3600,1))")
cat > $D/README.ko.md <<EOT
# $NAME — PREFILL step-1 layer-0 창 (b1×S$S run = b32×S$S run의 step 1과 동일한 4,096행 chunk), llm_sim_tree_s31 전용

- 캡처: $MODEL / $CFG (batch $BATCH, S $S). prefill 창 = 커널 $A–$B ($((B-A+1))개), warp 명령 $WARP (thread ≈ $THREAD_INSN) → **-gpgpu_max_insn 0** (CAP_EXCEPTION.txt).
- 왜 step 1 하나인가: vLLM은 b32×S4096 prefill을 4,096행 step 32개로 쪼개고(chunked prefill), step들은 GEMM·가중치 읽기·활성값이 동일하며 attention 산술 ±1 %, decode 행의 KV 읽기만 뒤 step에서 늘어난다(llm_exp/docs/s30_b32_s4096_mem128_capture_plan.ko.md §9.1–9.2). b1 run의 step 1은 b32 run의 step 1과 같은 토큰·행·KV 블록이다(§9.3).
- 트레이스: \`traces/$CFG/\` = 창 커널 .traceg.xz + \`kernelslist_prefill.g\`(MemcpyHtoD 줄 + 창 커널; 시뮬 입력). \`capture/\`에 capture_info·va_map·steps·run.log·S6 PASS 로그.
- 시뮬레이터: **llm_sim_tree_s31**(bin \`TREE_MD5.txt\`; 번들 \`llm_sim_tree_s31_src_20261111.tar\`로 빌드·verify PASS 뒤). config = h1 batch-4 레시피 + 시뮬 메모리 **$MEMTXT** + \`-gpgpu_perf_sim_memcpy 0\`; ptwhop0_mcm은 \`../ref_mcm/placement_mcm.map\` exact 적재(\`-gmmu_placement_exact 1\`) + placement_check 덤프.
- arm 3개 \`experiments/${CFG}_prefill/\`: \`ref_mcm\`(first touch, v3 맵 덤프), \`ptwhop0_mcm\`(ref 맵 exact 적재 → **프로토콜**), \`ptwhop0_ft\`(first touch, 맵 의존 없음 → **병렬 실행용**, 자기 맵 덤프).

## 실행
\`\`\`bash
export TREE=/path/to/llm_sim_tree_s31
cd experiments/${CFG}_prefill
# (A) 프로토콜: ref → ptwhop0(ref 맵 exact) → X-3.  arm당 ≈ $EST_H h(우리 16t 실측 3.3e4 warp/s 기준), 합 ≈ $(python3 -c "print(round(2*$EST_S/3600,1))") h
TREE=\$TREE setsid nohup bash queue_arms.sh > queue.log 2>&1 &
# (B) 시간 절약: ref 와 ptwhop0_ft 동시(2×16t), 끝에 두 맵의 소유자 일치율을 PLACEMENT_CHECK.txt에 기록. 합 ≈ $EST_H h. 사용자 승인 뒤에만.
TREE=\$TREE setsid nohup bash queue_arms_parallel.sh > queue_parallel.log 2>&1 &
\`\`\`
- 끝나면 각 arm의 \`exit_code.txt\`·run.log 통계 블록·placement_mcm.map(ref)·X3_CHECK.txt 또는 PLACEMENT_CHECK.txt를 보내 주면 됨. 4지표: \`python3 h1_slice_four_metrics.py experiments/${CFG}_prefill 64k $A-$B $BATCH\`.
- RAM: warp 명령 비례(b1024 decode 1.15e9 → 24 GB) ≈ $(python3 -c "print(round(24*$WARP/1.15e9,1))") GB/arm.
EOT
echo "$(date '+%F %T') made by scripts/make_prefill_kit_addon_s31.sh from $CAP ($MEM)" > $D/.made_by
( cd $D && find . -type f ! -name MANIFEST.md5 -print0 | sort -z | xargs -0 md5sum > MANIFEST.md5 )
( cd $OUT && tar -cf $NAME.tar $NAME && md5sum $NAME.tar > $NAME.tar.md5 )
echo "add-on: $D ($(du -sh $D | cut -f1)); tar: $OUT/$NAME.tar ($(du -sh $OUT/$NAME.tar | cut -f1)) md5 $(cut -c1-8 $OUT/$NAME.tar.md5)…"
echo "pull: rsync -av --partial --progress -e 'ssh -p 2380' user@<REMOTE_HOST>:~/kits/outbound/llm_addons_s31/$NAME.tar{,.md5} ~/kits_20261111/ && cd ~/kits_20261111 && md5sum -c $NAME.tar.md5 && tar -xf $NAME.tar"
