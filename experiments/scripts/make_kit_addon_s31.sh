#!/bin/bash
# make_kit_addon_s31.sh <capture dir (host)>  -- s31 variant of make_b32_kit_addon.sh (docs/s30_b32_s4096_mem128_capture_plan.ko.md section 8).
# Package one finished decode-only capture (b32 or b256 x S4096 series) as a self-contained add-on to llm_sim_kit_20261111 for the
# NEW simulator tree llm_sim_tree_s31 (bin cee28716): arm configs = batch-4 recipe + 128 GB memory (4 x 32 GB, 37-bit mapping)
# + -gpgpu_perf_sim_memcpy 0 (all arms) + -gmmu_placement_exact 1 and a placement_check.map dump (base/ptwhop0). The capture's
# sim/ arms are regenerated from the s31 templates when they are older (moved aside, never deleted; aborts if an arm already ran).
# Output: ~/kits/outbound/llm_addons_s31/<model>_<cfg>_s31_<date>{,.tar,.tar.md5}.
# Requires: capture EXIT=0, H1 CAPTURE PASS, post-processing EXIT=0, S6 TRACES PASS for the decode window, s31 V-gates PASS.
set -u
CAP=${1:?capture dir}; CAP=$(cd $CAP && pwd)
H=/home/user/accel-sim-docker/llm_exp; OUT=/home/user/kits/outbound/llm_addons_s31; KIT=llm_sim_kit_20261111
TPL=$H/llm_hypothesis1/_templates/mem128_s31_20261111; BIN_MD5=cee287166ea343c67ccc86d33628affa; SO_MD5=8f0bffefff5176210632181b57c3f0eb
CFG=$(basename $CAP); MODEL=$(basename $(dirname $CAP)); DATE=${ADDON_DATE:-$(date +%Y%m%d)}; NAME=${MODEL}_${CFG}_s31_${DATE}
die(){ echo "ABORT: $*" >&2; exit 1; }
grep -q '^EXIT=0' $CAP/exit_code.txt || die "capture exit_code is not 0"
grep -q '^H1 CAPTURE PASS' $CAP/run.log || die "no 'H1 CAPTURE PASS' in run.log"
grep -q '^EXIT=0' $CAP/h1_postprocess_exit.txt 2>/dev/null || die "post-processing not finished (h1_postprocess_exit.txt)"
S6=$(ls $CAP/h1_check_traces_decode_layer0_*.log 2>/dev/null | head -1); [ -n "$S6" ] && grep -q '^S6 TRACES PASS' $S6 || die "decode-window S6 PASS log missing"
grep -q 'ALL V-GATES PASS' $TPL/validate/GATES.txt 2>/dev/null || die "s31 V-gates not passed yet ($TPL/validate/GATES.txt)"
[ "$(md5sum $H/llm_sim_tree_s31/gpu-simulator/bin/release/accel-sim.out | cut -c1-32)" = $BIN_MD5 ] || die "llm_sim_tree_s31 binary md5 changed"
read A B WARP BATCH S KEY SEQMIN SEQMAX <<< "$(python3 - $CAP/capture_info.json <<'PY'
import json, sys
ci = json.load(open(sys.argv[1])); w = ci["windows"]["decode_layer0"]
wi = ci.get("window_warp_insts_by_window", {}).get("decode_layer0", ci.get("window_warp_insts"))
print(w[0], w[1], wi, ci["batch"], ci["S"], ci["model"], ci.get("seq_lens_min"), ci.get("seq_lens_max"))
PY
)"
ROUTING=$(grep -m1 '^ROUTING' $CAP/run.log | cut -c1-300); FORCE=$(python3 -c "import json;print(json.load(open('$CAP/capture_info.json')).get('force_routing') or 'natural')")
THREAD_INSN=$((WARP * 32)); CAP0=0; [ $THREAD_INSN -gt 10000000000 ] && CAP0=1
echo "== $NAME: window $A-$B ($((B-A+1)) kernels), warp insts $WARP (x32 = $THREAD_INSN -> cap0=$CAP0), batch $BATCH S $S seq $SEQMIN-$SEQMAX, routing $FORCE"

# 1. local arms: must be byte-identical to the s31 templates; older sim/ (64 GB or pre-s31 128 GB configs) is moved aside and regenerated
if [ -d $CAP/sim ] && ! cmp -s $TPL/sim/ref_mcm/gpgpusim.config $CAP/sim/ref_mcm/gpgpusim.config; then
  ls $CAP/sim/*/exit_code.txt >/dev/null 2>&1 && die "$CAP/sim holds finished arms with pre-s31 configs; not touching it"
  OLD=$CAP/sim_pre_s31_$(date +%H%M%S); mv $CAP/sim $OLD; echo "moved pre-s31 arms aside: $OLD"; fi
[ -x $CAP/sim/launch_h1.sh ] || bash $H/scripts/h1_make_decode_arms_s31.sh $CAP $A $B > /dev/null || die "h1_make_decode_arms_s31 failed"
for arm in ref_mcm base_mcm ptwhop0_mcm; do cmp -s $TPL/sim/$arm/gpgpusim.config $CAP/sim/$arm/gpgpusim.config || die "sim/$arm config differs from the s31 template"; done
[ -s $CAP/traces/kernelslist_decode.g ] || die "kernelslist_decode.g missing"
NK=$(grep -vc MemcpyHtoD $CAP/traces/kernelslist_decode.g); [ "$NK" -eq $((B-A+1)) ] || die "decode list holds $NK kernels, window has $((B-A+1))"

# 2. add-on tree
D=$OUT/$NAME; rm -rf $D; mkdir -p $D/traces/$CFG $D/capture $D/experiments/$CFG
for f in $(grep -v MemcpyHtoD $CAP/traces/kernelslist_decode.g); do cp -p $CAP/traces/$f $D/traces/$CFG/ || die "trace file $f missing"; done
cp -p $CAP/traces/kernelslist_decode.g $CAP/traces/kernelslist.g $D/traces/$CFG/
cp -p $CAP/capture_info.json $CAP/va_map.json $CAP/steps.tsv $CAP/run.log $S6 $CAP/exit_code.txt $D/capture/
grep -E "$CFG" $H/llm_hypothesis1/$MODEL/chain_v2.log > $D/capture/chain_v2_lines.log
E=$D/experiments/$CFG
for arm in ref_mcm base_mcm ptwhop0_mcm; do mkdir -p $E/$arm; cp -p $CAP/sim/$arm/gpgpusim.config $CAP/sim/$arm/trace.config $CAP/sim/$arm/config_ampere_islip.icnt $E/$arm/
  if [ $CAP0 = 1 ]; then sed -i 's/^-gpgpu_max_insn 10000000000$/-gpgpu_max_insn 0/' $E/$arm/gpgpusim.config; grep -q '^-gpgpu_max_insn 0$' $E/$arm/gpgpusim.config || die "cap patch failed"; fi; done
if [ $CAP0 = 1 ]; then cat > $E/CAP_EXCEPTION.txt <<EOT
-gpgpu_max_insn 0 (no cap) in all three arms of $CFG.
Reason: the decode window holds $WARP warp instructions = about $THREAD_INSN thread instructions, above the standard cap
1e10, so the cap would cut the layer short (the standard cap stays in every other experiment). Same exception as the
prefill arms of llm_sim_kit_20261111 (experiments/h1_s1024_b4_prefill/CAP_EXCEPTION.txt). $(date '+%F %T')
EOT
  echo "cap 1e10 kept here; the shipped add-on ($D) runs the same configs with -gpgpu_max_insn 0, see its CAP_EXCEPTION.txt" > $CAP/sim/CAP_NOTE.txt; fi
echo "$BIN_MD5  gpu-simulator/bin/release/accel-sim.out
$SO_MD5  gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-12020/release/libcudart.so" > $D/TREE_MD5.txt
cat > $E/run_arm.sh <<'EOT'
#!/bin/bash
# run_arm.sh <ref_mcm|base_mcm|ptwhop0_mcm> -- run one arm of this add-on in the foreground (launch_addon.sh detaches it).
# Paths are relative to this file. TREE = simulator tree root (the directory holding gpu-simulator/; must be llm_sim_tree_s31,
# see ../../TREE_MD5.txt), from $TREE or the file ../../TREE. Threads: TH (default 16). Output: run.log, exit_code.txt, provenance.txt.
A=${1:?arm}; E=$(cd $(dirname $0) && pwd); ADDON=$(cd $E/../.. && pwd); CFG=$(basename $E)
TREE=${TREE:-$(cat $ADDON/TREE 2>/dev/null)}; [ -n "$TREE" ] && [ -x $TREE/gpu-simulator/bin/release/accel-sim.out ] || { echo "set TREE=<dir holding gpu-simulator/> (binary missing)"; exit 1; }
TH=${TH:-16}; TRACE=$ADDON/traces/$CFG/kernelslist_decode.g
cd $E/$A || exit 1
source $TREE/gpu-simulator/setup_environment.sh release >/dev/null 2>&1
{ echo "h1 $CFG arm=$A tree=$TREE threads=$TH trace=$TRACE max_insn=$(grep '^-gpgpu_max_insn ' gpgpusim.config | awk '{print $2}')"; date; hostname
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
# launch_addon.sh <ref_mcm|base_mcm|ptwhop0_mcm> -- preflight, then start run_arm.sh detached (setsid). Run it where the
# simulator binary is reachable (inside the container if the tree lives there). TH=16 by default.
set -u
A=$1; E=$(cd $(dirname $0) && pwd); ADDON=$(cd $E/../.. && pwd); CFG=$(basename $E)
die(){ echo "ABORT: $*" | tee -a $E/launch.log; exit 1; }
TREE=${TREE:-$(cat $ADDON/TREE 2>/dev/null)}; [ -n "$TREE" ] && [ -x $TREE/gpu-simulator/bin/release/accel-sim.out ] || die "TREE not set or binary missing"
BIN=$(md5sum $TREE/gpu-simulator/bin/release/accel-sim.out | cut -c1-32); grep -q "^$BIN " $ADDON/TREE_MD5.txt || echo "WARNING: $TREE binary md5 $BIN is not the s31 build listed in TREE_MD5.txt (rebuilt from the same source is fine; a different tree is not)" | tee -a $E/launch.log
grep -q '^S6 TRACES PASS' $ADDON/capture/h1_check_traces_decode_layer0_*.log || die "S6 PASS log missing"
n=$(grep -vc MemcpyHtoD $ADDON/traces/$CFG/kernelslist_decode.g); [ "$n" -ge 10 ] && [ "$n" -le 60 ] || die "decode trace list holds $n kernels"
for f in $(grep -v MemcpyHtoD $ADDON/traces/$CFG/kernelslist_decode.g); do [ -s $ADDON/traces/$CFG/$f ] || die "trace file $f missing"; done
grep -qx -- '-gpgpu_perf_sim_memcpy 0' $E/$A/gpgpusim.config || die "config is not an s31 config (-gpgpu_perf_sim_memcpy 0 missing)"
if [ $A != ref_mcm ]; then grep -q '^EXIT=0' $E/ref_mcm/exit_code.txt 2>/dev/null || die "ref_mcm not finished"
  grep -q '^# gpgpu-sim placement snapshot v3' $E/ref_mcm/placement_mcm.map 2>/dev/null || die "ref placement map missing or not v3 (ref must run on the s31 tree)"
  grep -q '^R 0x' $E/ref_mcm/placement_mcm.map || die "ref map has no root line"; grep -qx -- '-gmmu_placement_exact 1' $E/$A/gpgpusim.config || die "exact replay option missing"; fi
[ -e $E/$A/exit_code.txt ] && die "$A already ran (remove exit_code.txt/run.log to rerun)"
for pid in $(pgrep -x accel-sim.out); do [ "$(readlink /proc/$pid/cwd 2>/dev/null)" = "$E/$A" ] && die "$A is already running (pid $pid)"; done
echo "$(date '+%F %T') launch $A (TH=${TH:-16}, TREE=$TREE)" | tee -a $E/launch.log
TREE=$TREE TH=${TH:-16} setsid nohup bash $E/run_arm.sh $A > /dev/null 2>&1 < /dev/null &
EOT
cat > $E/queue_arms.sh <<'EOT'
#!/bin/bash
# queue_arms.sh -- ref_mcm (16t), then base_mcm and ptwhop0_mcm together (2 x 16t); waits on exit_code.txt; then the X-3 check
# (base/ptwhop0 placement_check.map byte-identical to ref's placement_mcm.map). Detach it:
#   TREE=... setsid nohup bash queue_arms.sh > queue.log 2>&1 &
E=$(cd $(dirname $0) && pwd)
wait_arm(){ while ! grep -q '^EXIT=' $E/$1/exit_code.txt 2>/dev/null; do sleep 120; done; echo "$(date '+%F %T') $1: $(cat $E/$1/exit_code.txt)"; }
bash $E/launch_addon.sh ref_mcm || exit 1; wait_arm ref_mcm
grep -q '^EXIT=0' $E/ref_mcm/exit_code.txt || { echo "ref failed; base/ptwhop0 not started"; exit 1; }
bash $E/launch_addon.sh base_mcm; sleep 5; bash $E/launch_addon.sh ptwhop0_mcm
wait_arm base_mcm; wait_arm ptwhop0_mcm
for a in base_mcm ptwhop0_mcm; do if cmp -s $E/$a/placement_check.map $E/ref_mcm/placement_mcm.map; then echo "$(date '+%F %T') X-3 $a PASS (placement_check.map == ref map)"; else echo "$(date '+%F %T') X-3 $a FAIL (placement_check.map differs from ref map)"; fi; done > $E/X3_CHECK.txt; cat $E/X3_CHECK.txt
echo "$(date '+%F %T') all arms done"
EOT
chmod +x $E/run_arm.sh $E/launch_addon.sh $E/queue_arms.sh
cp -p $H/scripts/h1_slice_four_metrics.py $D/ 2>/dev/null || true
EST_S=$(( WARP / 21000 )); EST_H=$(python3 -c "print(round($EST_S/3600,1))"); RAM_GB=$(python3 -c "print(round(24*$WARP/1.15e9,0))")
if [ "$BATCH" = 32 ]; then
  MEMNOTE="메모리 128 GB에서 전체 모델이 들어가는 조합(Qwen3-32B 101 GB, Qwen3-30B-A3B 74 GB)이라는 것이 이 분석점의 근거(docs/s30_b32_s4096_mem128_capture_plan.ko.md §1–2)."
  TILENOTE="M = 32라 cuBLAS GEMM 타일·split-K가 b256(256x128)과 다르다(b4/b16은 64x64 + split-K). b256 결과와 직접 비교하지 말고 새 분석점으로 볼 것."
else
  MEMNOTE="b${BATCH}×S${S}는 전체 모델 기준 128 GB에 들어가지 않는 조합(Qwen3-32B 363 GB, Qwen3-30B-A3B 173 GB; s30 §2)이라 '압력 감도' 자료이고, 메모리 설정만 b32 시리즈와 같은 128 GB로 통일했다(설정 통일 목적, 현실성 근거는 b32 쪽)."
  TILENOTE="M = ${BATCH}라 cuBLAS GEMM 타일(256x128)·커널 구성이 b4/b16(64x64 + split-K)·b32와 다르다(s27 §5.12). 다른 배치의 결과와 직접 비교하지 말 것."
fi
cat > $D/README.ko.md <<EOT
# $NAME — llm_sim_kit_20261111 추가 킷 (decode 전용 캡처 1건 + 가설 1 arm 3종, **s31 트리 전용**)

- 캡처: $MODEL / $CFG (batch $BATCH, S $S, 창 step 시퀀스 길이 $SEQMIN~$SEQMAX, 라우팅: $FORCE)
- decode 창: 커널 $A–$B ($((B-A+1))개), warp 명령 $WARP (thread 명령 ≈ $THREAD_INSN)
- $ROUTING
- 트레이스: \`traces/$CFG/\` = 창 커널의 .traceg.xz + \`kernelslist_decode.g\`(MemcpyHtoD 줄 + 창 커널; 시뮬 입력) + \`kernelslist.g\`(참고). 원시 .trace.xz는 넣지 않음.
- 검사: \`capture/\`에 capture_info.json, va_map.json(가중치·KV 주소), steps.tsv(step별 스케줄), run.log(ROUTING·CHECK 줄), S6 PASS 로그.
- **시뮬레이터**: 반드시 \`llm_sim_tree_s31\`(bin md5 \`TREE_MD5.txt\`; = exactmap 배치 재현 + 빠른 트레이스 파서 + memcpy 주소 64비트)로. 옛 트리(pwcstat_l1fwd, bin b4876c28)는 \`-gmmu_placement_exact\` 옵션을 몰라 base/ptwhop0가 뜨지 않음.
- arm: \`experiments/$CFG/{ref_mcm,base_mcm,ptwhop0_mcm}/\` — config는 킷의 h1 batch-4 레시피(64 KiB, KERNEL_PARTITION + cursor, L2 TLB set hash 1, L1 TLB 64K 32, L2 TLB 64K 1024, merge 무제한) + **시뮬 메모리 128 GB**(\`-gpu_mem_size_mb 131072\`, 37비트 주소 매핑 \`000…0DD R×15 BBB CCC B DDDDD CCC SSSSS\`; 칩렛 4 × 32 GB, s30 §3) + **s31 옵션**: 세 arm 모두 \`-gpgpu_perf_sim_memcpy 0\`(memcpy L2 워밍업 끔 — 번역 안 된 VA로 L2를 채우는 결함, s31 §6), base/ptwhop0는 \`-gmmu_placement_load ../ref_mcm/placement_mcm.map\` + \`-gmmu_placement_exact 1\`(데이터·PT 노드·루트 PFN까지 ref 그대로 재현; owner만 재현하던 옛 base는 PT 노드 DRAM row 아티팩트로 cold-miss 커널이 느렸음, docs/s30_exact_placement_replay.ko.md) + 검사용 \`-gmmu_placement_dump ./placement_check.map\`. ptwhop0는 \`-icnt_ptw_ring_hop_latency 0\` 추가$( [ $CAP0 = 1 ] && echo ". **단 -gpgpu_max_insn 0** (CAP_EXCEPTION.txt: thread 명령 > 1e10)" ).
- 순서 제약: ref가 v3 맵(\`# gpgpu-sim placement snapshot v3\`, N 줄 PFN, \`R\` 줄)을 쓰고 base/ptwhop0가 그것을 적재하므로 **ref → base/ptwhop0** 순서, 같은 트리·같은 메모리 설정. launch_addon.sh가 검사함.

## 실행
\`\`\`bash
export TREE=/path/to/llm_sim_tree_s31      # gpu-simulator/ 가 들어 있는 디렉토리; md5sum -c TREE_MD5.txt 로 확인 (같은 소스 재빌드면 md5가 달라도 됨)
cd experiments/$CFG
TREE=\$TREE setsid nohup bash queue_arms.sh > queue.log 2>&1 &     # ref(16t) → base+ptwhop0(2×16t) → X-3 검사 자동
# 또는 arm 하나씩: TREE=\$TREE bash launch_addon.sh ref_mcm   (끝나면 base_mcm, ptwhop0_mcm)
\`\`\`
- 예상 시간(16t, Llama b256·S256 ref 실측 2.1만 warp 명령/s 기준; s31 파서로 적재는 더 빠름): arm당 ≈ $EST_S s ≈ $EST_H h. RAM: b1024(1.15e9 warp 명령) arm RSS 24 GB 비례 추정 ≈ $RAM_GB GB/arm.
- 판정 X-3: \`X3_CHECK.txt\`(queue_arms.sh가 씀) — base/ptwhop0의 placement_check.map이 ref의 placement_mcm.map과 byte 동일해야 함(base는 fault를 안 내므로 끝 상태 == 적재 상태). FAIL이면 맵·트리 불일치이므로 결과를 쓰지 말 것.
- 끝나면 \`exit_code.txt\`(EXIT=0 SECONDS=…)와 run.log 끝의 통계 블록, ref의 placement_mcm.map, X3_CHECK.txt를 다시 보내 주면 됨. 4지표: \`python3 h1_slice_four_metrics.py experiments/$CFG 64k $A-$B $BATCH\`(이 add-on 루트에서; 커널 범위 차분).

## 읽을 때 전제 (s27 §5.12·§5.11, s30 §7)
1. KV 페이지 배치: decode 전용 캡처라 prefill이 KV를 쓴 커널이 없음 → ref에서는 attention이 읽는 순간 first-touch(읽는 CTA의 칩렛, 낙관적). "prefill이 썼다면"의 맵은 steps.tsv + va_map.json으로 해석적으로 만들어 뒤에 보냄(placement_injection arm).
2. $TILENOTE $MEMNOTE
3. 창 step의 시퀀스 길이가 $SEQMIN~$SEQMAX로 퍼짐(chunked prefill 순서). 자연 라우팅 캡처의 창에는 후크 기록용 커널(float 캐스트, topk; 원소 $BATCH×E)이 들어 있을 수 있음 — 커널별 표에서 이름(elementwise/gatherTopK/bitonicSort)으로 제외.
4. 옛 트리로 돌린 같은 캡처의 결과(있다면)와는 base/ptwhop0 정의가 달라졌으므로(exact 재현 + memcpy 0) 섞지 말 것. 알려진 결함은 킷 README와 같음(b64 STG 라이브락 미해결 → 새 커널 조합에서 재발 가능, ref부터 감시; SIM-FIND-1).
EOT
echo "$(date '+%F %T') made by scripts/make_kit_addon_s31.sh from $CAP (templates $TPL, tree bin $BIN_MD5)" > $D/.made_by
( cd $D && find . -type f ! -name MANIFEST.md5 -print0 | sort -z | xargs -0 md5sum > MANIFEST.md5 )
( cd $OUT && tar -cf $NAME.tar $NAME && md5sum $NAME.tar > $NAME.tar.md5 )
echo "add-on: $D  ($(du -sh $D | cut -f1)); tar: $OUT/$NAME.tar ($(du -sh $OUT/$NAME.tar | cut -f1)) md5 $(cut -c1-8 $OUT/$NAME.tar.md5)…"
echo "pull (on the other server):  rsync -av --partial --progress -e 'ssh -p 2380' user@<REMOTE_HOST>:~/kits/outbound/llm_addons_s31/$NAME.tar{,.md5} ~/kits_20261111/ && cd ~/kits_20261111 && md5sum -c $NAME.tar.md5 && tar -xf $NAME.tar"
