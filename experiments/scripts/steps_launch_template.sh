#!/bin/bash
# launch.sh -- piece @DIR@ of a step-piece capture (@EXP@) on llm_sim_tree.
#   ref_batch4 (REF_TH threads); with MEASURE=1 also base_pcc and ptwhop0_pcc (ARM_TH threads each) after it,
#   side by side when PARALLEL=1, base_pcc then ptwhop0_pcc when PARALLEL=0.
# Explained in llm_exp/docs/s8_b32s256_step_pieces.ko.md. Generated from llm_exp/scripts/steps_launch_template.sh (copy of s8_launch_template.sh with the paths as parameters).
# Run inside mcm-accelsim as uid 1035:
#   docker exec -d -u 1035:1035 mcm-accelsim bash /accel-sim/llm_exp/experiments/@EXP@/@DIR@/launch.sh
# env: DRY=1 builds and checks every arm dir, launches nothing (the step-3 acceptance check is reported, not enforced).
# Arm recipe follows graphalytics_exp/experiments/basep0_4k_spmv344Mk1/launch_pcc_perf.sh:
#   base     = ref_batch4 config with the placement dump line replaced by a load of ../ref_batch4's map (config only)
#   ptwhop0  = base + "-icnt_ptw_ring_hop_latency 0" (config only)
#   <src>_pcc = <src> + "-gpgpu_cta_chiplet_cursor 1" (measurement arms)
# Each arm's run.sh runs in a child bash (the set -u launcher defect of 2026-09-12).
set -u
X=/accel-sim/llm_exp/experiments/@EXP@/@DIR@
N_KERNELS=@N_KERNELS@
MAX_INSN=@MAX_INSN@
MEASURE=@MEASURE@
TREE=/accel-sim/llm_exp/llm_sim_tree
BIN=$TREE/gpu-simulator/bin/release/accel-sim.out
SO=$TREE/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-12020/release/libcudart.so
BIN_MD5=f7666e0232c85ba921ae44eca75eb633
SO_MD5=e5b1840838bb890778373f8bc5807fcb
TRACE=/accel-sim/llm_exp/traces/@TRACE_DIR@/traces/kernelslist_@DIR@.g
TRACE_MD5=@TRACE_MD5@
SRC_CFG=/accel-sim/llm_exp/experiments/v3_decode_S256/ref_batch4
REF_CFG_MD5=608a40b4f2e77c77633fc983bb3b0678
BASE_PCC_MD5=3a22508dca9f6a3136f58a2205958cbe
PTWHOP0_PCC_MD5=30a4d477ada220d570e13e7e7ace9c47
S3_LOG=/accel-sim/llm_exp/experiments/s3_ldgsts_fix/s3_check_fix.log
S3_ACCEPT=/accel-sim/llm_exp/experiments/s3_ldgsts_fix/USER_ACCEPTED_S3-7.txt
MAP=placement_batch4.map
REF_TH=@REF_TH@; ARM_TH=@ARM_TH@; PARALLEL=@PARALLEL@; DRY=${DRY:-0}
LOG=$X/launch.log
log(){ echo "$(date '+%F %T') [$(basename $X)] $*" | tee -a $LOG; }
die(){ log "ABORT: $*"; echo "ABORT $*" >> $X/status.txt; exit 1; }
md5(){ md5sum "$1" | awk '{print $1}'; }

exec {LK}>$X/.launch.lock
flock -n $LK || { echo "another launch.sh holds $X/.launch.lock"; exit 1; }

# ---------- preflight: nothing runs unless every check passes ----------
if grep -q '^S3 FIX PASS' $S3_LOG 2>/dev/null; then log "step 3: S3 FIX PASS"
elif [ -s $S3_ACCEPT ]; then log "step 3: S3-7 FAIL accepted by the user ($S3_ACCEPT)"
elif [ "$DRY" = 1 ]; then log "step 3: no PASS and no user acceptance yet (DRY run continues)"
else die "step 3 (LDGSTS fix) has neither 'S3 FIX PASS' nor the user acceptance record $S3_ACCEPT"; fi
[ "$(md5 $BIN)" = $BIN_MD5 ] || die "binary md5 $(md5 $BIN) != $BIN_MD5"
[ "$(md5 $SO)" = $SO_MD5 ] || die "libcudart.so md5 $(md5 $SO) != $SO_MD5"
[ "$(md5 $TRACE)" = $TRACE_MD5 ] || die "trace list md5 $(md5 $TRACE) != $TRACE_MD5"
[ "$(md5 $SRC_CFG/gpgpusim.config)" = $REF_CFG_MD5 ] || die "V3 ref config md5 changed"
grep -qx -- '-gpgpu_max_insn 10000000000' $SRC_CFG/gpgpusim.config || die "ref config lacks -gpgpu_max_insn 10000000000"
grep -qx -- "-gmmu_placement_dump ./$MAP" $SRC_CFG/gpgpusim.config || die "ref config does not dump ./$MAP"
grep -qE '^-(gpgpu_cta_chiplet_cursor|icnt_ptw_ring_hop_latency|gpgpu_kernel_rebind)[[:space:]]' $SRC_CFG/gpgpusim.config \
  && die "ref config already sets cursor / ptw hop / rebind -- appended lines would not be the only value"
[ "$(grep -c '^MemcpyHtoD,' $TRACE)" -ge 518 ] && [ "$(grep -vc '^MemcpyHtoD,' $TRACE)" = $N_KERNELS ] \
  || die "trace list does not hold >= 518 copies + $N_KERNELS kernels"
# Instruction cap: 1e10 unless the user recorded an exception for this piece (CAP_EXCEPTION.txt in this dir).
if [ "$MAX_INSN" != 10000000000 ]; then
  if [ -s $X/CAP_EXCEPTION.txt ]; then log "instruction cap $MAX_INSN (user exception: $X/CAP_EXCEPTION.txt)"
  elif [ "$DRY" = 1 ]; then log "instruction cap $MAX_INSN needs $X/CAP_EXCEPTION.txt (missing; DRY run continues)"
  else die "instruction cap $MAX_INSN without a user exception record ($X/CAP_EXCEPTION.txt)"; fi
fi

ARMS=(ref_batch4); [ "$MEASURE" = 1 ] && ARMS+=(base_pcc ptwhop0_pcc)
for A in "${ARMS[@]}"; do [ -e $X/$A/run.log ] && die "$A already has run.log -- refusing to overwrite"; done

# ---------- arm directories ----------
mkdir -p $X/ref_batch4
cp $SRC_CFG/trace.config $SRC_CFG/config_ampere_islip.icnt $X/ref_batch4/
sed "s|^-gpgpu_max_insn 10000000000\$|-gpgpu_max_insn $MAX_INSN|" $SRC_CFG/gpgpusim.config > $X/ref_batch4/gpgpusim.config
if [ "$MAX_INSN" = 10000000000 ]; then cmp -s $SRC_CFG/gpgpusim.config $X/ref_batch4/gpgpusim.config || die "ref config differs from V3 ref"
else [ "$(diff $SRC_CFG/gpgpusim.config $X/ref_batch4/gpgpusim.config | grep -c '^[<>]')" = 2 ] \
  && diff $SRC_CFG/gpgpusim.config $X/ref_batch4/gpgpusim.config | grep -qx -- "> -gpgpu_max_insn $MAX_INSN" \
  || die "ref config is not V3 ref with only the cap line changed"; fi
if [ "$MEASURE" = 1 ]; then
  mkdir -p $X/base $X/ptwhop0 $X/base_pcc $X/ptwhop0_pcc
  for A in base ptwhop0 base_pcc ptwhop0_pcc; do cp $SRC_CFG/trace.config $SRC_CFG/config_ampere_islip.icnt $X/$A/; done
  sed "s|^-gmmu_placement_dump ./$MAP\$|-gmmu_placement_load ../ref_batch4/$MAP|" $X/ref_batch4/gpgpusim.config > $X/base/gpgpusim.config
  { cat $X/base/gpgpusim.config; echo "-icnt_ptw_ring_hop_latency 0"; } > $X/ptwhop0/gpgpusim.config
  { cat $X/base/gpgpusim.config; echo "-gpgpu_cta_chiplet_cursor 1"; } > $X/base_pcc/gpgpusim.config
  { cat $X/ptwhop0/gpgpusim.config; echo "-gpgpu_cta_chiplet_cursor 1"; } > $X/ptwhop0_pcc/gpgpusim.config
  one_line_added(){ [ "$(diff "$1" "$2" | grep -c '^[<>]')" = 1 ] && diff "$1" "$2" | grep -qx -- "> $3"; }
  [ "$(diff $X/ref_batch4/gpgpusim.config $X/base/gpgpusim.config | grep -c '^[<>]')" = 2 ] \
    && diff $X/ref_batch4/gpgpusim.config $X/base/gpgpusim.config | grep -qx -- "> -gmmu_placement_load ../ref_batch4/$MAP" \
    || die "base is not ref_batch4 with only the placement line changed"
  one_line_added $X/base/gpgpusim.config $X/ptwhop0/gpgpusim.config "-icnt_ptw_ring_hop_latency 0" || die "ptwhop0 config"
  one_line_added $X/base/gpgpusim.config $X/base_pcc/gpgpusim.config "-gpgpu_cta_chiplet_cursor 1" || die "base_pcc config"
  one_line_added $X/ptwhop0/gpgpusim.config $X/ptwhop0_pcc/gpgpusim.config "-gpgpu_cta_chiplet_cursor 1" || die "ptwhop0_pcc config"
  if [ "$MAX_INSN" = 10000000000 ]; then
    [ "$(md5 $X/base_pcc/gpgpusim.config)" = $BASE_PCC_MD5 ] || die "base_pcc config md5 differs from the graphalytics/SpMV arms"
    [ "$(md5 $X/ptwhop0_pcc/gpgpusim.config)" = $PTWHOP0_PCC_MD5 ] || die "ptwhop0_pcc config md5 differs from the graphalytics/SpMV arms"
  fi
  printf 'CONFIG ONLY (not run). ref_batch4 config with the placement dump replaced by -gmmu_placement_load ../ref_batch4/%s\n' $MAP > $X/base/README_ARM.md
  printf 'CONFIG ONLY (not run). base config + -icnt_ptw_ring_hop_latency 0\n' > $X/ptwhop0/README_ARM.md
fi

write_run(){ # $1 arm, $2 threads
  cat > $X/$1/run.sh <<EOF
#!/bin/bash
# $(basename $X) arm $1 (generated by ../launch.sh). Binary: llm_sim_tree (patch 29). Threads $2, unpinned.
# Trace: piece $(basename $X) ($N_KERNELS kernels) of the @EXP@ capture. max_insn $MAX_INSN.
TREE=$TREE
TH=$2
TRACE=$TRACE
cd $X/$1
source \$TREE/gpu-simulator/setup_environment.sh release >/dev/null 2>&1
{ echo "step8 piece $(basename $X) tree=\$TREE arm=$1 threads=\$TH trace=\$TRACE max_insn=$MAX_INSN"; date
  md5sum \$TREE/gpu-simulator/bin/release/accel-sim.out \$GPGPUSIM_ROOT/lib/\$GPGPUSIM_CONFIG/libcudart.so gpgpusim.config trace.config config_ampere_islip.icnt \$TRACE
  ldd \$TREE/gpu-simulator/bin/release/accel-sim.out | grep cudart
} > provenance.txt
export ACCELSIM_NO_PIN=1
t0=\$SECONDS
env OMP_NUM_THREADS=\$TH ACCELSIM_ALLOW_UNVERIFIED_THREADS=1 nice -n 10 \\
  \$TREE/gpu-simulator/bin/release/accel-sim.out \\
  -config ./gpgpusim.config -config ./trace.config -trace \$TRACE > run.log 2>&1
echo "EXIT=\$? SECONDS=\$((SECONDS-t0)) load=\$(cut -d' ' -f1 /proc/loadavg)" > exit_code.txt
EOF
  chmod +x $X/$1/run.sh; }
write_run ref_batch4 $REF_TH
[ "$MEASURE" = 1 ] && { write_run base_pcc $ARM_TH; write_run ptwhop0_pcc $ARM_TH; }
log "preflight OK: piece=$(basename $X) kernels=$N_KERNELS max_insn=$MAX_INSN measure=$MEASURE bin=$BIN_MD5 so=$SO_MD5 trace=$TRACE_MD5 ref_th=$REF_TH arm_th=$ARM_TH parallel=$PARALLEL dry=$DRY"
[ "$DRY" = 1 ] && { log "DRY run finished -- nothing launched"; exit 0; }

# ---------- ref_batch4 ----------
echo "RUNNING ref_batch4 $(date '+%F %T')" >> $X/status.txt
log "ref_batch4 start (threads $REF_TH)"
bash $X/ref_batch4/run.sh
log "ref_batch4 end: $(cat $X/ref_batch4/exit_code.txt 2>/dev/null)"
grep -q '^EXIT=0 ' $X/ref_batch4/exit_code.txt 2>/dev/null || die "ref_batch4 not EXIT=0"
grep -q "^\[PLACEMENT\] dump: ./$MAP" $X/ref_batch4/run.log && [ -s $X/ref_batch4/$MAP ] || die "ref_batch4 placement map missing"
[ "$MEASURE" = 1 ] || { echo "DONE $(date '+%F %T')" >> $X/status.txt; log "ref_batch4 only -- finished"; exit 0; }

# ---------- measurement arms ----------
if [ "$PARALLEL" = 1 ]; then
  echo "RUNNING base_pcc ptwhop0_pcc $(date '+%F %T')" >> $X/status.txt
  log "base_pcc + ptwhop0_pcc start side by side (threads $ARM_TH each)"
  bash $X/base_pcc/run.sh & P1=$!
  bash $X/ptwhop0_pcc/run.sh & P2=$!
  wait $P1; wait $P2
else
  for A in base_pcc ptwhop0_pcc; do
    echo "RUNNING $A $(date '+%F %T')" >> $X/status.txt
    log "$A start (threads $ARM_TH, serial)"
    bash $X/$A/run.sh
    log "$A returned: $(cat $X/$A/exit_code.txt 2>/dev/null)"
  done
fi
for A in base_pcc ptwhop0_pcc; do
  L=$X/$A/run.log; hop=-1; bad=""
  [ $A = ptwhop0_pcc ] && hop=0
  grep -q '^EXIT=0 ' $X/$A/exit_code.txt 2>/dev/null || bad="$bad exit"
  grep -qE '^-gpgpu_cta_chiplet_cursor[[:space:]]+1 ' $L || bad="$bad cursor"
  grep -qE "^-icnt_ptw_ring_hop_latency[[:space:]]+$hop " $L || bad="$bad ptw_hop"
  grep -qE '^-gpgpu_kernel_rebind[[:space:]]+0 ' $L || bad="$bad rebind"
  grep -q "^\[PLACEMENT\] load: ../ref_batch4/$MAP" $L || bad="$bad placement_load"
  grep -q 'per-chiplet cursors on' $L || bad="$bad pcc_line"
  if [ -z "$bad" ]; then log "$A end: $(cat $X/$A/exit_code.txt) option check OK (cursor=1 ptw_hop=$hop rebind=0 placement=load PCC on)"
  else log "$A end: $(cat $X/$A/exit_code.txt 2>/dev/null) option check FAILED:$bad"; fi
done
echo "DONE $(date '+%F %T')" >> $X/status.txt
log "all arms finished"
