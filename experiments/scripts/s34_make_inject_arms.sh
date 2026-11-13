#!/bin/bash
# s34_make_inject_arms.sh <decode capture dir> <placement map (host path)>  -- sim_prefillchunked/{base_prefill,ptwhop0_prefill}:
# base_prefill = s31 base template loading the predicted map (owner-only, -gmmu_placement_exact 0: no PFNs) and dumping its
# end state (prefill-injected + decode first touch, v3 with PFNs) to placement_end.map; ptwhop0_prefill = s31 ptwhop0 template
# loading ../base_prefill/placement_end.map with -gmmu_placement_exact 1 (user rule 11-12 01:25: the baseline's saved placement
# is inserted before ptwhop0) and dumping placement_check.map for the X-3 cmp; run.sh = the capture's sim/ref_mcm/run.sh
# with the arm name swapped. Preflight launcher sim/launch_inj.sh <arm>. Nothing is launched here.
set -u
CAP=$1; MAP=$2; H=/home/user/accel-sim-docker/llm_exp; T=$H/llm_hypothesis1/_templates/mem128_s31_20261111/sim
RC=$CAP/sim_noprefill/ref_mcm/gpgpusim.config; [ -s $RC ] || RC=$CAP/sim/ref_mcm/gpgpusim.config
grep -q '^-gpu_mem_size_mb 262144' $RC 2>/dev/null && T=$H/llm_hypothesis1/_templates/mem256_s31_20261112/sim   # Llama-70B: 256 GB templates
[ -s $MAP ] || { echo "map missing: $MAP"; exit 1; }; SRC=$CAP/sim_noprefill; [ -d $SRC ] || SRC=$CAP/sim; [ -x $SRC/ref_mcm/run.sh ] || { echo "no ref_mcm/run.sh under $CAP/sim_noprefill or sim/"; exit 1; }
OUTD=$CAP/sim_prefillchunked
CMAP=/accel-sim/llm_exp/${MAP#$H/}
for arm in base_prefill ptwhop0_prefill; do src=base_mcm; [ $arm = ptwhop0_prefill ] && src=ptwhop0_mcm; X=$OUTD/$arm; mkdir -p $X
  if [ $arm = base_prefill ]; then   # predicted map (owner only, no PFN) -> premap by owner; dumps the full end state (v3, PFNs) for ptwhop0
    sed -e "s#^-gmmu_placement_load ../ref_mcm/placement_mcm.map#-gmmu_placement_load $CMAP#" -e 's#^-gmmu_placement_exact 1#-gmmu_placement_exact 0#' -e 's#^-gmmu_placement_dump ./placement_check.map#-gmmu_placement_dump ./placement_end.map#' $T/$src/gpgpusim.config > $X/gpgpusim.config; want=6
  else                                # user rule 11-12 01:25: ptwhop0 replays the BASELINE's saved placement (prefill-injected + decode first touch), PFN-exact
    sed -e "s#^-gmmu_placement_load ../ref_mcm/placement_mcm.map#-gmmu_placement_load ../base_prefill/placement_end.map#" $T/$src/gpgpusim.config > $X/gpgpusim.config; want=2; fi
  n=$(diff $T/$src/gpgpusim.config $X/gpgpusim.config | grep -c '^[<>]'); [ "$n" = $want ] || { echo "$arm: unexpected diff ($n lines, want $want)"; exit 1; }
  cp -p $T/$src/trace.config $T/$src/config_ampere_islip.icnt $X/
  sed -e "s#arm=ref_mcm#arm=$arm#" -e "s#/sim[a-z_]*/ref_mcm#/sim_prefillchunked/$arm#" -e "s#h1 \(.*\) / ref_mcm:#h1 \1 / $arm:#" $SRC/ref_mcm/run.sh > $X/run.sh; chmod +x $X/run.sh
  grep -q "sim_prefillchunked/$arm" $X/run.sh || { echo "$arm: run.sh rewrite failed"; exit 1; }; done
cat > $OUTD/launch_inj.sh <<EOT
#!/bin/bash
# launch_inj.sh <base_prefill|ptwhop0_prefill> -- preflight (map present, <=2 sims system-wide, not already run/running), then run.sh detached in the container.
set -u
A=\$1; HP=$CAP; CP=/accel-sim/llm_exp/${CAP#$H/}; X=\$HP/sim_prefillchunked/\$A
die(){ echo "ABORT: \$*" | tee -a \$HP/sim_prefillchunked/launch.log; exit 1; }
[ -s $MAP ] || die "map missing"
if [ \$A = base_prefill ]; then grep -q "^-gmmu_placement_load $CMAP" \$X/gpgpusim.config || die "base config does not load the s34 map"
else grep -q '^EXIT=0' \$HP/sim_prefillchunked/base_prefill/exit_code.txt 2>/dev/null || die "base_prefill not finished"
  M=\$HP/sim_prefillchunked/base_prefill/placement_end.map; grep -q '^# gpgpu-sim placement snapshot v3' \$M 2>/dev/null && grep -q '^R 0x' \$M || die "base_prefill end map missing or not v3"
  [ "\$(grep -c 'ffffffffffffffff' \$M)" = 0 ] || die "base_prefill end map has PFN -1 entries"; grep -q '^-gmmu_placement_load ../base_prefill/placement_end.map' \$X/gpgpusim.config || die "ptwhop0 config does not load the baseline map"; fi
[ -e \$X/exit_code.txt ] && die "\$A already ran"
n=\$(ps -o stat= -C accel-sim.out | grep -vc '^Z'); [ "\$n" -lt 2 ] || die "\$n simulations already running (limit 2)"
for pid in \$(pgrep -x accel-sim.out); do [ "\$(readlink /proc/\$pid/cwd 2>/dev/null)" = "\$CP/sim_prefillchunked/\$A" ] && die "\$A is already running"; done
echo "\$(date '+%F %T') launch \$A (TH=\${TH:-16}, map $CMAP)" | tee -a \$HP/sim_prefillchunked/launch.log
docker exec -d -u 1035:1035 -e TH=\${TH:-16} mcm-accelsim bash \$CP/sim_prefillchunked/\$A/run.sh; sleep 15
EOT
chmod +x $OUTD/launch_inj.sh; bash -n $OUTD/launch_inj.sh && echo "injection arms ready: $OUTD/{base_prefill,ptwhop0_prefill} (not launched)"
