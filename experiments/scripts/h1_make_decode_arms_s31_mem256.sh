#!/bin/bash
# h1_make_decode_arms_s31_mem256.sh <capture dir (host)> <first decode kernel id> <last decode kernel id>  -- s31 variant: 256 GB memory (Llama-70B, docs/s30 section 11) + exact placement replay + memcpy warm-up off, tree llm_sim_tree_s31
# Builds sim/{ref_mcm,base_mcm,ptwhop0_mcm} for a v2 capture from the hypothesis-1 batch-4 recipe (configs byte-identical to the s31 templates
# llm_hypothesis1/_templates/mem256_s31_20261112/sim/<arm> = v1 b4 arm configs + -gpu_mem_size_mb 262144 + 38-bit address mapping + -gpgpu_perf_sim_memcpy 0 (+ -gmmu_placement_exact 1 and placement_check.map dump in base/ptwhop0); model-independent), traces/kernelslist_decode.g
# (MemcpyHtoD lines + the decode-window kernels), run.sh per arm and sim/launch_h1.sh (preflight + cwd-based duplicate check).
set -u
CAP=$1; A=$2; B=$3; SIMDIR=${4:-sim}   # 4th arg: arm dir name (default sim; use sim_noprefill for s31 re-runs of captures whose sim/ holds old-tree results)
H=/home/user/accel-sim-docker/llm_exp; V4=$H/llm_hypothesis1/_templates/mem256_s31_20261112/sim
REL=${CAP#$H/}; CP=/accel-sim/llm_exp/$REL; TAG=$(basename $(dirname $CAP))/$(basename $CAP)
[ -s $CAP/traces/kernelslist.g ] || { echo "no post-processed kernelslist.g in $CAP"; exit 1; }
grep -E "MemcpyHtoD" $CAP/traces/kernelslist.g > $CAP/traces/kernelslist_decode.g
grep -vE "MemcpyHtoD" $CAP/traces/kernelslist.g | awk -v a=$A -v b=$B '{match($0,/kernel-([0-9]+)-/,m); if (m[1]+0>=a && m[1]+0<=b) print}' >> $CAP/traces/kernelslist_decode.g
echo "kernelslist_decode.g: $(grep -vc MemcpyHtoD $CAP/traces/kernelslist_decode.g) kernels + $(grep -c MemcpyHtoD $CAP/traces/kernelslist_decode.g) copies"
for arm in ref_mcm base_mcm ptwhop0_mcm; do X=$CAP/$SIMDIR/$arm; mkdir -p $X; cp -p $V4/$arm/gpgpusim.config $V4/$arm/trace.config $V4/$arm/config_ampere_islip.icnt $X/
cat > $X/run.sh <<EOF2
#!/bin/bash
# h1 $TAG / $arm: decode layer 0 window (kernels $A-$B), hypothesis-1 batch-4 recipe: 64 KiB pages, KERNEL_PARTITION + per-chiplet
# cursors + L2 TLB set hash 1, L1 TLB 64K 32, L2 TLB 64K 1024, merge caps unlimited, cap 1e10. Config = v1 arm config + 256 GB memory (s30 section 11) + s31 options (memcpy warm-up off, exact map replay); byte-identical to
# _templates/mem256_s31_20261112/sim/$arm (ref dumps ./placement_mcm.map; base/ptwhop0 load ../ref_mcm/placement_mcm.map with -gmmu_placement_exact 1 and dump ./placement_check.map; ptwhop0 adds
# -icnt_ptw_ring_hop_latency 0). Child shell, no set -u. Threads: TH (default 16).
TREE=/accel-sim/llm_exp/llm_sim_tree_s31
TH=\${TH:-16}
TRACE=$CP/traces/kernelslist_decode.g
cd $CP/$SIMDIR/$arm
source \$TREE/gpu-simulator/setup_environment.sh release >/dev/null 2>&1
{ echo "h1 $TAG arm=$arm tree=\$TREE threads=\$TH trace=\$TRACE max_insn=10000000000"; date
  md5sum \$TREE/gpu-simulator/bin/release/accel-sim.out \$GPGPUSIM_ROOT/lib/\$GPGPUSIM_CONFIG/libcudart.so gpgpusim.config trace.config config_ampere_islip.icnt \$TRACE
  ldd \$TREE/gpu-simulator/bin/release/accel-sim.out | grep cudart
} > provenance.txt
export ACCELSIM_NO_PIN=1
t0=\$SECONDS
env OMP_NUM_THREADS=\$TH ACCELSIM_ALLOW_UNVERIFIED_THREADS=1 nice -n 10 \\
  \$TREE/gpu-simulator/bin/release/accel-sim.out -config ./gpgpusim.config -config ./trace.config -trace \$TRACE > run.log 2>&1
echo "EXIT=\$? SECONDS=\$((SECONDS-t0)) load=\$(cut -d' ' -f1 /proc/loadavg)" > exit_code.txt
EOF2
chmod +x $X/run.sh; done
cat > $CAP/$SIMDIR/launch_h1.sh <<EOF3
#!/bin/bash
# launch_h1.sh <ref_mcm|base_mcm|ptwhop0_mcm> -- preflight then start sim/<arm>/run.sh detached in mcm-accelsim (uid 1035).
set -u
A=\$1; HP=$CAP; CP=$CP; TREE=$H/llm_sim_tree_s31; V4=$V4
BIN_MD5=cee287166ea343c67ccc86d33628affa; SO_MD5=8f0bffefff5176210632181b57c3f0eb
md5(){ md5sum "\$1" | awk '{print \$1}'; }
die(){ echo "ABORT: \$*" | tee -a \$HP/$SIMDIR/launch.log; exit 1; }
X=\$HP/$SIMDIR/\$A
exec {LK}>\$HP/$SIMDIR/.launch_\$A.lock; flock -n \$LK || die "\$A launch already in progress"
[ "\$(md5 \$TREE/gpu-simulator/bin/release/accel-sim.out)" = \$BIN_MD5 ] || die "binary md5 changed"
[ "\$(md5 \$TREE/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-12020/release/libcudart.so)" = \$SO_MD5 ] || die "libcudart md5 changed"
cmp -s \$V4/\$A/gpgpusim.config \$X/gpgpusim.config || die "config differs from the s31 template \$A"
grep -q '^S6 TRACES PASS' \$HP/h1_check_traces_decode_layer0_*.log 2>/dev/null || die "decode-window S6 PASS log missing"
n=\$(grep -vc MemcpyHtoD \$HP/traces/kernelslist_decode.g); [ "\$n" -ge 10 ] && [ "\$n" -le 40 ] || die "decode trace list holds \$n kernels"
if [ \$A != ref_mcm ]; then grep -q '^EXIT=0' \$HP/$SIMDIR/ref_mcm/exit_code.txt 2>/dev/null || die "ref_mcm not finished"; grep -q '^# page_shift 16' \$HP/$SIMDIR/ref_mcm/placement_mcm.map || die "ref map missing"; fi
[ -e \$X/exit_code.txt ] && die "\$A already ran"
for pid in \$(pgrep -f "accel-sim.out -config"); do [ "\$(readlink /proc/\$pid/cwd 2>/dev/null)" = "\$CP/$SIMDIR/\$A" ] && die "\$A is already running (pid \$pid)"; done
echo "\$(date '+%F %T') launch \$A (TH=\${TH:-16})" | tee -a \$HP/$SIMDIR/launch.log
docker exec -d -u 1035:1035 -e TH=\${TH:-16} mcm-accelsim bash \$CP/$SIMDIR/\$A/run.sh; sleep 15
EOF3
chmod +x $CAP/$SIMDIR/launch_h1.sh; bash -n $CAP/$SIMDIR/launch_h1.sh && bash -n $CAP/$SIMDIR/ref_mcm/run.sh && echo "arms ready under $CAP/$SIMDIR"
