#!/bin/bash
# s34_vb_kernel_override.sh <prefill capture> <kernel position in kernelslist (1-based, kernel- lines only)> <out dir> [TH=8]
# Reference-simulate ONE prefill kernel (MemcpyHtoD lines + that kernel) with the s31 ref template (standard cap) and dump its
# placement: the simulated owners replace the predictor's for split-K kernels (s26 V-a/--override mechanism). Waits until fewer
# than 2 accel-sim.out are alive. Output: <out>/placement_mcm.map, run.log, exit_code.txt. Idempotent: returns at once if EXIT=0 exists.
set -u
PRE=$1; K=$2; O=$3; TH=${4:-8}; H=/home/user/accel-sim-docker/llm_exp; T=$H/llm_hypothesis1/_templates/mem128_s31_20261111/sim/ref_mcm
grep -q '^EXIT=0' $O/exit_code.txt 2>/dev/null && { echo "override sim already done: $O"; exit 0; }
mkdir -p $O; KL=$PRE/traces/kernelslist.g; [ -s $KL ] || KL=$(ls $PRE/traces/kernelslist*.g | head -1); F=$(grep 'kernel-' $KL | sed -n "${K}p"); [ -n "$F" ] || { echo "no kernel at position $K"; exit 1; }
{ grep MemcpyHtoD $KL; echo $F; } > $PRE/traces/kernelslist_vb_k$K.g
cp -p $T/gpgpusim.config $T/trace.config $T/config_ampere_islip.icnt $O/
CP=/accel-sim/llm_exp/${PRE#$H/}; CO=/accel-sim/llm_exp/${O#$H/}
cat > $O/run.sh <<EOT
#!/bin/bash
TREE=/accel-sim/llm_exp/llm_sim_tree_s31; TH=\${TH:-$TH}; TRACE=$CP/traces/kernelslist_vb_k$K.g
cd $CO; source \$TREE/gpu-simulator/setup_environment.sh release >/dev/null 2>&1; export ACCELSIM_NO_PIN=1
{ echo "s34 single-kernel ref (override source) k=$K file=$F tree=\$TREE threads=\$TH"; date; md5sum \$TREE/gpu-simulator/bin/release/accel-sim.out gpgpusim.config; } > provenance.txt
t0=\$SECONDS; env OMP_NUM_THREADS=\$TH ACCELSIM_ALLOW_UNVERIFIED_THREADS=1 nice -n 10 \$TREE/gpu-simulator/bin/release/accel-sim.out -config ./gpgpusim.config -config ./trace.config -trace \$TRACE > run.log 2>&1
echo "EXIT=\$? SECONDS=\$((SECONDS-t0))" > exit_code.txt
EOT
chmod +x $O/run.sh
while [ "$(ps -o stat= -C accel-sim.out | grep -vc '^Z')" -ge 2 ]; do sleep 60; done
echo "$(date '+%F %T') override sim launch k$K ($F, TH $TH)"; docker exec -u 1035:1035 -e TH=$TH mcm-accelsim bash $CO/run.sh
echo "$(date '+%F %T') override sim k$K: $(cat $O/exit_code.txt) D=$(grep -c '^D ' $O/placement_mcm.map 2>/dev/null)"; grep -q '^EXIT=0' $O/exit_code.txt
