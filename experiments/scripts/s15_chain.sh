#!/bin/bash
# Step 15 chain (docs/s15_chiplet_hash_apply.ko.md section 3), run on the host:
#   1. wait for the fast arms of batch 1 (a2_p64k_idx0, e1_p64k, e1_p4k)
#   2. launch E2 refs (ref4k, ref64k) and the E3 pair (idx0, idx7)
#   3. when both E2 refs end with EXIT=0, build and launch the E2 measurement arms
#      (base4k_pcc, base64k_pcc on the fast tree, ideal4k_pcc, ideal64k_pcc on the serial tree)
set -u
R=/home/user/accel-sim-docker
X=$R/llm_exp/experiments/s15_hash
XC=/accel-sim/llm_exp/experiments/s15_hash
D=/accel-sim/llm_exp/experiments/deepseek_llama8b_b4s256_steps/s12_D1L1
TD=/accel-sim/llm_exp/traces/s10_deepseek_b4s256_steps/traces/kernelslist_D1-L1.g
M=$R/llm_exp/scripts/s15_make_arm.sh
log(){ echo "$(date '+%F %T') $*" >> $X/launch.log; }
launch(){ docker exec -d -u 1035:1035 mcm-accelsim bash $XC/$1/run.sh; log "launched $1"; }
done_ok(){ [ -f $X/$1/exit_code.txt ]; }
export HOSTROOT=$R

until done_ok a2_p64k_idx0 && done_ok e1_p64k && done_ok e1_p4k; do sleep 60; done
log "batch-1 fast arms finished"
for a in e2_d1l1/ref4k e2_d1l1/ref64k e3_sssp_k14/idx0 e3_sssp_k14/idx7; do launch $a; done

until done_ok e2_d1l1/ref4k && done_ok e2_d1l1/ref64k; do sleep 60; done
for a in ref4k ref64k; do
  grep -q "^EXIT=0" $X/e2_d1l1/$a/exit_code.txt || { log "ABORT: e2_d1l1/$a did not exit 0"; exit 1; }
done
log "E2 refs finished"
bash $M $XC/e2_d1l1/base4k_pcc  $D/base4k_pcc  7 fast   8 $TD >> $X/launch.log 2>&1 &&
bash $M $XC/e2_d1l1/base64k_pcc $D/base64k_pcc 7 fast   8 $TD >> $X/launch.log 2>&1 &&
bash $M $XC/e2_d1l1/ideal4k_pcc  $D/ideal4k_pcc  7 serial 1 $TD >> $X/launch.log 2>&1 &&
bash $M $XC/e2_d1l1/ideal64k_pcc $D/ideal64k_pcc 7 serial 1 $TD >> $X/launch.log 2>&1 || { log "ABORT: arm build failed"; exit 1; }
for a in base4k_pcc base64k_pcc ideal4k_pcc ideal64k_pcc; do launch e2_d1l1/$a; done
log "chain done"
