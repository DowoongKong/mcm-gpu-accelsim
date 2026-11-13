#!/bin/bash
# Step 26 driver: run s22_ft_predict2.py --config for every kernel of a capture's kernelslist (post-processed .traceg.xz),
# at most P processes at a time, nice 10 (CPU headroom rule), one k<i>.json + k<i>.log per kernel.
# Usage: s26_run_predict.sh <capture dir> <gpgpusim.config> <out dir> [P=3] [k list, default: all]
set -u
CAP=$1; CFG=$2; OUT=$3; P=${4:-3}; shift 4 2>/dev/null || shift $#
H=/home/user/accel-sim-docker/llm_exp
KL=$CAP/traces/kernelslist.g; [ -s $KL ] || KL=$(ls $CAP/traces/kernelslist*.g | head -1)   # prefer the full list (s34 fix 11-12)
[ -s "$KL" ] || { echo "no post-processed kernelslist (*.g) in $CAP/traces"; exit 1; }
N=$(grep -c 'kernel-' $KL)
mkdir -p $OUT
if [ $# -gt 0 ]; then KS="$*"; else KS=$(seq 1 $N); fi
echo "$(date '+%F %T') predict: $KL ($N kernels) config $CFG -> $OUT, P=$P, k = $(echo $KS | tr '\n' ' ')" | tee -a $OUT/run.log
printf '%s\n' $KS | xargs -P $P -I{} bash -c "t0=\$SECONDS; nice -n 10 python3 $H/scripts/s22_ft_predict2.py $KL {} $OUT/k{}.json --config $CFG > $OUT/k{}.log 2>&1; echo \"k{} EXIT=\$? \$((SECONDS-t0)) s: \$(tail -n 1 $OUT/k{}.log)\" >> $OUT/run.log"
echo "$(date '+%F %T') predict done: $(ls $OUT/k*.json 2>/dev/null | wc -l)/$N json" | tee -a $OUT/run.log
