#!/bin/bash
# Step 37: make <arm>_l2log next to an existing simulator arm: same gpgpusim.config / trace.config / icnt (byte copies),
# same run.sh except the arm name, its own directory, and ACCELSIM_L2MISS_LOG=./l2miss.log (walk event log; timing-neutral,
# s29 gate L2M-1). Does not launch anything.
# Usage: s37_make_l2log_arm.sh <host arm dir>
set -eu
SRC=$(cd "$1" && pwd); DST=${SRC}_l2log
H=/home/user/accel-sim-docker; CSRC=${SRC/#$H//accel-sim}; CDST=${DST/#$H//accel-sim}
[ -e "$DST" ] && { echo "exists: $DST"; exit 1; }
mkdir "$DST"
cp -p "$SRC"/gpgpusim.config "$SRC"/trace.config "$SRC"/config_ampere_islip.icnt "$DST"/
A=$(basename "$SRC")
# The source run.sh may still cd into the pre-rename path (sim/ -> sim_noprefill/); always cd into the new arm dir.
sed -E -e "s#^cd .*#cd $CDST#" -e "s#arm=$A #arm=${A}_l2log #" \
       -e "s#^export ACCELSIM_NO_PIN=1\$#export ACCELSIM_NO_PIN=1; export ACCELSIM_L2MISS_LOG=./l2miss.log   \# s37: walk event log (timing-neutral, s29 L2M-1)#" \
       "$SRC/run.sh" > "$DST/run.sh"
chmod +x "$DST/run.sh"
grep -q "ACCELSIM_L2MISS_LOG" "$DST/run.sh" || { echo "log export not inserted"; exit 1; }
grep -q "^cd $CDST\$" "$DST/run.sh" || { echo "cd not rewritten"; exit 1; }
cmp -s "$SRC/gpgpusim.config" "$DST/gpgpusim.config" && echo "made $DST (config identical to $A)"
