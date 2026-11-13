#!/bin/bash
# s6_parallel.sh <capture dir> <window name> <first> <last> [P=4] -- run scripts/s6_check_traces.py on P kernel sub-ranges at once (nice 10),
# merge the part logs into the standard h1_check_traces_<window>_<first>_<last>.log with one final "S6 TRACES PASS|FAIL (n/n)" line.
# Same checks as the serial run (P-1 is global and cheap; P-2 is per kernel), 3-4x faster on the long prefill windows.
set -u
T=$1; W=$2; A=$3; B=$4; P=${5:-4}; H=/home/user/accel-sim-docker/llm_exp; OPC=${OPCODES:-$H/scripts/opcodes_table.txt}
[ -s "$OPC" ] || OPC=$(grep -oE 'OPCODES=[^ ]+' $H/scripts/h1_capture_v2_chain_c.sh | head -1 | cut -d= -f2 | tr -d '"'); OPC=$(eval echo $OPC)
[ -s "$OPC" ] || { echo "opcode table not found ($OPC)"; exit 1; }
OUT=$T/h1_check_traces_${W}_${A}_${B}.log; TMP=$T/.s6_parts_${W}; mkdir -p $TMP; n=$((B-A+1)); step=$(( (n + P - 1) / P )); pids=""
for i in $(seq 0 $((P-1))); do a=$((A + i*step)); b=$((a + step - 1)); [ $b -gt $B ] && b=$B; [ $a -gt $B ] && break
  nice -n 10 python3 $H/scripts/s6_check_traces.py $T/traces $OPC $T/h1_postprocess.log $T/h1_postprocess_exit.txt $a $b > $TMP/part_${a}_${b}.log 2>&1 & pids="$pids $!"; done
fail=0; for p in $pids; do wait $p || fail=1; done
{ echo "# merged from $P parallel sub-ranges (scripts/s6_parallel.sh) $(date '+%F %T')"; np=0; nt=0
  for f in $(ls $TMP/part_*.log | sort -t_ -k2 -n); do echo "## $(basename $f)"; grep -vE '^S6 TRACES' $f; r=$(grep '^S6 TRACES' $f); echo "$r"
    c=$(echo "$r" | grep -oE '\([0-9]+/[0-9]+\)' | tr -d '()'); np=$((np + ${c%%/*})); nt=$((nt + ${c##*/})); done
  if [ $fail = 0 ] && ! grep -q '^S6 TRACES FAIL' $TMP/part_*.log && [ $np = $nt ]; then echo "S6 TRACES PASS ($np/$nt)"; else echo "S6 TRACES FAIL ($np/$nt)"; fi; } > $OUT
tail -n 1 $OUT
