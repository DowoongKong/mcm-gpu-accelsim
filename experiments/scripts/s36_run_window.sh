#!/bin/bash
# s36_run_window.sh <capture dir> <window name> <out dir> [parallel=1] -- run s36_cta_reuse.py on every kernel of one
# capture window (ids from capture_info.json window_kernels), P at a time under nice, then build KERNELS.md.
# Log: <out dir>/run.log. Skips kernels whose JSON already exists.
set -u
H=/home/user/accel-sim-docker/llm_exp
C=$1; W=$2; O=$3; P=${4:-1}; mkdir -p "$O"; L=$O/run.log
ids=$(python3 -I -c "
import json,sys; ci=json.load(open('$C/capture_info.json'))
print(' '.join(str(k['id']) for k in ci['window_kernels'] if k['window']=='$W'))")
echo "$(date '+%F %T') start window $W capture $C kernels: $ids (P=$P)" >> "$L"
for k in $ids; do
  f=$(ls "$C"/traces/kernel-$k-*.traceg.xz 2>/dev/null | head -n 1); [ -n "$f" ] || { echo "k$k: trace missing" >> "$L"; continue; }
  [ -s "$O/k$k.json" ] && continue
  echo "$k $f"
done | xargs -P "$P" -L 1 bash -c 'k=$0; f=$1; t0=$SECONDS; nice -n 10 python3 -I '"$H"'/scripts/s36_cta_reuse.py "$f" '"$C"'/va_map.json '"$O"'/k$k.json > '"$O"'/k$k.log 2>&1; echo "$(date "+%F %T") k$k exit=$? $((SECONDS-t0)) s: $(tail -n 1 '"$O"'/k$k.log)" >> '"$L"
python3 -I $H/scripts/s36_summarize.py "$O" "$C/capture_info.json" "$W" >> "$L" 2>&1
echo "$(date '+%F %T') done window $W -> $O/KERNELS.md" >> "$L"
