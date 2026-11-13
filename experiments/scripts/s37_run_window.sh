#!/bin/bash
# Step 37: run s37_sharing_class.py on every kernel of one kernelslist (window), P kernels in parallel, niced.
# Usage: s37_run_window.sh <capture_dir> <kernelslist> <out_dir> [P=4]
# Skips kernels whose k<id>.json already exists, so a rerun resumes. Log: <out_dir>/run.log
set -u
CAP=$1; KL=$2; OUT=$3; P=${4:-4}
S=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
grep '^kernel-' "$KL" | while read -r f; do
  k=$(echo "$f" | sed -E 's/kernel-([0-9]+)-.*/\1/')
  [ -s "$OUT/k$k.json" ] || echo "$k $CAP/traces/$f"
done | xargs -P "$P" -L 1 bash -c 't0=$SECONDS; nice -n 10 python3 -I '"$S"'/s37_sharing_class.py "$1" '"$CAP"'/va_map.json '"$OUT"'/k$0 >> '"$OUT"'/run.log 2>&1; echo "k$0 rc=$? $((SECONDS-t0)) s" >> '"$OUT"'/run.log'
echo "window done $(date '+%F %T')" >> "$OUT/run.log"
