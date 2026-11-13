#!/bin/bash
# s31 (docs/s31_sim_speed_review_llm.ko.md section 6): make an experiment config for llm_sim_tree_s31 from an existing
# arm config. The ONLY change is -gpgpu_perf_sim_memcpy 0 (the memcpy L2 warm-up fills the cache with untranslated
# destination VAs, so the MCM experiments run without it). Fails unless exactly that one line differs.
# Usage: s31_arm_config.sh <src gpgpusim.config> <dst gpgpusim.config>
set -eu
src=$1; dst=$2
[ -s "$src" ] || { echo "no $src"; exit 1; }
[ -e "$dst" ] && { echo "refusing to overwrite $dst"; exit 1; }
n=$(grep -c '^-gpgpu_perf_sim_memcpy ' "$src" || true)
if [ "$n" = 1 ]; then sed 's#^-gpgpu_perf_sim_memcpy .*#-gpgpu_perf_sim_memcpy 0#' "$src" > "$dst"
elif [ "$n" = 0 ]; then { cat "$src"; echo "-gpgpu_perf_sim_memcpy 0"; } > "$dst"
else echo "-gpgpu_perf_sim_memcpy appears $n times in $src"; exit 1; fi
grep -qx -- '-gpgpu_perf_sim_memcpy 0' "$dst" || { echo "check failed"; exit 1; }
d=$(diff "$src" "$dst" | grep -c '^[<>]' || true); [ "$d" -le 2 ] || { echo "unexpected diff ($d lines)"; exit 1; }
echo "ok: $dst (-gpgpu_perf_sim_memcpy 0)"
