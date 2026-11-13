#!/bin/bash
# s34_kernel_breakdown.sh <view dir with ref_mcm/base_mcm/ptwhop0_mcm> <kernel list .g> <tokens> [title] -- per-kernel four metrics of a
# layer window: kernel name, ref cycles, ref unique L2 misses + MPKI, ref AT/PTW, base->hop0 cycles, NormIPC/NormAT/NormPTW (base = 1),
# via h1_slice_four_metrics.py on slices k-k. Prints a markdown table.
set -u
V=$1; KL=$2; T=$3; TITLE=${4:-}; H=/home/user/accel-sim-docker/llm_exp
files=$(grep -v MemcpyHtoD $KL | grep kernel-); N=$(echo "$files" | wc -l); dir=$(dirname $KL)
echo "### $TITLE"; echo "| k | kernel | ref cycle | ref miss / MPKI | ref AT | ref PTW | base→hop0 cycle | NormIPC | NormAT | NormPTW |"; echo "|---|---|---|---|---|---|---|---|---|---|"
i=0; for f in $files; do i=$((i+1)); name=$(xz -dc $dir/$f 2>/dev/null | head -3 | grep -o 'kernel name = .*' | cut -c15- | sed -E 's/^_ZN[0-9]+//; s/^(at6native[0-9]+|vllm[0-9]+|flash[0-9]+|cutlass7Kernel2I[0-9]+|cublasLt[0-9]+|vllm3moe[0-9]+)//' | cut -c1-34)
  o=$(python3 $H/scripts/h1_slice_four_metrics.py $V 64k $i-$i $T 2>&1)
  rc=$(echo "$o" | grep -oE 'cycles [0-9,]+' | head -1 | awk '{print $2}'); miss=$(echo "$o" | grep -oE 'unique L2 miss [0-9,]+' | awk '{print $4}'); mpki=$(echo "$o" | grep -oE 'MPKI [0-9.]+' | awk '{print $2}')
  at=$(echo "$o" | grep -oE 'AT [0-9]+ \(lookups' | awk '{print $2}'); ptw=$(echo "$o" | grep -oE 'PTW latency [0-9]+' | awk '{print $3}')
  n=$(echo "$o" | grep '^\[norm\]'); ipc=$(echo "$n" | grep -oE 'IPC base [0-9.]+ -> hop0 [0-9.]+ = [0-9.]+' | awk '{print $NF}'); nat=$(echo "$n" | grep -oE 'AT [0-9.]+ -> [0-9.]+ = [0-9.]+' | awk '{print $NF}'); nptw=$(echo "$n" | grep -oE 'PTW [0-9.]+ -> [0-9.]+ = [0-9.]+' | awk '{print $NF}'); bc=$(echo "$n" | grep -oE 'cycles base [0-9,]+ hop0 [0-9,]+' | awk '{print $3" → "$5}')
  echo "| $i | $name | ${rc:-} | ${miss:-} / ${mpki:-} | ${at:-} | ${ptw:-} | ${bc:-} | ${ipc:-} | ${nat:-} | ${nptw:-} |"; done
