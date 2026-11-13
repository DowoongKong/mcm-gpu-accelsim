#!/bin/bash
# Compare decode_S256_L1 (llm_sim_tree, patch 29) with decode_S256_L1_fast
# (llm_sim_tree_fast = patch 29 + ST cleanup + T1~T16): per arm, wall time,
# gpu_tot_sim_cycle / insn, and the A-class digest diff (verify/xtract_amrs.py).
R=/home/user/accel-sim-docker; O=$R/llm_exp/experiments/decode_S256_L1; N=$R/llm_exp/experiments/decode_S256_L1_fast
printf "%-12s %-10s %-10s %-8s %-12s %-12s %-12s %s\n" arm old_s new_s speedup cyc_old cyc_new insn_same digest_diff
for A in ref_batch4 base_pcc ptwhop0_pcc; do
  [ -f $N/$A/exit_code.txt ] || { printf "%-12s (not finished)\n" $A; continue; }
  os=$(grep -o 'SECONDS=[0-9]*' $O/$A/exit_code.txt | cut -d= -f2); ns=$(grep -o 'SECONDS=[0-9]*' $N/$A/exit_code.txt | cut -d= -f2)
  co=$(grep '^gpu_tot_sim_cycle = ' $O/$A/run.log | tail -1 | awk '{print $3}'); cn=$(grep '^gpu_tot_sim_cycle = ' $N/$A/run.log | tail -1 | awk '{print $3}')
  io=$(grep '^gpu_tot_sim_insn = ' $O/$A/run.log | tail -1 | awk '{print $3}'); in=$(grep '^gpu_tot_sim_insn = ' $N/$A/run.log | tail -1 | awk '{print $3}')
  # The 09-18 stats cleanup (in the new tree) no longer prints the all-zero 64k/2m TLB lines; drop them from the old digest (same rule as gate G-A).
  python3 $R/verify/xtract_amrs.py $O/$A/run.log | tail -n +2 | grep -vE '^A[[:space:]]+\[L[12]_TLB_(64k|2m)_' > /tmp/dg_old_$A.txt; python3 $R/verify/xtract_amrs.py $N/$A/run.log | tail -n +2 > /tmp/dg_new_$A.txt
  if cmp -s /tmp/dg_old_$A.txt /tmp/dg_new_$A.txt; then dd="0/$(wc -l < /tmp/dg_old_$A.txt)"; else dd="$(diff /tmp/dg_old_$A.txt /tmp/dg_new_$A.txt | grep -c '^[<>]')/$(wc -l < /tmp/dg_old_$A.txt)"; fi
  sp=$(python3 -c "print(f'{$os/$ns:.2f}x')" 2>/dev/null)
  printf "%-12s %-10s %-10s %-8s %-12s %-12s %-12s %s\n" $A "$os" "$ns" "$sp" "$co" "$cn" "$([ "$io" = "$in" ] && echo yes || echo NO)" "$dd"
done
