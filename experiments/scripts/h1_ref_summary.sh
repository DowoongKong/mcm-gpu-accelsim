#!/bin/bash
# One-line metrics of a finished hypothesis-1 reference run (docs/h1_llama70b_decode_batch_sweep.ko.md section 8.1):
# builds the per-kernel static profiles if missing (s23_kernel_tlb_profile.py, 64 KiB pages), joins them with the run.log
# (s23_kernel_tlb_summary.py -> analysis/s23_summary_ref_64k.md) and appends a row to <model>/REF_SUMMARY.tsv:
#   config, batch, cycles, thread insn, warp insn, IPC, L1 lookups, L1 hit%, L2 hit, L2 unique miss, L2 merge, L2 hit%,
#   MPKI (unique L2 miss per warp kilo-insn, GPU-wide), cold lower bound, capacity (miss - cold), seconds, exit.
# Usage: h1_ref_summary.sh <model dir name> <config dir name>
set -u
MODEL=$1; CFG=$2
H=/home/user/accel-sim-docker/llm_exp
P=$H/llm_hypothesis1/$MODEL/$CFG
R=$P/sim/ref_mcm
SUM=$H/llm_hypothesis1/$MODEL/REF_SUMMARY.tsv
[ -s $R/exit_code.txt ] || { echo "no exit_code.txt in $R"; exit 1; }
mkdir -p $P/analysis/profiles
i=0
grep -v '^MemcpyHtoD' $P/traces/kernelslist.g | while read -r k; do
  i=$((i+1)); out=$P/analysis/profiles/k$(printf %02d $i).json
  [ -s $out ] || nice -n 15 python3 $H/scripts/s23_kernel_tlb_profile.py $P/traces/$k $out 16 > ${out%.json}.log 2>&1
done
NK=$(grep -vc '^MemcpyHtoD' $P/traces/kernelslist.g)
python3 $H/scripts/s23_kernel_tlb_summary.py $R/run.log 64k $P/analysis/profiles $(seq 1 $NK) > $P/analysis/s23_summary_ref_64k.md 2>&1
[ -s $SUM ] || printf 'config\tbatch\tcycles\tthread_insn\twarp_insn\tipc\tl1_lookups\tl1_hit_pct\tl2_hit\tl2_unique_miss\tl2_merge\tl2_hit_pct\tmpki\tcold_lb\tcapacity\tseconds\texit\n' > $SUM
python3 - "$P" "$CFG" "$R" "$SUM" <<'EOF'
import json, re, sys
p, cfg, r, summ = sys.argv[1:5]
j = json.load(open(f"{p}/capture_info.json"))
log = open(f"{r}/run.log", errors="ignore").read()
def last(pat):
    m = re.findall(pat, log); return m[-1] if m else None
cyc = int(last(r"gpu_tot_sim_cycle = (\d+)")); insn = int(last(r"gpu_tot_sim_insn = (\d+)")); ipc = float(last(r"gpu_tot_ipc =\s+([\d.]+)"))
tot = [ln for ln in open(f"{p}/analysis/s23_summary_ref_64k.md") if ln.startswith("| 합 ")][0].split("|")
# columns of the 합 row: k, L1 lookups, L1 hit, L1 merge, L2 hit, L2 unique miss, L2 merge, L2 hit%, lookups/miss, MSHR, cold, capacity, carry, share
num = lambda s: int(s.strip().replace(",", "").split(" ")[0]) if s.strip() else 0
l1_lookups = num(tot[2]); l1_hit = tot[3].strip(); l2_hit = num(tot[5]); l2_miss = num(tot[6]); l2_merge = num(tot[7]); l2_hit_pct = tot[8].strip()
cold = num(tot[11]); cap = num(tot[12])
warp = j["window_warp_insts"]; mpki = l2_miss / (warp / 1000.0)
ex = open(f"{r}/exit_code.txt").read().strip(); sec = re.search(r"SECONDS=(\d+)", ex).group(1); exit_ = re.search(r"EXIT=(\d+)", ex).group(1)
row = [cfg, j["batch"], cyc, insn, warp, f"{ipc:.1f}", l1_lookups, l1_hit, l2_hit, l2_miss, l2_merge, l2_hit_pct, f"{mpki:.2f}", cold, cap, sec, exit_]
lines = [ln for ln in open(summ) if not ln.startswith(cfg + "\t")]
lines.append("\t".join(str(x) for x in row) + "\n")
open(summ, "w").writelines(lines)
print("\t".join(str(x) for x in row))
EOF
