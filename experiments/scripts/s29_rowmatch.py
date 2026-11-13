#!/usr/bin/env python3
# s29 follow-up (docs/s29 section 8): does the page-table read of a walk land in the same DRAM row/bank as the data page it
# maps (ref: node and data frames allocated at the same first touch; base: nodes placed after all data), and is that what
# separates fast from slow walks? Per kernel and arm: row/bank match fractions, service time split by match, service time by
# time bin, slow-walk attribution, icnt backpressure. Pure python. Usage: s29_rowmatch.py <exp dir> [kernels]
import sys, os, statistics as st
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s29_l2miss_timeline import kernel_bounds, load, in_range, q, chan, bank, row, desc

ARMS = ("ref_mcm", "base_mcm")
def bank_hi(pa): return (pa >> 17) & 7      # bank bits outside a 64 KiB page (bit 13 is inside the page)

def main():
    H = sys.argv[1]; sel = [int(k) for k in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2, 9, 12, 14]
    KB = {a: kernel_bounds(os.path.join(H, a, "run.log")) for a in ARMS}
    D = {a: load(os.path.join(H, a, "l2miss.log")) for a in ARMS}
    for k in sel:
        print(f"\n==================== kernel {k}: {KB['ref_mcm'][k-1][1][:60]}")
        for a in ARMS:
            _, name, s, e = KB[a][k - 1]; C = [r for r in in_range(D[a][1], s, e) if r[5] > 0]
            n = len(C)
            same_row = [r for r in C if row(r[12]) == row(r[13])]
            same_rb = [r for r in C if row(r[12]) == row(r[13]) and bank_hi(r[12]) == bank_hi(r[13])]
            sv = lambda rows: [r[0] - r[4] for r in rows]
            print(f"-- {a}: walks with a memory step {n:,}; PT row == data row {100*len(same_row)/n:.1f} %  (row+bank {100*len(same_rb)/n:.1f} %)")
            print(f"   service  same-row {desc(sv(same_row))}")
            print(f"   service  other    {desc(sv([r for r in C if row(r[12]) != row(r[13])]))}")
            # PT node rows vs the data rows streamed in this kernel: is any PT row also a data row of the kernel?
            data_rows = Counter(row(r[13]) for r in C); pt_rows = Counter(row(r[12]) for r in C)
            shared = sum(c for rr, c in pt_rows.items() if rr in data_rows)
            print(f"   PT rows {len(pt_rows)} (top {pt_rows.most_common(3)}), data rows {len(data_rows)}; PT reads whose row is also a data row of this kernel {100*shared/n:.1f} %")
            # |PT frame - data frame| distance in 1 MiB rows
            dist = [abs(row(r[12]) - row(r[13])) for r in C]
            print(f"   |row(PT) - row(data)|: p50 {q(dist,.5)} p90 {q(dist,.9)}; PT frame index range {min(r[12]>>12 for r in C):#x}..{max(r[12]>>12 for r in C):#x}, data frame range {min(r[13]>>12 for r in C):#x}..{max(r[13]>>12 for r in C):#x}")
            # service by time bin
            nb = 10; w = (e - s) / nb; bins = [[] for _ in range(nb)]
            for r in C: bins[min(nb-1, int((r[0]-s)/w))].append(r[0]-r[4])
            print("   service mean per 1/10 of kernel : " + " ".join(f"{st.fmean(b):.0f}" if b else "-" for b in bins))
            print("   service p90  per 1/10 of kernel : " + " ".join(f"{q(b,.9)}" if b else "-" for b in bins))
            slow = [r for r in C if r[0]-r[4] > 5000]
            print(f"   slow walks (service > 5000): {len(slow):,} = {100*len(slow)/n:.1f} %; by chiplet {dict(sorted(Counter(r[1] for r in slow).items()))}; "
                  f"by in-chiplet channel {dict(Counter(chan(r[12]) & 31 for r in slow))}; same-row share among slow {100*sum(1 for r in slow if row(r[12])==row(r[13]))/len(slow) if slow else 0:.1f} %")
            bp = [r[9] for r in C]
            print(f"   icnt backpressure cycles: mean {st.fmean(bp):.1f} p90 {q(bp,.9)} max {max(bp)}; walks with bp>0 {100*sum(1 for x in bp if x>0)/n:.1f} %")
            lat = [r[11] for r in C]
            print(f"   memory-step latency (max per walk): {desc(lat)}")
if __name__ == "__main__":
    main()
