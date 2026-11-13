#!/usr/bin/env python3
# s29b follow-up (docs/s29b_l2miss_timeline_base_vs_ptwhop0.ko.md): for every RE-miss of a (chiplet, vpn) inside one
# kernel window, was the previous fill of that pair started by the SAME SM or by ANOTHER SM, and how long ago (cycles)?
# A fill's starter is the SM of the M kind-0 line whose cycle equals the C line's t_pwq_enq (same chiplet, vpn).
# Usage: s29b_remiss_who.py <debug_l2miss_timeline dir> <kernel> <arm> [<arm> ...]
import sys
from collections import Counter

D, K, ARMS = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
EDGES = [0, 2000, 5000, 10000, 20000, 40000, 10**12]


def window(run_log, k):
    prev = 0; ends = []
    for line in open(run_log, errors="ignore"):
        if line.startswith("gpu_tot_sim_cycle = "):
            ends.append(int(line.split("=")[1]))
    return (ends[k - 2] if k >= 2 else 0), ends[k - 1]


for arm in ARMS:
    a, b = window(f"{D}/{arm}/run.log", K)
    starter = {}     # (chiplet, vpn, t_enq) -> sm
    ev = []
    for line in open(f"{D}/{arm}/l2miss.log", errors="ignore"):
        p = line.split()
        t = int(p[1])
        if not (a <= t < b):
            continue
        if p[0] == "M" and p[5] == "0":
            ch, sm, vpn = int(p[2]), int(p[3]), int(p[4], 16)
            starter[(ch, vpn, t)] = sm
            ev.append((t, 1, ch, vpn, sm))
        elif p[0] == "C":
            ch, vpn, te = int(p[2]), int(p[3], 16), int(p[4])
            ev.append((t, 0, ch, vpn, te))
    ev.sort()
    last_fill = {}   # (ch, vpn) -> (fill cycle, starter sm)
    cnt = Counter(); first = 0
    for t, typ, ch, vpn, x in ev:
        if typ == 0:
            last_fill[(ch, vpn)] = (t, starter.get((ch, vpn, x), -1))
        else:
            lf = last_fill.get((ch, vpn))
            if lf is None:
                first += 1; continue
            gap = t - lf[0]
            who = "same SM" if lf[1] == x else "other SM"
            bin_ = next(i for i in range(len(EDGES) - 1) if EDGES[i] <= gap < EDGES[i + 1])
            cnt[(who, bin_)] += 1
    print(f"== {arm} kernel {K} window [{a:,}, {b:,}) first-misses {first:,}")
    for who in ("same SM", "other SM"):
        row = [cnt[(who, i)] for i in range(len(EDGES) - 1)]
        print(f"  {who:9s} re-misses {sum(row):>10,}  by gap since previous fill: " +
              "  ".join(f"[{EDGES[i]//1000}k,{'inf' if EDGES[i+1] > 10**9 else str(EDGES[i+1]//1000)+'k'}) {row[i]:,}"
                        for i in range(len(row))))
