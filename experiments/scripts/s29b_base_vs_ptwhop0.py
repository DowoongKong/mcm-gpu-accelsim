#!/usr/bin/env python3
# s29b (docs/s29b_l2miss_timeline_base_vs_ptwhop0.ko.md section 5): why does ptwhop0 take more L2 TLB misses than base on
# the same trace and the same placement map? Reads the L2MISSLOG files of base_mcm and ptwhop0_mcm
#   M <cycle> <chiplet> <cluster> 0x<vpn> <kind 0=MISS 1=MERGE> <ft>
#   C <cycle> <chiplet> 0x<vpn> <t_pwq_enq> <t_walk_start> <steps> <remote> <pwc> <faults> <icnt_bp> <lat_sum> <lat_max> 0x<entry_pa> 0x<data_pa>
# and, per kernel window (run.log cumulative gpu_tot_sim_cycle), replays fills (C) and misses (M kind 0) in time order per
# chiplet. For every RE-miss of a (chiplet, vpn) already filled in the window it records the time since that fill and the
# number of fills into the same L2 TLB set in between (set = 128-set XOR-fold, -gpgpu_l2_tlb_set_hash 1; 8 ways, LRU).
# Usage: s29b_base_vs_ptwhop0.py <debug_l2miss_timeline dir> [kernels, default 9,14] [arms, default base_mcm,ptwhop0_mcm]
import statistics as st
import sys
from collections import defaultdict

D = sys.argv[1]
KS = [int(k) for k in (sys.argv[2] if len(sys.argv) > 2 else "9,14").split(",")]
ARMS = (sys.argv[3] if len(sys.argv) > 3 else "base_mcm,ptwhop0_mcm").split(",")
SETS, SET_BITS, WAYS = 128, 7, 8
NAMES = {2: "QKV", 9: "O", 12: "gate_up", 14: "down"}


def set_idx(vpn):
    s = 0
    while vpn:
        s ^= vpn & (SETS - 1)
        vpn >>= SET_BITS
    return s


def kernel_bounds(run_log):
    out = []; prev = 0
    for line in open(run_log, errors="ignore"):
        if line.startswith("gpu_tot_sim_cycle = "):
            end = int(line.split("=")[1]); out.append((prev, end)); prev = end
    return out


def q(xs, p):
    if not xs: return float("nan")
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]


def analyse(arm):
    bounds = kernel_bounds(f"{D}/{arm}/run.log")
    win = {k: bounds[k - 1] for k in KS if k <= len(bounds)}
    ev = {k: [] for k in win}          # (cycle, order, type, chiplet, sm, vpn, t_enq, t_start)
    with open(f"{D}/{arm}/l2miss.log", errors="ignore") as f:
        for line in f:
            p = line.split()
            if not p: continue
            t = int(p[1])
            for k, (a, b) in win.items():
                if a <= t < b:
                    if p[0] == "M":
                        ev[k].append((t, 1, int(p[5]), int(p[2]), int(p[3]), int(p[4], 16), 0, 0))
                    elif p[0] == "C":
                        ev[k].append((t, 0, 2, int(p[2]), -1, int(p[3], 16), int(p[4]), int(p[5])))
                    break
    res = {}
    for k, E in ev.items():
        a, b = win[k]; dur = b - a
        E.sort(key=lambda r: (r[0], r[1]))          # fills before misses at the same cycle
        setcnt = defaultdict(int); last_fill = {}; miss_seen = set()
        remiss_gap, remiss_between, first_miss = [], [], 0
        n_miss = n_merge = n_fill = 0; pwq, svc = [], []
        fills_ch = defaultdict(int); sm_last = {}; sm_gap = []
        vpn_sms = defaultdict(set)
        for t, _, typ, ch, sm, vpn, te, ts in E:
            if typ == 2:                               # walk completed -> entry filled
                n_fill += 1; fills_ch[ch] += 1
                s = set_idx(vpn); setcnt[(ch, s)] += 1
                last_fill[(ch, vpn)] = (t, setcnt[(ch, s)])
                pwq.append(ts - te); svc.append(t - ts)
            elif typ == 0:                             # new miss
                n_miss += 1; vpn_sms[(ch, vpn)].add(sm)
                lf = last_fill.get((ch, vpn))
                if lf is None:
                    first_miss += 1
                else:
                    remiss_gap.append(t - lf[0])
                    remiss_between.append(setcnt[(ch, set_idx(vpn))] - lf[1])
                key = (sm, vpn)
                if key in sm_last: sm_gap.append(t - sm_last[key])
                sm_last[key] = t
            else:                                      # merge into an in-flight walk
                n_merge += 1; vpn_sms[(ch, vpn)].add(sm)
        nre = len(remiss_gap)
        res[k] = dict(
            dur=dur, miss=n_miss, merge=n_merge, fill=n_fill, first=first_miss, remiss=nre,
            fill_rate_set=n_fill / (4 * SETS * dur) if dur else 0,
            pwq_med=q(pwq, .5), svc_med=q(svc, .5), svc_mean=st.mean(svc) if svc else 0,
            gap_med=q(remiss_gap, .5), gap_p10=q(remiss_gap, .1), gap_p90=q(remiss_gap, .9),
            btw_med=q(remiss_between, .5), btw_ge8=sum(1 for x in remiss_between if x >= WAYS) / nre if nre else 0,
            btw_lt8=sum(1 for x in remiss_between if x < WAYS),
            smgap_med=q(sm_gap, .5), smgap_n=len(sm_gap),
            sms_per_vpn=st.mean(len(v) for v in vpn_sms.values()) if vpn_sms else 0,
            uniq=len(vpn_sms))
    return res


R = {arm: analyse(arm) for arm in ARMS}
rows = [("window cycles", "dur", "{:,}"), ("new misses (walks)", "miss", "{:,}"), ("merges", "merge", "{:,}"),
        ("fills (C)", "fill", "{:,}"), ("(chiplet,vpn) pairs touched by misses", "uniq", "{:,}"),
        ("first miss of a pair in window", "first", "{:,}"), ("RE-misses (pair filled before)", "remiss", "{:,}"),
        ("fills per set per kcycle", "fill_rate_set", None), ("PWQ wait median", "pwq_med", "{:,}"),
        ("walk service median / mean", "svc_med", None), ("re-miss: cycles since its last fill p10/med/p90", "gap_med", None),
        ("re-miss: same-set fills in between, median", "btw_med", "{:,}"), ("re-miss: share with >= 8 fills in between", "btw_ge8", "{:.3f}"),
        ("re-miss with < 8 fills in between (not plain LRU)", "btw_lt8", "{:,}"),
        ("same-SM same-VPN consecutive-miss gap median", "smgap_med", "{:,}"), ("SMs per (chiplet,vpn)", "sms_per_vpn", "{:.2f}")]
for k in KS:
    print(f"\n== kernel {k} ({NAMES.get(k, '?')}) ==")
    print(f"{'metric':52s}" + "".join(f"{a:>26s}" for a in ARMS))
    for label, key, fmt in rows:
        cells = []
        for a in ARMS:
            r = R[a].get(k)
            if r is None: cells.append("n/a"); continue
            if key == "fill_rate_set": cells.append(f"{1000 * r[key]:.3f}")
            elif key == "svc_med": cells.append(f"{r['svc_med']:,} / {r['svc_mean']:.0f}")
            elif key == "gap_med": cells.append(f"{r['gap_p10']:,}/{r['gap_med']:,}/{r['gap_p90']:,}")
            else: cells.append(fmt.format(r[key]))
        print(f"{label:52s}" + "".join(f"{c:>26s}" for c in cells))
    if all(k in R[a] for a in ARMS) and len(ARMS) == 2:
        b, p = R[ARMS[0]][k], R[ARMS[1]][k]
        if b["remiss"] and b["fill_rate_set"] and b["gap_med"] == b["gap_med"]:
            print(f"  ratio {ARMS[1]}/{ARMS[0]}: misses {p['miss']/b['miss']:.3f}, re-misses {p['remiss']/max(1,b['remiss']):.3f}, "
                  f"fill rate per set {p['fill_rate_set']/b['fill_rate_set']:.3f}, re-miss gap median {p['gap_med']/b['gap_med']:.3f}, "
                  f"same-SM gap median {p['smgap_med']/b['smgap_med']:.3f}")
