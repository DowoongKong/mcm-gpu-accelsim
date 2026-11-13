#!/usr/bin/env python3
# s29 (docs/s29_l2miss_timeline_refbase_debug.ko.md section 7): compare the L2 TLB miss / page walk timelines of the ref
# (first touch) and base (ref map pre-loaded) arms from the L2MISSLOG files written by llm_sim_tree_l2misslog.
#   M <cycle> <chiplet> <cluster> 0x<vpn> <kind 0=MISS 1=MERGE> <ft>
#   C <cycle> <chiplet> 0x<vpn> <t_pwq_enq> <t_walk_start> <steps> <remote> <pwc> <faults> <icnt_bp> <lat_sum> <lat_max> 0x<entry_pa> 0x<data_pa>
# Kernel boundaries come from run.log (cumulative gpu_tot_sim_cycle per kernel block). Pure python, no numpy.
# Usage: s29_l2miss_timeline.py <exp dir with ref_mcm/ base_mcm/> [kernels, e.g. 2,9,12,14] [nbins]
import sys, os, re, statistics as st
from collections import defaultdict, Counter

ARMS = ("ref_mcm", "base_mcm")
NBINS = 20

def kernel_bounds(run_log):
    """[(k, name, start, end)] from the per-kernel stat blocks; start = previous block's cumulative end."""
    out = []; name = None; prev = 0
    for line in open(run_log, errors="ignore"):
        if line.startswith("kernel_name = "): name = line.split("=", 1)[1].strip().split()[0]
        elif line.startswith("gpu_tot_sim_cycle = "):
            end = int(line.split("=")[1]); out.append((len(out) + 1, name, prev, end)); prev = end
    return out

def load(path):
    M = []; C = []
    with open(path, errors="ignore") as f:
        for line in f:
            p = line.split()
            if not p: continue
            if p[0] == "M" and len(p) == 7:
                M.append((int(p[1]), int(p[2]), int(p[3]), int(p[4], 16), int(p[5]), int(p[6])))
            elif p[0] == "C" and len(p) == 15:
                C.append((int(p[1]), int(p[2]), int(p[3], 16), int(p[4]), int(p[5]), int(p[6]), int(p[7]), int(p[8]),
                          int(p[9]), int(p[10]), int(p[11]), int(p[12]), int(p[13], 16), int(p[14], 16)))
    M.sort(key=lambda r: r[0]); C.sort(key=lambda r: r[0])
    return M, C

def in_range(rows, a, b, idx=0):
    # rows sorted by rows[idx]; binary search for [a, b)
    import bisect
    keys = [r[idx] for r in rows]
    return rows[bisect.bisect_left(keys, a):bisect.bisect_left(keys, b)]

def q(v, p):
    if not v: return 0
    s = sorted(v); return s[min(len(s) - 1, int(p * len(s)))]

def desc(v):
    if not v: return "n=0"
    return f"n={len(v)} mean={st.fmean(v):.0f} p50={q(v,.5)} p90={q(v,.9)} p99={q(v,.99)} max={max(v)}"

# address decode of -gpgpu_mem_addr_mapping 0000000000000000000000000000DDRRRRRRRRRRRRRRBBBCCCBDDDDDCCCSSSSS (bit 0 rightmost)
def chan(pa):  return (((pa >> 34) & 3) << 5) | ((pa >> 8) & 31)     # 7 bits = 128 channels; >>5 = chiplet
def bank(pa):  return (((pa >> 17) & 7) << 1) | ((pa >> 13) & 1)      # 4 bits = 16 banks
def row(pa):   return (pa >> 20) & 0x3fff

def hist(ts, a, b, n):
    h = [0] * n; w = (b - a) / n if b > a else 1
    for t in ts:
        i = int((t - a) / w); h[min(n - 1, max(0, i))] += 1
    return h

def spearman(x, y):
    n = len(x)
    if n < 3: return float("nan")
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for rank, i in enumerate(order): r[i] = rank
        return r
    rx, ry = ranks(x), ranks(y); mx = my = (n - 1) / 2
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); var = sum((a - mx) ** 2 for a in rx)
    return cov / var if var else float("nan")

def main():
    H = sys.argv[1]
    sel = [int(k) for k in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else NBINS
    KB = {a: kernel_bounds(os.path.join(H, a, "run.log")) for a in ARMS}
    D = {a: load(os.path.join(H, a, "l2miss.log")) for a in ARMS}
    nk = min(len(KB[a]) for a in ARMS)
    print(f"== s29 L2 miss timeline: {H}")
    for a in ARMS:
        M, C = D[a]; print(f"   {a}: M lines {len(M):,} (MISS {sum(1 for r in M if r[4]==0):,} MERGE {sum(1 for r in M if r[4]==1):,}, ft=1 {sum(1 for r in M if r[5]==1):,}), C lines {len(C):,}, kernels {len(KB[a])}, total cycles {KB[a][-1][3]:,}")
    print()
    print("== per-kernel summary (M0 = new walks, UV = unique VPNs, C = completed walks; latencies in cycles)")
    print(" k arm       cycles    M0      MERGE   UV      ft    pwq_wait(mean/p50/p90)  service(mean/p50/p90)  steplat(mean/max p90)  steps  remote  pwc   faults  inflight_avg")
    per = {}
    for k in range(1, nk + 1):
        for a in ARMS:
            _, name, s, e = KB[a][k - 1]; M, C = D[a]
            Mk = in_range(M, s, e); Ck = in_range(C, s, e)
            M0 = [r for r in Mk if r[4] == 0]; M1 = [r for r in Mk if r[4] == 1]
            uv = len(set(r[3] for r in M0)); ft = sum(1 for r in M0 if r[5] == 1)
            pw = [r[4] - r[3] for r in Ck]; sv = [r[0] - r[4] for r in Ck]; sl = [r[10] for r in Ck]; sm = [r[11] for r in Ck]
            inflight = sum(r[0] - r[3] for r in Ck) / (e - s) if e > s else 0
            per[(k, a)] = dict(name=name, s=s, e=e, M0=M0, M1=M1, C=Ck)
            print(f"{k:2d} {a:9s} {e-s:9,d} {len(M0):7,d} {len(M1):7,d} {uv:7,d} {ft:6,d}  "
                  f"{(st.fmean(pw) if pw else 0):7.0f}/{q(pw,.5):6d}/{q(pw,.9):6d}  {(st.fmean(sv) if sv else 0):7.0f}/{q(sv,.5):6d}/{q(sv,.9):6d}  "
                  f"{(st.fmean(sl) if sl else 0):7.0f}/{q(sm,.9):6d}  {(st.fmean([r[5] for r in Ck]) if Ck else 0):5.2f} {(st.fmean([r[6] for r in Ck]) if Ck else 0):6.2f} "
                  f"{(st.fmean([r[7] for r in Ck]) if Ck else 0):5.2f} {sum(r[8] for r in Ck):7,d} {inflight:8.1f}")
        print(f"   kernel {k}: {per[(k,'ref_mcm')]['name'][:60]}")
    kernels = sel or [k for k in range(1, nk + 1) if len(per[(k, "ref_mcm")]["M0"]) >= 1000]
    for k in kernels:
        print(f"\n==================== kernel {k}: {per[(k,'ref_mcm')]['name'][:70]}")
        for a in ARMS:
            P = per[(k, a)]; s, e = P["s"], P["e"]
            print(f"-- {a}: cycles {e-s:,}  new walks {len(P['M0']):,}  merges {len(P['M1']):,}  completed {len(P['C']):,}")
            # H1: when do the misses happen (relative time bins), and how many walks are in flight
            h = hist([r[0] for r in P["M0"]], s, e, nb)
            print(f"   M0 per 1/{nb} of the kernel : " + " ".join(f"{x:d}" for x in h))
            hc = hist([r[0] for r in P["C"]], s, e, nb)
            print(f"   C  per 1/{nb} of the kernel : " + " ".join(f"{x:d}" for x in hc))
            # in-flight walks sampled at bin edges
            ev = sorted([(r[3], 1) for r in P["C"]] + [(r[0], -1) for r in P["C"]])
            samples = []; cur = 0; j = 0
            for i in range(1, nb + 1):
                t = s + (e - s) * i / nb
                while j < len(ev) and ev[j][0] < t: cur += ev[j][1]; j += 1
                samples.append(cur)
            print(f"   walks in flight at bin edges: " + " ".join(f"{x:d}" for x in samples))
            # latency decomposition
            pw = [r[4] - r[3] for r in P["C"]]; sv = [r[0] - r[4] for r in P["C"]]; tot = [r[0] - r[3] for r in P["C"]]
            print(f"   enq->done  {desc(tot)}")
            print(f"   pwq wait   {desc(pw)}")
            print(f"   service    {desc(sv)}")
            print(f"   step lat   sum {desc([r[10] for r in P['C']])} | max {desc([r[11] for r in P['C']])}")
            print(f"   steps mean {st.fmean([r[5] for r in P['C']]) if P['C'] else 0:.3f} remote {st.fmean([r[6] for r in P['C']]) if P['C'] else 0:.3f} pwc {st.fmean([r[7] for r in P['C']]) if P['C'] else 0:.3f} icnt_bp mean {st.fmean([r[9] for r in P['C']]) if P['C'] else 0:.1f} faults {sum(r[8] for r in P['C']):,}")
            # H2: fault vs no-fault walks (ref only has faults)
            f1 = [r[0] - r[4] for r in P["C"] if r[8] > 0]; f0 = [r[0] - r[4] for r in P["C"] if r[8] == 0]
            if f1: print(f"   service by fault: fault>0 {desc(f1)} | fault=0 {desc(f0)}")
            # SM view
            first = {}; cnt = Counter()
            for r in P["M0"]:
                cnt[r[2]] += 1
                if r[2] not in first: first[r[2]] = r[0] - s
            fv = list(first.values()); cv = list(cnt.values())
            print(f"   SMs with misses {len(cnt)}; first-miss time per SM min/p50/max {min(fv) if fv else 0}/{q(fv,.5)}/{max(fv) if fv else 0}; misses per SM min/p50/max {min(cv) if cv else 0}/{q(cv,.5)}/{max(cv) if cv else 0}")
            chc = Counter(r[1] for r in P["M0"]); print(f"   new walks per chiplet: {dict(sorted(chc.items()))}")
            # H3: where do the page-table reads go
            ch = Counter(chan(r[12]) & 31 for r in P["C"] if r[5] > 0); bk = Counter(bank(r[12]) for r in P["C"] if r[5] > 0)
            rows = Counter(row(r[12]) for r in P["C"] if r[5] > 0)
            top = ch.most_common(4)
            print(f"   PT-entry reads: in-chiplet channel values used {len(ch)} top {top} | banks used {len(bk)} top {bk.most_common(3)} | distinct rows {len(rows)} top {rows.most_common(2)}")
            # H4: data page frames
            fr = Counter((bank(r[13]), row(r[13])) for r in P["C"]); bd = Counter(bank(r[13]) for r in P["C"])
            print(f"   data frames: distinct (bank,row) {len(fr)} max pages per (bank,row) {fr.most_common(1)} | bank spread {sorted(bd.values())[:3]}..{sorted(bd.values())[-3:]}")
        # VPN-matched comparison (first new walk per VPN in this kernel)
        fa = {}; fb = {}
        for r in per[(k, "ref_mcm")]["M0"]:
            if r[3] not in fa: fa[r[3]] = (r[0] - per[(k,"ref_mcm")]["s"], r[2], r[1])
        for r in per[(k, "base_mcm")]["M0"]:
            if r[3] not in fb: fb[r[3]] = (r[0] - per[(k,"base_mcm")]["s"], r[2], r[1])
        common = [v for v in fa if v in fb]
        if common:
            dt = [fa[v][0] - fb[v][0] for v in common]
            same_sm = sum(1 for v in common if fa[v][1] == fb[v][1]); same_ch = sum(1 for v in common if fa[v][2] == fb[v][2])
            rho = spearman([fa[v][0] for v in common], [fb[v][0] for v in common])
            print(f"-- VPN-matched (first new walk per VPN): common {len(common):,} of ref {len(fa):,} / base {len(fb):,}; "
                  f"t_ref - t_base: mean {st.fmean(dt):.0f} p10 {q(dt,.1)} p50 {q(dt,.5)} p90 {q(dt,.9)}; same first SM {100*same_sm/len(common):.1f} %, same first chiplet {100*same_ch/len(common):.1f} %, Spearman(order) {rho:.3f}")
            # relative order: fraction of kernel time at which each VPN first misses
            ra = [fa[v][0] / (per[(k,"ref_mcm")]["e"] - per[(k,"ref_mcm")]["s"]) for v in common]
            rb = [fb[v][0] / (per[(k,"base_mcm")]["e"] - per[(k,"base_mcm")]["s"]) for v in common]
            print(f"   normalized first-miss time (fraction of kernel): ref p50 {q(ra,.5):.3f} base p50 {q(rb,.5):.3f}; |diff| p50 {q([abs(x-y) for x,y in zip(ra,rb)],.5):.3f}")

if __name__ == "__main__":
    main()
