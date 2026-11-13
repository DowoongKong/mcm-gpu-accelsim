#!/usr/bin/env python3
# s34_k11_walk_timeline.py <arm dir A> <arm dir B> <kernel block k> -- compare the page-walk event log (ACCELSIM_L2MISS_LOG 'C' lines) of two
# arms inside one kernel block: per-walk PWQ wait (walk_start - pwq_enq), walk service (complete - walk_start), icnt backpressure, step
# latency max, remote flag and the owner chiplet of the last PT entry (PA >> 35 for the 128 GB mapping). Kernel block boundaries come from
# the cumulative gpu_tot_sim_cycle lines of run.log. Prints distributions (mean / p50 / p90 / p99) and per-owner-chiplet splits.
import sys, statistics as st, collections
def kbounds(run_log, k):
    c = [int(l.split("=")[1]) for l in open(run_log, errors="ignore") if l.startswith("gpu_tot_sim_cycle =")]
    return (c[k-2] if k >= 2 else 0), c[k-1]
def walks(d, k):
    lo, hi = kbounds(d + "/run.log", k); out = []
    for l in open(d + "/l2miss.log", errors="ignore"):
        if not l.startswith("C "): continue
        t = l.split()
        cyc = int(t[1]); 
        if not (lo < cyc <= hi): continue
        chip, vpn, enq, start, steps, remote, pwc, faults, bp, lsum, lmax = int(t[2]), int(t[3], 16), int(t[4]), int(t[5]), int(t[6]), int(t[7]), int(t[8]), int(t[9]), int(t[10]), int(t[11]), int(t[12])
        last_pa = int(t[13], 16); out.append(dict(cyc=cyc, chip=chip, vpn=vpn, pwq=start - enq, svc=cyc - start, steps=steps, remote=remote, pwc=pwc, bp=bp, lsum=lsum, lmax=lmax, owner=last_pa >> 35))
    return out, (lo, hi)
def q(v, p): v = sorted(v); return v[min(len(v)-1, int(p * len(v)))] if v else 0
A, B, k = sys.argv[1], sys.argv[2], int(sys.argv[3]); wa, ra = walks(A, k); wb, rb = walks(B, k)
print(f"kernel {k}: A {A.split('/')[-1]} cycles {ra[1]-ra[0]:,} walks {len(wa):,} | B {B.split('/')[-1]} cycles {rb[1]-rb[0]:,} walks {len(wb):,}")
for name, w in (("A", wa), ("B", wb)):
    for f in ("pwq", "svc", "bp", "lmax"):
        v = [x[f] for x in w]; print(f"  {name} {f:<4} mean {st.mean(v) if v else 0:8.1f}  p50 {q(v,.5):7d}  p90 {q(v,.9):7d}  p99 {q(v,.99):7d}")
    print(f"  {name} remote {sum(x['remote'] for x in w)}/{len(w)}  owner chiplet of last PT entry {dict(sorted(collections.Counter(x['owner'] for x in w).items()))}  requesting chiplet {dict(sorted(collections.Counter(x['chip'] for x in w).items()))}")
# paired by (requesting chiplet, vpn) order
ka = collections.defaultdict(list); kb = collections.defaultdict(list)
for x in wa: ka[(x['chip'], x['vpn'])].append(x)
for x in wb: kb[(x['chip'], x['vpn'])].append(x)
pairs = [(a[0], b[0]) for key, a in ka.items() for b in [kb.get(key)] if b]
if pairs:
    d_svc = [b['svc'] - a['svc'] for a, b in pairs]; d_pwq = [b['pwq'] - a['pwq'] for a, b in pairs]
    print(f"  paired walks {len(pairs):,}: B-A svc mean {st.mean(d_svc):+.1f} (p50 {q(d_svc,.5):+d}, p90 {q(d_svc,.9):+d}) | pwq mean {st.mean(d_pwq):+.1f}")
    by_owner = collections.defaultdict(list)
    for a, b in pairs: by_owner[a['owner']].append(b['svc'] - a['svc'])
    print("  B-A svc by owner chiplet of the PT entry:", {o: f"n={len(v)} mean {st.mean(v):+.0f}" for o, v in sorted(by_owner.items())})
