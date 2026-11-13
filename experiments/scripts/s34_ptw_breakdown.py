#!/usr/bin/env python3
# s34_ptw_breakdown.py <run.log> <first block> <last block> -- translation-latency stage breakdown of a kernel-block range: for
# [XLATSUM] out=ALL and out=PTW, the n-weighted average of every stage field over SMs (per-kernel cumulative stats are differenced
# block-by-block like h1_slice_four_metrics.py), plus the [GMMU_STAT] per-chiplet lines of the last block. Usage: ... run.log 11 11
import re, sys, collections
path, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
blocks = []; cur = None
for line in open(path, errors="ignore"):
    if line.startswith("kernel_launch_uid ="): cur = {"x": collections.defaultdict(dict), "g": []}; blocks.append(cur); continue
    if cur is None: continue
    if line.startswith("[XLATSUM] GPU kind="):   # GPU-wide cumulative lines, one per outcome (ALL, L1_HIT, L1_MERGE, L2_HIT, L2_MERGE, PTW)
        m = re.match(r"\[XLATSUM\] GPU kind=(\w+) out=(\w+) n=(\d+)(.*)", line)
        if not m: continue
        out, n = m.group(2), int(m.group(3))
        f = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", m.group(4))}
        if "avg" not in f: continue
        cur["x"][(out, 0)] = (n, f)
    elif line.startswith("[GMMU_STAT]") or (cur["g"] and len(cur["g"]) < 40 and line.startswith(" ") and "=" in line): cur["g"].append(line.rstrip())
def agg(bi, out):   # cumulative n-weighted sums at block bi (n and n*avg per field)
    S = collections.defaultdict(float); N = 0
    if bi < 0: return N, S
    for (o, sm), (n, f) in blocks[bi]["x"].items():
        if o != out: continue
        N += n
        for k, v in f.items(): S[k] += n * v
    return N, S
for out in ("ALL", "L2_HIT", "L2_MERGE", "PTW"):
    n1, s1 = agg(b - 1, out); n0, s0 = agg(a - 2, out); n = n1 - n0
    if n <= 0: print(f"{out}: no lookups in blocks {a}-{b}"); continue
    print(f"{out:<8} blocks {a}-{b}: n={n:,} avg={(s1['avg']-s0['avg'])/n:.1f} | " + " ".join(f"{k}={(s1[k]-s0[k])/n:.1f}" for k in s1 if k not in ("avg","svc","wait") and (s1[k]-s0[k])/n >= 0.05))
if "--gmmu" in sys.argv: print("GMMU_STAT (last block):"); print("\n".join(blocks[b-1]["g"][:40]))
