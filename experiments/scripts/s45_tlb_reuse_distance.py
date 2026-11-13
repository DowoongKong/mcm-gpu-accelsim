#!/usr/bin/env python3
# Step 45 (docs/s45_h1_context_length_theory.ko.md section 13): TLB reuse distance of one kernel trace, no simulation.
# Access = (warp memory instruction, 64 KB page). L1 level: per CTA (one CTA per SM here), warps interleaved round-robin by their
# instruction index; LRU-32 hit/miss and stack distance (distinct pages since the last access to the same page). L2 level: the L1
# misses of every CTA on chiplet 0 merged by normalized progress (CTAs assumed to advance at the same rate); LRU-1024 hit/miss and
# stack distance. Timing (in-flight walks, MSHR merges) is not modelled.
# Usage: s45_tlb_reuse_distance.py <kernel .traceg.xz> [chip_for_L2 = 0]
import lzma, math, sys, collections, bisect
f = sys.argv[1]; CHIP = int(sys.argv[2]) if len(sys.argv) > 2 else 0; N_CHIP = 4; SHIFT = 16
MEM_OPS = ("LDG", "STG", "LD", "ST", "LDGSTS", "ATOMG", "RED", "LDGDEPBAR")
def addrs(t, k):
    if int(t[k]) == 0: return []
    fmt = int(t[k + 1]); base = int(t[k + 2], 16); act = bin(int(t[1], 16)).count("1")
    if fmt == 1: s = int(t[k + 3]); return [base + i * s for i in range(act)]
    if fmt == 2:
        out = [base]; a = base
        for d in t[k + 3:k + 3 + act - 1]: a += int(d); out.append(a)
        return out
    return [int(x, 16) for x in t[k + 2:k + 2 + act]]
h = {}; gx = gy = 1; chunk = 1; n_total = 0; lin = -1; warp = -1
cta = collections.defaultdict(lambda: collections.defaultdict(list))   # lin -> warp -> [page, ...]
with lzma.open(f, "rt") as fh:
    for ln in fh:
        if ln.startswith("-"):
            k, v = ln.lstrip("-").split("=", 1); h[k.strip()] = v.strip(); continue
        if ln.startswith("#"):
            if "grid dim" in h and not n_total:
                gx, gy, gz = (int(x) for x in h["grid dim"].strip("() ").split(",")); n_total = gx * gy * gz; chunk = math.ceil(n_total / N_CHIP)
            continue
        if ln.startswith("thread block ="):
            x, y, z = (int(v) for v in ln.split("=")[1].split(",")); lin = x + y * gx + z * gx * gy; continue
        if ln.startswith("warp ="): warp = int(ln.split("=")[1]); continue
        if lin < 0: continue
        t = ln.split()
        if len(t) < 5: continue
        try: dn = int(t[2]); oi = 3 + dn
        except ValueError: continue
        if t[oi].split(".")[0] not in MEM_OPS: continue
        try: A = addrs(t, oi + 2 + int(t[oi + 1]))
        except Exception: continue
        for P in sorted({a >> SHIFT for a in A}): cta[lin][warp].append(P)
class Fen:
    def __init__(s, n): s.n = n; s.t = [0] * (n + 1)
    def add(s, i, v):
        i += 1
        while i <= s.n: s.t[i] += v; i += i & -i
    def sum(s, i):
        i += 1; r = 0
        while i > 0: r += s.t[i]; i -= i & -i
        return r
def stack(stream, cap):
    """exact stack distances (None = first access) and LRU-cap hit flags"""
    fen = Fen(len(stream)); last = {}; out = []
    for i, p in enumerate(stream):
        j = last.get(p)
        if j is None: d = None
        else: d = fen.sum(i - 1) - fen.sum(j); fen.add(j, -1)
        fen.add(i, 1); last[p] = i; out.append(d)
    return out
bins = [0, 1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 1 << 30]
def hist(ds):
    n = len(ds); cold = sum(1 for d in ds if d is None); H = collections.Counter()
    for d in ds:
        if d is None: continue
        H[bisect.bisect_right(bins, d) - 1] += 1
    return n, cold, H
L1n = L1cold = L1hit = 0; L1H = collections.Counter(); l2_streams = {}
for c, wd in cta.items():
    ws = sorted(wd); m = max(len(wd[w]) for w in ws)
    s = [wd[w][i] for i in range(m) for w in ws if i < len(wd[w])]     # round-robin warp interleave
    ds = stack(s, 32); n, cold, H = hist(ds); L1n += n; L1cold += cold; L1H.update(H)
    L1hit += sum(1 for d in ds if d is not None and d < 32)
    if min(N_CHIP - 1, c // chunk) == CHIP:
        miss = [(i / len(s), s[i]) for i, d in enumerate(ds) if d is None or d >= 32]
        l2_streams[c] = miss
merged = [p for _, _, p in sorted((pos, c, p) for c, ms in l2_streams.items() for pos, p in ms)]
d2 = stack(merged, 1024); n2, cold2, H2 = hist(d2); hit2 = sum(1 for d in d2 if d is not None and d < 1024)
lab = lambda k: f"{bins[k]}-{bins[k+1]-1}" if bins[k + 1] < (1 << 30) else f">={bins[k]}"
print(f"trace {f.split('/')[-1]}: grid {h.get('grid dim')}, CTAs {len(cta)}")
print(f"L1 (per CTA, LRU 32): accesses {L1n:,}, first-touch {100*L1cold/L1n:.1f} %, predicted hit {100*L1hit/L1n:.1f} %; reuse distance of re-accesses: "
      + ", ".join(f"{lab(k)}: {100*L1H[k]/max(1,L1n-L1cold):.1f}%" for k in sorted(L1H)))
print(f"L2 (chiplet {CHIP}, {len(l2_streams)} CTAs, L1 misses merged by progress, LRU 1024): lookups {n2:,}, first-touch on chiplet {100*cold2/max(1,n2):.1f} %, predicted hit {100*hit2/max(1,n2):.1f} %; reuse distance: "
      + ", ".join(f"{lab(k)}: {100*H2[k]/max(1,n2-cold2):.1f}%" for k in sorted(H2)))
