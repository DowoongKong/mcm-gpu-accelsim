# Usage: tlb_3c_model.py <kernel-N-...traceg.xz> [page_shift=12]   (docs/llm_tlb_miss_3c_analysis.ko.md)
# 3C decomposition of L2 TLB misses for one GEMM kernel trace (approximate interleaving model).
# CTA -> chiplet: CHUNK_BINDING chunk 4 (cta // 4 % 4); CTA -> SM: resident CTAs spread over 64 SMs per chiplet.
# Per chiplet the resident warps advance round-robin one memory instruction at a time. Each instruction's pages go through
# a per-SM L1 TLB (32 entries, fully associative LRU); L1 misses go to the chiplet L2 TLB modeled three ways:
# infinite (compulsory), fully associative LRU 1024 (compulsory + capacity), 128 sets x 8 ways LRU, set = vpn % 128 (all).
import collections, subprocess, sys
from collections import OrderedDict
path = sys.argv[1]; SHIFT = int(sys.argv[2]) if len(sys.argv) > 2 else 12
streams = collections.defaultdict(list)     # (cta, warp) -> list of page lists
cta = warp = None; gx = gy = 1
p = subprocess.Popen(["xz", "-dc", path], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
for line in p.stdout:
    if line.startswith("-grid dim"):
        g = line.split("=")[1].strip().strip("()").split(","); gx, gy = int(g[0]), int(g[1]); continue
    if line.startswith("thread block"):
        x, y, z = (int(v) for v in line.split("=")[1].split(",")); cta = x + y * gx + z * gx * gy; continue
    if line.startswith("warp ="): warp = int(line.split("=")[1]); continue
    t = line.split()
    if len(t) < 5 or not t[2].isdigit(): continue
    i = 3 + int(t[2]); op = t[i]; i += 2 + int(t[i + 1]); w = int(t[i])
    if w == 0 or len(t) <= i + 1 or not op.startswith(("LDG", "STG", "LDGSTS", "ATOMG", "RED")): continue
    mode, n = int(t[i + 1]), bin(int(t[1], 16)).count("1")
    if mode == 0: a = [int(v, 16) for v in t[i + 2:i + 2 + n]]
    elif mode == 1: b, s = int(t[i + 2], 16), int(t[i + 3]); a = [b + k * s for k in range(n)]
    else:
        a = [int(t[i + 2], 16)]
        for d in t[i + 3:i + 2 + n]: a.append(a[-1] + int(d))
    streams[(cta, warp)].append(sorted({v >> SHIFT for v in a}))
p.wait()

class LRU:
    def __init__(s, cap): s.cap, s.d = cap, OrderedDict()
    def access(s, k):
        if k in s.d: s.d.move_to_end(k); return True
        s.d[k] = None
        if len(s.d) > s.cap: s.d.popitem(last=False)
        return False
class SetAssoc:
    def __init__(s, sets, ways): s.sets = [LRU(ways) for _ in range(sets)]; s.n = sets
    def access(s, k): return s.sets[k % s.n].access(k)

by_chip = collections.defaultdict(list)
for key in sorted(streams): by_chip[(key[0] // 4) % 4].append(key)
tot = collections.Counter()
for chip, keys in by_chip.items():
    ctas = sorted({k[0] for k in keys}); sm_of = {c: i % 64 for i, c in enumerate(ctas)}
    l1 = {s: LRU(32) for s in range(64)}; seen = set(); fa = LRU(1024); sa = SetAssoc(128, 8)
    ptr = {k: 0 for k in keys}; live = list(keys)
    while live:
        nxt = []
        for k in live:
            pages = streams[k][ptr[k]]; ptr[k] += 1
            for pg in pages:
                tot["refs"] += 1
                if l1[sm_of[k[0]]].access(pg): continue
                tot["l1_miss"] += 1
                if pg not in seen: seen.add(pg); tot["compulsory"] += 1
                if not fa.access(pg): tot["fa_miss"] += 1
                if not sa.access(pg): tot["sa_miss"] += 1
            if ptr[k] < len(streams[k]): nxt.append(k)
        live = nxt
c = tot["compulsory"]; cap = tot["fa_miss"] - c; conf = tot["sa_miss"] - tot["fa_miss"]
print(f"{path.split('/')[-1][:24]} page_shift {SHIFT}: page refs {tot['refs']:,}, L1 misses {tot['l1_miss']:,}, model L2 misses {tot['sa_miss']:,}")
print(f"   compulsory {c:,} ({c/tot['sa_miss']*100:.1f}%)  capacity {cap:,} ({cap/tot['sa_miss']*100:.1f}%)  conflict {conf:,} ({conf/tot['sa_miss']*100:.1f}%)")
