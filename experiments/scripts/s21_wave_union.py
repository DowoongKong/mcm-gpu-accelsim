#!/usr/bin/env python3
# Union of distinct 64 KB / 4 KB pages over the first W thread blocks of a kernel trace (the first wave of chiplet 0
# under chunk binding with one CTA per SM) and the per-CTA mean. Usage: wave_union.py <trace.xz> <W>
import lzma, re, sys
f = sys.argv[1]; W = int(sys.argv[2])
MEM = re.compile(r'\b(LDG|STG|LDGSTS|LD|ST|ATOM|ATOMG|RED)\b')
def addrs(t, k):
    if int(t[k]) == 0: return []
    fmt = int(t[k+1]); base = int(t[k+2], 16); act = bin(int(t[1], 16)).count('1')
    if fmt == 1: s = int(t[k+3]); return [base + i*s for i in range(act)]
    if fmt == 2:
        out = [base]; a = base
        for d in t[k+3:k+3+act-1]: a += int(d); out.append(a)
        return out
    return [int(x, 16) for x in t[k+2:k+2+act]]
u64 = set(); u4 = set(); n = 0; per = []; cur = None
with lzma.open(f, 'rt') as fh:
    for line in fh:
        if line.startswith('thread block ='):
            if cur is not None: per.append(len(cur))
            n += 1
            if n > W: cur = None; break
            cur = set(); continue
        if cur is None: continue
        t = line.split()
        if len(t) < 5 or not re.fullmatch(r'[0-9a-f]{4,}', t[0]) or not MEM.search(line): continue
        dn = int(t[2]); oi = 3 + dn; sn = int(t[oi+1]); k = oi + 2 + sn
        try: A = addrs(t, k)
        except Exception: continue
        for a in A: p = a >> 16; cur.add(p); u64.add(p); u4.add(a >> 12)
if cur is not None: per.append(len(cur))
print(f"{f.split('/')[-1]} first {len(per)} CTAs: union pages64={len(u64):,} union pages4={len(u4):,} per-CTA pages64 mean={sum(per)/max(1,len(per)):.0f} min={min(per) if per else 0} max={max(per) if per else 0}")
