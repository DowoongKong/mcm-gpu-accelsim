#!/usr/bin/env python3
# S25-3 prototype (2026-11-10). Dump the global-memory instruction sequence of one CTA of a GEMM trace, decoded as (pc, op, mask, lanes, region, row, col-offset),
# plus a per-PC summary for that CTA. A0 = activation base (prev kernel STG min), weights = memcpy regions.
import lzma, sys, collections
klist, gemm, A0, want = sys.argv[1], sys.argv[2], int(sys.argv[3], 16), tuple(int(v) for v in sys.argv[4].split(','))
W = sorted((int(l.split(',')[1], 16), int(l.split(',')[2])) for l in open(klist) if l.startswith('MemcpyHtoD'))
def wbase(a):
    for b, s in W:
        if b <= a < b + s: return b
    return None
def addrs(t, k):
    fmt = int(t[k + 1]); base = int(t[k + 2], 16); act = bin(int(t[1], 16)).count("1")
    if fmt == 1: s = int(t[k + 3]); return [base + i * s for i in range(act)]
    if fmt == 2:
        out = [base]; a = base
        for d in t[k + 3:k + 3 + act - 1]: a += int(d); out.append(a)
        return out
    return [int(x, 16) for x in t[k + 2:k + 2 + act]]
GLOBAL = ('LDG', 'STG', 'LDGSTS', 'ATOM', 'RED', 'LD', 'ST')
tb = None; n = 0; per = collections.defaultdict(lambda: [0, set(), set(), set()]); warp = None
with lzma.open(gemm, 'rt') as fh:
    for line in fh:
        if line.startswith('thread block'):
            tb = tuple(int(v) for v in line.split('=')[1].split(','))
            if tb != want and n: break
            continue
        if tb != want: continue
        if line.startswith('warp'): warp = int(line.split('=')[1]); continue
        t = line.split()
        if len(t) < 4 or line[0] in '-#tw' or not t[0].isalnum(): continue
        try: dn = int(t[2])
        except ValueError: continue
        oi = 3 + dn; op = t[oi].split('.')[0]
        if op not in GLOBAL: continue
        sn = int(t[oi + 1]); kk = oi + 2 + sn
        if kk >= len(t) or int(t[kk]) == 0: continue
        A = addrs(t, kk); w = int(t[kk]); mask = t[1]
        if not A: continue   # predicate mask 0: no lane recorded an address
        wb = wbase(A[0])
        if wb is not None: reg = 'W'; rows = sorted({(a - wb) // 16384 for a in A}); cols = sorted({(a - wb) % 16384 for a in A})
        else:
            reg = 'A' if abs(A[0] - A0) < (1 << 24) else 'O'
            base = A0 if reg == 'A' else A[0] & ~0xfffff
            rows = sorted({(a - base) // 16384 for a in A}); cols = sorted({(a - base) % 16384 for a in A})
        n += 1
        if n <= 60:
            print(f'w{warp} {t[0]} {op:6s} mask={mask} lanes={len(A):2d} {reg} rows={rows[:4]}{"..." if len(rows)>4 else ""}({len(rows)}) cols={cols[:3]}..{cols[-1]} ({len(cols)}) w={w}')
        p = per[(t[0], op, reg)]; p[0] += 1; p[1].update(rows); p[2].update(cols); p[3].add(mask)
print(f'--- CTA {want}: {n} global mem warp-instructions; per PC:')
for (pc, op, reg), (c, rows, cols, masks) in sorted(per.items()):
    print(f'  {pc} {op:6s} {reg} count={c:5d} rows={len(rows):4d} [{min(rows)}..{max(rows)}] cols={len(cols):4d} [{min(cols)}..{max(cols)}] masks={sorted(masks)}')
