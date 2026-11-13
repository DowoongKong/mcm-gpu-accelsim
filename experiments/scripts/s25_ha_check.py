#!/usr/bin/env python3
# S25-2/3 prototype (2026-11-10). H-A check on one GEMM kernel trace: classify every global-memory instruction's addresses as weight (memcpy regions),
# A-operand region (the activation written by the preceding kernel), or other; for the A region report how many distinct
# 16 KB rows are touched per CTA and overall (M=1 means only row 0 is valid). Also prints per-PC LDGSTS region split.
import lzma, sys, collections
klist, gemm, prev = sys.argv[1], sys.argv[2], sys.argv[3]
W = []
for l in open(klist):
    if l.startswith('MemcpyHtoD'):
        _, va, sz = l.strip().split(','); W.append((int(va, 16), int(sz)))
W.sort()
def in_w(a):
    for b, s in W:
        if b <= a < b + s: return True
    return False
def addrs(t, k):
    if int(t[k]) == 0: return []
    fmt = int(t[k + 1]); base = int(t[k + 2], 16); act = bin(int(t[1], 16)).count("1")
    if fmt == 1:
        s = int(t[k + 3]); return [base + i * s for i in range(act)]
    if fmt == 2:
        out = [base]; a = base
        for d in t[k + 3:k + 3 + act - 1]: a += int(d); out.append(a)
        return out
    return [int(x, 16) for x in t[k + 2:k + 2 + act]]
GLOBAL = ('LDG', 'STG', 'LDGSTS', 'ATOM', 'RED', 'LD', 'ST')
def scan(f, cb):
    tb = None
    with lzma.open(f, 'rt') as fh:
        for line in fh:
            if line.startswith('thread block'):
                tb = tuple(int(v) for v in line.split('=')[1].split(',')); continue
            t = line.split()
            if len(t) < 4 or not t[0].isalnum() or line[0] in '-#tw': continue
            try: dn = int(t[2])
            except ValueError: continue
            oi = 3 + dn; op = t[oi].split('.')[0]
            if op not in GLOBAL: continue
            sn = int(t[oi + 1]); kk = oi + 2 + sn
            if kk >= len(t) or int(t[kk]) == 0: continue
            cb(tb, t[0], op, int(t[kk]), addrs(t, kk))
# 1) previous kernel's STG range = A operand (valid rows)
lo, hi = [None], [None]
def prev_cb(tb, pc, op, w, A):
    if op != 'STG': return
    for a in A:
        lo[0] = a if lo[0] is None or a < lo[0] else lo[0]; hi[0] = a + w if hi[0] is None or a + w > hi[0] else hi[0]
scan(prev, prev_cb)
print(f'prev kernel STG range: 0x{lo[0]:x}..0x{hi[0]:x}  = {hi[0]-lo[0]} B')
A0 = lo[0]
# 2) GEMM kernel
pcreg = collections.defaultdict(lambda: collections.Counter())
rows_all = set(); rows_per_cta = collections.defaultdict(set); bytes_cls = collections.Counter(); cta_bytes = collections.defaultdict(collections.Counter)
other_lo, other_hi = [None], [None]
def gemm_cb(tb, pc, op, w, A):
    for a in A:
        if in_w(a): c = 'weight'
        elif A0 - (1 << 20) <= a < A0 + (1 << 24): c = 'Aregion'   # within 1 MB before .. 16 MB after the activation base
        else: c = 'other'
        pcreg[(pc, op)][c] += 1; bytes_cls[c] += w; cta_bytes[tb][c] += w
        if c == 'Aregion':
            r = (a - A0) // 16384; rows_all.add(r); rows_per_cta[tb].add(r)
        elif c == 'other':
            other_lo[0] = a if other_lo[0] is None or a < other_lo[0] else other_lo[0]
            other_hi[0] = a if other_hi[0] is None or a > other_hi[0] else other_hi[0]
scan(gemm, gemm_cb)
tot = sum(bytes_cls.values())
print('bytes by class:', {k: f'{v/2**20:.1f} MB ({100*v/tot:.1f} %)' for k, v in bytes_cls.items()}, 'CTAs', len(cta_bytes))
print(f'A-region rows touched overall: {len(rows_all)} (min {min(rows_all) if rows_all else None}, max {max(rows_all) if rows_all else None}); rows per CTA: min {min(len(s) for s in rows_per_cta.values())} max {max(len(s) for s in rows_per_cta.values())}')
print(f'other-region span: 0x{other_lo[0]:x}..0x{other_hi[0]:x}' if other_lo[0] is not None else 'no other')
ex = next(iter(cta_bytes)); print('example CTA', ex, {k: f'{v/1024:.0f} KB' for k, v in cta_bytes[ex].items()})
print('per-PC LDGSTS/STG region split (pc op: weight/Aregion/other lane-accesses):')
for (pc, op), c in sorted(pcreg.items(), key=lambda x: -sum(x[1].values()))[:14]:
    print(f'  {pc} {op:7s} {dict(c)}')
