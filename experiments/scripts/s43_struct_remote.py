#!/usr/bin/env python3
# Step 43 (docs/s43_anomaly_review.ko.md): structural remote ratio of page-table and data accesses for one decode window, no simulation.
# For every (page, chiplet that touches it) pair of every kernel (s37 tables, KERNEL_PARTITION chip masks) compare the accessing chiplet
# with the owner chiplet of the data page and of each page-table level, taken from a placement snapshot v3 (D / N 1-3 / R lines).
# A pair is a lower bound of "this chiplet must translate this page at least once"; repeated TLB misses are not modelled.
# Usage: s43_struct_remote.py <s37 window dir> <placement_end.map> [--mem-gib 128|256] [--by-chip]
import sys, gzip, glob, json, re, collections
args = [a for a in sys.argv[1:] if not a.startswith('--')]
win, mp = args[0], args[1]
MEM = 128
if '--mem-gib' in sys.argv: MEM = int(sys.argv[sys.argv.index('--mem-gib') + 1]); args = [a for a in args if a != str(MEM)]
BYCHIP = '--by-chip' in sys.argv
D, N = {}, {1: {}, 2: {}, 3: {}}; root_chip = 0
chip_shift = (MEM * 2**30 // 4).bit_length() - 1          # bytes per chiplet = 2^chip_shift (37-bit map: 32 GiB -> 35)
for l in open(mp):
    t = l.split()
    if not t or t[0].startswith('#'): continue
    if t[0] == 'D': D[int(t[1], 16)] = int(t[2])
    elif t[0] == 'N': N[int(t[1])][int(t[2], 16)] = int(t[3])
    elif t[0] == 'R': root_chip = (int(t[1], 16) << 16) >> chip_shift
LV = (('data', lambda v: D.get(v)), ('leaf', lambda v: N[3].get(v >> 5)), ('PD', lambda v: N[2].get(v >> 14)),
      ('PDP', lambda v: N[1].get(v >> 23)), ('root', lambda v: root_chip))
grp = lambda lab: 'weight' if lab.startswith('w.') else ('kv' if lab.startswith('kv_cache') else ('act' if lab.startswith('m.') else 'other'))
tot = collections.Counter(); out = []
for f in sorted(glob.glob(win + '/k*.json'), key=lambda x: int(re.search(r'k(\d+)\.json', x).group(1))):
    J = json.load(open(f)); k = f.split('/')[-1][:-5]
    c = collections.Counter(); chipc = collections.Counter()
    for l in gzip.open(f.replace('.json', '.pages.tsv.gz'), 'rt'):
        t = l.rstrip('\n').split('\t')
        if t[0] == 'vpn': continue
        v = int(t[0], 16); mask = int(t[4]); g = grp(t[1])
        if v not in D: c['unmapped'] += 1; continue
        for ch in range(4):
            if not mask >> ch & 1: continue
            c['pairs'] += 1; c['pairs.' + g] += 1; chipc[(ch, 'pairs')] += 1
            for name, fn in LV:
                o = fn(v)
                if o is None: c['nolvl.' + name] += 1; continue
                if o != ch:
                    c[name] += 1; c[name + '.' + g] += 1; chipc[(ch, name)] += 1
    for kk, vv in c.items(): tot[kk] += vv
    out.append((k, J['grid'], J['kernel'], c, chipc))
pct = lambda a, b: f"{100 * a / b:5.1f}" if b else "    -"
name = lambda s: re.sub(r'^_ZN?\d*', '', s)[:34]
print(f"{'kernel':7s} {'grid':12s} {'name':34s} {'pairs':>7s} | remote %: {'data':>5s} {'leaf':>5s} {'PD':>5s} {'PDP':>5s} {'root':>5s} | leaf % by group (weight / kv / act)")
for k, g, nm, c, chipc in out:
    p = c['pairs']
    gl = ' / '.join(pct(c['leaf.' + x], c['pairs.' + x]) for x in ('weight', 'kv', 'act'))
    print(f"{k:7s} {g:12s} {name(nm):34s} {p:7d} | {'':9s} {pct(c['data'], p)} {pct(c['leaf'], p)} {pct(c['PD'], p)} {pct(c['PDP'], p)} {pct(c['root'], p)} | {gl}")
    if BYCHIP: print('        by chip (pairs, leaf remote %): ' + '  '.join(f"c{ch} {chipc[(ch, 'pairs')]} {pct(chipc[(ch, 'leaf')], chipc[(ch, 'pairs')])}" for ch in range(4)))
p = tot['pairs']
print(f"{'ALL':7s} {'':12s} {'':34s} {p:7d} | {'':9s} {pct(tot['data'], p)} {pct(tot['leaf'], p)} {pct(tot['PD'], p)} {pct(tot['PDP'], p)} {pct(tot['root'], p)} | "
      + ' / '.join(pct(tot['leaf.' + x], tot['pairs.' + x]) for x in ('weight', 'kv', 'act')) + f"   (unmapped pages {tot['unmapped']}, missing level {sum(v for k2, v in tot.items() if k2.startswith('nolvl'))})")
lv = collections.Counter((lvl, o) for lvl in (1, 2, 3) for o in N[lvl].values())
print('PT node owners: ' + '  '.join(f"L{lvl}({'PDP' if lvl == 1 else 'PD' if lvl == 2 else 'leaf'}) " + '/'.join(str(lv[(lvl, ch)]) for ch in range(4)) for lvl in (1, 2, 3)) + f"  root chip {root_chip}")
