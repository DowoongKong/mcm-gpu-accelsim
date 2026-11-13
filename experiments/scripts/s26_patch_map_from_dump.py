#!/usr/bin/env python3
# Replace the owners of a predicted placement map (s26_make_placement_map.py) with a reference run's first-touch dump for
# every page the dump covers (used for the kernels whose first touch is a same-wave race, e.g. split-K GEMMs, where the
# config predictor can only guess "lowest chiplet"). N (node) lines are recomputed: a region's owner = the owner of its
# first page in the patched D set (the dump's N lines are used where the dump covers the region).
# Usage: s26_patch_map_from_dump.py <map in> <dump placement_mcm.map> <map out>
import sys
mi, dump, mo = sys.argv[1:4]
D = {}; order = []; hdr = []
for l in open(mi):
    if l.startswith('#'): hdr.append(l)
    elif l.startswith('D '):
        vpn, own = int(l.split()[1], 0), int(l.split()[2]); D[vpn] = own; order.append(vpn)
ref = {int(l.split()[1], 0): int(l.split()[2]) for l in open(dump) if l.startswith('D ')}
refN = {(int(l.split()[1]), int(l.split()[2], 0)): int(l.split()[3]) for l in open(dump) if l.startswith('N ')}
changed = sum(1 for p in order if p in ref and ref[p] != D[p]); covered = sum(1 for p in order if p in ref)
for p in order:
    if p in ref: D[p] = ref[p]
shift = int(next(h for h in hdr if h.startswith('# page_shift')).split()[2])
def idx(va): return (va >> 39) & 0x1ff, (va >> 30) & 0x1ff, (va >> 21) & 0x1ff
node = {1: {}, 2: {}, 3: {}}
for p in order:
    pml4, pdp, pd = idx(p << shift)
    for lvl, key in ((1, pml4), (2, (pml4 << 9) | pdp), (3, (pml4 << 18) | (pdp << 9) | pd)):
        if (lvl, key) in refN: node[lvl][key] = refN[(lvl, key)]
        else: node[lvl].setdefault(key, D[p])
with open(mo, 'w') as f:
    for h in hdr: f.write(h)
    f.write(f'# patched by s26_patch_map_from_dump.py from {dump}: {covered} pages covered, {changed} owners changed\n')
    for lvl in (1, 2, 3):
        for key in sorted(node[lvl]): f.write(f'N {lvl} 0x{key:x} {node[lvl][key]}\n')
    for p in order: f.write(f'D 0x{p:x} {D[p]} 0x0\n')
from collections import Counter
share = Counter(D.values())
print(f'{mi} -> {mo}: D {len(order)}, dump covers {covered}, owners changed {changed} ({100*changed/len(order):.1f} %), N {sum(len(v) for v in node.values())}, share ' +
      ' '.join(f'c{c}={share[c]}' for c in sorted(share)))
