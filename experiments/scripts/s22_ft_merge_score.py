#!/usr/bin/env python3
# Merge the per-kernel first-touch candidates of s22_ft_predict2.py (k1..kN, in kernel order) into one owner map with
# rule C (lowest (wave, insn_idx, local_cta_idx); tie -> lowest chiplet) and score it against a reference placement map:
# overall agreement, agreement split by tied/untied pages, per first-touch kernel, and the owner share per chiplet.
# Usage: s22_ft_merge_score.py <candidate dir with k<N>.json> <placement_mcm.map> <n kernels>
import json, sys, os, collections
cdir, mapf, nk = sys.argv[1], sys.argv[2], int(sys.argv[3])
ref = {int(l.split()[1], 0): int(l.split()[2]) for l in open(mapf) if l.startswith('D ')}
owner, first_k, tied = {}, {}, {}
for k in range(1, nk + 1):
    J = json.load(open(os.path.join(cdir, f'k{k}.json')))
    for p, d in J['pages'].items():
        p = int(p)
        if p in owner: continue
        keys = {int(c): tuple(v[:3]) for c, v in d.items()}
        mn = min(keys.values()); ties = [c for c in keys if keys[c] == mn]
        owner[p] = min(ties); first_k[p] = k; tied[p] = len(ties) > 1
common = [p for p in owner if p in ref]
print(f'pages predicted {len(owner):,}  ref {len(ref):,}  common {len(common):,}')
ok = lambda ps: sum(owner[p] == ref[p] for p in ps)
print(f'rule C agreement: {ok(common):,}/{len(common):,} = {100*ok(common)/len(common):.1f} %')
t = [p for p in common if tied[p]]; u = [p for p in common if not tied[p]]
print(f'  tied pages {len(t):,}: {100*ok(t)/max(1,len(t)):.1f} %   untied pages {len(u):,}: {100*ok(u)/max(1,len(u)):.1f} %')
per = collections.defaultdict(lambda: [0, 0])
for p in common: per[first_k[p]][1] += 1; per[first_k[p]][0] += (owner[p] == ref[p])
for k in sorted(per): print(f'  k{k:02d}: {per[k][0]:6d}/{per[k][1]:6d} = {100*per[k][0]/per[k][1]:5.1f} %')
sp = [sum(1 for p in common if owner[p] == c) for c in range(4)]; sr = [sum(1 for p in common if ref[p] == c) for c in range(4)]
print('owner share predicted', [f'{100*x/len(common):.1f}' for x in sp], ' ref', [f'{100*x/len(common):.1f}' for x in sr])
