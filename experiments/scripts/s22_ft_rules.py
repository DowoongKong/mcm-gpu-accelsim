#!/usr/bin/env python3
# Evaluate first-touch owner rules on one kernel's per-chiplet candidates against the reference map, and compare the
# aggregate placement statistics (owner share per chiplet, remote-access share) of the best rule with the reference.
# Usage: ft_rules.py <cand.json> <placement_mcm.map>
import json, sys
J = json.load(open(sys.argv[1])); ref = {}
for line in open(sys.argv[2]):
    if line.startswith('D '):
        t = line.split(); ref[int(t[1], 0)] = int(t[2])
pages = {int(p): {int(c): v for c, v in d.items()} for p, d in J['pages'].items()}
pages = {p: d for p, d in pages.items() if p in ref}
contended = sum(1 for d in pages.values() if len(d) > 1)
print(f"k{J['kernel']} grid={J['grid']} chunk={J['chunk']} per_wave={J['per_wave']} pages(in map)={len(pages):,} touched by >1 chiplet={contended:,}")
def score(rule, name):
    agree = 0; own = [0]*4; rem = 0; tot = 0
    for p, d in pages.items():
        o = min(d, key=lambda c: rule(d[c], c)); agree += (o == ref[p]); own[o] += 1
        for c, v in d.items(): tot += v[3]; rem += v[3] if c != o else 0
    print(f"  {name:48s} agree={100*agree/len(pages):6.2f}%  owner share={[f'{100*x/len(pages):.1f}' for x in own]}  remote access={100*rem/tot:5.1f}%")
# v = [wave, insn, local, count]
score(lambda v, c: (v[0], v[1], c), "A: (wave, insn), tie -> lower chiplet")
score(lambda v, c: (v[0], v[1], -c), "B: (wave, insn), tie -> higher chiplet")
score(lambda v, c: (v[0], v[1], v[2], c), "C: (wave, insn, local), tie -> lower chiplet")
score(lambda v, c: (v[0], v[2], v[1], c), "D: (wave, local, insn), tie -> lower chiplet")
for a in (1, 2, 4, 8, 16, 32, 64):
    score(lambda v, c, a=a: (v[0], v[1] + a*v[2], c), f"E{a}: (wave, insn + {a}*local), tie -> lower chiplet")
    score(lambda v, c, a=a: (v[0], v[1] + a*v[2], -c), f"F{a}: (wave, insn + {a}*local), tie -> higher chiplet")
own = [0]*4; rem = 0; tot = 0
for p, d in pages.items():
    own[ref[p]] += 1
    for c, v in d.items(): tot += v[3]; rem += v[3] if c != ref[p] else 0
print(f"  {'REFERENCE':48s} agree=100.00%  owner share={[f'{100*x/len(pages):.1f}' for x in own]}  remote access={100*rem/tot:5.1f}%")
