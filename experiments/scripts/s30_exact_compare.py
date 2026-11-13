#!/usr/bin/env python3
# s30 (docs/s30_exact_placement_replay.ko.md): per-kernel cycles and walks (L2 TLB unique misses) of the exact-replay arms
# next to the h1 owner-only arms, plus walk service p50/p90 from the L2MISSLOG C lines of the exact arms.
# Usage: s30_exact_compare.py <debug_exactmap dir> <h1 sim dir>
import re, sys
X, O = sys.argv[1], sys.argv[2]
NAMES = {2: 'QKV', 9: 'O', 12: 'gate_up', 14: 'down'}

def per_kernel(log):
    blocks = []; b = None
    for line in open(log, errors='ignore'):
        if line.startswith('kernel_launch_uid ='): b = {'c': 0, 'm': {}}; blocks.append(b); continue
        if b is None: continue
        if line.startswith('gpu_sim_cycle ='): b['c'] = int(line.split('=')[1])
        m = re.match(r'^\[L2_TLB_64k_STAT\] chiplet=(\d+) hits=\d+ misses=(\d+)', line)
        if m: b['m'][int(m.group(1))] = int(m.group(2))
    out = []; p = 0
    for b in blocks:
        t = sum(b['m'].values()); out.append((b['c'], t - p)); p = t
    return out

def svc(arm_dir, bounds):
    res = {}
    import bisect
    vals = {k: [] for k in bounds}
    for line in open(f'{arm_dir}/l2miss.log', errors='ignore'):
        if line[0] != 'C': continue
        p = line.split(); t = int(p[1]); s = t - int(p[5])
        for k, (a, b) in bounds.items():
            if a <= t < b: vals[k].append(s); break
    for k, v in vals.items():
        v.sort(); res[k] = (v[len(v) // 2], v[int(.9 * len(v))]) if v else (0, 0)
    return res

arms = [('h1 ref', f'{O}/ref_mcm'), ('h1 base', f'{O}/base_mcm'), ('h1 ptwhop0', f'{O}/ptwhop0_mcm'),
        ('ref (v3)', f'{X}/ref_mcm'), ('base_exact', f'{X}/base_exact'), ('ptwhop0_exact', f'{X}/ptwhop0_exact')]
R = {n: per_kernel(f'{d}/run.log') for n, d in arms}
print(f"{'kernel':10s}" + ''.join(f"{n:>22s}" for n, _ in arms))
for k in (2, 9, 12, 14):
    print(f"{NAMES[k]:10s}" + ''.join(f"{R[n][k-1][0]:>12,}/{R[n][k-1][1]:>9,}" for n, _ in arms))
print(f"{'total':10s}" + ''.join(f"{sum(x[0] for x in R[n]):>12,}/{sum(x[1] for x in R[n]):>9,}" for n, _ in arms))
print('(cycles / walks per kernel)')
c = {n: sum(x[0] for x in R[n]) for n, _ in arms}
print(f"cycle ratios: ref/base_exact {c['ref (v3)']/c['base_exact']:.4f} (owner-only h1 ref/base {c['h1 ref']/c['h1 base']:.4f}); "
      f"ptwhop0_exact/base_exact {c['ptwhop0_exact']/c['base_exact']:.4f} (h1 {c['h1 ptwhop0']/c['h1 base']:.4f})")
for n, d in arms[3:]:
    tot = 0; bounds = {}
    for i, (cy, _) in enumerate(R[n], 1):
        bounds[i] = (tot, tot + cy); tot += cy
    s = svc(d, {k: bounds[k] for k in (2, 9, 12, 14)})
    print(f"walk service p50/p90 {n:14s}: " + ', '.join(f"{NAMES[k]} {s[k][0]}/{s[k][1]}" for k in (2, 9, 12, 14)))
