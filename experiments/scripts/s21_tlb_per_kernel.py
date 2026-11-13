#!/usr/bin/env python3
# Step 21 follow-up (docs/s21_llm_metrics_by_phase.ko.md section 7): per-kernel TLB behaviour of one simulation arm.
# For every kernel (statistics block, matched by ORDER: the n-th block is the n-th kernel of the kernel list, and the
# "kernel_name =" line inside a block belongs to the NEXT launch, so names are taken from the trace headers instead):
#   cycles, thread instructions, L1 TLB lookups (= (warp, page) requests: one per distinct page a warp memory
#   instruction touches), L1 hit rate, L2 TLB hits / unique misses (walks) / merges, L2 hit rate = hits/(hits+misses)
#   as the simulator prints it, and lookups per unique miss.
# Usage: s21_tlb_per_kernel.py <run.log> <kind: 4k|64k> <kernel list .g> [first-last ...]
import lzma, os, re, sys
log, kind, klist = sys.argv[1:4]
ranges = [tuple(int(x) for x in r.split('-')) for r in sys.argv[4:]] or [(1, 10**9)]
files = [l.strip() for l in open(klist) if 'kernel-' in l]
names = {}
for i, f in enumerate(files, 1):
    p = os.path.join(os.path.dirname(klist), f)
    try:
        with lzma.open(p, 'rt') as fh:
            for _ in range(5):
                l = fh.readline()
                if l.startswith('-kernel name'): names[i] = l.split('=', 1)[1].strip()
                if l.startswith('-grid dim'): names[i] += ' ' + l.split('=', 1)[1].strip()
    except OSError: names[i] = f
B = []; cur = None
for line in open(log, errors='ignore'):
    if line.startswith('kernel_launch_uid ='):
        cur = {'l1': [0, 0, 0], 'l2': [0, 0, 0], 'insn': 0, 'cyc': 0}; B.append(cur); continue
    if cur is None: continue
    if line.startswith(f'[L1_TLB_{kind}_STAT] SM='):
        m = re.search(r'hits=(\d+) misses=(\d+) mshr_hits=(\d+)', line)
        for i in range(3): cur['l1'][i] += int(m.group(i + 1))
    elif line.startswith(f'[L2_TLB_{kind}_STAT] chiplet='):
        m = re.search(r'hits=(\d+) misses=(\d+) mshr_hits=(\d+)', line)
        for i in range(3): cur['l2'][i] += int(m.group(i + 1))
    elif line.startswith('gpu_tot_sim_insn ='): cur['insn'] = int(line.split('=')[1])
    elif line.startswith('gpu_tot_sim_cycle ='): cur['cyc'] = int(line.split('=')[1])
B = [b for b in B if b['cyc']]
print(f"# {log}: {len(B)} kernel blocks, kind={kind}")
print(f"{'k':>3s} {'kernel (grid)':52s} {'cycles':>11s} {'thr_insn':>15s} {'L1_look':>11s} {'L1hit%':>6s} {'L2_hit':>11s} {'L2_miss':>10s} {'L2_merge':>11s} {'L2hit%':>6s} {'look/miss':>9s}")
prev = {'l1': [0, 0, 0], 'l2': [0, 0, 0], 'insn': 0, 'cyc': 0}
tot = {'cyc': 0, 'insn': 0, 'l1': [0, 0, 0], 'l2': [0, 0, 0]}
for i, b in enumerate(B, 1):
    l1 = [b['l1'][j] - prev['l1'][j] for j in range(3)]; l2 = [b['l2'][j] - prev['l2'][j] for j in range(3)]
    cyc = b['cyc'] - prev['cyc']; ins = b['insn'] - prev['insn']; prev = b
    if not any(a <= i <= z for a, z in ranges): continue
    for j in range(3): tot['l1'][j] += l1[j]; tot['l2'][j] += l2[j]
    tot['cyc'] += cyc; tot['insn'] += ins
    nm = names.get(i, '?'); nm = re.sub(r'^_ZN\d+', '', nm)
    print(f"{i:3d} {nm[:52]:52s} {cyc:>11,} {ins:>15,} {sum(l1):>11,} {100 * l1[0] / max(1, l1[0] + l1[1]):6.1f} {l2[0]:>11,} {l2[1]:>10,} {l2[2]:>11,} {100 * l2[0] / max(1, l2[0] + l2[1]):6.1f} {sum(l2) / max(1, l2[1]):9.1f}")
l1, l2 = tot['l1'], tot['l2']
print(f"{'sum':>3s} {'(selected kernels)':52s} {tot['cyc']:>11,} {tot['insn']:>15,} {sum(l1):>11,} {100 * l1[0] / max(1, l1[0] + l1[1]):6.1f} {l2[0]:>11,} {l2[1]:>10,} {l2[2]:>11,} {100 * l2[0] / max(1, l2[0] + l2[1]):6.1f} {sum(l2) / max(1, l2[1]):9.1f}")
