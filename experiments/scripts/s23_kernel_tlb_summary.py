#!/usr/bin/env python3
# Step 23 (docs/s23_llm_kernel_tlb_anatomy.ko.md): join the per-kernel trace profiles (s23_kernel_tlb_profile.py JSON)
# with the per-kernel statistics of one simulation run and print markdown tables. Statistics blocks are matched to
# kernels by ORDER (block n = n-th kernel of the kernel list); every printed counter is cumulative, so each kernel's
# value is the difference between consecutive blocks. The "cold lower bound" is the number of distinct (chiplet, page)
# pairs the kernel touches: with one L2 TLB per chiplet every such pair costs at least one unique miss unless the entry
# survived from an earlier kernel (TLBs are not flushed between kernels). capacity = unique misses - cold lower bound.
# Usage: s23_kernel_tlb_summary.py <run.log> <4k|64k> <profile dir> <k ...>
import glob, json, os, re, sys
log, kind, pdir = sys.argv[1:4]; ks = [int(k) for k in sys.argv[4:]]
B = []; cur = None
XL = re.compile(r'^\[XLATSUM\] GPU kind=(\w+) out=(\w+) n=(\d+) avg=([\d.]+)')
for line in open(log, errors='ignore'):
    if line.startswith('kernel_launch_uid ='):
        cur = {'l1': [0, 0, 0], 'l2': [0, 0, 0, 0], 'insn': 0, 'cyc': 0, 'walks': [0] * 4, 'ptw_t': [0.0] * 4, 'x': {}}; B.append(cur); continue
    if cur is None: continue
    if line.startswith(f'[L1_TLB_{kind}_STAT] SM='):
        m = re.search(r'hits=(\d+) misses=(\d+) mshr_hits=(\d+)', line)
        for i in range(3): cur['l1'][i] += int(m.group(i + 1))
    elif line.startswith(f'[L2_TLB_{kind}_STAT] chiplet='):
        m = re.search(r'chiplet=(\d+) hits=(\d+) misses=(\d+) mshr_hits=(\d+).*entries_full=(\d+)', line)
        for i in range(3): cur['l2'][i] += int(m.group(i + 2))
        cur['l2'][3] += int(m.group(5))
    elif line.startswith('[GMMU_STAT] chiplet='): cur['_c'] = int(line.split('=')[1])
    elif line.strip().startswith('completed walks'): cur['walks'][cur['_c']] = int(line.split(':')[1])
    elif line.strip().startswith('avg PTW time'): cur['ptw_t'][cur['_c']] = float(line.split(':')[1].split()[0])
    elif line.startswith('gpu_tot_sim_insn ='): cur['insn'] = int(line.split('=')[1])
    elif line.startswith('gpu_tot_sim_cycle ='): cur['cyc'] = int(line.split('=')[1])
    else:
        m = XL.match(line)
        if m and m.group(1) == kind: cur['x'][m.group(2)] = (int(m.group(3)), float(m.group(4)))
B = [b for b in B if b['cyc']]
def diff(i):
    b = B[i - 1]; p = B[i - 2] if i >= 2 else {'l1': [0, 0, 0], 'l2': [0, 0, 0, 0], 'insn': 0, 'cyc': 0, 'walks': [0] * 4, 'x': {}}
    d = {'l1': [b['l1'][j] - p['l1'][j] for j in range(3)], 'l2': [b['l2'][j] - p['l2'][j] for j in range(4)],
         'insn': b['insn'] - p['insn'], 'cyc': b['cyc'] - p['cyc'], 'walks': [b['walks'][j] - p['walks'][j] for j in range(4)], 'x': {}}
    for o, (n, a) in b['x'].items():
        n0, a0 = p['x'].get(o, (0, 0.0)); dn = n - n0
        d['x'][o] = (dn, (n * a - n0 * a0) / dn if dn > 0 else 0.0)
    return d
P = {}
for k in ks:
    f = os.path.join(pdir, f'k{k:02d}.json')
    if os.path.exists(f): P[k] = json.load(open(f))
def short(n):
    n = re.sub(r'^_ZN\d+', '', n); n = n.replace('cutlass7Kernel2I', 'cutlass ').replace('EEvNT_6ParamsE', '')
    return n
layer_miss = sum(diff(k)['l2'][1] for k in ks)
print(f"# run={log} kind={kind} blocks={len(B)} kernels={ks}\n")
print("## A. 실행 형상\n")
print("| k | 커널 | grid | block | regs | shmem B | CTA 수 | CTA/SM | CTA/칩렛 | wave/칩렛 | cycles | warp insn |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in ks:
    p = P.get(k); d = diff(k)
    if not p: print(f"| {k} | (profile pending) | | | | | | | | | {d['cyc']:,} | |"); continue
    print(f"| {k} | {short(p['kernel_name'])[:60]} | {p['grid']} | {p['block']} | {p['nregs']} | {p['shmem']:,} | {p['n_cta']:,} | {p['ctas_per_sm']} | {p['chunk']:,} | {p['waves_per_chiplet']} | {d['cyc']:,} | {d['insn'] // 32:,} |")
print("\n## B. 접근 패턴 (트레이스)\n")
print("| k | mem warp insn | 조회(lookup) | 페이지/명령 | 고유 페이지 (MiB) | CTA당 페이지 (읽기/쓰기) | CTA당 재조회/페이지 | CTA/페이지 | 칩렛/페이지 | 칩렛0 첫 wave live (읽기/쓰기) | 명령 종류 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for k in ks:
    p = P.get(k)
    if not p: print(f"| {k} | pending |"); continue
    ops = ' '.join(f"{o}:{n:,}" for o, n in sorted(p['insn_by_op'].items(), key=lambda x: -x[1]))
    print(f"| {k} | {p['mem_warp_insn']:,} | {p['lookups']:,} | {p['pages_per_insn']} | {p['distinct_pages']:,} ({p['distinct_MiB']}) | {p['cta_pages_mean']} ({p['cta_read_pages_mean']}/{p['cta_write_pages_mean']}) | {p['reuse_per_cta_page']} | {p['ctas_per_page_mean']} | {p['chips_per_page_mean']} | {p['wave0_pages']} ({p['wave0_read_pages']}/{p['wave0_write_pages']}) | {ops} |")
print("\n## C. TLB 실측과 miss 분해 (시뮬)\n")
print("| k | L1 조회 | L1 hit | L1 merge | L2 hit | L2 unique miss | L2 merge | L2 hit% | 조회/miss | MSHR full 재시도 | cold 하한 (칩렛,페이지) | capacity = miss − cold | 이월(cold − miss) | layer miss 비중 | walk/칩렛 | PTW 평균 cycle |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
T = {'look': 0, 'l1h': 0, 'l2': [0, 0, 0], 'cold': 0, 'cap': 0, 'carry': 0}
for k in ks:
    p = P.get(k); d = diff(k); l1, l2 = d['l1'], d['l2']
    cold = p['cold_lower_bound'] if p else 0; cap = max(0, l2[1] - cold); carry = max(0, cold - l2[1])
    T['look'] += sum(l1); T['l1h'] += l1[0]; T['cold'] += cold; T['cap'] += cap; T['carry'] += carry
    for j in range(3): T['l2'][j] += l2[j]
    n_ptw, t_ptw = d['x'].get('PTW', (0, 0.0)); ptw = f"{t_ptw:.0f}" if n_ptw >= 100 else "–"
    print(f"| {k} | {sum(l1):,} | {100 * l1[0] / max(1, l1[0] + l1[1]):.1f}% | {l1[2]:,} | {l2[0]:,} | {l2[1]:,} | {l2[2]:,} | {100 * l2[0] / max(1, l2[0] + l2[1]):.1f}% | {sum(l2[:3]) / max(1, l2[1]):.1f} | {l2[3]:,} | {cold:,} | {cap:,} ({100 * cap / max(1, l2[1]):.0f}%) | {carry:,} | {100 * l2[1] / max(1, layer_miss):.1f}% | {'/'.join(f'{w:,}' for w in d['walks'])} | {ptw} |")
l2 = T['l2']
print(f"| 합 | {T['look']:,} | {100 * T['l1h'] / max(1, T['look']):.1f}% | | {l2[0]:,} | {l2[1]:,} | {l2[2]:,} | {100 * l2[0] / max(1, l2[0] + l2[1]):.1f}% | {sum(l2) / max(1, l2[1]):.1f} | | {T['cold']:,} | {T['cap']:,} ({100 * T['cap'] / max(1, l2[1]):.0f}%) | {T['carry']:,} | 100% | | |")
print("\n## D. 텐서(연속 주소 구간)별 접근 — 조회 수 상위\n")
for k in ks:
    p = P.get(k)
    if not p: continue
    print(f"\n### k{k} {short(p['kernel_name'])[:70]} — 구간 {p['n_segments']}개\n")
    print("| 시작 VA | 페이지 | MiB | 조회 | 읽기 명령 | 쓰기 명령 | CTA/페이지 | 칩렛/페이지 | 조회/페이지 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in p['segments'][:8]:
        print(f"| {s['first_va']} | {s['pages']:,} | {s['MiB']} | {s['lookups']:,} | {s['reads']:,} | {s['writes']:,} | {s['ctas_per_page']} | {s['chips_per_page']} | {s['lookups_per_page']} |")
