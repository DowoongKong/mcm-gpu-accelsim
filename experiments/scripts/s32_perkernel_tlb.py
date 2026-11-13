#!/usr/bin/env python3
# s32 (docs/s32_qwen_s4096_decode_local_runs.ko.md, result 2): per-kernel deltas of the cumulative 64 KB TLB stats in an
# accel-sim run.log -- L1 lookups/hit rate, L2 hit rate, walks (= L2 unique misses), MPKI on the simulator warp count
# (gpgpu_n_tot_w_icount delta), warp insts per walk. Kernels below MIN share of walks and of warps are hidden.
# Usage: s32_perkernel_tlb.py <run.log> [MIN=0.02]
import re, sys
# per-kernel deltas of cumulative TLB stats in an accel-sim run.log
log = sys.argv[1]; MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
ks = []; cur = None
def short(n):
    for key in ['fused_moe', 'flash_fwd_splitkv_combine', 'flash_fwd_splitkv', 'flash_fwd', 'rms_norm', 'rotary', 'reshape_and_cache', 'act_and_mul', 'silu', 'moe_align', 'topk', 'moe_sum', 'elementwise', 'reduce', 'gemv', 'splitKreduce']:
        if key in n: return key
    m = re.search(r'(cutlass_80_\w+?|ampere_\w+?|sm80_\w+?|gemm\w*?)(EE|$|_ex)', n)
    return (m.group(1) if m else n)[:60]
st = None
for l in open(log, errors='ignore'):
    if l.startswith('-kernel name = '):
        cur = {'name': short(l.split('=', 1)[1].strip())}
    elif l.startswith('-grid dim = ') and cur is not None: cur['grid'] = l.split('=')[1].strip()
    elif l.startswith('-block dim = ') and cur is not None: cur['blk'] = l.split('=')[1].strip()
    elif l.startswith('kernel_launch_uid'):
        st = {'l1h': 0, 'l1m': 0, 'l1mh': 0, 'l2h': 0, 'l2m': 0, 'l2mh': 0, 'w': 0, 'cyc': 0}; ks.append((cur, st))
    elif st is not None:
        x = re.match(r'^\[L1_TLB_64k_STAT\] SM=\d+ hits=(\d+) misses=(\d+) mshr_hits=(\d+)', l)
        if x: st['l1h'] += int(x.group(1)); st['l1m'] += int(x.group(2)); st['l1mh'] += int(x.group(3)); continue
        x = re.match(r'^\[L2_TLB_64k_STAT\] chiplet=\d+ hits=(\d+) misses=(\d+) mshr_hits=(\d+)', l)
        if x: st['l2h'] += int(x.group(1)); st['l2m'] += int(x.group(2)); st['l2mh'] += int(x.group(3)); continue
        if l.startswith('gpgpu_n_tot_w_icount = '): st['w'] = int(l.split('=')[1])
        if l.startswith('gpu_sim_cycle = '): st['cyc'] = int(l.split('=')[1])
prev = {'l1h': 0, 'l1m': 0, 'l1mh': 0, 'l2h': 0, 'l2m': 0, 'l2mh': 0, 'w': 0}
rows = []
for i, (k, s) in enumerate(ks):
    d = {a: s[a] - prev[a] for a in prev}; prev = {a: s[a] for a in prev}
    rows.append((i + 1, k, d, s['cyc']))
TW = sum(r[2]['w'] for r in rows); TM = sum(r[2]['l2m'] for r in rows)
print(f"total warp {TW/1e6:.1f} M, walks {TM:,}, MPKI {TM/(TW/1e3):.2f}")
print(f"{'k':>3} {'kernel':34s} {'grid':>12s} {'warpM':>7s} {'L1look':>9s} {'L1hit%':>6s} {'L2hit%':>6s} {'walks':>9s} {'wk%':>5s} {'MPKI':>7s} {'w/page':>7s}")
for i, k, d, c in rows:
    look = d['l1h'] + d['l1m'] + d['l1mh']
    if d['l2m'] < MIN * TM and d['w'] < MIN * TW: continue
    l2look = d['l2h'] + d['l2m'] + d['l2mh']
    print(f"{i:>3} {k['name'][:34]:34s} {k.get('grid',''):>12s} {d['w']/1e6:7.2f} {look:9,} {100*d['l1h']/max(look,1):6.1f} {100*d['l2h']/max(l2look,1):6.1f} {d['l2m']:9,} {100*d['l2m']/TM:5.1f} {d['l2m']/max(d['w']/1e3,1e-9):7.2f} {d['w']/max(d['l2m'],1):7.0f}")
