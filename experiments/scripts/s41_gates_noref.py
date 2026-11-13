#!/usr/bin/env python3
# Step 41 gates for maps built without a decode-own ref run (scripts/s41_build_map_noref.sh). Writes <out>/GATES.txt, last line
# "GATES: PASS" or "GATES: FAIL".
#   A-1  (as s34): every split-K weight GEMM among the prefill kernels is overridden by a single-kernel ref dump.
#   A-2' : layer-0 weight pages (va_map "weights", ALL of them -- a superset of what the decode window touches, so stricter than
#          s34's A-2 for dense models; for MoE only the active experts are read) covered by the prefill prediction >= 95 %.
#   A-3  : every KV page holding a token before the captured step (prompt AND generated tokens, from the capture-time block table)
#          is in the map -- i.e. the decode-step replay covered the generated part.
# Usage: s41_gates_noref.py <out dir> <decode capture> <n prefill cand>
import json, sys, os, glob, re
out, dec, na = sys.argv[1], sys.argv[2], int(sys.argv[3]); PS = 16; G = []
splitk = []
for f in sorted(glob.glob(out + '/pred_prefill/k*.json'), key=lambda x: int(re.search(r'k(\d+)\.json', x).group(1))):
    J = json.load(open(f)); g = J.get('grid') or ''; n = J.get('name') or ''
    z = int(g.strip('()').split(',')[2]) if g.count(',') == 2 else 1
    gemm = any(t in n for t in ('gemm', 'Gemm', 'cutlass', 'Kernel2', 'xmma', 'wmma', 'fused_moe')) and 'flash' not in n.lower()
    if gemm and (z > 1 or 'split' in n.lower()): splitk.append((os.path.basename(f), n[:60], g))
ovr = {}
if os.path.exists(out + '/splitk_overrides.txt'):
    for l in open(out + '/splitk_overrides.txt'):
        t = l.split(); ovr[t[0]] = t[1] if len(t) > 1 else 'NONE'
missing = [a for a, b, c in splitk if ovr.get(a.replace('.json', ''), 'NONE') == 'NONE']
G.append(('A-1', not missing, f'split-K weight GEMMs among prefill kernels: {len(splitk)}' + (f', overridden {len(splitk) - len(missing)}/{len(splitk)}' if splitk else '')))
vm = json.load(open(dec + '/va_map.json')); ci = json.load(open(dec + '/capture_info.json'))
wpages = set()
for v in vm['weights'].values(): wpages.update(range(v['ptr'] >> PS, ((v['ptr'] + v['bytes'] - 1) >> PS) + 1))
covered = set()
for k in range(1, na + 1): covered.update(int(q) for q in json.load(open(f'{out}/cand/k{k}.json'))['pages'])
cov = len(wpages & covered) / max(1, len(wpages))
G.append(("A-2'", cov >= 0.95, f'layer-0 weight pages {len(wpages):,}, covered by the prefill prediction {len(wpages & covered):,} = {100*cov:.1f} % (threshold 95 %)'))
own = {int(l.split()[1], 16) for l in open(out + '/placement_prefill_chunked.map') if l.startswith('D ')}
kv = vm['kv_cache_layer0']; ptr, st, bs = kv['ptr'], kv['stride'], kv['shape'][2]; vb = ptr + st[0] * 2; bt = vm['block_table']
need = set(); gen_need = set()
for r, n in enumerate(vm['num_computed_tokens']):   # tokens 0..n-1 were written before the captured step
    for t in range(n):
        blk = bt[r][t // bs]; off = (blk * st[1] + (t % bs) * st[2]) * 2
        for p in ((ptr + off) >> PS, (vb + off) >> PS):
            need.add(p)
            if t >= ci['S']: gen_need.add(p)
miss = need - own; gmiss = gen_need - own
G.append(('A-3', not miss, f'KV pages before the captured step {len(need):,} (generated part {len(gen_need):,}), missing from the map {len(miss):,} (generated {len(gmiss):,})'))
ok = all(p for _, p, _ in G)
with open(out + '/GATES.txt', 'w') as f:
    for name, p, msg in G: f.write(f'{"PASS" if p else "FAIL"} {name}: {msg}\n')
    f.write('GATES: ' + ('PASS' if ok else 'FAIL') + '\n')
print(open(out + '/GATES.txt').read(), end='')
