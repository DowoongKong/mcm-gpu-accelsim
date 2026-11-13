#!/usr/bin/env python3
# Step 34 accuracy gates (docs/s34 section 3), run by s34_build_prefill_map.sh after the map is built. Writes <out>/GATES.txt whose
# last line is "GATES: PASS" or "GATES: FAIL". A-1: no split-K GEMM among the prefill kernels whose prediction places the weights
# (the predictor is validated at 100 % only for non-split-K kernels, s26 V-a). A-2: decode-window weight pages covered by the
# prefill step-1 prediction >= 95 % (the rest fall back to decode-own owners). Usage: s34_gates.py <out dir> <decode capture> <n prefill cand>
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
G.append(('A-1', not missing, f'split-K weight GEMMs among prefill kernels: {len(splitk)} ' + '; '.join(f'{a} {b} {c}' for a, b, c in splitk) + (f' | overridden by single-kernel ref dumps: {len(splitk) - len(missing)}/{len(splitk)}' if splitk else '')))
wd = json.load(open(dec + '/va_map.json'))['weights']; wpages = set()
for v in wd.values(): wpages.update(range(v['ptr'] >> PS, ((v['ptr'] + v['bytes'] - 1) >> PS) + 1))
simd = dec + '/sim_noprefill' if os.path.isdir(dec + '/sim_noprefill') else dec + '/sim'
decp = {int(l.split()[1], 16) for l in open(simd + '/ref_mcm/placement_mcm.map') if l.startswith('D ')}
covered = set()
for k in range(1, na + 1): covered.update(int(q) for q in json.load(open(f'{out}/cand/k{k}.json'))['pages'])
dw = decp & wpages; cov = len(dw & covered) / max(1, len(dw))
G.append(('A-2', cov >= 0.95, f'decode-window weight pages {len(dw):,}, covered by the prefill step-1 prediction {len(dw & covered):,} = {100*cov:.1f} % (threshold 95 %)'))
ok = all(p for _, p, _ in G)
with open(out + '/GATES.txt', 'w') as f:
    for name, p, msg in G: f.write(f'{"PASS" if p else "FAIL"} {name}: {msg}\n')
    f.write('GATES: ' + ('PASS' if ok else 'FAIL') + '\n')
print(open(out + '/GATES.txt').read(), end='')
