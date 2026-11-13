#!/usr/bin/env python3
# Step 42 sweep consistency gate (docs/s42_decode_gen_sweep.ko.md section 4). Compares a later decode capture against the earlier
# points of the same sweep (same model, S, batch). Writes <new>/SWEEP_GATE.txt, last line "SWEEP GATE: PASS" or "SWEEP GATE: FAIL".
#   C-2  context at the captured step within [min_ctx, min_ctx + batch], all requests still running (no finish / preemption)
#   C-3  KV tensor: num_blocks as expected (arg), layer-0 KV base aligned to 32 MB, and num_blocks == every earlier point's (mod 1024)
#   C-4  trajectory: for each earlier point, its schedule (steps before its capture step) equals the new schedule over those steps,
#        and every request's block ids recorded at the earlier capture are a prefix of the new block table row
# Usage: s42_sweep_gate.py <new capture> <expected num_blocks> <min_ctx> <earlier capture> [<earlier capture> ...]
import json, sys
new, nb_exp, min_ctx, olds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
load = lambda d: (json.load(open(d + '/capture_info.json')), json.load(open(d + '/va_map.json')))
ci, vm = load(new); G = []
n = vm['num_computed_tokens']; B = ci['batch']
last = [s for s in ci['sched'] if s['step'] == ci['capture_step']]
running = len(last[0]['reqs']) if last else -1
ok2 = min(n) + 1 >= min_ctx and max(n) + 1 <= min_ctx + B and running == B and len(n) == B
G.append(('C-2', ok2, f'context at captured step {min(n) + 1}..{max(n) + 1} (window [{min_ctx}, {min_ctx + B}]), requests in captured step {running}/{B}'))
kv = vm['kv_cache_layer0']; nb = kv['shape'][1]; base = kv['ptr']
mods = []
for d in olds:
    _, ov = load(d); mods.append((d.rstrip('/').split('/')[-1], ov['kv_cache_layer0']['shape'][1] % 1024, ov['kv_cache_layer0']['ptr'] % (1 << 25)))
ok3 = nb == nb_exp and base % (1 << 25) == 0 and all(m == nb % 1024 and p == 0 for _, m, p in mods)
G.append(('C-3', ok3, f'num_blocks {nb} (expected {nb_exp}), mod 1024 {nb % 1024} vs earlier {[m for _, m, _ in mods]}, base mod 32 MB {base % (1 << 25)} vs earlier {[p for _, _, p in mods]}'))
norm = lambda reqs: [(int(rid.split('-')[0]), n, pos) for rid, n, pos in reqs]   # request ids carry a per-run random suffix
sched = {s['step']: norm(s['reqs']) for s in ci['sched']}
for d in olds:
    oc, ov = load(d); name = d.rstrip('/').split('/')[-1]
    steps = [s for s in oc['sched'] if s['step'] < oc['capture_step']]
    bad_s = [s['step'] for s in steps if sched.get(s['step']) != norm(s['reqs'])]
    bad_b = [r for r in range(B) if vm['block_table'][r][:len(ov['block_table'][r])] != ov['block_table'][r]]
    G.append((f'C-4 {name}', not bad_s and not bad_b and len(steps) > 0,
              f'schedule steps 1..{oc["capture_step"] - 1}: {len(steps) - len(bad_s)}/{len(steps)} identical (first diff {bad_s[:3]}); '
              f'block table prefix identical for {B - len(bad_b)}/{B} requests (diff {bad_b[:5]})'))
ok = all(p for _, p, _ in G)
with open(new + '/SWEEP_GATE.txt', 'w') as f:
    for name, p, msg in G: f.write(f'{"PASS" if p else "FAIL"} {name}: {msg}\n')
    f.write('SWEEP GATE: ' + ('PASS' if ok else 'FAIL') + '\n')
print(open(new + '/SWEEP_GATE.txt').read(), end='')
