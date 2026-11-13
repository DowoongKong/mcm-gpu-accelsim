#!/usr/bin/env python3
# Step 26-2 (docs/s26_init_placement_injection.ko.md): build a gpgpu-sim placement snapshot v2 from the per-kernel
# first-touch candidates of s22_ft_predict2.py (one k<i>.json per kernel of the capture's kernelslist, in launch order).
# Rule C, exactly the simulator's first-write-wins order: a page's owner is the chiplet with the smallest
# (kernel order, wave, insn_idx, local_cta_idx) key, ties -> lowest chiplet (ref mode commits deferred touches
# chiplet-ascending, gpu-module.cc commit_deferred_first_touch). PT/PD/PDP node owners (N lines) follow the same rule
# over ALL pages of the region, even when the D lines are restricted to one window (gpu-module.cc record_first_touch:
# k_pdp = pml4, k_pd = (pml4<<9)|pdp, k_pt = (pml4<<18)|(pdp<<9)|pd; first_touch_node_owner asserts the N line exists).
# Usage: s26_make_placement_map.py <candidate dir> <out.map> --windows name:first-last[,name:first-last...]
#        [--select name[,name]] [--ref placement_mcm.map] [--stats out.json] [--override dump.map[,dump.map...]]
#   --override: placement dumps of single-kernel reference simulations (V-a); for every page present in such a dump the
#   simulated owner replaces the predicted one (used for split-K kernels whose same-wave ties the predictor cannot decide).
#   Node (N) owners are recomputed after the overrides. Later dumps win over earlier ones for the same page.
#   windows: 1-based k indices into the candidate dir, in launch order; --select limits the D lines to those windows'
#   pages (default: every window). The PFN column is 0x0: premap_from_placement() re-derives PFNs from the owner.
import json, os, sys, collections

def parse(argv):
    pos, w, sel, ref, stats, ovr = [], None, None, None, None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--override': ovr += argv[i+1].split(','); i += 2
        elif a == '--windows': w = argv[i+1]; i += 2
        elif a == '--select': sel = argv[i+1].split(','); i += 2
        elif a == '--ref': ref = argv[i+1]; i += 2
        elif a == '--stats': stats = argv[i+1]; i += 2
        else: pos.append(a); i += 1
    if len(pos) != 2 or not w: sys.exit('usage: s26_make_placement_map.py <cand dir> <out.map> --windows n:a-b[,...] [--select n[,n]] [--ref map] [--stats json]')
    wins = []
    for item in w.split(','):
        name, rng = item.split(':'); a, b = (int(v) for v in rng.split('-')); wins.append((name, a, b))
    return pos[0], pos[1], wins, sel or [n for n, _, _ in wins], ref, stats, ovr

cdir, outp, wins, select, refp, statsp, overrides = parse(sys.argv[1:])
order = [(name, k) for name, a, b in wins for k in range(a, b + 1)]
best = {}            # page -> (key tuple, chip, window, k)
touched_by = collections.defaultdict(set)   # page -> windows that touch it
shift = None; n_chip = None; per_kernel = []
for pos, (wname, k) in enumerate(order):
    J = json.load(open(os.path.join(cdir, f'k{k}.json')))
    m = J.get('model', {})
    if shift is None: shift, n_chip = m.get('shift', 16), m.get('n_chip', 4)
    elif (m.get('shift', 16), m.get('n_chip', 4)) != (shift, n_chip): sys.exit(f'k{k}: shift/n_chip differ from k{order[0][1]}')
    new = 0
    for p, d in J['pages'].items():
        p = int(p); touched_by[p].add(wname)
        cands = sorted((tuple(v[:3]), int(c)) for c, v in d.items())   # min key, then lowest chiplet on ties
        key = (pos,) + cands[0][0]; chip = cands[0][1]
        tied = len(cands) > 1 and cands[1][0] == cands[0][0]
        if p not in best:
            best[p] = (key, chip, wname, k, tied); new += 1
    per_kernel.append({'k': k, 'window': wname, 'name': J.get('name'), 'grid': J.get('grid'), 'cps': J.get('cps'),
                       'pages': len(J['pages']), 'new_pages': new})
    print(f'k{k:<3} {wname:<14} pages {len(J["pages"]):>8,}  first-touched here {new:>8,}  {str(J.get("name"))[:70]}')

# overrides from single-kernel reference dumps (V-a): simulated owner replaces the predicted one, page by page
n_over = n_changed = 0
for ov in overrides:
    for l in open(ov):
        if not l.startswith('D '): continue
        p, chip = int(l.split()[1], 0), int(l.split()[2])
        if p in best:
            key, old_chip, wname, k, tied = best[p]
            best[p] = (key, chip, wname, k, tied); n_over += 1; n_changed += (chip != old_chip)
    print(f'override {ov}: {n_over:,} pages applied, {n_changed:,} owners changed')

# node owners over all pages: the region's owner is the owner of its earliest-touched page
def idx(va): return (va >> 39) & 0x1ff, (va >> 30) & 0x1ff, (va >> 21) & 0x1ff
node = {1: {}, 2: {}, 3: {}}
for p, (key, chip, wname, k, tied) in sorted(best.items(), key=lambda kv: kv[1][0]):
    va = p << shift; pml4, pdp, pd = idx(va)
    for lvl, nk in ((1, pml4), (2, (pml4 << 9) | pdp), (3, (pml4 << 18) | (pdp << 9) | pd)):
        node[lvl].setdefault(nk, chip)

emit = sorted(p for p in best if touched_by[p] & set(select))
need = {1: set(), 2: set(), 3: set()}
for p in emit:
    pml4, pdp, pd = idx(p << shift)
    need[1].add(pml4); need[2].add((pml4 << 9) | pdp); need[3].add((pml4 << 18) | (pdp << 9) | pd)
with open(outp, 'w') as f:
    f.write('# gpgpu-sim placement snapshot v2\n')
    f.write(f'# page_shift {shift}\n')
    f.write('# D <vpn> <owner_chiplet> <data_pfn>   (data page, vpn = va >> page_shift; pfn 0x0 = re-derived at premap)\n')
    f.write('# N <level> <region_key> <owner_chiplet>   (PT node, level 1=PDP 2=PD 3=PT)\n')
    f.write(f'# source: s26_make_placement_map.py rule C over {cdir} windows {",".join(f"{n}:{a}-{b}" for n, a, b in wins)} '
            f'select {",".join(select)} n_chiplets {n_chip}\n')
    for lvl in (1, 2, 3):
        for nk in sorted(need[lvl]): f.write(f'N {lvl} 0x{nk:x} {node[lvl][nk]}\n')
    for p in emit: f.write(f'D 0x{p:x} {best[p][1]} 0x0\n')

share = collections.Counter(best[p][1] for p in emit)
by_first = collections.Counter(best[p][2] for p in emit)
tied_n = sum(1 for p in emit if best[p][4])
print(f'\nmap {outp}: D {len(emit):,} (of {len(best):,} predicted pages)  N1 {len(need[1])} N2 {len(need[2])} N3 {len(need[3])}  shift {shift}')
print('owner share: ' + '  '.join(f'chip{c} {share[c]:,} ({100*share[c]/max(1,len(emit)):.1f} %)' for c in range(n_chip)))
print('first-touch window of the emitted pages: ' + '  '.join(f'{w} {n:,}' for w, n in by_first.most_common()))
print(f'tied pages (same key on >1 chiplet, lowest chiplet chosen): {tied_n:,} ({100*tied_n/max(1,len(emit)):.1f} %)')
stats = {'D': len(emit), 'predicted_pages': len(best), 'N': {l: len(need[l]) for l in (1, 2, 3)}, 'shift': shift, 'n_chip': n_chip,
         'overrides': overrides, 'override_pages': n_over, 'override_changed': n_changed,
         'owner_share': dict(share), 'first_touch_window': dict(by_first), 'tied': tied_n, 'per_kernel': per_kernel}
if refp:
    ref = {int(l.split()[1], 0): int(l.split()[2]) for l in open(refp) if l.startswith('D ')}
    common = [p for p in emit if p in ref]; ok = sum(best[p][1] == ref[p] for p in common)
    print(f'vs ref {refp}: ref D {len(ref):,}  common {len(common):,}  agreement {ok:,}/{len(common):,} = {100*ok/max(1,len(common)):.1f} %  '
          f'ref-only {len(set(ref)-set(emit)):,}  predicted-only {len(set(emit)-set(ref)):,}')
    stats['ref'] = {'file': refp, 'ref_D': len(ref), 'common': len(common), 'agree': ok,
                    'ref_only': len(set(ref) - set(emit)), 'pred_only': len(set(emit) - set(ref))}
if statsp: json.dump(stats, open(statsp, 'w'), indent=1)
