# Per FlashAttention CTA of the captured decode step: how many chiplets own the KV pages (and their 2 MB leaf PT nodes) that the CTA reads.
# CTA -> chiplet = KERNEL_PARTITION (linear id // ceil(nCTA/4)); FA2 split-KV grid = (1, splits or batch, batch*kv_heads or kv_heads).
import json, sys, collections, math
cap, mp, splits = sys.argv[1], sys.argv[2], int(sys.argv[3])
ci = json.load(open(cap + '/capture_info.json')); vm = json.load(open(cap + '/va_map.json'))
kv = vm['kv_cache_layer0']; ptr, st, bs = kv['ptr'], kv['stride'], kv['shape'][2]; H = kv['shape'][3]; vb = ptr + st[0] * 2; S = ci['S']
D, N3 = {}, {}
for l in open(mp):
    t = l.split()
    if t and t[0] == 'D': D[int(t[1], 16)] = int(t[2])
    elif t and t[0] == 'N' and t[1] == '3': N3[int(t[2], 16)] = int(t[3])
B = len(vm['num_computed_tokens']); n_cta = B * H * splits; chunk = math.ceil(n_cta / 4)
res = collections.Counter(); own_cnt = collections.Counter(); leaf_cnt = collections.Counter(); part = collections.Counter()
for r, ncomp in enumerate(vm['num_computed_tokens']):
    L = ncomp + 1                                   # the step attends to every cached token plus its own new token
    nblk = math.ceil(L / 128); per = math.ceil(nblk / splits)
    for h in range(H):
        for s in range(splits):
            lin = (s + splits * (r * H + h)) if splits > 1 else (r + B * h)
            chip = lin // chunk
            t0, t1 = s * per * 128, min(L, (s + 1) * per * 128)
            if t0 >= t1: continue
            pages = set()
            for t in range(t0, t1):
                blk = vm['block_table'][r][t // bs]; off = (blk * st[1] + (t % bs) * st[2]) * 2
                pages.add(((ptr + off) >> 16, t >= S)); pages.add(((vb + off) >> 16, t >= S))
            owners = {D[p] for p, g in pages}; leaves = {N3.get(p >> 5) for p, g in pages}
            own_cnt[len(owners)] += 1; leaf_cnt[len(leaves)] += 1; res['cta'] += 1
            for p, g in pages:
                res['pages'] += 1; res['remote_data'] += D[p] != chip; res['remote_leaf'] += N3.get(p >> 5) != chip
                part[('gen' if g else 'prompt', 'n')] += 1; part[('gen' if g else 'prompt', 'remote')] += D[p] != chip
c = res['cta']; pp = res['pages']
fmt = lambda cnt: ' / '.join(f"{100*cnt[k]/c:.0f}%" for k in (1, 2, 3, 4))
print(f"  CTAs {c}, KV pages per CTA {pp/c:,.0f} | chiplets owning one CTA's KV pages (1/2/3/4): {fmt(own_cnt)} | leaf PT owners (1/2/3/4): {fmt(leaf_cnt)}")
print(f"  pages remote to the CTA: data {100*res['remote_data']/pp:.1f} %, leaf PT {100*res['remote_leaf']/pp:.1f} % | prompt part remote {100*part[('prompt','remote')]/max(1,part[('prompt','n')]):.1f} % ({part[('prompt','n')]/c:,.0f}/CTA), generated part remote {100*part[('gen','remote')]/max(1,part[('gen','n')]):.1f} % ({part[('gen','n')]/c:,.0f}/CTA)")
