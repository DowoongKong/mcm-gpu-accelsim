# kv_owner_check.py <capture dir>  -- compare KV-page owners in the decode ref map (first touch by the decode window itself)
# with what a chunked-prefill replay (vLLM steps as recorded in capture_info 'sched', 1 CTA per token, KERNEL_PARTITION
# = 4 contiguous CTA quarters) and a hypothetical single 131k-token prefill step would assign. Read-only.
import json, sys, collections
d = sys.argv[1]; PS = 16
ci = json.load(open(d + "/capture_info.json")); vm = json.load(open(d + "/va_map.json"))
own = {}
for line in open(d + "/sim/ref_mcm/placement_mcm.map"):
    if line.startswith("D "):
        _, vpn, o, pfn = line.split()[:4]; own[int(vpn, 16)] = int(o)
kv = vm["kv_cache_layer0"]; ptr = kv["ptr"]; shape = kv["shape"]; stride = kv["stride"]; esz = 2
nblk, bs = shape[1], shape[2]; vbase = ptr + stride[0] * esz
bt = vm["block_table"]; B = ci["batch"]; S = ci["S"]
def pages_of(r, t):            # 64 KB pages holding token t of request r (K and V)
    blk = bt[r][t // bs]; off = (blk * stride[1] + (t % bs) * stride[2]) * esz
    return ((ptr + off) >> PS, (vbase + off) >> PS)
# chunked replay: rows of each step in scheduled order (cached requests first, then the new chunk), chiplet = row*4//rows
first_chunk, first_single = {}, {}
for st in ci["sched"]:
    rows = []
    for rid, n, pos in st["reqs"]:
        r = int(rid.split("-")[0]); rows += [(r, pos + i) for i in range(n)]
    R = len(rows)
    for i, (r, t) in enumerate(rows):
        if t >= S: continue                      # prompt tokens only (prefill-written pages)
        c = i * 4 // R
        for p in pages_of(r, t): first_chunk.setdefault(p, c)
for r in range(B):
    for t in range(S):
        c = (r * S + t) * 4 // (B * S)
        for p in pages_of(r, t): first_single.setdefault(p, c)
kvpages = set(first_chunk)
inmap = [p for p in kvpages if p in own]
print(f"{d}: KV prompt pages {len(kvpages):,}; present in decode ref map {len(inmap):,}")
def dist(f): 
    c = collections.Counter(f[p] for p in inmap); return " ".join(f"c{k}:{100*c[k]/len(inmap):.0f}%" for k in sorted(c))
print("  owner distribution  ref map:", dist(own), "| chunked replay:", dist(first_chunk), "| single 131k step:", dist(first_single))
agree_rc = sum(own[p] == first_chunk[p] for p in inmap); agree_rs = sum(own[p] == first_single[p] for p in inmap); agree_cs = sum(first_chunk[p] == first_single[p] for p in inmap)
print(f"  owner agreement: ref==chunked {100*agree_rc/len(inmap):.1f}%  ref==single {100*agree_rs/len(inmap):.1f}%  chunked==single {100*agree_cs/len(inmap):.1f}%")
# per-request striping: how many chiplets hold a request's KV pages under each assignment
def stripes(f):
    out = []
    for r in range(B):
        ps = {q for t in range(S) for q in pages_of(r, t) if q in own}
        out.append(len({f[p] for p in ps}))
    return collections.Counter(out)
print("  chiplets per request's KV (count of requests): ref map", dict(stripes(own)), "| chunked", dict(stripes(first_chunk)), "| single", dict(stripes(first_single)))
