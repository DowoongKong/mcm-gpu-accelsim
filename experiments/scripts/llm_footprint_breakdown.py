# Per-array first-touch footprint of an LLM L1seq ref_batch4 run: classify every MemcpyHtoD (weight tensor) by its byte
# size using the model dims, count the pages of placement_batch4.map that fall inside each copy, and group the touched
# pages outside every HtoD range (activations / KV cache / workspace) into contiguous runs.
# Usage: llm_footprint_breakdown.py <exp L1seq dir> <deepseek|qwen>
import os, sys, collections
X, model = sys.argv[1], sys.argv[2]
DIMS = {"deepseek": dict(h=4096, kv=1024, inter=14336, vocab=128256, L=32, bias=False),
        "qwen":     dict(h=3584, kv=512,  inter=18944, vocab=152064, L=28, bias=True)}[model]
h, kv, inter, vocab = DIMS["h"], DIMS["kv"], DIMS["inter"], DIMS["vocab"]
NAMES = {vocab*h*2: "embed_tokens / lm_head (vocab x h)", h*h*2: "q_proj / o_proj (h x h)", kv*h*2: "k_proj / v_proj (kv x h)",
         inter*h*2: "gate / up / down (inter x h)", h*2: "norm weight (h)", kv*2: "k/v bias (kv)" , 4: "scalar 4 B", 16: "scalar 16 B", 32: "scalar 32 B"}
PAGE = 4096
copies = []
for ln in open(os.path.join(X, "ref_batch4", "run.log"), errors="ignore"):
    if ln.startswith("launching memcpy command : MemcpyHtoD"):
        p = ln.strip().split(","); copies.append((int(p[1], 16), int(p[2])))
pages = sorted(int(l.split()[1], 16) for l in open(os.path.join(X, "ref_batch4", "placement_batch4.map")) if l.startswith("D "))
# page -> copy index (copies do not overlap); binary search over copies sorted by address
import bisect
cs = sorted(range(len(copies)), key=lambda i: copies[i][0]); starts = [copies[i][0] for i in cs]
touched = collections.Counter(); outside = []
for vpn in pages:
    va = vpn * PAGE; k = bisect.bisect_right(starts, va) - 1
    if k >= 0:
        i = cs[k]; a, b = copies[i]
        if va < a + b: touched[i] += 1; continue
    outside.append(vpn)
# report per size class: number of copies, copies touched, pages touched, bytes theoretical per copy
by_size = collections.defaultdict(lambda: [0, 0, 0])
for i, (a, b) in enumerate(copies):
    r = by_size[b]; r[0] += 1
    if touched[i]: r[1] += 1; r[2] += touched[i]
print(f"model={model} copies={len(copies)} HtoD_total={sum(b for a,b in copies):,} B  map_pages={len(pages):,} ({len(pages)*PAGE/1e6:,.1f} MB)")
print(f"{'size B':>14} {'name':42} {'#copies':>7} {'touched':>7} {'pages':>8} {'MB':>9} {'pages/copy':>10} {'full pages/copy':>15}")
tot_in = 0
for b, (n, nt, pg) in sorted(by_size.items(), key=lambda kv: -kv[0]):
    if b <= 1024 and pg == 0: continue
    tot_in += pg
    print(f"{b:>14,} {NAMES.get(b, '?'):42} {n:>7} {nt:>7} {pg:>8,} {pg*PAGE/1e6:>9.2f} {pg/max(nt,1):>10.1f} {-(-b//PAGE):>15,}")
print(f"HtoD arrays touched: {tot_in:,} pages = {tot_in*PAGE/1e6:,.1f} MB; outside HtoD: {len(outside):,} pages = {len(outside)*PAGE/1e6:,.1f} MB")
# which copies (by order) of each big class were touched -> layer identification
big = [i for i,(a,b) in enumerate(copies) if b >= h*h*2 and touched[i]]
print("touched large copies (order index among all copies):", big[:40])
runs = []; 
if outside:
    s = p = outside[0]
    for v in outside[1:]:
        if v != p + 1: runs.append((s, p)); s = v
        p = v
    runs.append((s, p))
print(f"outside-HtoD contiguous runs: {len(runs)}; largest:")
for s, e in sorted(runs, key=lambda r: -(r[1]-r[0]))[:10]:
    print(f"  0x{s*PAGE:x} {e-s+1:>8,} pages = {(e-s+1)*PAGE/1e6:9.2f} MB")
