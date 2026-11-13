#!/usr/bin/env python3
# Step 45 section 14: the weight tile of a GEMM kernel from its trace. For every CTA: distinct weight rows (output features), the K
# (column) range, and distinct 64 KB weight pages; for one warp, the column span covered per weight-load instruction.
# Usage: s45_gemm_tile_probe.py <kernel .traceg.xz> <va_map.json> <weight name substring, e.g. down_proj> <K>
import lzma, math, sys, json, collections
f, vmp, wname, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
vm = json.load(open(vmp)); W = [(n, v['ptr'], v['bytes']) for n, v in vm['weights'].items() if wname in n][0]
base, nbytes = W[1], W[2]
MEM = ("LDG", "LDGSTS", "LD")
def addrs(t, k):
    if int(t[k]) == 0: return []
    fmt = int(t[k + 1]); b = int(t[k + 2], 16); act = bin(int(t[1], 16)).count("1")
    if fmt == 1: s = int(t[k + 3]); return [b + i * s for i in range(act)]
    if fmt == 2:
        out = [b]; a = b
        for d in t[k + 3:k + 3 + act - 1]: a += int(d); out.append(a)
        return out
    return [int(x, 16) for x in t[k + 2:k + 2 + act]]
h = {}; gx = gy = 1; n_total = chunk = 0; lin = -1; warp = -1
rows = collections.defaultdict(set); cols = collections.defaultdict(lambda: [1 << 62, -1]); pages = collections.defaultdict(set); first_warp = collections.defaultdict(list)
with lzma.open(f, "rt") as fh:
    for ln in fh:
        if ln.startswith("-"): k, v = ln.lstrip("-").split("=", 1); h[k.strip()] = v.strip(); continue
        if ln.startswith("#"):
            if "grid dim" in h and not n_total: gx, gy, gz = (int(x) for x in h["grid dim"].strip("() ").split(",")); n_total = gx * gy * gz; chunk = math.ceil(n_total / 4)
            continue
        if ln.startswith("thread block ="): x, y, z = (int(v) for v in ln.split("=")[1].split(",")); lin = x + y * gx + z * gx * gy; continue
        if ln.startswith("warp ="): warp = int(ln.split("=")[1]); continue
        t = ln.split()
        if lin < 0 or len(t) < 5: continue
        try: oi = 3 + int(t[2])
        except ValueError: continue
        if t[oi].split(".")[0] not in MEM: continue
        try: A = [a for a in addrs(t, oi + 2 + int(t[oi + 1])) if base <= a < base + nbytes]
        except Exception: continue
        if not A: continue
        for a in A:
            r, c = divmod((a - base) // 2, K); rows[lin].add(r); cc = cols[lin]; cc[0] = min(cc[0], c); cc[1] = max(cc[1], c); pages[lin].add(a >> 16)
        if warp == 0 and len(first_warp[lin]) < 12:
            cs = sorted({((a - base) // 2) % K for a in A}); rs = {((a - base) // 2) // K for a in A}
            first_warp[lin].append((len(rs), cs[0], cs[-1], len({a >> 16 for a in A})))
print(f"{f.split('/')[-1]}: grid {h.get('grid dim')}, weight {W[0]} ({nbytes/2**20:.0f} MiB, K = {K}, rows {nbytes//2//K})")
R = [len(rows[c]) for c in rows]; P = [len(pages[c]) for c in pages]
print(f"CTAs touching the weight {len(rows)}: rows per CTA min/mean/max {min(R)}/{sum(R)/len(R):.0f}/{max(R)}, pages per CTA mean {sum(P)/len(P):.0f}")
for c in sorted(rows)[:3] + [sorted(rows)[len(rows) // 2]]:
    print(f"  CTA {c} (chiplet {c // chunk}): rows {min(rows[c])}-{max(rows[c])} ({len(rows[c])}), K columns {cols[c][0]}-{cols[c][1]}, pages {len(pages[c])}; warp 0 first loads (rows, col from, col to, pages): {first_warp[c][:6]}")
per_chip = collections.defaultdict(set)
for c in pages: per_chip[min(3, c // chunk)] |= pages[c]
print("distinct weight pages read per chiplet over the kernel: " + ", ".join(f"c{k} {len(v):,}" for k, v in sorted(per_chip.items())))
