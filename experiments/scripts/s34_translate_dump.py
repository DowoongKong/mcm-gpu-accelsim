#!/usr/bin/env python3
# Step 34: translate the D lines of a single-kernel reference dump taken on the prefill capture into the decode capture's VA space
# through the per-tensor weight addresses (same mapping as s34_translate_candidates.py; non-weight pages dropped). Output is an
# override map for s26_make_placement_map.py --override (lines "D 0x<vpn> <owner> 0x0").
# Usage: s34_translate_dump.py <src capture> <dst capture> <dump.map> <out.map>
import json, sys
src, dst, dump, outp = sys.argv[1:5]; PS = 16
ws = json.load(open(src + "/va_map.json"))["weights"]; wd = json.load(open(dst + "/va_map.json"))["weights"]
ten = sorted((v["ptr"], v["ptr"] + v["bytes"], wd[n]["ptr"], n) for n, v in ws.items() if n in wd and wd[n]["bytes"] == v["bytes"])
out = {}; n_in = n_drop = 0
for l in open(dump):
    if not l.startswith("D "): continue
    p, o = int(l.split()[1], 16), int(l.split()[2]); a, b = p << PS, (p + 1) << PS; hit = False; n_in += 1
    for s0, s1, d0, name in ten:
        lo, hi = max(a, s0), min(b, s1)
        if lo >= hi: continue
        hit = True
        for q in range((d0 + (lo - s0)) >> PS, ((d0 + (hi - s0) - 1) >> PS) + 1): out.setdefault(q, o)
    n_drop += (not hit)
with open(outp, "w") as f:
    f.write(f"# s34 override translated from {dump} (weights only)\n")
    for q in sorted(out): f.write(f"D 0x{q:x} {out[q]} 0x0\n")
print(f"translate dump {dump}: D {n_in:,} -> weight pages in dst {len(out):,} (dropped non-weight {n_drop:,}) -> {outp}")
