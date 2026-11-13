#!/usr/bin/env python3
# Step 34: move first-touch candidates predicted on one capture (the b1 prefill-step run) into another capture's VA space
# (the b32 decode run) through the per-tensor weight addresses of both va_map.json files. Only pages inside a layer-0
# weight tensor are translated (activations and the prefill run's own KV are dropped and counted). A source page that
# maps onto two destination pages (tensor bases are not 64 KB aligned) contributes its candidates to both.
# Usage: s34_translate_candidates.py <src capture> <dst capture> <dst cand dir> <first k> <src k.json>...
import json, os, sys
src, dst, cdir, k0 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); files = sys.argv[5:]; PS = 16
ws = json.load(open(src + "/va_map.json"))["weights"]; wd = json.load(open(dst + "/va_map.json"))["weights"]
ten = []
for name, v in ws.items():
    if name not in wd or wd[name]["bytes"] != v["bytes"]: sys.exit(f"weight {name} missing or size differs in dst")
    ten.append((v["ptr"], v["ptr"] + v["bytes"], wd[name]["ptr"], name))
ten.sort(); os.makedirs(cdir, exist_ok=True); k = k0; tot_in = tot_out = tot_drop = 0
for f in files:
    J = json.load(open(f)); out = {}
    for ps, cands in J["pages"].items():
        p = int(ps); a, b = p << PS, (p + 1) << PS; hit = False
        for s0, s1, d0, name in ten:
            lo, hi = max(a, s0), min(b, s1)
            if lo >= hi: continue
            hit = True
            for q in range((d0 + (lo - s0)) >> PS, ((d0 + (hi - s0) - 1) >> PS) + 1):
                e = out.setdefault(q, {})
                for c, v in cands.items():
                    if c not in e or tuple(v[:3]) < tuple(e[c][:3]): e[c] = list(v)
        tot_in += 1; tot_drop += (not hit)
    tot_out += len(out)
    json.dump({"kernel": k, "file": J.get("file"), "name": J.get("name"), "grid": J.get("grid"), "block": J.get("block"),
               "model": J.get("model", {"shift": PS, "n_chip": 4}), "translated_from": f, "src_pages": len(J["pages"]), "dst_pages": len(out),
               "pages": {str(q): {str(c): v for c, v in e.items()} for q, e in out.items()}}, open(os.path.join(cdir, f"k{k}.json"), "w"))
    print(f"k{k} <- {os.path.basename(f)} {str(J.get('name'))[:50]}: src pages {len(J['pages']):,} -> weight pages in dst {len(out):,} (dropped non-weight {sum(1 for ps in J['pages'] if not any(max(int(ps)<<PS,s0)<min((int(ps)+1)<<PS,s1) for s0,s1,_,_ in ten)):,})")
    k += 1
print(f"translate: {len(files)} kernels, src pages {tot_in:,}, dropped (not weights) {tot_drop:,}, dst weight pages {tot_out:,} -> k{k0}..k{k-1}")
