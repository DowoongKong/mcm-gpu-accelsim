#!/usr/bin/env python3
# Step 34: a decode-own candidate from a ref placement dump (v2/v3): every D page with its simulated owner, key (0,0,0).
# Placed last in the window order so it only fills pages that prefill (weights) and the KV replay did not touch, and
# it defines the decode window's page set for --select. Usage: s34_refmap_candidate.py <placement map> <out k.json> <k>
import json, sys
m, out, k = sys.argv[1], sys.argv[2], int(sys.argv[3]); pages = {}
for l in open(m):
    if l.startswith("D "): _, vpn, o = l.split()[:3]; pages[str(int(vpn, 16))] = {o: [0, 0, 0, 1]}
json.dump({"kernel": k, "file": m, "name": "decode_own_refmap", "grid": None, "block": None, "model": {"shift": 16, "n_chip": 4, "source": "ref dump"}, "pages": pages}, open(out, "w"))
print(f"refmap candidate k{k}: {len(pages):,} decode pages from {m}")
