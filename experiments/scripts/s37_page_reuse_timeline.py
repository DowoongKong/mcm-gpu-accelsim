#!/usr/bin/env python3
# Step 37 add-on: per-page reuse timeline of ONE kernel trace (no simulation). For every (page, CTA) pair it records how
# many warp memory instructions touched the page and WHERE in the warps' instruction streams those touches fall
# (position = instruction index / warp length, 0..1). A page that is "cold-missed, reused, then dead" shows a short span
# (all touches inside one loop iteration) followed by no touches until the warp ends.
# Usage: s37_page_reuse_timeline.py <trace.xz> [--page-shift 16]
import collections, lzma, math, sys

MEM_OPS = {"LDG", "STG", "LDGSTS", "LD", "ST", "ATOM", "ATOMG", "RED"}


def addrs(t, k):
    if int(t[k]) == 0: return []
    fmt = int(t[k + 1]); base = int(t[k + 2], 16); act = bin(int(t[1], 16)).count("1")
    if fmt == 1:
        s = int(t[k + 3]); return [base + i * s for i in range(act)]
    if fmt == 2:
        out = [base]; a = base
        for d in t[k + 3:k + 3 + act - 1]: a += int(d); out.append(a)
        return out
    return [int(x, 16) for x in t[k + 2:k + 2 + act]]


def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def main():
    f = sys.argv[1]; SHIFT = int(sys.argv[sys.argv.index("--page-shift") + 1]) if "--page-shift" in sys.argv else 16
    h = {}; gx = gy = 1; cta = -1; wlen = 1; idx = 0
    pc = collections.defaultdict(lambda: [2.0, -1.0, 0])     # (page, cta) -> [first pos, last pos, warp-level touches]
    page_ctas = collections.defaultdict(set)
    with lzma.open(f, "rt") as fh:
        for ln in fh:
            if ln.startswith("-"):
                k, v = ln.lstrip("-").split("=", 1); h[k.strip()] = v.strip()
                if k.strip() == "grid dim": gx, gy, _ = (int(x) for x in v.strip("() \n").split(","))
                continue
            if ln.startswith("thread block ="):
                x, y, z = (int(v) for v in ln.split("=")[1].split(",")); cta = x + y * gx + z * gx * gy; continue
            if ln.startswith("warp ="): idx = 0; continue
            if ln.startswith("insts ="): wlen = max(1, int(ln.split("=")[1])); continue
            if cta < 0 or ln.startswith("#"): continue
            t = ln.split()
            if len(t) < 5 or not t[2].isdigit(): continue
            idx += 1
            dn = int(t[2]); oi = 3 + dn; op = t[oi].split(".")[0]
            if op not in MEM_OPS: continue
            kk = oi + 2 + int(t[oi + 1])
            try: A = addrs(t, kk)
            except Exception: continue
            pos = idx / wlen
            for P in {a >> SHIFT for a in A}:
                e = pc[(P, cta)]
                if pos < e[0]: e[0] = pos
                if pos > e[1]: e[1] = pos
                e[2] += 1; page_ctas[P].add(cta)
    spans = [e[1] - e[0] for e in pc.values()]; touches = [e[2] for e in pc.values()]; lasts = [e[1] for e in pc.values()]
    nct = [len(s) for s in page_ctas.values()]
    print(f"{(h.get('kernel name') or '?')[:70]}  grid {h.get('grid dim')}")
    print(f"pages {len(page_ctas):,}, (page, CTA) pairs {len(pc):,}; CTAs per page p50 {pct(nct, .5)} max {max(nct)}")
    print(f"warp-instruction touches per (page, CTA): p10 {pct(touches, .1)} p50 {pct(touches, .5)} p90 {pct(touches, .9)}")
    print(f"span of those touches in the warp stream (last - first, fraction of warp length): "
          f"p50 {pct(spans, .5):.3f} p90 {pct(spans, .9):.3f} p99 {pct(spans, .99):.3f}")
    print(f"position of the last touch: p10 {pct(lasts, .1):.2f} p50 {pct(lasts, .5):.2f} p90 {pct(lasts, .9):.2f}"
          f"  (after it the page is dead for that CTA)")
    for th in (0.05, 0.10, 0.25):
        print(f"  pairs with span <= {th:.2f}: {100 * sum(s <= th for s in spans) / len(spans):5.1f}%")


main()
