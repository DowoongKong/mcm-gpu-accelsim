#!/usr/bin/env python3
# Step 37 (docs/s37_access_class_remote_pt.ko.md): classify the memory accesses of ONE accel-sim kernel trace by how many
# thread blocks (CTAs) share each 128 B line and each 64 KB page: private (1 CTA), shared (>= SHARED_FRAC of all CTAs),
# partial (in between). Also records which chiplets touch each page and each 2 MB leaf page-table region
# (chiplet = KERNEL_PARTITION split, the simulator's CTA scheduler; same formula as s36). No simulation.
# Output: <out_prefix>.pages.tsv.gz (one row per page) and <out_prefix>.json (per-operand summary).
# Usage: s37_sharing_class.py <trace.xz> <va_map.json> <out_prefix> [--page-shift 16] [--shared-frac 0.9]
import gzip, json, lzma, math, sys, time, bisect, collections

MEM_OPS = {"LDG", "STG", "LDGSTS", "LD", "ST", "ATOM", "ATOMG", "RED"}
N_CHIP = 4
CLS = ("private", "partial", "shared")
LEAF_SHIFT = 21          # one leaf page-table node covers 2 MB (vpn >> 5 for 64 KB pages)


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


def hdr_int3(v): return tuple(int(x) for x in v.strip("() ").split(","))


def load_ranges(va_map_path):
    """Sorted (start, end, label) from va_map.json; exact duplicate ranges keep the first label (as s36)."""
    m = json.load(open(va_map_path)); R = []
    def add(label, e):
        if isinstance(e, dict) and "ptr" in e and "bytes" in e:
            R.append((int(e["ptr"]), int(e["ptr"]) + int(e["bytes"]), label))
    for n, e in (m.get("weights") or {}).items(): add("w." + n, e)
    for n, e in (m.get("modules") or {}).items():
        for i, x in enumerate(e.get("inputs") or []): add(f"m.{n}.in{i}", x)
        for i, x in enumerate(e.get("outputs") or []): add(f"m.{n}.out{i}", x)
    kv = m.get("kv_cache_layer0")
    if isinstance(kv, dict) and "ptr" in kv: add("kv_cache_layer0", kv)
    seen = set(); out = []
    for r in sorted(R):
        if (r[0], r[1]) in seen: continue
        seen.add((r[0], r[1])); out.append(r)
    return out


def main():
    f, vmp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    SHIFT = 16; SHARED_FRAC = 0.9
    for i, a in enumerate(sys.argv):
        if a == "--page-shift": SHIFT = int(sys.argv[i + 1])
        if a == "--shared-frac": SHARED_FRAC = float(sys.argv[i + 1])
    R = load_ranges(vmp); starts = [r[0] for r in R]
    def label_of(a):
        i = bisect.bisect_right(starts, a) - 1
        while i >= 0:
            if R[i][0] <= a < R[i][1]: return R[i][2]
            i -= 1
        return "other"
    t0 = time.time(); h = {}; gx = gy = 1; chunk = 1; n_total = 0
    line = {}      # line -> [n_ctas, last_cta, accesses]
    page = {}      # page -> [n_ctas, last_cta, chip_mask, label]
    lin = chip = -1; n_cta = 0; n_local = 0
    with lzma.open(f, "rt") as fh:
        for ln in fh:
            if ln.startswith("-"):
                k, v = ln.lstrip("-").split("=", 1); h[k.strip()] = v.strip(); continue
            if ln.startswith("#"):
                if "grid dim" not in h or n_total: continue
                gx, gy, gz = hdr_int3(h["grid dim"])
                n_total = gx * gy * gz; chunk = math.ceil(n_total / N_CHIP); continue
            if ln.startswith("thread block ="):
                x, y, z = (int(v) for v in ln.split("=")[1].split(","))
                lin = x + y * gx + z * gx * gy; chip = min(N_CHIP - 1, lin // chunk); n_cta += 1
                continue
            if lin < 0 or ln.startswith("warp ="): continue
            t = ln.split()
            if len(t) < 5: continue
            try:
                dn = int(t[2]); oi = 3 + dn
            except ValueError:
                continue
            op = t[oi].split(".")[0]
            if op in ("LDL", "STL"): n_local += 1   # local memory: the simulator remaps these to its own region
            if op not in MEM_OPS: continue
            sn = int(t[oi + 1]); kk = oi + 2 + sn
            try:
                A = addrs(t, kk)
            except Exception:
                continue
            if not A: continue
            lab = None; cbit = 1 << chip
            for L in {a >> 7 for a in A}:            # one access per line per instruction (coalesced)
                e = line.get(L)
                if e is None: line[L] = [1, lin, 1]
                else:
                    e[2] += 1
                    if e[1] != lin: e[0] += 1; e[1] = lin
                P = L >> (SHIFT - 7)
                p = page.get(P)
                if p is None:
                    if lab is None: lab = label_of(A[0])
                    page[P] = [1, lin, cbit, lab]
                else:
                    p[2] |= cbit
                    if p[1] != lin: p[0] += 1; p[1] = lin
    thr = max(2, math.ceil(SHARED_FRAC * n_cta))
    def cls(n): return 0 if n == 1 else (2 if n >= thr else 1)
    # per-page line-class counts
    pl = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])   # page -> lines p/pa/s, accesses p/pa/s
    hist = collections.Counter()
    for L, (n, _, acc) in line.items():
        c = cls(n); r = pl[L >> (SHIFT - 7)]; r[c] += 1; r[3 + c] += acc
        hist[min(n, 1 << 20).bit_length()] += 1
    leaf = collections.defaultdict(int)
    for P, (_, _, m, _) in page.items(): leaf[P >> (LEAF_SHIFT - SHIFT)] |= m
    summ = collections.defaultdict(lambda: {"lines": [0, 0, 0], "acc": [0, 0, 0], "pages": [0, 0, 0], "page_chips": [0, 0, 0, 0]})
    with gzip.open(outp + ".pages.tsv.gz", "wt") as g:
        g.write("vpn\tlabel\tn_ctas\tpage_class\tchip_mask\tleaf_chip_mask\tl_priv\tl_part\tl_shared\ta_priv\ta_part\ta_shared\n")
        for P in sorted(page):
            n, _, m, lab = page[P]; r = pl[P]; pc = cls(n); lm = leaf[P >> (LEAF_SHIFT - SHIFT)]
            g.write(f"{P:x}\t{lab}\t{n}\t{CLS[pc]}\t{m}\t{lm}\t" + "\t".join(str(v) for v in r) + "\n")
            s = summ[lab]
            for c in range(3): s["lines"][c] += r[c]; s["acc"][c] += r[3 + c]
            s["pages"][pc] += 1; s["page_chips"][bin(m).count("1") - 1] += 1
    tot = {"lines": [0, 0, 0], "acc": [0, 0, 0], "pages": [0, 0, 0], "page_chips": [0, 0, 0, 0]}
    for s in summ.values():
        for k in tot:
            for c in range(len(tot[k])): tot[k][c] += s[k][c]
    out = {"trace": f, "kernel": h.get("kernel name"), "grid": h.get("grid dim"), "n_cta": n_cta, "chunk": chunk,
           "shared_threshold_ctas": thr, "trivial": n_cta < 4, "classes": CLS, "local_mem_warp_insts": n_local,
           "total": tot, "by_operand": summ, "ctas_per_line_log2_hist": dict(sorted(hist.items())),
           "unique_lines": len(line), "unique_pages": len(page), "leaf_nodes": len(leaf), "elapsed_s": round(time.time() - t0, 1)}
    json.dump(out, open(outp + ".json", "w"), indent=1)
    a = tot["acc"]; sa = sum(a) or 1
    print(f"{(h.get('kernel name') or '?')[:60]} ctas {n_cta} pages {len(page):,} acc p/pa/s "
          f"{100*a[0]/sa:.1f}/{100*a[1]/sa:.1f}/{100*a[2]/sa:.1f}% {out['elapsed_s']} s")


main()
