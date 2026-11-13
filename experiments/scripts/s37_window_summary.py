#!/usr/bin/env python3
# Step 37: which sharing class dominates one window (trace only). Reads the per-kernel page tables written by
# s37_sharing_class.py and prints, for the whole window and per kernel, the share of memory accesses (one access =
# one 128 B line per warp instruction) by LINE class and by PAGE class, plus pages by page class and by chiplet count.
# Usage: s37_window_summary.py <kernelslist> <class_dir> [out.json]
import collections, gzip, json, os, re, sys

CLS = ("private", "partial", "shared")


def group(label):
    return "weight" if label.startswith("w.") else "kv" if label.startswith("kv_cache") else \
        "activation" if label.startswith("m.") else "other"


def main():
    klist, cdir = sys.argv[1], sys.argv[2]
    kids = [int(re.search(r"kernel-(\d+)-", l).group(1)) for l in open(klist) if l.startswith("kernel-")]
    W = {"acc_line": [0, 0, 0], "acc_page": [0, 0, 0], "pages": [0, 0, 0], "page_chips": [0, 0, 0, 0],
         "acc_by_group_line": collections.defaultdict(lambda: [0, 0, 0]),
         "acc_by_group_page": collections.defaultdict(lambda: [0, 0, 0])}
    rows = []
    for i, k in enumerate(kids):
        js = json.load(open(os.path.join(cdir, f"k{k}.json")))
        al = [0, 0, 0]; ap = [0, 0, 0]; pg = [0, 0, 0]
        with gzip.open(os.path.join(cdir, f"k{k}.pages.tsv.gz"), "rt") as g:
            next(g)
            for row in g:
                v = row.rstrip("\n").split("\t"); a = [int(x) for x in v[9:12]]; pc = CLS.index(v[3]); s = sum(a)
                gr = group(v[1])
                for c in range(3): al[c] += a[c]; W["acc_by_group_line"][gr][c] += a[c]
                ap[pc] += s; pg[pc] += 1; W["acc_by_group_page"][gr][pc] += s
                W["page_chips"][bin(int(v[4])).count("1") - 1] += 1
        for c in range(3): W["acc_line"][c] += al[c]; W["acc_page"][c] += ap[c]; W["pages"][c] += pg[c]
        rows.append((i + 1, k, (js["kernel"] or "?")[:44], js["n_cta"], al, ap, pg))
    pct = lambda a: "/".join(f"{100 * x / max(1, sum(a)):.0f}" for x in a)
    print(f"window {klist}: {len(kids)} kernels")
    print(f"  accesses by LINE class  p/pa/s % = {pct(W['acc_line'])}   (total {sum(W['acc_line']):,})")
    print(f"  accesses by PAGE class  p/pa/s % = {pct(W['acc_page'])}")
    print(f"  pages by page class     p/pa/s % = {pct(W['pages'])}   pages touched by 1/2/3/4 chiplets = {W['page_chips']}")
    for gname in sorted(W["acc_by_group_line"]):
        print(f"  {gname:10s} line p/pa/s % = {pct(W['acc_by_group_line'][gname]):12s} page p/pa/s % = {pct(W['acc_by_group_page'][gname]):12s}"
              f" share of accesses {100 * sum(W['acc_by_group_line'][gname]) / max(1, sum(W['acc_line'])):.1f}%")
    print("  # | kernel | CTAs | acc by line class p/pa/s % | acc by page class p/pa/s %")
    for (i, k, name, nc, al, ap, pg) in rows:
        print(f"  {i:2d} k{k} {name:44s} {nc:6d}  {pct(al):12s} {pct(ap):12s}")
    if len(sys.argv) > 3:
        W["acc_by_group_line"] = dict(W["acc_by_group_line"]); W["acc_by_group_page"] = dict(W["acc_by_group_page"])
        W["kernels"] = [{"pos": i, "id": k, "name": n, "n_cta": nc, "acc_line": al, "acc_page": ap, "pages": pg}
                        for (i, k, n, nc, al, ap, pg) in rows]
        json.dump(W, open(sys.argv[3], "w"), indent=1)


main()
