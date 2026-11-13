# Step 11: per-piece address-translation metrics for a multi-layer sequence run (L1seq, L2seq, ...).
# Kernel positions come from regions.tsv + the sequence kernelslist itself, so the same code serves any --layers N list:
# for every step, <tag>-O = prep + tail kernels, <tag>-L<k> = the kernels of layer k (k = 0-based layer index).
# Metric definitions and the per-kernel cumulative-dump parsing are those of s6_compare_layers.py (as in s8_extract_l1seq.py).
# Usage: s11_extract_lseq.py <regions.tsv> <kernelslist_LNseq.g> <run.log> [<run.log> ...]
import os, re, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "s6_compare_layers.py")).read().split("\ndata = {}")[0]
ns = {}; exec(compile(src.replace("E = sys.argv[1]", "E = ''"), "s6_compare_layers.py", "exec"), ns)
blocks, dump, derived, minus = ns["blocks"], ns["dump"], ns["derived"], ns["minus"]
TAGS = {"prefill": "P", "decode1": "D1", "decode2": "D2", "decode3": "D3"}


def pieces(regions, seqlist):
    rng, steps = {}, []
    for ln in open(regions).read().splitlines()[1:]:
        step, piece, a, b = ln.split("\t")[:4]
        rng[(step, piece)] = range(int(a), int(b) + 1)
        if step not in steps: steps.append(step)
    kids = [int(re.match(r"kernel-(\d+)-ctx", l).group(1)) for l in open(seqlist).read().splitlines() if l.startswith("kernel-")]
    pos = {k: i + 1 for i, k in enumerate(kids)}                    # 1-based index in the sequence list
    out = {}
    for step in steps:
        t = TAGS.get(step, step)
        out[f"{t}-O"] = [pos[k] for p in ("prep", "tail") for k in rng[(step, p)] if k in pos]
        for piece in sorted(p for (s, p) in rng if s == step and p.startswith("layer")):
            ks = [pos[k] for k in rng[(step, piece)] if k in pos]
            if ks: out[f"{t}-L{piece[5:]}"] = ks
    return out, len(kids)


def per_kernel(path):
    ds = []
    for b in blocks(open(path, errors="replace").read()):
        try: ds.append(dump(b))
        except Exception: break                                    # last block may be incomplete
    return [ds[0]] + [minus(ds[i], ds[i - 1]) for i in range(1, len(ds))]


def total(pk, idx):
    t = None
    for i in idx:
        if i - 1 >= len(pk): return None
        t = dict(pk[i - 1]) if t is None else {k: t[k] + pk[i - 1][k] for k in t}
    return t


if __name__ == "__main__":
    regions, seqlist, logs = sys.argv[1], sys.argv[2], sys.argv[3:]
    P, nk = pieces(regions, seqlist)
    print(f"{'run':12s} {'piece':6s} {'kern':>4s} {'warp':>11s} {'cycles':>11s} {'L2miss':>10s} {'MPKI':>7s} {'IPC':>6s} {'AT':>7s} {'PTW':>7s} {'rr':>6s}")
    for log in logs:
        pk = per_kernel(log)
        name = log.split("/")[-3] + "/" + log.split("/")[-2]
        for p in list(P) + ["ALL"]:
            idx = P[p] if p != "ALL" else list(range(1, nk + 1))
            t = total(pk, idx)
            if t is None: print(f"{name[:12]:12s} {p:6s} (not reached)"); continue
            x = derived(t)
            print(f"{name[:12]:12s} {p:6s} {len(idx):4d} {t['warp']:11d} {t['cycle']:11d} {t['l2_miss']:10d} {x['MPKI']:7.2f} {x['IPC']:6.1f} {x['AT']:7.1f} {x['PTW']:7.1f} {x['rr']:6.4f}")
