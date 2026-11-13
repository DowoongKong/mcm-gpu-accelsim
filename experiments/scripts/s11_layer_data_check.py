# Step 11 pre-check: do the layer-0 and layer-1 kernels of a 2-layer capture touch different data?
# For every kernel of layer0 / layer1 of each step (regions.tsv), collect the 4 KiB pages its memory instructions
# address (post-processed .traceg.xz; address modes 0 list / 1 base+stride / 2 base+delta, all active lanes), then
# split the pages into HtoD weight copies (MemcpyHtoD ranges of kernelslist.g) and everything else (activations,
# KV cache, workspace), and report per step and per kernel pair how much layer 0 and layer 1 share.
# Usage: s11_layer_data_check.py <traces dir> <regions.tsv> [-j N]
import bisect, collections, os, re, subprocess, sys
from multiprocessing import Pool

PAGE_SHIFT = 12


def kernel_pages(path):
    """Set of 4 KiB page numbers addressed by the memory instructions of one .traceg.xz."""
    pages = set()
    p = subprocess.Popen(f"xz -dc '{path}' | grep -F ' 0x'", shell=True, stdout=subprocess.PIPE)
    for line in p.stdout:
        t = line.split()
        if len(t) < 4 or not t[2].isdigit():   # header lines such as '-shmem base_addr = 0x...'
            continue
        i = 3 + int(t[2])                      # skip PC, mask, dest count and dest regs -> opcode
        i += 2 + int(t[i + 1])                 # skip opcode, src count and src regs -> mem_width
        if int(t[i]) == 0:
            continue
        mode, n = int(t[i + 1]), bin(int(t[1], 16)).count("1")
        if mode == 0:                          # one address per active lane
            addrs = [int(x, 16) for x in t[i + 2:i + 2 + n]]
        elif mode == 1:                        # base + fixed stride
            base, stride = int(t[i + 2], 16), int(t[i + 3])
            addrs = [base + k * stride for k in range(n)]
        else:                                  # base + per-lane deltas
            addrs = [int(t[i + 2], 16)]
            for d in t[i + 3:i + 2 + n]:
                addrs.append(addrs[-1] + int(d))
        pages.update(a >> PAGE_SHIFT for a in addrs)
    p.wait()
    return path, pages


def main():
    args = sys.argv[1:]
    jobs = 4
    if "-j" in args:
        k = args.index("-j"); jobs = int(args[k + 1]); del args[k:k + 2]
    tdir, regions = args
    lines = open(os.path.join(tdir, "kernelslist.g")).read().splitlines()
    copies = sorted({(int(l.split(",")[1], 16), int(l.split(",")[2])) for l in lines if l.startswith("MemcpyHtoD,")})
    order = {}                                 # copy -> position in transfer order (weights load layer by layer)
    for l in lines:
        if l.startswith("MemcpyHtoD,"):
            c = (int(l.split(",")[1], 16), int(l.split(",")[2])); order.setdefault(c, len(order))
    starts = [a for a, _ in copies]
    trace_of = {int(re.match(r"kernel-(\d+)-ctx", l).group(1)): os.path.join(tdir, l) for l in lines if l.startswith("kernel-")}
    rng = {}
    for ln in open(regions).read().splitlines()[1:]:
        step, piece, a, b = ln.split("\t")[:4]
        rng[(step, piece)] = list(range(int(a), int(b) + 1))
    steps = [s for s in dict.fromkeys(s for s, _ in rng) if (s, "layer1") in rng]
    todo = [trace_of[k] for s in steps for lp in ("layer0", "layer1") for k in rng[(s, lp)]]
    with Pool(jobs) as pool:
        got = dict(pool.map(kernel_pages, todo, chunksize=1))

    def split(pages):
        """(weight pages by copy, other pages)"""
        by_copy, other = collections.defaultdict(set), set()
        for vpn in pages:
            va = vpn << PAGE_SHIFT; k = bisect.bisect_right(starts, va) - 1
            if k >= 0 and va < copies[k][0] + copies[k][1]:
                by_copy[copies[k]].add(vpn)
            else:
                other.add(vpn)
        return by_copy, other

    mib = lambda n: n * 4096 / 2**20
    for s in steps:
        L0 = set().union(*(got[trace_of[k]] for k in rng[(s, "layer0")]))
        L1 = set().union(*(got[trace_of[k]] for k in rng[(s, "layer1")]))
        w0, o0 = split(L0); w1, o1 = split(L1)
        wp0, wp1 = set().union(*w0.values()), set().union(*w1.values())
        c0 = sorted(order[c] for c in w0); c1 = sorted(order[c] for c in w1)
        print(f"== {s}: layer0 kernels {rng[(s, 'layer0')][0]}-{rng[(s, 'layer0')][-1]}, layer1 {rng[(s, 'layer1')][0]}-{rng[(s, 'layer1')][-1]}")
        print(f"  weight (HtoD) pages : L0 {len(wp0):8,} ({mib(len(wp0)):7.1f} MiB)  L1 {len(wp1):8,} ({mib(len(wp1)):7.1f} MiB)  shared {len(wp0 & wp1):,}")
        print(f"  weight copies used  : L0 {len(c0)} (transfer order {c0})")
        print(f"                        L1 {len(c1)} (transfer order {c1})  shared copies {len(set(w0) & set(w1))}")
        print(f"  other pages         : L0 {len(o0):8,} ({mib(len(o0)):7.1f} MiB)  L1 {len(o1):8,} ({mib(len(o1)):7.1f} MiB)  shared {len(o0 & o1):,} ({mib(len(o0 & o1)):.1f} MiB)")
        print(f"  {'pair':>11} {'kernel (layer0)':46} {'wt L0':>7} {'wt L1':>7} {'wt sh':>6} {'oth L0':>7} {'oth L1':>7} {'oth sh':>7}")
        for k0, k1 in zip(rng[(s, "layer0")], rng[(s, "layer1")]):
            a, b = split(got[trace_of[k0]]), split(got[trace_of[k1]])
            wa, wb = set().union(*a[0].values()) if a[0] else set(), set().union(*b[0].values()) if b[0] else set()
            kn = subprocess.run(f"xz -dc '{trace_of[k0]}' | head -1", shell=True, capture_output=True, text=True).stdout.split("=")[-1].strip()
            kn1 = subprocess.run(f"xz -dc '{trace_of[k1]}' | head -1", shell=True, capture_output=True, text=True).stdout.split("=")[-1].strip()
            tag = "" if kn == kn1 else "  (NAME DIFFERS)"
            print(f"  {k0:>5}/{k1:<5} {kn[:46]:46} {len(wa):7,} {len(wb):7,} {len(wa & wb):6,} {len(a[1]):7,} {len(b[1]):7,} {len(a[1] & b[1]):7,}{tag}")


if __name__ == "__main__":
    main()
