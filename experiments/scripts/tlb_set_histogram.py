# Per-layer L2 TLB set histogram for an LLM capture (docs/llm_tlb_miss_3c_analysis.ko.md section 9).
# For every kernel of one layer (regions.tsv), replay the page references with the interleaving model of
# tlb_3c_model.py (CTA -> chiplet by CHUNK_BINDING chunk 4, per-SM 32-entry fully associative LRU L1 TLB) and count
# the L1-missing references that reach the chiplet L2 TLB by set = vpn % 128.
# Outputs JSON: whole-layer set histogram (all chiplets) and, for chiplet 0, per-window set counts (windows of W refs)
# to show which sets are live at the same time.
# Usage: tlb_set_histogram.py <traces dir> <regions.tsv> <step> <layerN> <out.json> [W=1024]
import collections, json, os, re, subprocess, sys
from collections import OrderedDict

SETS = 128
tdir, regions, step, layer, out = sys.argv[1:6]
W = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
rng = {}
for ln in open(regions).read().splitlines()[1:]:
    s, p, a, b = ln.split("\t")[:4]
    rng[(s, p)] = range(int(a), int(b) + 1)
lines = open(os.path.join(tdir, "kernelslist.g")).read().splitlines()
trace_of = {int(re.match(r"kernel-(\d+)-ctx", l).group(1)): os.path.join(tdir, l) for l in lines if l.startswith("kernel-")}


class LRU:
    def __init__(self, cap): self.cap, self.d = cap, OrderedDict()
    def access(self, k):
        if k in self.d: self.d.move_to_end(k); return True
        self.d[k] = None
        if len(self.d) > self.cap: self.d.popitem(last=False)
        return False


def streams_of(path):
    st = collections.defaultdict(list); cta = warp = None; gx = gy = 1; name = ""
    p = subprocess.Popen(["xz", "-dc", path], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    for line in p.stdout:
        if line.startswith("-kernel name"): name = line.split("=", 1)[1].strip(); continue
        if line.startswith("-grid dim"):
            g = line.split("=")[1].strip().strip("()").split(","); gx, gy = int(g[0]), int(g[1]); continue
        if line.startswith("thread block"):
            x, y, z = (int(v) for v in line.split("=")[1].split(",")); cta = x + y * gx + z * gx * gy; continue
        if line.startswith("warp ="): warp = int(line.split("=")[1]); continue
        t = line.split()
        if len(t) < 5 or not t[2].isdigit(): continue
        i = 3 + int(t[2]); op = t[i]; i += 2 + int(t[i + 1]); w = int(t[i])
        if w == 0 or len(t) <= i + 1 or not op.startswith(("LDG", "STG", "LDGSTS", "ATOMG", "RED")): continue
        mode, n = int(t[i + 1]), bin(int(t[1], 16)).count("1")
        if mode == 0: a = [int(v, 16) for v in t[i + 2:i + 2 + n]]
        elif mode == 1: b, s = int(t[i + 2], 16), int(t[i + 3]); a = [b + k * s for k in range(n)]
        else:
            a = [int(t[i + 2], 16)]
            for d in t[i + 3:i + 2 + n]: a.append(a[-1] + int(d))
        st[(cta, warp)].append(sorted({v >> 12 for v in a}))
    p.wait()
    return st, name


hist = [0] * SETS; windows = []; kernels = []
for kid in rng[(step, layer)]:
    st, name = streams_of(trace_of[kid])
    by_chip = collections.defaultdict(list)
    for key in sorted(st): by_chip[(key[0] // 4) % 4].append(key)
    khist = [0] * SETS; cur = [0] * SETS; filled = 0
    for chip, keys in by_chip.items():
        ctas = sorted({k[0] for k in keys}); sm_of = {c: i % 64 for i, c in enumerate(ctas)}
        l1 = {s: LRU(32) for s in range(64)}
        ptr = {k: 0 for k in keys}; live = list(keys)
        while live:
            nxt = []
            for k in live:
                for pg in st[k][ptr[k]]:
                    if l1[sm_of[k[0]]].access(pg): continue
                    s = pg % SETS; hist[s] += 1; khist[s] += 1
                    if chip == 0:
                        cur[s] += 1; filled += 1
                        if filled == W: windows.append([kid] + cur); cur = [0] * SETS; filled = 0
                ptr[k] += 1
                if ptr[k] < len(st[k]): nxt.append(k)
            live = nxt
    kernels.append({"kernel": kid, "name": name[:60], "l2_refs": sum(khist)})
    print(f"{step} {layer} kernel {kid} {name[:40]} L2 refs {sum(khist):,}", flush=True)
json.dump({"step": step, "layer": layer, "sets": SETS, "window": W, "hist": hist, "kernels": kernels, "windows": windows}, open(out, "w"))
