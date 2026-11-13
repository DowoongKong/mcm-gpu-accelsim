# How many L2 slices do the memory requests of one kernel spread over? (docs/s13_l2_bw_codepath.ko.md)
# Every global memory instruction of the trace is turned into 32 B sector requests, each sector's VA is
# translated with a placement map (the same one the simulator loads), and the destination L2 slice is
# computed with the simulator's own address-decoder rule for this config:
#   chip (DRAM channel, 0..127) = PA[12:8] | PA[35:34] << 5     (addr_dec_mask[CHIP] = 0x0000000c00001f00)
#   sub-partition (L2 slice)    = chip * 2 + PA[13]             (sub_partition_id_mask = 0x2000)
#   chiplet                     = PA[35:34]                     (= chip >> 5, mem [32c, 32c+32) per chiplet)
# with -gpgpu_memory_partition_indexing 0 (CONSECUTIVE: no hashing).
# Two views:
#   per instruction: sectors vs. distinct slices of ONE warp instruction (independent of timing);
#   round:           all warps of the kernel advance one memory instruction per round (lockstep); a round
#                    is every warp's i-th memory instruction, i.e. an upper bound on what is in flight together.
#                    Distinct slices / chiplets over the whole round are counted.
#   window:          the sectors of a round are shuffled (fixed seed) so that a window mixes requests from
#                    many warps, as simultaneous issue from 256 SMs would; the stream is cut into windows of
#                    W = 256 requests (= one request per slice, what the L2 could take in ONE cycle at peak).
# The same stream is also scored with an XOR-folded slice index (hash inside a chiplet, chiplet bits kept)
# to show what a hashed mapping would do. That hash is a what-if, not something the simulator implements.
# Usage: l2_slice_spread.py <trace dir> <placement map> <kernel id> [<kernel id> ...]
import collections, os, random, re, subprocess, sys

tdir, mappath = sys.argv[1], sys.argv[2]
kids = [int(k) for k in sys.argv[3:]]
W = 256
N_SLICE = 256

shift = None; vmap = {}
for line in open(mappath):
    if line.startswith("# page_shift"): shift = int(line.split()[-1])
    elif line.startswith("D "):
        _, vpn, own, pfn = line.split()[:4]
        vmap[int(vpn, 16)] = (int(own), int(pfn, 16))
off_mask = (1 << shift) - 1


def slice_consecutive(pa):
    chip = ((pa >> 8) & 0x1f) | (((pa >> 34) & 3) << 5)
    return chip * 2 + ((pa >> 13) & 1)


def ipoly64(a, b):
    # hashing.cc ipoly_hash_function(), bank_set_num == 64 branch, bit for bit
    A = lambda i: (a >> i) & 1
    B = lambda i: (b >> i) & 1
    n = [A(18) ^ A(17) ^ A(16) ^ A(15) ^ A(12) ^ A(10) ^ A(6) ^ A(5) ^ A(0) ^ B(0),
         A(15) ^ A(13) ^ A(12) ^ A(11) ^ A(10) ^ A(7) ^ A(5) ^ A(1) ^ A(0) ^ B(1),
         A(16) ^ A(14) ^ A(13) ^ A(12) ^ A(11) ^ A(8) ^ A(6) ^ A(2) ^ A(1) ^ B(2),
         A(17) ^ A(15) ^ A(14) ^ A(13) ^ A(12) ^ A(9) ^ A(7) ^ A(3) ^ A(2) ^ B(3),
         A(18) ^ A(16) ^ A(15) ^ A(14) ^ A(13) ^ A(10) ^ A(8) ^ A(4) ^ A(3) ^ B(4),
         A(17) ^ A(16) ^ A(15) ^ A(14) ^ A(11) ^ A(9) ^ A(5) ^ A(4) ^ B(5)]
    return sum(v << i for i, v in enumerate(n))


def slice_chiplet_ipoly(pa):
    # Proposed -gpgpu_memory_partition_indexing 7 (docs/chiplet_local_slice_hash_design.ko.md):
    # local slice = PA[12:8]*2 + PA[13]; high = partition_address(pa) >> 8 = PA[33:14] | PA[63:36] << 20.
    local = ((pa >> 8) & 31) * 2 + ((pa >> 13) & 1)
    high = ((pa >> 14) & ((1 << 20) - 1)) | ((pa >> 36) << 20)
    return ((pa >> 34) & 3) * 64 + ipoly64(high, local)


def slice_xorfold(pa):
    local = ((pa >> 8) & 63) ^ ((pa >> 14) & 63) ^ ((pa >> 20) & 63) ^ ((pa >> 26) & 63)
    return ((pa >> 34) & 3) * 64 + local


lines = open(os.path.join(tdir, "kernelslist.g")).read().splitlines()
trace_of = {int(re.match(r"kernel-(\d+)-ctx", l).group(1)): os.path.join(tdir, l) for l in lines if l.startswith("kernel-")}


def streams_of(path):
    st = collections.defaultdict(list); cta = warp = None; gx = gy = 1; name = ""; bad = [0, 0]
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
        pas = set()
        for va in {v >> 5 << 5 for v in a}:
            m = vmap.get(va >> shift)
            if m is None: bad[1] += 1; continue
            bad[0] += 1
            pas.add((m[1] << 12) + (va & off_mask))
        if pas: st[(cta, warp)].append(sorted(pas))
    p.wait()
    return st, name, bad


def pct(v, q): return v[min(len(v) - 1, int(q * len(v)))]


for kid in kids:
    st, name, bad = streams_of(trace_of[kid])
    inst_sec = inst_sl = n_inst = 0
    hist_ratio = collections.Counter()
    stream = []; rounds = []
    rng = random.Random(12345)
    keys = sorted(st); ptr = {k: 0 for k in keys}; live = list(keys)
    while live:                       # lockstep: one memory instruction per warp per round
        nxt = []; rnd = []
        for k in live:
            pas = st[k][ptr[k]]
            sl = {slice_consecutive(pa) for pa in pas}
            n_inst += 1; inst_sec += len(pas); inst_sl += len(sl)
            hist_ratio[len(sl)] += len(pas)
            rnd.extend(pas)
            ptr[k] += 1
            if ptr[k] < len(st[k]): nxt.append(k)
        rounds.append((len(live), len({slice_consecutive(pa) for pa in rnd}), len({slice_xorfold(pa) for pa in rnd}),
                       len({(pa >> 34) & 3 for pa in rnd}), len(rnd), len({slice_chiplet_ipoly(pa) for pa in rnd})))
        rng.shuffle(rnd); stream.extend(rnd)
        live = nxt
    res = {}
    for fname, f in (("consecutive", slice_consecutive), ("xorfold", slice_xorfold), ("chiplet_ipoly", slice_chiplet_ipoly)):
        d = []; top = []
        for s in range(0, len(stream) - W + 1, W):
            c = collections.Counter(f(pa) for pa in stream[s:s + W])
            d.append(len(c)); top.append(max(c.values()))
        d.sort(); top.sort()
        res[fname] = (d, top)
    chips = sorted(len({(pa >> 34) & 3 for pa in stream[s:s + W]}) for s in range(0, len(stream) - W + 1, W))
    uni = N_SLICE * (1 - (1 - 1 / N_SLICE) ** W)
    print(f"kernel {kid} {name[:70]}")
    print(f"  sectors {len(stream):,} (unmapped sectors skipped {bad[1]:,}), warp mem insts {n_inst:,}, "
          f"sectors/inst {inst_sec / n_inst:.2f}, distinct slices/inst {inst_sl / n_inst:.2f}")
    tot = sum(hist_ratio.values())
    print("  share of sectors by 'distinct slices in their instruction': " +
          ", ".join(f"{k}:{100 * v / tot:.1f}%" for k, v in sorted(hist_ratio.items()) if 100 * v / tot >= 0.5))
    print(f"  destination chiplets per window: median {pct(chips, .5)} (p10 {pct(chips, .1)}, p90 {pct(chips, .9)}) of 4")
    wsum = sum(r[4] for r in rounds)   # sector-weighted round statistics

    def wmed(ix):
        acc = 0
        for r in sorted(rounds, key=lambda r: r[ix]):
            acc += r[4]
            if acc >= wsum / 2: return r[ix]
    print(f"  rounds {len(rounds):,}: warps/round median {wmed(0)}, sectors/round median {wmed(4)}; per round (sector-weighted median) "
          f"distinct slices consecutive {wmed(1)} / xorfold {wmed(2)} / chiplet_ipoly {wmed(5)} of {N_SLICE}, chiplets {wmed(3)} of 4")
    for fname, (d, top) in res.items():
        print(f"  [{fname:13s}] window {W} sectors: distinct slices median {pct(d, .5)} (p10 {pct(d, .1)}, p90 {pct(d, .9)}) "
              f"of {N_SLICE} [uniform random would give {uni:.0f}]; busiest slice in window median {pct(top, .5)} "
              f"(p90 {pct(top, .9)}) requests")
    sys.stdout.flush()
