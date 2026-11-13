# Step 6 (P-3): build kernelslist_L<N>.g = every MemcpyHtoD line + decode layers 0..N-1 (12 kernels each) of the
# post-processed full-decode list, lines copied verbatim and kept in their original order.
# Checks: kernel ids are exactly 1348 .. 1348+12N-1; each 12-kernel block follows the layer layout of the kernel map
# (input_norm, self_attn x7, post_attn_norm, mlp x3) with layer index k; all copy lines are kept in order.
# Explained in docs/s6_decode_layer_scaling.ko.md 1.1. Usage: s6_make_kernelslists.py <traces dir> <kernel map tsv> N [N ...]
import hashlib
import os
import re
import sys

traces_dir, map_path = sys.argv[1:3]
layer_counts = [int(x) for x in sys.argv[3:]]
FIRST, PER_LAYER = 1348, 12
LAYOUT = ["input_norm"] + ["self_attn"] * 7 + ["post_attn_norm"] + ["mlp"] * 3

# kernelslist.g is written line by line while post-processing runs, so the lists are built from the tracer's
# kernelslist_ctx_* with post-traces-processing.cpp's own rule (MemcpyHtoD lines verbatim, ".trace.xz" -> ".traceg.xz").
# P-1 (scripts/s6_check_traces.py) later checks that the finished kernelslist.g is exactly this conversion.
ctx = [name for name in os.listdir(traces_dir) if name.startswith("kernelslist_ctx_")]
assert len(ctx) == 1, ctx
lines = [ln[:-3] + "g.xz" if ln.endswith(".trace.xz") else ln
         for ln in open(os.path.join(traces_dir, ctx[0])).read().splitlines()]
copies = [ln for ln in lines if ln.startswith("MemcpyHtoD,")]
kmap = {}
for ln in open(map_path).read().splitlines()[1:]:
    f = ln.split("\t")
    kmap[int(f[0])] = (f[3], f[4])
stats = {}
for name in os.listdir(traces_dir):
    if name.startswith("stats_ctx_"):
        for ln in open(os.path.join(traces_dir, name)).read().splitlines()[1:]:
            parts = ln.split(",")
            stats[int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1))] = int(parts[-1])

all_ok = True
for n in layer_counts:
    wanted = set(range(FIRST, FIRST + PER_LAYER * n))
    out = [ln for ln in lines if ln.startswith("MemcpyHtoD,") or int(re.match(r"kernel-(\d+)-ctx", ln).group(1)) in wanted]
    kids = [int(re.match(r"kernel-(\d+)-ctx", ln).group(1)) for ln in out if not ln.startswith("MemcpyHtoD,")]
    ok_ids = kids == sorted(wanted)
    ok_layout = all(kmap[FIRST + PER_LAYER * k + i] == (str(k), LAYOUT[i]) for k in range(n) for i in range(PER_LAYER))
    ok_copies = [ln for ln in out if ln.startswith("MemcpyHtoD,")] == copies and len(copies) == 534
    ok = ok_ids and ok_layout and ok_copies
    all_ok &= ok
    path = os.path.join(traces_dir, f"kernelslist_L{n}.g")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    warp = sum(stats[k] for k in kids)
    print(f"L{n}: {path} lines={len(out)} kernels={len(kids)} ({kids[0]}-{kids[-1]}) copies={len(copies)} "
          f"ids {'OK' if ok_ids else 'BAD'} layout {'OK' if ok_layout else 'BAD'} copies {'OK' if ok_copies else 'BAD'} "
          f"stats_warp={warp} md5={md5}")
print("S6 KERNELSLISTS", "PASS" if all_ok else "FAIL")
