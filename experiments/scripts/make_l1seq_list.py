# Build kernelslist_L1seq.g for a step-piece capture: every MemcpyHtoD line + the kernels of prep, layer 0 and tail of
# every step (prefill, decode1, decode2, ...), all lines copied verbatim from the post-processed kernelslist.g in their
# original order (so the simulator sees prefill -> decode1 -> decode2 as one run, layer 0 only). Layer-1 kernels are
# left out on purpose (a capture with 2 traced layers still gives a layer-0-only sequence). The kernel ranges come
# from regions.tsv written by steps_capture.py. Prints the piece index ranges (1-based, as s8_extract_l1seq.py uses
# them), the kernel / copy counts and the md5 of the written file.
# --layers N keeps layers 0..N-1 of every step instead (N=2 -> kernelslist_L2seq.g, step 11: prefill and both decode
# steps with two layers each); it needs a capture whose regions.tsv holds layer0..layer<N-1> for every step.
# Usage: make_l1seq_list.py <traces dir> <regions.tsv> [--layers N] [--out <path>]
#        (default out: <traces dir>/kernelslist_L<N>seq.g, N = 1)
import hashlib
import os
import re
import sys

args = sys.argv[1:]
out_path = None
if "--out" in args:
    i = args.index("--out"); out_path = args[i + 1]; del args[i:i + 2]
n_layers = 1
if "--layers" in args:
    i = args.index("--layers"); n_layers = int(args[i + 1]); del args[i:i + 2]
layer_pieces = [f"layer{k}" for k in range(n_layers)]
traces_dir, regions_path = args[:2]
out_path = out_path or os.path.join(traces_dir, f"kernelslist_L{n_layers}seq.g")
rng = {}
steps = []
for ln in open(regions_path).read().splitlines()[1:]:
    step, piece, a, b = ln.split("\t")[:4]
    rng[(step, piece)] = (int(a), int(b))
    if step not in steps:
        steps.append(step)
keep = set()
for step in steps:
    for piece in ["prep"] + layer_pieces + ["tail"]:
        a, b = rng[(step, piece)]
        keep.update(range(a, b + 1))
lines = open(os.path.join(traces_dir, "kernelslist.g")).read().splitlines()
out, kept_ids = [], []
for ln in lines:
    if ln.startswith("MemcpyHtoD,"):
        out.append(ln)
        continue
    kid = int(re.match(r"kernel-(\d+)-ctx", ln).group(1))
    if kid in keep:
        out.append(ln)
        kept_ids.append(kid)
assert kept_ids == sorted(keep), "kernelslist.g does not hold every kernel of the regions in order"
with open(out_path, "w") as f:
    f.write("\n".join(out) + "\n")
n_copy = sum(1 for ln in out if ln.startswith("MemcpyHtoD,"))
pos = {kid: i + 1 for i, kid in enumerate(kept_ids)}       # 1-based kernel index inside the L1seq list
tags = {"prefill": "P", "decode1": "D1", "decode2": "D2", "decode3": "D3"}
for step in steps:
    t = tags.get(step, step)
    pr, tl = rng[(step, "prep")], rng[(step, "tail")]
    ls = [rng[(step, lp)] for lp in layer_pieces]
    line = f"{t}-O\t{pos[pr[0]]}-{pos[pr[1]]},{pos[tl[0]]}-{pos[tl[1]]}\t{t}-L{n_layers}\t{pos[ls[0][0]]}-{pos[ls[-1][1]]}"
    if n_layers > 1:   # per-layer ranges inside the multi-layer piece
        line += "".join(f"\t{lp}\t{pos[a]}-{pos[b]}" for lp, (a, b) in zip(layer_pieces, ls))
    print(line)
md5 = hashlib.md5(open(out_path, "rb").read()).hexdigest()
print(f"L{n_layers}SEQ LIST\tkernels={len(kept_ids)}\tcopies={n_copy}\tmd5={md5}\t{out_path}")
