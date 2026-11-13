# Step 8: build kernelslist_<piece>.g for every piece = every MemcpyHtoD line + the piece's kernels, lines copied
# verbatim from the post-processed kernelslist.g and kept in their original order. Piece ranges come from regions.tsv
# written by scripts/s8_steps_capture.py. Explained in docs/s8_b32s256_step_pieces.ko.md section 4.
# Usage: s8_make_piece_lists.py <traces dir> <regions.tsv>
import hashlib
import os
import re
import sys

traces_dir, regions_path = sys.argv[1:3]
rng = {}
for ln in open(regions_path).read().splitlines()[1:]:
    step, piece, a, b = ln.split("\t")[:4]
    rng[(step, piece)] = range(int(a), int(b) + 1)
PIECES = {}
for tag, step in (("P", "prefill"), ("D1", "decode1"), ("D2", "decode2")):
    PIECES[f"{tag}-O"] = list(rng[(step, "prep")]) + list(rng[(step, "tail")])
    PIECES[f"{tag}-L1"] = list(rng[(step, "layer0")])
    if (step, "layer1") in rng:          # captures that stop after layer 0 have no layer-1 piece
        PIECES[f"{tag}-L2"] = list(rng[(step, "layer0")]) + list(rng[(step, "layer1")])
lines = open(os.path.join(traces_dir, "kernelslist.g")).read().splitlines()
copies = [ln for ln in lines if ln.startswith("MemcpyHtoD,")]
stats = {}
for name in os.listdir(traces_dir):
    if name.startswith("stats_ctx_"):
        for ln in open(os.path.join(traces_dir, name)).read().splitlines()[1:]:
            parts = ln.split(",")
            stats[int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1))] = int(parts[-1])
ok_all = True
for piece, ids in PIECES.items():
    want = set(ids)
    out = [ln for ln in lines if ln.startswith("MemcpyHtoD,") or int(re.match(r"kernel-(\d+)-ctx", ln).group(1)) in want]
    kids = [int(re.match(r"kernel-(\d+)-ctx", ln).group(1)) for ln in out if not ln.startswith("MemcpyHtoD,")]
    ok = kids == sorted(ids) and [ln for ln in out if ln.startswith("MemcpyHtoD,")] == copies
    ok_all &= ok
    path = os.path.join(traces_dir, f"kernelslist_{piece}.g")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    print(f"{piece}\t{len(kids)}\t{sum(stats[k] for k in kids)}\t{md5}\t{'OK' if ok else 'BAD'}\t{path}")
print("S8 PIECE LISTS", "PASS" if ok_all else "FAIL")
