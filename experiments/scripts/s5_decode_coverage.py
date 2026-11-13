# Step 5 (B): how much of one decode step the step-4 (A) sample covers, from the tracer stats of the full decode
# capture (reported warp instructions per kernel, before post-processing). Sections come from the kernel map
# (scripts/s256_kernel_map.py). Usage: s5_decode_coverage.py <full-decode stats_ctx file> <kernel map tsv>
import collections
import re
import sys

stats_path, map_path = sys.argv[1:3]
insts = {}
for ln in open(stats_path).read().splitlines()[1:]:
    parts = ln.split(",")
    insts[int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1))] = int(parts[-1])
sections = {}
for ln in open(map_path).read().splitlines()[1:]:
    f = ln.split("\t")
    sections[int(f[0])] = (f[1], f[2], f[3], f[4])
decode = [k for k, (step, _, _, _) in sections.items() if step == "decode"]
total = sum(insts[k] for k in decode)
groups = collections.Counter()
per_layer = collections.Counter()
for k in decode:
    _, section, layer, module = sections[k]
    if layer == "":
        groups[f"outside layers: {section}"] += insts[k]
    elif module in ("self_attn", "mlp"):
        groups["layers: attention + FFN"] += insts[k]
        per_layer[int(layer)] += insts[k]
    else:
        groups["layers: norms"] += insts[k]
print(f"decode step: {len(decode)} kernels, {total} reported warp instructions")
for name, v in groups.items():
    print(f"  {name}: {v} ({100 * v / total:.2f}%)")
sample = sum(per_layer[i] for i in (0, 15, 31))
print(f"step-4 sample (layers 0/15/31 attention + FFN, 30 kernels): {sample} ({100 * sample / total:.2f}%)")
print(f"per-layer attention + FFN: min {min(per_layer.values())} max {max(per_layer.values())}")
print(f"sample x 32/3 = {sample * 32 / 3:.0f} vs all 32 layers {groups['layers: attention + FFN']} "
      f"({100 * (sample * 32 / 3 - groups['layers: attention + FFN']) / groups['layers: attention + FFN']:+.4f}%)")
for k in decode:
    if sections[k][2] == "" and insts[k] > 100000:
        print(f"  large kernel outside layers: {k} {insts[k]} ({100 * insts[k] / total:.2f}%) {sections[k][1]}")
