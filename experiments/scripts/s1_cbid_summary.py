# Step 1 (build D): summarize "[memcpy-cbid]" discovery lines from a vLLM run log.
# Counts callbacks per (cbid, name), sums bytes on entry, checks nesting, and splits the run into
# engine init and generate using vLLM's "init engine (...) took" log line as the boundary.
# Explained in llm_exp/docs/s1_tracer_async_memcpy.ko.md.
import re
import sys
from collections import Counter, defaultdict

CHECKPOINT_TENSOR_BYTES = 16060522496
log_path = sys.argv[1]

pat = re.compile(r"\[memcpy-cbid\] (enter|exit) +cbid=(\d+) name=(\S+) bytes=(-?\d+)")
phase = "init"
calls = Counter()
bytes_sum = defaultdict(int)
sizes = defaultdict(Counter)
depth = 0
max_depth = 0
nested_examples = []
prev = None
for line in open(log_path, errors="replace"):
    if "init engine (profile, create kv cache, warmup model) took" in line:
        phase = "generate"
        continue
    m = pat.search(line)
    if not m:
        continue
    kind, cbid, name, nbytes = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    key = (phase, cbid, name)
    if kind == "enter":
        if depth > 0 and len(nested_examples) < 5:
            nested_examples.append((prev, (cbid, name, nbytes)))
        depth += 1
        max_depth = max(max_depth, depth)
        calls[key] += 1
        if nbytes >= 0:
            bytes_sum[key] += nbytes
            sizes[key][nbytes] += 1
        prev = (cbid, name, nbytes)
    else:
        depth -= 1

print(f"{'phase':<9} {'cbid':>4} {'name':<28} {'enters':>7} {'bytes':>15} {'GiB':>8}")
for key in sorted(calls, key=lambda k: (k[0] != "init", k[1])):
    ph, cbid, name = key
    print(f"{ph:<9} {cbid:>4} {name:<28} {calls[key]:>7} {bytes_sum[key]:>15} {bytes_sum[key] / 2**30:>8.3f}")
print("max nesting depth:", max_depth, "(1 = no callback enters inside another)")
for a, b in nested_examples:
    print("   nested:", a, "->", b)
print("final depth (0 = every enter had an exit):", depth)
total_by_cbid = defaultdict(int)
for (ph, cbid, name), b in bytes_sum.items():
    total_by_cbid[(cbid, name)] += b
for (cbid, name), b in sorted(total_by_cbid.items()):
    print(f"total {cbid} {name}: {b} bytes, >= checkpoint tensor bytes ({CHECKPOINT_TENSOR_BYTES}): {b >= CHECKPOINT_TENSOR_BYTES}")
for key in sorted(sizes, key=lambda k: (k[0] != "init", k[1])):
    top = sizes[key].most_common(6)
    print("sizes", key, "distinct", len(sizes[key]), "top", top)
