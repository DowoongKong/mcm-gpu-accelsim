# Model-generic weight-copy check: every model tensor of 1 MiB or more (sizes from the safetensors headers) appears as
# a MemcpyHtoD record of exactly that size in the capture's kernelslist (multiset equality). Replaces the
# batch-1-specific "first 518 copy lines" check (docs/s7_b32_prefill_layer.ko.md 8.2).
# Usage: check_weight_copies.py <model dir> <traces dir>
import glob
import json
import os
import struct
import sys

LARGE = 1 << 20
BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1}
model_dir, traces_dir = sys.argv[1:3]
tensors = []
for shard in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        size = BYTES[meta["dtype"]]
        for d in meta["shape"]:
            size *= d
        tensors.append((name, size))
copies = [int(ln.split(",")[2]) for ln in open(glob.glob(os.path.join(traces_dir, "kernelslist_ctx_*"))[0])
          if ln.startswith("MemcpyHtoD,")]
big_t = sorted(s for _, s in tensors if s >= LARGE)
big_c = sorted(c for c in copies if c >= LARGE)
print(f"model: {len(tensors)} tensors, {sum(s for _, s in tensors)} B; >= 1 MiB: {len(big_t)} tensors, {sum(big_t)} B")
print(f"capture: {len(copies)} copies, {sum(copies)} B; >= 1 MiB: {len(big_c)} copies, {sum(big_c)} B")
print("WEIGHT COPIES", "PASS" if big_t == big_c else "FAIL", "(every tensor >= 1 MiB copied once with its exact size)")
