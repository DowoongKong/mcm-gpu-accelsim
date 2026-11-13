# Steps 7-8, replacement copy check (G7-4' / G8-5'): every model weight copy of the batch-1 reference capture is
# recorded in the batch-32 capture. The registered G7-4 / G8-5 assumed that the first 518 copy lines are exactly the
# model-load copies; at batch 32 the last small copies of that stretch are batch-dependent buffers, so that criterion
# cannot hold (docs/s7_b32_prefill_layer.ko.md 8.2). Weights are the copies of 1 MiB or more.
# Usage: s7_check_copies.py <batch-32 traces dir> <batch-1 reference traces dir>
import glob
import sys

LARGE = 1 << 20


def copies(d):
    lines = open(glob.glob(d + "/kernelslist_ctx_*")[0]).read().splitlines()
    return [int(ln.split(",")[2]) for ln in lines if ln.startswith("MemcpyHtoD,")]


new, ref = copies(sys.argv[1]), copies(sys.argv[2])
big_new, big_ref = sorted(c for c in new if c >= LARGE), sorted(c for c in ref if c >= LARGE)
same_prefix = next((i for i in range(min(len(new), len(ref))) if new[i] != ref[i]), min(len(new), len(ref)))
print(f"reference: {len(ref)} copies, {sum(ref)} B; capture: {len(new)} copies, {sum(new)} B")
print(f"copies >= 1 MiB: reference {len(big_ref)} ({sum(big_ref)} B), capture {len(big_new)} ({sum(big_new)} B)")
print(f"identical leading copy lines: {same_prefix}")
ok = big_new == big_ref and len(big_ref) == 226
print("COPIES", "PASS" if ok else "FAIL", "(weight copies identical to the batch-1 reference)")
