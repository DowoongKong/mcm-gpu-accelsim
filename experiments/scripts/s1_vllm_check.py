# Step 1 (builds R, N): check the kernelslist of a vLLM tracer run and compare it with the V2 reference.
# Usage: s1_vllm_check.py <test dir> <V2 reference summary json> <expected lines> <expected bytes> [other test dir]
# The optional last argument compares the MemcpyHtoD size sequence with another run (N against R).
# Explained in llm_exp/docs/s1_tracer_async_memcpy.ko.md.
import glob
import json
import os
import sys
from collections import Counter

CHECKPOINT_TENSOR_BYTES = 16060522496
test_dir, ref_json, exp_lines, exp_bytes = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
other_dir = sys.argv[5] if len(sys.argv) > 5 else None


def kernelslist_lines(d):
    paths = glob.glob(os.path.join(d, "traces", "kernelslist_ctx_*"))
    assert len(paths) == 1, paths
    return open(paths[0]).read().splitlines()


lines = kernelslist_lines(test_dir)
memcpy = [ln for ln in lines if ln.startswith("MemcpyHtoD,")]
sizes = [int(ln.split(",")[2]) for ln in memcpy]
addrs = [ln.split(",")[1] for ln in memcpy]
total = sum(sizes)
summary = json.load(open(glob.glob(os.path.join(test_dir, "logs", "v2_summary_tracer_S*.json"))[0]))
ref = json.load(open(ref_json))

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


check("kernelslist has only MemcpyHtoD lines", len(lines) == len(memcpy), f"{len(lines)} lines")
check("MemcpyHtoD bytes >= checkpoint tensor bytes", total >= CHECKPOINT_TENSOR_BYTES, f"{total} vs {CHECKPOINT_TENSOR_BYTES}")
check("MemcpyHtoD line count == expected (no duplicates)", len(memcpy) == exp_lines, f"{len(memcpy)} vs {exp_lines}")
check("MemcpyHtoD bytes == expected", total == exp_bytes, f"{total} vs {exp_bytes}")
check("output ids == V2", summary["output_ids"] == ref["output_ids"], summary["output_ids"])
check("hook events == V2", summary["events"] == ref["events"], f"{len(summary['events'])} events")
check("window kernel ranges == V2", summary.get("windows") == ref.get("windows"))
check("kernel launches == V2", summary.get("n_kernel_launches") == ref.get("n_kernel_launches"),
      summary.get("n_kernel_launches"))
if other_dir:
    other = [int(ln.split(",")[2]) for ln in kernelslist_lines(other_dir) if ln.startswith("MemcpyHtoD,")]
    check("MemcpyHtoD size sequence == other run", sizes == other, f"{len(sizes)} vs {len(other)}")

print("sizes (top):", Counter(sizes).most_common(8))
print("distinct destination addresses:", len(set(addrs)), "of", len(addrs))
print("init_s", round(summary["init_s"], 1), "generate_s", round(summary["generate_s"], 3))
print("S1 CHECK", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
