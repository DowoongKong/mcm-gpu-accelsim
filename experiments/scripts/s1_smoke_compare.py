# Step 1: compare cublas_smoke tracer outputs of two tags (kernelslist structure and per-kernel stats).
# Addresses and CUDA context pointers change between runs, so records are compared as (kind, size) and
# (kernel name, grid, block, total_insts). Explained in llm_exp/docs/s1_tracer_async_memcpy.ko.md.
import glob
import os
import sys

ROOT = sys.argv[3] if len(sys.argv) > 3 else "/accel-sim/llm_exp/tracer_tests"


def load(tag):
    d = os.path.join(ROOT, f"smoke_{tag}", "traces")
    kl = glob.glob(os.path.join(d, "kernelslist_ctx_*"))
    st = glob.glob(os.path.join(d, "stats_ctx_*"))
    assert len(kl) == 1 and len(st) == 1, (tag, kl, st)
    records = []
    for ln in open(kl[0]).read().splitlines():
        if ln.startswith("MemcpyHtoD,"):
            records.append(("MemcpyHtoD", int(ln.split(",")[2])))
        else:
            records.append(("kernel", ln.split("-ctx_")[0]))
    kernels = []
    for ln in open(st[0]).read().splitlines()[1:]:
        parts = [p.strip() for p in ln.split(",")]
        kernels.append((parts[0].split("-ctx_")[0], ",".join(parts[1:-10]), parts[-10:-2], int(parts[-2]), int(parts[-1])))
    return records, kernels


def show(tag, records, kernels):
    print(f"--- {tag}: kernelslist {len(records)} lines")
    for r in records:
        print("   ", r)
    for k in kernels:
        print("    stats", k[0], k[1][:70], "total_insts", k[3], "reported", k[4])


a_tag, b_tag = sys.argv[1], sys.argv[2]
ra, ka = load(a_tag)
rb, kb = load(b_tag)
show(a_tag, ra, ka)
show(b_tag, rb, kb)
print("SAME kernelslist (kind,size) sequence:", ra == rb)
print("SAME stats (id,name,grid,block,insts):", ka == kb)
