# Step 17 layer-internal comparison (docs/s17d_layer_internals_8b_vs_70b.ko.md): per-kernel memory footprint of a kernel
# range in a post-processed capture. For every kernel: warp instructions, global memory warp instructions, distinct
# 4 KiB / 64 KiB / 2 MiB pages, and 32 B sectors touched (sum over instructions, not unique). Address decoding follows
# scripts/s11_layer_data_check.py (modes 0 list / 1 base+stride / 2 base+delta, active lanes from the mask).
# Usage: s17_layer_scan.py <traces dir> <first kernel id> <last kernel id> [<first2> <last2> ...]
import glob
import os
import re
import subprocess
import sys

traces_dir = sys.argv[1]
ranges = [(int(sys.argv[i]), int(sys.argv[i + 1])) for i in range(2, len(sys.argv), 2)]
GLOBAL_OPS = ("LDG", "STG", "LDGSTS", "ATOMG", "RED")


def short_name(name):
    rules = [("rms_norm_kernel", "rms_norm"), ("fused_add_rms_norm", "fused_add_rms_norm"), ("rotary_embedding", "rotary_embedding"),
             ("reshape_and_cache_flash", "reshape_and_cache_flash"), ("flash_fwd_splitkv_combine", "flash_fwd_splitkv_combine"),
             ("flash_fwd_splitkv", "flash_fwd_splitkv"), ("act_and_mul", "silu_and_mul"), ("elementwise_kernel", "torch_elementwise_copy"),
             ("splitKreduce", "cublasLt_splitKreduce")]
    for key, val in rules:
        if key in name:
            return val
    m = re.search(r"(cutlass_80_[a-z0-9_]+?)(EEvNT|$)", name)
    if m:
        return m.group(1)
    m = re.search(r"(ampere_bf16_[a-z0-9_]+)", name)
    return m.group(1) if m else name[:60]


def scan(path):
    name = None
    n_warp = n_mem = sectors = 0
    p4 = set()
    proc = subprocess.Popen(["xz", "-dc", path], stdout=subprocess.PIPE, text=True, bufsize=1 << 22)
    for line in proc.stdout:
        if line.startswith("-kernel name"):
            name = line.split("=", 1)[1].strip()
            continue
        if line[0] == "-" or line.startswith("thread block") or line.startswith("warp") or line.startswith("insts"):
            continue
        t = line.split()
        if len(t) < 5 or not t[2].isdigit():
            continue
        n_warp += 1
        i = 3 + int(t[2])
        op = t[i]
        i += 2 + int(t[i + 1])
        if i >= len(t):
            continue
        w = int(t[i])
        if w == 0 or len(t) <= i + 1 or not op.startswith(GLOBAL_OPS):
            continue
        mode, n = int(t[i + 1]), bin(int(t[1], 16)).count("1")
        if mode == 0:
            addrs = [int(x, 16) for x in t[i + 2:i + 2 + n]]
        elif mode == 1:
            base, stride = int(t[i + 2], 16), int(t[i + 3])
            addrs = [base + k * stride for k in range(n)]
        else:
            addrs = [int(t[i + 2], 16)]
            for d in t[i + 3:i + 2 + n]:
                addrs.append(addrs[-1] + int(d))
        n_mem += 1
        sectors += len({a >> 5 for a in addrs})
        p4.update(a >> 12 for a in addrs)
    proc.wait()
    return name, n_warp, n_mem, sectors, p4


print("kernel\tname\twarp_insts\tglobal_mem_warp_insts\tsectors32B\tpages4K\tpages64K\tpages2M\tbytes_touched_est_MB")
tot = {}
for a, b in ranges:
    union4 = set()
    for k in range(a, b + 1):
        files = glob.glob(os.path.join(traces_dir, f"kernel-{k}-*.traceg.xz"))
        if not files:
            print(f"{k}\tMISSING")
            continue
        name, n_warp, n_mem, sectors, p4 = scan(files[0])
        union4 |= p4
        p64 = {x >> 4 for x in p4}
        p2m = {x >> 9 for x in p4}
        print(f"{k}\t{short_name(name)}\t{n_warp}\t{n_mem}\t{sectors}\t{len(p4)}\t{len(p64)}\t{len(p2m)}\t{sectors * 32 / 1e6:.1f}", flush=True)
    print(f"RANGE {a}-{b}\tunique 4K pages {len(union4)} = {len(union4) * 4096 / 1e6:.1f} MB\tunique 64K {len({x >> 4 for x in union4})}\tunique 2M {len({x >> 9 for x in union4})}", flush=True)
