#!/usr/bin/env python3
# Step 19 (docs/s19_mem_latency_microbench.ko.md): per-level memory-instruction latency of every simulation arm of the
# pointer-chase microbenchmark, next to the RTX 3090 numbers, from the per-kernel statistics the simulator prints.
# The trace holds 22 kernels in launch order: 6 k_init, then per level the N and 2N chase launches (the DRAM level has a
# k_flush before each). Kernels run one at a time, so the n-th statistics block belongs to the n-th kernel; blocks are
# matched by ORDER because some blocks print an empty "kernel_launch_uid =" line.
# Two estimates per level:
#   diff = (cycles[2N] - cycles[N]) / N      valid when both launches start from the same cache state:
#          L1/T1 (the simulator flushes L1 at every kernel end, -gpgpu_flush_l1_cache 1), L2R (both launches find the
#          data in L2), DRAM (k_flush before both).
#   warm = (cycles[2N] - F) / 2N             for the levels whose data lives in L2 (L2, T2, T3): in trace mode each
#          launch runs once, so the N launch is cold (k_init of the 256 MB buffer and k_flush evicted the data) and the
#          2N launch is warm; the diff is then biased low. F = fixed cycles of a launch with no cold pass, measured on
#          the ideal arm's L2R pair (F = cycles[L2R N] - N * diff(L2R) = kernel launch latency 5000 + ~400) and used
#          for every arm: on the translating arms the L2R N launch also pays a cold TLB pass, which is not fixed cost.
# averagemflatency (mem_fetch creation -> response, L1D misses only) of the 2N kernel and the remote share of its
# memory accesses are printed as cross-checks.
# Usage: s19_latency.py <hardware time.txt> <arm run.log> [<arm run.log> ...]
import re
import sys

LEVELS = [  # name, block of the N launch, block of the 2N launch, N, estimate to trust
    ("L1", 7, 8, 65536, "diff"), ("L2", 9, 10, 16384, "warm"), ("L2R", 11, 12, 16384, "diff"),
    ("DRAM", 14, 16, 8192, "diff"), ("T1", 17, 18, 16384, "diff"), ("T2", 19, 20, 16384, "warm"),
    ("T3", 21, 22, 16384, "warm")]
CUM = ["mcm_remote_accesses", "mcm_local_accesses", "gpu_tot_sim_insn"]


def blocks(path):
    out = []; cur = None
    for line in open(path, errors="ignore"):
        s = line.strip()
        if line.startswith("kernel_launch_uid ="):
            cur = {}; out.append(cur); continue
        if cur is None: continue
        if s.startswith("gpu_sim_cycle ="): cur["cyc"] = int(s.split("=")[1])
        elif s.startswith("averagemflatency ="): cur["mflat"] = int(s.split("=")[1])
        else:
            for k in CUM:
                if s.startswith(k + " =") or s.startswith(k + "="):
                    cur[k] = int(float(re.split(r"=\s*", s, 1)[1].split()[0]))
    return [b for b in out if "cyc" in b]


def hardware(path):
    hw = {}
    for line in open(path, errors="ignore"):
        t = line.split()
        if len(t) >= 6 and t[0] in {l[0] for l in LEVELS}:
            hw[t[0]] = float(t[5])
    return hw


hw = hardware(sys.argv[1])
arms = [(p.split("/")[-2], blocks(p)) for p in sys.argv[2:]]
FIXED = 5405.0   # fallback: the value measured on the step-19 ideal arm
for name_, b_ in arms:
    if name_ == "ideal" and len(b_) >= 12:
        FIXED = b_[10]["cyc"] - (b_[11]["cyc"] - b_[10]["cyc"])
print(f"{'level':5s} {'N':>6s} {'use':>4s} {'RTX3090':>8s} | " + " | ".join(f"{n + ': diff   warm  mflat rem%':>34s}" for n, _ in arms))
for name, k1, k2, n, use in LEVELS:
    cells = []
    for _, b in arms:
        if len(b) < 22:
            cells.append(f"{'(blocks ' + str(len(b)) + '/22)':>34s}"); continue
        fixed = FIXED
        diff = (b[k2 - 1]["cyc"] - b[k1 - 1]["cyc"]) / n
        warm = (b[k2 - 1]["cyc"] - fixed) / (2 * n)
        rem = b[k2 - 1].get("mcm_remote_accesses", 0) - b[k2 - 2].get("mcm_remote_accesses", 0)
        loc = b[k2 - 1].get("mcm_local_accesses", 0) - b[k2 - 2].get("mcm_local_accesses", 0)
        share = 100.0 * rem / (rem + loc) if rem + loc else 0.0
        mark = lambda v, w: f"{v:>7.1f}{'*' if use == w else ' '}"
        cells.append(f"{mark(diff, 'diff')} {mark(warm, 'warm')} {b[k2 - 1].get('mflat', 0):>5d} {share:>5.1f}".rjust(34))
    print(f"{name:5s} {n:>6d} {use:>4s} {hw.get(name, 0):>8.1f} | " + " | ".join(cells))
for name, b in arms:
    print(f"# {name}: blocks {len(b)}/22, thread insn {b[-1].get('gpu_tot_sim_insn', 0) if b else 0:,}")
print(f"# fixed cycles per launch F = {FIXED:,.0f} (ideal arm, L2R pair)")
print("# '*' marks the estimate to trust for that level (see the header comment).")
