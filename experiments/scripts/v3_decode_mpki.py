# Per-kernel footprint pressure of the V3 decode run (S=256, layers 0/15/31, attention and FFN windows).
# MPKI follows graphalytics_exp/scripts/cugraph_mst/mst_ref_metrics.py (2026-09-13/15 convention): unique (non-merged)
# L2 TLB misses summed over the 4 chiplets from [L2_TLB_4k_STAT], per 1000 warp instructions, GPU-wide.
# The simulator dumps these stats after every kernel (cumulative), so per-kernel values are differences.
# Two denominators are printed: the simulator's gpgpu_n_tot_w_icount (inflated by the LDGSTS double count) and the
# .traceg instruction lines recomputed by scripts/v3_check_sim_per_kernel.py (its log is read here).
# Usage: v3_decode_mpki.py <run dir> <per-kernel check log> <kernel map tsv>
import re
import sys

run_dir, check_log, map_tsv = sys.argv[1:4]
modules = {}
for ln in open(map_tsv).read().splitlines()[1:]:
    f = ln.split("\t")
    modules[int(f[0])] = (f[3], f[4], f[6][:60])
traceg = {int(m[0]): int(m[1]) for m in re.findall(r"kernel (\d+): warp traceg=(\d+)", open(check_log).read())}

kernels, l2_dumps, warp_dumps, cur = [], [], [], []
for line in open(f"{run_dir}/run.log", errors="replace"):
    if line.startswith("Header info loaded for kernel command"):
        kernels.append(int(re.search(r"kernel-(\d+)-ctx", line).group(1)))
    elif line.startswith("[L2_TLB_4k_STAT]"):
        m = re.search(r"hits=(\d+) misses=(\d+) mshr_hits=(\d+)", line)
        cur.append(tuple(int(x) for x in m.groups()))
        if len(cur) == 4:
            l2_dumps.append(tuple(sum(r[i] for r in cur) for i in range(3)))
            cur = []
    elif line.startswith("gpgpu_n_tot_w_icount ="):
        warp_dumps.append(int(line.split("=")[1]))
if not len(kernels) == len(l2_dumps) == len(warp_dumps):
    sys.exit(f"cannot pair: {len(kernels)} kernels, {len(l2_dumps)} L2 dumps, {len(warp_dumps)} warp dumps")

prev_l2, prev_w = (0, 0, 0), 0
groups = {}
print("kernel\tlayer\tmodule\twarp_traceg\tunique_L2_miss\tMPKI(traceg)\tMPKI(sim)\tname")
for kid, l2, w in zip(kernels, l2_dumps, warp_dumps):
    miss = l2[1] - prev_l2[1]
    sim_w = w - prev_w
    prev_l2, prev_w = l2, w
    layer, module, name = modules[kid]
    tg = traceg[kid]
    print(f"{kid}\t{layer}\t{module}\t{tg}\t{miss}\t{miss / (tg / 1000):.2f}\t{miss / (sim_w / 1000):.2f}\t{name}")
    g = groups.setdefault(module, [0, 0, 0])
    g[0] += miss
    g[1] += tg
    g[2] += sim_w
tot = [sum(g[i] for g in groups.values()) for i in range(3)]
for module, g in list(groups.items()) + [("total", tot)]:
    print(f"{module}: unique L2 misses {g[0]} | warp traceg {g[1]} | MPKI traceg {g[0] / (g[1] / 1000):.2f} "
          f"| MPKI sim {g[0] / (g[2] / 1000):.2f}")
print(f"final dump: hits {l2_dumps[-1][0]} misses {l2_dumps[-1][1]} mshr_hits {l2_dumps[-1][2]}")
