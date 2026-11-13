# Per-kernel bandwidth of the step-14 microbenchmark (docs/s14_bw_microbench.ko.md), from the
# cumulative statistics dump the simulator prints after every kernel (deltas between dumps).
# Peaks (docs/s13_l2_bw_codepath.ko.md sections 1-2): L2 256 sector accesses per core cycle
# (11.43 TB/s at 1.395 GHz), DRAM 128 channels x 32 B per tCCD = 2 ns (2.048 TB/s).
# Usage: s14_kernel_bw.py <run.log> [<run.log> ...]
import re
import sys

CORE_HZ = 1.395e9
N_SLICE = 256
DRAM_PEAK = 128 * 32 / 2e-9
L2_PEAK = N_SLICE * 32 * CORE_HZ
NAMES = ["K1 init_big (write)", "K2 stream (read)", "K3 rowgather (read)", "K4 init_l2 (write)", "K5 l2read (read)"]
DESIGN = [256 << 20, 256 << 20, 256 << 20, 16 << 20, 1024 * (1 << 20)]   # bytes each kernel moves by design
CUM = ["L2_total_cache_accesses", "L2_total_cache_misses", "mcm_remote_accesses", "mcm_local_accesses",
       "gpu_tot_sim_insn"]


def dumps(path):
    out = []; cur = None; dram = None
    for line in open(path, errors="ignore"):
        s = line.strip()
        if line.startswith("kernel_launch_uid ="):
            cur = {"dram": []}; out.append(cur); dram = cur["dram"]; continue
        if cur is None: continue
        if s.startswith("gpu_sim_cycle ="): cur["cyc"] = int(s.split("=")[1])
        elif s.startswith("L2_BW  ="): cur["l2bw_log"] = float(s.split("=")[1].split()[0])
        else:
            for k in CUM:
                if s.startswith(k + " =") or s.startswith(k + "="):
                    cur[k] = int(float(re.split(r"=\s*", s, 1)[1].split()[0]))
        if line.startswith("DRAM[0]:"): cur["dram"] = []; dram = cur["dram"]
        m = re.search(r" n_rd=(\d+) .*n_write=(\d+) n_wr_bk=(\d+)", line)
        if m and dram is not None: dram.append(sum(int(x) for x in m.groups()))
    return [d for d in out if "cyc" in d and d["dram"]]


for path in sys.argv[1:]:
    ds = dumps(path)
    print(f"== {path}  (kernels {len(ds)})")
    if not ds:
        continue
    print(f"{'kernel':22s} {'cycles':>9s} {'design GB/s':>11s} {'L2 sect/cyc':>11s} {'L2 %peak':>8s} "
          f"{'L2_BW log':>9s} {'DRAM GB/s':>9s} {'DRAM %':>6s} {'L2 hit':>6s} {'remote':>6s} {'ch max/mean':>11s}")
    prev = None
    for i, d in enumerate(ds):
        dl = lambda k: d.get(k, 0) - (prev.get(k, 0) if prev else 0)
        ch = [a - (prev["dram"][j] if prev else 0) for j, a in enumerate(d["dram"])]
        t = d["cyc"] / CORE_HZ
        l2 = dl("L2_total_cache_accesses"); miss = dl("L2_total_cache_misses")
        rem, loc = dl("mcm_remote_accesses"), dl("mcm_local_accesses")
        dram_B = sum(ch) * 32; mean = sum(ch) / len(ch)
        name = NAMES[i] if i < len(NAMES) else f"kernel {i + 1}"
        design = DESIGN[i] / t / 1e9 if i < len(DESIGN) else 0
        print(f"{name:22s} {d['cyc']:>9,} {design:>11.1f} {l2 / d['cyc']:>11.2f} {100 * l2 / d['cyc'] / N_SLICE:>7.1f}% "
              f"{d.get('l2bw_log', 0):>9.1f} {dram_B / t / 1e9:>9.1f} {100 * dram_B / t / DRAM_PEAK:>5.1f}% "
              f"{100 * (1 - miss / l2) if l2 else 0:>5.1f}% {100 * rem / (rem + loc) if rem + loc else 0:>5.1f}% "
              f"{(max(ch) / mean) if mean else 0:>11.2f}")
        prev = d
    print(f"  thread insn {ds[-1].get('gpu_tot_sim_insn', 0):,}; peaks: L2 {L2_PEAK / 1e12:.2f} TB/s, DRAM {DRAM_PEAK / 1e12:.3f} TB/s")
