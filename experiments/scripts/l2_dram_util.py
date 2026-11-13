# L2 and DRAM utilization of a finished run, from its run.log final (cumulative) dump
# (docs/s13_l2_bw_codepath.ko.md sections 1-3).
# Peaks for this MCM config (derived from the code, see the doc):
#   L2:   256 sub-partitions x 1 sector access per L2 cycle (cache_cycle() serves one queue head per call)
#   DRAM: 128 channels x 32 B (BL16 x 2 B bus) per tCCD = 2 DRAM cycles at 1 GHz = 16 GB/s per channel
# Usage: l2_dram_util.py <run.log> [<run.log> ...]
import re
import sys

CORE_HZ = 1.395e9
N_SLICE = 256
DRAM_PEAK = 128 * 32 / (2 / 1e9)
KEYS = ["gpu_tot_sim_cycle", "gpu_tot_sim_insn", "L2_total_cache_accesses", "L2_total_cache_misses",
        "L2_BW_total", "averagemflatency", "avg_icnt2mem_latency", "avg_icnt2sh_latency",
        "gpu_stall_dramfull", "gpu_stall_icnt2sh"]

for path in sys.argv[1:]:
    last = {}; dram = []
    for line in open(path, errors="ignore"):
        s = line.strip()
        for k in KEYS:
            if s.startswith(k + " ") or s.startswith(k + "="):
                try: last[k] = float(re.split(r"=\s*", s, 1)[1].split()[0])
                except ValueError: pass
        if line.startswith("DRAM[0]:"): dram = []
        m = re.search(r" n_rd=(\d+) .*n_write=(\d+) n_wr_bk=(\d+)", line)
        if m: dram.append(sum(int(x) for x in m.groups()))
    cyc = last["gpu_tot_sim_cycle"]; t = cyc / CORE_HZ
    l2 = last["L2_total_cache_accesses"]; dram_B = sum(dram) * 32
    print(f"== {path}")
    print(f"  cycles {cyc:,.0f} ({t * 1e3:.3f} ms), thread insn {last['gpu_tot_sim_insn']:,.0f}")
    print(f"  L2 sector accesses {l2:,.0f} = {l2 / cyc:.2f}/cycle of {N_SLICE} ({100 * l2 / cyc / N_SLICE:.2f}%), "
          f"miss {100 * last.get('L2_total_cache_misses', 0) / l2:.1f}%")
    print(f"  L2_BW_total (log) {last['L2_BW_total']:.2f} GB/s vs L2 accesses x 32 B / time {l2 * 32 / t / 1e9:.2f} GB/s")
    print(f"  DRAM {dram_B / t / 1e9:.1f} GB/s of {DRAM_PEAK / 1e9:.0f} ({100 * dram_B / t / DRAM_PEAK:.2f}%), channels {len(dram)}")
    print(f"  latency: mem {last['averagemflatency']:.0f}, icnt2mem {last['avg_icnt2mem_latency']:.0f}, "
          f"icnt2sh {last['avg_icnt2sh_latency']:.0f}")
    print(f"  slice input full: {100 * last['gpu_stall_dramfull'] / cyc / N_SLICE:.2f}% of slice-cycles, "
          f"stall_icnt2sh {last['gpu_stall_icnt2sh']:,.0f}")
