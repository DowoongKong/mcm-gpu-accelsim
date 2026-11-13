# V3 (after simulation): check the LDGSTS double-count explanation per kernel.
# Prediction registered before the simulation ended (docs/v3_postprocess_sim.ko.md 8.3):
#   simulator warp instructions of a kernel = .traceg instruction lines + active LDGSTS lines
# where an active LDGSTS line has a non-zero lane mask (it issues memory accesses, so ldst_unit::writeback
# calls warp_inst_complete a second time). warp_inst_complete also adds the active lane count to gpu_sim_insn
# (shader.cc:2117-2119), so the thread count gains the lanes of every active LDGSTS line as well.
# All counts are recomputed here from the .traceg files and run.log.
# Usage: v3_check_sim_per_kernel.py <run dir> <traces dir>
import lzma
import os
import re
import sys

run_dir, traces_dir = sys.argv[1], sys.argv[2]
log = open(os.path.join(run_dir, "run.log"), errors="replace").read()
inst_re = re.compile(r"^[0-9a-f]{4,} ([0-9a-f]{8}) (\d+) ")

# Kernels run one after another; each kernel's stat dump follows its header load, and w_icount is cumulative.
loaded = re.findall(r"Header info loaded for kernel command : \S*?kernel-(\d+)-ctx", log)
cumulative = [int(x) for x in re.findall(r"^gpgpu_n_tot_w_icount = (\d+)", log, re.M)]
thread_cumulative = [int(x) for x in re.findall(r"^gpu_tot_sim_insn = (\d+)", log, re.M)]
if not len(loaded) == len(cumulative) == len(thread_cumulative):
    sys.exit(f"cannot pair kernels with stat dumps: {len(loaded)} headers, {len(cumulative)} / "
             f"{len(thread_cumulative)} dumps")

ok = thread_ok = 0
prev = thread_prev = 0
tot_lines = tot_active = tot_lanes = tot_active_lanes = 0
for kid, cum, thread_cum in zip(loaded, cumulative, thread_cumulative):
    sim = cum - prev
    prev = cum
    thread_sim = thread_cum - thread_prev
    thread_prev = thread_cum
    lines = active = lanes = active_lanes = 0
    path = next(os.path.join(traces_dir, f) for f in os.listdir(traces_dir)
                if f.startswith(f"kernel-{kid}-") and f.endswith(".traceg.xz"))
    with lzma.open(path, "rt", errors="replace") as f:
        for line in f:
            m = inst_re.match(line)
            if not m:
                continue
            lines += 1
            n_lanes = bin(int(m.group(1), 16)).count("1")
            lanes += n_lanes
            ndest = int(m.group(2))
            opcode = line.split()[3 + ndest]
            if opcode.startswith("LDGSTS") and n_lanes:
                active += 1
                active_lanes += n_lanes
    match = sim == lines + active
    thread_match = thread_sim == lanes + active_lanes
    ok += match
    thread_ok += thread_match
    tot_lines += lines
    tot_active += active
    tot_lanes += lanes
    tot_active_lanes += active_lanes
    print(f"kernel {kid}: warp traceg={lines} active_LDGSTS={active} predicted={lines + active} simulated={sim}",
          "MATCH" if match else "DIFF",
          f"| thread traceg={lanes} active_LDGSTS_lanes={active_lanes} predicted={lanes + active_lanes} "
          f"simulated={thread_sim}", "MATCH" if thread_match else "DIFF")
print(f"total warp: traceg={tot_lines} active_LDGSTS={tot_active} predicted={tot_lines + tot_active} "
      f"simulated={cumulative[-1]}")
print(f"total thread: traceg={tot_lanes} active_LDGSTS_lanes={tot_active_lanes} "
      f"predicted={tot_lanes + tot_active_lanes} simulated={thread_cumulative[-1]}")
print("PER-KERNEL WARP", "PASS" if ok == len(loaded) else "FAIL", f"({ok}/{len(loaded)})")
print("PER-KERNEL THREAD", "PASS" if thread_ok == len(loaded) else "FAIL", f"({thread_ok}/{len(loaded)})")
