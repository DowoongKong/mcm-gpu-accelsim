# V3 (step 3): check the short validation simulation of the step-2 decode traces.
# Usage: v3_check_sim.py <run dir> <traces dir>. Explained in docs/v3_postprocess_sim.ko.md.
import glob
import os
import re
import sys

run_dir, traces_dir = sys.argv[1], sys.argv[2]
log = open(os.path.join(run_dir, "run.log"), errors="replace").read()
exit_line = open(os.path.join(run_dir, "exit_code.txt")).read().strip()
kernels = [ln for ln in open(os.path.join(traces_dir, "kernelslist.g")).read().splitlines()
           if not ln.startswith("MemcpyHtoD,")]
# The simulator executes .traceg.xz, where post-processing merged each two-line LDGSTS into one line
# (post-traces-processing.cpp:378-386; audited in scripts/v3_ldgsts_audit.py). Expected executed instructions are
# therefore the .traceg instruction lines, not the tracer stats total.
import lzma
inst_re = re.compile(r"^[0-9a-f]{4,} [0-9a-f]{8} \d+ ")
traceg_total = 0
for k in kernels:
    with lzma.open(os.path.join(traces_dir, k), "rt", errors="replace") as f:
        traceg_total += sum(1 for line in f if inst_re.match(line))


def last_value(key):
    vals = re.findall(rf"^{key} = (\d+)", log, re.M)
    return int(vals[-1]) if vals else None


results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


check("simulator exit code 0", exit_line.startswith("EXIT=0"), exit_line)
check("no undefined instruction / command", "undefined instruction" not in log and "Undefined Command" not in log)
check("534 memcpy commands processed", log.count("launching memcpy command") == 534, log.count("launching memcpy command"))
check("30 kernels loaded", log.count("Header info loaded for kernel command") == len(kernels) == 30,
      log.count("Header info loaded for kernel command"))
check("30 kernel stat dumps", len(re.findall(r"^kernel_name = ", log, re.M)) == 30, len(re.findall(r"^kernel_name = ", log, re.M)))
w = last_value("gpgpu_n_tot_w_icount")
check("gpgpu_n_tot_w_icount == .traceg instruction lines", w == traceg_total, f"{w} vs {traceg_total}")

maps = glob.glob(os.path.join(run_dir, "*.map"))
data_pages = sum(1 for ln in open(maps[0]) if ln.startswith("D ")) if maps else None
print("gpu_tot_sim_cycle", last_value("gpu_tot_sim_cycle"), "| gpu_tot_sim_insn", last_value("gpu_tot_sim_insn"),
      "| placement data pages", data_pages)
print("V3 SIM", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
