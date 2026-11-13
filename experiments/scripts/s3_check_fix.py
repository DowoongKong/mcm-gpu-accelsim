# Step 3: check the LDGSTS double-count fix with an A/B run of the same kernelslist.
# old = unmodified pcc_perf_tree, new = llm_sim_tree with patch 29. Expected instruction counts are recomputed
# from the .traceg files. Criteria S3-4..S3-7 are fixed in docs/s3_sim_ldgsts_double_count_fix.ko.md section 6.
# Usage: s3_check_fix.py <old run dir> <new run dir> <traces dir> <kernelslist> [<reference run dir>]
# The optional reference run (V3 full run) is compared as information only, not as a criterion.
import hashlib
import lzma
import os
import re
import sys

old_dir, new_dir, traces_dir, kernelslist = sys.argv[1:5]
ref_dir = sys.argv[5] if len(sys.argv) > 5 else None
WARP_SIZE = 32
COUNT_KEYS = {"gpu_sim_insn", "gpu_tot_sim_insn", "gpgpu_n_tot_w_icount", "gpgpu_n_tot_thrd_icount"}
HOST_TIME_KEYS = {"gpu_total_sim_rate", "gpgpu_simulation_rate", "gpgpu_simulation_time"}
inst_re = re.compile(r"^[0-9a-f]{4,} ([0-9a-f]{8}) (\d+) ")

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


def load(run_dir):
    log = open(os.path.join(run_dir, "run.log"), errors="replace").read()
    exit_line = open(os.path.join(run_dir, "exit_code.txt")).read().strip()
    headers = [m.start() for m in re.finditer(r"^Header info loaded for kernel command", log, re.M)]
    kids = re.findall(r"^Header info loaded for kernel command : \S*?kernel-(\d+)-ctx", log, re.M)
    blocks = {}
    for i, kid in enumerate(kids):
        # A kernel's stat block starts at its "kernel_name =" line and ends before the next header (or EOF).
        end = headers[i + 1] if i + 1 < len(headers) else len(log)
        start = log.find("\nkernel_name = ", headers[i], end)
        blocks[int(kid)] = log[start + 1:end].splitlines() if start >= 0 else []
    maps = [f for f in os.listdir(run_dir) if f.endswith(".map")]
    map_md5 = hashlib.md5(open(os.path.join(run_dir, maps[0]), "rb").read()).hexdigest() if maps else None
    return log, exit_line, [int(k) for k in kids], blocks, map_md5


def key_of(line):
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\[\]\.]*)\s*=", line)
    return m.group(1) if m else None


def value(block, key):
    for line in block:
        if key_of(line) == key:
            return int(line.split("=", 1)[1].split()[0])
    return None


expected_kernels = [int(re.search(r"kernel-(\d+)-", ln).group(1)) for ln in open(kernelslist).read().splitlines()
                    if not ln.startswith("MemcpyHtoD,")]
n_copies = sum(1 for ln in open(kernelslist).read().splitlines() if ln.startswith("MemcpyHtoD,"))

counts = {}
for kid in expected_kernels:
    path = next(os.path.join(traces_dir, f) for f in os.listdir(traces_dir)
                if f.startswith(f"kernel-{kid}-") and f.endswith(".traceg.xz"))
    lines = active = lanes = active_lanes = 0
    with lzma.open(path, "rt", errors="replace") as f:
        for line in f:
            m = inst_re.match(line)
            if not m:
                continue
            n = bin(int(m.group(1), 16)).count("1")
            lines += 1
            lanes += n
            if line.split()[3 + int(m.group(2))].startswith("LDGSTS") and n:
                active += 1
                active_lanes += n
    counts[kid] = (lines, active, lanes, active_lanes)

runs = {}
for name, d in (("old", old_dir), ("new", new_dir)):
    log, exit_line, kids, blocks, map_md5 = load(d)
    runs[name] = (kids, blocks, map_md5)
    check(f"S3-4 {name}: exit 0, {n_copies} copies, kernels {expected_kernels[0]}-{expected_kernels[-1]}",
          exit_line.startswith("EXIT=0") and log.count("launching memcpy command") == n_copies
          and kids == expected_kernels and all(blocks[k] for k in kids)
          and len(re.findall(r"^kernel_name = ", log, re.M)) == len(expected_kernels),
          f"{exit_line} | copies {log.count('launching memcpy command')} | kernels {kids}")


def per_kernel_deltas(blocks, kids, key):
    prev = 0
    out = {}
    for kid in kids:
        cum = value(blocks[kid], key)
        out[kid] = cum - prev
        prev = cum
    return out


for name in ("new", "old"):
    kids, blocks, _ = runs[name]
    warp = per_kernel_deltas(blocks, kids, "gpgpu_n_tot_w_icount")
    thread = per_kernel_deltas(blocks, kids, "gpu_tot_sim_insn")
    thrd = per_kernel_deltas(blocks, kids, "gpgpu_n_tot_thrd_icount")
    ok = 0
    for kid in kids:
        lines, active, lanes, active_lanes = counts[kid]
        if name == "new":
            exp_w, exp_t = lines, lanes
        else:
            exp_w, exp_t = lines + active, lanes + active_lanes
        match = warp[kid] == exp_w and thread[kid] == exp_t and thrd[kid] == WARP_SIZE * exp_w
        ok += match
        print(f"  {name} kernel {kid}: warp {warp[kid]} (expected {exp_w}) thread {thread[kid]} (expected {exp_t}) "
              f"thrd_icount {thrd[kid]} (expected {WARP_SIZE * exp_w}) active_LDGSTS {active}",
              "MATCH" if match else "DIFF")
    label = "S3-5 new: warp = .traceg lines, thread = lanes" if name == "new" else \
        "S3-6 old: warp = lines + active LDGSTS, thread = lanes + active LDGSTS lanes"
    check(label, ok == len(kids), f"{ok}/{len(kids)}")


def compare_blocks(a_blocks, b_blocks, kids, allowed):
    other = {}
    allowed_seen = set()
    for kid in kids:
        a, b = a_blocks[kid], b_blocks[kid]
        if len(a) != len(b):
            other.setdefault(kid, []).append(f"line count {len(a)} vs {len(b)}")
            continue
        for la, lb in zip(a, b):
            if la == lb:
                continue
            key = key_of(la)
            if key in allowed and key == key_of(lb):
                allowed_seen.add(key)
            else:
                other.setdefault(kid, []).append(f"{la.strip()[:120]} || {lb.strip()[:120]}")
    return other, allowed_seen


old_kids, old_blocks, old_map = runs["old"]
new_kids, new_blocks, new_map = runs["new"]
other, seen = compare_blocks(old_blocks, new_blocks, old_kids, COUNT_KEYS | HOST_TIME_KEYS)
for kid, diffs in other.items():
    print(f"  S3-7 kernel {kid}: {len(diffs)} lines outside the allowed list differ, first: {diffs[:3]}")
check("S3-7 old vs new stat blocks differ only in allowed keys", not other,
      f"allowed keys that differed: {sorted(seen)}")
check("S3-7 placement map identical", old_map is not None and old_map == new_map, f"{old_map} vs {new_map}")

if ref_dir:
    log, exit_line, kids, ref_blocks, _ = load(ref_dir)
    ref_other, ref_seen = compare_blocks(ref_blocks, old_blocks, old_kids, HOST_TIME_KEYS)
    for kid, diffs in ref_other.items():
        print(f"  INFO reference kernel {kid}: {len(diffs)} lines differ, first: {diffs[:3]}")
    print("INFO reference (V3 full run) vs old truncated run, same kernels:",
          "identical except host time" if not ref_other else "DIFFERENT", f"| host-time keys differed: {sorted(ref_seen)}")

print("S3 FIX", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
