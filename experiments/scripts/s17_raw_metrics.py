#!/usr/bin/env python3
# Step 17 (docs/s17_llama70b_prep.ko.md section 4.12): raw (not normalized) four metrics of one arm of an L1seq run --
# address-translation (AT) latency, PTW latency, IPC, L2 TLB MPKI -- with their components, for the whole run and per
# piece of the L1seq list. Every number is a difference of the cumulative statistics the simulator prints after each
# kernel (blocks matched by order). Conventions (feedback_four_metrics_reporting): AT = [XLATSUM] out=ALL avg;
# PTW latency = l2_mshr_full + pwq_wait + walk_svc of out=PTW; MPKI = unique L2 TLB misses / warp kilo-instructions,
# GPU-wide; remote ratio = remote walk steps / all walk steps ([GMMU_STAT]).
# Usage: s17_raw_metrics.py <page kind: 4k|64k> <pieces spec "name=1-9,20-27;..."> <arm dir> [<arm dir> ...]
import re
import sys

kind, pieces_spec, arm_dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
STAGES = ["l1_probe", "l1_mshr_full", "merge_wait", "link_l1l2", "l2_probe", "l2_mshr_full", "pwq_admit", "pwq_wait",
          "walk_svc", "fill_return", "rest_merged", "reissue", "replay_wait"]
OUTS = ["L1_HIT", "L1_MERGE", "L2_HIT", "L2_MERGE", "PTW", "ALL"]


def blocks(path):
    out = []; cur = None
    for line in open(path, errors="ignore"):
        if line.startswith("kernel_launch_uid ="):
            cur = {"l2miss": [], "gm": [], "x": {}}; out.append(cur); continue
        if cur is None: continue
        if line.startswith(f"[L2_TLB_{kind}_STAT]"):
            cur["l2miss"].append(int(re.search(r"misses=(\d+)", line).group(1)))
        elif line.startswith("[GMMU_STAT] chiplet="):
            cur["gm"].append({})
        elif line.startswith(f"[XLATSUM] GPU kind={kind} out="):
            body = line.split("out=", 1)[1]
            cur["x"][body.split()[0]] = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", body)}
        else:
            s = line.strip()
            if cur["gm"]:
                m = re.match(r"(local steps|remote steps|completed walks|faults)\s*:\s*(\d+)", s)
                if m: cur["gm"][-1][m.group(1)] = int(m.group(2))
            for k in ("gpgpu_n_tot_w_icount", "gpu_tot_sim_cycle", "gpu_tot_sim_insn", "mcm_remote_accesses", "mcm_local_accesses"):
                if s.startswith(k + " =") or s.startswith(k + "="):
                    cur[k] = int(float(re.split(r"=\s*", s, 1)[1].split()[0]))
    return [b for b in out if "gpgpu_n_tot_w_icount" in b and len(b["l2miss"]) >= 4 and "ALL" in b["x"]]


def cum(b):
    """Cumulative totals of one block: counts, and latency sums (avg x n) so that pieces can be differenced."""
    c = {"w": b["gpgpu_n_tot_w_icount"], "cyc": b["gpu_tot_sim_cycle"], "insn": b["gpu_tot_sim_insn"],
         "miss": sum(b["l2miss"][-4:]), "rem": b.get("mcm_remote_accesses", 0), "loc": b.get("mcm_local_accesses", 0)}
    for k in ("local steps", "remote steps", "completed walks", "faults"):
        c[k] = sum(g.get(k, 0) for g in b["gm"][-4:])
    for o in OUTS:
        row = b["x"].get(o, {})
        n = row.get("n", 0.0); c[f"n_{o}"] = n; c[f"sum_{o}"] = row.get("avg", 0.0) * n
        for st in STAGES: c[f"{o}_{st}"] = row.get(st, 0.0) * n
    return c


def rng(spec):
    ks = []
    for part in spec.split(","):
        a, b = part.split("-"); ks += list(range(int(a), int(b) + 1))
    return ks


pieces = [p.split("=") for p in pieces_spec.split(";") if p]
for d in arm_dirs:
    bl = blocks(f"{d}/run.log"); cs = [cum(b) for b in bl]
    zero = {k: 0 for k in cs[0]}
    print(f"== {d}  (kernel blocks {len(cs)})")
    hdr = ["piece", "warp_insn", "cycles", "IPC", "L2TLB_miss", "MPKI", "AT_n", "AT_avg", "PTW_n", "PTW_lat", "=l2_mshr_full", "+pwq_wait",
           "+walk_svc", "walk_remote%", "data_remote%", "faults"]
    print("\t".join(hdr))
    for name, spec in pieces:
        t = dict(zero)
        for k in rng(spec):
            if k > len(cs): continue
            prev = cs[k - 2] if k > 1 else zero
            for key in t: t[key] += cs[k - 1][key] - prev[key]
        npw = t["n_PTW"] or 1; nall = t["n_ALL"] or 1
        steps = t["local steps"] + t["remote steps"]; acc = t["rem"] + t["loc"]
        ptw = [t["PTW_l2_mshr_full"] / npw, t["PTW_pwq_wait"] / npw, t["PTW_walk_svc"] / npw]
        print("\t".join([name, f"{t['w']:,}", f"{t['cyc']:,}", f"{t['insn'] / t['cyc']:.2f}", f"{t['miss']:,}",
                         f"{t['miss'] / (t['w'] / 1000):.2f}", f"{int(t['n_ALL']):,}", f"{t['sum_ALL'] / nall:.1f}",
                         f"{int(t['n_PTW']):,}", f"{sum(ptw):.1f}", f"{ptw[0]:.1f}", f"{ptw[1]:.1f}", f"{ptw[2]:.1f}",
                         f"{100 * t['remote steps'] / steps if steps else 0:.1f}", f"{100 * t['rem'] / acc if acc else 0:.1f}",
                         f"{t['faults']:,}"]))
    last = cs[-1]; n = last["n_ALL"]
    print("# whole run, AT latency by outcome (n, share, avg):  " + "  ".join(
        f"{o} {int(last['n_' + o]):,} {100 * last['n_' + o] / n:.1f}% {last['sum_' + o] / (last['n_' + o] or 1):.1f}" for o in OUTS[:-1]))
    print("# whole run, AT latency by stage (cycles per translation, sum = AT avg):  " + "  ".join(
        f"{st} {last['ALL_' + st] / n:.1f}" for st in STAGES if last["ALL_" + st] / n >= 0.05))
