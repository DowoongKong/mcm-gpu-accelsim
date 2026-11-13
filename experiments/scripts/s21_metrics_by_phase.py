#!/usr/bin/env python3
# Four metrics by phase for the distributed (KERNEL_PARTITION) L1seq runs: whole run, prefill / decode1 / decode2,
# and the layer-only pieces, raw and normalized (ptwhop0_mcm / base_mcm; base = 1). Every number is a difference of
# the cumulative per-kernel statistics (blocks matched by order), so a phase value is exactly the contribution of
# its kernels. Conventions (feedback_four_metrics_reporting): MPKI = unique L2 TLB misses / warp kilo-insn of the
# reference arm (first touch, GPU-wide); AT = [XLATSUM] out=ALL avg; PTW = l2_mshr_full + pwq_wait + walk_svc of
# out=PTW; IPC = thread instructions / cycles; remote = GMMU remote walk steps / all steps; data remote =
# mcm_remote / (remote + local). Phase latencies are weighted by the phase's own translation counts.
# Usage: s21_metrics_by_phase.py <out.tsv> <label>=<exp dir>=<kind 4k|64k>=<pieces "name=a-b,c-d;..."> [...]
import re
import sys

OUTS = ["ALL", "PTW"]
STAGES = ["l2_mshr_full", "pwq_wait", "walk_svc"]
ARMS = ("ref_mcm", "base_mcm", "ptwhop0_mcm")


def blocks(path, kind):
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
                m = re.match(r"(local steps|remote steps)\s*:\s*(\d+)", s)
                if m: cur["gm"][-1][m.group(1)] = int(m.group(2))
            for k in ("gpgpu_n_tot_w_icount", "gpu_tot_sim_cycle", "gpu_tot_sim_insn", "mcm_remote_accesses", "mcm_local_accesses"):
                if s.startswith(k + " =") or s.startswith(k + "="):
                    cur[k] = int(float(re.split(r"=\s*", s, 1)[1].split()[0]))
    return [b for b in out if "gpgpu_n_tot_w_icount" in b and len(b["l2miss"]) >= 4 and "ALL" in b["x"]]


def cum(b):
    c = {"w": b["gpgpu_n_tot_w_icount"], "cyc": b["gpu_tot_sim_cycle"], "insn": b["gpu_tot_sim_insn"],
         "miss": sum(b["l2miss"][-4:]), "rem": b.get("mcm_remote_accesses", 0), "loc": b.get("mcm_local_accesses", 0),
         "lsteps": sum(g.get("local steps", 0) for g in b["gm"][-4:]), "rsteps": sum(g.get("remote steps", 0) for g in b["gm"][-4:])}
    for o in OUTS:
        row = b["x"].get(o, {}); n = row.get("n", 0.0)
        c[f"n_{o}"] = n; c[f"sum_{o}"] = row.get("avg", 0.0) * n
        for st in STAGES: c[f"{o}_{st}"] = row.get(st, 0.0) * n
    return c


def rng(spec):
    ks = []
    for part in spec.split(","):
        a, b = part.split("-"); ks += list(range(int(a), int(b) + 1))
    return ks


def piece_totals(cs, spec):
    zero = {k: 0 for k in cs[0]}; t = dict(zero)
    for k in rng(spec):
        if k > len(cs): continue
        prev = cs[k - 2] if k > 1 else zero
        for key in t: t[key] += cs[k - 1][key] - prev[key]
    return t


def metrics(t):
    nall = t["n_ALL"] or 1; npw = t["n_PTW"] or 1
    steps = t["lsteps"] + t["rsteps"]; acc = t["rem"] + t["loc"]
    return dict(ipc=t["insn"] / t["cyc"] if t["cyc"] else 0.0, at=t["sum_ALL"] / nall,
                ptw=sum(t[f"PTW_{s}"] for s in STAGES) / npw, mpki=t["miss"] / (t["w"] / 1000) if t["w"] else 0.0,
                walk_rem=100 * t["rsteps"] / steps if steps else 0.0, data_rem=100 * t["rem"] / acc if acc else 0.0,
                cyc=t["cyc"], w=t["w"], miss=t["miss"])


out_path = sys.argv[1]
rows = []
hdr = ["workload", "piece", "warp_insn", "MPKI_ref", "IPC_ref", "IPC_base", "IPC_hop0", "normIPC", "AT_base", "AT_hop0", "normAT",
       "PTW_base", "PTW_hop0", "normPTW", "walk_remote%_base", "walk_remote%_hop0", "data_remote%_base", "cycles_base", "cycles_hop0"]
print("\t".join(hdr))
for spec in sys.argv[2:]:
    label, expdir, kind, pieces = spec.split("=", 3)
    cs = {a: [cum(b) for b in blocks(f"{expdir}/{a}/run.log", kind)] for a in ARMS}
    for name_spec in pieces.split(";"):
        if not name_spec: continue
        name, prange = name_spec.split(":")
        m = {a: metrics(piece_totals(cs[a], prange)) for a in ARMS}
        r, b, h = m["ref_mcm"], m["base_mcm"], m["ptwhop0_mcm"]
        row = [label, name, f"{b['w']}", f"{r['mpki']:.2f}", f"{r['ipc']:.2f}", f"{b['ipc']:.2f}", f"{h['ipc']:.2f}", f"{h['ipc'] / b['ipc']:.4f}",
               f"{b['at']:.1f}", f"{h['at']:.1f}", f"{h['at'] / b['at']:.4f}", f"{b['ptw']:.1f}", f"{h['ptw']:.1f}", f"{h['ptw'] / b['ptw']:.4f}",
               f"{b['walk_rem']:.1f}", f"{h['walk_rem']:.1f}", f"{b['data_rem']:.1f}", f"{b['cyc']}", f"{h['cyc']}"]
        rows.append(row); print("\t".join(row))
    print(f"# {label}: blocks ref/base/hop0 = {len(cs['ref_mcm'])}/{len(cs['base_mcm'])}/{len(cs['ptwhop0_mcm'])}")
with open(out_path, "w") as fh:
    fh.write("\t".join(hdr) + "\n")
    for row in rows: fh.write("\t".join(row) + "\n")
