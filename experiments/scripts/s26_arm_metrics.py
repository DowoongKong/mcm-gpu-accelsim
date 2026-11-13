#!/usr/bin/env python3
# Final-total four metrics of individual simulation runs (any arm), one row per run.log, same definitions as
# h1_slice_four_metrics.py (unique L2 TLB misses, [XLATSUM] out=ALL / out=PTW, GMMU steps, mcm_remote/local), taken from
# the last per-kernel cumulative block (= whole run). Usage: s26_arm_metrics.py <kind 4k|64k> <label>=<run.log> [...]
import re, sys
STAGES = ["l2_mshr_full", "pwq_wait", "walk_svc"]

def final_block(path, kind):
    cur = None; last = None
    for line in open(path, errors="ignore"):
        if line.startswith("kernel_launch_uid ="):
            if cur and "gpgpu_n_tot_w_icount" in cur and "ALL" in cur["x"]: last = cur
            cur = {"l2": [], "l1h": 0, "l1m": 0, "gm": [], "x": {}}; continue
        if cur is None: continue
        if line.startswith(f"[L2_TLB_{kind}_STAT]"):
            h = re.search(r"\bhits=(\d+)", line); m = re.search(r"misses=(\d+)", line); cur["l2"].append((int(h.group(1)) if h else 0, int(m.group(1))))
        elif line.startswith(f"[L1_TLB_{kind}_STAT] SM="):
            cur["l1h"] += int(re.search(r"\bhits=(\d+)", line).group(1)); cur["l1m"] += int(re.search(r"misses=(\d+)", line).group(1))
        elif line.startswith("[GMMU_STAT] chiplet="): cur["gm"].append({})
        elif line.startswith(f"[XLATSUM] GPU kind={kind} out="):
            body = line.split("out=", 1)[1]; cur["x"][body.split()[0]] = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", body)}
        else:
            s = line.strip()
            if cur["gm"]:
                m = re.match(r"(local steps|remote steps)\s*:\s*(\d+)", s)
                if m: cur["gm"][-1][m.group(1)] = int(m.group(2))
            for k in ("gpgpu_n_tot_w_icount", "gpu_tot_sim_cycle", "gpu_tot_sim_insn", "mcm_remote_accesses", "mcm_local_accesses"):
                if s.startswith(k + " =") or s.startswith(k + "="): cur[k] = int(float(re.split(r"=\s*", s, 1)[1].split()[0]))
    if cur and "gpgpu_n_tot_w_icount" in cur and "ALL" in cur["x"]: last = cur
    return last

def metrics(b):
    w = b["gpgpu_n_tot_w_icount"]; cyc = b["gpu_tot_sim_cycle"]; insn = b["gpu_tot_sim_insn"]
    miss = sum(m for h, m in b["l2"][-4:]); l2h = sum(h for h, m in b["l2"][-4:])
    ls = sum(g.get("local steps", 0) for g in b["gm"][-4:]); rs = sum(g.get("remote steps", 0) for g in b["gm"][-4:])
    A = b["x"]["ALL"]; Pw = b["x"].get("PTW", {})
    rem = b.get("mcm_remote_accesses", 0); loc = b.get("mcm_local_accesses", 0)
    return dict(cyc=cyc, ipc=insn / cyc, w=w, miss=miss, mpki=miss / (w / 1000), l1hit=100 * b["l1h"] / max(1, b["l1h"] + b["l1m"]),
                l2hit=100 * l2h / max(1, l2h + miss), remote=rs / max(1, ls + rs), ptw=sum(Pw.get(s, 0.0) for s in STAGES),
                mshr=Pw.get("l2_mshr_full", 0.0), pwq=Pw.get("pwq_wait", 0.0), walk=Pw.get("walk_svc", 0.0), ptw_incl=Pw.get("avg", 0.0),
                at=A.get("avg", 0.0), lookups=A.get("n", 0.0), data_rem=100 * rem / max(1, rem + loc))

kind = sys.argv[1]
print("label\tcycles\tIPC\twarp_insn\tunique_miss\tMPKI\tL1hit%\tL2hit%\tremote_ratio\tPTW_lat\tmshr_full\tpwq_wait\twalk_svc\tPTW_incl_L1\tAT\tlookups\tdata_remote%")
for spec in sys.argv[2:]:
    label, path = spec.split("=", 1); m = metrics(final_block(path, kind))
    print(f"{label}\t{m['cyc']:,}\t{m['ipc']:.1f}\t{m['w']:,}\t{m['miss']:,}\t{m['mpki']:.2f}\t{m['l1hit']:.1f}\t{m['l2hit']:.1f}\t{m['remote']:.3f}\t{m['ptw']:.0f}\t{m['mshr']:.0f}\t{m['pwq']:.0f}\t{m['walk']:.0f}\t{m['ptw_incl']:.0f}\t{m['at']:.0f}\t{m['lookups']:,.0f}\t{m['data_rem']:.1f}")
