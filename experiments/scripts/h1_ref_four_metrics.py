#!/usr/bin/env python3
# Raw four metrics of hypothesis-1 reference runs (docs/h1_llama70b_decode_batch_sweep.ko.md section 8.5), one row per
# batch, using the same definitions as graphalytics_exp/scripts/four_metrics_tsv.py but for 64 KiB logs and a single arm
# (no base/ptwhop0 arm exists yet, so nothing is normalized):
#   remote PTW ratio   = remote steps / (local + remote steps) over the four chiplets' final [GMMU_STAT] blocks
#   PTW latency        = l2_mshr_full + pwq_wait + walk_svc of the final "[XLATSUM] GPU kind=64k out=PTW" row (cycles)
#                        (ptw_avg = that row's avg, which also includes the L1 probe and link stages)
#   AT latency         = avg of the final "[XLATSUM] GPU kind=64k out=ALL" row (every L1 lookup, hits included)
#   IPC                = gpu_tot_ipc (thread instructions per cycle, whole GPU)
# Usage: h1_ref_four_metrics.py <model dir name> <config> [<config> ...]   (writes <model>/REF_FOUR_METRICS.tsv, prints it)
import os
import re
import sys

MODEL = sys.argv[1]
ROOT = f"/home/user/accel-sim-docker/llm_exp/llm_hypothesis1/{MODEL}"
GMMU_KEYS = ("requests", "local steps", "remote steps", "completed walks", "avg queuing delay", "avg PTW time")
SCALARS = ("gpu_tot_ipc", "gpu_tot_sim_insn", "gpu_tot_sim_cycle")


def parse(path):
    gmmu, xlatsum, scalars = [], {}, {}
    block = None
    for line in open(path, errors="ignore"):
        if line.startswith("[GMMU_STAT] chiplet="):
            block = {}
            gmmu.append(block)
        elif line.startswith("[XLATSUM] GPU kind=64k out="):
            body = line.split("out=", 1)[1]
            xlatsum[body.split()[0]] = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", body)}
        else:
            if block is not None:
                m = re.match(r"\s+([A-Za-z][A-Za-z ]*?)\s*:\s*([\d.]+)", line)
                if m and m.group(1) in GMMU_KEYS:
                    block[m.group(1)] = float(m.group(2))
            for key in SCALARS:
                if line.startswith(key + " ="):
                    scalars[key] = float(line.split("=", 1)[1].split()[0])
    if len(gmmu) < 4 or "ALL" not in xlatsum or "PTW" not in xlatsum:
        raise SystemExit(f"incomplete final stats in {path}")
    gm = gmmu[-4:]
    local = sum(b["local steps"] for b in gm)
    remote = sum(b["remote steps"] for b in gm)
    walks = sum(b["completed walks"] for b in gm)
    at, ptw = xlatsum["ALL"], xlatsum["PTW"]
    return dict(local=local, remote=remote, rr=remote / (local + remote), walks=walks,
                at_n=at["n"], at=at["avg"], ptw_n=ptw["n"], ptw_avg=ptw["avg"],
                resv=ptw["l2_mshr_full"], pwq=ptw["pwq_wait"], walk=ptw["walk_svc"],
                ptw=ptw["l2_mshr_full"] + ptw["pwq_wait"] + ptw["walk_svc"],
                gmmu_q=sum(b["avg queuing delay"] * b["completed walks"] for b in gm) / walks,
                gmmu_t=sum(b["avg PTW time"] * b["completed walks"] for b in gm) / walks,
                ipc=scalars["gpu_tot_ipc"], insn=scalars["gpu_tot_sim_insn"], cycles=scalars["gpu_tot_sim_cycle"])


header = ["config", "batch", "remote_ratio", "local_steps", "remote_steps", "ptw_walks_n", "ptw_latency_cycles",
          "l2_mshr_full", "pwq_wait", "walk_svc", "ptw_request_avg_incl_L1", "gmmu_queue_wavg", "gmmu_walk_wavg",
          "at_lookups_n", "at_avg_cycles", "ipc", "cycles", "thread_insn"]
rows = []
for cfg in sys.argv[2:]:
    log = f"{ROOT}/{cfg}/sim/ref_mcm/run.log"
    ex = f"{ROOT}/{cfg}/sim/ref_mcm/exit_code.txt"
    if not (os.path.exists(ex) and open(ex).read().startswith("EXIT=0")):
        print(f"# {cfg}: no clean ref run ({open(ex).read().strip() if os.path.exists(ex) else 'no exit_code'})", file=sys.stderr)
        continue
    v = parse(log)
    b = int(re.search(r"_b(\d+)_", cfg).group(1))
    rows.append([cfg, b, f"{v['rr']:.4f}", int(v["local"]), int(v["remote"]), int(v["ptw_n"]), f"{v['ptw']:.1f}",
                 f"{v['resv']:.1f}", f"{v['pwq']:.1f}", f"{v['walk']:.1f}", f"{v['ptw_avg']:.1f}", f"{v['gmmu_q']:.1f}",
                 f"{v['gmmu_t']:.1f}", int(v["at_n"]), f"{v['at']:.1f}", f"{v['ipc']:.1f}", int(v["cycles"]), int(v["insn"])])
out = f"{ROOT}/REF_FOUR_METRICS.tsv"
with open(out, "w") as f:
    f.write("\t".join(header) + "\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print("\t".join(header))
for r in rows:
    print("\t".join(str(x) for x in r))
