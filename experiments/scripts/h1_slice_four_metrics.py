#!/usr/bin/env python3
# Four metrics of ONE kernel range (a "slice") of a finished 3-arm distributed run, in the hypothesis-1 format
# (docs/h1_llama70b_decode_batch_sweep.ko.md 8.3/8.5: ref raw values) plus the base/ptwhop0 normalization of
# docs/s21_llm_metrics_by_phase.ko.md. Every number is the difference of the per-kernel cumulative statistics
# (blocks matched by launch order, as in s21_metrics_by_phase.py), so it is exactly the slice's contribution:
#   MPKI = unique L2 TLB misses (sum of the [L2_TLB_<kind>_STAT] misses over the chiplets) / warp kilo-insn, ref arm
#   remote PTW ratio = GMMU remote steps / (local + remote), PTW latency = l2_mshr_full + pwq_wait + walk_svc of
#   [XLATSUM] out=PTW (slice-weighted), PTW avg incl. L1 = out=PTW avg, AT = out=ALL avg, IPC = thread insn / cycles,
#   L1/L2 TLB hit = hits/(hits+misses) summed over SMs/chiplets, first-touched pages = [REF_DEFER] new_pages (ref).
# Usage: h1_slice_four_metrics.py <exp dir> <kind 4k|64k> <first>-<last> <tokens in the slice> [<out.tsv>]
import re, sys

OUTS = ["ALL", "PTW"]; STAGES = ["l2_mshr_full", "pwq_wait", "walk_svc"]; ARMS = ("ref_mcm", "base_mcm", "ptwhop0_mcm")

def blocks(path, kind):
    out = []; cur = None
    for line in open(path, errors="ignore"):
        if line.startswith("kernel_launch_uid ="):
            cur = {"l2": [], "l1h": 0, "l1m": 0, "gm": [], "x": {}, "newp": 0}; out.append(cur); continue
        if cur is None: continue
        if line.startswith(f"[L2_TLB_{kind}_STAT]"):
            h = re.search(r"\bhits=(\d+)", line); m = re.search(r"misses=(\d+)", line)
            cur["l2"].append((int(h.group(1)) if h else 0, int(m.group(1))))
        elif line.startswith(f"[L1_TLB_{kind}_STAT] SM="):
            cur["l1h"] += int(re.search(r"\bhits=(\d+)", line).group(1)); cur["l1m"] += int(re.search(r"misses=(\d+)", line).group(1))
        elif line.startswith("[GMMU_STAT] chiplet="):
            cur["gm"].append({})
        elif line.startswith(f"[XLATSUM] GPU kind={kind} out="):
            body = line.split("out=", 1)[1]
            cur["x"][body.split()[0]] = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", body)}
        elif line.startswith("[REF_DEFER]"):
            m = re.search(r"new_pages=(\d+)", line)
            if m: cur["newp"] = int(m.group(1))
        else:
            s = line.strip()
            if cur["gm"]:
                m = re.match(r"(local steps|remote steps)\s*:\s*(\d+)", s)
                if m: cur["gm"][-1][m.group(1)] = int(m.group(2))
            for k in ("gpgpu_n_tot_w_icount", "gpu_tot_sim_cycle", "gpu_tot_sim_insn", "mcm_remote_accesses", "mcm_local_accesses"):
                if s.startswith(k + " =") or s.startswith(k + "="):
                    cur[k] = int(float(re.split(r"=\s*", s, 1)[1].split()[0]))
    return [b for b in out if "gpgpu_n_tot_w_icount" in b and len(b["l2"]) >= 4 and "ALL" in b["x"]]

def cum(b):
    c = {"w": b["gpgpu_n_tot_w_icount"], "cyc": b["gpu_tot_sim_cycle"], "insn": b["gpu_tot_sim_insn"],
         "l2hit": sum(h for h, m in b["l2"][-4:]), "miss": sum(m for h, m in b["l2"][-4:]), "l1hit": b["l1h"], "l1miss": b["l1m"],
         "rem": b.get("mcm_remote_accesses", 0), "loc": b.get("mcm_local_accesses", 0), "newp": b["newp"],
         "lsteps": sum(g.get("local steps", 0) for g in b["gm"][-4:]), "rsteps": sum(g.get("remote steps", 0) for g in b["gm"][-4:])}
    for o in OUTS:
        row = b["x"].get(o, {}); n = row.get("n", 0.0)
        c[f"n_{o}"] = n; c[f"sum_{o}"] = row.get("avg", 0.0) * n
        for st in STAGES: c[f"{o}_{st}"] = row.get(st, 0.0) * n
    return c

def slice_totals(cs, a, b):
    zero = {k: 0 for k in cs[0]}; t = dict(zero)
    for k in range(a, b + 1):
        prev = cs[k - 2] if k > 1 else zero
        for key in t: t[key] += cs[k - 1][key] - prev[key]
    return t

def metrics(t, tokens):
    nall = t["n_ALL"] or 1; npw = t["n_PTW"] or 1; steps = t["lsteps"] + t["rsteps"]; acc = t["rem"] + t["loc"]
    st = {s: t[f"PTW_{s}"] / npw for s in STAGES}
    return dict(w=t["w"], insn=t["insn"], cyc=t["cyc"], ipc=t["insn"] / t["cyc"], miss=t["miss"], mpki=t["miss"] / (t["w"] / 1000),
                l1hit=100 * t["l1hit"] / max(1, t["l1hit"] + t["l1miss"]), l2hit=100 * t["l2hit"] / max(1, t["l2hit"] + t["miss"]),
                lookups=nall, at=t["sum_ALL"] / nall, ptw_n=npw, ptw=sum(st.values()), ptw_incl_l1=t["sum_PTW"] / npw, **st,
                remote=t["rsteps"] / steps if steps else 0.0, lsteps=t["lsteps"], rsteps=t["rsteps"],
                data_rem=100 * t["rem"] / acc if acc else 0.0, newp=t["newp"], tok_per_mcyc=tokens / (t["cyc"] / 1e6),
                miss_per_tok=t["miss"] / tokens)

exp, kind, rng_, tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); outp = sys.argv[5] if len(sys.argv) > 5 else None
a, b = (int(v) for v in rng_.split("-"))
M = {}
for arm in ARMS:
    cs = [cum(x) for x in blocks(f"{exp}/{arm}/run.log", kind)]
    assert len(cs) >= b, f"{arm}: only {len(cs)} kernel blocks, need {b}"
    M[arm] = metrics(slice_totals(cs, a, b), tokens)
r, bs, h = M["ref_mcm"], M["base_mcm"], M["ptwhop0_mcm"]
print(f"slice kernels {a}-{b} of {exp} (kind {kind}, tokens {tokens})")
print(f"[ref]  warp insn {r['w']:,}  thread insn {r['insn']:,}  cycles {r['cyc']:,}  IPC {r['ipc']:.1f}  L1 TLB hit {r['l1hit']:.1f} %  L2 TLB hit {r['l2hit']:.1f} %  "
      f"unique L2 miss {r['miss']:,}  MPKI {r['mpki']:.2f}  miss/token {r['miss_per_tok']:.0f}  first-touched pages {r['newp']:,} ({r['newp']*64/1024:.0f} MB if 64K)")
print(f"[ref]  remote PTW ratio {r['remote']:.3f} (local {r['lsteps']:,} remote {r['rsteps']:,})  PTW latency {r['ptw']:.0f} = {r['l2_mshr_full']:.0f} + {r['pwq_wait']:.0f} + {r['walk_svc']:.0f}  "
      f"PTW avg incl. L1 {r['ptw_incl_l1']:.0f}  AT {r['at']:.0f} (lookups {r['lookups']:,.0f})  tokens/Mcycle {r['tok_per_mcyc']:.3f}")
print(f"[norm] IPC base {bs['ipc']:.2f} -> hop0 {h['ipc']:.2f} = {h['ipc']/bs['ipc']:.4f} | AT {bs['at']:.1f} -> {h['at']:.1f} = {h['at']/bs['at']:.4f} | "
      f"PTW {bs['ptw']:.1f} -> {h['ptw']:.1f} = {h['ptw']/bs['ptw']:.4f} | walk remote % base {100*bs['remote']:.1f} hop0 {100*h['remote']:.1f} | data remote % base {bs['data_rem']:.1f} | cycles base {bs['cyc']:,} hop0 {h['cyc']:,}")
if outp:
    hdr = ["slice", "tokens", "warp_insn", "ref_cycles", "ref_IPC", "ref_L1hit%", "ref_L2hit%", "ref_unique_miss", "ref_MPKI", "ref_miss_per_token", "ref_first_touched_pages",
           "ref_remote_ratio", "ref_PTW_lat", "ref_l2_mshr_full", "ref_pwq_wait", "ref_walk_svc", "ref_PTW_incl_L1", "ref_AT", "ref_tokens_per_Mcycle",
           "IPC_base", "IPC_hop0", "normIPC", "AT_base", "AT_hop0", "normAT", "PTW_base", "PTW_hop0", "normPTW", "walk_remote%_base", "walk_remote%_hop0", "data_remote%_base", "cycles_base", "cycles_hop0"]
    row = [f"k{a}-{b}", tokens, r["w"], r["cyc"], f"{r['ipc']:.2f}", f"{r['l1hit']:.1f}", f"{r['l2hit']:.1f}", r["miss"], f"{r['mpki']:.2f}", f"{r['miss_per_tok']:.1f}", r["newp"],
           f"{r['remote']:.4f}", f"{r['ptw']:.1f}", f"{r['l2_mshr_full']:.1f}", f"{r['pwq_wait']:.1f}", f"{r['walk_svc']:.1f}", f"{r['ptw_incl_l1']:.1f}", f"{r['at']:.1f}", f"{r['tok_per_mcyc']:.4f}",
           f"{bs['ipc']:.2f}", f"{h['ipc']:.2f}", f"{h['ipc']/bs['ipc']:.4f}", f"{bs['at']:.1f}", f"{h['at']:.1f}", f"{h['at']/bs['at']:.4f}", f"{bs['ptw']:.1f}", f"{h['ptw']:.1f}", f"{h['ptw']/bs['ptw']:.4f}",
           f"{100*bs['remote']:.1f}", f"{100*h['remote']:.1f}", f"{bs['data_rem']:.1f}", bs["cyc"], h["cyc"]]
    with open(outp, "w") as f: f.write("\t".join(hdr) + "\n" + "\t".join(str(v) for v in row) + "\n")
    print("wrote", outp)
