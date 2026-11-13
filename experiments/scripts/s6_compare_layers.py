# Step 6: do address-translation metrics scale with the number of decode layers? (docs/s6_decode_layer_scaling.ko.md 4-5)
# Every value comes from the per-kernel cumulative stat dumps in run.log; per-layer values are differences at the
# layer boundaries (12 kernels per layer). Only arms with EXIT=0 are used.
# Usage: s6_compare_layers.py <experiments dir>
import os
import re
import sys

E = sys.argv[1]
RUNS = ("L1", "L2", "L4")
ARMS = ("ref_batch4", "base_pcc", "ptwhop0_pcc")
PER_LAYER = 12
NONDET = re.compile(r"^(gpgpu_stall_shd_mem\[|traffic_breakdown_coretomem\[|gpu_total_sim_rate|gpgpu_simulation_rate|"
                    r"gpgpu_simulation_time|gpgpu_silicon_slowdown)")


def blocks(log):
    parts = log.split("\nkernel_name = ")[1:]
    return [p.split("\nHeader info loaded")[0] for p in parts]


def dump(b):
    one = lambda k: int(re.search(rf"^{k} = (\d+)", b, re.M).group(1))
    l2 = [tuple(int(x) for x in m) for m in re.findall(r"^\[L2_TLB_4k_STAT\] chiplet=\d+ hits=(\d+) misses=(\d+) mshr_hits=(\d+)", b, re.M)]
    assert len(l2) == 4, len(l2)
    g = lambda k: sum(float(x) for x in re.findall(rf"^\s+{k}\s*:\s*([\d.]+)", b, re.M))
    xa = re.search(r"^\[XLATSUM\] GPU kind=4k out=ALL n=(\d+) avg=([\d.]+)", b, re.M)
    xp = re.search(r"^\[XLATSUM\] GPU kind=4k out=PTW n=(\d+) .*l2_mshr_full=([\d.]+) .*pwq_wait=([\d.]+) walk_svc=([\d.]+)", b, re.M)
    return {"cycle": one("gpu_tot_sim_cycle"), "thread": one("gpu_tot_sim_insn"), "warp": one("gpgpu_n_tot_w_icount"),
            "l2_hit": sum(r[0] for r in l2), "l2_miss": sum(r[1] for r in l2), "l2_merge": sum(r[2] for r in l2),
            "walks": g("completed walks"), "local": g("local steps"), "remote": g("remote steps"),
            "at_n": int(xa.group(1)), "at_sum": int(xa.group(1)) * float(xa.group(2)),
            "ptw_n": int(xp.group(1)), "ptw_sum": int(xp.group(1)) * sum(float(xp.group(i)) for i in (2, 3, 4))}


def derived(d):
    return {"MPKI": d["l2_miss"] / (d["warp"] / 1000), "IPC": d["thread"] / d["cycle"],
            "AT": d["at_sum"] / d["at_n"] if d["at_n"] else 0, "PTW": d["ptw_sum"] / d["ptw_n"] if d["ptw_n"] else 0,
            "rr": d["remote"] / (d["local"] + d["remote"]) if d["local"] + d["remote"] else 0}


def minus(a, b):
    return {k: a[k] - b[k] for k in a}


data = {}
for r in RUNS:
    for a in ARMS:
        d = os.path.join(E, f"decode_S256_{r}", a)
        try:
            if not open(os.path.join(d, "exit_code.txt")).read().startswith("EXIT=0"):
                continue
        except FileNotFoundError:
            continue
        log = open(os.path.join(d, "run.log"), errors="replace").read()
        bl = blocks(log)
        dumps = [dump(b) for b in bl]
        pages = sum(1 for ln in open(os.path.join(E, f"decode_S256_{r}", "ref_batch4", "placement_batch4.map")) if ln.startswith("D "))
        data[(r, a)] = {"blocks": bl, "dumps": dumps, "pages": pages}

print("== totals per run (ref_batch4 and measurement arms) ==")
hdr = f"{'run':3s} {'arm':12s} {'warp':>12s} {'cycles':>11s} {'L2miss':>10s} {'walks':>10s} {'MPKI':>7s} {'IPC':>7s} {'AT':>7s} {'PTW':>7s} {'rr':>6s} {'pages':>8s}"
print(hdr)
for (r, a), v in data.items():
    t = v["dumps"][-1]
    x = derived(t)
    print(f"{r:3s} {a:12s} {t['warp']:12d} {t['cycle']:11d} {t['l2_miss']:10d} {t['walks']:10.0f} {x['MPKI']:7.2f} {x['IPC']:7.1f} "
          f"{x['AT']:7.1f} {x['PTW']:7.1f} {x['rr']:6.4f} {v['pages'] if a == 'ref_batch4' else '':>8}")

print("\n== per layer (difference at layer boundaries) ==")
print(f"{'run':3s} {'arm':12s} {'layer':>5s} {'warp':>11s} {'cycles':>10s} {'L2miss':>9s} {'walks':>9s} {'MPKI':>7s} {'IPC':>7s} {'AT':>7s} {'PTW':>7s} {'rr':>6s}")
for (r, a), v in data.items():
    prev = {k: 0 for k in v["dumps"][0]}
    n_layers = len(v["dumps"]) // PER_LAYER
    for k in range(n_layers):
        cur = v["dumps"][PER_LAYER * (k + 1) - 1]
        d = minus(cur, prev)
        x = derived(d)
        print(f"{r:3s} {a:12s} {k:5d} {d['warp']:11d} {d['cycle']:10d} {d['l2_miss']:9d} {d['walks']:9.0f} {x['MPKI']:7.2f} "
              f"{x['IPC']:7.1f} {x['AT']:7.1f} {x['PTW']:7.1f} {x['rr']:6.4f}")
        prev = cur

print("\n== normalized four metrics (ptwhop0_pcc / base_pcc, base_pcc = 1) ==")
for r in RUNS:
    if (r, "base_pcc") in data and (r, "ptwhop0_pcc") in data:
        b, h = data[(r, "base_pcc")]["dumps"][-1], data[(r, "ptwhop0_pcc")]["dumps"][-1]
        xb, xh = derived(b), derived(h)
        print(f"{r}: NormAT {xh['AT'] / xb['AT']:.4f} NormPTW {xh['PTW'] / xb['PTW']:.4f} NormIPC {xh['IPC'] / xb['IPC']:.4f} "
              f"rr base {xb['rr']:.4f} hop0 {xh['rr']:.4f} | raw AT {xb['AT']:.1f}->{xh['AT']:.1f} PTW {xb['PTW']:.1f}->{xh['PTW']:.1f} IPC {xb['IPC']:.2f}->{xh['IPC']:.2f}")

print("\n== L-6 prefix identity (later run's first kernels vs earlier run, nondeterministic/host-time keys excluded) ==")
for small, big in (("L1", "L2"), ("L2", "L4"), ("L1", "L4")):
    for a in ARMS:
        if (small, a) not in data or (big, a) not in data:
            continue
        s_bl, b_bl = data[(small, a)]["blocks"], data[(big, a)]["blocks"]
        diff_keys = set()
        for i in range(len(s_bl)):
            ls = [ln for ln in s_bl[i].splitlines() if not NONDET.match(ln)]
            lb = [ln for ln in b_bl[i].splitlines() if not NONDET.match(ln)]
            for x, y in zip(ls, lb):
                if x != y:
                    diff_keys.add(re.split(r"[ =:\[]", x.strip())[0][:40])
            if len(ls) != len(lb):
                diff_keys.add(f"line count block {i}")
        print(f"{big} vs {small} {a}: first {len(s_bl)} kernels", "IDENTICAL" if not diff_keys else f"DIFFER in {sorted(diff_keys)[:8]}")
