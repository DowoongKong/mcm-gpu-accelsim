#!/usr/bin/env python3
# Step 37 (docs/s37_access_class_remote_pt.ko.md): attribute every page walk of one simulator arm to the sharing class
# of the page it translated. Walks come from the arm's L2MISSLOG (`C` lines: cycle, chiplet, vpn, steps = page-table
# reads that went to memory, remote = those that crossed a chiplet); the kernel a walk belongs to comes from the
# per-kernel cumulative gpu_tot_sim_cycle in run.log; page classes come from s37_sharing_class.py output of the same
# kernel trace. Gates: G2 walks == [XLATSUM] GPU out=PTW n, G3 every walked vpn is in its kernel's page table.
# Usage: s37_remote_pt_by_class.py <arm_dir> <kernelslist> <class_dir> <out.json>
import bisect, collections, gzip, json, os, re, sys

CLS = ("private", "partial", "shared")


def op_group(label):
    if label.startswith("w."): return "weight"
    if label.startswith("kv_cache"): return "kv"
    if label.startswith("m."): return "activation"
    return "other"


def main():
    arm, klist, cdir, outp = sys.argv[1:5]
    kids = [int(re.search(r"kernel-(\d+)-", l).group(1)) for l in open(klist) if l.startswith("kernel-")]
    names, ends, ptw_n = [], [], None
    for l in open(os.path.join(arm, "run.log"), errors="replace"):
        if l.startswith("kernel_name = "): names.append(l.split("=", 1)[1].strip())
        elif l.startswith("gpu_tot_sim_cycle = "): ends.append(int(l.split("=")[1]))
        elif l.startswith("[XLATSUM] GPU") and " out=PTW " in l: ptw_n = int(re.search(r" n=(\d+)", l).group(1))
    assert len(names) == len(ends) == len(kids), (len(names), len(ends), len(kids))
    pages = {}
    def table(i):
        if i not in pages:
            js = json.load(open(os.path.join(cdir, f"k{kids[i]}.json")))
            assert (js["kernel"] or "")[:40] == names[i][:40], (i, js["kernel"][:40], names[i][:40])
            T = {}
            with gzip.open(os.path.join(cdir, f"k{kids[i]}.pages.tsv.gz"), "rt") as g:
                next(g)
                for row in g:
                    v = row.rstrip("\n").split("\t")
                    a = [int(x) for x in v[9:12]]
                    T[int(v[0], 16)] = (v[3], op_group(v[1]), bin(int(v[4])).count("1"), bin(int(v[5])).count("1"),
                                        CLS[a.index(max(a))], int(v[4]))
            pages[i] = (T, js["n_cta"])
        return pages[i]
    # Owner chiplet of a physical address = PA >> log2(bytes per chiplet) (the 37-bit mapping puts the chiplet bits on top).
    cfg = open(os.path.join(arm, "gpgpusim.config")).read()
    mem_mb = int(re.search(r"^-gpu_mem_size_mb\s+(\d+)", cfg, re.M).group(1))
    nchip = int(re.search(r"^-gpgpu_n_chiplets\s+(\d+)", cfg, re.M).group(1))
    oshift = (mem_mb // nchip * (1 << 20)).bit_length() - 1
    # walks, PT steps, remote steps, walks with >=1 remote step, remote leaf-PT reads, remote upper-level reads,
    # walks whose data page is on another chiplet
    Z = lambda: [0, 0, 0, 0, 0, 0, 0]
    agg = {k: collections.defaultdict(Z) for k in ("page_class", "line_major", "operand", "page_chips", "leaf_chips",
                                                     "config_vs_share", "kernel_class", "locality")}
    tot = Z(); missing = 0; bad_missing = 0; not_member = 0; n = 0
    def local_ops(i):
        return json.load(open(os.path.join(cdir, f"k{kids[i]}.json"))).get("local_mem_warp_insts", 0) > 0
    for l in open(os.path.join(arm, "l2miss.log")):
        if not l.startswith("C "): continue
        t = l.split(); cyc, chip, vpn, steps, rem = int(t[1]), int(t[2]), int(t[3], 16), int(t[6]), int(t[7])
        leaf_pa, data_pa = int(t[13], 16), int(t[14], 16)
        leaf_rem = 1 if (steps and leaf_pa and (leaf_pa >> oshift) != chip) else 0   # the leaf entry is the last memory step
        data_rem = 1 if (data_pa >> oshift) != chip else 0
        loc = ("data local" if not data_rem else "data REMOTE") + (", leaf PT local" if not leaf_rem else ", leaf PT REMOTE")
        i = min(bisect.bisect_left(ends, cyc), len(ends) - 1); T, nc = table(i); n += 1
        e = T.get(vpn)
        if e is None:
            # Not in the trace: local memory (LDL/STL), which the simulator remaps to its own region. Counted as its
            # own class; G3 fails only if the kernel has no local-memory instructions.
            missing += 1
            if not local_ops(i): bad_missing += 1
            e = ("local(sim)", "local", 0, 0, "local(sim)", 0)
        pc, og, npc, nlc, lm, mask = e
        if mask and not (mask >> chip) & 1: not_member += 1
        v = (1, steps, rem, 1 if rem else 0, leaf_rem, rem - leaf_rem, data_rem)
        keys = {"page_class": pc, "line_major": lm, "operand": og, "page_chips": npc, "leaf_chips": nlc,
                "config_vs_share": "page touched by 1 chiplet" if npc == 1 else "page touched by >=2 chiplets",
                "kernel_class": f"{i + 1:02d} {names[i][:34]} | {pc}", "locality": loc}
        for k, key in keys.items():
            a = agg[k][key]
            for j in range(7): a[j] += v[j]
        for j in range(7): tot[j] += v[j]
    out = {"arm": arm, "kernels": len(kids), "walks_logged": n, "xlatsum_ptw_n": ptw_n,
           "G2_walks_match": n == ptw_n, "local_mem_walks": missing, "G3_unexplained_missing_vpn": bad_missing,
           "walker_not_in_page_chips": not_member,
           "total": tot, **{k: {str(kk): vv for kk, vv in sorted(v.items(), key=lambda x: str(x[0]))} for k, v in agg.items()}}
    json.dump(out, open(outp, "w"), indent=1)
    print(f"walks {n} (XLATSUM {ptw_n}, G2 {'PASS' if n == ptw_n else 'FAIL'}), local-memory walks {missing}, "
          f"unexplained missing {bad_missing} (G3 {'PASS' if not bad_missing else 'FAIL'}), walker not in page chips {not_member}")
    print(f"total: walks {tot[0]} steps {tot[1]} remote {tot[2]} ({100 * tot[2] / max(1, tot[1]):.1f}% of steps; "
          f"leaf {tot[4]}, upper levels {tot[5]}); walks with data on another chiplet {tot[6]} (owner shift {oshift})")
    for k in ("page_class", "line_major", "operand", "page_chips", "config_vs_share", "locality"):
        print(f"-- by {k}: class | walks (share) | PT steps | remote steps (share of all remote) | remote/steps | remote leaf / upper | data-remote walks")
        for kk, (w, s, r, wr, rl, ru, dr) in sorted(agg[k].items(), key=lambda x: str(x[0])):
            print(f"   {str(kk):34s} {w:8d} ({100 * w / max(1, tot[0]):5.1f}%) {s:8d} {r:8d} ({100 * r / max(1, tot[2]):5.1f}%) "
                  f"{100 * r / max(1, s):5.1f}%  {rl:7d} / {ru:7d}  {dr:7d}")


main()
