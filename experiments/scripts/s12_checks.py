# Step 12 checks (docs/s12_d1l1_xlat_diag.ko.md section 4).
#   s12_checks.py stats <run.log>            key end-of-run statistics (last cumulative dump)
#   s12_checks.py same  <run.log A> <run.log B>   V-S3: compare the model statistics of two runs
#   s12_checks.py vk    <4KB map> <64KB map>      V-K2..V-K4: placement-map consistency
import re
import sys

PA_CHIPLET_SHIFT = 22        # current config: -gpgpu_mem_addr_mapping puts the top two channel (D) bits at PA[35:34]
                             # (64 GB, 16 GB per chiplet), so chiplet = pfn >> 22 for 4 KB frames. The 06-25 A.1 note
                             # (PA[33:32], pfn >> 20) describes an older, smaller configuration.


def stats(path):
    s = {"kernels": 0, "capped": False}
    l1 = {}; l2 = {}; gm = {}; dram = []; cur_gm = None; xl = {}
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("launching kernel name:"):
                s["kernels"] += 1
            elif line.startswith("GPGPU-Sim: ** break due to reaching the maximum"):
                s["capped"] = True
            elif line.startswith("[RUN_MODE]"):
                s["run_mode"] = line.split("]", 1)[1].strip()
            elif line.startswith("[PAGE_TABLE] pt_nodes="):
                s["pt_nodes"] = int(re.search(r"pt_nodes=(\d+)", line).group(1))
            elif line.startswith("[PAGE_TABLE] mutations="):
                s["pt_mutations"] = int(re.search(r"mutations=(\d+)", line).group(1))
            elif line.startswith("[PLACEMENT]"):
                s.setdefault("placement", []).append(line.strip())
            elif m := re.match(r"\[L1_TLB_(4k|64k)_STAT\] SM=(\d+) hits=(\d+) misses=(\d+)", line):
                l1[(m.group(1), m.group(2))] = (int(m.group(3)), int(m.group(4)))
            elif m := re.match(r"\[L2_TLB_(4k|64k)_STAT\] chiplet=(\d+) hits=(\d+) misses=(\d+)", line):
                l2[(m.group(1), m.group(2))] = (int(m.group(3)), int(m.group(4)))
            elif m := re.match(r"\[GMMU_STAT\] chiplet=(\d+)", line):
                cur_gm = gm.setdefault(m.group(1), {})
            elif cur_gm is not None and (m := re.match(r"\s+([a-zA-Z][a-zA-Z ]*?)\s*:\s*([\d.]+)", line)):
                cur_gm[m.group(1)] = float(m.group(2))
            elif m := re.match(r"\[XLATSUM\] GPU kind=(4k|64k) out=(ALL|PTW) n=(\d+) avg=([\d.]+)", line):
                xl[(m.group(1), m.group(2))] = (int(m.group(3)), float(m.group(4)))
            elif line.startswith("DRAM[0]:"):
                dram = []
            elif m := re.search(r" n_rd=(\d+) .*n_write=(\d+) n_wr_bk=(\d+)", line):
                dram.append(sum(int(x) for x in m.groups()))
            else:
                for k in ("gpu_tot_sim_cycle", "gpu_tot_sim_insn", "gpgpu_n_tot_w_icount",
                          "mcm_remote_accesses", "mcm_local_accesses", "L2_total_cache_accesses"):
                    if line.strip().startswith(k + " =") or line.startswith(k + "="):
                        s[k] = int(float(re.split(r"=\s*", line.strip(), 1)[1].split()[0]))
    for kind in ("4k", "64k"):
        s[f"l1tlb_{kind}_access"] = sum(h + m for (k, _), (h, m) in l1.items() if k == kind)
        s[f"l2tlb_{kind}_access"] = sum(h + m for (k, _), (h, m) in l2.items() if k == kind)
        s[f"l2tlb_{kind}_miss"] = sum(m for (k, _), (h, m) in l2.items() if k == kind)
        for out in ("ALL", "PTW"):
            if (kind, out) in xl:
                s[f"xlat_{kind}_{out}_n"], s[f"xlat_{kind}_{out}_avg"] = xl[(kind, out)]
    for key, name in (("requests", "gmmu_requests"), ("completed walks", "gmmu_walks"),
                      ("local steps", "pt_local_steps"), ("remote steps", "pt_remote_steps")):
        s[name] = int(sum(b.get(key, 0) for b in gm.values()))
    s["dram_atoms"] = sum(dram)
    return s


MODEL_KEYS = ("kernels", "capped", "gpu_tot_sim_cycle", "gpu_tot_sim_insn", "gpgpu_n_tot_w_icount",
              "l1tlb_4k_access", "l2tlb_4k_access", "l2tlb_4k_miss", "gmmu_requests", "gmmu_walks",
              "pt_local_steps", "pt_remote_steps", "dram_atoms", "mcm_remote_accesses", "mcm_local_accesses",
              "L2_total_cache_accesses", "pt_nodes", "pt_mutations")


def same(a, b):
    sa, sb = stats(a), stats(b)
    bad = 0
    for k in MODEL_KEYS:
        ok = sa.get(k) == sb.get(k)
        bad += not ok
        print(f"{'SAME' if ok else 'DIFF'}  {k:26s} {sa.get(k)!s:>16} {sb.get(k)!s:>16}")
    print("V-S3", "PASS" if bad == 0 else f"FAIL ({bad} keys differ)")


def load_map(path):
    shift = None; data = {}; leaf = {}
    for line in open(path):
        if line.startswith("# page_shift"):
            shift = int(line.split()[-1])
        elif line.startswith("D "):
            _, vpn, own, pfn = line.split()[:4]
            data[int(vpn, 16)] = (int(own), int(pfn, 16))
        elif line.startswith("N 3 "):
            _, _, key, own = line.split()[:4]
            leaf[int(key, 16)] = int(own)
    return shift, data, leaf


def vk(m4, m64):
    s4, d4, l4 = load_map(m4)
    s64, d64, l64 = load_map(m64)
    print(f"4KB map : page_shift {s4}, data pages {len(d4):,}, leaf PT nodes {len(l4):,}")
    print(f"64KB map: page_shift {s64}, data pages {len(d64):,}, leaf PT nodes {len(l64):,}")
    res = {}
    res["V-K1 page_shift"] = (s4 == 12 and s64 == 16)
    collapsed = {v >> 4 for v in d4}
    res["V-K2 64KB pages == 4KB pages >> 4"] = (collapsed == set(d64))
    if not res["V-K2 64KB pages == 4KB pages >> 4"]:
        print(f"   only in 4KB>>4: {len(collapsed - set(d64))}, only in 64KB: {len(set(d64) - collapsed)}")
    # leaf node region = 2 MB: 4KB vpn >> 9, 64KB vpn >> 5
    res["V-K3 leaf regions equal"] = ({v >> 9 for v in d4} == {v >> 5 for v in d64} == set(l4) == set(l64))
    res["V-K4a 64KB frames 16-aligned"] = all(pfn % 16 == 0 for _, pfn in d64.values())
    res["V-K4b 64KB owner == PA chiplet"] = all(own == (pfn >> PA_CHIPLET_SHIFT) & 3 for own, pfn in d64.values())
    res["V-K4c 4KB owner == PA chiplet"] = all(own == (pfn >> PA_CHIPLET_SHIFT) & 3 for own, pfn in d4.values())
    for k, v in res.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stats":
        for k, v in stats(sys.argv[2]).items():
            print(f"{k:28s} {v}")
    elif cmd == "same":
        same(sys.argv[2], sys.argv[3])
    elif cmd == "vk":
        vk(sys.argv[2], sys.argv[3])
