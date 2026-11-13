#!/usr/bin/env python3
# Step 23 (docs/s23_llm_kernel_tlb_anatomy.ko.md): page-level profile of ONE accel-sim kernel trace, in the simulator's
# own units. A "lookup" is one (warp memory instruction, distinct page) pair, exactly what the L1 TLB sees
# (gpu_tlb.cc access() is keyed by (warp_id, vpn)). CTAs are bound to chiplets as KERNEL_PARTITION does
# (chip = lin // ceil(nCTA / 4)), issued per chiplet in linear order, 64 SMs x ctas_per_sm concurrent = one wave.
# Reported per kernel: header, instruction/lookup counts by opcode, pages per instruction, per-CTA distinct pages and
# re-lookups per page (reuse factor), per-chiplet distinct pages (= lower bound of compulsory L2 TLB misses, one per
# (chiplet, page)), the live set of chiplet 0's first wave (reads / writes), sharing (CTAs and chiplets per page), and
# contiguous address segments (tensors) with their size and read/write mix.
# Usage: s23_kernel_tlb_profile.py <trace.xz> <out.json> [page_shift=16]
import json, lzma, math, sys, time
f, outp = sys.argv[1], sys.argv[2]
SHIFT = int(sys.argv[3]) if len(sys.argv) > 3 else 16
N_CHIP, SMS_PER_CHIP = 4, 64
MEM_OPS = {"LDG", "STG", "LDGSTS", "LD", "ST", "ATOM", "ATOMG", "RED", "LDL", "STL"}
WRITE_OPS = {"STG", "ST", "STL", "RED", "ATOM", "ATOMG"}
t_start = time.time()

def addrs(t, k):
    if int(t[k]) == 0: return []
    fmt = int(t[k + 1]); base = int(t[k + 2], 16); act = bin(int(t[1], 16)).count("1")
    if fmt == 1:
        s = int(t[k + 3]); return [base + i * s for i in range(act)]
    if fmt == 2:
        out = [base]; a = base
        for d in t[k + 3:k + 3 + act - 1]: a += int(d); out.append(a)
        return out
    return [int(x, 16) for x in t[k + 2:k + 2 + act]]

def hdr_int3(v): return tuple(int(x) for x in v.strip("() ").split(","))

h = {}; gx = gy = gz = 1; chunk = per_wave = cps = 1
op_insn = {}; op_look = {}; n_mem = 0; n_look = 0; n_lanes = 0; hist = {}
cta_look = []; cta_pages = []; cta_rd = []; cta_wr = []   # per CTA
cur = None   # per-CTA state: dict(pages=set, rd=set, wr=set, look=int)
page = {}    # page -> [cnt_c0..c3, ctas(set), rd, wr]
wave0_rd = set(); wave0_wr = set(); n_cta = 0; chip = 0; local = 0; wave = 0

def close_cta():
    if cur is None: return
    cta_look.append(cur["look"]); cta_pages.append(len(cur["pages"])); cta_rd.append(len(cur["rd"])); cta_wr.append(len(cur["wr"]))

with lzma.open(f, "rt") as fh:
    for line in fh:
        if line.startswith("-"):
            k, v = line.lstrip("-").split("=", 1); h[k.strip()] = v.strip(); continue
        if line.startswith("#"):
            if "grid dim" not in h or chunk != 1 or per_wave != 1: continue
            gx, gy, gz = hdr_int3(h["grid dim"]); bx, by, bz = hdr_int3(h["block dim"])
            threads = bx * by * bz; nregs = int(h.get("nregs", 32)); shmem = int(h.get("shmem", 0))
            n_total = gx * gy * gz; chunk = math.ceil(n_total / N_CHIP)
            regs_per_warp = math.ceil(nregs * 32 / 256) * 256; warps = math.ceil(threads / 32)
            cps = max(1, min(32, 2048 // threads, 65536 // (regs_per_warp * warps), (102400 // shmem) if shmem else 32))
            per_wave = SMS_PER_CHIP * cps; continue
        if line.startswith("thread block ="):
            close_cta()
            x, y, z = (int(v) for v in line.split("=")[1].split(","))
            lin = x + y * gx + z * gx * gy; chip = min(N_CHIP - 1, lin // chunk); local = lin - chip * chunk
            wave = local // per_wave; n_cta += 1
            cur = {"pages": set(), "rd": set(), "wr": set(), "look": 0, "cid": lin}; continue
        if cur is None or line.startswith("warp ="): continue
        t = line.split()
        if len(t) < 5: continue
        try:
            dn = int(t[2]); oi = 3 + dn
        except ValueError:
            continue
        op = t[oi].split(".")[0]
        if op not in MEM_OPS: continue
        sn = int(t[oi + 1]); kk = oi + 2 + sn
        try: A = addrs(t, kk)
        except Exception: continue
        if not A: continue
        p64 = {a >> SHIFT for a in A}; npg = len(p64)
        n_mem += 1; n_look += npg; n_lanes += len(A); hist[npg] = hist.get(npg, 0) + 1
        op_insn[op] = op_insn.get(op, 0) + 1; op_look[op] = op_look.get(op, 0) + npg
        wr = op in WRITE_OPS
        cur["look"] += npg; cur["pages"] |= p64
        (cur["wr"] if wr else cur["rd"]).update(p64)
        if chip == 0 and wave == 0:
            (wave0_wr if wr else wave0_rd).update(p64)
        for p in p64:
            e = page.get(p)
            if e is None: e = page[p] = [0, 0, 0, 0, set(), 0, 0]
            e[chip] += 1; e[4].add(cur["cid"])
            if wr: e[6] += 1
            else: e[5] += 1
close_cta()

# derived
pages_sorted = sorted(page)
chip_pages = [sum(1 for p in pages_sorted if page[p][c] > 0) for c in range(N_CHIP)]
cold_lb = sum(chip_pages)
ctas_per_page = [len(page[p][4]) for p in pages_sorted]
chips_per_page = [sum(1 for c in range(N_CHIP) if page[p][c] > 0) for p in pages_sorted]
mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
# segments: split the sorted page list where the gap exceeds 32 pages (2 MiB at 64 KiB pages)
segs = []; start = None; prev = None
for p in pages_sorted + [None]:
    if p is not None and (prev is None or p - prev <= 32):
        if start is None: start = p
        prev = p; continue
    if start is not None:
        ps = [q for q in pages_sorted if start <= q <= prev]
        segs.append({"first_va": f"0x{start << SHIFT:x}", "pages": len(ps), "span_pages": prev - start + 1,
                     "MiB": round(len(ps) * (1 << SHIFT) / 2**20, 1),
                     "lookups": sum(sum(page[q][:4]) for q in ps), "reads": sum(page[q][5] for q in ps),
                     "writes": sum(page[q][6] for q in ps),
                     "ctas_per_page": round(mean([len(page[q][4]) for q in ps]), 1),
                     "chips_per_page": round(mean([sum(1 for c in range(N_CHIP) if page[q][c] > 0) for q in ps]), 2),
                     "lookups_per_page": round(mean([sum(page[q][:4]) for q in ps]), 1)})
    start = p; prev = p
segs.sort(key=lambda s: -s["lookups"])
out = {
    "trace": f.split("/")[-1], "kernel_name": h.get("kernel name", ""), "grid": h.get("grid dim"), "block": h.get("block dim"),
    "nregs": int(h.get("nregs", 0)), "shmem": int(h.get("shmem", 0)), "n_cta": n_cta, "ctas_per_sm": cps, "chunk": chunk,
    "per_wave": per_wave, "waves_per_chiplet": math.ceil(chunk / per_wave), "page_shift": SHIFT,
    "mem_warp_insn": n_mem, "lookups": n_look, "lanes": n_lanes, "pages_per_insn": round(n_look / max(1, n_mem), 3),
    "pages_per_insn_hist": dict(sorted(hist.items())), "insn_by_op": op_insn, "lookups_by_op": op_look,
    "distinct_pages": len(pages_sorted), "distinct_MiB": round(len(pages_sorted) * (1 << SHIFT) / 2**20, 1),
    "chip_distinct_pages": chip_pages, "cold_lower_bound": cold_lb,
    "cta_lookups_mean": round(mean(cta_look), 1), "cta_pages_mean": round(mean(cta_pages), 1),
    "cta_pages_min": min(cta_pages) if cta_pages else 0, "cta_pages_max": max(cta_pages) if cta_pages else 0,
    "cta_read_pages_mean": round(mean(cta_rd), 1), "cta_write_pages_mean": round(mean(cta_wr), 1),
    "reuse_per_cta_page": round(sum(cta_look) / max(1, sum(cta_pages)), 2),
    "ctas_per_page_mean": round(mean(ctas_per_page), 2), "chips_per_page_mean": round(mean(chips_per_page), 3),
    "lookups_per_chip_page": round(n_look / max(1, cold_lb), 2),
    "wave0_read_pages": len(wave0_rd), "wave0_write_pages": len(wave0_wr), "wave0_pages": len(wave0_rd | wave0_wr),
    "segments": segs[:12], "n_segments": len(segs), "parse_seconds": round(time.time() - t_start),
}
json.dump(out, open(outp, "w"), indent=1)
print(f"{out['trace']} {out['kernel_name'][:50]} grid={out['grid']} ctas={n_cta} cps={cps} waves/chip={out['waves_per_chiplet']} "
      f"mem_insn={n_mem:,} lookups={n_look:,} p/insn={out['pages_per_insn']} pages={len(pages_sorted):,} cold_lb={cold_lb:,} "
      f"cta_pages={out['cta_pages_mean']} reuse={out['reuse_per_cta_page']} wave0={out['wave0_pages']} ({len(wave0_rd)}r/{len(wave0_wr)}w) "
      f"segs={len(segs)} t={out['parse_seconds']}s")
