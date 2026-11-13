#!/usr/bin/env python3
# Step 36 (docs/s36_cta_work_and_reuse.ko.md): per-CTA work and data reuse of ONE accel-sim kernel trace, counted
# directly from the warp memory instructions (no simulation). For every CTA: instruction mix (HMMA/LDG/LDGSTS/STG/...),
# requested bytes and distinct 128 B lines / 64 KB pages per operand (operand = VA range from the capture's va_map.json),
# and for 2-D tensors the row range the CTA touched (GEMM tile rows). Across CTAs: per-line access count, distinct CTAs,
# and reuse distance in CTA launch order, classified as same CTA / same chiplet+wave / same chiplet other wave /
# other chiplet (chiplet = KERNEL_PARTITION split, wave = 64 SMs x CTAs-per-SM, same formula as s23).
# Usage: s36_cta_reuse.py <trace.xz> <va_map.json> <out.json> [--detail-ctas 4] [--page-shift 16]
import json, lzma, math, sys, time, bisect, collections

MEM_OPS = {"LDG", "STG", "LDGSTS", "LD", "ST", "ATOM", "ATOMG", "RED"}
WRITE_OPS = {"STG", "ST", "RED", "ATOM", "ATOMG"}
N_CHIP, SMS_PER_CHIP = 4, 64
HMMA_FLOP = 16 * 8 * 16 * 2   # m16n8k16, one warp-level HMMA instruction

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

def load_ranges(va_map_path):
    """Return sorted (start, end, label, row_stride_bytes or 0) from va_map.json."""
    m = json.load(open(va_map_path)); R = []
    def add(label, e):
        if not (isinstance(e, dict) and "ptr" in e and "bytes" in e): return
        st = e.get("stride"); sh = e.get("shape"); rs = 0
        if st and sh and len(sh) >= 2 and len(st) >= 1: rs = int(st[0]) * 2   # bf16 elements -> bytes
        R.append((int(e["ptr"]), int(e["ptr"]) + int(e["bytes"]), label, rs))
    for n, e in (m.get("weights") or {}).items(): add("w." + n, e)
    for n, e in (m.get("modules") or {}).items():
        for i, x in enumerate(e.get("inputs") or []): add(f"m.{n}.in{i}", x)
        for i, x in enumerate(e.get("outputs") or []): add(f"m.{n}.out{i}", x)
    kv = m.get("kv_cache_layer0")
    if isinstance(kv, dict) and "ptr" in kv: add("kv_cache_layer0", kv)
    # drop exact duplicates (same range) keeping the first label; keep overlapping distinct ranges (bisect picks the
    # last start <= addr, so inner/later-starting ranges win).
    seen = set(); out = []
    for r in sorted(R):
        if (r[0], r[1]) in seen: continue
        seen.add((r[0], r[1])); out.append(r)
    return out

def main():
    f, vmp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    DET = 4; SHIFT = 16
    for i, a in enumerate(sys.argv):
        if a == "--detail-ctas": DET = int(sys.argv[i + 1])
        if a == "--page-shift": SHIFT = int(sys.argv[i + 1])
    R = load_ranges(vmp); starts = [r[0] for r in R]
    def label_of(a):
        i = bisect.bisect_right(starts, a) - 1
        while i >= 0:
            if R[i][0] <= a < R[i][1]: return i
            i -= 1
        return -1
    t0 = time.time(); h = {}; gx = gy = gz = 1; chunk = per_wave = cps = 1; n_total = 0
    line = {}          # line -> [acc, n_ctas, last_cta, first_cta]
    page_cta = {}      # page -> [n_ctas, last_cta]
    op_tot = collections.Counter()     # operand -> total requested bytes (all CTAs)
    op_lines = collections.defaultdict(set)   # operand -> distinct lines (memory: ok for <= ~10M lines)
    dist_hist = collections.defaultdict(lambda: collections.Counter())   # operand -> {class: count}
    dist_log = collections.defaultdict(lambda: collections.Counter())    # operand -> {log2 bin: count}
    cta_rows = []      # per-CTA summary dicts (compact)
    details = []       # first DET CTAs: operand row ranges
    cur = None; n_cta = 0; lin = chip = local = wave = 0
    def close_cta():
        nonlocal cur
        if cur is None: return
        row = {"cid": cur["cid"], "chip": cur["chip"], "wave": cur["wave"], "insts": cur["insts"],
               "mix": dict(cur["mix"]), "bytes_rd": sum(v for k, v in cur["bytes"].items() if not k.endswith("|w")),
               "bytes_wr": sum(v for k, v in cur["bytes"].items() if k.endswith("|w")),
               "lines": len(cur["lines"]), "pages": len(cur["pages"]), "line_acc": cur["line_acc"], "page_look": cur["page_look"],
               "op_pages": {k: len(v) for k, v in cur["op_pages"].items()}}
        cta_rows.append(row)
        if len(details) < DET:
            d = {"cid": cur["cid"], "chip": cur["chip"], "wave": cur["wave"], "operands": {}}
            for k, v in cur["bytes"].items():
                name, rw = k.rsplit("|", 1); rr = cur["rows"].get(k)
                d["operands"][k] = {"bytes": v, "lines": len(cur["op_lines"][k]), "pages": len(cur["op_pages"][k]),
                                    "rows": [rr[0], rr[1], len(rr[2])] if rr else None}
            details.append(d)
        cur = None
    with lzma.open(f, "rt") as fh:
        for ln in fh:
            if ln.startswith("-"):
                k, v = ln.lstrip("-").split("=", 1); h[k.strip()] = v.strip(); continue
            if ln.startswith("#"):
                if "grid dim" not in h or n_total: continue
                gx, gy, gz = hdr_int3(h["grid dim"]); bx, by, bz = hdr_int3(h["block dim"])
                threads = bx * by * bz; nregs = int(h.get("nregs", 32)); shmem = int(h.get("shmem", 0))
                n_total = gx * gy * gz; chunk = math.ceil(n_total / N_CHIP)
                regs_per_warp = math.ceil(nregs * 32 / 256) * 256; warps = math.ceil(threads / 32)
                cps = max(1, min(32, 2048 // threads, 65536 // (regs_per_warp * warps), (102400 // shmem) if shmem else 32))
                per_wave = SMS_PER_CHIP * cps; continue
            if ln.startswith("thread block ="):
                close_cta()
                x, y, z = (int(v) for v in ln.split("=")[1].split(","))
                lin = x + y * gx + z * gx * gy; chip = min(N_CHIP - 1, lin // chunk); local = lin - chip * chunk
                wave = local // per_wave; n_cta += 1
                cur = {"cid": lin, "chip": chip, "wave": wave, "insts": 0, "mix": collections.Counter(), "bytes": collections.Counter(),
                       "lines": set(), "pages": set(), "line_acc": 0, "page_look": 0, "rows": {}, "op_lines": collections.defaultdict(set), "op_pages": collections.defaultdict(set)}
                continue
            if cur is None or ln.startswith("warp ="): continue
            t = ln.split()
            if len(t) < 5: continue
            try:
                dn = int(t[2]); oi = 3 + dn
            except ValueError:
                continue
            cur["insts"] += 1
            op = t[oi].split(".")[0]
            if op.startswith("HMMA"): cur["mix"]["HMMA"] += 1
            elif op in ("LDS", "STS", "LDSM"): cur["mix"]["SHM"] += 1
            elif op in MEM_OPS: cur["mix"][op] += 1
            if op not in MEM_OPS: continue
            sn = int(t[oi + 1]); kk = oi + 2 + sn
            try:
                A = addrs(t, kk); width = int(t[kk])
            except Exception:
                continue
            if not A: continue
            rw = "w" if op in WRITE_OPS else "r"
            seen_lines = set()
            for a in A:
                L = a >> 7
                if L in seen_lines: continue   # one line per instruction counts once (coalesced)
                seen_lines.add(L)
            li = label_of(A[0]); name = R[li][2] if li >= 0 else "other"; key = name + "|" + rw
            cur["bytes"][key] += width * len(A)
            op_tot[key] += width * len(A)
            if li >= 0 and R[li][3]:
                rs = R[li][3]; base = R[li][0]
                rr = cur["rows"].get(key)
                if rr is None: rr = cur["rows"][key] = [1 << 62, -1, set()]
                for a in A:
                    r = (a - base) // rs
                    if r < rr[0]: rr[0] = r
                    if r > rr[1]: rr[1] = r
                    if len(rr[2]) < 100000: rr[2].add(r)
            cl = cur["lines"]; pg = cur["pages"]; opl = cur["op_lines"][key]; opL = op_lines[name]
            cur["line_acc"] += len(seen_lines)
            pgs = {L >> (SHIFT - 7) for L in seen_lines}; cur["page_look"] += len(pgs); cur["op_pages"][key].update(pgs)
            dh = dist_hist[name]; dl = dist_log[name]
            for L in seen_lines:
                cl.add(L); opl.add(L); opL.add(L); pg.add(L >> (SHIFT - 7))
                e = line.get(L)
                if e is None:
                    line[L] = [1, 1, lin, lin]; dh["first"] += 1
                else:
                    e[0] += 1
                    if e[2] != lin:
                        d = lin - e[2]; e[1] += 1
                        pchip = min(N_CHIP - 1, e[2] // chunk); pwave = (e[2] - pchip * chunk) // per_wave
                        if pchip != chip: dh["other_chiplet"] += 1
                        elif pwave == wave: dh["same_chiplet_same_wave"] += 1
                        else: dh["same_chiplet_other_wave"] += 1
                        dl[d.bit_length()] += 1
                        e[2] = lin
                    else:
                        dh["same_cta"] += 1
            # (page-level distinct CTA count)
            for P in {L >> (SHIFT - 7) for L in seen_lines}:
                e = page_cta.get(P)
                if e is None: page_cta[P] = [1, lin]
                elif e[1] != lin: e[0] += 1; e[1] = lin
    close_cta()
    # summaries
    def dist(v):
        v = sorted(v); n = len(v)
        return {"min": v[0], "p50": v[n // 2], "max": v[-1], "mean": sum(v) / n} if n else None
    ops = {}
    for key, tot in op_tot.items():
        name, rw = key.rsplit("|", 1)
        uniq_lines = len(op_lines[name]); uniq_bytes = uniq_lines * 128
        ops[key] = {"total_bytes": tot, "unique_lines": uniq_lines, "unique_bytes": uniq_bytes,
                    "reuse_factor_bytes": (tot / uniq_bytes) if uniq_bytes else None}
    op_reuse = {}
    for name in op_lines:
        ls = op_lines[name]; acc = sum(line[L][0] for L in ls); nct = [line[L][1] for L in ls]
        op_reuse[name] = {"unique_lines": len(ls), "accesses": acc, "reuse_factor_lines": acc / len(ls),
                          "ctas_per_line": dist(nct), "dist_class": dict(dist_hist[name]),
                          "dist_log2": {int(k): v for k, v in sorted(dist_log[name].items())}}
    pages_ctas = [v[0] for v in page_cta.values()]
    out = {"trace": f, "kernel": h.get("kernel name"), "grid": h.get("grid dim"), "block": h.get("block dim"),
           "nregs": h.get("nregs"), "shmem": h.get("shmem"), "n_cta": n_cta, "chunk": chunk, "cps": cps, "per_wave": per_wave,
           "insts_total": sum(r["insts"] for r in cta_rows),
           "cta": {"insts": dist([r["insts"] for r in cta_rows]),
                   "mix": {k: dist([r["mix"].get(k, 0) for r in cta_rows]) for k in ("HMMA", "LDG", "LDGSTS", "STG", "SHM", "LD", "ST")},
                   "bytes_rd": dist([r["bytes_rd"] for r in cta_rows]), "bytes_wr": dist([r["bytes_wr"] for r in cta_rows]),
                   "lines": dist([r["lines"] for r in cta_rows]), "pages": dist([r["pages"] for r in cta_rows]),
                   "intra_reuse_lines": dist([r["line_acc"] / r["lines"] for r in cta_rows if r["lines"]]),
                   "page_lookups": dist([r["page_look"] for r in cta_rows]),
                   "intra_reuse_pages": dist([r["page_look"] / r["pages"] for r in cta_rows if r["pages"]]),
                   "op_pages_p50": {k: sorted(r["op_pages"].get(k, 0) for r in cta_rows)[len(cta_rows) // 2] for k in op_tot},
                   "flop_hmma": (dist([r["mix"].get("HMMA", 0) for r in cta_rows])["p50"] * HMMA_FLOP) if cta_rows else 0},
           "operands": ops, "operand_reuse": op_reuse,
           "pages": {"unique": len(page_cta), "ctas_per_page": dist(pages_ctas)},
           "lines_unique": len(line), "details": details, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(out, open(outp, "w"), indent=1)
    print(f"{h.get('kernel name','?')[:70]} grid {out['grid']} ctas {n_cta} per_wave {per_wave} insts {out['insts_total']:,} lines {len(line):,} pages {len(page_cta):,} {out['elapsed_s']} s")

main()
