#!/usr/bin/env python3
# s36_summarize.py <out dir> <capture_info.json> <window> -- KERNELS.md from the per-kernel JSONs of s36_cta_reuse.py:
# per kernel: tile (from the name), CTAs, per-CTA instruction mix / bytes / pages, per-operand reuse factor and the
# reuse-distance classes, and the row ranges of the first CTA (what one CTA computes).
import json, os, re, sys
O, CI, W = sys.argv[1], sys.argv[2], sys.argv[3]
ci = json.load(open(CI)); ks = [k for k in ci["window_kernels"] if k["window"] == W]
def short(n):
    m = re.search(r"(s16816gemm[\w]*?_(\d+x\d+)[\w]*?_(\d+x\d+))", n)
    if m: return f"GEMM tile {m.group(2)} K{m.group(3)}"
    if "flash_fwd_splitkv" in n: return "FA2 splitkv"
    for t in ("rms_norm", "fused_add_rms_norm", "rotary_embedding", "reshape_and_cache_flash", "act_and_mul", "elementwise", "gather", "slot_mapping", "index_elementwise"):
        if t in n: return t
    return n[:28]
def f(x, d=1): return "-" if x is None else (f"{x:,.{d}f}" if isinstance(x, float) else f"{x:,}")
rows = ["# s36 per-CTA work and reuse — " + os.path.basename(os.path.dirname(CI)) + " / " + W, "",
        "| pos | kernel | type | CTAs | per_wave | insts/CTA | HMMA | LDG | LDGSTS | STG | rd B/CTA | wr B/CTA | pages/CTA | page lookups/CTA | page reuse/CTA (L1 TLB) | line reuse/CTA |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
op_rows = ["", "## operand reuse (read side, all CTAs)", "", "| pos | kernel | operand | unique MB | total MB | reuse (bytes) | CTAs/line p50/max | same chiplet+wave % | same chiplet other wave % | other chiplet % |", "|---|---|---|---|---|---|---|---|---|---|"]
det_rows = ["", "## first CTA: operand bytes and row ranges (2-D tensors: [first row, last row, #rows])", ""]
for i, k in enumerate(ks, 1):
    p = f"{O}/k{k['id']}.json"
    if not os.path.exists(p): rows.append(f"| {i} | k{k['id']} | {short(k['name'])} | (pending) |"); continue
    o = json.load(open(p)); c = o["cta"]; mx = c["mix"]
    rows.append(f"| {i} | k{k['id']} | {short(k['name'])} | {o['n_cta']:,} | {o['per_wave']} | {f(c['insts']['p50'])} | {f(mx['HMMA']['p50'])} | {f(mx['LDG']['p50'])} | {f(mx['LDGSTS']['p50'])} | {f(mx['STG']['p50'])} | {f(c['bytes_rd']['p50'])} | {f(c['bytes_wr']['p50'])} | {f(c['pages']['p50'])} | {f((c.get('page_lookups') or {}).get('p50'))} | {f((c.get('intra_reuse_pages') or {}).get('p50'), 1)} | {f(c['intra_reuse_lines']['p50'] if c['intra_reuse_lines'] else None, 2)} |")
    for key, v in sorted(o["operands"].items(), key=lambda kv: -kv[1]["total_bytes"]):
        name, rw = key.rsplit("|", 1)
        if rw != "r" or v["total_bytes"] < 65536: continue
        r = o["operand_reuse"].get(name, {}); dc = r.get("dist_class", {}); tot = sum(dc.values()) - dc.get("first", 0)
        pc = lambda x: f"{100*dc.get(x,0)/tot:.0f}" if tot else "-"
        cpl = r.get("ctas_per_line") or {}
        op_rows.append(f"| {i} | k{k['id']} | {name} | {v['unique_bytes']/1048576:,.2f} | {v['total_bytes']/1048576:,.2f} | {f(v['reuse_factor_bytes'],2)} | {cpl.get('p50','-')}/{cpl.get('max','-')} | {pc('same_chiplet_same_wave')} | {pc('same_chiplet_other_wave')} | {pc('other_chiplet')} |")
    if o["details"]:
        d = o["details"][0]
        det_rows.append(f"- **{i}. k{k['id']} {short(k['name'])}** CTA {d['cid']} (chip {d['chip']}, wave {d['wave']}): " + "; ".join(
            f"{kk} {v['bytes']:,} B" + (f" rows {v['rows']}" if v['rows'] else "") for kk, v in sorted(d["operands"].items(), key=lambda kv: -kv[1]["bytes"]) if v["bytes"] >= 1024))
open(f"{O}/KERNELS.md", "w").write("\n".join(rows + op_rows + det_rows) + "\n")
print("wrote", f"{O}/KERNELS.md")
