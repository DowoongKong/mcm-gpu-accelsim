#!/usr/bin/env python3
# Count, for one accel-sim .traceg.xz kernel trace: memory warp instructions, distinct 64 KB (and 4 KB) pages touched per
# memory warp instruction, and the distinct-page working set per CTA. Address formats: 1 = base+stride, 2 = base+deltas,
# 0 = full list. Usage: trace_pages.py <trace.xz> [max_ctas_for_ws]
import lzma, re, sys
f = sys.argv[1]; maxws = int(sys.argv[2]) if len(sys.argv) > 2 else 64
MEM = re.compile(r'\b(LDG|STG|LDGSTS|LD|ST|ATOM|ATOMG|RED|LDL|STL)\b')
n_mem = 0; sum_p64 = 0; sum_p4 = 0; n_lanes = 0; hist = {}
cta_ws = []; cur_ws64 = None; cur_ws4 = None; n_cta = 0; mem_per_cta = []; cur_mem = 0
def addrs(t, k):  # t = tokens, k = index of mem_width; returns list of addresses
    mw = int(t[k]); 
    if mw == 0: return []
    fmt = int(t[k+1]); base = int(t[k+2], 16)
    mask = int(t[1], 16); act = bin(mask).count('1')
    if fmt == 1:
        stride = int(t[k+3]); return [base + i*stride for i in range(act)]
    if fmt == 2:
        out = [base]; a = base
        for d in t[k+3:k+3+act-1]: a += int(d); out.append(a)
        return out
    return [int(x, 16) for x in t[k+2:k+2+act]]
with lzma.open(f, 'rt') as fh:
    for line in fh:
        if line.startswith('thread block ='):
            if cur_ws64 is not None and n_cta <= maxws: cta_ws.append((len(cur_ws64), len(cur_ws4))); mem_per_cta.append(cur_mem)
            n_cta += 1; cur_ws64 = set(); cur_ws4 = set(); cur_mem = 0; continue
        t = line.split()
        if len(t) < 5 or not re.fullmatch(r'[0-9a-f]{4,}', t[0]): continue
        if not MEM.search(line): continue
        # locate opcode: t[2]=dest_num, dests follow, then opcode, src_num, srcs, mem_width
        dn = int(t[2]); op_i = 3 + dn; sn = int(t[op_i+1]); k = op_i + 2 + sn
        try: A = addrs(t, k)
        except Exception as e: continue
        if not A: continue
        n_mem += 1; cur_mem += 1; n_lanes += len(A)
        p64 = {a >> 16 for a in A}; p4 = {a >> 12 for a in A}
        sum_p64 += len(p64); sum_p4 += len(p4); hist[len(p64)] = hist.get(len(p64), 0) + 1
        if n_cta <= maxws: cur_ws64 |= p64; cur_ws4 |= p4
    if cur_ws64 is not None and n_cta <= maxws: cta_ws.append((len(cur_ws64), len(cur_ws4))); mem_per_cta.append(cur_mem)
print(f"file={f}")
print(f"ctas={n_cta} mem_warp_insn={n_mem:,} lanes={n_lanes:,} sum_pages64={sum_p64:,} avg_pages64/insn={sum_p64/max(1,n_mem):.2f} sum_pages4={sum_p4:,} avg_pages4/insn={sum_p4/max(1,n_mem):.2f}")
print("pages64/insn histogram:", dict(sorted(hist.items())))
if cta_ws:
    import statistics
    print(f"per-CTA working set over first {len(cta_ws)} CTAs: pages64 mean={statistics.mean(w[0] for w in cta_ws):.0f} min={min(w[0] for w in cta_ws)} max={max(w[0] for w in cta_ws)} | pages4 mean={statistics.mean(w[1] for w in cta_ws):.0f} | mem insn/CTA mean={statistics.mean(mem_per_cta):.0f}")
