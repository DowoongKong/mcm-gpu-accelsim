#!/usr/bin/env python3
"""Recover the active experts of one fused_moe kernel from its accel-sim trace.

Usage: s27_moe_routing_from_trace.py <trace.traceg.xz> <weight base hex> <bytes per expert> <n experts>
Every memory address inside [base, base + n * bytes) is mapped to expert = (addr - base) // bytes.
Per CTA: the experts whose weights it read (a CTA whose M-block lies past num_tokens_post_padded exits
before touching weights). Prints CTAs per expert, idle CTAs and warp instructions per expert.
"""
import lzma
import sys
from collections import Counter

path, base, per, n = sys.argv[1], int(sys.argv[2], 16), int(sys.argv[3]), int(sys.argv[4])
end = base + per * n


def addrs(t, k):
    if int(t[k]) == 0:
        return []
    fmt = int(t[k + 1]); b = int(t[k + 2], 16); act = bin(int(t[1], 16)).count("1")
    if fmt == 1:
        s = int(t[k + 3]); return [b + i * s for i in range(act)]
    if fmt == 2:
        out = [b]; a = b
        for d in t[k + 3:k + 3 + act - 1]:
            a += int(d); out.append(a)
        return out
    return [int(x, 16) for x in t[k + 2:k + 2 + act]]


cta_exp = []   # per CTA: set of experts
cta_insn = []  # per CTA: warp instructions
cur = None
for line in lzma.open(path, "rt"):
    if line.startswith("thread block"):
        cur = [set(), 0]; cta_exp.append(cur[0]); cta_insn.append(0); continue
    if cur is None or not line[:1].isdigit() and not line.startswith("0"):
        continue
    t = line.split()
    try:
        nd = int(t[2]); op = t[3 + nd]; ns = int(t[4 + nd]); k = 5 + nd + ns
    except (ValueError, IndexError):
        continue
    cta_insn[-1] += 1
    if not (op.startswith("LD") or op.startswith("ST")):
        continue
    try:
        for a in addrs(t, k):
            if base <= a < end:
                cur[0].add((a - base) // per)
    except (ValueError, IndexError):
        pass

q = len(cta_exp) // 4
print("per chiplet (KERNEL_PARTITION quarters): " + " | ".join(f"c{c}: active {sum(1 for s in cta_exp[c*q:(c+1)*q] if s)}/{q}, experts {sorted(set().union(*cta_exp[c*q:(c+1)*q]))}, insn {sum(cta_insn[c*q:(c+1)*q])/1e6:.2f}M" for c in range(4)))
per_exp = Counter(); insn_exp = Counter(); idle = 0; multi = 0
for s, ni in zip(cta_exp, cta_insn):
    if not s:
        idle += 1; continue
    if len(s) > 1:
        multi += 1
    for e in s:
        per_exp[e] += 1; insn_exp[e] += ni / len(s)
print(f"{path.split('/')[-1]}: CTAs {len(cta_exp)}, idle (no weight read) {idle}, multi-expert CTAs {multi}")
print(f"active experts {len(per_exp)}: " + ", ".join(f"e{e}:{per_exp[e]} CTA/{insn_exp[e]/1e6:.2f}M insn" for e in sorted(per_exp)))
