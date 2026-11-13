# Usage: tlb_gemm_page_pattern.py <kernel-N-...traceg.xz>   (docs/llm_tlb_miss_3c_analysis.ko.md)
# Page-level access pattern of a GEMM kernel trace: pages per warp memory instruction, VPN stride inside one instruction,
# and per-CTA page reuse (how many separate instructions touch the same page).
import collections, subprocess, sys
path = sys.argv[1]
p = subprocess.Popen(["xz", "-dc", path], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
name = grid = None; cta = None
n_w = 0; n_mem = 0; pages_sum = 0; stride = collections.Counter(); per_cta = collections.defaultdict(collections.Counter)
ops = collections.Counter()
for line in p.stdout:
    if line.startswith("-kernel name"): name = line.split("=")[1].strip()[:70]; continue
    if line.startswith("-grid dim"): grid = line.split("=")[1].strip(); continue
    if line.startswith("thread block"): cta = line.split("=")[1].strip(); continue
    t = line.split()
    if len(t) < 5 or not t[2].isdigit(): continue
    n_w += 1
    i = 3 + int(t[2]); op = t[i]; i += 2 + int(t[i + 1]); w = int(t[i])
    if w == 0 or len(t) <= i + 1 or not op.startswith(("LDG", "STG", "LDGSTS", "ATOMG", "RED")): continue
    mode, n = int(t[i + 1]), bin(int(t[1], 16)).count("1")
    if mode == 0: a = [int(x, 16) for x in t[i + 2:i + 2 + n]]
    elif mode == 1: b, s = int(t[i + 2], 16), int(t[i + 3]); a = [b + k * s for k in range(n)]
    else:
        a = [int(t[i + 2], 16)]
        for d in t[i + 3:i + 2 + n]: a.append(a[-1] + int(d))
    pg = sorted({x >> 12 for x in a})
    n_mem += 1; pages_sum += len(pg); ops[op.split(".")[0]] += 1
    for x, y in zip(pg, pg[1:]): stride[y - x] += 1
    for x in pg: per_cta[cta][x] += 1
p.wait()
touches = sum(sum(c.values()) for c in per_cta.values()); distinct = sum(len(c) for c in per_cta.values())
print(f"{name}  grid {grid}  CTAs {len(per_cta)}")
print(f"  warp insn {n_w:,}, global mem warp insn {n_mem:,} {dict(ops)}; avg pages per mem insn {pages_sum/max(n_mem,1):.2f}; page touches per kilo warp insn {pages_sum/n_w*1000:.0f}")
print(f"  VPN gap between pages of one instruction (top): {[(g, f'{c/sum(stride.values())*100:.0f}%') for g, c in stride.most_common(4)]}")
print(f"  per CTA: distinct pages {distinct/len(per_cta):.0f}, instructions touching each page {touches/distinct:.1f} (re-visits)")
