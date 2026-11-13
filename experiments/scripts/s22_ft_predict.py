#!/usr/bin/env python3
# Offline first-touch placement predictor (feasibility test). For kernels k_from..k_to of a kernel list, in order:
# every CTA is bound to chiplet lin // ceil(nCTA/4) (KERNEL_PARTITION), issued in linear order per chiplet with
# 64 SMs x ctas_per_sm concurrent CTAs (wave index), and a page's first touch on a chiplet is approximated by the
# earliest (wave, instruction index within the warp) among that chiplet's CTAs. The predicted owner is the chiplet with
# the smallest key; ties go to the lower chiplet index (the simulator's same-cycle commit order). Pages are keyed by
# va >> 16 (64 KB). Then compare with the reference run's placement map (D lines).
# Usage: ft_predict.py <kernel list .g> <k_from> <k_to> <placement_mcm.map> <out prefix>
import lzma, math, os, re, sys
klist, k_from, k_to, mapf, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
N_CHIP, SMS_PER_CHIP, SHIFT = 4, 64, 16
MEM = re.compile(r'\b(LDG|STG|LDGSTS|LD|ST|ATOM|ATOMG|RED)\b')
files = [l.strip() for l in open(klist) if 'kernel-' in l]
def addrs(t, k):
    if int(t[k]) == 0: return []
    fmt = int(t[k+1]); base = int(t[k+2], 16); act = bin(int(t[1], 16)).count('1')
    if fmt == 1: s = int(t[k+3]); return [base + i*s for i in range(act)]
    if fmt == 2:
        out = [base]; a = base
        for d in t[k+3:k+3+act-1]: a += int(d); out.append(a)
        return out
    return [int(x, 16) for x in t[k+2:k+2+act]]
owner = {}; owner_key = {}; owner_kernel = {}
def hdr_int3(v): return tuple(int(x) for x in v.strip('() ').split(','))
for ki in range(k_from, k_to + 1):
    f = os.path.join(os.path.dirname(klist), files[ki - 1]); h = {}
    best = {}  # page -> (wave, insn_idx, chiplet)
    with lzma.open(f, 'rt') as fh:
        for line in fh:
            if line.startswith('-'):
                k, v = line.lstrip('-').split('=', 1); h[k.strip()] = v.strip(); continue
            if line.startswith('#'):
                if 'grid dim' not in h: continue
                gx, gy, gz = hdr_int3(h['grid dim']); bx, by, bz = hdr_int3(h['block dim'])
                threads = bx * by * bz; nregs = int(h.get('nregs', 32)); shmem = int(h.get('shmem', 0))
                n_cta = gx * gy * gz; chunk = math.ceil(n_cta / N_CHIP)
                regs_per_warp = math.ceil(nregs * 32 / 256) * 256; warps = math.ceil(threads / 32)
                cps = min(32, 2048 // threads, 65536 // (regs_per_warp * warps), (102400 // shmem) if shmem else 32)
                cps = max(1, cps); per_wave = SMS_PER_CHIP * cps
                continue
            if line.startswith('thread block ='):
                x, y, z = (int(v) for v in line.split('=')[1].split(','))
                lin = x + y * gx + z * gx * gy; chip = min(N_CHIP - 1, lin // chunk)
                wave = (lin - chip * chunk) // per_wave; insn = 0; continue
            if line.startswith('warp ='): insn = 0; continue
            t = line.split()
            if len(t) < 5 or not re.fullmatch(r'[0-9a-f]{4,}', t[0]): continue
            insn += 1
            if not MEM.search(line): continue
            dn = int(t[2]); oi = 3 + dn; sn = int(t[oi+1]); kk = oi + 2 + sn
            try: A = addrs(t, kk)
            except Exception: continue
            key = (wave, insn, chip)
            for a in A:
                p = a >> SHIFT; b = best.get(p)
                if b is None or key < b: best[p] = key
    new = 0
    for p, (w, i, c) in best.items():
        if p not in owner: owner[p] = c; owner_key[p] = (w, i); owner_kernel[p] = ki; new += 1
    print(f"k{ki:02d} {h.get('kernel name','')[:40]:40s} grid={h.get('grid dim')} ctas/SM={cps} per_wave={per_wave} pages_touched={len(best):,} new_pages={new:,}", flush=True)
# compare with the reference map
ref = {}
for line in open(mapf):
    if line.startswith('D '):
        t = line.split(); ref[int(t[1], 0)] = int(t[2])
common = [p for p in owner if p in ref]
agree = sum(1 for p in common if owner[p] == ref[p])
print(f"predicted pages={len(owner):,} in ref map={len(common):,} (ref map D total {len(ref):,}) agree={agree:,} ({100*agree/max(1,len(common)):.2f}%)")
by_k = {}
for p in common:
    d = by_k.setdefault(owner_kernel[p], [0, 0]); d[0] += 1; d[1] += (owner[p] == ref[p])
for k in sorted(by_k): print(f"  first touched in k{k:02d}: pages={by_k[k][0]:,} agree={100*by_k[k][1]/by_k[k][0]:.2f}%")
with open(outp + '_pred.map', 'w') as fo:
    for p in sorted(owner): fo.write(f"D 0x{p:x} {owner[p]} k{owner_kernel[p]} w{owner_key[p][0]} i{owner_key[p][1]} ref={ref.get(p,-1)}\n")
