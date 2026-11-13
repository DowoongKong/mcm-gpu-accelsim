#!/usr/bin/env python3
# S25-2/3/4/5 in one streaming pass over an Accel-Sim .traceg.xz kernel trace (docs/s25_decode_kernel_ldst_method.ko.md).
# Per PC: opcode, memory space, width, dynamic count, lane/sector/line statistics, lane-stride signature, loop stride,
# operand split. Per CTA and operand: byte counts (read/write), 128 B line intervals, 4 KB page byte counts. Then, from the
# per-CTA data: inter-CTA sharing at 5 granularities (byte/line/4K/64K/2M) with sub-group inference, and partition locality
# under 4 KB RR, 64 KB RR, a reference placement map (data owners D and PT-node owners N), and a per-page majority ceiling.
# CTA -> chiplet follows LASP policy 4 (KERNEL_PARTITION): lin = x + y*gx + z*gx*gy, chunk = ceil(n/4), chip = lin // chunk.
# Operands: weight regions come from the MemcpyHtoD list (one operand per contiguous group of copies); every other address
# is bucketed by 64 KB page and merged into contiguous "R<n>" regions after the pass (to be renamed from a VA map later).
# Usage: s25_kernel_profile.py <kernelslist.g> <kernel .traceg.xz> <outdir> [--refmap placement_mcm.map] [--nchip 4]
import bisect, collections, json, lzma, math, os, sys, time

args = sys.argv[1:]
refmap = None; nchip = 4
if '--refmap' in args: i = args.index('--refmap'); refmap = args[i + 1]; del args[i:i + 2]
if '--nchip' in args: i = args.index('--nchip'); nchip = int(args[i + 1]); del args[i:i + 2]
klist, tracef, outdir = args[0], args[1], args[2]
os.makedirs(outdir, exist_ok=True)
t0 = time.time()

# ---------------- weight operands from the memcpy list ----------------
copies = sorted((int(l.split(',')[1], 16), int(l.split(',')[2])) for l in open(klist) if l.startswith('MemcpyHtoD'))
copies = [(b, s) for b, s in copies if s >= 1 << 20]          # weight-sized copies only (>= 1 MB)
wregions = []                                                 # merge copies that are adjacent in VA (vLLM merged qkv / gate_up)
for b, s in copies:
    if wregions and b == wregions[-1][0] + wregions[-1][1]: wregions[-1][1] += s; wregions[-1][2] += 1
    else: wregions.append([b, s, 1])
wstarts = [r[0] for r in wregions]
def weight_region(a):
    i = bisect.bisect_right(wstarts, a) - 1
    if i >= 0 and a < wregions[i][0] + wregions[i][1]: return i
    return -1

# ---------------- reference placement map ----------------
ref_d, ref_n = {}, {}
if refmap:
    for l in open(refmap):
        p = l.split()
        if not p: continue
        if p[0] == 'D': ref_d[int(p[1], 0)] = int(p[2])
        elif p[0] == 'N' and p[1] == '3': ref_n[int(p[2], 0)] = int(p[3])

# ---------------- trace line decoding ----------------
def addrs(t, k, act):
    fmt = int(t[k + 1]); base = int(t[k + 2], 16)
    if fmt == 1:
        s = int(t[k + 3]); return [base + i * s for i in range(act)]
    if fmt == 2:
        out = [base]; a = base
        for d in t[k + 3:k + 3 + act - 1]: a += int(d); out.append(a)
        return out
    return [int(x, 16) for x in t[k + 2:k + 2 + act]]
GLOBAL_OPS = {'LDG', 'STG', 'LDGSTS', 'ATOM', 'ATOMG', 'RED', 'LD', 'ST'}
SHARED_OPS = {'LDS', 'STS', 'LDSM', 'ATOMS'}
LOCAL_OPS = {'LDL', 'STL'}
def space_of(op):
    if op in GLOBAL_OPS: return 'G'
    if op in SHARED_OPS: return 'S'
    if op in LOCAL_OPS: return 'L'
    return '?'
def is_write(op): return op in {'STG', 'ST', 'STS', 'STL', 'RED', 'ATOM', 'ATOMG', 'ATOMS'}
def lane_sig(A, w):
    # signature of the sorted lane addresses: contiguous run length at stride w, then the first jump
    if len(A) == 1: return '1L'
    S = sorted(set(A)); d = [S[i + 1] - S[i] for i in range(len(S) - 1)]
    run = 1
    for x in d:
        if x == w: run += 1
        else: break
    jump = next((x for x in d if x != w), 0)
    bc = len(A) - len(S)
    return f'{len(A)}L:{run}x{w}' + (f'+{jump}' if jump else '') + (f' bcast{bc}' if bc else '')

# ---------------- accumulators ----------------
hdr = {}
pc_stats = {}                     # pc -> dict
cta_bytes = {}                    # tb -> {operand: [rbytes, wbytes]}
cta_lines = {}                    # tb -> {operand: [sorted line intervals (start, n)]} built at CTA end
cta_page4k = {}                   # tb -> Counter{(operand, page4k): bytes}
nonw_pages = set()                # 64 KB pages of non-weight accesses (for region clustering)
cur_tb = None; cur_lines = None; cur_bytes = None; cur_p4k = None; warp = None
last_base = {}                    # (warp, pc) -> last base address for loop stride
n_lines = 0; n_mem = 0

def flush_cta():
    global cur_tb, cur_lines
    if cur_tb is None: return
    iv = {}
    for opnd, (rs, ws) in cur_lines.items():
        out = {}
        for tag, s in (('r', rs), ('w', ws)):
            if not s: continue
            L = sorted(s); runs = []; start = L[0]; prev = L[0]
            for x in L[1:]:
                if x == prev + 1: prev = x; continue
                runs.append((start, prev - start + 1)); start = prev = x
            runs.append((start, prev - start + 1)); out[tag] = runs
        iv[opnd] = out
    cta_lines[cur_tb] = iv; cta_bytes[cur_tb] = cur_bytes; cta_page4k[cur_tb] = cur_p4k

def operand_key(a):
    wi = weight_region(a)
    if wi >= 0: return ('W', wi)
    p = a >> 16; nonw_pages.add(p); return ('N', p)

shm_base = loc_base = -1; shm_win = loc_win = 0
with lzma.open(tracef, 'rt') as fh:
    for line in fh:
        if line[0] == '-':
            k, _, v = line[1:].partition('='); hdr[k.strip()] = v.strip()
            if k.strip() == 'shmem base_addr': shm_base = int(v, 16); shm_win = 1 << 24
            if k.strip() == 'local mem base_addr': loc_base = int(v, 16); loc_win = 1 << 24
            continue
        if line.startswith('thread block'):
            flush_cta()
            cur_tb = tuple(int(v) for v in line.split('=')[1].split(','))
            cur_lines = collections.defaultdict(lambda: (set(), set())); cur_bytes = collections.defaultdict(lambda: [0, 0])
            cur_p4k = collections.Counter(); last_base = {}
            continue
        if line.startswith('warp'): warp = int(line.split('=')[1]); continue
        t = line.split()
        if len(t) < 4 or line[0] in '#i' or not t[0].isalnum(): continue
        try: dn = int(t[2])
        except ValueError: continue
        n_lines += 1
        oi = 3 + dn; opfull = t[oi]; op = opfull.split('.')[0]
        sn = int(t[oi + 1]); kk = oi + 2 + sn
        if kk >= len(t): continue
        w = int(t[kk])
        if w == 0: continue
        act = bin(int(t[1], 16)).count('1')
        if act == 0: continue
        A = addrs(t, kk, act); n_mem += 1
        sp = space_of(op); wr = is_write(op)
        if op in ('LD', 'ST'):   # generic addressing: resolve by the shared / local windows of the trace header
            a0 = A[0]
            if shm_base <= a0 < shm_base + shm_win: sp = 'S'
            elif loc_base <= a0 < loc_base + loc_win: sp = 'L'
        st = pc_stats.get(t[0])
        if st is None:
            st = pc_stats[t[0]] = {'op': opfull, 'space': sp, 'width': w, 'n': 0, 'lanes': 0, 'sectors': 0, 'lines': 0,
                                  'bytes': 0, 'operands': collections.Counter(), 'lane_sig': collections.Counter(),
                                  'loop_stride': collections.Counter(), 'mask_pop': collections.Counter(), 'write': wr}
        st['n'] += 1; st['lanes'] += act; st['bytes'] += act * w; st['mask_pop'][act] += 1
        st['sectors'] += len({a >> 5 for a in A}); st['lines'] += len({a >> 7 for a in A})
        st['lane_sig'][lane_sig(A, w)] += 1
        lb = last_base.get((warp, t[0]))
        if lb is not None: st['loop_stride'][A[0] - lb] += 1
        last_base[(warp, t[0])] = A[0]
        if sp != 'G': continue
        for a in A:
            key = operand_key(a)
            st['operands'][key] += w
            cb = cur_bytes[key]; cb[1 if wr else 0] += w
            cur_lines[key][1 if wr else 0].add(a >> 7)
            cur_p4k[(key, a >> 12)] += w
flush_cta()
t_parse = time.time() - t0

# ---------------- non-weight regions: merge adjacent 64 KB pages ----------------
regions = []
for p in sorted(nonw_pages):
    if regions and p == regions[-1][1] + 1: regions[-1][1] = p
    else: regions.append([p, p])
rstart = [r[0] for r in regions]
def canon(key):
    if key[0] == 'W': return f'W{key[1]}'
    i = bisect.bisect_right(rstart, key[1]) - 1
    return f'R{i}'
def remap_counter(c):
    out = collections.Counter()
    for k, v in c.items(): out[canon(k)] += v
    return out
operand_names = {f'W{i}': {'va': hex(r[0]), 'bytes': r[1], 'copies': r[2]} for i, r in enumerate(wregions)}
for i, (p0, p1) in enumerate(regions):
    operand_names[f'R{i}'] = {'va': hex(p0 << 16), 'bytes': (p1 - p0 + 1) << 16, 'pages64k': p1 - p0 + 1}
for tb in cta_bytes:
    nb = collections.defaultdict(lambda: [0, 0])
    for k, v in cta_bytes[tb].items(): nb[canon(k)][0] += v[0]; nb[canon(k)][1] += v[1]
    cta_bytes[tb] = dict(nb)
    nl = {}
    for k, v in cta_lines[tb].items():
        d = nl.setdefault(canon(k), {})
        for tag, runs in v.items(): d.setdefault(tag, []).extend(runs)
    cta_lines[tb] = nl
    np4 = collections.Counter()
    for (k, pg), v in cta_page4k[tb].items(): np4[(canon(k), pg)] += v
    cta_page4k[tb] = np4
for st in pc_stats.values(): st['operands'] = remap_counter(st['operands'])
used = sorted({k for tb in cta_bytes for k in cta_bytes[tb]}, key=lambda s: (s[0], int(s[1:])))

# ---------------- CTA -> chiplet (LASP 4) ----------------
def hdr3(v): return tuple(int(x) for x in v.strip('() ').split(','))
gx, gy, gz = hdr3(hdr.get('grid dim', '(1,1,1)')); bx, by, bz = hdr3(hdr.get('block dim', '(1,1,1)'))
ncta = gx * gy * gz; chunk = math.ceil(ncta / nchip)
def chip_of(tb): return min(nchip - 1, (tb[0] + tb[1] * gx + tb[2] * gx * gy) // chunk)

# ---------------- sharing at 5 granularities + sub-group inference ----------------
def units_of(runs, shift):
    # line runs (start_line, n) -> set of units at granularity 2^shift bytes (shift >= 7)
    s = shift - 7; out = set()
    for a, n in runs:
        out.update(range(a >> s, ((a + n - 1) >> s) + 1))
    return out
sharing = {}
for opnd in used:
    row = {}
    for gname, shift in (('line128', 7), ('page4k', 12), ('page64k', 16), ('pt2m', 21)):
        cover = collections.defaultdict(list)
        for tb, d in cta_lines.items():
            if opnd not in d: continue
            runs = d[opnd].get('r', []) + d[opnd].get('w', [])
            for u in units_of(runs, shift): cover[u].append(tb)
        if not cover: continue
        hist = collections.Counter(len(v) for v in cover.values())
        tot = len(cover); shared = sum(1 for v in cover.values() if len(v) > 1)
        # sub-group inference: among sharers of a unit, which grid coordinates are equal
        eq = collections.Counter(); nsh = 0
        for v in cover.values():
            if len(v) < 2: continue
            nsh += 1
            eq['x'] += all(c[0] == v[0][0] for c in v); eq['y'] += all(c[1] == v[0][1] for c in v); eq['z'] += all(c[2] == v[0][2] for c in v)
        row[gname] = {'units': tot, 'shared_units': shared, 'degree_hist': dict(sorted(hist.items())),
                      'max_degree': max(hist), 'sharers_equal_coord_frac': {k: round(eq[k] / nsh, 3) for k in ('x', 'y', 'z')} if nsh else {}}
    # byte-level sharing: intervals exact at 128 B lines already cover bytes for 128 B-aligned accesses; report line128 as proxy
    nctas = sum(1 for tb in cta_bytes if opnd in cta_bytes[tb])
    l = row.get('line128', {})
    cls = 'private' if l and l.get('shared_units', 0) == 0 else ('global' if l and l.get('max_degree', 0) >= nctas and nctas > 1 else 'partial')
    sharing[opnd] = {'ctas': nctas, 'class_line128': cls, **row}

# ---------------- partition locality ----------------
locality = {}
for opnd in used:
    tot = 0; loc = collections.Counter(); by64 = collections.defaultdict(collections.Counter); by2m = collections.defaultdict(collections.Counter)
    for tb, cnt in cta_page4k.items():
        c = chip_of(tb)
        for (k, pg), v in cnt.items():
            if k != opnd: continue
            tot += v
            loc['rr4k'] += v if pg % nchip == c else 0
            p64 = pg >> 4; loc['rr64k'] += v if p64 % nchip == c else 0
            own = ref_d.get(p64, -1); loc['refmap'] += v if own == c else 0; loc['refmap_known'] += v if own >= 0 else 0
            k2m = pg >> 9; nown = ref_n.get(k2m, -1); loc['pt_refmap'] += v if nown == c else 0; loc['pt_rr'] += v if k2m % nchip == c else 0
            by64[p64][c] += v; by2m[k2m][c] += v
    if tot == 0: continue
    ceil64 = sum(max(cc.values()) for cc in by64.values()); ceil2m = sum(max(cc.values()) for cc in by2m.values())
    locality[opnd] = {'bytes': tot, 'rr4k': round(loc['rr4k'] / tot, 4), 'rr64k': round(loc['rr64k'] / tot, 4),
                      'refmap': round(loc['refmap'] / tot, 4) if loc['refmap_known'] else None,
                      'refmap_coverage': round(loc['refmap_known'] / tot, 4),
                      'ceiling_page64k_majority': round(ceil64 / tot, 4), 'pt_node_rr': round(loc['pt_rr'] / tot, 4),
                      'pt_node_refmap': round(loc['pt_refmap'] / tot, 4) if loc['refmap_known'] else None,
                      'pt_node_ceiling_majority': round(ceil2m / tot, 4), 'pages64k': len(by64), 'pt_nodes': len(by2m)}

# ---------------- per-CTA operand footprint summary ----------------
fp = {}
for opnd in used:
    rb = [cta_bytes[tb][opnd][0] for tb in cta_bytes if opnd in cta_bytes[tb]]
    wb = [cta_bytes[tb][opnd][1] for tb in cta_bytes if opnd in cta_bytes[tb]]
    ul = [sum(n for tag in cta_lines[tb][opnd] for _, n in cta_lines[tb][opnd][tag]) for tb in cta_lines if opnd in cta_lines[tb]]
    fp[opnd] = {'ctas': len(rb), 'read_bytes_per_cta': {'min': min(rb), 'max': max(rb), 'mean': round(sum(rb) / len(rb))},
                'write_bytes_per_cta': {'min': min(wb), 'max': max(wb), 'mean': round(sum(wb) / len(wb))},
                'unique_lines_per_cta': {'min': min(ul), 'max': max(ul), 'mean': round(sum(ul) / len(ul))},
                'total_read_bytes': sum(rb), 'total_write_bytes': sum(wb)}

# ---------------- outputs ----------------
kid = hdr.get('kernel id', '?')
with open(os.path.join(outdir, f'k{kid}_pc.tsv'), 'w') as f:
    f.write('pc\topcode\tspace\twidth\tdyn\tlanes_avg\tsectors_per_req\tlines_per_req\tbytes\twrite\tmask_pop\tlane_sig_top\tloop_stride_top\toperands\n')
    for pc, st in sorted(pc_stats.items(), key=lambda x: int(x[0], 16)):
        f.write('\t'.join([pc, st['op'], st['space'], str(st['width']), str(st['n']), f"{st['lanes']/st['n']:.1f}",
                           f"{st['sectors']/st['n']:.2f}", f"{st['lines']/st['n']:.2f}", str(st['bytes']), str(int(st['write'])),
                           ';'.join(f'{k}:{v}' for k, v in st['mask_pop'].most_common(3)),
                           ';'.join(f'{k}:{v}' for k, v in st['lane_sig'].most_common(2)),
                           ';'.join(f'{k}:{v}' for k, v in st['loop_stride'].most_common(3)),
                           ';'.join(f'{k}:{v}' for k, v in st['operands'].most_common(4))]) + '\n')
json.dump({'tb': [list(tb) for tb in cta_lines], 'chip': [chip_of(tb) for tb in cta_lines],
           'bytes': [cta_bytes[tb] for tb in cta_lines], 'lines': [cta_lines[tb] for tb in cta_lines]},
          open(os.path.join(outdir, f'k{kid}_cta.json'), 'w'))
summary = {'kernel': hdr.get('kernel name'), 'kernel_id': kid, 'grid': [gx, gy, gz], 'block': [bx, by, bz], 'ctas': ncta,
           'chunk_per_chiplet': chunk, 'shmem': hdr.get('shmem'), 'nregs': hdr.get('nregs'),
           'warp_insts_total': n_lines, 'mem_insts_total': n_mem, 'seconds_parse': round(t_parse, 1),
           'operands': {k: operand_names[k] for k in used}, 'footprint': fp, 'sharing': sharing, 'locality': locality,
           'pc_summary': {sp: {'static': sum(1 for s in pc_stats.values() if s['space'] == sp),
                               'dyn': sum(s['n'] for s in pc_stats.values() if s['space'] == sp),
                               'bytes': sum(s['bytes'] for s in pc_stats.values() if s['space'] == sp)} for sp in ('G', 'S', 'L')}}
json.dump(summary, open(os.path.join(outdir, f'k{kid}_summary.json'), 'w'), indent=1)
print(f"k{kid} {hdr.get('kernel name','')[:60]} grid={gx},{gy},{gz} ctas={ncta} warp_insts={n_lines:,} mem={n_mem:,} parse={t_parse:.0f}s total={time.time()-t0:.0f}s")
for opnd in used:
    L = locality.get(opnd, {}); S = sharing[opnd]
    print(f"  {opnd:4s} {operand_names[opnd]['va']} {operand_names[opnd]['bytes']/2**20:9.1f} MB  ctas={S['ctas']:4d} class={S['class_line128']:7s}"
          f" deg64k_max={S.get('page64k',{}).get('max_degree','-')} rd/cta={fp[opnd]['read_bytes_per_cta']['mean']/1024:8.1f} KB"
          f" loc rr4k={L.get('rr4k')} rr64k={L.get('rr64k')} ref={L.get('refmap')} ceil={L.get('ceiling_page64k_majority')} pt_ref={L.get('pt_node_refmap')} pt_ceil={L.get('pt_node_ceiling_majority')}")
