#!/usr/bin/env python3
# Per-page, per-chiplet first-touch candidates of ONE kernel: best (wave, insn_idx, local_cta_idx) per chiplet and the
# number of (warp instruction, page) accesses per chiplet. Output: JSON {page: {chip: [wave, insn, local, count]}}.
# Usage: s22_ft_predict2.py <kernel list .g> <k> <out.json> [--config gpgpusim.config] [--opt "-name value"]... [--force]
#   --config  (step 26-1) take n_chiplets, SM count, occupancy limits and the page shift from the simulator config
#             instead of the built-in constants (4 chiplets, 64 SM/chiplet, 64 KB pages, 2048 thr / 65536 regs /
#             32 CTA / 102400 B shmem per SM) and refuse a config whose CTA->chiplet rule is not the one modelled here
#             (-gpgpu_lasp_policy 4 KERNEL_PARTITION with -gpgpu_cta_chiplet_cursor 1); --force only prints the refusal.
#   --opt     extra "-name value" pairs that the run adds on top of the file (later values win, like the simulator).
# Model (llm_sim_tree_pwcstat_l1fwd, 2026-11-10):
#   chiplet  = ((x + y*gx + z*gx*gy) // ceil(n_cta / n_chiplets)) % n_chiplets        shader.cc issue_block2core,
#                                                                                       gpu-sim.cc KERNEL_PARTITION
#   CTA/SM   = shader_core_config::max_cta(): min(thr/padded_cta, shmem/smem, regs/(padded_cta*round4(regs)), max_cta),
#              then ceil(n_cta / n_SM) when the grid cannot fill every SM                shader.cc:4443
#   wave     = local_cta // (SM per chiplet * CTA/SM): each chiplet issues its own CTAs in id order (PCC cursor)
#   page key = va >> page_shift (uniform -gpgpu_page_size_policy 0/1/2 -> 12/16/21)     gpu-sim.cc decide_page_shift
#   address space of generic LD/ST = the simulator's own test on the FIRST active address (trace_driven.cc:353-375):
#              shared iff shmem_base <= a < local_base, local iff local_base <= a < local_base + 16 KB, else global;
#              only shared accesses skip translation (local ones are translated). With local_base < shmem_base, as in
#              our captures, the shared test never fires and shmem-window generic loads count as global -- reproduced
#              on purpose so the map matches what the simulator translates (SIM-FIND-1, docs/s25b... section 5.3).
import json, lzma, math, os, re, sys

def parse_args(argv):
    pos, opts, extra, force = [], None, [], False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--config': opts = argv[i + 1]; i += 2
        elif a == '--opt': extra.append(argv[i + 1]); i += 2
        elif a == '--force': force = True; i += 1
        else: pos.append(a); i += 1
    if len(pos) != 3: sys.exit(__doc__ or 'usage: s22_ft_predict2.py <klist> <k> <out.json> [--config cfg] [--opt "-x v"]')
    return pos[0], int(pos[1]), pos[2], opts, extra, force

def read_config(path, extra):
    vals = {}
    def take(tokens):
        j = 0
        while j < len(tokens):
            if tokens[j].startswith('-') and not re.fullmatch(r'-?\d+(\.\d+)?', tokens[j]):
                vals[tokens[j]] = tokens[j + 1] if j + 1 < len(tokens) and not tokens[j + 1].startswith('-') else ''
                j += 2
            else: j += 1
    for line in open(path):
        line = line.split('#', 1)[0].strip()
        if line: take(line.split())
    for e in extra: take(e.split())
    return vals

def model_from_config(vals, force):
    need = ['-gpgpu_n_chiplets', '-gpgpu_n_clusters', '-gpgpu_shader_registers', '-gpgpu_shader_core_pipeline',
            '-gpgpu_shader_cta', '-gpgpu_shmem_size', '-gpgpu_page_size_policy', '-gpgpu_lasp_policy',
            '-gpgpu_cta_chiplet_cursor']
    missing = [n for n in need if n not in vals]
    if missing: sys.exit(f'config lacks {missing}')
    problems = []
    if vals['-gpgpu_lasp_policy'] != '4': problems.append(f"-gpgpu_lasp_policy {vals['-gpgpu_lasp_policy']} (model needs 4 KERNEL_PARTITION)")
    if vals['-gpgpu_cta_chiplet_cursor'] != '1': problems.append(f"-gpgpu_cta_chiplet_cursor {vals['-gpgpu_cta_chiplet_cursor']} (model needs 1)")
    shift = {'0': 12, '1': 16, '2': 21}.get(vals['-gpgpu_page_size_policy'])
    if shift is None: problems.append(f"-gpgpu_page_size_policy {vals['-gpgpu_page_size_policy']} is size-dependent (3-5); one page key per VA is impossible")
    if problems:
        print('REFUSE: ' + '; '.join(problems), file=sys.stderr)
        if not force or shift is None: sys.exit(2)
    n_chip = int(vals['-gpgpu_n_chiplets']); n_sm = int(vals['-gpgpu_n_clusters']) * int(vals.get('-gpgpu_n_cores_per_cluster', '1'))
    thr, warp = (int(v) for v in vals['-gpgpu_shader_core_pipeline'].split(':'))
    return {'n_chip': n_chip, 'n_sm': n_sm, 'sms_per_chip': n_sm // n_chip, 'shift': shift, 'n_thread_per_shader': thr,
            'warp_size': warp, 'regs_per_sm': int(vals['-gpgpu_shader_registers']), 'max_cta_per_core': int(vals['-gpgpu_shader_cta']),
            'shmem_size': int(vals['-gpgpu_shmem_size']), 'source': 'config'}

BUILTIN = {'n_chip': 4, 'n_sm': 256, 'sms_per_chip': 64, 'shift': 16, 'n_thread_per_shader': 2048, 'warp_size': 32,
           'regs_per_sm': 65536, 'max_cta_per_core': 32, 'shmem_size': 102400, 'source': 'builtin constants (pre-26-1 behaviour)'}

def max_cta(M, threads, nregs, smem, n_cta):
    # shader_core_config::max_cta() of the simulator, line for line
    padded = threads if threads % M['warp_size'] == 0 else (threads // M['warp_size'] + 1) * M['warp_size']
    r = M['n_thread_per_shader'] // padded
    if smem > 0: r = min(r, M['shmem_size'] // smem)
    if nregs > 0: r = min(r, M['regs_per_sm'] // (padded * ((nregs + 3) & ~3)))
    r = min(r, M['max_cta_per_core'])
    if n_cta < r * M['n_sm']: r = math.ceil(n_cta / M['n_sm'])
    return max(r, 1)

MEM = re.compile(r'\b(LDG|STG|LDGSTS|LD|ST|ATOM|ATOMG|RED)\b')
def addrs(t, k):
    if int(t[k]) == 0: return []
    fmt = int(t[k+1]); base = int(t[k+2], 16); act = bin(int(t[1], 16)).count('1')
    if fmt == 1: s = int(t[k+3]); return [base + i*s for i in range(act)]
    if fmt == 2:
        out = [base]; a = base
        for d in t[k+3:k+3+act-1]: a += int(d); out.append(a)
        return out
    return [int(x, 16) for x in t[k+2:k+2+act]]
def hdr_int3(v): return tuple(int(x) for x in v.strip('() ').split(','))

klist, ki, outp, cfgp, extra, force = parse_args(sys.argv[1:])
M = model_from_config(read_config(cfgp, extra), force) if cfgp else BUILTIN
N_CHIP, SMS_PER_CHIP, SHIFT = M['n_chip'], M['sms_per_chip'], M['shift']
files = [l.strip() for l in open(klist) if 'kernel-' in l]
f = os.path.join(os.path.dirname(klist), files[ki - 1]); h = {}; cand = {}
shm_base = loc_base = 0; LOCAL_MEM_SIZE_MAX = 1 << 14; skipped_shared = 0
with lzma.open(f, 'rt') as fh:
    for line in fh:
        if line.startswith('-'):
            k, v = line.lstrip('-').split('=', 1); h[k.strip()] = v.strip(); continue
        if line.startswith('#'):
            if 'grid dim' not in h: continue
            shm_base = int(h.get('shmem base_addr', '0'), 16); loc_base = int(h.get('local mem base_addr', '0'), 16)
            gx, gy, gz = hdr_int3(h['grid dim']); bx, by, bz = hdr_int3(h['block dim'])
            threads = bx * by * bz; nregs = int(h.get('nregs', 32)); shmem = int(h.get('shmem', 0))
            n_cta = gx * gy * gz; chunk = math.ceil(n_cta / N_CHIP)
            cps = max_cta(M, threads, nregs, shmem, n_cta)
            per_wave = SMS_PER_CHIP * cps; continue
        if line.startswith('thread block ='):
            x, y, z = (int(v) for v in line.split('=')[1].split(','))
            lin = x + y * gx + z * gx * gy; chip = min(N_CHIP - 1, (lin // chunk) % N_CHIP); local = lin - chip * chunk
            wave = local // per_wave; insn = 0; continue
        if line.startswith('warp ='): insn = 0; continue
        t = line.split()
        if len(t) < 5 or not re.fullmatch(r'[0-9a-f]{4,}', t[0]): continue
        insn += 1
        if not MEM.search(line): continue
        dn = int(t[2]); oi = 3 + dn; sn = int(t[oi+1]); kk = oi + 2 + sn
        try: A = addrs(t, kk)
        except Exception: continue
        if A and shm_base and loc_base and shm_base <= A[0] < loc_base:
            skipped_shared += 1; continue           # simulator: shared space, never translated
        for p in {a >> SHIFT for a in A}:
            d = cand.setdefault(p, {}); c = d.get(chip)
            key = [wave, insn, local]
            if c is None: d[chip] = key + [1]
            else:
                if key < c[:3]: c[:3] = key
                c[3] += 1
json.dump({'kernel': ki, 'file': files[ki - 1], 'name': h.get('kernel name'), 'grid': h.get('grid dim'), 'block': h.get('block dim'),
           'nregs': h.get('nregs'), 'shmem': h.get('shmem'), 'chunk': chunk, 'per_wave': per_wave, 'cps': cps,
           'model': M, 'config_file': cfgp, 'shmem_base_addr': hex(shm_base), 'local_base_addr': hex(loc_base),
           'skipped_shared_insts': skipped_shared,
           'pages': {str(p): {str(c): v for c, v in d.items()} for p, d in cand.items()}}, open(outp, 'w'))
print(f"k{ki} grid={h.get('grid dim')} cps={cps} chunk={chunk} per_wave={per_wave} pages={len(cand):,} shift={SHIFT} model={M['source']}")
