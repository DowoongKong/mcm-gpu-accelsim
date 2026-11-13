#!/usr/bin/env python3
# Compact per-kernel access anatomy from the s25_kernel_profile outputs (k<id>_summary.json + k<id>_pc.tsv):
# for every kernel and every global operand: bytes read/written per CTA and in total, the dominant memory instruction
# (opcode, width), its lane pattern (contiguous run x width + jump), the loop stride (base advance between successive
# executions of the same PC by the same warp), sectors per request, and the dynamic count per CTA. Usage: <outdir>
import collections, glob, json, os, sys
d = sys.argv[1]
for sf in sorted(glob.glob(os.path.join(d, 'k*_summary.json')), key=lambda p: int(os.path.basename(p)[1:].split('_')[0])):
    J = json.load(open(sf)); kid = J['kernel_id']
    pcs = [l.rstrip('\n').split('\t') for l in open(os.path.join(d, f'k{kid}_pc.tsv'))][1:]
    print(f"\n### k{kid} {J['kernel'][:70]}\n grid={J['grid']} block={J['block']} ctas={J['ctas']} warp_insts={J['warp_insts_total']:,} mem_insts={J['mem_insts_total']:,} "
          f"G bytes={J['pc_summary']['G']['bytes']/2**20:.1f} MB S bytes={J['pc_summary']['S']['bytes']/2**20:.1f} MB shmem={J['shmem']} nregs={J['nregs']}")
    # per operand: aggregate the PCs touching it
    per = collections.defaultdict(list)
    for r in pcs:
        if r[2] != 'G': continue
        for item in r[13].split(';'):
            if not item: continue
            name, b = item.rsplit(':', 1); per[name].append((int(b), r))
    for name, info in J['operands'].items():
        fp = J['footprint'][name]
        rows = sorted(per.get(name, []), key=lambda x: -x[0])
        tot_rd = fp['total_read_bytes']; tot_wr = fp['total_write_bytes']
        print(f"  {name:3s} va={info['va']} size={info['bytes']/2**20:8.2f} MB  ctas={fp['ctas']:4d}  rd/cta={fp['read_bytes_per_cta']['mean']/1024:9.1f} KB  wr/cta={fp['write_bytes_per_cta']['mean']/1024:7.1f} KB  "
              f"total rd={tot_rd/2**20:8.1f} MB wr={tot_wr/2**20:6.2f} MB  uniq lines/cta={fp['unique_lines_per_cta']['mean']}")
        shown = 0
        for b, r in rows:
            if shown >= 3: break
            pc, op, _, w, dyn, lanes, sect, lines, _, wr, mpop, sig, loop = r[:13]
            print(f"       pc {pc} {op[:34]:34s} w={w:>2s}B dyn={int(dyn):8d} ({int(dyn)/J['ctas']:7.1f}/cta) lanes={lanes:>4s} sect/req={sect} lane_sig={sig.split(';')[0]} loop_stride={loop.split(';')[0] or '-'}")
            shown += 1
