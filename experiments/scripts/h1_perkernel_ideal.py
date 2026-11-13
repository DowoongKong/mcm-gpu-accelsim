import re, sys
def blocks(path):
    out=[]; cur=None
    for l in open(path, errors='ignore'):
        if l.startswith('kernel_launch_uid ='): cur={}; out.append(cur); continue
        if cur is None: continue
        if l.startswith('gpu_tot_sim_cycle ='): cur['cyc']=int(l.split('=')[1])
        elif l.startswith('gpgpu_n_tot_w_icount ='): cur['w']=int(l.split('=')[1])
        elif l.startswith('kernel_name ='): cur['next_name']=l.split('=',1)[1].strip()
    return [b for b in out if 'cyc' in b]
exp=sys.argv[1]; arms=['ref_mcm','base_mcm','ptwhop0_mcm','ideal_mcm']
D={a:blocks(f'{exp}/{a}/run.log') for a in arms}
n=len(D["base_mcm"]); names=["rms_norm (first kernel, name not printed)"]+[x.get("next_name","")[:34] for x in D["base_mcm"]]   # block k-1 prints kernel k+1 name -> kernel k label = names[k-1]
def d(bl,k): return bl[k-1]['cyc']-(bl[k-2]['cyc'] if k>1 else 0)
print(f"{'k':>2} {'kernel':34s} {'warp-insn':>10} {'base':>9} {'ref/base':>8} {'hop0/base':>9} {'ideal/base':>10} {'hop0 gain / ideal gain':>22}")
tot={a:0 for a in arms}
for k in range(1,n+1):
    c={a:d(D[a],k) for a in arms}; w=D['base_mcm'][k-1]['w']-(D['base_mcm'][k-2]['w'] if k>1 else 0)
    for a in arms: tot[a]+=c[a]
    ig=c['base_mcm']-c['ideal_mcm']; hg=c['base_mcm']-c['ptwhop0_mcm']
    print(f"{k:>2} {names[k-1][:34]:34s} {w:>10} {c['base_mcm']:>9} {c['ref_mcm']/c['base_mcm']:>8.3f} {c['ptwhop0_mcm']/c['base_mcm']:>9.3f} {c['ideal_mcm']/c['base_mcm']:>10.3f} {hg/ig if ig else float('nan'):>22.2f}")
print(f"{'':>2} {'TOTAL':34s} {'':>10} {tot['base_mcm']:>9} {tot['ref_mcm']/tot['base_mcm']:>8.3f} {tot['ptwhop0_mcm']/tot['base_mcm']:>9.3f} {tot['ideal_mcm']/tot['base_mcm']:>10.3f} {(tot['base_mcm']-tot['ptwhop0_mcm'])/(tot['base_mcm']-tot['ideal_mcm']):>22.2f}")
