#!/usr/bin/env python3
# Step 17 (docs/s17_llama70b_prep.ko.md section 4.10): build the Llama-3.3-70B L1seq distributed-scheduling experiments
#   llm_exp/experiments_distributed/llama70b_b4s256_L1seq_mcm      (4 KiB pages)
#   llm_exp/experiments_distributed/llama70b_b4s256_L1seq64k_mcm   (64 KiB pages)
# each with arms ref_mcm (first-touch placement, dumps placement_mcm.map), base_mcm (loads the map) and ptwhop0_mcm
# (base + PTW ring hop latency 0), exactly the 2026-10-16 recipe of deepseek_b4s256_L1seq_mcm, plus the TLB settings
# the user asked for on 2026-11-04:
#   L1 TLB 64K  16 -> 32 entries (and its MSHR entries 16 -> 32, the config's "MSHR entries = TLB entries" rule),
#   L2 TLB 64K  512 -> 1024 entries (64 -> 128 sets x 8-way; power of two, as the set hash requires),
#   MSHR merge caps of every TLB (L1 4K/64K/2M 192, L2 4K/64K/2M 46) -> 1000000 = effectively unlimited
#   (the simulator compares waiters >= cap, so 0 would block every merge; nothing is allocated per cap).
# Nothing is launched. A per-experiment launch_mcm.sh (ref_mcm 16 threads, then base_mcm + ptwhop0_mcm 8 threads each,
# side by side, each arm behind a RAM gate) is written with every md5 it must see; run it with DRY=1 to check.
# Usage: s17_make_mcm_arms.py [--force]   (refuses to overwrite an experiment directory unless --force and no run.log)
import hashlib
import os
import shutil
import sys

ROOT_HOST = "/home/user/accel-sim-docker/llm_exp"
ROOT_CONT = "/accel-sim/llm_exp"
SRC_CFG = f"{ROOT_HOST}/experiments_distributed/deepseek_b4s256_L1seq_mcm/ref_mcm/gpgpusim.config"
V3_DIR = f"{ROOT_HOST}/experiments/v3_decode_S256/ref_batch4"
TREE = f"{ROOT_CONT}/llm_sim_tree_pwcstat_l1fwd"
BIN_MD5 = "b4876c28ad2e290989c6460f86498e4c"
SO_MD5 = "dbaed8e6c2c9b9ca20c23fc163cfe513"
TRACE = f"{ROOT_CONT}/traces/s17_llama70b_b4s256_steps/traces/kernelslist_L1seq.g"
TRACE_MD5 = "39dcc14dd2af1d07793d29b82f630a31"
N_KERNELS = 91
N_COPIES = 78
REF_TH, ARM_TH = 16, 8
RESERVE_GB = 36
GATE_LOCK = f"{ROOT_CONT}/experiments_distributed/.ram_gate_llama70b.lock"
EXPS = {"llama70b_b4s256_L1seq_mcm": 0, "llama70b_b4s256_L1seq64k_mcm": 1}

TLB_EDITS = [  # (exact old line, new line) -- each must occur exactly once in the source config
    ("-gpgpu_l1_tlb_64k_assoc 16", "-gpgpu_l1_tlb_64k_assoc 32"),
    ("-gpgpu_l1_tlb_64k_mshr_entries 16", "-gpgpu_l1_tlb_64k_mshr_entries 32"),
    ("-gpgpu_l2_tlb_64k_sets 64", "-gpgpu_l2_tlb_64k_sets 128"),
    ("-gpgpu_l1_tlb_4k_mshr_max_merge 192", "-gpgpu_l1_tlb_4k_mshr_max_merge 1000000"),
    ("-gpgpu_l1_tlb_64k_mshr_max_merge 192", "-gpgpu_l1_tlb_64k_mshr_max_merge 1000000"),
    ("-gpgpu_l1_tlb_2m_mshr_max_merge 192", "-gpgpu_l1_tlb_2m_mshr_max_merge 1000000"),
    ("-gpgpu_l2_tlb_mshr_max_merge 46", "-gpgpu_l2_tlb_mshr_max_merge 1000000"),
    ("-gpgpu_l2_tlb_64k_mshr_max_merge 46", "-gpgpu_l2_tlb_64k_mshr_max_merge 1000000"),
    ("-gpgpu_l2_tlb_2m_mshr_max_merge 46", "-gpgpu_l2_tlb_2m_mshr_max_merge 1000000"),
    ("# L1 TLB (64KB): 16-entry fully-associative, 10-cycle, per-SM (MCM-GPU reference config).",
     "# L1 TLB (64KB): 32-entry fully-associative, 10-cycle, per-SM (2026-11-04 user setting; MCM-GPU reference config had 16)."),
    ("# L2 TLB (64KB): 512-entry (64 sets x 8-way), 80-cycle, chiplet-private.",
     "# L2 TLB (64KB): 1024-entry (128 sets x 8-way), 80-cycle, chiplet-private (2026-11-04 user setting; was 512 = 64 sets)."),
    ("# is also the L1 TLB capacity in this config (sets 1 x assoc 32/16/8,",
     "# is also the L1 TLB capacity in this config (sets 1 x assoc 32/32/8,"),
    ("# L1 TLB MSHR entries = that TLB's entry count (32/16/8 for 4KB/64KB/2MB), which",
     "# L1 TLB MSHR entries = that TLB's entry count (32/32/8 for 4KB/64KB/2MB), which"),
]
MERGE_NOTE = """
# 2026-11-04 (step 17, Llama-70B distributed runs): every TLB MSHR merge cap (-gpgpu_l{1,2}_tlb_*_mshr_max_merge) is set
# to 1000000 = effectively unlimited, on the user's instruction. The simulator blocks a request when the pending list
# of its VPN reaches the cap (waiters >= cap), so 0 cannot mean unlimited; no storage is sized by the cap.
"""
PAGE_EDIT = [("# 0 = every allocation is 4KB.\n-gpgpu_page_size_policy 0", "# 1 = every allocation is 64KB.\n-gpgpu_page_size_policy 1")]


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def replace_once(text, old, new, where):
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT: '{old[:60]}' occurs {n} times in {where} (need exactly 1)")
    return text.replace(old, new)


def make_ref_config(policy):
    s = open(SRC_CFG).read()
    for old, new in TLB_EDITS:
        s = replace_once(s, old, new, "source ref_mcm config")
    s = replace_once(s, "# MCM-GPU Configurations\n", "# MCM-GPU Configurations\n" + MERGE_NOTE, "source ref_mcm config")
    if policy == 1:
        for old, new in PAGE_EDIT:
            s = replace_once(s, old, new, "source ref_mcm config")
    return s


def arm_run_sh(exp, arm, th, gate):
    gate_block = "" if not gate else f"""# RAM gate: /dev/shm holds another user's 58 GB, leaving this host about 61 GB for simulations, and a 70B L1seq
# simulation grows to 25-30 GB RSS. Start only when MemAvailable minus the headroom still reserved for every running
# accel-sim.out (RESERVE_GB each, less its current RSS) leaves RESERVE_GB for this arm; one arm checks at a time (flock,
# held until the new process is visible), rechecking every 60 s. Results are unaffected, only the start time moves.
RESERVE_GB={RESERVE_GB}
GATE_LOCK={GATE_LOCK}
ram_ok(){{
  local avail short rss p
  avail=$(awk '/MemAvailable/{{print int($2/1024)}}' /proc/meminfo); short=0
  for p in $(pgrep -x accel-sim.out); do
    rss=$(awk '/VmRSS/{{print int($2/1024)}}' /proc/$p/status 2>/dev/null); rss=${{rss:-0}}
    [ "$rss" -lt $((RESERVE_GB*1024)) ] && short=$((short + RESERVE_GB*1024 - rss))
  done
  echo "$(date '+%F %T') RAM gate: MemAvailable ${{avail}} MB, reserved-but-unused ${{short}} MB, sims $(pgrep -cx accel-sim.out)" >> ram_gate.log
  [ $((avail - short)) -ge $((RESERVE_GB*1024)) ]
}}
exec {{GL}}>$GATE_LOCK; flock $GL
until ram_ok; do sleep 60; done
"""
    start = ("""env OMP_NUM_THREADS=$TH ACCELSIM_ALLOW_UNVERIFIED_THREADS=1 nice -n 10 \\
  $TREE/gpu-simulator/bin/release/accel-sim.out -config ./gpgpusim.config -config ./trace.config -trace $TRACE > run.log 2>&1
echo "EXIT=$? SECONDS=$((SECONDS-t0)) load=$(cut -d' ' -f1 /proc/loadavg)" > exit_code.txt
""" if not gate else """env OMP_NUM_THREADS=$TH ACCELSIM_ALLOW_UNVERIFIED_THREADS=1 nice -n 10 \\
  $TREE/gpu-simulator/bin/release/accel-sim.out -config ./gpgpusim.config -config ./trace.config -trace $TRACE > run.log 2>&1 &
SIM=$!
sleep 10; echo "$(date '+%F %T') RAM gate: passed, accel-sim.out pid $SIM started" >> ram_gate.log
exec {GL}>&-
wait $SIM; rc=$?
echo "EXIT=$rc SECONDS=$((SECONDS-t0)) load=$(cut -d' ' -f1 /proc/loadavg)" > exit_code.txt
""")
    pages = "64 KiB" if "64k" in exp else "4 KiB"
    return f"""#!/bin/bash
# {exp}/{arm}: Llama-3.3-70B-Instruct (unsloth mirror, layers 0-2 real weights) b4 s256, L1seq piece ({N_KERNELS} kernels) of the
# s17_llama70b_b4s256_steps capture, {pages} pages; KERNEL_PARTITION + per-chiplet cursors + L2 TLB set hash 1 (the 2026-10-16
# deepseek_b4s256_L1seq_mcm recipe) with the 2026-11-04 TLB settings: L1 TLB 64K 32 entries, L2 TLB 64K 1024 entries, merge caps
# unlimited. Generated by llm_exp/scripts/s17_make_mcm_arms.py; tree llm_sim_tree_pwcstat_l1fwd. Threads {th}, unpinned.
TREE={TREE}
TH=${{TH:-{th}}}
TRACE={TRACE}
cd {ROOT_CONT}/experiments_distributed/{exp}/{arm}
source $TREE/gpu-simulator/setup_environment.sh release >/dev/null 2>&1
{{ echo "arm={arm} tree=$TREE threads=$TH trace=$TRACE max_insn=0"; date
  md5sum $TREE/gpu-simulator/bin/release/accel-sim.out $GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG/libcudart.so gpgpusim.config trace.config config_ampere_islip.icnt $TRACE
  ldd $TREE/gpu-simulator/bin/release/accel-sim.out | grep cudart
}} > provenance.txt
export ACCELSIM_NO_PIN=1
{gate_block}t0=$SECONDS
{start}"""


def launcher(exp, policy, cfg_md5):
    X = f"{ROOT_CONT}/experiments_distributed/{exp}"
    return f"""#!/bin/bash
# launch_mcm.sh -- {exp}: ref_mcm (REF_TH threads) first, then base_mcm and ptwhop0_mcm side by side (ARM_TH threads
# each; every arm's run.sh waits behind the RAM gate). Generated by llm_exp/scripts/s17_make_mcm_arms.py (step 17,
# docs/s17_llama70b_prep.ko.md section 4.10). Run inside mcm-accelsim as uid 1035:
#   docker exec -d -u 1035:1035 mcm-accelsim bash {X}/launch_mcm.sh
# env DRY=1: run every preflight check, launch nothing.
set -u
X={X}
TREE={TREE}
BIN=$TREE/gpu-simulator/bin/release/accel-sim.out
SO=$TREE/gpu-simulator/gpgpu-sim/lib/gcc-11.4.0/cuda-12020/release/libcudart.so
BIN_MD5={BIN_MD5}; SO_MD5={SO_MD5}
TRACE={TRACE}; TRACE_MD5={TRACE_MD5}; N_KERNELS={N_KERNELS}; N_COPIES={N_COPIES}
REF_MD5={cfg_md5['ref_mcm']}; BASE_MD5={cfg_md5['base_mcm']}; PTW_MD5={cfg_md5['ptwhop0_mcm']}
PAGE_POLICY={policy}
REF_TH={REF_TH}; ARM_TH={ARM_TH}; DRY=${{DRY:-0}}
LOG=$X/launch.log
log(){{ echo "$(date '+%F %T') [$(basename $X)] $*" | tee -a $LOG; }}
die(){{ log "ABORT: $*"; echo "ABORT $*" >> $X/status.txt; exit 1; }}
md5(){{ md5sum "$1" | awk '{{print $1}}'; }}
exec {{LK}}>$X/.launch.lock
flock -n $LK || {{ echo "another launch_mcm.sh holds $X/.launch.lock"; exit 1; }}

# ---------- preflight: nothing runs unless every check passes ----------
[ "$(md5 $BIN)" = $BIN_MD5 ] || die "binary md5 $(md5 $BIN) != $BIN_MD5"
[ "$(md5 $SO)" = $SO_MD5 ] || die "libcudart.so md5 $(md5 $SO) != $SO_MD5"
[ "$(md5 $TRACE)" = $TRACE_MD5 ] || die "trace list md5 $(md5 $TRACE) != $TRACE_MD5"
[ "$(grep -c '^MemcpyHtoD,' $TRACE)" -ge $N_COPIES ] && [ "$(grep -vc '^MemcpyHtoD,' $TRACE)" = $N_KERNELS ] \\
  || die "trace list does not hold >= $N_COPIES copies + $N_KERNELS kernels"
for A in ref_mcm base_mcm ptwhop0_mcm; do
  [ -s $X/$A/gpgpusim.config ] && [ -s $X/$A/trace.config ] && [ -s $X/$A/config_ampere_islip.icnt ] && [ -x $X/$A/run.sh ] || die "$A files missing"
  [ -e $X/$A/run.log ] && die "$A already has run.log -- refusing to overwrite"
done
[ "$(md5 $X/ref_mcm/gpgpusim.config)" = $REF_MD5 ] || die "ref_mcm config md5 changed since generation"
[ "$(md5 $X/base_mcm/gpgpusim.config)" = $BASE_MD5 ] || die "base_mcm config md5 changed since generation"
[ "$(md5 $X/ptwhop0_mcm/gpgpusim.config)" = $PTW_MD5 ] || die "ptwhop0_mcm config md5 changed since generation"
[ -s $X/CAP_EXCEPTION.txt ] || die "instruction cap 0 without a user exception record ($X/CAP_EXCEPTION.txt)"
grep -qx -- '-gpgpu_max_insn 0' $X/ref_mcm/gpgpusim.config || die "ref_mcm config is not uncapped"
for KV in "-gpgpu_page_size_policy $PAGE_POLICY" "-gpgpu_lasp_policy 4" "-gpgpu_cta_chiplet_cursor 1" "-gpgpu_l2_tlb_set_hash 1" \\
          "-gpgpu_l1_tlb_64k_assoc 32" "-gpgpu_l1_tlb_64k_mshr_entries 32" "-gpgpu_l2_tlb_64k_sets 128" "-gpgpu_l2_tlb_64k_assoc 8" \\
          "-gpgpu_l1_tlb_4k_mshr_max_merge 1000000" "-gpgpu_l1_tlb_64k_mshr_max_merge 1000000" \\
          "-gpgpu_l2_tlb_mshr_max_merge 1000000" "-gpgpu_l2_tlb_64k_mshr_max_merge 1000000" "-gmmu_placement_dump ./placement_mcm.map"; do
  grep -qx -- "$KV" $X/ref_mcm/gpgpusim.config || die "ref_mcm config lacks '$KV'"; done
[ "$(diff $X/ref_mcm/gpgpusim.config $X/base_mcm/gpgpusim.config | grep -c '^[<>]')" = 2 ] \\
  && diff $X/ref_mcm/gpgpusim.config $X/base_mcm/gpgpusim.config | grep -qx -- '> -gmmu_placement_load ../ref_mcm/placement_mcm.map' \\
  || die "base_mcm is not ref_mcm with only the placement line changed"
[ "$(diff $X/base_mcm/gpgpusim.config $X/ptwhop0_mcm/gpgpusim.config | grep -c '^[<>]')" = 1 ] \\
  && diff $X/base_mcm/gpgpusim.config $X/ptwhop0_mcm/gpgpusim.config | grep -qx -- '> -icnt_ptw_ring_hop_latency 0' \\
  || die "ptwhop0_mcm is not base_mcm plus the PTW hop line"
log "preflight OK: kernels=$N_KERNELS page_policy=$PAGE_POLICY bin=$BIN_MD5 so=$SO_MD5 trace=$TRACE_MD5 ref=$REF_MD5 ref_th=$REF_TH arm_th=$ARM_TH dry=$DRY"
[ "$DRY" = 1 ] && {{ log "DRY run finished -- nothing launched"; exit 0; }}

# ---------- ref_mcm ----------
echo "RUNNING ref_mcm $(date '+%F %T')" >> $X/status.txt
log "ref_mcm start (threads $REF_TH)"
TH=$REF_TH bash $X/ref_mcm/run.sh
log "ref_mcm end: $(cat $X/ref_mcm/exit_code.txt 2>/dev/null)"
grep -q '^EXIT=0 ' $X/ref_mcm/exit_code.txt 2>/dev/null || die "ref_mcm not EXIT=0"
grep -q '^\\[PLACEMENT\\] dump: ./placement_mcm.map' $X/ref_mcm/run.log && [ -s $X/ref_mcm/placement_mcm.map ] || die "ref_mcm placement map missing"

# ---------- measurement arms (side by side; each run.sh waits behind the RAM gate) ----------
echo "RUNNING base_mcm ptwhop0_mcm $(date '+%F %T')" >> $X/status.txt
log "base_mcm + ptwhop0_mcm start side by side (threads $ARM_TH each, RAM gate)"
TH=$ARM_TH bash $X/base_mcm/run.sh & P1=$!
TH=$ARM_TH bash $X/ptwhop0_mcm/run.sh & P2=$!
wait $P1; wait $P2
for A in base_mcm ptwhop0_mcm; do
  L=$X/$A/run.log; hop=-1; bad=""
  [ $A = ptwhop0_mcm ] && hop=0
  grep -q '^EXIT=0 ' $X/$A/exit_code.txt 2>/dev/null || bad="$bad exit"
  grep -qE '^-gpgpu_lasp_policy[[:space:]]+4 ' $L || bad="$bad lasp"
  grep -qE '^-gpgpu_cta_chiplet_cursor[[:space:]]+1 ' $L || bad="$bad cursor"
  grep -qE '^-gpgpu_l2_tlb_set_hash[[:space:]]+1 ' $L || bad="$bad set_hash"
  grep -qE "^-gpgpu_page_size_policy[[:space:]]+$PAGE_POLICY " $L || bad="$bad page_policy"
  grep -qE '^-gpgpu_l1_tlb_64k_assoc[[:space:]]+32 ' $L || bad="$bad l1_64k_assoc"
  grep -qE '^-gpgpu_l2_tlb_64k_sets[[:space:]]+128 ' $L || bad="$bad l2_64k_sets"
  grep -qE "^-icnt_ptw_ring_hop_latency[[:space:]]+$hop " $L || bad="$bad ptw_hop"
  grep -q '^\\[PLACEMENT\\] load: ../ref_mcm/placement_mcm.map' $L || bad="$bad placement_load"
  grep -q 'per-chiplet cursors on' $L || bad="$bad pcc_line"
  if [ -z "$bad" ]; then log "$A end: $(cat $X/$A/exit_code.txt) option check OK (lasp=4 cursor=1 set_hash=1 page=$PAGE_POLICY ptw_hop=$hop placement=load PCC on)"
  else log "$A end: $(cat $X/$A/exit_code.txt 2>/dev/null) option check FAILED:$bad"; fi
done
echo "DONE $(date '+%F %T')" >> $X/status.txt
log "all arms finished"
"""


def main():
    force = "--force" in sys.argv
    src_v3 = f"{V3_DIR}/gpgpusim.config"
    for exp, policy in EXPS.items():
        X = f"{ROOT_HOST}/experiments_distributed/{exp}"
        if os.path.exists(X):
            if any(os.path.exists(f"{X}/{a}/run.log") for a in ("ref_mcm", "base_mcm", "ptwhop0_mcm")):
                sys.exit(f"ABORT: {X} has a run.log -- not touching an experiment that ran")
            if not force:
                sys.exit(f"ABORT: {X} exists (use --force to regenerate)")
            shutil.rmtree(X)
        ref = make_ref_config(policy)
        base = replace_once(ref, "-gmmu_placement_dump ./placement_mcm.map", "-gmmu_placement_load ../ref_mcm/placement_mcm.map", "ref config")
        ptw = replace_once(base, "-gpgpu_cta_chiplet_cursor 1\n", "-gpgpu_cta_chiplet_cursor 1\n-icnt_ptw_ring_hop_latency 0\n", "base config")
        cfgs = {"ref_mcm": ref, "base_mcm": base, "ptwhop0_mcm": ptw}
        cfg_md5 = {}
        for arm, text in cfgs.items():
            d = f"{X}/{arm}"
            os.makedirs(d)
            with open(f"{d}/gpgpusim.config", "w") as f:
                f.write(text)
            shutil.copy(f"{V3_DIR}/trace.config", d)
            shutil.copy(f"{V3_DIR}/config_ampere_islip.icnt", d)
            th = REF_TH if arm == "ref_mcm" else ARM_TH
            with open(f"{d}/run.sh", "w") as f:
                f.write(arm_run_sh(exp, arm, th, gate=(arm != "ref_mcm")))
            os.chmod(f"{d}/run.sh", 0o755)
            cfg_md5[arm] = md5(f"{d}/gpgpusim.config")
        with open(f"{X}/launch_mcm.sh", "w") as f:
            f.write(launcher(exp, policy, cfg_md5))
        os.chmod(f"{X}/launch_mcm.sh", 0o755)
        pages = "64 KiB" if policy == 1 else "4 KiB"
        with open(f"{X}/TRACE_INPUT.txt", "w") as f:
            f.write(f"Llama-3.3-70B-Instruct (unsloth/Llama-3.3-70B-Instruct, layers 0-2 real weights, step 17) b4 s256 steps capture "
                    f"s17_llama70b_b4s256_steps (llm_exp/README.ko.md), piece kernelslist_L1seq.g = {N_KERNELS} kernels + {N_COPIES} copies, "
                    f"md5 {TRACE_MD5}. {pages} pages (-gpgpu_page_size_policy {policy}). Distributed arms created 2026-11-04 by "
                    f"scripts/s17_make_mcm_arms.py: deepseek_b4s256_L1seq_mcm ref_mcm config (lasp 4 + cursor 1 + placement_mcm.map + "
                    f"l2_tlb_set_hash 1, max_insn 0) with L1 TLB 64K 32 entries (MSHR 32), L2 TLB 64K 1024 entries (128 sets x 8-way), "
                    f"all TLB MSHR merge caps 1000000. ref_mcm {REF_TH} threads, base_mcm/ptwhop0_mcm {ARM_TH} threads (RAM gate), "
                    f"tree llm_sim_tree_pwcstat_l1fwd ({BIN_MD5}).\n")
        with open(f"{X}/CAP_EXCEPTION.txt", "w") as f:
            f.write(f"Instruction cap exception (-gpgpu_max_insn 0) for experiments_distributed/{exp} (all three arms), recorded 2026-11-04 by Claude on the\n"
                    "user's instruction of 2026-11-04 (\"distributed CTA scheduling으로 64KB, 4KB 돌릴 준비해줘\") under the standing 2026-09-19 b4s256 rule:\n"
                    f"the L1seq piece's stats warp count x 32 = 4.83e10 thread instructions exceeds the 1e10 cap, which would cut the prefill layer at 31%;\n"
                    "the 8B distributed arms (deepseek_b4s256_L1seq_mcm) ran uncapped for the same reason.\n"
                    "If this exception is not intended, delete this file before launching (launch_mcm.sh then refuses to start).\n")
        # report: ref config vs the V3 reference and vs the 8B distributed ref
        import difflib
        def changed(a, b):
            return [l for l in difflib.unified_diff(open(a).read().splitlines(), b.splitlines(), lineterm="", n=0)
                    if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---"))]
        print(f"== {exp}: ref_mcm config vs 8B deepseek ref_mcm ({len(changed(SRC_CFG, ref))} changed lines)")
        for l in changed(SRC_CFG, ref):
            print("  ", l)
        print(f"   vs V3 ref_batch4: {len(changed(src_v3, ref))} changed lines; md5 ref={cfg_md5['ref_mcm']} base={cfg_md5['base_mcm']} ptwhop0={cfg_md5['ptwhop0_mcm']}")
    print("MCM ARMS BUILT")


if __name__ == "__main__":
    main()
