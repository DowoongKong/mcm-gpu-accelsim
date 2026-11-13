#!/bin/bash
# Step 27 section 4.7 (docs/s27_prefill_capture_and_decode_plan.ko.md): Mixtral S1024 x b4 decode layer 0 with FORCED layer-0
# routing, scenarios min2 (all tokens -> experts 0,1) and max8 (token t -> 2t, 2t+1); the natural run stays as it is.
#   1. wait until the Qwen3 capture sequence has released the GPU, then until llmtrace runs no capture/smoke and the GPU is idle
#   2. per scenario: h1_capture_v2_chain_c.sh (decode window only) -> checks: FORCE ROUTE PASS, decode kernels == natural,
#      routing recovered from the trace == expected experts, va_map weights == natural capture
#   3. arms ref/base/ptwhop0 per scenario = the natural Mixtral arms with paths changed (configs byte-identical)
#   4. simulations after the natural Mixtral base/ptwhop0 have started, at most 2 at a time, refs first
# Run on the host, detached. Log: llm_hypothesis1/Mixtral-8x7B-v0.1/force_seq.log
set -u
H=/home/user/accel-sim-docker/llm_exp
M=$H/llm_hypothesis1/Mixtral-8x7B-v0.1
NAT=$M/s1024_b0004_prefill_l0_decode_layer0_v2
QLOG=$H/llm_hypothesis1/qwen3_capture_seq.log
LOG=$M/force_seq.log
log(){ echo "$(date '+%F %T') [mixtral_force] $*" | tee -a $LOG; }
exec {LK}>$M/.force_seq.lock
flock -n $LK || { echo "force sequence already running"; exit 1; }
nsims(){ ps -eo stat,args | grep "accel-sim.out -config" | grep -v grep | grep -vc '^Z'; }
gpu_idle(){ ! docker exec llmtrace pgrep -f 's17_smoke_70b|h1_capture_v2' > /dev/null 2>&1 && \
            [ "$(docker exec llmtrace nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc 0-9)" -lt 1000 ]; }

# ---------- 1. GPU ----------
log "waiting for the Qwen3 capture sequence to release the GPU"
until grep -q 'GPU part done' $QLOG 2>/dev/null; do sleep 120; done
until gpu_idle && sleep 60 && gpu_idle; do sleep 60; done
log "GPU idle"

# ---------- 2. captures ----------
for R in min2 max8; do
  CFG=s1024_b0004_decode_layer0_force_${R}_v2; T=$M/$CFG
  until gpu_idle && sleep 30 && gpu_idle; do sleep 60; done
  log "$R: capture chain start"
  FORCE_ROUTE=$R NAME=decode_layer0_force_$R INIT_FIRST=0 DISK_MIN=8 \
    bash $H/scripts/h1_capture_v2_chain_c.sh mixtral Mixtral-8x7B-v0.1 1024 32 4096 4294967296 4 >> $M/force_chain_$R.out 2>&1
  log "$R: chain returned: $(grep "$CFG" $M/chain_v2_force.log | tail -n 1 | cut -c1-200)"
  python3 - "$NAT" "$T" "$R" >> $LOG 2>&1 <<'EOF'
import json, re, subprocess, sys
nat, t, r = sys.argv[1:4]
ok = True
def check(label, cond, detail=""):
    global ok; ok &= bool(cond); print(f"  CHECK {'PASS' if cond else 'FAIL'} {r} {label} | {detail}")
run = open(f"{t}/run.log", errors="ignore").read()
m = re.search(r"^FORCE ROUTE (\w+) .*$", run, re.M)
check("FORCE ROUTE line", m and m.group(1) == "PASS", m.group(0)[:200] if m else "missing")
cn, cf = json.load(open(f"{nat}/capture_info.json")), json.load(open(f"{t}/capture_info.json"))
dn = [(k["name"], k["grid"], k["block"]) for k in cn["window_kernels"] if k["window"] == "decode_layer0"]
df = [(k["name"], k["grid"], k["block"]) for k in cf["window_kernels"] if k["window"] == "decode_layer0"]
check("decode kernels (name, grid, block) == natural", dn == df, f"{len(df)} vs {len(dn)} kernels, window {cf['windows'].get('decode_layer0')}")
wn, wf = json.load(open(f"{nat}/va_map.json"))["weights"], json.load(open(f"{t}/va_map.json"))["weights"]
check("layer-0 weight VAs == natural capture", {k: v["ptr"] for k, v in wn.items()} == {k: v["ptr"] for k, v in wf.items()})
exp = {"min2": {0, 1}, "max8": set(range(8))}[r]
moe = [k for k in cf["window_kernels"] if k["window"] == "decode_layer0" and "fused_moe" in k["name"]]
for k, (wname, per) in zip(sorted(moe, key=lambda k: k["id"]), (("block_sparse_moe.experts.w13_weight", 234881024), ("block_sparse_moe.experts.w2_weight", 117440512))):
    tr = subprocess.run(["ls", f"{t}/traces"], capture_output=True, text=True).stdout.split()
    f = [x for x in tr if x.startswith(f"kernel-{k['id']}-") and x.endswith(".traceg.xz")][0]
    out = subprocess.run(["python3", "-I", "/home/user/accel-sim-docker/llm_exp/scripts/s27_moe_routing_from_trace.py", f"{t}/traces/{f}",
                          f"{wf[wname]['ptr']:x}", str(per), "8"], capture_output=True, text=True).stdout
    cnt = {int(e): int(c) for e, c in re.findall(r"e(\d+):(\d+) CTA", out)}
    act = {e for e, c in cnt.items() if c >= 0.1 * max(cnt.values())} if cnt else set()
    print("   ", out.strip().replace("\n", "\n    "))
    check(f"k{k['id']} {wname.split('.')[-1]} active experts == {sorted(exp)}", act == exp, f"recovered {sorted(act)}")
print(f"  FORCE CAPTURE {'PASS' if ok else 'FAIL'} {r}")
EOF
  grep -q "FORCE CAPTURE PASS $R" $LOG || { log "$R: capture checks FAILED -- no arms for this scenario"; continue; }
  grep -q '^S6 TRACES PASS' $T/h1_check_traces_decode_layer0_*.log 2>/dev/null || { log "$R: decode S6 not PASS -- no arms"; continue; }

  # ---------- 3. arms ----------
  cp $T/traces/kernelslist.g $T/traces/kernelslist_decode.g
  mkdir -p $T/sim
  for a in ref base ptwhop0; do
    mkdir -p $T/sim/${a}_mcm
    cp $NAT/sim/${a}_mcm/gpgpusim.config $NAT/sim/${a}_mcm/trace.config $NAT/sim/${a}_mcm/config_ampere_islip.icnt $T/sim/${a}_mcm/
    sed -e "s#s1024_b0004_prefill_l0_decode_layer0_v2#$CFG#g" -e "s#section 4.5): Mixtral-8x7B decode layer 0#section 4.7): Mixtral-8x7B decode layer 0 (forced routing $R)#" \
      $NAT/sim/${a}_mcm/run.sh > $T/sim/${a}_mcm/run.sh; chmod +x $T/sim/${a}_mcm/run.sh
  done
  sed -e "s#s1024_b0004_prefill_l0_decode_layer0_v2#$CFG#g" -e 's#h1_check_traces_decode_layer0_313_329.log#h1_check_traces_decode_layer0_*.log#' \
    -e "s#section 4.5)#section 4.7, forced routing $R)#" $NAT/sim/launch_h1.sh > $T/sim/launch_h1.sh; chmod +x $T/sim/launch_h1.sh
  log "$R: arms ready in $T/sim"
done

# ---------- 4. simulations ----------
log "waiting until the natural Mixtral base/ptwhop0 have started"
until [ -e $NAT/sim/base_mcm/run.log ] && [ -e $NAT/sim/ptwhop0_mcm/run.log ]; do sleep 300; done
JOBS="min2:ref_mcm max8:ref_mcm min2:base_mcm min2:ptwhop0_mcm max8:base_mcm max8:ptwhop0_mcm"
pending=$JOBS
while [ -n "$pending" ]; do
  next=""; launched=0
  for j in $pending; do
    R=${j%%:*}; A=${j#*:}; T=$M/s1024_b0004_decode_layer0_force_${R}_v2
    [ -x $T/sim/launch_h1.sh ] || { log "$j skipped (no arms)"; continue; }
    if [ $A != ref_mcm ]; then
      [ -e $T/sim/ref_mcm/exit_code.txt ] || { next="$next $j"; continue; }
      grep -q '^EXIT=0' $T/sim/ref_mcm/exit_code.txt || { log "$j skipped (ref failed)"; continue; }
    fi
    if [ $launched = 0 ] && [ "$(nsims)" -le 1 ]; then
      bash $T/sim/launch_h1.sh $A >> $LOG 2>&1 && log "$j launched" || log "$j launch refused"
      launched=1; sleep 60
    else next="$next $j"; fi
  done
  pending=$(echo $next)
  [ -n "$pending" ] && sleep 180
done
log "all forced-routing arms launched"
