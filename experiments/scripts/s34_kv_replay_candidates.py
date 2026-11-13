#!/usr/bin/env python3
# Step 34 (docs/s34_prefill_placement_without_sim.ko.md): per-step KV first-write candidates from the captured vLLM schedule,
# no simulation. For every step before the decode window, rows are laid out in the recorded order (running requests'
# decode rows, then the previous request's remainder, then the new chunk); reshape_and_cache has one CTA per row, and
# KERNEL_PARTITION gives chiplet = (row // ceil(R/4)) % 4 (s22 model). A page's candidate per chiplet is its lowest row in
# that step with the s22 key [wave, insn 0, local cta, count], wave = local // per_wave (per_wave = SMs per chiplet x CTAs per SM
# of reshape_and_cache_flash = 64 x 2 = 128 for block 512 / 54 regs; override with --per-wave N), local = row - chiplet*chunk,
# chunk = ceil(rows/4). The map maker orders steps by window position and breaks ties by lowest chiplet, exactly the simulator's
# first-write-wins rule. V-b (11-12 03:05): without the wave term the 6 pages straddling the chiplet boundaries were wrong (a later
# wave of chiplet c loses to wave 0 of chiplet c+1); with it the single-kernel reference dump matches. Pages = 64 KB in the capture's VA space.
# Usage: s34_kv_replay_candidates.py <decode capture dir> <cand dir> <first k> [--per-wave N]   -> k<first>.json ... (one per step)
import json, os, sys, collections, math
PER_WAVE = 128
if '--per-wave' in sys.argv: i = sys.argv.index('--per-wave'); PER_WAVE = int(sys.argv[i+1]); del sys.argv[i:i+2]
d, cdir, k0 = sys.argv[1], sys.argv[2], int(sys.argv[3]); PS = 16; ESZ = 2; NCHIP = 4
ci = json.load(open(d + "/capture_info.json")); vm = json.load(open(d + "/va_map.json"))
kv = vm["kv_cache_layer0"]; ptr, shape, stride = kv["ptr"], kv["shape"], kv["stride"]; bs = shape[2]; vbase = ptr + stride[0] * ESZ
bt = vm["block_table"]; B = ci["batch"]; S = ci["S"]; cap_step = ci["capture_step"]
os.makedirs(cdir, exist_ok=True)
def pages_of(r, t):
    blk = bt[r][t // bs]; off = (blk * stride[1] + (t % bs) * stride[2]) * ESZ
    return ((ptr + off) >> PS, (vbase + off) >> PS)
k = k0; total = set(); nsteps = 0; per_req_chips = collections.defaultdict(set); seen = {}
sched = ci.get("sched")
if not sched:   # older v2 captures (S1024 x b4, prefill in ONE step: max_num_batched_tokens == batch x S, H1v3-1 PASS) record no schedule:
    # synthesize it -- step 1 = every request's whole prompt in request order, then one decode token per request per step.
    assert ci.get("prefill_step") == 1 and ci.get("prefill_tokens") == B * S, "no sched and not a single-step prefill capture"
    sched = [{"step": 1, "reqs": [[f"{r}-synth", S, 0] for r in range(B)]}]
    for stp in range(2, cap_step + 1): sched.append({"step": stp, "reqs": [[f"{r}-synth", 1, S + stp - 2] for r in range(B)]})
    print(f"note: no 'sched' in capture_info -> synthesized single-step prefill schedule (batch {B} x S {S} in step 1, then 1 token/request/step)")
for st in sched:
    if st["step"] >= cap_step: break
    rows = []
    for rid, n, pos in st["reqs"]:
        r = int(rid.split("-")[0]); rows += [(r, pos + i) for i in range(n)]
    R = len(rows); q = math.ceil(R / NCHIP); cand = {}
    for i, (r, t) in enumerate(rows):
        if t // bs >= len(bt[r]): continue
        c = min(NCHIP - 1, (i // q) % NCHIP); local = i - c * q; key = [local // PER_WAVE, 0, local]
        for p in pages_of(r, t):
            e = cand.setdefault(p, {})
            if c not in e or key < e[c][:3]: e[c] = key + [e[c][3] if c in e else 0]
            e[c][3] += 1
            total.add(p)
    for p, e in cand.items():                       # owner of this step's pages under rule C (min key, then lowest chiplet)
        if p not in seen:
            c = sorted((tuple(v[:3]), int(cc)) for cc, v in e.items())[0][1]; seen[p] = c
            for r in {rr for rr, tt in rows if tt // bs < len(bt[rr]) and p in pages_of(rr, tt)}: per_req_chips[r].add(c)
    json.dump({"kernel": k, "file": None, "name": f"kv_step{st['step']}", "grid": f"({R},1,1)", "block": "(512,1,1)", "chunk": q, "per_wave": PER_WAVE, "model": {"shift": PS, "n_chip": NCHIP, "source": "s34 kv replay (wave-aware)"},
               "pages": {str(p): {str(c): v for c, v in e.items()} for p, e in cand.items()}}, open(os.path.join(cdir, f"k{k}.json"), "w"))
    k += 1; nsteps += 1
chips = collections.Counter(len(v) for v in per_req_chips.values())
print(f"kv replay {d}: steps {nsteps} (before capture step {cap_step}) -> k{k0}..k{k-1}, KV pages {len(total):,}, "
      f"owner share {dict(collections.Counter(seen.values()))}, chiplets per request {dict(sorted(chips.items()))}")
