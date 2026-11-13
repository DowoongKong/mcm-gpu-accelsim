# V3: explain the instruction-count gap between tracer stats and post-processed .traceg.xz.
# post-traces-processing.cpp:378-386 drops every other LDGSTS line per warp ("one actual LDGSTS instruction
# includes 2 LDGSTS instructions in the trace because it has two memory references"). This audit checks, per
# kernel: raw lines == stats, raw - traceg == raw LDGSTS / 2, and that raw LDGSTS lines come in adjacent
# same-PC pairs within each warp. Usage: v3_ldgsts_audit.py <traces dir>. See docs/v3_postprocess_sim.ko.md.
import glob, lzma, os, re, sys
d = sys.argv[1]
stats = {}
for ln in open(glob.glob(os.path.join(d, "stats_ctx_*"))[0]).read().splitlines()[1:]:
    parts = ln.split(","); stats[int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1))] = int(parts[-1])
all_ok = True
tot = [0, 0, 0, 0]
for raw_path in sorted(glob.glob(os.path.join(d, "kernel-*.trace.xz")), key=lambda p: int(re.search(r"kernel-(\d+)-", p).group(1))):
    kid = int(re.search(r"kernel-(\d+)-", raw_path).group(1))
    raw_n = raw_ld = 0
    pending = {}   # (tb, warp) -> PC of an unmatched first LDGSTS line
    last_pc = {}   # (tb, warp) -> PC of previous instruction line in that warp
    bad_pairs = 0
    with lzma.open(raw_path, "rt", errors="replace") as f:
        for line in f:
            if not line or line[0] in "-#\n":
                continue
            t = line.split()
            if len(t) < 8:
                continue
            key = (t[0], t[1], t[2], t[3]); pc = t[4]; ndest = int(t[6]); op = t[7 + ndest]
            raw_n += 1
            if "LDGSTS" in op:
                raw_ld += 1
                if key in pending:
                    if pending.pop(key) != pc or last_pc.get(key) != pc:
                        bad_pairs += 1
                else:
                    pending[key] = pc
            elif key in pending:
                bad_pairs += 1  # an unpaired LDGSTS followed by a different instruction
                pending.pop(key)
            last_pc[key] = pc
    bad_pairs += len(pending)
    g_n = g_ld = 0
    with lzma.open(raw_path[:-3] + "g.xz", "rt", errors="replace") as f:
        for line in f:
            if re.match(r"^[0-9a-f]{4,} [0-9a-f]{8} \d+ ", line):
                t = line.split(); g_n += 1
                if "LDGSTS" in t[3 + int(t[2])]:
                    g_ld += 1
    ok = raw_n == stats[kid] and raw_n - g_n == raw_ld // 2 and raw_ld % 2 == 0 and g_ld == raw_ld // 2 and bad_pairs == 0
    all_ok &= ok
    tot = [tot[0] + raw_n, tot[1] + g_n, tot[2] + raw_ld, tot[3] + g_ld]
    print(f"kernel {kid}: stats={stats[kid]} raw={raw_n} traceg={g_n} raw_LDGSTS={raw_ld} traceg_LDGSTS={g_ld} unpaired={bad_pairs} {'OK' if ok else 'MISMATCH'}")
print(f"total raw={tot[0]} traceg={tot[1]} raw_LDGSTS={tot[2]} traceg_LDGSTS={tot[3]} raw-traceg={tot[0]-tot[1]} raw_LDGSTS/2={tot[2]//2}")
print("LDGSTS AUDIT", "PASS" if all_ok else "FAIL")
