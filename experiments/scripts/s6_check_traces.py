# Step 6 preparation checks on the post-processed full-decode traces (docs/s6_decode_layer_scaling.ko.md, P-1 and P-2).
# P-1 (whole list): post-processing exit 0 and no "Undefined command"; kernelslist.g == kernelslist_ctx_* with
#     .trace.xz renamed to .traceg.xz; every listed kernel has a non-empty .traceg.xz.
# P-2 (the given kernel ids only): every opcode exists in the simulator's Ampere opcode table (parsing as in
#     scripts/v3_check_traces.py), and the LDGSTS merge audit of scripts/v3_ldgsts_audit.py holds per kernel:
#     raw lines == stats, raw - traceg == raw LDGSTS / 2, LDGSTS lines come in adjacent same-PC pairs per warp.
# Usage: s6_check_traces.py <traces dir> <ampere_opcode.h> <postprocess log> <postprocess exit file> <first id> <last id>
import collections
import glob
import lzma
import os
import re
import sys

traces_dir, table_path, pp_log, pp_exit, first_id, last_id = sys.argv[1:7]
ids = list(range(int(first_id), int(last_id) + 1))
known = set(re.findall(r'\{"([A-Z0-9_]+)",\s*OpcodeChar', open(table_path).read()))
inst_re = re.compile(r"^[0-9a-f]{4,} [0-9a-f]{8} \d+ ")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


raw = open(glob.glob(os.path.join(traces_dir, "kernelslist_ctx_*"))[0]).read().splitlines()
expected = [ln[:-3] + "g.xz" if ln.endswith(".trace.xz") else ln for ln in raw]
if os.path.exists(pp_exit):
    exit_line = open(pp_exit).read().strip()
    check("P-1 post-processing exit 0, no 'Undefined command'",
          exit_line.startswith("EXIT=0") and "Undefined command" not in open(pp_log).read(), exit_line)
    converted = open(os.path.join(traces_dir, "kernelslist.g")).read().splitlines()
    check("P-1 kernelslist.g == kernelslist_ctx with .traceg.xz names", converted == expected, f"{len(converted)} lines")
    listed = [ln for ln in converted if not ln.startswith("MemcpyHtoD,")]
    check("P-1 every listed kernel has a non-empty .traceg.xz",
          all(os.path.isfile(os.path.join(traces_dir, k)) and os.path.getsize(os.path.join(traces_dir, k)) > 0
              for k in listed), f"{len(listed)} kernels")
else:
    # Post-processing is still running: only P-2 is checked, on kernels whose .traceg.xz is already finished
    # (a truncated xz stream would make lzma raise while reading it).
    print("P-1 SKIPPED: post-processing has not finished (no", pp_exit, ")")
kernel_lines = [ln for ln in expected if not ln.startswith("MemcpyHtoD,")]

stats = {}
for ln in open(glob.glob(os.path.join(traces_dir, "stats_ctx_*"))[0]).read().splitlines()[1:]:
    parts = ln.split(",")
    stats[int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1))] = int(parts[-1])

used = collections.Counter()
audit_bad = {}
for kid in ids:
    name = next(k for k in kernel_lines if re.match(rf"kernel-{kid}-ctx", k))
    raw_n = raw_ld = bad_pairs = 0
    pending, last_pc = {}, {}
    with lzma.open(os.path.join(traces_dir, name.replace(".traceg.xz", ".trace.xz")), "rt", errors="replace") as f:
        for line in f:
            if not line or line[0] in "-#\n":
                continue
            t = line.split()
            if len(t) < 8:
                continue
            key = (t[0], t[1], t[2], t[3])
            pc = t[4]
            op = t[7 + int(t[6])]
            raw_n += 1
            if "LDGSTS" in op:
                raw_ld += 1
                if key in pending:
                    if pending.pop(key) != pc or last_pc.get(key) != pc:
                        bad_pairs += 1
                else:
                    pending[key] = pc
            elif key in pending:
                bad_pairs += 1
                pending.pop(key)
            last_pc[key] = pc
    bad_pairs += len(pending)
    g_n = g_ld = 0
    with lzma.open(os.path.join(traces_dir, name), "rt", errors="replace") as f:
        for line in f:
            if not inst_re.match(line):
                continue
            t = line.split()
            op = t[3 + int(t[2])]
            used[op.split(".")[0]] += 1
            g_n += 1
            if "LDGSTS" in op:
                g_ld += 1
    ok = (raw_n == stats[kid] and raw_ld % 2 == 0 and raw_n - g_n == raw_ld // 2 and g_ld == raw_ld // 2
          and bad_pairs == 0)
    print(f"  kernel {kid}: stats={stats[kid]} raw={raw_n} traceg={g_n} raw_LDGSTS={raw_ld} traceg_LDGSTS={g_ld} "
          f"unpaired={bad_pairs} {'OK' if ok else 'MISMATCH'}")
    if not ok:
        audit_bad[kid] = (stats[kid], raw_n, g_n, raw_ld, g_ld, bad_pairs)
unknown = {op: c for op, c in used.items() if op not in known}
check(f"P-2 no opcode missing from the simulator table (kernels {first_id}-{last_id})", not unknown,
      unknown if unknown else f"{len(used)} distinct opcodes")
check(f"P-2 LDGSTS merge audit (kernels {first_id}-{last_id})", not audit_bad,
      audit_bad if audit_bad else f"{len(ids)}/{len(ids)}")
print("opcodes by count:", used.most_common())
print("S6 TRACES", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
