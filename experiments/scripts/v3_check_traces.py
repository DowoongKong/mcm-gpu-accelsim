# V3 (steps 1-2): check post-processed traces before simulation.
# - kernelslist.g equals the tracer's kernelslist_ctx_* with .trace.xz renamed to .traceg.xz
# - every kernel line has a non-empty .traceg.xz
# - every opcode in the traces exists in the simulator's Ampere opcode table (the simulator asserts otherwise)
# - instruction lines per kernel equal the tracer's reported instruction count in stats
# Opcode parsing follows graphalytics_exp/scripts/cugraph_mst/check_opcodes.py (used for the cuGraph traces).
# Usage: v3_check_traces.py <traces dir> <ampere_opcode.h> <postprocess log>. Explained in docs/v3_postprocess_sim.ko.md.
import collections
import glob
import lzma
import os
import re
import sys

traces_dir, table_path, pp_log = sys.argv[1], sys.argv[2], sys.argv[3]
known = set(re.findall(r'\{"([A-Z0-9_]+)",\s*OpcodeChar', open(table_path).read()))
inst_re = re.compile(r"^[0-9a-f]{4,} [0-9a-f]{8} \d+ ")

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


raw = open(glob.glob(os.path.join(traces_dir, "kernelslist_ctx_*"))[0]).read().splitlines()
converted = open(os.path.join(traces_dir, "kernelslist.g")).read().splitlines()
expected = [ln[:-3] + "g.xz" if ln.endswith(".trace.xz") else ln for ln in raw]
memcpy = [ln for ln in converted if ln.startswith("MemcpyHtoD,")]
kernels = [ln for ln in converted if not ln.startswith("MemcpyHtoD,")]
check("post-processing log has no 'Undefined command'", "Undefined command" not in open(pp_log).read())
check("kernelslist.g == kernelslist_ctx with .traceg.xz names", converted == expected, f"{len(converted)} lines")
check("kernelslist.g: 534 MemcpyHtoD + 30 kernel lines", (len(memcpy), len(kernels)) == (534, 30), (len(memcpy), len(kernels)))
check("every kernel line has a non-empty .traceg.xz",
      all(os.path.isfile(os.path.join(traces_dir, k)) and os.path.getsize(os.path.join(traces_dir, k)) > 0 for k in kernels))

stats = {}
for ln in open(glob.glob(os.path.join(traces_dir, "stats_ctx_*"))[0]).read().splitlines()[1:]:
    parts = ln.split(",")
    stats[int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1))] = int(parts[-1])

used = collections.Counter()
per_kernel = {}
noinc = {}
for k in kernels:
    kid = int(re.search(r"kernel-(\d+)-ctx", k).group(1))
    n = 0
    c = 0
    with lzma.open(os.path.join(traces_dir, k), "rt", errors="replace") as f:
        for line in f:
            if not inst_re.match(line):
                continue
            tok = line.split()
            ndest = int(tok[2])
            opcode = tok[3 + ndest]
            used[opcode.split(".")[0]] += 1
            if opcode.startswith("CALL.REL.NOINC"):
                c += 1
            n += 1
    per_kernel[kid] = n
    noinc[kid] = c

unknown = {op: c for op, c in used.items() if op not in known}
check("no opcode missing from the simulator table", not unknown, unknown if unknown else "none")
mismatch = {kid: (per_kernel[kid], stats[kid]) for kid in per_kernel if per_kernel[kid] != stats[kid]}
check("instruction lines per kernel == stats reported instructions", not mismatch, mismatch if mismatch else "30/30")
print("table opcodes:", len(known), "| distinct opcodes used:", len(used), "| instruction lines:", sum(per_kernel.values()))
print("CALL.REL.NOINC lines:", sum(noinc.values()), {k: v for k, v in noinc.items() if v})
print("opcodes by count:", used.most_common())
print("traceg bytes:", sum(os.path.getsize(os.path.join(traces_dir, k)) for k in kernels))
print("V3 TRACES", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
