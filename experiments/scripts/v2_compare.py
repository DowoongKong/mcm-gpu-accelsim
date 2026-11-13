# V2: compare the baseline run and the tracer-attached run of scripts/v2_tracer.py for one S.
import json
import os
import sys

S = int(sys.argv[1]) if len(sys.argv) > 1 else 256
LOG_DIR = sys.argv[2] if len(sys.argv) > 2 else "logs"
base = json.load(open(os.path.join(LOG_DIR, f"v2_summary_baseline_S{S}.json")))
trac = json.load(open(os.path.join(LOG_DIR, f"v2_summary_tracer_S{S}.json")))

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


check("output ids identical", base["output_ids"] == trac["output_ids"], f"{base['output_ids']} vs {trac['output_ids']}")
check("hook events identical (tag, kind, call, tokens)", base["events"] == trac["events"], f"{len(trac['events'])} events")
print(f"init_s      baseline {base['init_s']:.1f}  tracer {trac['init_s']:.1f}  ratio {trac['init_s'] / base['init_s']:.2f}x")
print(f"generate_s  baseline {base['generate_s']:.3f}  tracer {trac['generate_s']:.3f}  "
      f"ratio {trac['generate_s'] / base['generate_s']:.2f}x")
print(f"device_used_GiB baseline {base['device_used_gib']:.2f}  tracer {trac['device_used_gib']:.2f}")
print("V2 COMPARE", f"S={S}", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
