# Step 5 (B): trace one whole decode step (S=256) with a single NVBit window.
# The window opens when the second GPUModelRunner.execute_model call starts and closes when the second
# sample_tokens call returns (EngineCore.step runs execute_model then sample_tokens; step 1 = prefill,
# step 2 = decode). Both steps' kernel-id ranges go to regions.tsv. Explained in docs/s5_decode_full_step_capture.ko.md.
# Usage (inside llmtrace, tracer attached, ACTIVE_FROM_START=0): s5_decode_full.py <log dir>
import glob
import json
import os
import re
import sys
import threading
import time

LOG_DIR = sys.argv[1]
S = 256
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
assert "CUDA_INJECTION64_PATH" in os.environ, "step 5 needs the tracer attached"
assert os.environ.get("ACTIVE_FROM_START") == "0", "step 5 requires ACTIVE_FROM_START=0"
OUT = os.environ["TRACES_FOLDER"]
TRACES = os.path.join(OUT, "traces")
assert not os.path.exists(TRACES) or not os.listdir(TRACES), f"{TRACES} must start empty"

import torch
from vllm import SamplingParams

from vllm_setup import build_prompt_ids, load_tokenizer, make_llm

V0_OUTPUT_S256 = [7533, 9612]
EXPECTED_MEMCPY_S256 = (534, 16060528722)  # step 1 (docs/s1_tracer_async_memcpy.ko.md 8.3)
EXPECTED_STEPS = {"prefill": (1002, 1338), "decode": (1339, 1739)}  # docs/s256_full_step_estimate.ko.md 1
KERNEL_MAP = "/accel-sim/llm_exp/traces/s2_decode_S256/s256_generate_kernel_map.tsv"
S2_STATS = glob.glob("/accel-sim/llm_exp/traces/s2_decode_S256/traces/stats_ctx_*")


def stats_path():
    paths = glob.glob(os.path.join(TRACES, "stats_ctx_*"))
    assert len(paths) == 1, f"expected exactly one CUDA context stats file, found {paths}"
    return paths[0]


def launched_kernels():
    # One stats line per kernel launch, written whether or not the window is open (validated in V2).
    with open(stats_path()) as f:
        return sum(1 for _ in f) - 1


def read_stats(path):
    rows = {}
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            nums = [int(p) for p in parts[-10:]]
            rows[int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1))] = {
                "name": ",".join(parts[1:-10]).strip(), "reported_insts": nums[9]}
    return rows


print("S5 start", time.strftime("%Y-%m-%d %H:%M:%S"), "S", S)
tok = load_tokenizer()
ids, text_ids = build_prompt_ids(tok, S)
t0 = time.perf_counter()
llm = make_llm(S)
t_init = time.perf_counter() - t0

main_thread = threading.get_ident()
events = []
calls = {"execute_model": 0, "sample_tokens": 0}
window = []


def attach(worker):
    runner = worker.model_runner
    for fname in ("execute_model", "sample_tokens"):
        original = getattr(runner, fname)

        def wrapped(*args, _fname=fname, _original=original, **kwargs):
            calls[_fname] += 1
            call = calls[_fname]
            if _fname == "execute_model" and call == 2:
                assert not window, "window already open"
                window.append("decode")
                torch.cuda.profiler.start()  # cudaProfilerStart -> tracer sets active_region on exit
            events.append({"fn": _fname, "kind": "enter", "call": call, "launched": launched_kernels(),
                           "thread": threading.get_ident()})
            result = _original(*args, **kwargs)
            events.append({"fn": _fname, "kind": "exit", "call": call, "launched": launched_kernels(),
                           "thread": threading.get_ident()})
            if _fname == "sample_tokens" and call == 2:
                torch.cuda.profiler.stop()  # cudaProfilerStop -> tracer clears active_region on exit
                window.remove("decode")
            return result

        setattr(runner, fname, wrapped)


llm.collective_rpc(attach)
params = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True, detokenize=False)
t1 = time.perf_counter()
out = llm.generate({"prompt_token_ids": ids}, params)[0]
t_gen = time.perf_counter() - t1
out_ids = list(out.outputs[0].token_ids)
print("init_s", round(t_init, 1), "generate_s", round(t_gen, 3))
print("output_token_ids", out_ids)

ev = {(e["fn"], e["kind"], e["call"]): e for e in events}
steps = {}
for call, step in ((1, "prefill"), (2, "decode")):
    steps[step] = (ev[("execute_model", "enter", call)]["launched"] + 1, ev[("sample_tokens", "exit", call)]["launched"])
gap = (ev[("sample_tokens", "exit", 1)]["launched"] + 1, ev[("execute_model", "enter", 2)]["launched"])
with open(os.path.join(OUT, "regions.tsv"), "w") as f:
    f.write("tag\tphase\tfirst_kernel_id\tlast_kernel_id\tn_kernels\ttraced\n")
    for step, (first, last) in steps.items():
        f.write(f"step\t{step}\t{first}\t{last}\t{last - first + 1}\t{int(step == 'decode')}\n")
traced_ids = list(range(steps["decode"][0], steps["decode"][1] + 1))

rows = read_stats(stats_path())
s2_rows = read_stats(S2_STATS[0])
kernelslist = glob.glob(os.path.join(TRACES, "kernelslist_ctx_*"))
klines = open(kernelslist[0]).read().splitlines() if len(kernelslist) == 1 else []
memcpy = [ln for ln in klines if ln.startswith("MemcpyHtoD,")]
kl_ids = [int(re.match(r"kernel-(\d+)-ctx", ln).group(1)) for ln in klines if not ln.startswith("MemcpyHtoD,")]
trace_files = sorted(glob.glob(os.path.join(TRACES, "kernel-*.trace*")))
tf_ids = sorted(int(re.search(r"kernel-(\d+)-ctx", os.path.basename(p)).group(1)) for p in trace_files)
map_names = {}
for ln in open(KERNEL_MAP).read().splitlines()[1:]:
    f = ln.split("\t")
    map_names[int(f[0])] = f[6]
sample_ids = [int(ln.split("\t")[2]) + k for ln in open("/accel-sim/llm_exp/traces/s2_decode_S256/regions.tsv").read().splitlines()[1:]
              if ln.split("\t")[1] == "decode" for k in range(int(ln.split("\t")[4]))]

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


expected_order = [(fn, kind, call) for call in (1, 2) for fn in ("execute_model", "sample_tokens") for kind in ("enter", "exit")]
check("step call order: execute_model, sample_tokens per step, 2 steps",
      [(e["fn"], e["kind"], e["call"]) for e in events] == expected_order, f"{len(events)} events")
check("wrappers ran on the main thread", all(e["thread"] == main_thread for e in events))
check("G5-1 output equals V0", out_ids == V0_OUTPUT_S256, out_ids)
check("one CUDA context", len(kernelslist) == 1 and len(glob.glob(os.path.join(TRACES, "stats_ctx_*"))) == 1)
check("no kernel launched between the two steps", gap[1] < gap[0], f"gap ids {gap}")
check("G5-2 kernelslist kernel ids == decode step range", kl_ids == traced_ids, f"{len(kl_ids)} listed vs {len(traced_ids)}")
check("G5-2 trace file ids == decode step range", tf_ids == traced_ids, f"{len(tf_ids)} files")
check("G5-2 every trace file is non-empty", all(os.path.getsize(p) > 0 for p in trace_files))
check("G5-2 instructions reported only for window kernels",
      all((r["reported_insts"] > 0) == (k in set(traced_ids)) for k, r in rows.items()))
check("G5-3 prefill / decode step ranges == kernel map", steps == EXPECTED_STEPS, steps)
check("G5-3 kernel names 1002-1739 == kernel map",
      all(rows.get(k, {}).get("name") == map_names[k] for k in map_names) and max(rows) == 1739,
      f"{sum(rows.get(k, {}).get('name') == map_names[k] for k in map_names)}/{len(map_names)}, last id {max(rows)}")
check("G5-4 MemcpyHtoD lines/bytes == step 1",
      (len(memcpy), sum(int(ln.split(",")[2]) for ln in memcpy)) == EXPECTED_MEMCPY_S256,
      (len(memcpy), sum(int(ln.split(",")[2]) for ln in memcpy)))
diff5 = {k: (rows[k]["reported_insts"], s2_rows[k]["reported_insts"]) for k in sample_ids
         if rows[k]["reported_insts"] != s2_rows[k]["reported_insts"]}
check("G5-5 the 30 sample kernels report the same instructions as step 2", len(sample_ids) == 30 and not diff5,
      diff5 if diff5 else "30/30")

summary = {
    "S": S, "init_s": t_init, "generate_s": t_gen, "output_ids": out_ids, "steps": steps, "gap": gap,
    "traced_kernels": len(traced_ids), "trace_files": len(trace_files),
    "trace_bytes": sum(os.path.getsize(p) for p in trace_files),
    "reported_insts_total": sum(rows[k]["reported_insts"] for k in traced_ids if k in rows),
    "memcpy_lines": len(memcpy),
    "memcpy_positions": [i for i, ln in enumerate(klines) if ln.startswith("MemcpyHtoD,")][-16:],
}
with open(os.path.join(LOG_DIR, "s5_summary_S256.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("traced kernels", len(traced_ids), "trace files", len(trace_files), "trace bytes", summary["trace_bytes"],
      "reported insts", summary["reported_insts_total"])
print("S5 CAPTURE", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
