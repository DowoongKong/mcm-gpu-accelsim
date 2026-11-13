# Step 2: open NVBit trace windows around vLLM decoder sublayers with torch.cuda.profiler.start/stop,
# record kernel-id ranges in regions.tsv, and check that exactly the window kernels were traced.
# Usage (inside llmtrace, tracer attached, ACTIVE_FROM_START=0): s2_windows.py <S> <log dir> <phases>
#   phases: comma list of prefill, decode (step 2 uses decode only). Explained in docs/s2_trace_windows.ko.md.
import glob
import json
import lzma
import os
import re
import sys
import threading
import time

S = int(sys.argv[1])
LOG_DIR = sys.argv[2]
PHASES = set(sys.argv[3].split(","))
assert PHASES <= {"prefill", "decode"}, PHASES
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
assert "CUDA_INJECTION64_PATH" in os.environ, "step 2 needs the tracer attached"
# Windows are driven only by cudaProfilerStart/Stop, so tracing must start switched off.
assert os.environ.get("ACTIVE_FROM_START") == "0", "step 2 requires ACTIVE_FROM_START=0"
OUT = os.environ["TRACES_FOLDER"]
TRACES = os.path.join(OUT, "traces")
assert not os.path.exists(TRACES) or not os.listdir(TRACES), f"{TRACES} must start empty"

import torch
from vllm import SamplingParams

from vllm_setup import build_prompt_ids, load_tokenizer, make_llm

LAYERS = (0, 15, 31)
SUBLAYERS = ("self_attn", "mlp")
TARGETS = [f"model.layers.{i}.{sub}" for i in LAYERS for sub in SUBLAYERS]
V0_OUTPUT_S256 = [7533, 9612]
EXPECTED_MEMCPY_S256 = (534, 16060528722)  # step 1, build N (docs/s1_tracer_async_memcpy.ko.md 8.3)
STEP1_N_SUMMARY = "/accel-sim/llm_exp/tracer_tests/vllm_N_S256/logs/v2_summary_tracer_S256.json"


def stats_path():
    paths = glob.glob(os.path.join(TRACES, "stats_ctx_*"))
    assert len(paths) == 1, f"expected exactly one CUDA context stats file, found {paths}"
    return paths[0]


def launched_kernels():
    # One stats line per kernel launch, written whether or not the window is open (validated in V2).
    with open(stats_path()) as f:
        return sum(1 for _ in f) - 1


print("S2 start", time.strftime("%Y-%m-%d %H:%M:%S"), "S", S, "phases", sorted(PHASES))
tok = load_tokenizer()
ids, text_ids = build_prompt_ids(tok, S)
t0 = time.perf_counter()
llm = make_llm(S)
t_init = time.perf_counter() - t0

main_thread = threading.get_ident()
events = []
calls = {}
open_windows = []  # guards against overlapping windows


def hidden_tokens(args, kwargs):
    # LlamaDecoderLayer calls self_attn with keyword arguments and mlp positionally (llama.py:328, :332).
    hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
    return int(hs.shape[0])


def attach(worker):
    model = worker.get_model()
    for tag in TARGETS:
        mod = model.get_submodule(tag)

        def on_enter(m, args, kwargs, tag=tag):
            calls[tag] = calls.get(tag, 0) + 1
            phase = "prefill" if calls[tag] == 1 else "decode"
            traced = phase in PHASES
            launched = launched_kernels()
            if traced:
                assert not open_windows, f"window {tag} opened while {open_windows} is open"
                open_windows.append(tag)
                torch.cuda.profiler.start()  # cudaProfilerStart -> tracer sets active_region on exit
            events.append({"tag": tag, "kind": "enter", "call": calls[tag], "phase": phase, "traced": traced,
                           "tokens": hidden_tokens(args, kwargs), "thread": threading.get_ident(),
                           "launched": launched})

        def on_exit(m, args, kwargs, output, tag=tag):
            phase = "prefill" if calls[tag] == 1 else "decode"
            traced = phase in PHASES
            if traced:
                torch.cuda.profiler.stop()  # cudaProfilerStop -> tracer clears active_region on exit
                open_windows.remove(tag)
            events.append({"tag": tag, "kind": "exit", "call": calls[tag], "phase": phase, "traced": traced,
                           "tokens": int(output.shape[0]), "thread": threading.get_ident(),
                           "launched": launched_kernels()})

        mod.register_forward_pre_hook(on_enter, with_kwargs=True)
        mod.register_forward_hook(on_exit, with_kwargs=True)


llm.collective_rpc(attach)
params = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True, detokenize=False)
t1 = time.perf_counter()
out = llm.generate({"prompt_token_ids": ids}, params)[0]
t_gen = time.perf_counter() - t1
out_ids = list(out.outputs[0].token_ids)
print("init_s", round(t_init, 1), "generate_s", round(t_gen, 3))
print("output_token_ids", out_ids, "text", repr(tok.decode(out_ids)))

# Windows and regions.tsv (kernel ids first..last launched inside each hook pair).
windows = []
for i in range(0, len(events), 2):
    enter, leave = events[i], events[i + 1]
    windows.append({"tag": enter["tag"], "phase": enter["phase"], "call": enter["call"], "traced": enter["traced"],
                    "first": enter["launched"] + 1, "last": leave["launched"]})
with open(os.path.join(OUT, "regions.tsv"), "w") as f:
    f.write("tag\tphase\tfirst_kernel_id\tlast_kernel_id\tn_kernels\ttraced\n")
    for w in windows:
        f.write(f"{w['tag']}\t{w['phase']}\t{w['first']}\t{w['last']}\t{w['last'] - w['first'] + 1}\t{int(w['traced'])}\n")
traced_ids = sorted(k for w in windows if w["traced"] for k in range(w["first"], w["last"] + 1))

# Tracer outputs.
rows = []
with open(stats_path()) as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\n").split(",")
        nums = [int(p) for p in parts[-10:]]
        rows.append({"id": int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1)),
                     "name": ",".join(parts[1:-10]).strip(), "total_insts": nums[8], "reported_insts": nums[9]})
by_id = {r["id"]: r for r in rows}
kernelslist = glob.glob(os.path.join(TRACES, "kernelslist_ctx_*"))
klines = open(kernelslist[0]).read().splitlines() if len(kernelslist) == 1 else []
memcpy = [ln for ln in klines if ln.startswith("MemcpyHtoD,")]
kernel_lines = [ln for ln in klines if not ln.startswith("MemcpyHtoD,")]
kl_ids = [int(re.match(r"kernel-(\d+)-ctx", ln).group(1)) for ln in kernel_lines]
trace_files = sorted(glob.glob(os.path.join(TRACES, "kernel-*.trace*")))
tf_ids = sorted(int(re.search(r"kernel-(\d+)-ctx", os.path.basename(p)).group(1)) for p in trace_files)

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


expected_order = [(tag, kind) for step in (1, 2) for tag in TARGETS for kind in ("enter", "exit")]
check("hook event order (V1)", [(e["tag"], e["kind"]) for e in events] == expected_order, f"{len(events)} events")
check("prefill tokens == S, decode tokens == 1 (V1)",
      all(e["tokens"] == (S if e["call"] == 1 else 1) for e in events), f"S={S}")
check("all hooks on main thread (V1)", all(e["thread"] == main_thread for e in events))
if S == 256:
    check("output equals V0", out_ids == V0_OUTPUT_S256, out_ids)
check("one CUDA context", len(kernelslist) == 1 and len(glob.glob(os.path.join(TRACES, "stats_ctx_*"))) == 1)
check("kernelslist kernel ids == window kernel ids (only window kernels traced)", kl_ids == traced_ids,
      f"{len(kl_ids)} listed vs {len(traced_ids)} in windows")
check("trace file ids == window kernel ids", tf_ids == traced_ids, f"{len(tf_ids)} files")
check("every trace file is non-empty", all(os.path.getsize(p) > 0 for p in trace_files))
check("instructions reported only for window kernels",
      all((r["reported_insts"] > 0) == (r["id"] in set(traced_ids)) for r in rows))
if S == 256:
    check("MemcpyHtoD lines/bytes == step 1",
          (len(memcpy), sum(int(ln.split(",")[2]) for ln in memcpy)) == EXPECTED_MEMCPY_S256,
          (len(memcpy), sum(int(ln.split(",")[2]) for ln in memcpy)))
    ref = json.load(open(STEP1_N_SUMMARY))
    check("window kernel ranges == step 1 (build N, windows closed)",
          [[w["tag"], w["call"], w["first"], w["last"]] for w in windows] == ref["windows"])

# Records.
noinc = {}
insts = {}
for p in trace_files:
    kid = int(re.search(r"kernel-(\d+)-ctx", os.path.basename(p)).group(1))
    with lzma.open(p, "rt", errors="replace") as fh:
        noinc[kid] = sum(1 for ln in fh if "CALL.REL.NOINC" in ln)
    insts[kid] = by_id[kid]["reported_insts"]
with open(os.path.join(LOG_DIR, f"s2_windows_S{S}.tsv"), "w") as f:
    f.write("tag\tphase\tfirst_id\tlast_id\ttraced\treported_insts\ttrace_bytes\tcall_rel_noinc_lines\tkernels\n")
    for w in windows:
        ks = range(w["first"], w["last"] + 1)
        tb = sum(os.path.getsize(p) for p in trace_files
                 if int(re.search(r"kernel-(\d+)-ctx", os.path.basename(p)).group(1)) in ks)
        f.write(f"{w['tag']}\t{w['phase']}\t{w['first']}\t{w['last']}\t{int(w['traced'])}\t"
                f"{sum(insts.get(k, 0) for k in ks)}\t{tb}\t{sum(noinc.get(k, 0) for k in ks)}\t"
                f"{' | '.join(by_id[k]['name'][:60] for k in ks)}\n")
first_kernel_pos = next((i for i, ln in enumerate(klines) if not ln.startswith("MemcpyHtoD,")), None)
summary = {
    "S": S, "phases": sorted(PHASES), "init_s": t_init, "generate_s": t_gen, "output_ids": out_ids,
    "windows": [[w["tag"], w["call"], w["first"], w["last"], w["traced"]] for w in windows],
    "traced_kernels": len(traced_ids), "trace_files": len(trace_files),
    "trace_bytes": sum(os.path.getsize(p) for p in trace_files),
    "reported_insts_total": sum(insts.values()), "call_rel_noinc_lines": sum(noinc.values()),
    "memcpy_lines": len(memcpy), "memcpy_lines_before_first_kernel": first_kernel_pos,
}
with open(os.path.join(LOG_DIR, f"s2_summary_S{S}.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("traced kernels", len(traced_ids), "trace files", len(trace_files), "trace bytes", summary["trace_bytes"],
      "reported insts", summary["reported_insts_total"], "CALL.REL.NOINC lines", summary["call_rel_noinc_lines"])
print("S2 RESULT", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
print("S2 end", time.strftime("%Y-%m-%d %H:%M:%S"))
