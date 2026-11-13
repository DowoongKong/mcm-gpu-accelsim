# Step 7: batch 32 x 256 input tokens; trace the prefill call of decoder layer 0 (norm -> attention -> norm -> FFN)
# with one NVBit window, and record the kernel-id ranges of the three steps (prefill, decode, decode).
# Explained in docs/s7_b32_prefill_layer.ko.md.
# Usage (inside llmtrace, tracer attached, ACTIVE_FROM_START=0): s7_prefill_layer_capture.py <log dir>
import glob
import json
import os
import re
import sys
import threading
import time

LOG_DIR = sys.argv[1]
S, BATCH, STRIDE, MAX_TOKENS = 256, 32, 150, 3
KV_BYTES = 1536 << 20
TARGET = "model.layers.0"
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
assert "CUDA_INJECTION64_PATH" in os.environ, "step 7 needs the tracer attached"
assert os.environ.get("ACTIVE_FROM_START") == "0", "step 7 requires ACTIVE_FROM_START=0"
OUT = os.environ["TRACES_FOLDER"]
TRACES = os.path.join(OUT, "traces")
assert not os.path.exists(TRACES) or not os.listdir(TRACES), f"{TRACES} must start empty"

import torch
from vllm import SamplingParams

from vllm_setup import build_batch_prompts, load_tokenizer, make_llm_batch

MODEL_LOAD_COPIES = (518, 16060523412)  # step 1 D-run: model preparation copies (docs/s1_tracer_async_memcpy.ko.md 8)


def stats_path():
    paths = glob.glob(os.path.join(TRACES, "stats_ctx_*"))
    assert len(paths) == 1, f"expected exactly one CUDA context stats file, found {paths}"
    return paths[0]


def launched_kernels():
    with open(stats_path()) as f:
        return sum(1 for _ in f) - 1


print("S7 start", time.strftime("%Y-%m-%d %H:%M:%S"), "S", S, "batch", BATCH)
tok = load_tokenizer()
prompts = build_batch_prompts(tok, S, BATCH, STRIDE)
t0 = time.perf_counter()
llm = make_llm_batch(S, BATCH, KV_BYTES)
t_init = time.perf_counter() - t0

main_thread = threading.get_ident()
events = []
calls = {"layer": 0, "execute_model": 0, "sample_tokens": 0}


def attach(worker):
    runner = worker.model_runner
    for fname in ("execute_model", "sample_tokens"):
        original = getattr(runner, fname)

        def wrapped(*args, _fname=fname, _original=original, **kwargs):
            calls[_fname] += 1
            call = calls[_fname]
            events.append({"fn": _fname, "kind": "enter", "call": call, "launched": launched_kernels()})
            result = _original(*args, **kwargs)
            events.append({"fn": _fname, "kind": "exit", "call": call, "launched": launched_kernels()})
            return result

        setattr(runner, fname, wrapped)
    layer = worker.get_model().get_submodule(TARGET)

    def on_enter(m, args, kwargs):
        calls["layer"] += 1
        hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1]   # layer(positions, hidden_states, residual)
        if calls["layer"] == 1:
            torch.cuda.profiler.start()  # cudaProfilerStart -> tracer sets active_region on exit
        events.append({"fn": "layer", "kind": "enter", "call": calls["layer"], "tokens": int(hs.shape[0]),
                       "launched": launched_kernels(), "thread": threading.get_ident()})

    def on_exit(m, args, kwargs, output):
        if calls["layer"] == 1:
            torch.cuda.profiler.stop()   # cudaProfilerStop -> tracer clears active_region on exit
        events.append({"fn": "layer", "kind": "exit", "call": calls["layer"], "tokens": int(output[0].shape[0]),
                       "launched": launched_kernels(), "thread": threading.get_ident()})

    layer.register_forward_pre_hook(on_enter, with_kwargs=True)
    layer.register_forward_hook(on_exit, with_kwargs=True)


llm.collective_rpc(attach)
params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True, detokenize=False)
t1 = time.perf_counter()
outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
t_gen = time.perf_counter() - t1
out_ids = [list(o.outputs[0].token_ids) for o in outs]
print("init_s", round(t_init, 1), "generate_s", round(t_gen, 3))

ev = {(e["fn"], e["kind"], e["call"]): e for e in events}
layer_calls = [(e["call"], e["tokens"]) for e in events if e["fn"] == "layer" and e["kind"] == "enter"]
window = (ev[("layer", "enter", 1)]["launched"] + 1, ev[("layer", "exit", 1)]["launched"])
steps = {}
for call in range(1, calls["execute_model"] + 1):
    steps[f"step{call}"] = (ev[("execute_model", "enter", call)]["launched"] + 1, ev[("sample_tokens", "exit", call)]["launched"])
with open(os.path.join(OUT, "regions.tsv"), "w") as f:
    f.write("tag\tphase\tfirst_kernel_id\tlast_kernel_id\tn_kernels\ttraced\n")
    f.write(f"{TARGET}\tprefill\t{window[0]}\t{window[1]}\t{window[1] - window[0] + 1}\t1\n")
    for name, (first, last) in steps.items():
        f.write(f"step\t{name}\t{first}\t{last}\t{last - first + 1}\t0\n")
traced_ids = list(range(window[0], window[1] + 1))

rows = {}
with open(stats_path()) as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\n").split(",")
        nums = [int(p) for p in parts[-10:]]
        rows[int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1))] = {
            "name": ",".join(parts[1:-10]).strip(), "reported_insts": nums[9]}
kernelslist = glob.glob(os.path.join(TRACES, "kernelslist_ctx_*"))
klines = open(kernelslist[0]).read().splitlines() if len(kernelslist) == 1 else []
memcpy = [ln for ln in klines if ln.startswith("MemcpyHtoD,")]
kl_ids = [int(re.match(r"kernel-(\d+)-ctx", ln).group(1)) for ln in klines if not ln.startswith("MemcpyHtoD,")]
trace_files = sorted(glob.glob(os.path.join(TRACES, "kernel-*.trace*")))
tf_ids = sorted(int(re.search(r"kernel-(\d+)-ctx", os.path.basename(p)).group(1)) for p in trace_files)

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


check("G7-1 layer 0 called 3 times with 8192 / 32 / 32 tokens (prefill in one step)",
      layer_calls == [(1, BATCH * S), (2, BATCH), (3, BATCH)], layer_calls)
check("hooks and wrappers on the main thread", all(e.get("thread", main_thread) == main_thread for e in events))
check("G7-2 32 requests, 3 output tokens each", len(out_ids) == BATCH and all(len(o) == MAX_TOKENS for o in out_ids),
      f"{len(out_ids)} outputs")
check("one CUDA context", len(kernelslist) == 1 and len(glob.glob(os.path.join(TRACES, "stats_ctx_*"))) == 1)
check("G7-3 kernelslist kernel ids == window", kl_ids == traced_ids, f"{len(kl_ids)} listed, window {window}")
check("G7-3 trace file ids == window", tf_ids == traced_ids, f"{len(tf_ids)} files")
check("G7-3 every trace file is non-empty", all(os.path.getsize(p) > 0 for p in trace_files))
check("G7-3 instructions reported only for window kernels",
      all((r["reported_insts"] > 0) == (k in set(traced_ids)) for k, r in rows.items()))
load = memcpy[:MODEL_LOAD_COPIES[0]]
check("G7-4 model-load copies == step 1 (first 518 lines)",
      (len(load), sum(int(ln.split(",")[2]) for ln in load)) == MODEL_LOAD_COPIES,
      (len(load), sum(int(ln.split(",")[2]) for ln in load), "total copy lines", len(memcpy)))

print("window kernels:")
for k in traced_ids:
    print(f"  {k} reported_warp={rows[k]['reported_insts']} {rows[k]['name'][:90]}")
summary = {
    "S": S, "batch": BATCH, "stride": STRIDE, "max_tokens": MAX_TOKENS, "init_s": t_init, "generate_s": t_gen,
    "output_ids": out_ids, "layer_calls": layer_calls, "window": window, "steps": steps,
    "launched_total": len(rows), "trace_files": len(trace_files),
    "trace_bytes": sum(os.path.getsize(p) for p in trace_files),
    "reported_insts_window": sum(rows[k]["reported_insts"] for k in traced_ids),
    "memcpy_lines": len(memcpy), "memcpy_bytes": sum(int(ln.split(",")[2]) for ln in memcpy),
}
with open(os.path.join(LOG_DIR, "s7_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("steps", steps, "window", window, "trace bytes", summary["trace_bytes"],
      "reported warp in window", summary["reported_insts_window"])
print("S7 CAPTURE", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
