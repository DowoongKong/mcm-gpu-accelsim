# Hypothesis-1 decode batch sweep: trace ONLY the layer-0 kernels of the first step in which every one of the `batch`
# sequences is decoding (docs/h1_llama70b_decode_batch_sweep.ko.md). Large batches prefill over several steps
# (chunked prefill, fixed token budget), and vLLM mixes decode of finished prompts with prefill of the rest, so the
# window is armed from the scheduler output: the first execute_model whose num_scheduled_tokens has exactly `batch`
# entries, all equal to 1. The window opens at the layer-0 forward pre-hook and closes at its forward hook, so prep,
# other layers and the tail are not traced. MemcpyHtoD records are window-independent (weights are still listed).
# Usage (inside llmtrace, tracer attached, ACTIVE_FROM_START=0):
#   h1_decode_layer_capture.py <out dir> <model: llama70b|deepseek|qwen> <batch> <S> <max_tokens> <max_batched_tokens> <kv bytes>
import glob
import json
import os
import re
import sys
import threading
import time

OUT, MODEL_KEY = sys.argv[1], sys.argv[2]
BATCH, S, MAX_TOKENS, MAX_BATCHED, KV_BYTES = (int(x) for x in sys.argv[3:8])
MODELS = {"deepseek": ("/hf_cache/DeepSeek-R1-Distill-Llama-8B", 32, 128000),
          "qwen": ("/hf_cache/Qwen2.5-7B", 28, None),
          "llama70b": ("/hf_cache/Llama-3.3-70B-Instruct-L3", 3, 128000),
          # H2: Mixtral-8x7B-v0.1 truncated to decoder layers 0..2 (docs/h2_mixtral8x7b_prep.ko.md); BOS <s> = 1
          "mixtral": ("/hf_cache/Mixtral-8x7B-v0.1-L3", 3, 1)}
MODEL_DIR, N_LAYERS, BOS_ID = MODELS[MODEL_KEY]
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
assert "CUDA_INJECTION64_PATH" in os.environ, "the tracer must be attached"
assert os.environ.get("ACTIVE_FROM_START") == "0", "ACTIVE_FROM_START=0 is required"
assert os.path.realpath(os.environ["TRACES_FOLDER"]) == os.path.realpath(OUT), "TRACES_FOLDER must be the out dir"
TRACES = os.path.join(OUT, "traces")
assert not os.path.exists(TRACES) or not os.listdir(TRACES), f"{TRACES} must start empty"

import torch
from vllm import SamplingParams

from vllm_setup import build_prompts_adaptive, load_tokenizer_from, make_llm_decode_sweep


def stats_path():
    paths = glob.glob(os.path.join(TRACES, "stats_ctx_*"))
    assert len(paths) == 1, f"expected exactly one CUDA context stats file, found {paths}"
    return paths[0]


def launched_kernels():
    with open(stats_path()) as f:
        return sum(1 for _ in f) - 1


print("H1 start", time.strftime("%Y-%m-%d %H:%M:%S"), MODEL_KEY, "S", S, "batch", BATCH, "max_tokens", MAX_TOKENS,
      "max_batched", MAX_BATCHED, "kv_bytes", KV_BYTES)
tok = load_tokenizer_from(MODEL_DIR)
prompts, stride = build_prompts_adaptive(tok, S, BATCH, bos_id=BOS_ID)
print("prompt stride", stride)
t0 = time.perf_counter()
llm = make_llm_decode_sweep(MODEL_DIR, S, BATCH, KV_BYTES, MAX_TOKENS, MAX_BATCHED)
t_init = time.perf_counter() - t0

main_thread = threading.get_ident()
state = {"step": 0, "armed": False, "captured": False, "window": None, "capture_step": None,
         "layer0_tokens": None, "seq_lens": None, "steps": [], "hook_thread_ok": True}


def attach(worker):
    runner = worker.model_runner
    original = runner.execute_model

    def execute_model(*args, **kwargs):
        so = args[0] if args else kwargs["scheduler_output"]
        state["step"] += 1
        n_sched = so.num_scheduled_tokens
        n_new = len(so.scheduled_new_reqs)
        cached = so.scheduled_cached_reqs
        pure_decode = (n_new == 0 and len(cached.req_ids) == BATCH and len(n_sched) == BATCH
                       and so.total_num_scheduled_tokens == BATCH and all(v == 1 for v in n_sched.values()))
        state["steps"].append((state["step"], len(n_sched), so.total_num_scheduled_tokens, n_new, pure_decode))
        if pure_decode and not state["captured"] and not state["armed"]:
            state["armed"] = True
            state["capture_step"] = state["step"]
            state["seq_lens"] = [int(n) + 1 for n in cached.num_computed_tokens]
        return original(*args, **kwargs)

    runner.execute_model = execute_model
    model = worker.get_model()
    layer0 = model.get_submodule("model.layers.0")

    def on_enter(m, args, kwargs):
        if state["armed"]:
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1]
            state["layer0_tokens"] = int(hs.shape[0])
            state["hook_thread_ok"] &= threading.get_ident() == main_thread
            first = launched_kernels() + 1
            state["window"] = [first, None]
            torch.cuda.profiler.start()

    def on_exit(m, args, kwargs, output):
        if state["armed"]:
            torch.cuda.profiler.stop()
            state["window"][1] = launched_kernels()
            state["armed"] = False
            state["captured"] = True

    layer0.register_forward_pre_hook(on_enter, with_kwargs=True)
    layer0.register_forward_hook(on_exit, with_kwargs=True)


llm.collective_rpc(attach)
params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True, detokenize=False)
t1 = time.perf_counter()
outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
t_gen = time.perf_counter() - t1
out_ids = [list(o.outputs[0].token_ids) for o in outs]
print("init_s", round(t_init, 1), "generate_s", round(t_gen, 1), "steps", state["step"])

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


with open(os.path.join(OUT, "steps.tsv"), "w") as f:
    f.write("step\tn_reqs\ttotal_tokens\tn_new_reqs\tpure_decode\n")
    for row in state["steps"]:
        f.write("\t".join(str(x) for x in row) + "\n")

check("H1-1 a step with all sequences decoding (1 token each) occurred", state["captured"], f"step {state['capture_step']}")
check("H1-2 layer 0 saw exactly batch rows in that step", state["layer0_tokens"] == BATCH, state["layer0_tokens"])
check("H1-2 hooks ran on the main thread", state["hook_thread_ok"])
check("H1-3 every request produced max_tokens", len(out_ids) == BATCH and all(len(o) == MAX_TOKENS for o in out_ids),
      f"{len(out_ids)} requests")
seq = state["seq_lens"] or []
check("H1-4 sequence lengths at the captured step recorded", len(seq) == BATCH,
      f"min {min(seq) if seq else None} max {max(seq) if seq else None}")

rows = {}
with open(stats_path()) as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\n").split(",")
        nums = [int(p) for p in parts[-10:]]
        kid = int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1))
        rows[kid] = {"name": ",".join(parts[1:-10]).strip(), "grid": nums[0:3], "block": nums[4:7],
                     "total_insts": nums[8], "reported_insts": nums[9]}
first, last = state["window"] if state["window"] else (0, -1)
window_ids = list(range(first, last + 1))
kernelslist = glob.glob(os.path.join(TRACES, "kernelslist_ctx_*"))
klines = open(kernelslist[0]).read().splitlines() if len(kernelslist) == 1 else []
memcpy = [ln for ln in klines if ln.startswith("MemcpyHtoD,")]
kl_ids = [int(re.match(r"kernel-(\d+)-ctx", ln).group(1)) for ln in klines if not ln.startswith("MemcpyHtoD,")]
trace_files = sorted(glob.glob(os.path.join(TRACES, "kernel-*.trace*")))
tf_ids = sorted(int(re.search(r"kernel-(\d+)-ctx", os.path.basename(p)).group(1)) for p in trace_files)
check("one CUDA context", len(kernelslist) == 1 and len(glob.glob(os.path.join(TRACES, "stats_ctx_*"))) == 1)
check("H1-5 kernelslist kernel ids == the layer-0 window", kl_ids == window_ids, f"{len(kl_ids)} listed, window {first}-{last}")
check("H1-5 trace file ids == the layer-0 window", tf_ids == window_ids, f"{len(tf_ids)} files")
check("H1-5 every trace file is non-empty", all(os.path.getsize(p) > 0 for p in trace_files))
check("H1-5 instructions reported only for window kernels",
      all((r["reported_insts"] > 0) == (k in set(window_ids)) for k, r in rows.items()))
check("H1-6 window holds a plausible decoder layer (10..40 kernels)", 10 <= len(window_ids) <= 40, len(window_ids))

print("window kernels:")
for k in window_ids:
    r = rows[k]
    print(f"  {k} grid={tuple(r['grid'])} block={tuple(r['block'])} insts={r['reported_insts']} {r['name'][:100]}")
warp_insts = sum(rows[k]["reported_insts"] for k in window_ids)
summary = {"model": MODEL_KEY, "model_dir": MODEL_DIR, "n_layers": N_LAYERS, "S": S, "batch": BATCH,
           "max_tokens": MAX_TOKENS, "max_batched_tokens": MAX_BATCHED, "kv_cache_bytes": KV_BYTES, "prompt_stride": stride,
           "init_s": t_init, "generate_s": t_gen, "total_steps": state["step"], "capture_step": state["capture_step"],
           "layer0_tokens": state["layer0_tokens"], "seq_lens_min": min(seq) if seq else None,
           "seq_lens_max": max(seq) if seq else None, "window": [first, last], "n_window_kernels": len(window_ids),
           "window_kernels": [{"id": k, "name": rows[k]["name"], "grid": rows[k]["grid"], "block": rows[k]["block"],
                               "warp_insts": rows[k]["reported_insts"]} for k in window_ids],
           "window_warp_insts": warp_insts, "launched_total": len(rows), "trace_files": len(trace_files),
           "trace_bytes": sum(os.path.getsize(p) for p in trace_files),
           "memcpy_lines": len(memcpy), "memcpy_bytes": sum(int(ln.split(",")[2]) for ln in memcpy),
           "output_ids_first3": [o[:3] for o in out_ids[:4]]}
with open(os.path.join(OUT, "capture_info.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("window warp insts", warp_insts, "trace bytes", summary["trace_bytes"], "memcpy", len(memcpy), summary["memcpy_bytes"])
print("H1 CAPTURE", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
