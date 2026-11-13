# Model-generic version of scripts/s8_steps_capture.py (step 8 pieces for any decoder-only model served by vLLM
# with model.layers.N modules called as layer(positions, hidden_states, residual), e.g. Llama and Qwen2).
# For every step trace
#   window A: execute_model entry -> end of model.layers.1 (input preparation + layer 0 + layer 1)
#   window B: end of the last decoder layer -> sample_tokens exit (final norm, lm_head, sampling)
# and record the kernel-id range of every piece (prep, layer 0, layer 1, tail) per step in regions.tsv.
# The weight-copy check is done afterwards by scripts/check_weight_copies.py (model-generic).
# Explained in docs/s8_b32s256_step_pieces.ko.md (method) and docs/q0_qwen25_7b_prep.ko.md (Qwen).
# Usage (inside llmtrace, tracer attached, ACTIVE_FROM_START=0):
#   steps_capture.py <log dir> <model: deepseek|qwen> <batch> <S> <max_tokens> [<traced layers: 1|2> [<kernel map json>]]
# traced layers 1 closes window A after layer 0 (no layer-1 piece); the optional kernel map (from scripts/q0_smoke.py
# with the tracer attached) gives the expected kernel counts of prep / layer / tail per step.
import glob
import json
import os
import re
import sys
import threading
import time

LOG_DIR, MODEL_KEY = sys.argv[1], sys.argv[2]
BATCH, S, MAX_TOKENS = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
TRACED_LAYERS = int(sys.argv[6]) if len(sys.argv) > 6 else 2
KMAP = json.load(open(sys.argv[7])) if len(sys.argv) > 7 else None
assert TRACED_LAYERS in (1, 2)
STRIDE = 150
KV_BYTES = 1536 << 20
MODELS = {"deepseek": ("/hf_cache/DeepSeek-R1-Distill-Llama-8B", 32, 128000),   # dir, decoder layers, BOS id
          "qwen": ("/hf_cache/Qwen2.5-7B", 28, None),                          # Qwen2.5 adds no BOS
          # Step 17: Llama-3.3-70B-Instruct truncated to decoder layers 0..2 (docs/s17_llama70b_prep.ko.md 2.3);
          # the windows only need layers 0, 1 and the last one, and the kernels of a layer do not depend on the count.
          "llama70b": ("/hf_cache/Llama-3.3-70B-Instruct-L3", 3, 128000),
          # H2: Mixtral-8x7B-v0.1 truncated to decoder layers 0..2 (docs/h2_mixtral8x7b_prep.ko.md); BOS <s> = 1
          "mixtral": ("/hf_cache/Mixtral-8x7B-v0.1-L3", 3, 1)}
MODEL_DIR, N_LAYERS, BOS_ID = MODELS[MODEL_KEY]
LAST = N_LAYERS - 1
STEPS = ["prefill"] + [f"decode{i}" for i in range(1, MAX_TOKENS)]
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
assert "CUDA_INJECTION64_PATH" in os.environ, "step 8 needs the tracer attached"
assert os.environ.get("ACTIVE_FROM_START") == "0", "step 8 requires ACTIVE_FROM_START=0"
OUT = os.environ["TRACES_FOLDER"]
TRACES = os.path.join(OUT, "traces")
assert not os.path.exists(TRACES) or not os.listdir(TRACES), f"{TRACES} must start empty"

import torch
from vllm import SamplingParams

from vllm_setup import build_batch_prompts_from, load_tokenizer_from, make_llm_from


def stats_path():
    paths = glob.glob(os.path.join(TRACES, "stats_ctx_*"))
    assert len(paths) == 1, f"expected exactly one CUDA context stats file, found {paths}"
    return paths[0]


def launched_kernels():
    with open(stats_path()) as f:
        return sum(1 for _ in f) - 1


print("STEPS start", time.strftime("%Y-%m-%d %H:%M:%S"), MODEL_KEY, "S", S, "batch", BATCH, "max_tokens", MAX_TOKENS)
tok = load_tokenizer_from(MODEL_DIR)
prompts = build_batch_prompts_from(tok, S, BATCH, STRIDE, bos_id=BOS_ID)
t0 = time.perf_counter()
llm = make_llm_from(MODEL_DIR, S, BATCH, KV_BYTES)
t_init = time.perf_counter() - t0

main_thread = threading.get_ident()
ev = {}          # (name, kind, call) -> launched kernels at that moment
tokens = {}      # layer-0 call -> tokens
calls = {}


def record(name, kind):
    call = calls[name]
    ev[(name, kind, call)] = launched_kernels()
    assert threading.get_ident() == main_thread, "hook ran off the main thread"
    return call


def attach(worker):
    runner = worker.model_runner
    for fname in ("execute_model", "sample_tokens"):
        calls[fname] = 0
        original = getattr(runner, fname)

        def wrapped(*args, _fname=fname, _original=original, **kwargs):
            calls[_fname] += 1
            if _fname == "execute_model":
                torch.cuda.profiler.start()          # window A opens (input preparation onward)
            record(_fname, "enter")
            result = _original(*args, **kwargs)
            call = record(_fname, "exit")
            if _fname == "sample_tokens":
                torch.cuda.profiler.stop()           # window B closes
            return result

        setattr(runner, fname, wrapped)
    model = worker.get_model()
    for idx in ((0, 1, LAST) if TRACED_LAYERS == 2 else (0, LAST)):
        name = f"layer{idx}"
        calls[name] = 0
        mod = model.get_submodule(f"model.layers.{idx}")

        def on_enter(m, args, kwargs, _name=name):
            calls[_name] += 1
            if _name == "layer0":
                hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1]
                tokens[calls[_name]] = int(hs.shape[0])
            record(_name, "enter")

        def on_exit(m, args, kwargs, output, _name=name):
            record(_name, "exit")
            if _name == f"layer{TRACED_LAYERS - 1}":
                torch.cuda.profiler.stop()           # window A closes after the last traced layer (0 or 1)
            elif _name == f"layer{LAST}":
                torch.cuda.profiler.start()          # window B opens (final norm, lm_head, sampling)

        mod.register_forward_pre_hook(on_enter, with_kwargs=True)
        mod.register_forward_hook(on_exit, with_kwargs=True)


llm.collective_rpc(attach)
params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True, detokenize=False)
t1 = time.perf_counter()
outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
t_gen = time.perf_counter() - t1
out_ids = [list(o.outputs[0].token_ids) for o in outs]
print("init_s", round(t_init, 1), "generate_s", round(t_gen, 3))

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


n_exec = max((k[2] for k in ev if k[0] == "execute_model"), default=0)
n_samp = max((k[2] for k in ev if k[0] == "sample_tokens"), default=0)
check(f"G8-1 execute_model and sample_tokens called {MAX_TOKENS} times each", (n_exec, n_samp) == (MAX_TOKENS, MAX_TOKENS),
      (n_exec, n_samp))
want = [BATCH * S] + [BATCH] * (MAX_TOKENS - 1)
check("G8-1 layer 0 tokens: batch*S at prefill, batch per decode step", [tokens.get(i) for i in range(1, MAX_TOKENS + 1)] == want,
      [tokens.get(i) for i in range(1, MAX_TOKENS + 1)])
check("G8-2 32 requests, 3 output tokens each", len(out_ids) == BATCH and all(len(o) == MAX_TOKENS for o in out_ids))

pieces = {}      # (step, piece) -> (first, last)
windows = []
for k, step in enumerate(STEPS, start=1):
    e_in, l0_in, l0_out = ev[("execute_model", "enter", k)], ev[("layer0", "enter", k)], ev[("layer0", "exit", k)]
    a_end = l0_out
    if TRACED_LAYERS == 2:
        l1_in, l1_out = ev[("layer1", "enter", k)], ev[("layer1", "exit", k)]
        a_end = l1_out
    l31_out, s_out = ev[(f"layer{LAST}", "exit", k)], ev[("sample_tokens", "exit", k)]
    pieces[(step, "prep")] = (e_in + 1, l0_in)
    pieces[(step, "layer0")] = (l0_in + 1, l0_out)
    if TRACED_LAYERS == 2:
        pieces[(step, "layer1")] = (l1_in + 1, l1_out)
        check(f"{step}: no kernel between layer 0 end and layer 1 start", l1_in == l0_out, (l0_out, l1_in))
    pieces[(step, "tail")] = (l31_out + 1, s_out)
    windows += [(e_in + 1, a_end), (l31_out + 1, s_out)]
with open(os.path.join(OUT, "regions.tsv"), "w") as f:
    f.write("step\tpiece\tfirst_kernel_id\tlast_kernel_id\tn_kernels\n")
    for (step, piece), (a, b) in pieces.items():
        f.write(f"{step}\t{piece}\t{a}\t{b}\t{b - a + 1}\n")
traced_ids = sorted(i for a, b in windows for i in range(a, b + 1))

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

check("one CUDA context", len(kernelslist) == 1 and len(glob.glob(os.path.join(TRACES, "stats_ctx_*"))) == 1)
check("G8-3 kernelslist kernel ids == union of the 6 windows", kl_ids == traced_ids, f"{len(kl_ids)} listed vs {len(traced_ids)}")
check("G8-3 trace file ids == union of the 6 windows", tf_ids == traced_ids, f"{len(tf_ids)} files")
check("G8-3 every trace file is non-empty", all(os.path.getsize(p) > 0 for p in trace_files))
check("G8-3 instructions reported only for window kernels",
      all((r["reported_insts"] > 0) == (k in set(traced_ids)) for k, r in rows.items()))


def names(step, piece):
    a, b = pieces[(step, piece)]
    return [rows[i]["name"] for i in range(a, b + 1)]


PIECE_KINDS = ("prep", "layer0", "layer1", "tail") if TRACED_LAYERS == 2 else ("prep", "layer0", "tail")
if TRACED_LAYERS == 2:
    for step in STEPS:
        n0, n1 = names(step, "layer0"), names(step, "layer1")
        check(f"G8-4 {step}: layer 0 and layer 1 have the same kernel sequence (first norm may differ)",
              len(n0) == len(n1) and n0[1:] == n1[1:], (len(n0), len(n1)))
check("G8-4 every decode step has the same kernel sequences per piece",
      all(names(STEPS[1], p) == names(st, p) for st in STEPS[2:] for p in PIECE_KINDS))
if KMAP is not None:
    # kernel-map run (different process): same number of kernels per piece
    got, want = [], []
    for k, step in enumerate(STEPS, start=1):
        a, b = KMAP["steps"][str(k)]
        la, lb = KMAP["layers"][str(k)][0]
        want += [la - a, lb - la + 1, b - KMAP["layers"][str(k)][-1][1]]
        got += [len(names(step, "prep")), len(names(step, "layer0")), len(names(step, "tail"))]
    check("G8-4 kernels per piece (prep, layer 0, tail per step) == kernel map", got == want, (got, want))

print("pieces:")
for (step, piece), (a, b) in pieces.items():
    insts = sum(rows[i]["reported_insts"] for i in range(a, b + 1))
    print(f"  {step:8s} {piece:7s} {a}-{b} ({b - a + 1} kernels) reported_warp={insts}")
summary = {"model": MODEL_KEY, "model_dir": MODEL_DIR, "n_layers": N_LAYERS, "S": S, "batch": BATCH, "stride": STRIDE, "max_tokens": MAX_TOKENS, "init_s": t_init, "generate_s": t_gen,
           "output_ids": out_ids, "layer0_tokens": tokens, "pieces": {f"{s}/{p}": v for (s, p), v in pieces.items()},
           "windows": windows, "launched_total": len(rows), "trace_files": len(trace_files),
           "trace_bytes": sum(os.path.getsize(p) for p in trace_files),
           "memcpy_lines": len(memcpy), "memcpy_bytes": sum(int(ln.split(",")[2]) for ln in memcpy)}
with open(os.path.join(LOG_DIR, f"steps_summary_{MODEL_KEY}_b{BATCH}_s{S}_o{MAX_TOKENS}.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("trace bytes", summary["trace_bytes"])
print("STEPS CAPTURE", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
