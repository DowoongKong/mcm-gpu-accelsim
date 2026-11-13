# Q0 smoke for Qwen 2.5 7B (docs/q0_qwen25_7b_prep.ko.md, Q0-5 / Q0-6). One vLLM run per invocation.
# Without the tracer: checks that the model loads, generates, and has the expected decoder-layer modules.
# With the tracer attached (ACTIVE_FROM_START=0, no window ever opened): additionally records the kernel-id range of
# every step and of every decoder layer per step (from the stats file), i.e. the Qwen kernel map.
# Usage: q0_smoke.py <batch> <S> <max_tokens> <log dir>
import glob
import json
import os
import re
import sys
import threading
import time

BATCH, S, MAX_TOKENS, LOG_DIR = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
MODEL_DIR = "/hf_cache/Qwen2.5-7B"
N_LAYERS = 28
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
TRACER = "CUDA_INJECTION64_PATH" in os.environ
if TRACER:
    assert os.environ.get("ACTIVE_FROM_START") == "0", "the kernel-map run must keep every window closed"
    TRACES = os.path.join(os.environ["TRACES_FOLDER"], "traces")

from vllm import SamplingParams

from vllm_setup import build_batch_prompts_from, load_tokenizer_from, make_llm_from


def launched():
    if not TRACER:
        return -1
    paths = glob.glob(os.path.join(TRACES, "stats_ctx_*"))
    assert len(paths) == 1, paths
    with open(paths[0]) as f:
        return sum(1 for _ in f) - 1


tag = f"b{BATCH}_s{S}_o{MAX_TOKENS}{'_tracer' if TRACER else ''}"
print("Q0 start", time.strftime("%Y-%m-%d %H:%M:%S"), tag)
tok = load_tokenizer_from(MODEL_DIR)
prompts = build_batch_prompts_from(tok, S, BATCH, 150, bos_id=None)   # Qwen2.5 adds no BOS (tokenizer_config)
t0 = time.perf_counter()
llm = make_llm_from(MODEL_DIR, S, BATCH, 1536 << 20)
t_init = time.perf_counter() - t0
main_thread = threading.get_ident()
events = []
calls = {}
layer_names = []


def attach(worker):
    runner = worker.model_runner
    for fname in ("execute_model", "sample_tokens"):
        calls[fname] = 0
        original = getattr(runner, fname)

        def wrapped(*args, _f=fname, _o=original, **kwargs):
            calls[_f] += 1
            events.append((_f, "enter", calls[_f], launched(), None, threading.get_ident()))
            r = _o(*args, **kwargs)
            events.append((_f, "exit", calls[_f], launched(), None, threading.get_ident()))
            return r

        setattr(runner, fname, wrapped)
    model = worker.get_model()
    for name, _ in model.named_modules():
        if re.fullmatch(r"model\.layers\.\d+", name):
            layer_names.append(name)
    for idx in range(N_LAYERS):
        mod = model.get_submodule(f"model.layers.{idx}")
        key = f"layer{idx}"
        calls[key] = 0

        def pre(m, args, kwargs, _k=key):
            calls[_k] += 1
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1]
            events.append((_k, "enter", calls[_k], launched(), int(hs.shape[0]), threading.get_ident()))

        def post(m, args, kwargs, out, _k=key):
            events.append((_k, "exit", calls[_k], launched(), None, threading.get_ident()))

        mod.register_forward_pre_hook(pre, with_kwargs=True)
        mod.register_forward_hook(post, with_kwargs=True)
    sub = model.get_submodule("model.layers.0")
    return [n for n, _ in sub.named_modules()]


sub_modules = llm.collective_rpc(attach)[0]
params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True, detokenize=False)
t1 = time.perf_counter()
outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
t_gen = time.perf_counter() - t1
out_ids = [list(o.outputs[0].token_ids) for o in outs]
print("init_s", round(t_init, 1), "generate_s", round(t_gen, 3))
print("layer 0 submodules:", sub_modules)

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


ev = {(n, k, c): (l, t) for n, k, c, l, t, _ in events}
check("28 decoder layers named model.layers.N", layer_names == [f"model.layers.{i}" for i in range(N_LAYERS)], len(layer_names))
check(f"{MAX_TOKENS} steps (execute_model / sample_tokens)", (calls["execute_model"], calls["sample_tokens"]) == (MAX_TOKENS, MAX_TOKENS),
      (calls["execute_model"], calls["sample_tokens"]))
toks = [ev[("layer0", "enter", c)][1] for c in range(1, calls["layer0"] + 1)]
check("layer 0 tokens per step = batch*S, then batch", toks == [BATCH * S] + [BATCH] * (MAX_TOKENS - 1), toks)
check("every layer called once per step", all(calls[f"layer{i}"] == MAX_TOKENS for i in range(N_LAYERS)))
check("hooks on the main thread", all(t == main_thread for *_, t in events))
check(f"{BATCH} outputs of {MAX_TOKENS} tokens", len(out_ids) == BATCH and all(len(o) == MAX_TOKENS for o in out_ids))
print("first output ids:", out_ids[0], repr(tok.decode(out_ids[0])))

summary = {"tag": tag, "batch": BATCH, "S": S, "max_tokens": MAX_TOKENS, "init_s": t_init, "generate_s": t_gen,
           "out_ids": out_ids, "layer0_submodules": sub_modules}
if TRACER:
    steps, layers = {}, {}
    for c in range(1, MAX_TOKENS + 1):
        steps[c] = (ev[("execute_model", "enter", c)][0] + 1, ev[("sample_tokens", "exit", c)][0])
        layers[c] = [(ev[(f"layer{i}", "enter", c)][0] + 1, ev[(f"layer{i}", "exit", c)][0]) for i in range(N_LAYERS)]
    counts = {c: sorted({b - a + 1 for a, b in layers[c]}) for c in layers}
    check("tracer: no trace file written (every window closed)", not glob.glob(os.path.join(TRACES, "kernel-*.trace*")))
    summary.update({"steps": steps, "layers": layers, "kernels_per_layer": counts, "launched_total": launched()})
    for c in steps:
        a, b = steps[c]
        l0, lN = layers[c][0], layers[c][-1]
        print(f"step {c}: kernels {a}-{b} ({b - a + 1}); layer 0 {l0}, layer 27 {lN}; kernels per layer {counts[c]}; "
              f"prep {l0[0] - a}, tail {b - lN[1]}")
with open(os.path.join(LOG_DIR, f"q0_summary_{tag}.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("Q0 SMOKE", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
