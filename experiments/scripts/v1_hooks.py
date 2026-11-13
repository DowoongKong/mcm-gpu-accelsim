# V1: confirm the trace-window hook targets and their call pattern under vLLM, without the tracer.
# Explained in llm_exp/docs/v1_module_hooks.ko.md.
import json
import os
import sys
import threading
import time

assert "CUDA_INJECTION64_PATH" not in os.environ, "V1 must run without the NVBit tracer"
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
# In-process collective_rpc calls the function directly; V1 checks that no pickling switch is needed.
assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in os.environ, "V1 checks the path without this variable"

from vllm import SamplingParams

from vllm_setup import build_prompt_ids, load_tokenizer, make_llm

S = int(sys.argv[1]) if len(sys.argv) > 1 else 256
LOG_DIR = sys.argv[2] if len(sys.argv) > 2 else "logs"
LAYERS = (0, 15, 31)
SUBLAYERS = ("self_attn", "mlp")
TARGETS = [f"model.layers.{i}.{sub}" for i in LAYERS for sub in SUBLAYERS]
# Weight shapes implied by config.json (hidden 4096, FFN 14336, 32 heads / 8 KV heads, head dim 128, no bias).
EXPECTED_PARAMS = {
    "self_attn": {"qkv_proj.weight": (6144, 4096), "o_proj.weight": (4096, 4096)},
    "mlp": {"gate_up_proj.weight": (28672, 4096), "down_proj.weight": (4096, 14336)},
}
V0_OUTPUT_S256 = [7533, 9612]

print("V1 start", time.strftime("%Y-%m-%d %H:%M:%S"), "S", S)
tok = load_tokenizer()
ids, text_ids = build_prompt_ids(tok, S)
llm = make_llm(S)

main_thread = threading.get_ident()
events = []  # dicts in call order
calls = {}


def hidden_tokens(args, kwargs):
    # LlamaDecoderLayer calls self_attn with keyword arguments and mlp positionally (llama.py:328, :332).
    hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
    return int(hs.shape[0])


def inspect_and_attach(worker):
    model = worker.get_model()
    names = [(name, type(mod).__name__) for name, mod in model.named_modules()]
    with open(os.path.join(LOG_DIR, f"v1_named_modules_S{S}.txt"), "w") as f:
        for name, cls in names:
            f.write(f"{name}\t{cls}\n")
    info = {
        "model_class": type(model).__name__,
        "runner_model_is_get_model": worker.model_runner.model is model,
        "num_named_modules": len(names),
        "targets": {},
    }
    for tag in TARGETS:
        mod = model.get_submodule(tag)
        info["targets"][tag] = {
            "class": type(mod).__name__,
            "params": {pn: list(p.shape) for pn, p in mod.named_parameters()},
        }

        def on_enter(m, args, kwargs, tag=tag):
            calls[tag] = calls.get(tag, 0) + 1
            events.append({"tag": tag, "kind": "enter", "call": calls[tag],
                           "tokens": hidden_tokens(args, kwargs), "thread": threading.get_ident()})

        def on_exit(m, args, kwargs, output, tag=tag):
            events.append({"tag": tag, "kind": "exit", "call": calls[tag],
                           "tokens": int(output.shape[0]), "thread": threading.get_ident()})

        mod.register_forward_pre_hook(on_enter, with_kwargs=True)
        mod.register_forward_hook(on_exit, with_kwargs=True)
    return info


info = llm.collective_rpc(inspect_and_attach)[0]
print("module_info", json.dumps(info))
assert not events, "hooks fired before generate (should be registered after warm-up)"

params = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True, detokenize=False)
out = llm.generate({"prompt_token_ids": ids}, params)[0]
out_ids = list(out.outputs[0].token_ids)
print("output_token_ids", out_ids, "text", repr(tok.decode(out_ids)))
print("readme_next_ids", text_ids[S - 1:S + 1], "text", repr(tok.decode(text_ids[S - 1:S + 1])))

with open(os.path.join(LOG_DIR, f"v1_hook_events_S{S}.tsv"), "w") as f:
    f.write("seq\ttag\tkind\tcall\ttokens\tmain_thread\n")
    for i, e in enumerate(events):
        f.write(f"{i}\t{e['tag']}\t{e['kind']}\t{e['call']}\t{e['tokens']}\t{e['thread'] == main_thread}\n")
for i, e in enumerate(events):
    print("event", i, e["tag"], e["kind"], "call", e["call"], "tokens", e["tokens"])

# Checks.
results = []


def check(name, ok, detail):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


check("model class", info["model_class"] == "LlamaForCausalLM", info["model_class"])
check("runner.model is get_model()", info["runner_model_is_get_model"], info["runner_model_is_get_model"])
for tag in TARGETS:
    sub = tag.rsplit(".", 1)[1]
    t = info["targets"][tag]
    want_cls = "LlamaAttention" if sub == "self_attn" else "LlamaMLP"
    want_params = {k: list(v) for k, v in EXPECTED_PARAMS[sub].items()}
    check(f"{tag} class", t["class"] == want_cls, t["class"])
    check(f"{tag} weights", t["params"] == want_params, t["params"])
expected_order = [(tag, kind) for step in (1, 2) for tag in TARGETS for kind in ("enter", "exit")]
actual_order = [(e["tag"], e["kind"]) for e in events]
check("event count and order", actual_order == expected_order, f"{len(events)} events, expected {len(expected_order)}")
check("calls per target == 2", all(calls.get(tag) == 2 for tag in TARGETS), calls)
check("prefill call tokens == S", all(e["tokens"] == S for e in events if e["call"] == 1), f"S={S}")
check("decode call tokens == 1", all(e["tokens"] == 1 for e in events if e["call"] == 2), "")
check("all hooks on main thread", all(e["thread"] == main_thread for e in events), "")
if S == 256:
    check("output equals V0", out_ids == V0_OUTPUT_S256, out_ids)
print("V1 RESULT", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
print("V1 end", time.strftime("%Y-%m-%d %H:%M:%S"))
