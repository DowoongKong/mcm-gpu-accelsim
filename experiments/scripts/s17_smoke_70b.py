# Step 17 smoke (docs/s17_llama70b_prep.ko.md section 4, G17-3): load the truncated Llama-3.3-70B checkpoint in
# vLLM without the tracer, generate max_tokens tokens for a batch, and report decoder-layer count, GPU memory and
# timing. Same engine settings as the capture (vllm_setup.make_llm_from).
# Usage (inside llmtrace, VLLM_ENABLE_V1_MULTIPROCESSING=0): s17_smoke_70b.py <model dir> <batch> <S> <max_tokens> <expected layers> <bos id>
import os
import re
import sys
import time

MODEL_DIR, BATCH, S, MAX_TOKENS, N_LAYERS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
BOS_ID = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "none" else None
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
assert "CUDA_INJECTION64_PATH" not in os.environ, "smoke runs without the tracer"

import torch
from vllm import SamplingParams

from vllm_setup import build_batch_prompts_from, load_tokenizer_from, make_llm_from

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", label, "|", detail)


print("S17 smoke start", time.strftime("%Y-%m-%d %H:%M:%S"), MODEL_DIR, "batch", BATCH, "S", S, "max_tokens", MAX_TOKENS)
tok = load_tokenizer_from(MODEL_DIR)
prompts = build_batch_prompts_from(tok, S, BATCH, 150, bos_id=BOS_ID)
t0 = time.perf_counter()
llm = make_llm_from(MODEL_DIR, S, BATCH, 1536 << 20)
t_init = time.perf_counter() - t0
layer_names = []
tokens_seen = {}


def attach(worker):
    model = worker.get_model()
    for name, _ in model.named_modules():
        if re.fullmatch(r"model\.layers\.\d+", name):
            layer_names.append(name)
    first = model.get_submodule("model.layers.0")

    def pre(m, args, kwargs):
        hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1]
        tokens_seen[len(tokens_seen) + 1] = int(hs.shape[0])

    first.register_forward_pre_hook(pre, with_kwargs=True)
    return (len(layer_names), torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30)


info = llm.collective_rpc(attach)[0]
params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True, detokenize=False)
t1 = time.perf_counter()
outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
t_gen = time.perf_counter() - t1
out_ids = [list(o.outputs[0].token_ids) for o in outs]
mem = llm.collective_rpc(lambda w: (torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30))[0]

print("init_s", round(t_init, 1), "generate_s", round(t_gen, 3))
print("decoder layers:", info[0], "| GPU allocated after load GiB:", round(info[1], 2), "| peak GiB:", round(mem[1], 2))
print("layer-0 tokens per step:", tokens_seen)
print("output ids:", out_ids)
check("G17-3 decoder layer count", info[0] == N_LAYERS, f"{info[0]} vs expected {N_LAYERS}")
check("G17-3 layer 0 saw batch*S then batch per decode step",
      [tokens_seen.get(i) for i in range(1, MAX_TOKENS + 1)] == [BATCH * S] + [BATCH] * (MAX_TOKENS - 1), tokens_seen)
check("G17-3 every request produced max_tokens", len(out_ids) == BATCH and all(len(o) == MAX_TOKENS for o in out_ids))
check("G17-3 peak GPU memory below 20 GiB", mem[1] < 20.0, f"{mem[1]:.2f} GiB")
print("S17 SMOKE", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
