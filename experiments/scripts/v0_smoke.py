# V0: run DeepSeek-R1-Distill-Llama-8B under vLLM without the tracer (prefill S tokens + 1 decode step).
# Explained in llm_exp/docs/v0_vllm_smoke.ko.md.
import os
import sys
import time

# V0 isolates vLLM from the tracer, and needs the model in this process to read torch memory stats.
assert "CUDA_INJECTION64_PATH" not in os.environ, "V0 must run without the NVBit tracer"
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"

import torch
import vllm
from transformers import PreTrainedTokenizerFast
from vllm import LLM, SamplingParams

MODEL = "/hf_cache/DeepSeek-R1-Distill-Llama-8B"
S = int(sys.argv[1]) if len(sys.argv) > 1 else 256

print("V0 start", time.strftime("%Y-%m-%d %H:%M:%S"), "S", S)
print("versions vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)
print("env LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH"))
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))

# Exactly S prompt tokens from fixed real text: the model card shipped with the weights.
# transformers 5.x maps this checkpoint's "LlamaTokenizerFast" class to a Metaspace (SentencePiece-style)
# tokenizer, which drops spaces for this byte-level BPE vocabulary. PreTrainedTokenizerFast uses
# tokenizer.json as-is (verified identical to the raw tokenizers library). It does not add BOS, so add it.
tok = PreTrainedTokenizerFast.from_pretrained(MODEL)
bos_id = tok.convert_tokens_to_ids("<｜begin▁of▁sentence｜>")
assert bos_id == 128000
text = open(f"{MODEL}/README.md", encoding="utf-8").read()
ids = [bos_id] + tok(text, add_special_tokens=False).input_ids[: S - 1]
assert len(ids) == S
print("prompt_ids_head", ids[:8], "prompt_ids_tail", ids[-4:])
print("prompt_text_tail", repr(tok.decode(ids[-24:])))

t0 = time.perf_counter()
llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    enforce_eager=True,                   # no torch.compile, no CUDA graphs
    async_scheduling=False,               # one CPU thread drives the GPU
    attention_backend="FLASH_ATTN",       # fail loudly instead of silently falling back
    skip_tokenizer_init=True,             # vLLM's tokenizer load hits the transformers 5.x issue above
    max_model_len=S + 16,
    max_num_seqs=1,
    max_num_batched_tokens=S + 16,        # >= S so prefill runs in one step; <= max_model_len avoids a warning
    enable_prefix_caching=False,
    kv_cache_memory_bytes=512 << 20,      # fixed 512 MiB KV cache pool
)
t_init = time.perf_counter() - t0

params = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True, detokenize=False)
t1 = time.perf_counter()
out = llm.generate({"prompt_token_ids": ids}, params)[0]
t_gen = time.perf_counter() - t1

free_b, total_b = torch.cuda.mem_get_info()
print("init_s", round(t_init, 1), "generate_s", round(t_gen, 3))
out_ids = list(out.outputs[0].token_ids)
print("output_token_ids", out_ids, "text", repr(tok.decode(out_ids)))
print("max_allocated_GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2),
      "device_used_GiB", round((total_b - free_b) / 2**30, 2))
print("V0 end", time.strftime("%Y-%m-%d %H:%M:%S"))
