# Shared vLLM setup for llm_exp scripts from V1 on. Settings are identical to scripts/v0_smoke.py,
# whose reasons are documented in docs/v0_vllm_smoke.ko.md (section 2 and 7.2).
from transformers import PreTrainedTokenizerFast
from vllm import LLM

MODEL = "/hf_cache/DeepSeek-R1-Distill-Llama-8B"
BOS_TOKEN = "<｜begin▁of▁sentence｜>"
BOS_ID = 128000


def load_tokenizer():
    # transformers 5.x loads this checkpoint's "LlamaTokenizerFast" as a Metaspace tokenizer that drops
    # spaces; PreTrainedTokenizerFast uses tokenizer.json as-is.
    return PreTrainedTokenizerFast.from_pretrained(MODEL)


def build_prompt_ids(tok, s):
    """Return (prompt ids = BOS + first s-1 README tokens, all README tokens without BOS)."""
    bos_id = tok.convert_tokens_to_ids(BOS_TOKEN)
    assert bos_id == BOS_ID
    text = open(f"{MODEL}/README.md", encoding="utf-8").read()
    text_ids = tok(text, add_special_tokens=False).input_ids
    ids = [bos_id] + text_ids[: s - 1]
    assert len(ids) == s
    return ids, text_ids


def make_llm(s):
    return LLM(
        model=MODEL,
        dtype="bfloat16",
        enforce_eager=True,                   # no torch.compile, no CUDA graphs
        async_scheduling=False,               # one CPU thread drives the GPU
        attention_backend="FLASH_ATTN",       # fail loudly instead of silently falling back
        skip_tokenizer_init=True,             # avoid the transformers 5.x tokenizer issue inside vLLM
        max_model_len=s + 16,
        max_num_seqs=1,
        max_num_batched_tokens=s + 16,        # >= s so prefill runs in one step
        enable_prefix_caching=False,
        kv_cache_memory_bytes=512 << 20,      # fixed 512 MiB KV cache pool
    )


def build_batch_prompts(tok, s, batch, stride):
    """Return `batch` prompts of s tokens: BOS + README tokens [stride * i, stride * i + s - 1) for request i.
    The README has fewer than batch * (s - 1) tokens for step 7, so the windows overlap but stay distinct."""
    bos_id = tok.convert_tokens_to_ids(BOS_TOKEN)
    assert bos_id == BOS_ID
    text = open(f"{MODEL}/README.md", encoding="utf-8").read()
    text_ids = tok(text, add_special_tokens=False).input_ids
    assert stride * (batch - 1) + s - 1 <= len(text_ids), "README too short for these windows"
    prompts = [[bos_id] + text_ids[stride * i: stride * i + s - 1] for i in range(batch)]
    assert all(len(p) == s for p in prompts) and len({tuple(p) for p in prompts}) == batch
    return prompts


def make_llm_batch(s, batch, kv_cache_bytes):
    # Same settings as make_llm() except the batch-related limits (docs/s7_b32_prefill_layer.ko.md section 1).
    return LLM(
        model=MODEL,
        dtype="bfloat16",
        enforce_eager=True,
        async_scheduling=False,
        attention_backend="FLASH_ATTN",
        skip_tokenizer_init=True,
        max_model_len=s + 16,
        max_num_seqs=batch,
        max_num_batched_tokens=batch * (s + 16),   # >= batch * s so the whole batch prefills in one step
        enable_prefix_caching=False,
        kv_cache_memory_bytes=kv_cache_bytes,
    )


# Model-generic helpers (Q0, Qwen 2.5 7B). The DeepSeek functions above stay unchanged.
PROMPT_TEXT = f"{MODEL}/README.md"   # the same English text for every model (DeepSeek README)


def load_tokenizer_from(model_dir):
    return PreTrainedTokenizerFast.from_pretrained(model_dir)


def build_batch_prompts_from(tok, s, batch, stride, bos_id=None):
    """`batch` prompts of s tokens cut from PROMPT_TEXT at token offsets stride * i; BOS first when bos_id is given."""
    text_ids = tok(open(PROMPT_TEXT, encoding="utf-8").read(), add_special_tokens=False).input_ids
    body = s - (1 if bos_id is not None else 0)
    assert stride * (batch - 1) + body <= len(text_ids), "prompt text too short for these windows"
    head = [bos_id] if bos_id is not None else []
    prompts = [head + text_ids[stride * i: stride * i + body] for i in range(batch)]
    assert all(len(p) == s for p in prompts) and len({tuple(p) for p in prompts}) == batch
    return prompts


def make_llm_from(model_dir, s, batch, kv_cache_bytes):
    # Same engine settings as make_llm_batch(); only the model directory differs.
    return LLM(
        model=model_dir,
        dtype="bfloat16",
        enforce_eager=True,
        async_scheduling=False,
        attention_backend="FLASH_ATTN",
        skip_tokenizer_init=True,
        max_model_len=s + 16,
        max_num_seqs=batch,
        max_num_batched_tokens=batch * (s + 16),
        enable_prefix_caching=False,
        kv_cache_memory_bytes=kv_cache_bytes,
    )


# Hypothesis-1 decode batch sweep (docs/h1_llama70b_decode_batch_sweep.ko.md). The functions above stay unchanged.
def build_prompts_adaptive(tok, s, batch, bos_id=None, max_stride=150):
    """`batch` distinct prompts of s tokens from PROMPT_TEXT. The window stride is max_stride when the text is long
    enough, else the largest stride that still fits `batch` distinct windows (text has 4,948 tokens: batch 1024 ->
    stride 4). Returns (prompts, stride)."""
    text_ids = tok(open(PROMPT_TEXT, encoding="utf-8").read(), add_special_tokens=False).input_ids
    body = s - (1 if bos_id is not None else 0)
    stride = max_stride if batch == 1 else min(max_stride, (len(text_ids) - body) // (batch - 1))
    assert stride >= 1, "prompt text too short for this many distinct windows"
    head = [bos_id] if bos_id is not None else []
    prompts = [head + text_ids[stride * i: stride * i + body] for i in range(batch)]
    assert all(len(p) == s for p in prompts) and len({tuple(p) for p in prompts}) == batch
    return prompts, stride


def make_llm_decode_sweep(model_dir, s, batch, kv_cache_bytes, max_tokens, max_batched_tokens):
    """Same engine settings as make_llm_from() except the limits that a large batch needs: max_model_len leaves room
    for max_tokens outputs, max_num_batched_tokens is a fixed prefill budget (chunked prefill splits large batches
    over several steps instead of allocating batch*S activations at once), max_num_seqs = batch."""
    return LLM(
        model=model_dir,
        dtype="bfloat16",
        enforce_eager=True,
        async_scheduling=False,
        attention_backend="FLASH_ATTN",
        skip_tokenizer_init=True,
        max_model_len=s + max_tokens + 16,
        max_num_seqs=batch,
        max_num_batched_tokens=max_batched_tokens,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=kv_cache_bytes,
    )
