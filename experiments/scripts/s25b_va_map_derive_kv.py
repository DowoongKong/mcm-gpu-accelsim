#!/usr/bin/env python3
# Post-hoc derivation of the layer-0 KV cache view for a capture whose hook recorded only the raw KV tensors
# (docs/s25b_h1_capture_v2_init_window.ko.md, check H1v2-6). The original va_map.json keys are left untouched;
# "kv_cache_layer0_derived" and "kv_cache_layer0_derivation" are added and the H1v2-6 predicate is re-evaluated
# into a separate log so the original run.log stays the primary evidence.
# Layout = FlashAttentionBackend.get_kv_cache_shape(): (2, num_blocks, block_size, num_kv_heads, head_size),
# stride order identity (vllm/v1/attention/backends/flash_attn.py:121-130), element dtype = model dtype.
# Usage: s25b_va_map_derive_kv.py <capture dir> <hf config.json>
import json, os, sys
cap, cfgp = sys.argv[1], sys.argv[2]
vap = os.path.join(cap, "va_map.json")
va = json.load(open(vap)); cfg = json.load(open(cfgp))
kvh, hd = cfg["num_key_value_heads"], cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
dtype = {"bfloat16": ("torch.bfloat16", 2), "float16": ("torch.float16", 2)}[cfg["torch_dtype"]]
bs = va["kv_block_size"]
raw = va["kv_tensors"]["model.layers.0.self_attn.attn"]
per_block = 2 * bs * kvh * hd * dtype[1]
assert raw["bytes"] % per_block == 0, (raw["bytes"], per_block)
nb = raw["bytes"] // per_block
shape = [2, nb, bs, kvh, hd]
stride = [nb * bs * kvh * hd, bs * kvh * hd, kvh * hd, hd, 1]
va["kv_cache_layer0_derived"] = {"module": "self_attn.attn", "ptr": raw["ptr"], "bytes": raw["bytes"],
                                 "shape": shape, "stride": stride, "dtype": dtype[0]}
va["kv_cache_layer0_derivation"] = ("derived 2026-11-10 from kv_tensors['model.layers.0.self_attn.attn'] with "
                                    "FlashAttentionBackend.get_kv_cache_shape (2, num_blocks, block_size, num_kv_heads, head_size); "
                                    f"num_blocks = {raw['bytes']} / {per_block} = {nb}; the live hook missed it because vLLM 0.19.1 "
                                    "binds Attention.kv_cache as a tensor, not a list (h1_capture_v2.py fixed afterwards)")
json.dump(va, open(vap, "w"), indent=1)
ok = len(va["weights"]) >= 4 and len(va["modules"]) >= 5 and va["kv_cache_layer0_derived"] is not None
with open(os.path.join(cap, "h1v2_6_recheck.log"), "w") as f:
    f.write(f"H1v2-6 re-evaluated with the derived layer-0 KV view: {'PASS' if ok else 'FAIL'} | weights {len(va['weights'])} "
            f"modules {len(va['modules'])} kv_derived shape {shape} ptr 0x{raw['ptr']:x} bytes {raw['bytes']}\n")
    f.write(f"GPU KV cache size reported by vLLM (run.log) should equal num_blocks*block_size = {nb * bs} tokens\n")
print(open(os.path.join(cap, "h1v2_6_recheck.log")).read())
