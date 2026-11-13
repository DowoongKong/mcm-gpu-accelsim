# Build a truncated Hugging Face checkpoint that keeps only the first N decoder layers of a Llama-style model
# (docs/s17_llama70b_prep.ko.md section 2.3, docs/s17b_hf_model_layout_and_vllm_execution.ko.md section 6).
# Why: vLLM reads every tensor of every safetensors file listed in model.safetensors.index.json and fails on a
# tensor name that the (num_hidden_layers = N) model does not have, so the kept tensors must live in files of
# their own. The copy is done at the byte level (safetensors = 8-byte header length + JSON header + raw buffer),
# so no torch is needed and the bytes are preserved exactly.
# Usage: s17_make_truncated_ckpt.py <source dir with the needed shards> <output dir> --layers N
#   source dir: config.json, tokenizer files, model.safetensors.index.json and the shards that hold the kept tensors
#   output dir: same small files, config.json with num_hidden_layers = N, model-0000k-of-0000m.safetensors + index
# Prints one CHECK line per verification and "TRUNCATE PASS/FAIL (k/n)" at the end.
import json
import os
import re
import shutil
import struct
import sys

SMALL_FILES = ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "chat_template.jinja", "tokenizer.model"]   # tokenizer.model: SentencePiece models (Mixtral), copied when present
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
ALIGN = 8
CHUNK = 64 << 20

args = sys.argv[1:]
n_layers = int(args[args.index("--layers") + 1])
del args[args.index("--layers"):args.index("--layers") + 2]
src, dst = args


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return n, header


def keep(name):
    m = LAYER_RE.match(name)
    return int(m.group(1)) < n_layers if m else True


results = []


def check(label, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", label, "|", detail)


index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
weight_map = index["weight_map"]
kept = sorted(n for n in weight_map if keep(n))
needed_shards = sorted({weight_map[n] for n in kept})
missing = [s for s in needed_shards if not os.path.isfile(os.path.join(src, s))]
check("every shard that holds a kept tensor is present", not missing, f"need {needed_shards}, missing {missing}")
if missing:
    sys.exit(1)

# Source tensor descriptors: name -> (shard, dtype, shape, begin, end, header_len)
desc = {}
for shard in needed_shards:
    hlen, header = read_header(os.path.join(src, shard))
    for name, meta in header.items():
        if name == "__metadata__" or name not in weight_map or not keep(name):
            continue
        b, e = meta["data_offsets"]
        desc[name] = (shard, meta["dtype"], meta["shape"], b, e, hlen)
check("kept tensors found in shard headers", set(desc) == set(kept), f"{len(desc)} of {len(kept)}")

# Group: file 1 = embed + layers, file 2 = final norm + lm_head (both small enough; mirrors the source layout).
groups = [[n for n in kept if n.startswith("model.embed_tokens") or LAYER_RE.match(n)],
          [n for n in kept if not (n.startswith("model.embed_tokens") or LAYER_RE.match(n))]]
groups = [g for g in groups if g]
os.makedirs(dst, exist_ok=True)
new_map = {}
total = 0
for gi, names in enumerate(groups, start=1):
    out_name = f"model-{gi:05d}-of-{len(groups):05d}.safetensors"
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for n in names:
        shard, dtype, shape, b, e, hlen = desc[n]
        header[n] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + (e - b)]}
        offset += e - b
        new_map[n] = out_name
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * ((ALIGN - len(hjson) % ALIGN) % ALIGN)
    with open(os.path.join(dst, out_name), "wb") as out:
        out.write(struct.pack("<Q", len(hjson)))
        out.write(hjson)
        for n in names:
            shard, dtype, shape, b, e, hlen = desc[n]
            with open(os.path.join(src, shard), "rb") as f:
                f.seek(8 + hlen + b)
                left = e - b
                while left > 0:
                    buf = f.read(min(CHUNK, left))
                    assert buf, f"short read in {shard} for {n}"
                    out.write(buf)
                    left -= len(buf)
    total += offset
    print(f"wrote {out_name}: {len(names)} tensors, {offset / 1e9:.3f} GB")

json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
          open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
cfg = json.load(open(os.path.join(src, "config.json")))
cfg_layers_src = cfg["num_hidden_layers"]
cfg["num_hidden_layers"] = n_layers
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
for fn in SMALL_FILES:
    if os.path.isfile(os.path.join(src, fn)):
        shutil.copyfile(os.path.join(src, fn), os.path.join(dst, fn))

# Verification: re-read the new headers and compare dtype / shape / byte count per tensor with the source.
ok_meta = True
for gi in range(1, len(groups) + 1):
    out_name = f"model-{gi:05d}-of-{len(groups):05d}.safetensors"
    hlen, header = read_header(os.path.join(dst, out_name))
    size = os.path.getsize(os.path.join(dst, out_name))
    end = max(m["data_offsets"][1] for k, m in header.items() if k != "__metadata__")
    ok_meta &= size == 8 + hlen + end
    for n, m in header.items():
        if n == "__metadata__":
            continue
        shard, dtype, shape, b, e, _ = desc[n]
        ok_meta &= m["dtype"] == dtype and m["shape"] == shape and (m["data_offsets"][1] - m["data_offsets"][0]) == e - b
expected = sum(desc[n][4] - desc[n][3] for n in kept)
check("G17-2 tensor count", len(new_map) == len(kept), f"{len(new_map)} (embed 1 + {n_layers} layers x 9 + norm + lm_head)")
check("G17-2 byte total equals the source byte ranges", total == expected, f"{total} B = {total / 1e9:.3f} GB")
check("G17-2 dtype/shape/size per tensor and file sizes match", ok_meta)
check("G17-1 config num_hidden_layers rewritten", cfg["num_hidden_layers"] == n_layers, f"{cfg_layers_src} -> {n_layers}")
check("small files copied", all(os.path.isfile(os.path.join(dst, f)) for f in ["tokenizer.json", "tokenizer_config.json"]))
print("TRUNCATE", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
