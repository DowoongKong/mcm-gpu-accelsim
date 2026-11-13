# V2: repeat the V1 hook flow with the NVBit tracer attached but never activated (ACTIVE_FROM_START=0).
# Without CUDA_INJECTION64_PATH the same script runs as the baseline. Explained in docs/v2_tracer_attach.ko.md.
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

S = int(sys.argv[1]) if len(sys.argv) > 1 else 256
LOG_DIR = sys.argv[2] if len(sys.argv) > 2 else "logs"
MODE = "tracer" if "CUDA_INJECTION64_PATH" in os.environ else "baseline"
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
TRACES = None
if MODE == "tracer":
    # Nothing in V2 calls cudaProfilerStart, so with ACTIVE_FROM_START=0 no kernel is ever traced.
    assert os.environ.get("ACTIVE_FROM_START") == "0", "V2 requires ACTIVE_FROM_START=0"
    TRACES = os.path.join(os.environ["TRACES_FOLDER"], "traces")
    assert not os.path.exists(TRACES) or not os.listdir(TRACES), f"{TRACES} must start empty"

import torch
from vllm import SamplingParams

from vllm_setup import build_prompt_ids, load_tokenizer, make_llm

LAYERS = (0, 15, 31)
SUBLAYERS = ("self_attn", "mlp")
TARGETS = [f"model.layers.{i}.{sub}" for i in LAYERS for sub in SUBLAYERS]
V0_OUTPUT_S256 = [7533, 9612]


def stats_path():
    paths = glob.glob(os.path.join(TRACES, "stats_ctx_*"))
    assert len(paths) == 1, f"expected exactly one CUDA context stats file, found {paths}"
    return paths[0]


def launched_kernels():
    # The tracer appends one stats line per kernel launch even while tracing is inactive
    # (tracer_tool.cu:595-603 on entry, :647-651 on exit), after a one-line header.
    if MODE != "tracer":
        return -1
    with open(stats_path()) as f:
        return sum(1 for _ in f) - 1


print("V2 start", time.strftime("%Y-%m-%d %H:%M:%S"), "mode", MODE, "S", S)
tok = load_tokenizer()
ids, text_ids = build_prompt_ids(tok, S)
t0 = time.perf_counter()
llm = make_llm(S)
t_init = time.perf_counter() - t0
kernels_after_init = launched_kernels()

main_thread = threading.get_ident()
events = []
calls = {}


def hidden_tokens(args, kwargs):
    # LlamaDecoderLayer calls self_attn with keyword arguments and mlp positionally (llama.py:328, :332).
    hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
    return int(hs.shape[0])


def attach(worker):
    model = worker.get_model()
    for tag in TARGETS:
        mod = model.get_submodule(tag)

        def on_enter(m, args, kwargs, tag=tag):
            calls[tag] = calls.get(tag, 0) + 1
            events.append({"tag": tag, "kind": "enter", "call": calls[tag], "tokens": hidden_tokens(args, kwargs),
                           "thread": threading.get_ident(), "launched": launched_kernels()})

        def on_exit(m, args, kwargs, output, tag=tag):
            events.append({"tag": tag, "kind": "exit", "call": calls[tag], "tokens": int(output.shape[0]),
                           "thread": threading.get_ident(), "launched": launched_kernels()})

        mod.register_forward_pre_hook(on_enter, with_kwargs=True)
        mod.register_forward_hook(on_exit, with_kwargs=True)


llm.collective_rpc(attach)
params = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True, detokenize=False)
t1 = time.perf_counter()
out = llm.generate({"prompt_token_ids": ids}, params)[0]
t_gen = time.perf_counter() - t1
kernels_after_generate = launched_kernels()
out_ids = list(out.outputs[0].token_ids)
free_b, total_b = torch.cuda.mem_get_info()

print("init_s", round(t_init, 1), "generate_s", round(t_gen, 3))
print("output_token_ids", out_ids, "text", repr(tok.decode(out_ids)))
print("max_allocated_GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2),
      "device_used_GiB", round((total_b - free_b) / 2**30, 2))

summary = {
    "mode": MODE, "S": S, "init_s": t_init, "generate_s": t_gen, "output_ids": out_ids,
    "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "device_used_gib": (total_b - free_b) / 2**30,
    "events": [[e["tag"], e["kind"], e["call"], e["tokens"]] for e in events],
    "kernels_after_init": kernels_after_init, "kernels_after_generate": kernels_after_generate,
}

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


expected_order = [(tag, kind) for step in (1, 2) for tag in TARGETS for kind in ("enter", "exit")]
check("hook event order (V1)", [(e["tag"], e["kind"]) for e in events] == expected_order, f"{len(events)} events")
check("calls per target == 2 (V1)", all(calls.get(tag) == 2 for tag in TARGETS), calls)
check("prefill tokens == S, decode tokens == 1 (V1)",
      all(e["tokens"] == (S if e["call"] == 1 else 1) for e in events), f"S={S}")
check("all hooks on main thread (V1)", all(e["thread"] == main_thread for e in events))
if S == 256:
    check("output equals V0", out_ids == V0_OUTPUT_S256, out_ids)

if MODE == "tracer":
    # Stats file: "<kernel-ID-ctx_0x...trace.xz>, <mangled name>, gx, gy, gz, blocks, bx, by, bz, threads, total, reported"
    rows = []
    with open(stats_path()) as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            nums = [int(p) for p in parts[-10:]]
            rows.append({
                "id": int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1)),
                "name": ",".join(parts[1:-10]).strip(),
                "grid": nums[0:3], "block": nums[4:7], "total_insts": nums[8], "reported_insts": nums[9],
            })
    mangled = sorted({r["name"] for r in rows})
    demangled = subprocess.run(["c++filt"], input="\n".join(mangled), capture_output=True, text=True).stdout.split("\n")
    pretty = dict(zip(mangled, demangled))

    kernelslists = glob.glob(os.path.join(TRACES, "kernelslist_ctx_*"))
    klines = open(kernelslists[0]).read().splitlines() if len(kernelslists) == 1 else []
    memcpy = [ln for ln in klines if ln.startswith("MemcpyHtoD,")]
    memcpy_bytes = sum(int(ln.split(",")[2]) for ln in memcpy)
    trace_files = glob.glob(os.path.join(TRACES, "kernel-*.trace*"))

    check("one CUDA context (one stats + one kernelslist)", len(kernelslists) == 1, os.listdir(TRACES))
    check("no per-kernel trace files (tracing never active)", not trace_files, len(trace_files))
    check("kernelslist has only MemcpyHtoD lines", len(klines) == len(memcpy), f"{len(klines)} lines")
    check("stats ids consecutive from 1", [r["id"] for r in rows] == list(range(1, len(rows) + 1)), len(rows))
    check("stats count == launched_kernels() after generate", len(rows) == kernels_after_generate, kernels_after_generate)
    check("no instructions reported (tracing never active)", all(r["reported_insts"] == 0 for r in rows))

    by_id = {r["id"]: r for r in rows}
    windows = []
    for i in range(0, len(events), 2):
        enter, leave = events[i], events[i + 1]
        first, last = enter["launched"] + 1, leave["launched"]
        names = [by_id[k]["name"] for k in range(first, last + 1)]
        windows.append({"tag": enter["tag"], "call": enter["call"], "first": first, "last": last, "names": names})
    check("every window launched >= 1 kernel", all(w["last"] >= w["first"] for w in windows),
          [w["last"] - w["first"] + 1 for w in windows])
    check("windows are ordered and disjoint",
          all(windows[i + 1]["first"] > windows[i]["last"] for i in range(len(windows) - 1)))
    same_seq = True
    for call in (1, 2):
        for sub in SUBLAYERS:
            seqs = [w["names"] for w in windows if w["call"] == call and w["tag"].endswith(sub)]
            same_seq = same_seq and all(s == seqs[0] for s in seqs)
    check("kernel name sequence identical across layers 0/15/31 (per phase, per sublayer)", same_seq)

    with open(os.path.join(LOG_DIR, f"v2_kernels_S{S}.tsv"), "w") as f:
        f.write("id\tgrid\tblock\ttotal_insts\treported_insts\tdemangled_name\tmangled_name\n")
        for r in rows:
            f.write(f"{r['id']}\t{r['grid']}\t{r['block']}\t{r['total_insts']}\t{r['reported_insts']}\t"
                    f"{pretty[r['name']]}\t{r['name']}\n")
    with open(os.path.join(LOG_DIR, f"v2_windows_S{S}.tsv"), "w") as f:
        f.write("tag\tphase\tfirst_id\tlast_id\tn_kernels\tkernels\n")
        for w in windows:
            phase = "prefill" if w["call"] == 1 else "decode"
            short = " | ".join(pretty[n][:100] for n in w["names"])
            f.write(f"{w['tag']}\t{phase}\t{w['first']}\t{w['last']}\t{len(w['names'])}\t{short}\n")

    for w in windows:
        phase = "prefill" if w["call"] == 1 else "decode"
        print("window", w["tag"], phase, "ids", w["first"], "-", w["last"], "n", len(w["names"]))
        for n in w["names"]:
            print("   ", pretty[n][:140])
    counts = {}
    for r in rows:
        counts[r["name"]] = counts.get(r["name"], 0) + 1
    print("kernels launched: after init", kernels_after_init, "after generate", kernels_after_generate,
          "during generate", kernels_after_generate - kernels_after_init, "unique names", len(counts))
    print("MemcpyHtoD lines", len(memcpy), "bytes", memcpy_bytes, f"({memcpy_bytes / 2**30:.3f} GiB)")
    summary.update({"n_kernel_launches": len(rows), "unique_kernel_names": len(counts),
                    "memcpy_lines": len(memcpy), "memcpy_bytes": memcpy_bytes,
                    "windows": [[w["tag"], w["call"], w["first"], w["last"]] for w in windows]})

with open(os.path.join(LOG_DIR, f"v2_summary_{MODE}_S{S}.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("V2 RESULT", MODE, "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
print("V2 end", time.strftime("%Y-%m-%d %H:%M:%S"))
