# Hypothesis-1 decode capture v2 (docs/s25b_h1_capture_v2_init_window.ko.md): the v1 decode-step layer-0 window (W3,
# scripts/h1_decode_layer_capture.py) plus, with --init-windows, two initialization windows traced in the same process:
#   W1 init_layer0   : layer 0 of the memory-profiling dummy run (GPUModelRunner.profile_run -> _dummy_run(max_num_tokens));
#                      the first SM touch of every weight page
#   W2 kv_zero_fill  : GPUModelRunner._allocate_kv_cache_tensors (torch.zeros of the KV cache); the first SM touch of KV pages
# and a tensor address map (va_map.json): layer-0 weights, per-submodule input/output tensors of the captured decode forward,
# the layer-0 KV cache tensor, the raw KV tensors, block table rows and slot mapping of the captured step.
# W1/W2 do not depend on the batch (fixed token budget and KV size), so the chain requests them for the first batch only.
# Usage (inside llmtrace, tracer attached, ACTIVE_FROM_START=0):
#   h1_capture_v2.py <out dir> <model: llama70b|deepseek|qwen|mixtral> <batch> <S> <max_tokens> <max_batched_tokens> <kv bytes> [--init-windows]
import glob
import json
import os
import re
import sys
import threading
import time

INIT_WINDOWS = "--init-windows" in sys.argv
# --prefill-window (2026-11-11): trace the whole first step (the prefill of every request: prep + all layers + tail) as
# window "prefill_step", so a prefill-derived first-touch placement can be computed for the decode window (step 26/27).
PREFILL_WINDOW = "--prefill-window" in sys.argv
# --prefill-layer0 (2026-11-11): close the prefill window at the end of layer 0's forward of step 1 instead of the step end
# (prep + layer 0 only: the decode-layer-0 placement needs no layer 1-2 / lm_head pages, and the raw trace is 1/3 the size).
PREFILL_LAYER0 = "--prefill-layer0" in sys.argv
# --force-routing pair|spread (2026-11-11, MoE scenario captures): during the captured decode step every MoE layer's router
# logits are replaced so that "pair" sends all tokens to experts {0,1} and "spread" sends token i to {2i, 2i+1} mod E
# (b4 x top-2 -> all 8 experts once). The natural routing of every other step is untouched. Recorded in capture_info.
FORCE_ROUTING = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--force-routing"), None)
assert FORCE_ROUTING is None or FORCE_ROUTING in ("pair", "spread") or FORCE_ROUTING.startswith("active="), FORCE_ROUTING
FORCE_ACTIVE = int(FORCE_ROUTING.split("=")[1]) if FORCE_ROUTING and FORCE_ROUTING.startswith("active=") else None
# active=<n>: token i -> experts {(top_k*i + j) mod n}, so exactly n experts are active (n multiple of top_k keeps the
# per-token set distinct); pair == active=top_k, spread == active=min(E, batch*top_k).
# --min-context N (s41): open the decode window at the first pure-decode step in which EVERY request's context is at
# least N tokens (long-generation capture); 0 = the first pure-decode step (default, unchanged behaviour).
MIN_CTX = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--min-context"), 0)
argv = [a for i, a in enumerate(sys.argv[1:], 1)
        if not a.startswith("--") and sys.argv[i - 1] not in ("--force-routing", "--min-context")]
OUT, MODEL_KEY = argv[0], argv[1]
BATCH, S, MAX_TOKENS, MAX_BATCHED, KV_BYTES = (int(x) for x in argv[2:7])
MODELS = {"deepseek": ("/hf_cache/DeepSeek-R1-Distill-Llama-8B", 32, 128000),
          "qwen": ("/hf_cache/Qwen2.5-7B", 28, None),
          "llama70b": ("/hf_cache/Llama-3.3-70B-Instruct-L3", 3, 128000),
          "mixtral": ("/hf_cache/Mixtral-8x7B-v0.1-L3", 3, 1),
          "qwen3moe": ("/hf_cache/Qwen3-30B-A3B-L3", 3, None),     # Qwen3 prepends no BOS
          "qwen3dense": ("/hf_cache/Qwen3-32B-L3", 3, None)}
MODEL_DIR, N_LAYERS, BOS_ID = MODELS[MODEL_KEY]
assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
assert "CUDA_INJECTION64_PATH" in os.environ, "the tracer must be attached"
assert os.environ.get("ACTIVE_FROM_START") == "0", "ACTIVE_FROM_START=0 is required"
assert os.path.realpath(os.environ["TRACES_FOLDER"]) == os.path.realpath(OUT), "TRACES_FOLDER must be the out dir"
TRACES = os.path.join(OUT, "traces")
assert not os.path.exists(TRACES) or not os.listdir(TRACES), f"{TRACES} must start empty"

import torch
from vllm import SamplingParams
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_setup import build_prompts_adaptive, load_tokenizer_from, make_llm_decode_sweep


def stats_path():
    paths = glob.glob(os.path.join(TRACES, "stats_ctx_*"))
    assert len(paths) <= 1, f"expected at most one CUDA context stats file, found {paths}"
    return paths[0] if paths else None


def launched_kernels():
    p = stats_path()
    if p is None:
        return 0
    with open(p) as f:
        return sum(1 for _ in f) - 1


def tinfo(t):
    return {"ptr": int(t.data_ptr()), "bytes": int(t.numel() * t.element_size()), "shape": list(t.shape),
            "stride": list(t.stride()), "dtype": str(t.dtype)}


windows = {"init_layer0": None, "kv_zero_fill": None, "prefill_step": None, "prefill_layer0": None, "decode_layer0": None}
window_events = []            # (name, open|close, launched kernels so far) in order, for the non-overlap check
va = {"weights": {}, "weights_all": {}, "modules": {}, "kv_cache_layer0": None, "kv_tensors": {}, "block_table": None, "slot_mapping": None,
      "num_computed_tokens": None, "kv_block_size": None, "attn_metadata_keys": None, "errors": []}
state = {"step": 0, "armed": False, "captured": False, "capture_step": None, "layer0_tokens": None, "init_tokens": None,
         "routing_layer0": None, "routing_forced_calls": 0, "routing_steps": [], "sched": [],
         "prefill_step": None, "prefill_tokens": None, "prefill_new_reqs": None, "prefill_l0_armed": False, "prefill_l0_tokens": None,
         "seq_lens": None, "steps": [], "hook_thread_ok": True}
main_thread = threading.get_ident()


def open_window(name):
    assert windows[name] is None, f"window {name} opened twice"
    windows[name] = [launched_kernels() + 1, None]
    window_events.append((name, "open", windows[name][0]))
    torch.cuda.profiler.start()


def close_window(name):
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    windows[name][1] = launched_kernels()
    window_events.append((name, "close", windows[name][1]))


print("H1v2 start", time.strftime("%Y-%m-%d %H:%M:%S"), MODEL_KEY, "S", S, "batch", BATCH, "max_tokens", MAX_TOKENS,
      "max_batched", MAX_BATCHED, "kv_bytes", KV_BYTES, "init_windows", INIT_WINDOWS)

if INIT_WINDOWS:
    _orig_profile_run = GPUModelRunner.profile_run
    _orig_alloc = GPUModelRunner._allocate_kv_cache_tensors

    def profile_run_patched(runner):
        # W1: the first layer-0 forward inside the profiling dummy run
        model = runner.get_model() if hasattr(runner, "get_model") else runner.model
        layer0 = model.get_submodule("model.layers.0")
        local = {"done": False}

        def pre(m, a, kw):
            if local["done"] or windows["init_layer0"] is not None:
                return
            hs = kw["hidden_states"] if "hidden_states" in kw else a[1]
            state["init_tokens"] = int(hs.shape[0])
            open_window("init_layer0")

        def post(m, a, kw, out):
            if windows["init_layer0"] is not None and windows["init_layer0"][1] is None:
                close_window("init_layer0")
                local["done"] = True

        h1 = layer0.register_forward_pre_hook(pre, with_kwargs=True)
        h2 = layer0.register_forward_hook(post, with_kwargs=True)
        try:
            return _orig_profile_run(runner)
        finally:
            h1.remove()
            h2.remove()

    def alloc_patched(runner, kv_cache_config):
        # W2: torch.zeros of every raw KV cache tensor
        open_window("kv_zero_fill")
        out = _orig_alloc(runner, kv_cache_config)
        close_window("kv_zero_fill")
        try:
            va["kv_tensors"] = {str(k): tinfo(v) for k, v in out.items()}
        except Exception as e:  # noqa: BLE001 - recorded, not fatal
            va["errors"].append(f"kv_tensors: {e!r}")
        return out

    GPUModelRunner.profile_run = profile_run_patched
    GPUModelRunner._allocate_kv_cache_tensors = alloc_patched

tok = load_tokenizer_from(MODEL_DIR)
prompts, stride = build_prompts_adaptive(tok, S, BATCH, bos_id=BOS_ID)
print("prompt stride", stride)
t0 = time.perf_counter()
llm = make_llm_decode_sweep(MODEL_DIR, S, BATCH, KV_BYTES, MAX_TOKENS, MAX_BATCHED)
t_init = time.perf_counter() - t0
print("engine ready", round(t_init, 1), "s; windows so far", {k: v for k, v in windows.items() if v})


def attach(worker):
    runner = worker.model_runner
    original = runner.execute_model
    try:
        va["kv_block_size"] = int(runner.cache_config.block_size)
    except Exception as e:  # noqa: BLE001
        va["errors"].append(f"kv_block_size: {e!r}")

    def execute_model(*args, **kwargs):
        so = args[0] if args else kwargs["scheduler_output"]
        state["step"] += 1
        n_sched = so.num_scheduled_tokens
        n_new = len(so.scheduled_new_reqs)
        cached = so.scheduled_cached_reqs
        pure_decode = (n_new == 0 and len(cached.req_ids) == BATCH and len(n_sched) == BATCH
                       and so.total_num_scheduled_tokens == BATCH and all(v == 1 for v in n_sched.values()))
        state["steps"].append((state["step"], len(n_sched), so.total_num_scheduled_tokens, n_new, pure_decode))
        try:   # which request got how many tokens in this step, from which position (chunked prefill replay for KV placement)
            comp = {r.req_id: int(r.num_computed_tokens) for r in so.scheduled_new_reqs}
            comp.update({rid: int(nc) for rid, nc in zip(cached.req_ids, cached.num_computed_tokens)})
            state["sched"].append({"step": state["step"], "reqs": [[rid, int(n), comp.get(rid)] for rid, n in n_sched.items()]})
        except Exception as e:  # noqa: BLE001
            va["errors"].append(f"sched step {state['step']}: {e!r}")
        if (pure_decode and not state["captured"] and not state["armed"]
                and min(int(n) for n in cached.num_computed_tokens) + 1 >= MIN_CTX):
            state["armed"] = True
            state["capture_step"] = state["step"]
            state["seq_lens"] = [int(n) + 1 for n in cached.num_computed_tokens]
            va["num_computed_tokens"] = [int(n) for n in cached.num_computed_tokens]
        # prefill window: the step that admits every request at once (n_new == BATCH, BATCH*S prompt tokens)
        pw = "prefill_layer0" if PREFILL_LAYER0 else "prefill_step"
        prefill_now = (PREFILL_WINDOW and windows[pw] is None and n_new == BATCH
                       and so.total_num_scheduled_tokens == BATCH * S)
        if prefill_now:
            state["prefill_step"] = state["step"]; state["prefill_tokens"] = int(so.total_num_scheduled_tokens)
            state["prefill_new_reqs"] = n_new
            try:
                va["weights_all"] = {n: tinfo(t) for n, t in worker.get_model().named_parameters()}
            except Exception as e:  # noqa: BLE001
                va["errors"].append(f"weights_all: {e!r}")
            if PREFILL_LAYER0:
                state["prefill_l0_armed"] = True      # the layer-0 forward hook closes the window
            open_window(pw)
        out = original(*args, **kwargs)
        if prefill_now and not PREFILL_LAYER0:
            close_window("prefill_step")
        if prefill_now and PREFILL_LAYER0 and windows["prefill_layer0"][1] is None:
            close_window("prefill_layer0"); state["prefill_l0_armed"] = False   # safety: hook did not fire
        if state["capture_step"] == state["step"]:
            # block table rows of the captured step (after the step, the table is committed for these requests)
            try:
                bt = runner.input_batch.block_table[0]
                n_rows = min(BATCH, int(bt.num_blocks_per_row.shape[0]))
                cpu = bt.get_cpu_tensor()[:n_rows]
                va["block_table"] = [cpu[i, : int(bt.num_blocks_per_row[i])].tolist() for i in range(n_rows)]
            except Exception as e:  # noqa: BLE001
                va["errors"].append(f"block_table: {e!r}")
        return out

    runner.execute_model = execute_model
    model = worker.get_model()
    layer0 = model.get_submodule("model.layers.0")

    # MoE: record layer-0 routing of the captured step; optionally force it (pair / spread)
    moe0 = next((m for n, m in layer0.named_modules() if n.endswith("block_sparse_moe") or n.endswith("mlp") and hasattr(m, "experts") and hasattr(m, "gate")), None)
    if moe0 is not None and hasattr(moe0, "gate") and hasattr(moe0, "experts"):
        moe_cls = type(moe0)
        n_exp = int(getattr(moe0.experts, "global_num_experts", getattr(moe0.experts, "num_experts", 8)))
        top_k = int(getattr(moe0.experts, "top_k", 2))

        def forced_logits(hs):
            # Built on the CPU and moved once: the earlier element-wise GPU assignment issued one 1-element HtoD memcpy per
            # (token, slot) inside the traced window (b4: 32, b256: 2,048 memcpy lines in the kernel list). Measured with
            # torch.profiler on 2026-11-11: this version launches 0 kernels and 1 HtoD copy, output identical.
            n = hs.shape[0]
            n_act = top_k if FORCE_ROUTING == "pair" else (FORCE_ACTIVE if FORCE_ACTIVE else n_exp)
            lg = torch.full((n, n_exp), -1e4, dtype=hs.dtype)
            rows = torch.arange(n).repeat_interleave(top_k)
            cols = torch.tensor([(top_k * i + j) % n_act for i in range(n) for j in range(top_k)])
            lg[rows, cols] = torch.tensor([10.0 - j for i in range(n) for j in range(top_k)], dtype=hs.dtype)  # distinct values keep the top-k order stable
            return lg.to(hs.device)

        # Wrap the router (gate) of every MoE block instead of replacing the block's forward: the block bodies differ per
        # model (MixtralMoE: experts(hs, logits) -> tensor; Qwen3MoeSparseMoeBlock: experts(hidden_states=, router_logits=)
        # -> (shared_out, fused_out), no hidden_size attribute), while the gate is a ReplicatedLinear in both and returns
        # (logits, bias). Forcing replaces the logits while armed; layer 0's routing is recorded every step.
        def wrap_gate(mm):
            orig_gate_forward = mm.gate.forward
            is_layer0 = mm is moe0

            def gate_forward(x):
                out = orig_gate_forward(x)
                router_logits, rest = (out[0], tuple(out[1:])) if isinstance(out, tuple) else (out, None)
                if state["armed"] and FORCE_ROUTING:
                    router_logits = forced_logits(x); state["routing_forced_calls"] += 1
                if is_layer0:
                    # One DtoH copy, then CPU-only: the earlier GPU-side float() + topk added 3 kernels (cast, gatherTopK,
                    # bitonicSort) to every traced window (visible in the Mixtral pair/spread and hook-test kernel lists).
                    top = torch.topk(router_logits.detach().to("cpu", torch.float32), top_k, dim=-1).indices
                    hist = torch.bincount(top.flatten(), minlength=n_exp).tolist()
                    rec = {"step": state["step"], "tokens": int(x.shape[0]), "top_k": top_k, "n_experts": n_exp,
                           "expert_hist": hist, "active_experts": int(sum(1 for h in hist if h)), "forced": bool(state["armed"] and FORCE_ROUTING)}
                    if len(state["routing_steps"]) < 4096: state["routing_steps"].append(rec)
                    if state["armed"]:
                        state["routing_layer0"] = dict(rec, topk_ids=top.tolist())
                return (router_logits, *rest) if rest is not None else router_logits
            mm.gate.forward = gate_forward

        n_wrapped = 0
        for mm in model.modules():
            if type(mm) is moe_cls and hasattr(mm, "gate"):
                wrap_gate(mm); n_wrapped += 1
        print("MOE gate wrapped", n_wrapped, "blocks class", moe_cls.__name__, "experts", n_exp, "top_k", top_k, flush=True)

    def record_tensors(x):
        if torch.is_tensor(x):
            return [tinfo(x)]
        if isinstance(x, (list, tuple)):
            return [tinfo(t) for t in x if torch.is_tensor(t)]
        return []

    def make_pre(name):
        def pre(m, a, kw):
            if state["armed"]:
                va["modules"].setdefault(name, {})["inputs"] = record_tensors(list(a) + list(kw.values()))
        return pre

    def make_post(name):
        def post(m, a, kw, out):
            if state["armed"]:
                va["modules"].setdefault(name, {})["outputs"] = record_tensors(out)
        return post

    for name, m in layer0.named_modules():
        if name == "":
            continue
        m.register_forward_pre_hook(make_pre(name), with_kwargs=True)
        m.register_forward_hook(make_post(name), with_kwargs=True)

    def on_enter(m, args, kwargs):
        if state["armed"]:
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1]
            state["layer0_tokens"] = int(hs.shape[0])
            state["hook_thread_ok"] &= threading.get_ident() == main_thread
            va["weights"] = {n: tinfo(p) for n, p in layer0.named_parameters()}
            for name, sub in layer0.named_modules():
                kvc = getattr(sub, "kv_cache", None)
                # vLLM 0.19.1 bind_kv_cache() assigns the shaped view tensor itself (v1/worker/utils.py:
                # forward_context[layer_name].kv_cache = kv_cache); older versions kept a one-element list.
                if isinstance(kvc, (list, tuple)):
                    kvc = kvc[0] if kvc else None
                if torch.is_tensor(kvc) and kvc.numel() > 0:
                    va["kv_cache_layer0"] = {"module": name, **tinfo(kvc)}
            try:
                from vllm.forward_context import get_forward_context
                md = get_forward_context().attn_metadata
                if isinstance(md, list):
                    md = md[0]
                if isinstance(md, dict):
                    va["attn_metadata_keys"] = list(md.keys())[:8]
                    key = next((k for k in md if ".layers.0." in k), next(iter(md)))
                    m0 = md[key]
                    sm = getattr(m0, "slot_mapping", None)
                    if torch.is_tensor(sm):
                        va["slot_mapping"] = sm[:BATCH].cpu().tolist()
                    bt = getattr(m0, "block_table", None)
                    if torch.is_tensor(bt) and va["block_table"] is None:
                        va["block_table_tensor_shape"] = list(bt.shape)
            except Exception as e:  # noqa: BLE001
                va["errors"].append(f"attn_metadata: {e!r}")
            open_window("decode_layer0")

    def on_exit(m, args, kwargs, output):
        if state["prefill_l0_armed"]:
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1]
            state["prefill_l0_tokens"] = int(hs.shape[0])
            close_window("prefill_layer0")
            state["prefill_l0_armed"] = False
            return
        if state["armed"]:
            close_window("decode_layer0")
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

expected_windows = (["init_layer0", "kv_zero_fill"] if INIT_WINDOWS else []) + (["prefill_layer0" if PREFILL_LAYER0 else "prefill_step"] if PREFILL_WINDOW else []) + ["decode_layer0"]
got = {k: v for k, v in windows.items() if v}
check("H1v2-1 the expected windows were opened and closed", sorted(got) == sorted(expected_windows)
      and all(v[1] is not None and v[1] >= v[0] for v in got.values()), str(got))
spans = sorted(v for v in got.values())
check("H1v2-1 windows do not overlap", all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1)), str(spans))
if INIT_WINDOWS:
    check("H1v2-2 init layer-0 saw max_num_batched_tokens rows", state["init_tokens"] == MAX_BATCHED, state["init_tokens"])
if PREFILL_WINDOW:
    check("H1v3-1 the prefill window is step 1 and admitted every request at once", state["prefill_step"] == 1
          and state["prefill_new_reqs"] == BATCH, f"step {state['prefill_step']} new_reqs {state['prefill_new_reqs']}")
    check("H1v3-2 the prefill step scheduled exactly batch x S prompt tokens", state["prefill_tokens"] == BATCH * S, state["prefill_tokens"])
    check("H1v3-3 every model parameter recorded in va_map (weights_all)", len(va["weights_all"]) >= 4 * N_LAYERS,
          f"{len(va['weights_all'])} tensors, {sum(t['bytes'] for t in va['weights_all'].values())} B")
    if PREFILL_LAYER0:
        check("H1v3-4 the prefill window closed at the end of layer 0 with batch x S rows", state["prefill_l0_tokens"] == BATCH * S
              and windows["prefill_layer0"] is not None and windows["prefill_layer0"][1] is not None, f"rows {state['prefill_l0_tokens']} window {windows['prefill_layer0']}")
check("H1-1 a step with all sequences decoding (1 token each) occurred", state["captured"], f"step {state['capture_step']}")
check("H1-2 layer 0 saw exactly batch rows in that step", state["layer0_tokens"] == BATCH, state["layer0_tokens"])
check("H1-2 hooks ran on the main thread", state["hook_thread_ok"])
check("H1-3 every request produced max_tokens", len(out_ids) == BATCH and all(len(o) == MAX_TOKENS for o in out_ids),
      f"{len(out_ids)} requests")
seq = state["seq_lens"] or []
check("H1-4 sequence lengths at the captured step recorded", len(seq) == BATCH,
      f"min {min(seq) if seq else None} max {max(seq) if seq else None}")
if state["routing_layer0"] is not None:
    r = state["routing_layer0"]
    print("ROUTING layer0 step", state["capture_step"], "tokens", r["tokens"], "active experts", r["active_experts"], "hist", r["expert_hist"])
    if FORCE_ROUTING == "pair":
        check("H1v4-1 forced routing 'pair': every token on experts 0..top_k-1", all(sorted(t) == list(range(r["top_k"])) for t in r["topk_ids"]), str(r["expert_hist"]))
    elif FORCE_ROUTING == "spread":
        want = min(r["n_experts"], r["tokens"] * r["top_k"])
        check("H1v4-1 forced routing 'spread': experts used == min(E, batch x top_k)", r["active_experts"] == want, f"{r['active_experts']} of {want}, hist {r['expert_hist']}")
    elif FORCE_ACTIVE:
        want = min(FORCE_ACTIVE, r["tokens"] * r["top_k"])
        check(f"H1v4-1 forced routing active={FORCE_ACTIVE}: experts used == {want}", r["active_experts"] == want, f"{r['active_experts']} active, hist nonzero {sum(1 for h in r['expert_hist'] if h)}")
elif FORCE_ROUTING:
    check("H1v4-1 forced routing was applied to a MoE layer", False, "no MoE module found")

rows = {}
with open(stats_path()) as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\n").split(",")
        nums = [int(p) for p in parts[-10:]]
        kid = int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1))
        rows[kid] = {"name": ",".join(parts[1:-10]).strip(), "grid": nums[0:3], "block": nums[4:7],
                     "total_insts": nums[8], "reported_insts": nums[9]}
window_ids = sorted(k for v in got.values() for k in range(v[0], v[1] + 1))
kernelslist = glob.glob(os.path.join(TRACES, "kernelslist_ctx_*"))
klines = open(kernelslist[0]).read().splitlines() if len(kernelslist) == 1 else []
memcpy = [ln for ln in klines if ln.startswith("MemcpyHtoD,")]
kl_ids = [int(re.match(r"kernel-(\d+)-ctx", ln).group(1)) for ln in klines if not ln.startswith("MemcpyHtoD,")]
trace_files = sorted(glob.glob(os.path.join(TRACES, "kernel-*.trace*")))
tf_ids = sorted(int(re.search(r"kernel-(\d+)-ctx", os.path.basename(p)).group(1)) for p in trace_files)
check("one CUDA context", len(kernelslist) == 1 and len(glob.glob(os.path.join(TRACES, "stats_ctx_*"))) == 1)
check("H1v2-4 kernelslist kernel ids == union of the windows", kl_ids == window_ids, f"{len(kl_ids)} listed, windows {got}")
check("H1v2-4 trace file ids == union of the windows", tf_ids == window_ids, f"{len(tf_ids)} files")
check("H1v2-4 every trace file is non-empty", all(os.path.getsize(p) > 0 for p in trace_files))
check("H1v2-4 instructions reported only for window kernels",
      all((r["reported_insts"] > 0) == (k in set(window_ids)) for k, r in rows.items()))
dec = got["decode_layer0"]
dec_ids = list(range(dec[0], dec[1] + 1))
check("H1-6 decode window holds a plausible decoder layer (10..40 kernels)", 10 <= len(dec_ids) <= 40, len(dec_ids))
if INIT_WINDOWS:
    kv = got["kv_zero_fill"]
    kv_names = [rows[k]["name"] for k in range(kv[0], kv[1] + 1)]
    check("H1v2-3 KV zero-fill window holds only fill/elementwise kernels", len(kv_names) >= 1 and
          all(("fill" in n.lower() or "elementwise" in n.lower() or "vectorized" in n.lower()) for n in kv_names), str(kv_names[:3])[:200])
    check("H1v2-6 raw KV tensors recorded and sum to kv bytes", bool(va["kv_tensors"]) and
          abs(sum(t["bytes"] for t in va["kv_tensors"].values()) - KV_BYTES) <= KV_BYTES // 100,
          f"{len(va['kv_tensors'])} tensors, {sum(t['bytes'] for t in va['kv_tensors'].values()) if va['kv_tensors'] else 0} B")
check("H1v2-6 va_map has weights, module I/O and the layer-0 KV cache",
      len(va["weights"]) >= 4 and len(va["modules"]) >= 5 and va["kv_cache_layer0"] is not None,
      f"weights {len(va['weights'])} modules {len(va['modules'])} kv {va['kv_cache_layer0'] is not None} errors {va['errors'][:2]}")

print("window kernels:")
for wname, (a, b) in got.items():
    for k in range(a, b + 1):
        r = rows[k]
        print(f"  [{wname}] {k} grid={tuple(r['grid'])} block={tuple(r['block'])} insts={r['reported_insts']} {r['name'][:90]}")
warp_insts = {w: sum(rows[k]["reported_insts"] for k in range(v[0], v[1] + 1)) for w, v in got.items()}
summary = {"model": MODEL_KEY, "model_dir": MODEL_DIR, "n_layers": N_LAYERS, "S": S, "batch": BATCH, "version": 2,
           "init_windows": INIT_WINDOWS, "prefill_window": PREFILL_WINDOW, "prefill_layer0_only": PREFILL_LAYER0, "prefill_step": state["prefill_step"],
           "force_routing": FORCE_ROUTING, "min_context": MIN_CTX, "routing_layer0": state["routing_layer0"], "routing_forced_calls": state["routing_forced_calls"],
           "routing_steps": state["routing_steps"], "sched": state["sched"],
           "prefill_tokens": state["prefill_tokens"], "max_tokens": MAX_TOKENS, "max_batched_tokens": MAX_BATCHED, "kv_cache_bytes": KV_BYTES,
           "prompt_stride": stride, "init_s": t_init, "generate_s": t_gen, "total_steps": state["step"],
           "capture_step": state["capture_step"], "layer0_tokens": state["layer0_tokens"], "init_tokens": state["init_tokens"],
           "seq_lens_min": min(seq) if seq else None, "seq_lens_max": max(seq) if seq else None,
           "windows": windows, "window_events": window_events, "window": list(dec), "n_window_kernels": len(dec_ids),
           "window_kernels": [{"id": k, "window": w, "name": rows[k]["name"], "grid": rows[k]["grid"], "block": rows[k]["block"],
                               "warp_insts": rows[k]["reported_insts"]} for w, v in got.items() for k in range(v[0], v[1] + 1)],
           "window_warp_insts": warp_insts["decode_layer0"], "window_warp_insts_by_window": warp_insts,
           "launched_total": len(rows), "trace_files": len(trace_files),
           "trace_bytes": sum(os.path.getsize(p) for p in trace_files),
           "memcpy_lines": len(memcpy), "memcpy_bytes": sum(int(ln.split(",")[2]) for ln in memcpy),
           "output_ids_first3": [o[:3] for o in out_ids[:4]]}
with open(os.path.join(OUT, "capture_info.json"), "w") as f:
    json.dump(summary, f, indent=1)
with open(os.path.join(OUT, "va_map.json"), "w") as f:
    json.dump(va, f, indent=1)
print("windows", got, "warp insts", warp_insts, "trace bytes", summary["trace_bytes"], "memcpy", len(memcpy), summary["memcpy_bytes"])
print("H1 CAPTURE", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
