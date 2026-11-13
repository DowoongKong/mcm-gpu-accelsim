#!/usr/bin/env python3
# Step 27 section 4.7 (docs/s27_prefill_capture_and_decode_plan.ko.md): run h1_capture_v2.py with the layer-0 MoE routing of
# every DECODE step forced to a fixed expert pattern; prefill and dummy runs keep their natural routing.
# vLLM 0.19.1 BaseRouter.select_experts() calls router.capture_fn(topk_ids) right after the fused top-k (before EPLB mapping
# and dtype conversion). A capture_fn is installed on the router of the FusedMoE whose layer index is 0; when the call has
# exactly B tokens (a decode step of this capture) it overwrites topk_ids in place with the forced pattern. A same-dtype
# contiguous copy_ is a device-to-device cudaMemcpyAsync, not a kernel, so the traced kernel list does not change.
# Usage (inside llmtrace, same environment as h1_capture_v2.py):
#   FORCE_ROUTE=min2|max8 python h1_capture_v2_forced.py h1_capture_v2.py <h1_capture_v2.py arguments ...>
#   min2: every token -> experts (0, 1)        max8: token t -> experts (2t, 2t+1)   (top-2, B * 2 <= number of experts)
import atexit
import os
import runpy
import sys

MODE = os.environ.get("FORCE_ROUTE", "")
assert MODE in ("min2", "max8"), f"FORCE_ROUTE must be min2 or max8, got {MODE!r}"
SCRIPT = sys.argv[1]
ARGS = sys.argv[2:]          # h1_capture_v2.py: <out dir> <model key> <B> <S> <max_tokens> <max_batched> <kv bytes> [flags]
B = int(ARGS[2])

import torch  # noqa: E402
from vllm.model_executor.layers.fused_moe import layer as moe_layer  # noqa: E402
from vllm.model_executor.models.utils import extract_layer_index  # noqa: E402

state = {"layers": [], "forced_calls": 0, "natural_calls": 0, "pattern": None, "forced": None, "last_seen": None}


def make_pattern(top_k, n_experts):
    assert top_k == 2, f"forced patterns assume top-2, got top_k={top_k}"
    if MODE == "min2":
        return [[0, 1] for _ in range(B)]
    assert B * top_k <= n_experts, f"max8 needs B*top_k <= experts ({B}*{top_k} > {n_experts})"
    return [[2 * t, 2 * t + 1] for t in range(B)]


orig_init = moe_layer.FusedMoE.__init__


def patched_init(fm, *a, **k):
    orig_init(fm, *a, **k)
    if extract_layer_index(fm.layer_name) != 0:
        return
    state["layers"].append(fm.layer_name)
    state["pattern"] = make_pattern(fm.top_k, fm.global_num_experts)

    def capture_fn(topk_ids):
        if state["forced"] is None:   # first call = engine dummy run, outside every trace window
            state["forced"] = torch.tensor(state["pattern"], dtype=topk_ids.dtype, device=topk_ids.device)
        if topk_ids.shape[0] != B:
            state["natural_calls"] += 1
            return
        topk_ids.copy_(state["forced"])
        state["forced_calls"] += 1

    fm.router.set_capture_fn(capture_fn)


moe_layer.FusedMoE.__init__ = patched_init


@atexit.register
def report():
    ok = len(state["layers"]) == 1 and state["forced_calls"] >= 1 and state["forced"] is not None
    print(f"FORCE ROUTE {'PASS' if ok else 'FAIL'} mode={MODE} layer={state['layers']} pattern={state['pattern']} "
          f"forced_calls={state['forced_calls']} natural_calls={state['natural_calls']}", flush=True)


sys.argv = [SCRIPT] + ARGS
runpy.run_path(SCRIPT, run_name="__main__")
