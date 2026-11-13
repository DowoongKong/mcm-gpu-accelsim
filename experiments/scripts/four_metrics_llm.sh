#!/bin/bash
# Standard four-metric TSVs for the LLM L1seq runs (docs/four_metrics_llm.ko.md).
# Same extractor and columns as the graph workloads (graphalytics_exp/analysis/four_metrics_77zf),
# one row per model; add a LABEL=<model dir>/L1seq entry once all three arms of that model exit 0.
# LLM footprint = touched_pages_ref / touched_MB_ref (pages in the reference placement map).
# footprint_alloc_* stays NA on purpose: the cudaMalloc rule would count every layer's weights,
# while L1seq runs layer 0 only (docs/s9_b8s256_steps.ko.md 9.11).
set -euo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd)            # accel-sim-docker (host) or /accel-sim (container)
EXPS=(
  "DeepSeek_b4s256_L1seq=deepseek_llama8b_b4s256_steps/L1seq"
  "Qwen_b4s256_L1seq=qwen_b4s256_steps/L1seq"
)
exec python3 "$H/graphalytics_exp/scripts/four_metrics_tsv.py" \
  --root "$H/llm_exp/experiments" --out "$H/llm_exp/analysis/four_metrics_llm" "${EXPS[@]}"
