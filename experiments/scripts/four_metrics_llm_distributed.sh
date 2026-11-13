#!/bin/bash
# Standard four-metric TSVs for the LLM L1seq distributed (KERNEL_PARTITION + per-chiplet cursors + L2 TLB set hash 1)
# runs: same extractor and columns as four_metrics_llm.sh, but arms ref_mcm/base_mcm/ptwhop0_mcm and the
# experiments under llm_exp/experiments_distributed. Run once all three arms of a model exit 0.
set -euo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd)
EXPS=(
  "DeepSeek_b4s256_L1seq_mcm=deepseek_b4s256_L1seq_mcm"
  "Qwen_b4s256_L1seq_mcm=qwen_b4s256_L1seq_mcm"
)
exec python3 -I "$H/graphalytics_exp/scripts/four_metrics_tsv.py" --arms ref_mcm,base_mcm,ptwhop0_mcm \
  --root "$H/llm_exp/experiments_distributed" --out "$H/llm_exp/analysis/four_metrics_llm_distributed" "${EXPS[@]}"
