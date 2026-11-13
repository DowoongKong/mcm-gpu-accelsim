#!/bin/bash
# Step 1: capture workloads/custom/cublas_smoke under a given tracer build (runs inside the llmtrace container).
# Usage: s1_smoke_capture.sh <tag> <tracer .so path> [ACTIVE_FROM_START (default 1)] [extra VAR=value ...]
# Output: /accel-sim/llm_exp/tracer_tests/smoke_<tag>/{traces/, run.log, meta.txt}
# Explained in llm_exp/docs/s1_tracer_async_memcpy.ko.md.
set -u
TAG=$1; SO=$2; AFS=${3:-1}; shift 3 2>/dev/null || shift $#
OUT=/accel-sim/llm_exp/tracer_tests/smoke_$TAG
APP=/accel-sim/workloads/custom/cublas_smoke/cublas_smoke
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT"; exit 2; }
mkdir -p $OUT
{ echo "tag=$TAG"; echo "tracer=$SO"; md5sum "$SO"; echo "ACTIVE_FROM_START=$AFS"; echo "extra=$*"; date; } > $OUT/meta.txt
for try in 1 2 3; do
  rm -rf $OUT/traces
  env CUDA_INJECTION64_PATH=$SO ACTIVE_FROM_START=$AFS USER_DEFINED_FOLDERS=1 TRACES_FOLDER=$OUT NOBANNER=1 "$@" \
    $APP > $OUT/run_try$try.log 2>&1
  rc=$?
  k=$(ls $OUT/traces/kernel-*.trace* 2>/dev/null | wc -l)
  echo "try=$try rc=$rc trace_files=$k" | tee -a $OUT/meta.txt
  if [ $rc -eq 0 ] && grep -q "cublas_smoke done" $OUT/run_try$try.log; then
    if [ "$AFS" = "0" ] || [ "$k" -ge 4 ]; then cp $OUT/run_try$try.log $OUT/run.log; echo OK; exit 0; fi
  fi
done
echo FAILED; exit 1
