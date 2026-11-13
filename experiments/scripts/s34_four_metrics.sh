#!/bin/bash
# s34_four_metrics.sh <decode capture dir> [out.tsv] -- four metrics of the decode layer-0 window with the prefill-extrapolated placement:
# ref_mcm = decode-own first touch (sim_noprefill), base = base_prefill, ptwhop0 = ptwhop0_prefill (sim_prefillchunked), through
# scripts/h1_slice_four_metrics.py on a symlinked arm set (base_prefill = 1 normalization, ref raw values alongside).
set -u
CAP=$(cd ${1:?capture dir} && pwd); OUT=${2:-$CAP/sim_prefillchunked/four_metrics.tsv}; H=/home/user/accel-sim-docker/llm_exp
read A B BATCH <<< "$(python3 -c "import json;ci=json.load(open('$CAP/capture_info.json'));w=ci['windows']['decode_layer0'];print(w[0],w[1],ci['batch'])")"
X=$CAP/sim_prefillchunked/_metrics_view; mkdir -p $X; ln -sfn $CAP/sim_noprefill/ref_mcm $X/ref_mcm; ln -sfn $CAP/sim_prefillchunked/base_prefill $X/base_mcm; ln -sfn $CAP/sim_prefillchunked/ptwhop0_prefill $X/ptwhop0_mcm
for a in ref_mcm base_mcm ptwhop0_mcm; do grep -q '^EXIT=0' $X/$a/exit_code.txt 2>/dev/null || { echo "$a not finished"; exit 1; }; done
N=$(grep -vc MemcpyHtoD $CAP/traces/kernelslist_decode.g)   # h1_slice_four_metrics.py takes 1-based kernel BLOCK indices of the run, not kernel ids
python3 $H/scripts/h1_slice_four_metrics.py $X 64k 1-$N $BATCH $OUT
