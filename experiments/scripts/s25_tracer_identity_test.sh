#!/bin/bash
# Patch 30 identity test (docs/s25_tracer_sass_dump_patch30.ko.md P30-4/P30-5). Runs inside the llmtrace container.
# Traces a tiny torch workload (bf16 matmul + elementwise) three times with ACTIVE_FROM_START=1:
#   a) the pre-patch tracer backup, b) the patched tracer without SASS_DUMP, c) the patched tracer with SASS_DUMP=1,
# then compares the decompressed trace contents and kernel lists (a == b == c) and checks that only c has traces/sass/.
# Usage (inside the container): s25_tracer_identity_test.sh <out dir> <backup tracer .so>
set -u
OUT=$1; OLD=$2
T=/accel-sim/accel-sim-framework/util/tracer_nvbit/tracer_tool
NEW=$T/tracer_tool.so
mkdir -p $OUT
cat > $OUT/tiny.py <<'EOF'
import torch
torch.manual_seed(0)
a = torch.randn(256, 8192, device="cuda", dtype=torch.bfloat16)
w = torch.randn(1024, 8192, device="cuda", dtype=torch.bfloat16)
y = a @ w.t()
z = torch.nn.functional.silu(y) * y
s = z.float().sum()
torch.cuda.synchronize()
print("tiny done", float(s))
EOF
run() {  # $1 = tag, $2 = tracer .so, $3 = extra env
  local d=$OUT/$1; rm -rf $d; mkdir -p $d
  (cd $d && env CUDA_INJECTION64_PATH=$2 ACTIVE_FROM_START=1 USER_DEFINED_FOLDERS=1 TRACES_FOLDER=$d NOBANNER=1 $3 \
     /venv/bin/python $OUT/tiny.py > $d/run.log 2>&1; echo "EXIT=$?" >> $d/run.log)
  (cd $d/traces && for f in kernel-*.trace.xz; do echo "$f $(xz -dc $f | md5sum | cut -c1-32)"; done; echo "kernelslist $(md5sum kernelslist_ctx_* | cut -c1-32)") > $d/digest.txt
  echo "$1: $(grep -c '^kernel-' $d/digest.txt) kernels, exit $(tail -1 $d/run.log), sass dir: $([ -d $d/traces/sass ] && echo yes || echo no)"
}
run a $OLD ""
run b $NEW ""
run c $NEW "SASS_DUMP=1"
echo "== digests a==b: $(cmp -s $OUT/a/digest.txt $OUT/b/digest.txt && echo SAME || echo DIFF)   a==c: $(cmp -s $OUT/a/digest.txt $OUT/c/digest.txt && echo SAME || echo DIFF)"
if [ -d $OUT/c/traces/sass ]; then
  echo "== sass files: $(ls $OUT/c/traces/sass/*.sass | wc -l), index lines: $(wc -l < $OUT/c/traces/sass/index.tsv)"
  f=$(ls -S $OUT/c/traces/sass/*.sass | head -1); echo "largest: $f ($(wc -l < $f) lines)"; head -5 $f | cut -c1-120
  echo "predicated lines: $(grep -c '@P' $f), constant-bank operands: $(grep -c 'c\[0x' $f)"
  # P30-5: every static PC of the largest traced kernel exists in the matching sass file
  python3 - $OUT/c/traces <<'PY'
import glob, lzma, os, re, sys
d = sys.argv[1]
idx = {l.split('\t')[1]: l.split('\t')[0] for l in open(os.path.join(d, 'sass', 'index.tsv')).read().splitlines()}
big = max(glob.glob(os.path.join(d, 'kernel-*.trace.xz')), key=os.path.getsize)
pcs = set(); name = None
with lzma.open(big, 'rt') as fh:
    for line in fh:
        if line.startswith('-kernel name'): name = line.split('=', 1)[1].strip()
        t = line.split()
        if len(t) > 5 and re.fullmatch(r'[0-9a-f]{4,}', t[4] if t[0].isdigit() else t[0]): pcs.add(t[4] if t[0].isdigit() else t[0])
n = idx.get(name)
offs = set(l.split()[0] for l in open(os.path.join(d, 'sass', f'func-{n}.sass')) if not l.startswith('-')) if n is not None else set()
print(f'P30-5 kernel {os.path.basename(big)} {name[:50]} -> func-{n}.sass: trace PCs {len(pcs)}, in sass {len(pcs & offs)} ->', 'PASS' if pcs and pcs <= offs else 'FAIL')
PY
fi
