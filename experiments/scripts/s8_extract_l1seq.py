# Step 8, continuous run L1seq (prefill -> decode1 -> decode2, layer 0 only, 87 kernels in original order):
# per-piece address-translation metrics from the per-kernel cumulative stat dumps in run.log.
# Parsing and metric definitions are the same as s6_compare_layers.py (blocks / dump / derived / minus).
# Pieces are kernel-index ranges (1-based) of kernelslist_L1seq.g; P-O, D1-O, D2-O each have a head and a tail range.
# Usage: s8_extract_l1seq.py [--traces <trace dir>] <L1seq dir> [<P-L2 dir>]   (partial run.log handled per kernel; * = arm not finished)
import os, re, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# reuse the s6 parsing/metric functions without running its main body
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s6_compare_layers.py")).read().split("\ndata = {}")[0]
ns = {}; exec(compile(src.replace("E = sys.argv[1]", "E = ''"), "s6_compare_layers.py", "exec"), ns)
blocks, dump, derived, minus = ns["blocks"], ns["dump"], ns["derived"], ns["minus"]

ORDER = ["P-O", "P-L1", "D1-O", "D1-L1", "D2-O", "D2-L1"]
# Kernel-index ranges of each piece inside kernelslist_L1seq.g. Derived from the piece lists when the trace
# directory is given (--traces <dir with kernelslist_L1seq.g and kernelslist_<piece>.g>), otherwise the
# DeepSeek-Llama-8B s8 ranges below.
PIECES = {"P-O": [(1, 9), (20, 27)], "P-L1": [(10, 19)], "D1-O": [(28, 36), (50, 57)], "D1-L1": [(37, 49)],
          "D2-O": [(58, 66), (80, 87)], "D2-L1": [(67, 79)]}
if "--traces" in sys.argv:
    _i = sys.argv.index("--traces"); T = sys.argv[_i + 1]; del sys.argv[_i:_i + 2]
    seq = [l for l in open(os.path.join(T, "kernelslist_L1seq.g")).read().splitlines() if not l.startswith("MemcpyHtoD")]
    idx = {l: i + 1 for i, l in enumerate(seq)}
    def _ranges(ks):
        pos = sorted(idx[k] for k in ks); out = []; s0 = p0 = pos[0]
        for q in pos[1:]:
            if q != p0 + 1: out.append((s0, p0)); s0 = q
            p0 = q
        out.append((s0, p0)); return out
    PIECES = {p: _ranges([l for l in open(os.path.join(T, f"kernelslist_{p}.g")).read().splitlines() if not l.startswith("MemcpyHtoD")]) for p in ORDER}
ARMS = ("ref_batch4", "base_pcc", "ptwhop0_pcc")

def per_kernel(dumps):
    out = []
    prev = None
    for d in dumps:
        out.append(d if prev is None else minus(d, prev)); prev = d
    return out

def piece_sum(pk, ranges):
    tot = None
    for lo, hi in ranges:
        for i in range(lo, hi + 1):
            if i - 1 >= len(pk): return None   # not reached yet
            tot = dict(pk[i - 1]) if tot is None else {k: tot[k] + pk[i - 1][k] for k in tot}
    return tot

def load(d):
    try:
        ok = open(os.path.join(d, "exit_code.txt")).read().startswith("EXIT=0")
    except FileNotFoundError:
        ok = False
    log = open(os.path.join(d, "run.log"), errors="replace").read()
    bl = blocks(log)
    dumps = []
    for b in bl:
        try: dumps.append(dump(b))
        except Exception: break   # last block may be incomplete while running
    return ok, per_kernel(dumps)

def show(title, rows):
    print(f"== {title}")
    print(f"{'arm':12s} {'piece':6s} {'kernels':>7s} {'warp':>12s} {'cycles':>11s} {'L2miss':>10s} {'walks':>10s} {'MPKI':>7s} {'IPC':>7s} {'AT':>8s} {'PTW':>8s} {'rr':>7s}")
    for arm, piece, nk, t in rows:
        if t is None: print(f"{arm:12s} {piece:6s} {nk:7d} (not reached)"); continue
        x = derived(t)
        print(f"{arm:12s} {piece:6s} {nk:7d} {t['warp']:12d} {t['cycle']:11d} {t['l2_miss']:10d} {t['walks']:10.0f} {x['MPKI']:7.2f} {x['IPC']:7.1f} {x['AT']:8.1f} {x['PTW']:8.1f} {x['rr']:7.4f}")

X = sys.argv[1]
rows = []
for a in ARMS:
    d = os.path.join(X, a)
    if not os.path.exists(os.path.join(d, "run.log")): continue
    ok, pk = load(d)
    tag = a + ("" if ok else "*")   # * = not finished (EXIT=0 missing)
    for p in ORDER:
        nk = sum(hi - lo + 1 for lo, hi in PIECES[p])
        rows.append((tag, p, nk, piece_sum(pk, PIECES[p])))
    full = piece_sum(pk, [(1, len(pk))]) if pk else None
    rows.append((tag, "ALL", len(pk), full))
show(f"L1seq per piece ({X})", rows)
if len(sys.argv) > 2:   # P-L2 (prefill layers 0-1 standalone, 20 kernels): per-layer halves for the ~2x check
    Y = sys.argv[2]; rows = []
    for a in ARMS:
        d = os.path.join(Y, a)
        if not os.path.exists(os.path.join(d, "run.log")): continue
        ok, pk = load(d); tag = a + ("" if ok else "*")
        rows.append((tag, "L0", 10, piece_sum(pk, [(1, 10)]))); rows.append((tag, "L1", 10, piece_sum(pk, [(11, 20)])))
        rows.append((tag, "ALL", len(pk), piece_sum(pk, [(1, len(pk))]) if pk else None))
    show(f"P-L2 per layer ({Y})", rows)
