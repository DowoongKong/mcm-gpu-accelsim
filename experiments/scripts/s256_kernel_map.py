# Map every kernel launch of generate() (prefill 1 step + decode 1 step, S=256) to a model section, using only
# the tracer stats file of the step-2 run (one line per launch, names recorded even with the window closed)
# and regions.tsv (hook-measured kernel ranges of layers 0/15/31).
# The layer period is checked, not assumed: every layer block must repeat the kernel-name sequence of the
# hook-measured layer 15 block, and the hook ranges of layers 0/15/31 must fall where the period puts them.
# Usage: s256_kernel_map.py <stats_ctx file> <regions.tsv> <first generate kernel id> <out tsv>
import re
import sys

stats_path, regions_path, gen_first, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
N_LAYERS = 32

names = {}
insts = {}
for ln in open(stats_path).read().splitlines()[1:]:
    parts = ln.split(", ")
    kid = int(re.search(r"kernel-(\d+)-ctx", parts[0]).group(1))
    names[kid] = parts[1]
    insts[kid] = int(parts[-1].split(",")[-1])
gen_last = max(names)

regions = {}
for ln in open(regions_path).read().splitlines()[1:]:
    tag, phase, first, last = ln.split("\t")[:4]
    layer = int(tag.split(".")[2])
    regions[(phase, layer, tag.split(".")[3])] = (int(first), int(last))

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print("CHECK", "PASS" if ok else "FAIL", name, "|", detail)


rows = []   # (first, last, step, section)


def layer_blocks(phase):
    # A layer block runs from its input norm kernel to the kernel before the next layer's input norm.
    # Layer 15's hook ranges give the block layout: norm, attention range, norm, mlp range.
    a0, a1 = regions[(phase, 15, "self_attn")]
    m0, m1 = regions[(phase, 15, "mlp")]
    check(f"{phase}: layer 15 layout is norm + attn + norm + mlp", m0 == a1 + 2, (a0, a1, m0, m1))
    period = m1 - (a0 - 1) + 1
    first_norm = a0 - 1 - 15 * period
    ref = [names[k] for k in range(a0 - 1, m1 + 1)]
    same = 0
    for layer in range(N_LAYERS):
        base = first_norm + layer * period
        seq = [names[k] for k in range(base, base + period)]
        # layer 0 has a plain rms_norm first (no residual yet); every other layer starts with fused_add_rms_norm
        same += seq[1:] == ref[1:] and ("rms_norm_kernel" in seq[0])
    check(f"{phase}: all 32 layer blocks repeat the layer-15 kernel names", same == N_LAYERS,
          f"{same}/32, period {period}")
    for layer in (0, 31):
        base = first_norm + layer * period
        check(f"{phase}: hook ranges of layer {layer} match the period",
              regions[(phase, layer, "self_attn")] == (base + 1, base + a1 - a0 + 1)
              and regions[(phase, layer, "mlp")] == (base + m0 - a0 + 1, base + period - 1),
              (regions[(phase, layer, "self_attn")], regions[(phase, layer, "mlp")]))
    return first_norm, period


p_first, p_period = layer_blocks("prefill")
d_first, d_period = layer_blocks("decode")
p_layers_end = p_first + N_LAYERS * p_period - 1
d_layers_end = d_first + N_LAYERS * d_period - 1

# The kernels after the last layer until the next step's preparation: final norm, logits, sampling.
# The decode step starts with the same preparation sequence the prefill step starts with, which marks the
# boundary: find the decode preparation as the kernels between the prefill tail and the decode layer 0 norm.
p_tail = list(range(p_layers_end + 1, d_first))
d_tail = list(range(d_layers_end + 1, gen_last + 1))
tail_len = len(d_tail)   # the decode tail runs to the end of generate()
check("prefill tail == decode tail kernel names (final norm, logits, sampling)",
      [names[k] for k in p_tail[:tail_len]] == [names[k] for k in d_tail], f"{tail_len} kernels")
p_tail_end = p_layers_end + tail_len
d_prep = list(range(p_tail_end + 1, d_first))
p_prep = list(range(gen_first, p_first))
check("decode prep has the same length as prefill prep", len(d_prep) == len(p_prep), (len(p_prep), len(d_prep)))

rows += [(gen_first, p_first - 1, "prefill", "input preparation + embedding"),
         (p_first, p_layers_end, "prefill", f"32 layers x {p_period} kernels"),
         (p_layers_end + 1, p_tail_end, "prefill", "final norm + logits + sampling"),
         (p_tail_end + 1, d_first - 1, "decode", "input preparation + embedding"),
         (d_first, d_layers_end, "decode", f"32 layers x {d_period} kernels"),
         (d_layers_end + 1, gen_last, "decode", "final norm + logits + sampling")]
covered = sum(r[1] - r[0] + 1 for r in rows)
check("sections cover generate() exactly once", covered == gen_last - gen_first + 1 and
      all(rows[i][1] + 1 == rows[i + 1][0] for i in range(len(rows) - 1)), f"{covered} kernels")

with open(out_path, "w") as f:
    f.write("kernel_id\tstep\tsection\tlayer\tmodule\ttraced_reported_insts\tname\n")
    for first, last, step, section in rows:
        for k in range(first, last + 1):
            layer = module = ""
            if "layers" in section:
                period = p_period if step == "prefill" else d_period
                base = p_first if step == "prefill" else d_first
                layer = (k - base) // period
                off = (k - base) % period
                a0, a1 = regions[(step, 15, "self_attn")]
                m0, _ = regions[(step, 15, "mlp")]
                a_off, a_end, m_off = a0 - (base + 15 * period), a1 - (base + 15 * period), m0 - (base + 15 * period)
                module = ("input_norm" if off < a_off else "self_attn" if off <= a_end
                          else "post_attn_norm" if off < m_off else "mlp")
            f.write(f"{k}\t{step}\t{section}\t{layer}\t{module}\t{insts[k]}\t{names[k]}\n")

for first, last, step, section in rows:
    print(f"{step}\t{first}-{last}\t{last - first + 1}\t{section}")
print("KERNEL MAP", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)})")
