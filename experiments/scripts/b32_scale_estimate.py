# Scale estimate for batch 32, input 256, output 3 (prefill 1 step + decode 2 steps), from measured B=1 decode data.
# Model (explained in docs/b32_scale_estimate.ko.md):
#   * GEMM warp instructions grow with the number of 64-row M tiles. B=1 decode already pays one full M tile:
#     HMMA 6.25M warp/layer x 2048 MAC = 1.28e10 MAC of tensor-core work for 2.18e8 useful MAC (1/58.7 used).
#     Decode B=32 (M=32) therefore needs 1x (same 64-row tiles) to 2x (16-row tiles as in the qkv CUTLASS kernel).
#   * Prefill (M=8192) is bounded by useful MAC / 2048 per HMMA, divided by the HMMA share of all warp
#     instructions: 23.6% as measured in B=1 decode (upper estimate) to 45% for large-tile kernels (lower estimate).
#   * Attention and element-wise kernels scale with the number of tokens they touch.
# Everything measured comes from traces/s5_decode_full_S256 (tracer stats, reported warp instructions).
import glob
import re

stats = {}
for ln in open(glob.glob("traces/s5_decode_full_S256/traces/stats_ctx_*")[0]).read().splitlines()[1:]:
    p = ln.split(",")
    stats[int(re.search(r"kernel-(\d+)-", p[0]).group(1))] = int(p[-1])
layer = {k: stats[k] for k in range(1348, 1360)}
gemm = sum(layer[k] for k in (1349, 1350, 1355, 1357, 1359))           # qkv (+splitK reduce), o, gate_up, down
flash = layer[1353] + layer[1354]
elem = sum(layer[k] for k in (1348, 1351, 1352, 1356, 1358))
lm_head_b1 = stats[1735]
gate_up = layer[1357]
lanes = 29.6                                                             # thread / warp, V3 decode measurement
hmma_share = 12500992 / sum(stats[k] for k in range(1348, 1372))        # 2 layers, s6_check_traces_L2kernels.log
mac_per_token = 4096 * 6144 + 4096 * 4096 + 4096 * 28672 + 14336 * 4096  # qkv, o, gate_up, down
B, S, OUT, L = 32, 256, 3, 32
bytes_per_warp = 5951377288 / 994394914                                 # full-decode capture
tracer_s_per_warp = 3398 / 994394914
sim_warp_per_s = 24745655 / 5399                                         # L1 ref_batch4, 16 threads
print(f"B=1 decode layer: {sum(layer.values())} warp (GEMM {gemm}, flash {flash}, elementwise {elem}); "
      f"HMMA share {hmma_share:.3f}; lm_head gemv {lm_head_b1}")


def row(name, warp_lo, warp_hi, kernels):
    t_lo, t_hi = warp_lo * lanes, warp_hi * lanes
    print(f"{name:34s} kernels~{kernels:5d} warp {warp_lo:.2e}-{warp_hi:.2e} thread {t_lo:.2e}-{t_hi:.2e} "
          f"(cap x{t_lo / 1e10:.2f}-{t_hi / 1e10:.2f}) trace {warp_lo * bytes_per_warp / 1e9:.1f}-{warp_hi * bytes_per_warp / 1e9:.1f} GB "
          f"tracer {warp_lo * tracer_s_per_warp / 3600:.1f}-{warp_hi * tracer_s_per_warp / 3600:.1f} h "
          f"sim {warp_lo / sim_warp_per_s / 3600:.1f}-{warp_hi / sim_warp_per_s / 3600:.1f} h")
    return warp_lo, warp_hi


# decode, batch 32: 32 tokens per step, attention over ~257-258 cached tokens per request
dec_layer = (gemm * 1 + (flash + elem) * B, gemm * 2 + (flash + elem) * B)
lm_head_b32 = (gate_up * 128256 / 28672, gate_up * 2 * 128256 / 28672)
dec_step = (L * dec_layer[0] + lm_head_b32[0], L * dec_layer[1] + lm_head_b32[1])
# prefill, batch 32: 8192 tokens
tok = B * S
hmma = tok * mac_per_token / 2048
pre_gemm = (hmma / 0.45, hmma / hmma_share)
attn_mac = B * 32 * (S * (S + 1) / 2) * 128 * 2                          # causal QK^T and PV per head
pre_attn = attn_mac / 2048 / hmma_share
pre_layer = (pre_gemm[0] + elem * tok, pre_gemm[1] + pre_attn + elem * tok)
pre_step = (L * pre_layer[0] + lm_head_b32[0], L * pre_layer[1] + lm_head_b32[1])
print()
row("decode, 1 layer (12 kernels)", *dec_layer, 12)
row("decode, 1 step (401 kernels)", *dec_step, 401)
row("decode, 2 steps", 2 * dec_step[0], 2 * dec_step[1], 802)
row("prefill, 1 layer (10 kernels)", *pre_layer, 10)
row("prefill, 1 step (337 kernels)", *pre_step, 337)
row("total: prefill + 2 decode", pre_step[0] + 2 * dec_step[0], pre_step[1] + 2 * dec_step[1], 1139)
kv_per_layer = B * (S + OUT) * 2 * 8 * 128 * 2
print(f"\nKV cache: {kv_per_layer / 2**20:.1f} MiB per layer, {kv_per_layer * L / 2**30:.2f} GiB total "
      f"(vLLM blocks: {B * -(-(S + OUT) // 16)} x 16 tokens)")
print(f"prefill FFN intermediate (gate_up output): {tok * 28672 * 2 / 2**20:.0f} MiB")
# for comparison: batch 1, S=256 prefill layer with the same model (the README 1.3 flop x 0.437 value was 2.45e10 thread)
hmma1 = S * mac_per_token / 2048
print(f"B=1 prefill layer by the same model: warp {hmma1 / 0.45 + elem * S:.2e}-{hmma1 / hmma_share + elem * S:.2e}, "
      f"thread {(hmma1 / 0.45 + elem * S) * lanes:.2e}-{(hmma1 / hmma_share + elem * S) * lanes:.2e}")
