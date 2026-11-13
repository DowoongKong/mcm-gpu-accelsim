# LLM decode experiments (dense and MoE)

Scripts and simulator configurations used with `../simulator/` for Qwen3-32B, Llama-3.3-70B-Instruct (dense) and
Qwen3-30B-A3B, Mixtral-8x7B-v0.1 (MoE) decode on a 4-chiplet MCM GPU.

| path | contents |
|---|---|
| `scripts/` | trace capture (vLLM 0.19.1 + NVBit), prefill placement prediction, simulation kits and arms, context sweep, analysis |
| `configs/` | simulator configs (`mem128_s31_20261111`, `mem256_s31_20261112`: `sim/{ref_mcm,base_mcm,ptwhop0_mcm}`, `validate/`) |
| `tracer/` | NVBit tracer patch against Accel-Sim `3016c65`, capture container Dockerfile and package list |

| step | entry points in `scripts/` |
|---|---|
| trace capture | `h1_capture_v2.py`, `vllm_setup.py`, `h1_capture_v2_chain_c_v3.sh`, `h1_capture_v2_chain_c_v4.sh` |
| prefill placement prediction | `s22_ft_predict2.py`, `s26_make_placement_map.py`, `s34_build_prefill_map.sh`, `s41_build_map_noref.sh` |
| simulation kits / arms | `make_inject_hash_kit_s31.sh`, `h1_make_decode_arms_s31.sh` |
| context sweep | `s42_post_capture.sh`, `s42_sweep_gate.py` |
| analysis | `h1_slice_four_metrics.py`, `s37_sharing_class.py`, `s43_struct_remote.py`, `s45_tlb_reuse_distance.py` |

Paths in the scripts refer to the original setup (`/home/user/accel-sim-docker/llm_exp` on the host, `/accel-sim/llm_exp` in
the containers). Traces, model weights and simulation outputs are not included.
