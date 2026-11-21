# mem256_s34_20261121: changes from `mem256_s33_20261120`

Created 2026-11-21. Same directories and arms as s33. Every `gpgpusim.config` differs from its s33 counterpart in the
4 option lines below (plus their comments) and nothing else; `trace.config` and the `.icnt` file are unchanged.
Needs simulator v5. All four model actual hardware or driver behaviour; s33 left them at 0.

| option | s34 | what it changes | evidence | measured (64KB unless noted; fixed placement maps) |
|---|---|---|---|---|
| `-icnt_ring_credit` | 1 | A ring link starts a packet only while its wire holds fewer than width x (max hop latency + 2) packets (2,585 here); a local injection keeps one link slot free (bubble flow control) | Real inter-chip links send only while the receiver has buffer (credits); without it a blocked destination let the wire grow without bound (GUPS 2MB: system OOM). The cap is never reached while deliveries flow | identical where the wire never filled (SSSP, BFS, BC, GUPS 4KB/64KB); Mixtral b1 decode +1.9 % (fused_moe +7.8 % / -9.2 %), SpMV -0.07 %; GUPS 2MB now bounded (5.7 GB) |
| `-gpgpu_l1_tlb_mshr_wait_age` | 1 | MSHR-full L1 TLB lookups take freed entries before new lookups, oldest instruction first | Age-ordered allocation is the starvation-free hardware norm; the s33 order (newest first) came from the code (`push_front`) and starved requests up to 184,603 cycles | SSSP main kernel +28.5 % (64KB), +3.5 % or more (4KB), -17 % (2MB); GUPS IPC +11 % (4KB), +0.6 % (64KB), -1.2 % (2MB); others unchanged |
| `-gpgpu_l2_tlb_mshr_age_order` | 1 | A free L2 TLB MSHR entry goes to the request that has waited longest (ticket queue; the oldest min(free, waiters) hold reservations) instead of the first SM in a fixed order | Same principle; the fixed SM order let the first SMs of a chiplet take every freed entry | GUPS 64KB IPC +19 %, SpMV +1.4 %, Mixtral +1.1 % cycles; no L2 MSHR waits in BFS, BC, SSSP 4KB/2MB |
| `-gpgpu_shmem_per_cta_reserve` | 1024 | Kernels using shared memory count smem + 1 KB per CTA (CTA-per-SM limit, L1D carveout) | CUDA reserves 1 KB per block from sm_80 (`cudaDevAttrReservedSharedMemoryPerBlock`) | across 2,326 LLM kernels only the Llama-70B b1 wmma GEMM changes (6 -> 5 CTAs per SM) |

## Not changed (options available, left at their defaults)

- `-gpgpu_frfcfs_age_cap` (FR-FCFS starvation cap, new in v5): 0, as in Accel-Sim. Measured with 4000: bit identical on GUPS,
  SSSP, SpMV, BFS, BC (no DRAM wait reached 4,096 cycles); Mixtral b1 decode +0.18 % cycles, translation (PTW) -8.3 %.
- Instruction and memory latencies: Accel-Sim's values. Against an RTX 3090 pointer-chase microbenchmark, an L1 hit is 49.7
  cycles (real 38.9) and an L2 hit 260 (real 253); Accel-Sim 2.0 gives the same L1 value.

## To revisit

- 2 MB pages: walks are about 30 times fewer than at 64 KB but average about 20,000 cycles each (some over 130,000) in LLM
  decode; not caused by DRAM starvation (unchanged with the cap on). Cause open; experiments use 64 KB.
- 2 MB first touch: simultaneous first touches resolve in chiplet order, so most 2 MB data pages land on chiplet 0 (GUPS 95 %).
- Ring credit cap value: the cap is the smallest that never binds while deliveries flow; real link buffers may be smaller.
