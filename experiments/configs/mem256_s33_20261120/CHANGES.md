# mem256_s33_20261120: changes from `mem256_s32_20261118`

Created 2026-11-20. Same as `mem128_s33_20261120` except the memory size (256 GB = 64 GB per chiplet, one more row
bit in `-gpgpu_mem_addr_mapping`), exactly as between the two s32 sets. Directories: `sim/{ref_mcm,base_mcm,ptwhop0_mcm}`, `validate/{ref_mcm,base_exact}` (same arm
layout as s32). Every `gpgpusim.config` differs from its s32 counterpart in the 20 option lines below and nothing else
(checked option by option; comments were rewritten to match the values). 17 of them change the model; 3 only write
out the code default (`-gpgpu_l1_tlb_mshr_wait_fifo 0`, `-gpgpu_l2_tlb_starts_per_cycle 0`, `-icnt_icp_port_width 0`). `trace.config` and
`config_ampere_islip.icnt` are unchanged; the `.icnt` file is not read with `-network_mode 2`.

Needs a simulator with the options of branch `v4` (the s31 and s32 configs leave them all at 0 and reproduce).
Measured effects below were taken one option at a time against the s32 behaviour with a fixed placement map
(`-gmmu_placement_load <map> -gmmu_placement_exact 1`); their combined effect is measured on this config.

## SM model options (all 0 in s32)

| option | s33 | what it changes | evidence | measured alone |
|---|---|---|---|---|
| `-gpgpu_trace_atomic_no_stall` | 1 | A warp no longer stops issuing until every reply of its atomics is back; only readers of an ATOM result wait (scoreboard), and the warp exits after the last reply | The stall exists for PTX functional execution (its own comment says it is not needed for timing). PTX `red` returns nothing and is relaxed; SASS waits per register through dependency barriers (Jia et al. 2018). Accel-Sim 2.0 has the same stall | GUPS IPC +6.6 %, SSSP -13.2 % and BC -15.8 % cycles (finished kernels); no change without atomics |
| `-gpgpu_trace_rz_no_reg` | 1 | R255 (RZ) is no longer register 256: no false scoreboard waits after an instruction writing RZ, no register bank read for RZ | RZ reads as zero and discards writes. Accel-Sim 2.0 maps it to 256 as well | Mixtral -0.4 %, SSSP -3.4 %, others < 1 % |
| `-gpgpu_membar_wait_stores` | 1 | MEMBAR waits for the warp's store acks too | `__threadfence()` orders earlier writes before later ones; the upstream comment says "wait for all pending writes till they are acked" but the code checked loads only. 2.0 the same | LLM split-K GEMM tails only |
| `-gpgpu_strong_store_bypass_l1` | 1 | `.STRONG.GPU` / `.BYPASS` stores skip the L1D like such loads | GPU-scope strong accesses are performed at the L2; Accel-Sim 2.0 (`0db0445`) does this | split-K semaphore stores only |
| `-gpgpu_alu_ready_queue` | 1 | ALU results leave from an in-order ready queue (Accel-Sim 2.0) instead of a shift register that adds a cycle | Accel-Sim 2.0 model; dependent FFMA chain 6.9 -> 5.9 cycles (RTX 3090: 4.2) | reduction -3.0 %, Mixtral -1.1 %, LLM GEMM -1.0 % |
| `-gpgpu_ldst_bypass_send_per_bank` | 1 | L1D-bypassing accesses of one instruction are sent up to `-gpgpu_l1_banks` per cycle | Accel-Sim 2.0 (`987a23f8`) | LLM GEMM -6.6 % with the DRAM settings below |
| `-gpgpu_ldst_banked_response` | 1 | Up to `-gpgpu_l1_banks` responses taken and bypass responses written back per cycle | Accel-Sim 2.0 | LLM GEMM -1.0 % with the DRAM settings below |
| `-gpgpu_ldst_gate_skip_shared` | 1 | Shared-memory accesses do not take the per-warp translation gate | They need no translation | <= 0.5 % where it acts (fewer cycles) |
| `-gpgpu_ldst_gate_release_at_xlat` | 1 | A LOAD/STORE releases the gate once every page it touches is translated, not when it leaves the LSU dispatch | No hardware reason to hold a warp whose translation is done while its accesses drain | Mixtral -2.7 % (ideal TLB, direct effect); others about 0 |
| `-gpgpu_ldst_replay_first` | 1 | A dispatch slot freed in a cycle goes to a translated parked instruction before a newly issued one | Older work first | -3.3 % to +1.2 % |
| `-gpgpu_ldst_replay_skip_stale` | 1 | A replay id whose warp has nothing parked no longer uses up the cycle's replay | Stale ids are a model artefact (up to 0.09 % of SM-cycles); a stale pop lets a newly issued instruction take the slot first, so the waiting replay loses more than one cycle | Mixtral -5.8 % (fused_moe -9 % and -6 %), SSSP -3.4 % (finished kernels) |

## Translation options (all 0 in s32)

| option | s33 | what it changes | evidence | measured alone |
|---|---|---|---|---|
| `-gpgpu_l1d_vipt` | 1 | The L1 TLB lookup overlaps the L1D access | GPUs index the L1 virtually and tag it physically; pointer-chase L1 hit 59.7 -> 49.7 cycles (= ideal TLB). Valid because every L1 TLB latency (20) <= `-gpgpu_l1_latency` (39) | |
| `-gpgpu_l1_tlb_lookups_per_cycle` | 4 | Up to 4 new L1 TLB lookups per cycle (pipelined, each 20 cycles) instead of unlimited | Finite lookup bandwidth | Mixtral, GEMV unchanged; GUPS +0.1 % |
| `-gpgpu_l1_tlb_mshr_wait_fifo` | 0 | none: written out (the s32 order, most recently blocked first) | Oldest-first per lookup bounds the worst wait (GUPS 184,603 -> 52,609 cycles) but raised the mean wait 3.7x in SSSP: an instruction proceeds only when all its pages are translated, and FIFO interleaves the pages of many instructions. With the other s33 options, SSSP's main kernel was 27 % slower with FIFO (706,565 vs 554,726 cycles); GUPS +0.5 % IPC. An instruction-age order is the candidate model; until it is analysed the s32 order is kept | alone on s32: GUPS -1.5 %, SSSP -0.3 % |
| `-gpgpu_l2_tlb_starts_per_cycle` | 0 | none: written out (unlimited, the default) | Kept on purpose | |

## Interconnect options (unset in s32)

| option | s33 | what it changes | evidence | measured alone |
|---|---|---|---|---|
| `-icnt_xbar_input_limit` | 1 | Each SM and L2 input of a chiplet crossbar sends at most `-icnt_xbar_port_width` (14) flits per cycle; the ring ingress is not limited | A port has one width in both directions (intersim2 and Accel-Sim 2.0's LocalInterconnect also bound inputs) | Mixtral -1.6 %, GUPS -0.2 % |
| `-icnt_icp_port_width` | 0 | none: written out (the chiplet's send port to the ring has the crossbar port width) | Kept; set it only for inter-chiplet bandwidth sweeps | |

## DRAM scheduler

| option | s32 | s33 | why |
|---|---|---|---|
| `-gpgpu_frfcfs_dram_sched_queue_size` | 64 | 128 | With 64 entries and linear banks the steady-state DRAM use of a streaming GEMV was 0.48 (RTX 3090: 0.90); these three settings give 0.90 |
| `-dram_bnk_indexing_policy` | 0 | 1 | Bank index XORed with higher address bits, so strided streams spread over banks |
| `-dram_dual_bus_interface` | 0 | 1 | HBM has separate row and column command buses |

Turn these on only together with `-gpgpu_ldst_bypass_send_per_bank 1` (and `-gpgpu_ldst_banked_response 1`):
alone they make the LLM GEMM 7.2 % slower; with the per-bank bypass send it is 6.6 % faster.

## Kept (unchanged from s32)

- 768 GB/s DRAM per chiplet with the Accel-Sim 2.0 H100 cycle counts at 1.5 GHz (CCD 2, CCDL 4); not JEDEC values.
- L2 slice hash inside each chiplet (`-gpgpu_memory_partition_indexing 7`), L1 TLB latency 20 for all page sizes,
  L2 TLB latency 80 (includes the SM <-> L2 TLB round trip), `-gpgpu_kernel_rebind 1`.
- `-gpgpu_l1_latency 39` (same as Accel-Sim 2.0 RTX 3070; a dependent L1-hit load chain step measures 39 + 11 cycles
  against 38.9 on an RTX 3090).
- L2 capacity: `S:32:128:24` = 96 KB per slice, 24 MB in all, 6 MB per chiplet. The s31/s32 comment ("16 MB,
  16 ways, MCM-GPU Table 3") did not match the value and is corrected; the value is the one every result used.

## Comments corrected (no value change)

- Header: the s32 file still described an August local-PTW experiment ("baseline_4k_bfs256M", "24 runs").
- Crossbar port: the s32 comment gave width 5 for HBM2; the value is 14 (781 GB/s raw, 625 GB/s of 32 B data).
- `-gpgpu_page_size_policy`: the s32 comment said it came from an overlay; it is set in the file (1 = 64 KB).

## Validation

- Simulator branch `v4`: every new option at 0 reproduces s32 bit for bit; each option at 1 reproduces its
  test-tree run (`code_review` doc `20261120_all_sources_result_changing_options_v4_implementation_and_verification`).
- This config: `validate/ref_mcm` and `validate/base_exact` give the canonical digests of the v4 gate;
  1 thread = 8 threads.

## To revisit

- DRAM timings from a data sheet (JEDEC HBM3 or a vendor sheet).
- Inter-chiplet ring: wire capacity per link (unbounded now) and the ring ingress width; decide together.
- LDGSTS group buffer memory (about 5 GB in large LLM prefill GEMMs); a result-neutral fix is planned before MoE runs.
