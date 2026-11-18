# mem128_s32_20261118: changes from `mem128_s31_20261111`

Started 2026-11-18. Directories: `sim/{ref_mcm,base_mcm,ptwhop0_mcm}`, `validate/{ref_mcm,base_exact}`. Every
`gpgpusim.config` differs from the s31 one in the lines below only.

## DRAM: 768 GB/s per chiplet (was 512)

| option | s31 | s32 | why |
|---|---|---|---|
| `-gpgpu_clock_domains` | `1395:1395:1395:1000` | `1395:1395:1395:1500` | DRAM clock 1.0 -> 1.5 GHz |
| `-gpgpu_dram_buswidth` / `-gpgpu_dram_burst_length` / `-dram_data_command_freq_ratio` | 2 B / 16 / 12 | 8 B / 4 / 2 | written as in Accel-Sim 2.0 HBM configs (H100: 8 / 4 / 2, "HBM is DDR"). Still 32 B per column command; burst / ratio = 2 cycles = tCCD (s31: 16 / 12 = 1). Burst / ratio sets the read-to-precharge and read/write turnaround times (`dram.cc`, `gpu-sim.h`); 8 B / 4 / 2 and 2 B / 16 / 8 give identical results (GEMV check) |
| `-gpgpu_dram_timing_opt` RRD:RCD:RAS:RP:RC:CL:WL:CDLR:WR:RTPL | 4:14:34:14:48:14:5:6:12:5 | 6:19:45:19:50:19:6:8:21:8 | Accel-Sim 2.0 SM90_H100 (HBM3) cycle counts, used at 1.5 GHz (see below) |
| `-gpgpu_dram_timing_opt` nbk, CCD, nbkgrp, CCDL | 16, 2, 4, 4 | unchanged | the column rate follows the data rate |

- Peak per chiplet = 32 channels x 32 B x f_DRAM / tCCD = 512 GB/s at 1 GHz, 768 GB/s at 1.5 GHz (3.07 TB/s in all, the
  MCM-GPU ISCA'17 design point).
- Inter-chiplet limit per chiplet: ICP egress 625 GB/s, ring link 614 GB/s per direction. With 512 the link was larger
  than the DRAM; with 768 it is 0.81x the DRAM, as in MCM-GPU-style studies (link <= DRAM).
- The intra-chiplet crossbar port width (14 flits) was already sized for 768 GB/s (width ~= DRAM per chiplet / 55.8).
- Unchanged: `-dram_latency` (core cycles), address mapping, DRAM scheduler settings.
- Row and latency timings: the cycle counts of Accel-Sim 2.0's SM90_H100 config (HBM3), read at this 1.5 GHz clock:
  RCD / RP / CL 19 = 12.7 ns, RAS 45 = 30 ns, RC 50 = 33.3 ns, WR 21 = 14 ns. Read at H100's own 3106 MHz they would
  be 6.1 / 14.5 / 16.1 ns, too short for DRAM cells, so 1.5 GHz is the plausible reading. RC 50 < RAS + RP = 64, so
  RC never binds (as in the 2.0 config). Before this, s32 used the s31 values x1.5 (6:21:51:21:72:21:8:9:18:8, origin
  not recorded). Neither is a JEDEC data-sheet value; replace when one is chosen.
- Measured (GEMV, fixed placement): with this config as is, kernel 1 (local streaming) is 5.8% faster than with the
  earlier s32 timings (351 -> 373 GB/s per chiplet, row switching was the limit); with the block-5 DRAM scheduler
  settings the two timings differ by 0.09% in total. Unloaded DRAM load latency (pointer chase, ideal TLB):
  590.7 core cycles, 6.1 less than before (RTX 3090: 501).
- CCD 2 and CCDL 4 are not taken from H100: its CCD 1 would double the peak to 1,536 GB/s per chiplet, and its
  CCDL / CCD = 6 would cut the bandwidth when requests crowd a few bank groups. CCDL = 2 x CCD keeps the peak with two
  of the four bank groups busy.
- The structure matches one HBM3 stack at 6.0 Gb/s per chiplet: 32 pseudo-channels of 32 bits = the 32 channels here,
  CK 1.5 GHz, a 32 B burst every 2 CK (tCCD 2), 4 bank groups x 4 banks.

## L2: slices hashed inside each chiplet

| option | s31 | s32 |
|---|---|---|
| `-gpgpu_memory_partition_indexing` | 0 (slice = address bits) | 7 (IPoly over the 64 slices of a chiplet) |
| `-gpgpu_mem_hash_chiplets` | (unset) | 4 |

- The chiplet bits of the address are not hashed, so the chiplet that owns a page (first-touch placement) does not change.
  The simulator checks this at startup for every chiplet.
- With 0, requests 4 KB or 8 KB apart (GEMM weight rows) went to 2-4 of a chiplet's 64 slices. Real GPUs hash the slice
  index; Accel-Sim's RTX 3070 config uses IPoly (indexing 2), which here would also hash the chiplet bits.

## L1 TLB lookup latency: 20 cycles for every page size

| option | s31 | s32 |
|---|---|---|
| `-gpgpu_l1_tlb_4k_lookup_latency` | 20 | 20 |
| `-gpgpu_l1_tlb_64k_lookup_latency` | 10 | 20 |
| `-gpgpu_l1_tlb_2m_lookup_latency` | 10 | 20 |

- 2026-11-18 user setting: one L1 TLB lookup latency for all page sizes (s31 had 20 for 4KB only). L2 TLB stays 80.
- Still within the `-gpgpu_l1d_vipt` condition (every L1 TLB lookup latency <= `-gpgpu_l1_latency` 39).

## Kernel hand-over: every kernel waits the launch latency

| option | s31 | s32 |
|---|---|---|
| `-gpgpu_kernel_rebind` | (unset = 0) | 1 |

- 0 (code default, kept so s31 results reproduce): a core whose kernel finished goes straight to the next kernel the
  trace framework creates, without `select_kernel()`, so that kernel's CTAs on such cores skip
  `-gpgpu_kernel_launch_latency` (5000). Upstream does the same by accident (the finished kernel's address is reused).
  With 1 an idle core picks its kernel through `select_kernel()`, and every kernel waits the latency, as the upstream
  comment intends.
- Measured (LLM decode window, 30 kernels, `mem128_s32_20261118/sim/ref_mcm`): 2,365,613 -> 2,421,944 cycles
  (+2.4%). The three activation kernels take about 5,200 -> 10,200 cycles each (under 0 their CTAs skipped the
  5,000); the rest is GEMM timing shifts (-6% to +11% per kernel).
- With the s31 config (no L2 slice hash) the same window under 1 trips the deadlock detector: requests crowd two L2
  slices and no instruction completes for 50,000 cycles. With the detector off it finishes (+6.0%), so it is not a
  hang. Use 1 with s31 configs only with `-gpgpu_deadlock_detect 0` and a cycle cap.

## Kept

- `-gpgpu_l1_latency 39`: same as Accel-Sim 2.0 RTX 3070 and the same code path. A dependent load chain step measures
  39 + 11 cycles (4 to enter the L1 latency queue, 7 for the address IMAD) against 38.9 on an RTX 3090; recorded, not changed.

## To revisit before the next experiments

This config is a start, not final. Before new experiment runs, go through:

- DRAM timings from a data sheet (JEDEC HBM3 or a vendor sheet) instead of the Accel-Sim 2.0 H100 cycle counts.
- Block-5 DRAM scheduler settings, not in this config yet: `-gpgpu_frfcfs_dram_sched_queue_size 128
  -dram_bnk_indexing_policy 1 -dram_dual_bus_interface 1` (GEMV steady-state DRAM use 0.48 -> 0.90). Turn them on
  together with `-gpgpu_ldst_bypass_send_per_bank 1` (and `-gpgpu_ldst_banked_response 1`): alone they make LLM
  GEMM 7.2% slower, with the per-bank bypass send it is back at the old level.
- Core options in the repo that are off by default: `-gpgpu_l1d_vipt 1`, `-gpgpu_ldst_replay_first 1`,
  `-gpgpu_l1_tlb_lookups_per_cycle 4`, `-gpgpu_alu_ready_queue 1`, `-gpgpu_ldst_gate_skip_shared 1`,
  `-gpgpu_ldst_gate_release_at_xlat 1`. See `simulator/README.md` (Comparing model variants, Model notes).
- Kept on purpose, decide again only with new data: `-gpgpu_l1_latency 39`, L1 / L2 TLB latency 20 / 80,
  `-gpgpu_l2_tlb_starts_per_cycle 0`.
- Inter-chiplet ring: wire capacity per link and its ratio to DRAM per chiplet (0.81 now).
- Baselines must be measured again with these settings, with a fixed placement map
  (`-gmmu_placement_load <map> -gmmu_placement_exact 1`) for every model A/B comparison.
