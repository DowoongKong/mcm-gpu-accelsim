# MCM GPU simulator (Accel-Sim / GPGPU-Sim fork)

Trace-driven multi-chip-module GPU simulator with a virtual-memory model (per-chiplet TLBs, page-table walkers, page and
page-table placement).

| part | origin |
|---|---|
| `gpu-simulator/` | [Accel-Sim](https://github.com/accel-sim/accel-sim-framework) `d7f397a` + changes for this work |
| `gpu-simulator/gpgpu-sim/` | [GPGPU-Sim distribution](https://github.com/gpgpu-sim/gpgpu-sim_distribution) with its full history + changes for this work |
| `gpu-simulator/extern/pybind11/` | pybind11 `34566e1` (vendored) |

## Build
```bash
docker build -t mcm-accelsim -f simulator/Dockerfile .
docker run --rm -it -v "$PWD":/work -w /work mcm-accelsim bash simulator/build.sh simulator
# -> simulator/gpu-simulator/bin/release/accel-sim.out
```

## Run
```bash
source simulator/gpu-simulator/setup_environment.sh release
accel-sim.out -config gpgpusim.config -config trace.config -trace <traces>/kernelslist.g
```
Configurations used in the experiments: `../experiments/configs/`.

## Comparing model variants
A run without `-gmmu_placement_load` places pages itself: the chiplet whose L2 TLB misses on a page first owns it, and
frames are handed out in walk-completion order. Any timing change reorders both, so two such runs also differ in where
data and page-table nodes sit in DRAM (channel, bank, row), and that difference mixes into the comparison. To compare
model variants, run the reference variant once with `-gmmu_placement_dump ./placement_mcm.map`, then run every variant,
the reference included, with
```
-gmmu_placement_load <reference run>/placement_mcm.map -gmmu_placement_exact 1
```
A loaded run reproduces the dumping run cycle for cycle, so the remaining difference is the model change.

## Model notes
- `-gpgpu_l2_tlb*_lookup_latency` (80) includes the SM<->L2 TLB round trip; no separate transfer delay is modeled. A miss
  pays the same total (request trip + lookup + reply trip) before its walk instead of after it.

## Debug logs

Environment variables; all are off by default and only read simulator state.

- `ACCELSIM_L2MISS_LOG=<file>`: one line per event, cycles are absolute core cycles (`gpu_sim_cycle + gpu_tot_sim_cycle`).
  - `M <cycle> <chiplet> <cluster> 0x<vpn> <kind> <ft>`: an L2 TLB verdict that needs a walk. `kind` 0 = miss (a new
    walk starts), 1 = merge (joins a walk in flight). `ft` = 1 when the page has no present leaf PTE yet (first touch;
    reference mode only, a pinned run prints 0).
  - `C <cycle> <chiplet> 0x<vpn> <t_pwq_enq> <t_walk_start> <steps> <remote> <pwc_hits> <faults> <icnt_bp>
    <step_lat_sum> <step_lat_max> 0x<last_entry_pa> 0x<data_pa>`: a walk completed (L2 TLB fill). `steps` =
    page-table reads that went to memory, `remote` = those that crossed a chiplet, `faults` = nodes/leaf this walk
    created, `step_lat_*` = issue-to-reply cycles of the memory steps, `last_entry_pa` = PA of the last page-table
    entry read, `data_pa` = the translated PA.

## License
GPGPU-Sim: `gpu-simulator/gpgpu-sim/COPYRIGHT`. Accel-Sim: `LICENSE.accel-sim`. pybind11: `gpu-simulator/extern/pybind11/LICENSE`.
