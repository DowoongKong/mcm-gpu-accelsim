# mcm-gpu-accelsim

Trace-driven multi-chip-module (MCM) GPU simulator with a virtual-memory model, and the scripts used to run LLM decode
experiments on it.

| directory | contents |
|---|---|
| [`simulator/`](simulator/) | Accel-Sim frontend + GPGPU-Sim performance model (full history) with MCM and address-translation extensions |
| [`experiments/`](experiments/) | trace capture, placement prediction, simulation and analysis scripts; simulator configs |

Licenses: GPGPU-Sim (`simulator/gpu-simulator/gpgpu-sim/COPYRIGHT`), Accel-Sim (`simulator/LICENSE.accel-sim`),
pybind11 (`simulator/gpu-simulator/extern/pybind11/LICENSE`).
