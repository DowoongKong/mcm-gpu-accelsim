# MCM GPU simulator (Accel-Sim / GPGPU-Sim fork)

Trace-driven multi-chip-module GPU simulator with a virtual-memory model (per-chiplet TLBs, page-table walkers, page and
page-table placement). This is the source of the build used for the experiments in `../experiments/`.

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

## License
GPGPU-Sim: `gpu-simulator/gpgpu-sim/COPYRIGHT`. Accel-Sim: `LICENSE.accel-sim`. pybind11: `gpu-simulator/extern/pybind11/LICENSE`.
