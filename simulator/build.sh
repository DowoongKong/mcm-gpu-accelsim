#!/bin/bash
# build.sh -- clean build of the simulator. Run inside a container built from simulator/Dockerfile (CUDA 12.2, gcc 11.4), from the
# repository root or with the simulator directory as the first argument. Writes build_status.txt (MAKE_EXIT, md5s, BUILD OK/FAILED)
# and build.log next to this script. Binary: gpu-simulator/bin/release/accel-sim.out. md5 values depend on paths and compiler.
T=$(cd "${1:-$(dirname "$0")}" && pwd)
[ -d "$T/gpu-simulator" ] || { echo "no $T/gpu-simulator"; exit 1; }
echo "$(date '+%F %T') clean build start ($T)" > "$T/build_status.txt"
( cd "$T/gpu-simulator" && rm -rf build bin gpgpu-sim/lib gpgpu-sim/build && source ./setup_environment.sh release >/dev/null 2>&1 \
  && nice -n 10 make -j"${JOBS:-8}" > "$T/build.log" 2>&1; echo "MAKE_EXIT=$?" ) >> "$T/build_status.txt" 2>&1
BIN=$T/gpu-simulator/bin/release/accel-sim.out; LIB=$(ls "$T"/gpu-simulator/gpgpu-sim/lib/*/*/release/libcudart.so 2>/dev/null | head -1)
if grep -q "MAKE_EXIT=0" "$T/build_status.txt" && [ -x "$BIN" ] && [ -f "$LIB" ]; then md5sum "$BIN" "$LIB" >> "$T/build_status.txt"; echo "$(date '+%F %T') BUILD OK" >> "$T/build_status.txt"
else echo "$(date '+%F %T') BUILD FAILED (see $T/build.log)" >> "$T/build_status.txt"; fi; cat "$T/build_status.txt"
