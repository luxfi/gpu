#!/bin/bash
# tune_hip.sh — occupancy/register sweep of the SoA matcher on AMD (ROCm/HIP).
# Sweeps threads-per-block x kernel variant; prints M orders/sec + CPU speedup + parity.
#   variant 0 = preload (rp/rq/rs register-resident)   variant 1 = lowreg (only rem[] in regs)
# AMD 8060S tuned optimum: variant=0 (preload), TPB=64. (lowreg HURTS AMD — opposite of NVIDIA.)
set -e
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
here=$(cd "$(dirname "$0")" && pwd)
sed -e 's/#include <cuda_runtime.h>/#include <hip\/hip_runtime.h>/' \
    -e 's/cudaMalloc/hipMalloc/g' -e 's/cudaMemcpyHostToDevice/hipMemcpyHostToDevice/g' \
    -e 's/cudaMemcpyDeviceToHost/hipMemcpyDeviceToHost/g' -e 's/cudaMemcpy/hipMemcpy/g' \
    -e 's/cudaDeviceSynchronize/hipDeviceSynchronize/g' -e 's/cudaGetLastError/hipGetLastError/g' \
    -e 's/cudaError_t/hipError_t/g' -e 's/cudaSuccess/hipSuccess/g' \
    -e 's/cudaGetErrorString/hipGetErrorString/g' -e 's/cudaDeviceProp/hipDeviceProp_t/g' \
    -e 's/cudaGetDeviceProperties/hipGetDeviceProperties/g' \
    "$here/dex_tune.cu" > /tmp/dex_tune.hip.cpp
hipcc -O3 -fopenmp -o /tmp/dex_tune /tmp/dex_tune.hip.cpp 2>/dev/null
set +e
echo "=== AMD sweep (N=4M) — var0=preload var1=lowreg ==="
for V in 0 1; do for T in 32 64 128 256 512; do /tmp/dex_tune 4000000 "$T" "$V"; done; done
echo "--- tuned optimum confirm @N=16M (preload, TPB=64) ---"
/tmp/dex_tune 16000000 64 0
