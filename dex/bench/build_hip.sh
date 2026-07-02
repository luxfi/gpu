#!/bin/bash
# build_hip.sh — build the deterministic per-book DEX matcher for AMD (ROCm/HIP).
# Verified on evo (AMD Radeon 8060S, gfx1151 RDNA3.5 APU, ROCm 7.x).
# hipify-perl is often absent; we sed-map the CUDA runtime API to HIP (identical semantics).
# Usage: ./build_hip.sh [N]     (N = number of books/markets, default 4000000)
set -e
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
here=$(cd "$(dirname "$0")" && pwd)
N=${1:-4000000}
hipify() {
  sed -e 's/#include <cuda_runtime.h>/#include <hip\/hip_runtime.h>/' \
      -e 's/cudaMalloc/hipMalloc/g' -e 's/cudaMemcpyHostToDevice/hipMemcpyHostToDevice/g' \
      -e 's/cudaMemcpyDeviceToHost/hipMemcpyDeviceToHost/g' -e 's/cudaMemcpy/hipMemcpy/g' \
      -e 's/cudaDeviceSynchronize/hipDeviceSynchronize/g' -e 's/cudaGetLastError/hipGetLastError/g' \
      -e 's/cudaError_t/hipError_t/g' -e 's/cudaSuccess/hipSuccess/g' \
      -e 's/cudaGetErrorString/hipGetErrorString/g' -e 's/cudaDeviceProp/hipDeviceProp_t/g' \
      -e 's/cudaGetDeviceProperties/hipGetDeviceProperties/g' \
      "$1" > "$2"
}
hipify "$here/dex_books.cu"     /tmp/dex_books.hip.cpp
hipify "$here/dex_books_opt.cu" /tmp/dex_books_opt.hip.cpp
hipcc -O3 -fopenmp -o /tmp/dex_books     /tmp/dex_books.hip.cpp     2>/dev/null
hipcc -O3 -fopenmp -o /tmp/dex_books_opt /tmp/dex_books_opt.hip.cpp 2>/dev/null
echo "--- baseline (AoS) ---";        /tmp/dex_books     "$N"
echo "--- optimized (coalesced SoA) ---"; /tmp/dex_books_opt "$N"
