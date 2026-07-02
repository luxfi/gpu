#!/bin/bash
# build_cuda.sh — build the deterministic per-book DEX matcher for NVIDIA (CUDA).
# Verified on spark (NVIDIA GB10 Grace-Blackwell, CUDA 13.0, arm64/sbsa).
# Usage: ./build_cuda.sh [N]     (N = number of books/markets, default 4000000)
set -e
CUDA_ROOT=${CUDA_ROOT:-$(dirname "$(dirname "$(command -v nvcc)")")}
[ -x "$CUDA_ROOT/bin/nvcc" ] || CUDA_ROOT=/usr/local/cuda
NVCC=$CUDA_ROOT/bin/nvcc
# sbsa (arm64) vs x86_64 include/lib target dir
TGT=$(ls -d "$CUDA_ROOT"/targets/*/include 2>/dev/null | head -1)
INCF=""; LIBF=""
[ -n "$TGT" ] && INCF="-I$TGT" && LIBF="-L${TGT%/include}/lib"
here=$(cd "$(dirname "$0")" && pwd)
N=${1:-4000000}
echo "== nvcc $($NVCC --version | grep release) =="
$NVCC -O3 -arch=native $INCF $LIBF -Xcompiler -fopenmp -o /tmp/dex_books     "$here/dex_books.cu"
$NVCC -O3 -arch=native $INCF $LIBF -Xcompiler -fopenmp -o /tmp/dex_books_opt "$here/dex_books_opt.cu"
echo "--- baseline (AoS) ---";        /tmp/dex_books     "$N"
echo "--- optimized (coalesced SoA) ---"; /tmp/dex_books_opt "$N"
