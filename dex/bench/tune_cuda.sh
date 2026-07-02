#!/bin/bash
# tune_cuda.sh — occupancy/register sweep of the SoA matcher on NVIDIA (CUDA).
# Sweeps threads-per-block x kernel variant; prints M orders/sec + CPU speedup + parity.
#   variant 0 = preload (rp/rq/rs register-resident)   variant 1 = lowreg (only rem[] in regs)
# GB10 tuned optimum: variant=1 (lowreg), TPB=32.
set -e
CUDA_ROOT=${CUDA_ROOT:-$(dirname "$(dirname "$(command -v nvcc)")")}
[ -x "$CUDA_ROOT/bin/nvcc" ] || CUDA_ROOT=/usr/local/cuda
NVCC=$CUDA_ROOT/bin/nvcc
TGT=$(ls -d "$CUDA_ROOT"/targets/*/include 2>/dev/null | head -1)
INCF=""; LIBF=""; [ -n "$TGT" ] && INCF="-I$TGT" && LIBF="-L${TGT%/include}/lib"
here=$(cd "$(dirname "$0")" && pwd)
$NVCC -O3 -arch=native $INCF $LIBF -Xcompiler -fopenmp -o /tmp/dex_tune "$here/dex_tune.cu"
set +e
echo "=== NVIDIA sweep (N=4M) — var0=preload var1=lowreg ==="
for V in 0 1; do for T in 32 64 128 256 512; do /tmp/dex_tune 4000000 "$T" "$V"; done; done
echo "--- tuned optimum confirm @N=16M (lowreg, TPB=32) ---"
/tmp/dex_tune 16000000 32 1
