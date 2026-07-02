#!/bin/bash
# build_metal.sh — build the deterministic per-book DEX matcher for Apple Silicon (Metal).
# Verified on ra (Apple M1 Max, 64GB) and dbc (Apple M4 Max, 128GB).
# NOTE: on Apple unified memory the coalesced-SoA rewrite is NEUTRAL-to-NEGATIVE (the large
# shared cache already absorbs the strided AoS access); AoS is the tuned default here. Both are
# built so you can confirm on your own silicon.
# Usage: ./build_metal.sh [N]   (N = number of books/markets, default 2000000)
set -e
LIBOMP=${LIBOMP:-$( (command -v brew >/dev/null && brew --prefix libomp) 2>/dev/null || echo /opt/homebrew/opt/libomp)}
SDK=$(xcrun --show-sdk-path)
here=$(cd "$(dirname "$0")" && pwd)
N=${1:-2000000}
cc() { xcrun clang++ -O3 -std=c++17 -isysroot "$SDK" -I"$LIBOMP/include" -L"$LIBOMP/lib" \
        -Xpreprocessor -fopenmp -lomp -framework Metal -framework Foundation -o "$1" "$2"; }
cc /tmp/dex_books     "$here/dex_books.mm"
cc /tmp/dex_books_opt "$here/dex_books_opt.mm"
export DYLD_LIBRARY_PATH=$LIBOMP/lib
echo "--- baseline (AoS) — tuned default on Apple ---"; /tmp/dex_books     "$N"
echo "--- coalesced SoA (discrete-GPU optimization) ---";  /tmp/dex_books_opt "$N"
