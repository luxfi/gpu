# GPU-native DEX matcher — benchmark harnesses

Standalone microbenchmarks for the **deterministic per-book CLOB matcher** across all three GPU
makers: NVIDIA (CUDA), AMD (ROCm/HIP), Apple (Metal). Each harness runs the *same* algorithm on
GPU and on all CPU cores (OpenMP) on the same host, kernel-only timing, and checks GPU==CPU parity
on every run.

## Determinism (why one thread per book)

Production consensus requires every validator to re-execute a block **bit-identically**. So the
matcher assigns **one thread per order book** and walks that book's resting orders in strict
price-time priority — sequential *within* a book (no intra-book parallelism → no fill reordering →
no fork), while **millions of books run in parallel** for throughput. This is compute-bound real
work (`TAKERS × DEPTH` match steps per thread), not a dispatch microbenchmark. `parity … MATCH`
in the output is the guarantee that the GPU result equals the reference CPU result.

`DEPTH = 32` resting levels, `TAKERS = 32` taker orders per book.

## Files

| File | What |
|------|------|
| `dex_books.cu`      | Baseline matcher, **AoS** layout (`Order[book*DEPTH+i]`) — CUDA/HIP |
| `dex_books_opt.cu`  | Optimized matcher, **coalesced SoA** (order-major planes `plane[i*N+book]`) — CUDA/HIP |
| `dex_books.mm`      | Baseline matcher, AoS — Metal |
| `dex_books_opt.mm`  | Coalesced SoA — Metal |
| `dex_tune.cu`       | Occupancy/register sweep harness: `N [TPB] [variant]`; variant 0=preload, 1=lowreg |
| `build_cuda.sh` `build_hip.sh` `build_metal.sh` | Build + A/B run (baseline vs optimized) per backend |
| `tune_cuda.sh` `tune_hip.sh` | Full TPB × variant sweep, prints the tuned optimum |

## The optimization

Adjacent threads (book `b`, `b+1`) in the AoS layout read addresses `DEPTH` apart — a strided,
**uncoalesced** access. Transposing to **order-major SoA** (`plane[i*N + book]`) makes a warp's 32
threads read 32 *contiguous* words — one coalesced transaction. Same algorithm, same determinism,
pure memory-access change.

## Measured results (parity MATCH at every point)

Deterministic per-book matcher, tuned per backend. "ord/s" = taker orders matched per second.

| Backend | Silicon | Best layout / variant / TPB | Peak throughput | Baseline (AoS, TPB=256) | Speedup from tuning |
|---------|---------|-----------------------------|-----------------|-------------------------|---------------------|
| HIP (ROCm) | AMD Radeon 8060S (gfx1151, RDNA3.5 APU) | SoA · preload · 64 | **12.76 B ord/s** | 1.94 B | **6.6×** |
| CUDA | NVIDIA GB10 (Grace-Blackwell) | SoA · lowreg · 32 | **9.13 B ord/s** | 1.54 B | **5.9×** |
| Metal | Apple M4 Max (128 GB) | AoS · 256 | **5.60 B ord/s** | 5.60 B | — (SoA neutral/‑) |
| Metal | Apple M1 Max (64 GB) | AoS/SoA · 256 | **2.80 B ord/s** | 2.43 B | ~1.15× |

Two-node fabric (evo AMD + spark NVIDIA), pure deterministic matching:
**≈ 21.9 B orders/sec** aggregate, parity-verified.

## Findings ("to the nines")

1. **Coalescing (AoS→SoA) is the win on discrete/APU GDDR-style memory** — GB10 1.54→9.13 B (5.9×),
   AMD 1.94→12.76 B (6.6×). On **Apple unified memory it is neutral-to-negative**: the large shared
   cache already absorbs the strided AoS access, and the extra plane bindings add overhead — so AoS
   is the tuned default on Metal.
2. **The two discrete backends want opposite variants.** NVIDIA GB10 is **occupancy-bound** → it
   prefers *low register pressure* (`lowreg`: keep only the mutable `rem[]` in registers, re-read the
   read-only price/side coalesced from global) and *small blocks* (TPB=32). AMD 8060S is
   **bandwidth-rich** → it prefers *register-resident* data (`preload`) and TPB=64; `lowreg` costs it
   ~25% (12.76→9.4 B). No single config is optimal everywhere — tune per silicon.
3. **Determinism is free.** Every layout/occupancy/variant permutation produces `parity … MATCH`
   against the OpenMP CPU reference at every scale (N = 1M…16M). The tuning changes throughput only,
   never the fills.

## Run it

```bash
# NVIDIA (e.g. spark / GB10)
CUDA_ROOT=/usr/local/cuda-13.0 ./build_cuda.sh 4000000   # A/B baseline vs optimized
./tune_cuda.sh                                           # full TPB×variant sweep

# AMD (e.g. evo / 8060S) — needs ROCm; hipify is done inline via sed
./build_hip.sh 4000000
./tune_hip.sh

# Apple (M1/M4 Max) — needs libomp (brew install libomp)
./build_metal.sh 2000000
```
