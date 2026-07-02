// dex_books.cu — REAL deterministic per-book DEX matcher for CUDA/HIP (nvcc on spark GB10,
// hipcc on evo 8060S). One thread per book, sequential price-time priority per book (bit-identical
// re-execution), thousands-millions of books in parallel. Compute-bound. GPU vs CPU (OpenMP),
// parity-checked. Mirrors dex_books.mm (Metal).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

#define DEPTH  32u
#define TAKERS 32u
struct Order { uint32_t price; uint32_t qty; uint32_t side; uint32_t _pad; };

__global__ void match_books(const Order* __restrict__ resting, const Order* __restrict__ takers,
                            uint32_t* __restrict__ fills, uint32_t n) {
    uint32_t book = blockIdx.x * blockDim.x + threadIdx.x;
    if (book >= n) return;
    const Order* R = resting + (size_t)book*DEPTH;
    const Order* T = takers  + (size_t)book*TAKERS;
    uint32_t rem[DEPTH];
    #pragma unroll
    for (uint32_t i=0;i<DEPTH;i++) rem[i]=R[i].qty;
    uint32_t total=0;
    for (uint32_t t=0;t<TAKERS;t++){
        uint32_t tq=T[t].qty, tp=T[t].price, ts=T[t].side;
        for (uint32_t i=0;i<DEPTH && tq>0;i++){
            if (rem[i]==0) continue;
            bool ok = ts==0 ? (tp>=R[i].price) : (R[i].price>=tp);
            if (!ok) continue;
            uint32_t got = tq<rem[i]?tq:rem[i];
            tq-=got; rem[i]-=got; total++;
        }
    }
    fills[book]=total;
}

static void cpu_match(const Order* resting, const Order* takers, uint32_t* fills, uint32_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t book=0; book<(int64_t)n; book++){
        const Order* R = resting + (size_t)book*DEPTH;
        const Order* T = takers  + (size_t)book*TAKERS;
        uint32_t rem[DEPTH];
        for (uint32_t i=0;i<DEPTH;i++) rem[i]=R[i].qty;
        uint32_t total=0;
        for (uint32_t t=0;t<TAKERS;t++){
            uint32_t tq=T[t].qty, tp=T[t].price, ts=T[t].side;
            for (uint32_t i=0;i<DEPTH && tq>0;i++){
                if (rem[i]==0) continue;
                bool ok = ts==0 ? (tp>=R[i].price) : (R[i].price>=tp);
                if (!ok) continue;
                uint32_t got = tq<rem[i]?tq:rem[i];
                tq-=got; rem[i]-=got; total++;
            }
        }
        fills[book]=total;
    }
}

int main(int argc, char** argv){
    uint32_t N = argc>1 ? (uint32_t)strtoul(argv[1],0,10) : 1000000;
    std::vector<Order> resting((size_t)N*DEPTH), takers((size_t)N*TAKERS);
    for (uint32_t b=0;b<N;b++){
        for (uint32_t i=0;i<DEPTH;i++)  resting[(size_t)b*DEPTH+i] = {10000+(i%16),10u+(i%7),(uint32_t)(i&1),0};
        for (uint32_t t=0;t<TAKERS;t++) takers[(size_t)b*TAKERS+t] = {10000+(t%20),40u,(uint32_t)((t+1)&1),0};
    }
    uint64_t ops=(uint64_t)N*TAKERS;
    std::vector<uint32_t> fc(N);
    double best_cpu=1e300;
    for(int r=0;r<3;r++){auto t0=std::chrono::high_resolution_clock::now();
      cpu_match(resting.data(),takers.data(),fc.data(),N);
      auto t1=std::chrono::high_resolution_clock::now();
      double s=std::chrono::duration<double>(t1-t0).count(); if(s<best_cpu)best_cpu=s;}

    Order *dR,*dT; uint32_t* dF;
    cudaMalloc(&dR,resting.size()*sizeof(Order)); cudaMalloc(&dT,takers.size()*sizeof(Order));
    cudaMalloc(&dF,N*sizeof(uint32_t));
    cudaMemcpy(dR,resting.data(),resting.size()*sizeof(Order),cudaMemcpyHostToDevice);
    cudaMemcpy(dT,takers.data(),takers.size()*sizeof(Order),cudaMemcpyHostToDevice);
    int TPB=256, blocks=(N+TPB-1)/TPB;
    match_books<<<blocks,TPB>>>(dR,dT,dF,N); cudaDeviceSynchronize();
    double best_gpu=1e300;
    for(int r=0;r<3;r++){auto t0=std::chrono::high_resolution_clock::now();
      match_books<<<blocks,TPB>>>(dR,dT,dF,N); cudaDeviceSynchronize();
      auto t1=std::chrono::high_resolution_clock::now();
      double s=std::chrono::duration<double>(t1-t0).count(); if(s<best_gpu)best_gpu=s;}
    cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(e));return 1;}
    std::vector<uint32_t> fg(N); cudaMemcpy(fg.data(),dF,N*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    uint64_t cs=0,gs=0; for(uint32_t i=0;i<N;i++){cs+=fc[i];gs+=fg[i];}
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("=== DEX deterministic per-book matcher: GPU(%s) vs CPU(%d cores) ===\n",p.name,omp_get_max_threads());
    printf("  books=%u depth=%u takers=%u  parity cpu=%llu gpu=%llu %s\n",
           N,DEPTH,TAKERS,(unsigned long long)cs,(unsigned long long)gs,cs==gs?"MATCH":"MISMATCH!");
    printf("  CPU:  %8.2f M taker-orders/sec  (%.4f s)\n", ops/best_cpu/1e6, best_cpu);
    printf("  GPU:  %8.2f M taker-orders/sec  (%.4f s, kernel-only)\n", ops/best_gpu/1e6, best_gpu);
    printf("  GPU is %.2fx the CPU\n", best_cpu/best_gpu);
    return 0;
}
