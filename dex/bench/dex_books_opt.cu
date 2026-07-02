// dex_books_opt.cu — OPTIMIZED deterministic per-book DEX matcher (CUDA/HIP).
// Key optimization vs dex_books.cu: order-major (SoA) memory layout so a warp's 32 threads
// read CONTIGUOUS addresses (coalesced) instead of DEPTH-strided. Same deterministic algorithm
// (one thread per book, sequential price-time priority), parity-checked vs CPU.
//   AoS: resting[book*DEPTH + i]   -> warp reads stride DEPTH (uncoalesced)
//   SoA: resting[i*N   + book]     -> warp reads contiguous (coalesced)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

#define DEPTH  32u
#define TAKERS 32u

// SoA plane arrays: price[i*N+b], qty[i*N+b], side[i*N+b]
__global__ __launch_bounds__(256) void match_books_soa(
    const uint32_t* __restrict__ r_price, const uint32_t* __restrict__ r_qty, const uint8_t* __restrict__ r_side,
    const uint32_t* __restrict__ t_price, const uint32_t* __restrict__ t_qty, const uint8_t* __restrict__ t_side,
    uint32_t* __restrict__ fills, uint32_t n) {
    uint32_t book = blockIdx.x * blockDim.x + threadIdx.x;
    if (book >= n) return;
    uint32_t rem[DEPTH];
    uint32_t rp[DEPTH]; uint8_t rs[DEPTH];
    #pragma unroll
    for (uint32_t i=0;i<DEPTH;i++){ size_t o=(size_t)i*n+book; rem[i]=r_qty[o]; rp[i]=r_price[o]; rs[i]=r_side[o]; }
    uint32_t total=0;
    for (uint32_t t=0;t<TAKERS;t++){
        size_t to=(size_t)t*n+book;
        uint32_t tq=t_qty[to], tp=t_price[to]; uint8_t ts=t_side[to];
        for (uint32_t i=0;i<DEPTH && tq>0;i++){
            if (rem[i]==0) continue;
            bool ok = ts==0 ? (tp>=rp[i]) : (rp[i]>=tp);
            if (!ok) continue;
            uint32_t got = tq<rem[i]?tq:rem[i];
            tq-=got; rem[i]-=got; total++;
        }
    }
    fills[book]=total;
}

static void cpu_match_soa(const uint32_t* rp,const uint32_t* rq,const uint8_t* rs,
                          const uint32_t* tp,const uint32_t* tq,const uint8_t* ts,
                          uint32_t* fills, uint32_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t book=0; book<(int64_t)n; book++){
        uint32_t rem[DEPTH],rpr[DEPTH]; uint8_t rsi[DEPTH];
        for (uint32_t i=0;i<DEPTH;i++){ size_t o=(size_t)i*n+book; rem[i]=rq[o]; rpr[i]=rp[o]; rsi[i]=rs[o]; }
        uint32_t total=0;
        for (uint32_t t=0;t<TAKERS;t++){
            size_t to=(size_t)t*n+book; uint32_t q=tq[to],p=tp[to]; uint8_t s=ts[to];
            for (uint32_t i=0;i<DEPTH && q>0;i++){
                if (rem[i]==0) continue;
                bool ok = s==0 ? (p>=rpr[i]) : (rpr[i]>=p);
                if (!ok) continue;
                uint32_t got=q<rem[i]?q:rem[i]; q-=got; rem[i]-=got; total++;
            }
        }
        fills[book]=total;
    }
}

int main(int argc, char** argv){
    uint32_t N = argc>1 ? (uint32_t)strtoul(argv[1],0,10) : 4000000;
    size_t RS=(size_t)N*DEPTH, TS=(size_t)N*TAKERS;
    std::vector<uint32_t> rp(RS),rq(RS),tp(TS),tq(TS); std::vector<uint8_t> rs(RS),ts(TS);
    for (uint32_t b=0;b<N;b++){
        for (uint32_t i=0;i<DEPTH;i++){ size_t o=(size_t)i*N+b; rp[o]=10000+(i%16); rq[o]=10+(i%7); rs[o]=(i&1); }
        for (uint32_t t=0;t<TAKERS;t++){ size_t o=(size_t)t*N+b; tp[o]=10000+(t%20); tq[o]=40; ts[o]=((t+1)&1); }
    }
    uint64_t ops=(uint64_t)N*TAKERS;
    std::vector<uint32_t> fc(N);
    double best_cpu=1e300;
    for(int r=0;r<3;r++){auto t0=std::chrono::high_resolution_clock::now();
      cpu_match_soa(rp.data(),rq.data(),rs.data(),tp.data(),tq.data(),ts.data(),fc.data(),N);
      auto t1=std::chrono::high_resolution_clock::now(); double s=std::chrono::duration<double>(t1-t0).count(); if(s<best_cpu)best_cpu=s;}

    uint32_t *drp,*drq,*dtp,*dtq,*dF; uint8_t *drs,*dts;
    cudaMalloc(&drp,RS*4);cudaMalloc(&drq,RS*4);cudaMalloc(&drs,RS);cudaMalloc(&dtp,TS*4);cudaMalloc(&dtq,TS*4);cudaMalloc(&dts,TS);cudaMalloc(&dF,N*4);
    cudaMemcpy(drp,rp.data(),RS*4,cudaMemcpyHostToDevice);cudaMemcpy(drq,rq.data(),RS*4,cudaMemcpyHostToDevice);cudaMemcpy(drs,rs.data(),RS,cudaMemcpyHostToDevice);
    cudaMemcpy(dtp,tp.data(),TS*4,cudaMemcpyHostToDevice);cudaMemcpy(dtq,tq.data(),TS*4,cudaMemcpyHostToDevice);cudaMemcpy(dts,ts.data(),TS,cudaMemcpyHostToDevice);
    int TPB=256, blocks=(N+TPB-1)/TPB;
    match_books_soa<<<blocks,TPB>>>(drp,drq,drs,dtp,dtq,dts,dF,N); cudaDeviceSynchronize();
    double best_gpu=1e300;
    for(int r=0;r<3;r++){auto t0=std::chrono::high_resolution_clock::now();
      match_books_soa<<<blocks,TPB>>>(drp,drq,drs,dtp,dtq,dts,dF,N); cudaDeviceSynchronize();
      auto t1=std::chrono::high_resolution_clock::now(); double s=std::chrono::duration<double>(t1-t0).count(); if(s<best_gpu)best_gpu=s;}
    cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(e));return 1;}
    std::vector<uint32_t> fg(N); cudaMemcpy(fg.data(),dF,N*4,cudaMemcpyDeviceToHost);
    uint64_t cs=0,gs=0; for(uint32_t i=0;i<N;i++){cs+=fc[i];gs+=fg[i];}
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("=== OPT (coalesced SoA) per-book matcher: GPU(%s) vs CPU(%d cores), N=%u ===\n",p.name,omp_get_max_threads(),N);
    printf("  parity cpu=%llu gpu=%llu %s\n",(unsigned long long)cs,(unsigned long long)gs,cs==gs?"MATCH":"MISMATCH!");
    printf("  CPU:  %8.2f M ord/s (%.4f s)\n", ops/best_cpu/1e6, best_cpu);
    printf("  GPU:  %8.2f M ord/s (%.4f s, kernel-only)\n", ops/best_gpu/1e6, best_gpu);
    printf("  GPU is %.2fx the CPU\n", best_cpu/best_gpu);
    return 0;
}
