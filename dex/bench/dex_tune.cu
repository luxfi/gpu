// dex_tune.cu — occupancy/register tuning for the SoA per-book matcher (CUDA/HIP).
// argv: N [TPB] [variant]   variant: 0=preload(all in regs)  1=lowreg(only rem in regs, rp/rs coalesced-global)
// Deterministic per-book (one thread/book, sequential price-time), parity-checked vs OpenMP CPU.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

#define DEPTH  32u
#define TAKERS 32u

// variant 0: preload rp/rq/rs into registers (max traffic savings, max register pressure)
__global__ void match_preload(
    const uint32_t* __restrict__ r_price, const uint32_t* __restrict__ r_qty, const uint8_t* __restrict__ r_side,
    const uint32_t* __restrict__ t_price, const uint32_t* __restrict__ t_qty, const uint8_t* __restrict__ t_side,
    uint32_t* __restrict__ fills, uint32_t n) {
    uint32_t book = blockIdx.x*blockDim.x + threadIdx.x; if (book>=n) return;
    uint32_t rem[DEPTH], rp[DEPTH]; uint8_t rs[DEPTH];
    #pragma unroll
    for (uint32_t i=0;i<DEPTH;i++){ size_t o=(size_t)i*n+book; rem[i]=r_qty[o]; rp[i]=r_price[o]; rs[i]=r_side[o]; }
    uint32_t total=0;
    for (uint32_t t=0;t<TAKERS;t++){
        size_t to=(size_t)t*n+book; uint32_t tq=t_qty[to], tp=t_price[to]; uint8_t ts=t_side[to];
        for (uint32_t i=0;i<DEPTH && tq>0;i++){
            if (rem[i]==0) continue;
            bool ok = ts==0 ? (tp>=rp[i]) : (rp[i]>=tp);
            if (!ok) continue;
            uint32_t got = tq<rem[i]?tq:rem[i]; tq-=got; rem[i]-=got; total++;
        }
    }
    fills[book]=total;
}

// variant 1: only mutable rem[] in registers; read rp/rs coalesced from global inside loop (low reg pressure)
__global__ void match_lowreg(
    const uint32_t* __restrict__ r_price, const uint32_t* __restrict__ r_qty, const uint8_t* __restrict__ r_side,
    const uint32_t* __restrict__ t_price, const uint32_t* __restrict__ t_qty, const uint8_t* __restrict__ t_side,
    uint32_t* __restrict__ fills, uint32_t n) {
    uint32_t book = blockIdx.x*blockDim.x + threadIdx.x; if (book>=n) return;
    uint32_t rem[DEPTH];
    #pragma unroll
    for (uint32_t i=0;i<DEPTH;i++) rem[i]=r_qty[(size_t)i*n+book];
    uint32_t total=0;
    for (uint32_t t=0;t<TAKERS;t++){
        size_t to=(size_t)t*n+book; uint32_t tq=t_qty[to], tp=t_price[to]; uint8_t ts=t_side[to];
        for (uint32_t i=0;i<DEPTH && tq>0;i++){
            if (rem[i]==0) continue;
            uint32_t rpi=r_price[(size_t)i*n+book]; uint8_t rsi=r_side[(size_t)i*n+book];
            bool ok = ts==0 ? (tp>=rpi) : (rpi>=tp);
            if (!ok) continue;
            uint32_t got = tq<rem[i]?tq:rem[i]; tq-=got; rem[i]-=got; total++;
        }
    }
    fills[book]=total;
}

static void cpu_match(const uint32_t* rp,const uint32_t* rq,const uint8_t* rs,
                      const uint32_t* tp,const uint32_t* tq,const uint8_t* ts,uint32_t* fills,uint32_t n){
    #pragma omp parallel for schedule(static)
    for (int64_t book=0;book<(int64_t)n;book++){
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

int main(int argc,char**argv){
    uint32_t N   = argc>1?(uint32_t)strtoul(argv[1],0,10):4000000;
    int TPB      = argc>2?atoi(argv[2]):256;
    int variant  = argc>3?atoi(argv[3]):0;
    size_t RS=(size_t)N*DEPTH, TS=(size_t)N*TAKERS;
    std::vector<uint32_t> rp(RS),rq(RS),tp(TS),tq(TS); std::vector<uint8_t> rs(RS),ts(TS);
    for (uint32_t b=0;b<N;b++){
        for (uint32_t i=0;i<DEPTH;i++){ size_t o=(size_t)i*N+b; rp[o]=10000+(i%16); rq[o]=10+(i%7); rs[o]=(i&1); }
        for (uint32_t t=0;t<TAKERS;t++){ size_t o=(size_t)t*N+b; tp[o]=10000+(t%20); tq[o]=40; ts[o]=((t+1)&1); }
    }
    uint64_t ops=(uint64_t)N*TAKERS;
    std::vector<uint32_t> fc(N);
    double bc=1e300; for(int r=0;r<3;r++){auto t0=std::chrono::high_resolution_clock::now();
      cpu_match(rp.data(),rq.data(),rs.data(),tp.data(),tq.data(),ts.data(),fc.data(),N);
      double s=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count(); if(s<bc)bc=s;}
    uint32_t *drp,*drq,*dtp,*dtq,*dF; uint8_t *drs,*dts;
    cudaMalloc(&drp,RS*4);cudaMalloc(&drq,RS*4);cudaMalloc(&drs,RS);cudaMalloc(&dtp,TS*4);cudaMalloc(&dtq,TS*4);cudaMalloc(&dts,TS);cudaMalloc(&dF,N*4);
    cudaMemcpy(drp,rp.data(),RS*4,cudaMemcpyHostToDevice);cudaMemcpy(drq,rq.data(),RS*4,cudaMemcpyHostToDevice);cudaMemcpy(drs,rs.data(),RS,cudaMemcpyHostToDevice);
    cudaMemcpy(dtp,tp.data(),TS*4,cudaMemcpyHostToDevice);cudaMemcpy(dtq,tq.data(),TS*4,cudaMemcpyHostToDevice);cudaMemcpy(dts,ts.data(),TS,cudaMemcpyHostToDevice);
    int blocks=(N+TPB-1)/TPB;
    auto launch=[&](){ if(variant==1) match_lowreg<<<blocks,TPB>>>(drp,drq,drs,dtp,dtq,dts,dF,N);
                       else           match_preload<<<blocks,TPB>>>(drp,drq,drs,dtp,dtq,dts,dF,N); };
    launch(); cudaDeviceSynchronize();
    double bg=1e300; for(int r=0;r<3;r++){auto t0=std::chrono::high_resolution_clock::now();
      launch(); cudaDeviceSynchronize();
      double s=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count(); if(s<bg)bg=s;}
    cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(e));return 1;}
    std::vector<uint32_t> fg(N); cudaMemcpy(fg.data(),dF,N*4,cudaMemcpyDeviceToHost);
    uint64_t cs=0,gs=0; for(uint32_t i=0;i<N;i++){cs+=fc[i];gs+=fg[i];}
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("  [%s TPB=%d var=%d N=%u] parity %s  GPU %8.2f M ord/s (%.2fx CPU)\n",
           p.name,TPB,variant,N, cs==gs?"OK":"MISMATCH", ops/bg/1e6, bc/bg);
    return 0;
}
