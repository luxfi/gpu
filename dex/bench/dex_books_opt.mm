// dex_books_opt.mm — OPTIMIZED deterministic per-book DEX matcher (Metal).
// Same deterministic algorithm as dex_books.mm (one thread per book, sequential price-time
// priority) but ORDER-MAJOR (SoA) layout so adjacent threads read CONTIGUOUS addresses (coalesced):
//   AoS: resting[book*DEPTH + i]  -> thread b, b+1 read stride DEPTH apart (uncoalesced)
//   SoA: plane[i*N + book]        -> thread b, b+1 read adjacent (coalesced)
// Build: clang++ -O3 -std=c++17 -Xpreprocessor -fopenmp -lomp -framework Metal -framework Foundation
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <omp.h>

static const uint32_t DEPTH  = 32;
static const uint32_t TAKERS = 32;

static const char* kSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;
constant uint DEPTH  = 32;
constant uint TAKERS = 32;
kernel void match_books_soa(
    device const uint*  r_price [[buffer(0)]],
    device const uint*  r_qty   [[buffer(1)]],
    device const uchar* r_side  [[buffer(2)]],
    device const uint*  t_price [[buffer(3)]],
    device const uint*  t_qty   [[buffer(4)]],
    device const uchar* t_side  [[buffer(5)]],
    device uint*        fills    [[buffer(6)]],
    constant uint&      n        [[buffer(7)]],
    uint book [[thread_position_in_grid]])
{
    if (book >= n) return;
    uint rem[DEPTH]; uint rp[DEPTH]; uchar rs[DEPTH];
    for (uint i=0;i<DEPTH;i++){ uint o=i*n+book; rem[i]=r_qty[o]; rp[i]=r_price[o]; rs[i]=r_side[o]; }
    uint total=0;
    for (uint t=0;t<TAKERS;t++){
        uint o=t*n+book; uint tq=t_qty[o], tp=t_price[o]; uchar ts=t_side[o];
        for (uint i=0;i<DEPTH && tq>0;i++){
            if (rem[i]==0) continue;
            bool ok = ts==0 ? (tp>=rp[i]) : (rp[i]>=tp);
            if (!ok) continue;
            uint got = min(tq, rem[i]);
            tq-=got; rem[i]-=got; total++;
        }
    }
    fills[book]=total;
}
)METAL";

static void cpu_match_soa(const uint32_t* rp,const uint32_t* rq,const uint8_t* rs,
                          const uint32_t* tp,const uint32_t* tq,const uint8_t* ts,
                          uint32_t* fills, uint32_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t book=0; book<(int64_t)n; book++){
        uint32_t rem[DEPTH],rpr[DEPTH]; uint8_t rsi[DEPTH];
        for (uint32_t i=0;i<DEPTH;i++){ size_t o=(size_t)i*n+book; rem[i]=rq[o]; rpr[i]=rp[o]; rsi[i]=rs[o]; }
        uint32_t total=0;
        for (uint32_t t=0;t<TAKERS;t++){
            size_t o=(size_t)t*n+book; uint32_t q=tq[o],p=tp[o]; uint8_t s=ts[o];
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
 @autoreleasepool {
    uint32_t N = argc>1 ? (uint32_t)strtoul(argv[1],0,10) : 2000000;
    size_t RS=(size_t)N*DEPTH, TS=(size_t)N*TAKERS;
    std::vector<uint32_t> rp(RS),rq(RS),tp(TS),tq(TS); std::vector<uint8_t> rs(RS),ts(TS);
    for (uint32_t b=0;b<N;b++){
        for (uint32_t i=0;i<DEPTH;i++){ size_t o=(size_t)i*N+b; rp[o]=10000+(i%16); rq[o]=10+(i%7); rs[o]=(i&1); }
        for (uint32_t t=0;t<TAKERS;t++){ size_t o=(size_t)t*N+b; tp[o]=10000+(t%20); tq[o]=40; ts[o]=((t+1)&1); }
    }
    uint64_t ops=(uint64_t)N*TAKERS;
    std::vector<uint32_t> fc(N);
    double best_cpu=1e300;
    for(int r=0;r<3;r++){ auto t0=std::chrono::high_resolution_clock::now();
      cpu_match_soa(rp.data(),rq.data(),rs.data(),tp.data(),tq.data(),ts.data(),fc.data(),N);
      auto t1=std::chrono::high_resolution_clock::now();
      double s=std::chrono::duration<double>(t1-t0).count(); if(s<best_cpu)best_cpu=s; }

    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); if(!dev){printf("no Metal\n");return 1;}
    NSError* e=nil;
    id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&e];
    if(!lib){printf("shader fail: %s\n",e.localizedDescription.UTF8String);return 1;}
    id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"match_books_soa"] error:&e];
    if(!pso){printf("pso fail: %s\n",e.localizedDescription.UTF8String);return 1;}
    id<MTLCommandQueue> q=[dev newCommandQueue];
    auto buf=[&](void* p,size_t n){ return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared]; };
    id<MTLBuffer> bRp=buf(rp.data(),RS*4), bRq=buf(rq.data(),RS*4), bRs=buf(rs.data(),RS);
    id<MTLBuffer> bTp=buf(tp.data(),TS*4), bTq=buf(tq.data(),TS*4), bTs=buf(ts.data(),TS);
    id<MTLBuffer> bF=[dev newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bN=buf(&N,4);
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup; if(tg>256)tg=256;
    auto run=[&](){
      id<MTLCommandBuffer> cb=[q commandBuffer];
      id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:bRp offset:0 atIndex:0];[enc setBuffer:bRq offset:0 atIndex:1];[enc setBuffer:bRs offset:0 atIndex:2];
      [enc setBuffer:bTp offset:0 atIndex:3];[enc setBuffer:bTq offset:0 atIndex:4];[enc setBuffer:bTs offset:0 atIndex:5];
      [enc setBuffer:bF offset:0 atIndex:6];[enc setBuffer:bN offset:0 atIndex:7];
      [enc dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
      [enc endEncoding];[cb commit];[cb waitUntilCompleted];
    };
    run();
    double best_gpu=1e300;
    for(int r=0;r<3;r++){ auto t0=std::chrono::high_resolution_clock::now(); run();
      auto t1=std::chrono::high_resolution_clock::now();
      double s=std::chrono::duration<double>(t1-t0).count(); if(s<best_gpu)best_gpu=s; }

    uint32_t* gf=(uint32_t*)bF.contents; uint64_t cs=0,gs=0;
    for(uint32_t i=0;i<N;i++){cs+=fc[i];gs+=gf[i];}
    printf("=== OPT (coalesced SoA) per-book matcher: GPU(Metal, %s) vs CPU(%d cores), N=%u ===\n", dev.name.UTF8String, omp_get_max_threads(), N);
    printf("  parity cpu=%llu gpu=%llu %s\n",(unsigned long long)cs,(unsigned long long)gs, cs==gs?"MATCH":"MISMATCH!");
    printf("  CPU:  %8.2f M ord/s (%.4f s)\n", ops/best_cpu/1e6, best_cpu);
    printf("  GPU:  %8.2f M ord/s (%.4f s)\n", ops/best_gpu/1e6, best_gpu);
    printf("  GPU is %.2fx the CPU\n", best_cpu/best_gpu);
 }
 return 0;
}
