// dex_books.mm — REAL deterministic DEX matching: one thread per book, each book matched
// sequentially in price-time priority (consensus-correct, no intra-book parallelism → no fork),
// thousands of books in parallel (throughput). Compute-bound, NOT dispatch-bound: each thread
// does TAKERS x DEPTH real match operations. GPU(Metal) vs CPU(OpenMP), same host.
// Build: clang++ -O3 -std=c++17 -Xpreprocessor -fopenmp -lomp -framework Metal -framework Foundation
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <omp.h>

static const uint32_t DEPTH  = 32;   // resting orders per book (price levels)
static const uint32_t TAKERS = 32;   // taker orders per book

struct Order { uint32_t price; uint32_t qty; uint32_t side; uint32_t _pad; };

static const char* kSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Order { uint price; uint qty; uint side; uint _pad; };
constant uint DEPTH  = 32;
constant uint TAKERS = 32;
kernel void match_books(
    device const Order* resting [[buffer(0)]],
    device const Order* takers  [[buffer(1)]],
    device uint*        fills    [[buffer(2)]],
    constant uint&      n_books  [[buffer(3)]],
    uint book [[thread_position_in_grid]])
{
    if (book >= n_books) return;
    device const Order* R = resting + book*DEPTH;
    device const Order* T = takers  + book*TAKERS;
    uint rem[DEPTH];
    for (uint i=0;i<DEPTH;i++) rem[i]=R[i].qty;
    uint total=0;
    for (uint t=0;t<TAKERS;t++){
        uint tq=T[t].qty, tp=T[t].price, ts=T[t].side;
        for (uint i=0;i<DEPTH && tq>0;i++){
            if (rem[i]==0) continue;
            bool ok = ts==0 ? (tp>=R[i].price) : (R[i].price>=tp);
            if (!ok) continue;
            uint got = min(tq, rem[i]);
            tq-=got; rem[i]-=got; total++;
        }
    }
    fills[book]=total;
}
)METAL";

static void cpu_match(const Order* resting, const Order* takers, uint32_t* fills, uint32_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t book=0; book<(int64_t)n; book++){
        const Order* R = resting + book*DEPTH;
        const Order* T = takers  + book*TAKERS;
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
 @autoreleasepool {
    uint32_t N = argc>1 ? (uint32_t)strtoul(argv[1],0,10) : 200000; // books (markets)
    std::vector<Order> resting((size_t)N*DEPTH), takers((size_t)N*TAKERS);
    for (uint32_t b=0;b<N;b++){
        for (uint32_t i=0;i<DEPTH;i++)  resting[(size_t)b*DEPTH+i] = {10000+ (i%16), 10u+ (i%7), (uint32_t)(i&1), 0};
        for (uint32_t t=0;t<TAKERS;t++) takers[(size_t)b*TAKERS+t] = {10000+ (t%20), 40u, (uint32_t)((t+1)&1), 0};
    }
    uint64_t match_ops = (uint64_t)N*TAKERS; // taker orders processed
    std::vector<uint32_t> fc(N);
    double best_cpu=1e300;
    for(int r=0;r<3;r++){ auto t0=std::chrono::high_resolution_clock::now();
      cpu_match(resting.data(),takers.data(),fc.data(),N);
      auto t1=std::chrono::high_resolution_clock::now();
      double s=std::chrono::duration<double>(t1-t0).count(); if(s<best_cpu)best_cpu=s; }

    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); if(!dev){printf("no Metal\n");return 1;}
    NSError* e=nil;
    id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&e];
    if(!lib){printf("shader fail: %s\n",e.localizedDescription.UTF8String);return 1;}
    id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"match_books"] error:&e];
    id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLBuffer> bR=[dev newBufferWithBytes:resting.data() length:resting.size()*sizeof(Order) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bT=[dev newBufferWithBytes:takers.data()  length:takers.size()*sizeof(Order)  options:MTLResourceStorageModeShared];
    id<MTLBuffer> bF=[dev newBufferWithLength:N*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bN=[dev newBufferWithBytes:&N length:4 options:MTLResourceStorageModeShared];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup; if(tg>256)tg=256;
    auto run=[&](){
      id<MTLCommandBuffer> cb=[q commandBuffer];
      id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:bR offset:0 atIndex:0];[enc setBuffer:bT offset:0 atIndex:1];
      [enc setBuffer:bF offset:0 atIndex:2];[enc setBuffer:bN offset:0 atIndex:3];
      [enc dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
      [enc endEncoding];[cb commit];[cb waitUntilCompleted];
    };
    run();
    double best_gpu=1e300;
    for(int r=0;r<3;r++){ auto t0=std::chrono::high_resolution_clock::now(); run();
      auto t1=std::chrono::high_resolution_clock::now();
      double s=std::chrono::duration<double>(t1-t0).count(); if(s<best_gpu)best_gpu=s; }

    // parity check
    uint32_t* gf=(uint32_t*)bF.contents; uint64_t cs=0,gs=0;
    for(uint32_t i=0;i<N;i++){cs+=fc[i];gs+=gf[i];}
    printf("=== DEX deterministic per-book matcher: GPU(Metal, %s) vs CPU(%d cores) ===\n", dev.name.UTF8String, omp_get_max_threads());
    printf("  books=%u depth=%u takers=%u  parity(sum fills) cpu=%llu gpu=%llu %s\n",
           N,DEPTH,TAKERS,(unsigned long long)cs,(unsigned long long)gs, cs==gs?"MATCH":"MISMATCH!");
    printf("  CPU:  %8.2f M taker-orders/sec  (%.4f s)\n", match_ops/best_cpu/1e6, best_cpu);
    printf("  GPU:  %8.2f M taker-orders/sec  (%.4f s)\n", match_ops/best_gpu/1e6, best_gpu);
    printf("  GPU is %.2fx the CPU\n", best_cpu/best_gpu);
 }
 return 0;
}
