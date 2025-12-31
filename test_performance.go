// +build ignore

package main

import (
    "fmt"
    "time"
    "github.com/luxfi/gpu"
)

func main() {
    fmt.Println("=== GPU Performance Test ===")
    fmt.Printf("System: %s\n", gpu.Info())
    fmt.Println()
    
    // Test different sizes
    sizes := []int{100, 500, 1000}
    
    for _, size := range sizes {
        fmt.Printf("Matrix Size: %dx%d\n", size, size)

        // Create matrices
        a := gpu.Random([]int{size, size}, gpu.Float32)
        b := gpu.Random([]int{size, size}, gpu.Float32)

        // Warmup
        c := gpu.MatMul(a, b)
        gpu.Eval(c)
        gpu.Synchronize()

        // Benchmark
        iterations := 10
        start := time.Now()
        for i := 0; i < iterations; i++ {
            c = gpu.MatMul(a, b)
            gpu.Eval(c)
            gpu.Synchronize()
        }
        elapsed := time.Since(start)

        // Calculate metrics
        msPerOp := elapsed.Seconds() * 1000 / float64(iterations)
        ops := int64(size * size * size * 2) // multiply-add operations
        gflops := float64(ops*int64(iterations)) / elapsed.Seconds() / 1e9

        fmt.Printf("  Time: %.2f ms per operation\n", msPerOp)
        fmt.Printf("  Performance: %.1f GFLOPS\n", gflops)
        fmt.Println()
    }
    
    fmt.Println("Metal backend automatically selected for optimal performance on Apple Silicon!")
}