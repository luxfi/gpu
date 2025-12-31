// +build ignore

package main

import (
    "fmt"
    "time"
    "github.com/luxfi/gpu"
)

func main() {
    fmt.Println("=== GPU Realistic Performance Test ===")
    fmt.Printf("System: %s\n\n", gpu.Info())

    // Compare backends
    backends := []struct{
        backend gpu.Backend
        name string
    }{
        {gpu.Metal, "Metal (GPU)"},
        {gpu.CPU, "CPU"},
    }

    for _, b := range backends {
        err := gpu.SetBackend(b.backend)
        if err != nil {
            fmt.Printf("%s: Not available\n\n", b.name)
            continue
        }
        
        fmt.Printf("Testing %s Backend:\n", b.name)

        // Large matrix for realistic timing
        size := 2000
        fmt.Printf("  Matrix size: %dx%d\n", size, size)

        // Create matrices
        a := gpu.Random([]int{size, size}, gpu.Float32)
        b := gpu.Random([]int{size, size}, gpu.Float32)

        // Single operation timing
        start := time.Now()
        c := gpu.MatMul(a, b)
        gpu.Eval(c)
        gpu.Synchronize()
        elapsed := time.Since(start)

        // Calculate performance
        ops := int64(size) * int64(size) * int64(size) * 2 // multiply-add
        gflops := float64(ops) / elapsed.Seconds() / 1e9
        bandwidth := float64(size*size*4*3) / elapsed.Seconds() / 1e9 // 3 matrices, 4 bytes each

        fmt.Printf("  Time: %.1f ms\n", elapsed.Seconds()*1000)
        fmt.Printf("  Performance: %.1f GFLOPS\n", gflops)
        fmt.Printf("  Memory bandwidth: %.1f GB/s\n", bandwidth)
        fmt.Println()

        // Clean up
        gpu.DefaultContext.Free(c)
    }

    // Reset to auto
    gpu.SetBackend(gpu.Auto)
    fmt.Printf("Auto-detection selected: %s\n", gpu.GetBackend())
    fmt.Println("This is the optimal backend for your M-series Mac!")
}