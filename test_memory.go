// +build ignore

package main

import (
    "fmt"
    "github.com/luxfi/gpu"
)

func main() {
    fmt.Println("=== GPU Memory Detection Test ===")

    backend := gpu.GetBackend()
    device := gpu.GetDevice()

    fmt.Printf("Backend: %s\n", backend)
    fmt.Printf("Device: %s\n", device.Name)
    fmt.Printf("Memory: %.1f GB\n", float64(device.Memory)/(1024*1024*1024))

    // Show info
    fmt.Printf("\nFull Info: %s\n", gpu.Info())
}