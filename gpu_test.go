package gpu_test

import (
	"testing"

	"github.com/luxfi/gpu"
)

func TestBackendDetection(t *testing.T) {
	backend := gpu.GetBackend()
	t.Logf("Detected backend: %s", backend)

	device := gpu.GetDevice()
	if device != nil {
		t.Logf("Device: %s (Memory: %d GB)",
			device.Name,
			device.Memory/(1024*1024*1024))
	}
}

func TestArrayOperations(t *testing.T) {
	// Create arrays
	a := gpu.Zeros([]int{10, 10}, gpu.Float32)
	b := gpu.Ones([]int{10, 10}, gpu.Float32)

	// Add arrays
	c := gpu.Add(a, b)

	// Force evaluation
	gpu.Eval(c)

	// Wait for completion
	gpu.Synchronize()

	t.Log("Array operations completed")
}

func TestMatrixMultiplication(t *testing.T) {
	// Create matrices
	a := gpu.Random([]int{100, 50}, gpu.Float32)
	b := gpu.Random([]int{50, 75}, gpu.Float32)

	// Matrix multiplication
	c := gpu.MatMul(a, b)

	// Force evaluation
	gpu.Eval(c)
	gpu.Synchronize()

	t.Log("Matrix multiplication completed")
}

func TestReduction(t *testing.T) {
	// Create array
	a := gpu.Arange(0, 100, 1)

	// Sum all elements
	sum := gpu.Sum(a)

	// Mean of elements
	mean := gpu.Mean(a)

	// Evaluate
	gpu.Eval(sum, mean)
	gpu.Synchronize()

	t.Log("Reduction operations completed")
}

func TestStream(t *testing.T) {
	// Create compute stream
	_ = gpu.NewStream()

	// Create arrays
	a := gpu.Random([]int{1000, 1000}, gpu.Float32)
	b := gpu.Random([]int{1000, 1000}, gpu.Float32)

	// Perform operations
	c := gpu.MatMul(a, b)

	// Force evaluation
	gpu.Eval(c)

	// Synchronize stream
	gpu.Synchronize()

	t.Log("Stream operations completed")
}

func BenchmarkMatMul(b *testing.B) {
	// Create large matrices
	x := gpu.Random([]int{1000, 1000}, gpu.Float32)
	y := gpu.Random([]int{1000, 1000}, gpu.Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z := gpu.MatMul(x, y)
		gpu.Eval(z)
	}
	gpu.Synchronize()
}

func BenchmarkArrayCreation(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a := gpu.Zeros([]int{100, 100}, gpu.Float32)
		gpu.Eval(a)
	}
	gpu.Synchronize()
}

func BenchmarkReduction(b *testing.B) {
	a := gpu.Random([]int{10000}, gpu.Float32)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := gpu.Sum(a)
		gpu.Eval(sum)
	}
	gpu.Synchronize()
}