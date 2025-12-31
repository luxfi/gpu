// Package gpu provides Go bindings for GPU acceleration.
// This file contains type definitions shared between CGO and non-CGO builds.
package gpu

import (
	"errors"
	"unsafe"
)

// Backend represents the compute backend used by MLX
type Backend int

const (
	// CPU uses CPU-only computation
	CPU Backend = iota
	// Metal uses Apple Metal GPU acceleration
	Metal
	// CUDA uses NVIDIA CUDA GPU acceleration
	CUDA
	// ONNX uses ONNX Runtime (Windows fallback)
	ONNX
	// Auto automatically selects the best available backend
	Auto
)

// String returns the string representation of the backend
func (b Backend) String() string {
	switch b {
	case CPU:
		return "CPU"
	case Metal:
		return "Metal"
	case CUDA:
		return "CUDA"
	case ONNX:
		return "ONNX"
	case Auto:
		return "Auto"
	default:
		return "Unknown"
	}
}

// Device represents a compute device (CPU or GPU)
type Device struct {
	Type   Backend
	ID     int
	Name   string
	Memory int64 // Memory in bytes
}

// Array represents a multi-dimensional array
type Array struct {
	handle unsafe.Pointer
	shape  []int
	dtype  Dtype
}

// Shape returns the shape of the array
func (a *Array) Shape() []int {
	return a.shape
}

// Handle returns the internal handle (for validation purposes)
func (a *Array) Handle() unsafe.Pointer {
	return a.handle
}

// Dtype represents the data type of array elements
type Dtype int

const (
	Float32 Dtype = iota
	Float64
	Int32
	Int64
	Bool
)

// Stream represents a compute stream for async operations
type Stream struct {
	handle unsafe.Pointer
	device *Device
}

var (
	// ErrNoGPU is returned when GPU is requested but not available
	ErrNoGPU = errors.New("no GPU available")

	// ErrInvalidBackend is returned for invalid backend selection
	ErrInvalidBackend = errors.New("invalid backend")

	// Version is the MLX library version
	Version = "0.1.0"
)

// DeviceInfo is an alias for Device (for compatibility)
type DeviceInfo = Device
