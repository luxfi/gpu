// Package gpu provides Go bindings for GPU acceleration.
// This file contains type definitions shared between CGO and non-CGO builds.
package gpu

import (
	"errors"
	"unsafe"
)

// Backend represents the compute backend used by Lux GPU
type Backend int

const (
	// Auto automatically selects the best available backend
	Auto Backend = iota
	// CPU uses CPU-only computation with SIMD
	CPU
	// Metal uses Apple Metal GPU acceleration
	Metal
	// CUDA uses NVIDIA CUDA GPU acceleration
	CUDA
	// Dawn uses WebGPU via Dawn (cross-platform)
	Dawn
	// ONNX uses ONNX Runtime (Windows fallback)
	ONNX
)

// String returns the string representation of the backend
func (b Backend) String() string {
	switch b {
	case Auto:
		return "auto"
	case CPU:
		return "cpu"
	case Metal:
		return "metal"
	case CUDA:
		return "cuda"
	case Dawn:
		return "dawn"
	case ONNX:
		return "onnx"
	default:
		return "unknown"
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
	data   []float32 // CPU fallback storage (used when CGO disabled)
}

// Shape returns the shape of the array
func (a *Array) Shape() []int {
	return a.shape
}

// Handle returns the internal handle (for validation purposes)
func (a *Array) Handle() unsafe.Pointer {
	return a.handle
}

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

	// Version is the GPU library version
	Version = "0.1.0"
)

// DeviceInfo is an alias for Device (for compatibility)
type DeviceInfo = Device
