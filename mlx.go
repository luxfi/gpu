// +build cgo

// Package gpu provides Go bindings for GPU acceleration.
// Supports multiple backends:
//
// - Metal: Apple Silicon GPUs (macOS)
// - CUDA: NVIDIA GPUs (Linux/Windows)
// - CPU: Fallback with SIMD optimizations (all platforms)
//
// The package automatically detects and uses the best available backend.
// Build with CGO_ENABLED=1 and appropriate build tags for GPU support:
//
// - cuda: Enable CUDA backend (requires CUDA toolkit)
// - metal: Enable Metal backend (automatic on macOS)
package gpu

/*
#cgo CXXFLAGS: -std=c++17 -O3 -I${SRCDIR}
#cgo darwin CXXFLAGS: -I${SRCDIR}/mlx
#cgo darwin LDFLAGS: -L${SRCDIR}/lib -lmlx -framework Metal -framework Foundation -framework CoreGraphics -framework Accelerate
#cgo linux CXXFLAGS: -I${SRCDIR}/mlx
#cgo linux LDFLAGS: -L${SRCDIR}/lib -lmlx -lm -ldl -lstdc++
#cgo windows CXXFLAGS: -I${SRCDIR}/mlx
#cgo windows LDFLAGS: -L${SRCDIR}/lib -lmlx -lm -lstdc++
#include "mlx_c_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
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

// Context manages MLX runtime and resources
type Context struct {
	mu      sync.RWMutex
	backend Backend
	device  *Device
	stream  *Stream
	
	// Version info
	version string
	
	// Resource management
	arrays  map[unsafe.Pointer]*Array
	streams map[unsafe.Pointer]*Stream
}

var (
	// DefaultContext is the global MLX context
	DefaultContext *Context
)

func init() {
	// Initialize default context on package load
	DefaultContext = &Context{
		backend: Auto,
		arrays:  make(map[unsafe.Pointer]*Array),
		streams: make(map[unsafe.Pointer]*Stream),
		version: Version,
	}
	
	// Check environment variable for backend override
	if envBackend := os.Getenv("MLX_BACKEND"); envBackend != "" {
		switch strings.ToLower(envBackend) {
		case "cpu":
			DefaultContext.SetBackend(CPU)
		case "metal":
			DefaultContext.SetBackend(Metal)
		case "cuda":
			DefaultContext.SetBackend(CUDA)
		case "onnx":
			DefaultContext.SetBackend(ONNX)
		case "auto":
			DefaultContext.detectBackend()
		default:
			// Unknown backend, fall back to auto-detect
			DefaultContext.detectBackend()
		}
	} else {
		// Auto-detect best backend
		DefaultContext.detectBackend()
	}
}

// detectBackend automatically selects the best available backend
func (c *Context) detectBackend() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// Check for Metal on macOS
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		if hasMetalSupport() {
			c.backend = Metal
			c.device = &Device{
				Type:   Metal,
				ID:     0,
				Name:   getMetalDeviceName(),
				Memory: getMetalMemory(),
			}
			return
		}
	}
	
	// Check for CUDA on Linux/Windows
	if runtime.GOOS == "linux" || runtime.GOOS == "windows" {
		if hasCUDASupport() {
			c.backend = CUDA
			c.device = &Device{
				Type:   CUDA,
				ID:     0,
				Name:   getCUDADeviceName(),
				Memory: getCUDAMemory(),
			}
			return
		}
	}
	
	// Check for ONNX Runtime on Windows
	if runtime.GOOS == "windows" && detectONNXBackend() {
		c.backend = ONNX
		c.device = &Device{
			Type:   ONNX,
			ID:     0,
			Name:   "ONNX Runtime " + getONNXVersion(),
			Memory: getSystemMemory(),
		}
		return
	}

	// Fall back to CPU
	c.backend = CPU
	c.device = &Device{
		Type:   CPU,
		ID:     0,
		Name:   "CPU",
		Memory: getSystemMemory(),
	}
}

// SetBackend sets the compute backend
func SetBackend(backend Backend) error {
	return DefaultContext.SetBackend(backend)
}

// SetBackend sets the compute backend for this context
func (c *Context) SetBackend(backend Backend) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	switch backend {
	case Metal:
		if !hasMetalSupport() {
			return ErrNoGPU
		}
		c.backend = Metal
		c.device = &Device{
			Type:   Metal,
			ID:     0,
			Name:   getMetalDeviceName(),
			Memory: getMetalMemory(),
		}
	case CUDA:
		if !hasCUDASupport() {
			return ErrNoGPU
		}
		c.backend = CUDA
		c.device = &Device{
			Type:   CUDA,
			ID:     0,
			Name:   getCUDADeviceName(),
			Memory: getCUDAMemory(),
		}
	case CPU:
		c.backend = CPU
		c.device = &Device{
			Type:   CPU,
			ID:     0,
			Name:   "CPU",
			Memory: getSystemMemory(),
		}
	case ONNX:
		if !hasONNXSupport() {
			return fmt.Errorf("ONNX Runtime not available")
		}
		c.backend = ONNX
		c.device = &Device{
			Type:   ONNX,
			ID:     0,
			Name:   "ONNX Runtime " + getONNXVersion(),
			Memory: getSystemMemory(),
		}
	case Auto:
		// Detect without holding lock
		c.mu.Unlock()
		c.detectBackend()
		c.mu.Lock()
	default:
		return ErrInvalidBackend
	}
	
	return nil
}

// GetBackend returns the current compute backend
func GetBackend() Backend {
	return DefaultContext.GetBackend()
}

// GetBackend returns the current backend for this context
func (c *Context) GetBackend() Backend {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.backend
}

// GetDevice returns the current compute device
func GetDevice() *Device {
	return DefaultContext.GetDevice()
}

// GetDevice returns the current device for this context
func (c *Context) GetDevice() *Device {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.device
}

// Zeros creates a zero-filled array
func Zeros(shape []int, dtype Dtype) *Array {
	return DefaultContext.Zeros(shape, dtype)
}

// Ones creates an array filled with ones
func Ones(shape []int, dtype Dtype) *Array {
	return DefaultContext.Ones(shape, dtype)
}

// Random creates an array with random values
func Random(shape []int, dtype Dtype) *Array {
	return DefaultContext.Random(shape, dtype)
}

// Arange creates an array with sequential values
func Arange(start, stop, step float64) *Array {
	return DefaultContext.Arange(start, stop, step)
}

// FromSlice creates an array from a Go slice
func FromSlice(data []float32, shape []int, dtype Dtype) *Array {
	return DefaultContext.FromSlice(data, shape, dtype)
}

// Add performs element-wise addition
func Add(a, b *Array) *Array {
	return DefaultContext.Add(a, b)
}

// Maximum computes element-wise maximum of two arrays
func Maximum(a, b *Array) *Array {
	return DefaultContext.Maximum(a, b)
}

// Multiply performs element-wise multiplication
func Multiply(a, b *Array) *Array {
	return DefaultContext.Multiply(a, b)
}

// MatMul performs matrix multiplication
func MatMul(a, b *Array) *Array {
	return DefaultContext.MatMul(a, b)
}

// Sum computes the sum of array elements
func Sum(a *Array, axis ...int) *Array {
	return DefaultContext.Sum(a, axis...)
}

// Mean computes the mean of array elements
func Mean(a *Array, axis ...int) *Array {
	return DefaultContext.Mean(a, axis...)
}

// Eval forces evaluation of lazy operations
func Eval(arrays ...*Array) {
	DefaultContext.Eval(arrays...)
}

// Synchronize waits for all operations to complete
func Synchronize() {
	DefaultContext.Synchronize()
}

// NewStream creates a new compute stream
func NewStream() *Stream {
	return DefaultContext.NewStream()
}

// Info returns information about the MLX installation
func Info() string {
	device := GetDevice()
	return fmt.Sprintf("MLX %s - Backend: %s, Device: %s, Memory: %.2f GB",
		Version,
		GetBackend(),
		device.Name,
		float64(device.Memory)/(1024*1024*1024))
}

// DeviceInfo is an alias for Device (for compatibility)
type DeviceInfo = Device

// ArrayFromSlice creates an array from a typed Go slice with specified shape and dtype.
// Supports []int64, []float64, []float32, []int32 data types.
func ArrayFromSlice[T int64 | float64 | float32 | int32](data []T, shape []int, dtype Dtype) *Array {
	return DefaultContext.ArrayFromSlice(data, shape, dtype)
}

// ArrayFromSlice creates an array from a typed slice (Context method)
func (c *Context) ArrayFromSlice(data any, shape []int, dtype Dtype) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	cShape := intsToCInts(shape)
	cShapePtr := &cShape[0]
	cNdim := C.int(len(shape))
	var handle unsafe.Pointer

	switch d := data.(type) {
	case []int64:
		cData := (*C.longlong)(unsafe.Pointer(&d[0]))
		handle = C.mlx_from_slice_int64(cData, C.int(len(d)), cShapePtr, cNdim, C.int(dtype))
	case []float32:
		cData := (*C.float)(unsafe.Pointer(&d[0]))
		handle = C.mlx_from_slice(cData, C.int(len(d)), cShapePtr, cNdim, C.int(dtype))
	case []float64:
		// Convert float64 to float32 for MLX
		f32 := make([]float32, len(d))
		for i, v := range d {
			f32[i] = float32(v)
		}
		cData := (*C.float)(unsafe.Pointer(&f32[0]))
		handle = C.mlx_from_slice(cData, C.int(len(f32)), cShapePtr, cNdim, C.int(dtype))
	case []int32:
		// Convert int32 to int64 for MLX
		i64 := make([]int64, len(d))
		for i, v := range d {
			i64[i] = int64(v)
		}
		cData := (*C.longlong)(unsafe.Pointer(&i64[0]))
		handle = C.mlx_from_slice_int64(cData, C.int(len(i64)), cShapePtr, cNdim, C.int(dtype))
	default:
		// Fallback: create zeros array
		handle = C.mlx_zeros(cShapePtr, cNdim, C.int(dtype))
	}

	arr := &Array{
		handle: handle,
		shape:  shape,
		dtype:  dtype,
	}

	c.arrays[handle] = arr
	return arr
}