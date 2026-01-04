//go:build cgo && metal

// Package gpu provides Go bindings for cross-platform GPU acceleration.
//
// Backends (auto-detected by luxcpp/gpu):
//   - Metal: Apple Silicon GPUs (macOS)
//   - CUDA: NVIDIA GPUs (Linux/Windows)
//   - CPU: Optimized SIMD fallback (all platforms)
//
// The C++ library (lux-gpu) handles backend selection automatically.
// Build with CGO_ENABLED=1 for GPU support.
package gpu

/*
#cgo pkg-config: lux-gpu
#include <lux/gpu/gpu.h>
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"unsafe"
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