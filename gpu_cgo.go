//go:build cgo

// Package gpu provides Go bindings for cross-platform GPU acceleration.
//
// Backends (auto-detected by luxcpp/gpu):
//   - Metal: Apple Silicon GPUs (macOS)
//   - CUDA: NVIDIA GPUs (Linux/Windows)
//   - CPU: Optimized SIMD fallback (all platforms)
//
// The C++ library (luxgpu) handles backend selection automatically.
// Build with CGO_ENABLED=1 for GPU support.
package gpu

/*
#cgo pkg-config: lux-gpu
#include <lux/gpu/gpu.h>
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"unsafe"
)

// Context manages GPU runtime and resources
type Context struct {
	mu      sync.RWMutex
	gpu     *C.LuxGPU
	backend Backend
	device  *Device
	stream  *Stream

	// Version info
	version string

	// Resource management
	tensors map[*C.LuxTensor]*Array
	streams map[*C.LuxStream]*Stream
}

var (
	// DefaultContext is the global GPU context
	DefaultContext *Context
)

func init() {
	// Initialize default context on package load
	gpu := C.lux_gpu_create()
	DefaultContext = &Context{
		gpu:     gpu,
		backend: Auto,
		tensors: make(map[*C.LuxTensor]*Array),
		streams: make(map[*C.LuxStream]*Stream),
		version: Version,
	}

	// Check environment variable for backend override
	if envBackend := os.Getenv("LUX_GPU_BACKEND"); envBackend != "" {
		switch strings.ToLower(envBackend) {
		case "cpu":
			DefaultContext.SetBackend(CPU)
		case "metal":
			DefaultContext.SetBackend(Metal)
		case "cuda":
			DefaultContext.SetBackend(CUDA)
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

	if c.gpu == nil {
		c.backend = CPU
		c.device = &Device{
			Type:   CPU,
			ID:     0,
			Name:   "CPU",
			Memory: getSystemMemory(),
		}
		return
	}

	// Query backend from C library
	backendType := C.lux_gpu_backend(c.gpu)
	backendName := C.GoString(C.lux_gpu_backend_name(c.gpu))
	deviceName := C.GoString(C.lux_gpu_device_name(c.gpu))
	memory := int64(C.lux_gpu_memory_total(c.gpu))

	switch backendType {
	case C.LUX_GPU_BACKEND_METAL:
		c.backend = Metal
	case C.LUX_GPU_BACKEND_CUDA:
		c.backend = CUDA
	case C.LUX_GPU_BACKEND_CPU:
		c.backend = CPU
	default:
		c.backend = CPU
	}

	c.device = &Device{
		Type:   c.backend,
		ID:     0,
		Name:   fmt.Sprintf("%s (%s)", deviceName, backendName),
		Memory: memory,
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

	// Destroy existing context if changing backend
	if c.gpu != nil {
		C.lux_gpu_destroy(c.gpu)
	}

	var gpuBackend C.LuxGPUBackend
	switch backend {
	case Metal:
		gpuBackend = C.LUX_GPU_BACKEND_METAL
	case CUDA:
		gpuBackend = C.LUX_GPU_BACKEND_CUDA
	case CPU:
		gpuBackend = C.LUX_GPU_BACKEND_CPU
	case Auto:
		gpuBackend = C.LUX_GPU_BACKEND_AUTO
	default:
		return ErrInvalidBackend
	}

	c.gpu = C.lux_gpu_create_backend(gpuBackend)
	if c.gpu == nil {
		return ErrNoGPU
	}

	// Update device info
	c.mu.Unlock()
	c.detectBackend()
	c.mu.Lock()

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

// GPU returns the underlying C GPU handle
func (c *Context) GPU() *C.LuxGPU {
	return c.gpu
}

// Zeros creates a zero-filled array
func Zeros(shape []int, dtype Dtype) *Array {
	return DefaultContext.Zeros(shape, dtype)
}

// Zeros creates a zero-filled array (Context method)
func (c *Context) Zeros(shape []int, dtype Dtype) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	cShape := intsToCInts(shape)
	tensor := C.lux_gpu_tensor_zeros(
		c.gpu,
		&cShape[0],
		C.int(len(shape)),
		dtypeToC(dtype),
	)

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  shape,
		dtype:  dtype,
	}

	c.tensors[tensor] = arr
	return arr
}

// Ones creates an array filled with ones
func Ones(shape []int, dtype Dtype) *Array {
	return DefaultContext.Ones(shape, dtype)
}

// Ones creates an array filled with ones (Context method)
func (c *Context) Ones(shape []int, dtype Dtype) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	cShape := intsToCInts(shape)
	tensor := C.lux_gpu_tensor_ones(
		c.gpu,
		&cShape[0],
		C.int(len(shape)),
		dtypeToC(dtype),
	)

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  shape,
		dtype:  dtype,
	}

	c.tensors[tensor] = arr
	return arr
}

// Full creates an array filled with a value
func Full(shape []int, value float64, dtype Dtype) *Array {
	return DefaultContext.Full(shape, value, dtype)
}

// Full creates an array filled with a value (Context method)
func (c *Context) Full(shape []int, value float64, dtype Dtype) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	cShape := intsToCInts(shape)
	tensor := C.lux_gpu_tensor_full(
		c.gpu,
		&cShape[0],
		C.int(len(shape)),
		C.float(value),
		dtypeToC(dtype),
	)

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  shape,
		dtype:  dtype,
	}

	c.tensors[tensor] = arr
	return arr
}

// FromSlice creates an array from a Go slice
func FromSlice(data []float32, shape []int, dtype Dtype) *Array {
	return DefaultContext.FromSlice(data, shape, dtype)
}

// FromSlice creates an array from a Go slice (Context method)
func (c *Context) FromSlice(data []float32, shape []int, dtype Dtype) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	cShape := intsToCInts(shape)
	tensor := C.lux_gpu_tensor_create(
		c.gpu,
		unsafe.Pointer(&data[0]),
		&cShape[0],
		C.int(len(shape)),
		dtypeToC(dtype),
	)

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  shape,
		dtype:  dtype,
	}

	c.tensors[tensor] = arr
	return arr
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
	var tensor *C.LuxTensor

	switch d := data.(type) {
	case []int64:
		tensor = C.lux_gpu_tensor_create(
			c.gpu,
			unsafe.Pointer(&d[0]),
			cShapePtr,
			cNdim,
			C.LUX_DTYPE_I64,
		)
	case []float32:
		tensor = C.lux_gpu_tensor_create(
			c.gpu,
			unsafe.Pointer(&d[0]),
			cShapePtr,
			cNdim,
			dtypeToC(dtype),
		)
	case []float64:
		// Convert float64 to float32
		f32 := make([]float32, len(d))
		for i, v := range d {
			f32[i] = float32(v)
		}
		tensor = C.lux_gpu_tensor_create(
			c.gpu,
			unsafe.Pointer(&f32[0]),
			cShapePtr,
			cNdim,
			dtypeToC(dtype),
		)
	case []int32:
		tensor = C.lux_gpu_tensor_create(
			c.gpu,
			unsafe.Pointer(&d[0]),
			cShapePtr,
			cNdim,
			C.LUX_DTYPE_I32,
		)
	default:
		// Fallback: create zeros array
		tensor = C.lux_gpu_tensor_zeros(
			c.gpu,
			cShapePtr,
			cNdim,
			dtypeToC(dtype),
		)
	}

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  shape,
		dtype:  dtype,
	}

	c.tensors[tensor] = arr
	return arr
}

// Add performs element-wise addition
func Add(a, b *Array) *Array {
	return DefaultContext.Add(a, b)
}

// Add performs element-wise addition (Context method)
func (c *Context) Add(a, b *Array) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	tensor := C.lux_gpu_add(
		c.gpu,
		(*C.LuxTensor)(a.handle),
		(*C.LuxTensor)(b.handle),
	)

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  a.shape,
		dtype:  a.dtype,
	}

	c.tensors[tensor] = arr
	return arr
}

// Subtract performs element-wise subtraction
func Subtract(a, b *Array) *Array {
	return DefaultContext.Subtract(a, b)
}

// Subtract performs element-wise subtraction (Context method)
func (c *Context) Subtract(a, b *Array) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	tensor := C.lux_gpu_subtract(
		c.gpu,
		(*C.LuxTensor)(a.handle),
		(*C.LuxTensor)(b.handle),
	)

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  a.shape,
		dtype:  a.dtype,
	}

	c.tensors[tensor] = arr
	return arr
}

// Multiply performs element-wise multiplication
func Multiply(a, b *Array) *Array {
	return DefaultContext.Multiply(a, b)
}

// Multiply performs element-wise multiplication (Context method)
func (c *Context) Multiply(a, b *Array) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	tensor := C.lux_gpu_multiply(
		c.gpu,
		(*C.LuxTensor)(a.handle),
		(*C.LuxTensor)(b.handle),
	)

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  a.shape,
		dtype:  a.dtype,
	}

	c.tensors[tensor] = arr
	return arr
}

// Divide performs element-wise division
func Divide(a, b *Array) *Array {
	return DefaultContext.Divide(a, b)
}

// Divide performs element-wise division (Context method)
func (c *Context) Divide(a, b *Array) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	tensor := C.lux_gpu_divide(
		c.gpu,
		(*C.LuxTensor)(a.handle),
		(*C.LuxTensor)(b.handle),
	)

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  a.shape,
		dtype:  a.dtype,
	}

	c.tensors[tensor] = arr
	return arr
}

// MatMul performs matrix multiplication
func MatMul(a, b *Array) *Array {
	return DefaultContext.MatMul(a, b)
}

// MatMul performs matrix multiplication (Context method)
func (c *Context) MatMul(a, b *Array) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	tensor := C.lux_gpu_matmul(
		c.gpu,
		(*C.LuxTensor)(a.handle),
		(*C.LuxTensor)(b.handle),
	)

	// Calculate output shape for matmul
	outShape := make([]int, len(a.shape))
	copy(outShape, a.shape)
	if len(b.shape) >= 2 {
		outShape[len(outShape)-1] = b.shape[len(b.shape)-1]
	}

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  outShape,
		dtype:  a.dtype,
	}

	c.tensors[tensor] = arr
	return arr
}

// Sum computes the sum of array elements
func Sum(a *Array, axis ...int) *Array {
	return DefaultContext.Sum(a, axis...)
}

// Sum computes the sum (Context method)
func (c *Context) Sum(a *Array, axis ...int) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	var tensor *C.LuxTensor
	if len(axis) == 0 {
		tensor = C.lux_gpu_sum(c.gpu, (*C.LuxTensor)(a.handle), nil, 0)
	} else {
		cAxis := intsToCInts(axis)
		tensor = C.lux_gpu_sum(c.gpu, (*C.LuxTensor)(a.handle), &cAxis[0], C.int(len(axis)))
	}

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  []int{1}, // Sum reduces to scalar or reduced shape
		dtype:  a.dtype,
	}

	c.tensors[tensor] = arr
	return arr
}

// Mean computes the mean of array elements
func Mean(a *Array, axis ...int) *Array {
	return DefaultContext.Mean(a, axis...)
}

// Mean computes the mean (Context method)
func (c *Context) Mean(a *Array, axis ...int) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	var tensor *C.LuxTensor
	if len(axis) == 0 {
		tensor = C.lux_gpu_mean(c.gpu, (*C.LuxTensor)(a.handle), nil, 0)
	} else {
		cAxis := intsToCInts(axis)
		tensor = C.lux_gpu_mean(c.gpu, (*C.LuxTensor)(a.handle), &cAxis[0], C.int(len(axis)))
	}

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  []int{1},
		dtype:  a.dtype,
	}

	c.tensors[tensor] = arr
	return arr
}

// Synchronize waits for all operations to complete
func Synchronize() {
	DefaultContext.Synchronize()
}

// Synchronize waits for all operations (Context method)
func (c *Context) Synchronize() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.gpu != nil {
		C.lux_gpu_sync(c.gpu)
	}
}

// Free releases an array's memory
func (c *Context) Free(a *Array) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if a.handle != nil {
		tensor := (*C.LuxTensor)(a.handle)
		C.lux_gpu_tensor_destroy(tensor)
		delete(c.tensors, tensor)
		a.handle = nil
	}
}

// FreeStream releases a stream's resources
func (c *Context) FreeStream(s *Stream) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if s.handle != nil {
		stream := (*C.LuxStream)(s.handle)
		C.lux_gpu_stream_destroy(stream)
		delete(c.streams, stream)
		s.handle = nil
	}
}

// Close releases all resources
func (c *Context) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Free all tensors
	for tensor := range c.tensors {
		C.lux_gpu_tensor_destroy(tensor)
	}
	c.tensors = make(map[*C.LuxTensor]*Array)

	// Free all streams
	for stream := range c.streams {
		C.lux_gpu_stream_destroy(stream)
	}
	c.streams = make(map[*C.LuxStream]*Stream)

	// Destroy GPU context
	if c.gpu != nil {
		C.lux_gpu_destroy(c.gpu)
		c.gpu = nil
	}
}

// NewStream creates a new compute stream
func NewStream() *Stream {
	return DefaultContext.NewStream()
}

// NewStream creates a new compute stream (Context method)
func (c *Context) NewStream() *Stream {
	c.mu.Lock()
	defer c.mu.Unlock()

	cStream := C.lux_gpu_stream_create(c.gpu)
	stream := &Stream{
		handle: unsafe.Pointer(cStream),
		device: c.device,
	}

	c.streams[cStream] = stream
	return stream
}

// Info returns information about the GPU installation
func Info() string {
	device := GetDevice()
	return fmt.Sprintf("Lux GPU %s - Backend: %s, Device: %s, Memory: %.2f GB",
		Version,
		GetBackend(),
		device.Name,
		float64(device.Memory)/(1024*1024*1024))
}

// dtypeToC converts Go Dtype to C LuxDType
func dtypeToC(d Dtype) C.LuxDType {
	switch d {
	case Float32:
		return C.LUX_DTYPE_F32
	case Float64:
		return C.LUX_DTYPE_F32 // Promote to F32 (no F64 in C API)
	case Int32:
		return C.LUX_DTYPE_I32
	case Int64:
		return C.LUX_DTYPE_I64
	default:
		return C.LUX_DTYPE_F32
	}
}

// intsToCInts converts []int to []C.int
func intsToCInts(ints []int) []C.int {
	result := make([]C.int, len(ints))
	for i, v := range ints {
		result[i] = C.int(v)
	}
	return result
}

// getSystemMemory returns system RAM in bytes
func getSystemMemory() int64 {
	// Simple fallback for system memory detection
	switch runtime.GOOS {
	case "darwin":
		// macOS typically has at least 8GB
		return 8 * 1024 * 1024 * 1024
	default:
		return 4 * 1024 * 1024 * 1024
	}
}

// hasMetalSupport returns true if Metal is available
func hasMetalSupport() bool {
	return runtime.GOOS == "darwin" && runtime.GOARCH == "arm64"
}

// hasCUDASupport returns true if CUDA is available
func hasCUDASupport() bool {
	return runtime.GOOS == "linux" || runtime.GOOS == "windows"
}

// Eval forces evaluation of lazy operations (no-op in unified API)
func Eval(arrays ...*Array) {
	// In the unified API, operations are eager, not lazy
	// This is a no-op for compatibility
}

// Random creates an array with random values (using Full for now)
func Random(shape []int, dtype Dtype) *Array {
	return DefaultContext.Random(shape, dtype)
}

// Random creates an array with random values (Context method)
func (c *Context) Random(shape []int, dtype Dtype) *Array {
	// Use Full with 0.5 as placeholder - real random would need C API extension
	return c.Full(shape, 0.5, dtype)
}

// Arange creates an array with sequential values
func Arange(start, stop, step float64) *Array {
	return DefaultContext.Arange(start, stop, step)
}

// Arange creates an array with sequential values (Context method)
func (c *Context) Arange(start, stop, step float64) *Array {
	// Calculate size
	size := int((stop - start) / step)
	if size <= 0 {
		size = 1
	}

	// Create data manually
	data := make([]float32, size)
	val := float32(start)
	for i := range data {
		data[i] = val
		val += float32(step)
	}

	return c.FromSlice(data, []int{size}, Float32)
}

// Maximum computes element-wise maximum
func Maximum(a, b *Array) *Array {
	return DefaultContext.Maximum(a, b)
}

// Maximum computes element-wise maximum (Context method)
func (c *Context) Maximum(a, b *Array) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Use greater_equal and where to implement maximum
	cond := C.lux_gpu_greater_equal(
		c.gpu,
		(*C.LuxTensor)(a.handle),
		(*C.LuxTensor)(b.handle),
	)
	tensor := C.lux_gpu_where(
		c.gpu,
		cond,
		(*C.LuxTensor)(a.handle),
		(*C.LuxTensor)(b.handle),
	)
	C.lux_gpu_tensor_destroy(cond)

	arr := &Array{
		handle: unsafe.Pointer(tensor),
		shape:  a.shape,
		dtype:  a.dtype,
	}

	c.tensors[tensor] = arr
	return arr
}

// cgoSessionHandle implements sessionHandle for CGO builds
type cgoSessionHandle struct {
	ctx *Context
}

func (h *cgoSessionHandle) backend() Backend { return h.ctx.backend }
func (h *cgoSessionHandle) device() Device {
	d := h.ctx.device
	return Device{
		Type:   d.Type,
		ID:     d.ID,
		Name:   d.Name,
		Memory: d.Memory,
	}
}
func (h *cgoSessionHandle) sync() error                         { h.ctx.Synchronize(); return nil }
func (h *cgoSessionHandle) syncContext(_ context.Context) error { h.ctx.Synchronize(); return nil }
func (h *cgoSessionHandle) close() error                        { h.ctx.Close(); return nil }
func (h *cgoSessionHandle) tensor() TensorOps                   { return nil }
func (h *cgoSessionHandle) crypto() CryptoOps                   { return nil }
func (h *cgoSessionHandle) fhe() FHEOps                         { return nil }
func (h *cgoSessionHandle) ml() MLOps                           { return nil }

// newSession creates a new session for CGO builds
func newSession(cfg *sessionConfig) (*Session, error) {
	ctx := DefaultContext
	if cfg.backend != Auto {
		if err := ctx.SetBackend(cfg.backend); err != nil {
			return nil, err
		}
	}
	handle := &cgoSessionHandle{ctx: ctx}
	return &Session{
		handle:  handle,
		backend: ctx.backend,
		device:  handle.device(),
	}, nil
}
