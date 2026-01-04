//go:build !cgo

// Package gpu provides CPU-based array operations when CGO is disabled.
// For GPU-accelerated operations, build with CGO_ENABLED=1.
package gpu

import (
	"errors"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"unsafe"
)

var (
	// ErrGPUNotAvailable is returned when GPU library is not available
	ErrGPUNotAvailable = errors.New("GPU library not available - using CPU fallback")
)

// ONNX detection functions for non-CGO builds
func detectONNXBackend() bool {
	return false
}

func getONNXVersion() string {
	return "not available"
}

func hasONNXSupport() bool {
	return false
}

// Context provides a fallback implementation
type Context struct {
	mu      sync.RWMutex
	backend Backend
	device  *Device
	version string
	arrays  map[unsafe.Pointer]*Array
	streams map[unsafe.Pointer]*Stream
}

var (
	// DefaultContext is the global MLX context
	DefaultContext *Context
)

func init() {
	DefaultContext = &Context{
		backend: Auto,
		arrays:  make(map[unsafe.Pointer]*Array),
		streams: make(map[unsafe.Pointer]*Stream),
		version: Version,
	}
	DefaultContext.detectBackend()
}

// Backend detection functions that work without CGO
func hasMetalSupport() bool {
	return false // No Metal without MLX
}

func hasCUDASupport() bool {
	return false // No CUDA without MLX
}

func getMetalDeviceName() string {
	return "N/A"
}

func getMetalMemory() int64 {
	return 0
}

func getCUDADeviceName() string {
	return "N/A"
}

func getCUDAMemory() int64 {
	return 0
}

func getSystemMemory() int64 {
	// Estimate system memory (8GB default)
	return 8 * 1024 * 1024 * 1024
}

// detectBackend falls back to ONNX on Windows when MLX not available
func (c *Context) detectBackend() {
	// On Windows, try ONNX Runtime
	if runtime.GOOS == "windows" {
		if hasONNXSupport() {
			c.backend = ONNX
			c.device = &Device{
				Type:   ONNX,
				ID:     0,
				Name:   "ONNX Runtime " + getONNXVersion(),
				Memory: getSystemMemory(),
			}
			return
		}
	}

	// Fallback to CPU (limited functionality)
	c.backend = CPU
	c.device = &Device{
		Type:   CPU,
		ID:     0,
		Name:   "CPU (no MLX)",
		Memory: getSystemMemory(),
	}
}

// SetBackend sets the backend (limited without MLX)
func (c *Context) SetBackend(backend Backend) error {
	if backend == Metal || backend == CUDA {
		return fmt.Errorf("%w: %s backend requires GPU library", ErrGPUNotAvailable, backend)
	}

	if backend == ONNX {
		if !hasONNXSupport() {
			return errors.New("ONNX Runtime not available")
		}
		c.backend = ONNX
		c.device = &Device{
			Type:   ONNX,
			ID:     0,
			Name:   "ONNX Runtime " + getONNXVersion(),
			Memory: getSystemMemory(),
		}
		return nil
	}

	c.backend = CPU
	c.device = &Device{
		Type:   CPU,
		ID:     0,
		Name:   "CPU (no MLX)",
		Memory: getSystemMemory(),
	}
	return nil
}

// GetBackend returns current backend
func (c *Context) GetBackend() Backend {
	return c.backend
}

// GetDevice returns current device
func (c *Context) GetDevice() *Device {
	return c.device
}

// CPU implementations for array operations

func (c *Context) Zeros(shape []int, dtype Dtype) *Array {
	size := 1
	for _, s := range shape {
		size *= s
	}
	data := make([]float32, size)
	return &Array{shape: shape, dtype: dtype, data: data}
}

func (c *Context) Ones(shape []int, dtype Dtype) *Array {
	size := 1
	for _, s := range shape {
		size *= s
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = 1.0
	}
	return &Array{shape: shape, dtype: dtype, data: data}
}

func (c *Context) Random(shape []int, dtype Dtype) *Array {
	size := 1
	for _, s := range shape {
		size *= s
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = rand.Float32()
	}
	return &Array{shape: shape, dtype: dtype, data: data}
}

func (c *Context) Arange(start, stop, step float64) *Array {
	n := int((stop - start) / step)
	if n <= 0 {
		n = 0
	}
	data := make([]float32, n)
	val := start
	for i := range data {
		data[i] = float32(val)
		val += step
	}
	return &Array{shape: []int{n}, dtype: Float32, data: data}
}

func (c *Context) FromSlice(data []float32, shape []int, dtype Dtype) *Array {
	dataCopy := make([]float32, len(data))
	copy(dataCopy, data)
	return &Array{shape: shape, dtype: dtype, data: dataCopy}
}

func (c *Context) Add(a, b *Array) *Array {
	size := len(a.data)
	data := make([]float32, size)
	for i := 0; i < size; i++ {
		ai, bi := float32(0), float32(0)
		if i < len(a.data) {
			ai = a.data[i]
		}
		if i < len(b.data) {
			bi = b.data[i]
		}
		data[i] = ai + bi
	}
	return &Array{shape: a.shape, dtype: a.dtype, data: data}
}

func (c *Context) Maximum(a, b *Array) *Array {
	size := len(a.data)
	data := make([]float32, size)
	for i := 0; i < size; i++ {
		ai, bi := float32(0), float32(0)
		if i < len(a.data) {
			ai = a.data[i]
		}
		if i < len(b.data) {
			bi = b.data[i]
		}
		if ai > bi {
			data[i] = ai
		} else {
			data[i] = bi
		}
	}
	return &Array{shape: a.shape, dtype: a.dtype, data: data}
}

func (c *Context) Multiply(a, b *Array) *Array {
	size := len(a.data)
	data := make([]float32, size)
	for i := 0; i < size; i++ {
		ai, bi := float32(0), float32(0)
		if i < len(a.data) {
			ai = a.data[i]
		}
		if i < len(b.data) {
			bi = b.data[i]
		}
		data[i] = ai * bi
	}
	return &Array{shape: a.shape, dtype: a.dtype, data: data}
}

func (c *Context) MatMul(a, b *Array) *Array {
	if len(a.shape) < 2 || len(b.shape) < 2 {
		return &Array{shape: a.shape, dtype: a.dtype, data: make([]float32, len(a.data))}
	}
	m, k1 := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]
	if k1 != k2 {
		return &Array{shape: []int{m, n}, dtype: a.dtype, data: make([]float32, m*n)}
	}
	data := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for k := 0; k < k1; k++ {
				sum += a.data[i*k1+k] * b.data[k*n+j]
			}
			data[i*n+j] = sum
		}
	}
	return &Array{shape: []int{m, n}, dtype: a.dtype, data: data}
}

func (c *Context) Sum(a *Array, axis ...int) *Array {
	var sum float32
	for _, v := range a.data {
		sum += v
	}
	return &Array{shape: []int{1}, dtype: a.dtype, data: []float32{sum}}
}

func (c *Context) Mean(a *Array, axis ...int) *Array {
	if len(a.data) == 0 {
		return &Array{shape: []int{1}, dtype: a.dtype, data: []float32{0}}
	}
	var sum float32
	for _, v := range a.data {
		sum += v
	}
	return &Array{shape: []int{1}, dtype: a.dtype, data: []float32{sum / float32(len(a.data))}}
}

func (c *Context) Eval(arrays ...*Array) {
	// CPU operations are immediate, no lazy evaluation
}

func (c *Context) Synchronize() {
	// CPU operations are synchronous
}

func (c *Context) NewStream() *Stream {
	return &Stream{device: c.device}
}

// Info returns fallback mode information
func Info() string {
	backend := GetBackend()
	device := GetDevice()
	if backend == ONNX {
		return fmt.Sprintf("MLX Fallback Mode - Backend: %s, Device: %s (MLX library not available)",
			backend, device.Name)
	}
	return fmt.Sprintf("MLX Fallback Mode - Backend: %s, Device: %s (limited functionality)",
		backend, device.Name)
}

// Package-level functions that delegate to DefaultContext

// SetBackend sets the compute backend
func SetBackend(backend Backend) error {
	return DefaultContext.SetBackend(backend)
}

// GetBackend returns the current compute backend
func GetBackend() Backend {
	return DefaultContext.GetBackend()
}

// GetDevice returns the current compute device
func GetDevice() *Device {
	return DefaultContext.GetDevice()
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

// ArrayFromSlice creates an array from a typed Go slice with specified shape and dtype.
func ArrayFromSlice[T int64 | float64 | float32 | int32](data []T, shape []int, dtype Dtype) *Array {
	return DefaultContext.ArrayFromSlice(data, shape, dtype)
}

// ArrayFromSlice creates an array from a typed slice (Context method)
func (c *Context) ArrayFromSlice(data any, shape []int, dtype Dtype) *Array {
	return &Array{shape: shape, dtype: Float32}
}
