//go:build cgo

// Package gpu provides GPU-accelerated ZK operations via unified luxcpp/gpu.
// Automatically selects best backend: CUDA > Metal > WebGPU > CPU
package gpu

/*
#cgo pkg-config: lux-gpu
#include <lux/gpu/gpu.h>
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"sync"
	"unsafe"
)

// Error codes from C API
var (
	ErrZKInvalidArg = errors.New("zk: invalid argument")
	ErrZKSize       = errors.New("zk: invalid size")
	ErrZKAlloc      = errors.New("zk: allocation failed")
	ErrZKNotImpl    = errors.New("zk: not implemented")
	ErrZKNoDevice   = errors.New("zk: no GPU device available")
)

// Fr256 represents a 256-bit field element (BN254 scalar field).
// Uses 4 x 64-bit limbs in little-endian order.
type Fr256 [4]uint64

// Global GPU context
var (
	gpuOnce sync.Once
	gpuCtx  *C.LuxGPU
)

func getGPU() *C.LuxGPU {
	gpuOnce.Do(func() {
		gpuCtx = C.lux_gpu_create()
	})
	return gpuCtx
}

// ZKGPUAvailable returns true if GPU acceleration is available.
func ZKGPUAvailable() bool {
	gpu := getGPU()
	if gpu == nil {
		return false
	}
	backend := C.lux_gpu_backend(gpu)
	return backend != C.LUX_GPU_BACKEND_CPU
}

// ZKGetBackend returns the active backend name.
func ZKGetBackend() string {
	gpu := getGPU()
	if gpu == nil {
		return "None"
	}
	return C.GoString(C.lux_gpu_backend_name(gpu))
}

// ZKGetDeviceName returns the GPU device name.
func ZKGetDeviceName() string {
	gpu := getGPU()
	if gpu == nil {
		return "None"
	}
	return C.GoString(C.lux_gpu_device_name(gpu))
}

// ZKGetThreshold returns the recommended batch size threshold for GPU.
// opType: 1=Poseidon2, 2=Merkle, 3=MSM, 4=Commitment, 5=FRI
func ZKGetThreshold(opType int) int {
	switch opType {
	case 1:
		return C.LUX_ZK_THRESHOLD_POSEIDON2
	case 2:
		return C.LUX_ZK_THRESHOLD_MERKLE
	case 3:
		return C.LUX_ZK_THRESHOLD_MSM
	case 4:
		return C.LUX_ZK_THRESHOLD_COMMITMENT
	default:
		return 64
	}
}

// zkError converts C error code to Go error.
func zkError(code C.int) error {
	switch code {
	case C.LUX_GPU_OK:
		return nil
	case C.LUX_GPU_ERROR_INVALID_ARGS:
		return ErrZKInvalidArg
	case C.LUX_GPU_ERROR_NO_DEVICE:
		return ErrZKNoDevice
	case C.LUX_GPU_ERROR_OUT_OF_MEMORY:
		return ErrZKAlloc
	default:
		return errors.New("zk: unknown error")
	}
}

// Poseidon2Hash computes batch Poseidon2 hashes.
// Returns hash outputs where out[i] = Poseidon2(left[i], right[i]).
func Poseidon2Hash(left, right []Fr256) ([]Fr256, error) {
	n := len(left)
	if n != len(right) {
		return nil, ErrZKInvalidArg
	}
	if n == 0 {
		return nil, nil
	}

	gpu := getGPU()
	out := make([]Fr256, n)

	ret := C.lux_gpu_poseidon2(
		gpu,
		(*C.LuxFr256)(unsafe.Pointer(&out[0])),
		(*C.LuxFr256)(unsafe.Pointer(&left[0])),
		(*C.LuxFr256)(unsafe.Pointer(&right[0])),
		C.size_t(n),
	)

	if err := zkError(ret); err != nil {
		return nil, err
	}
	return out, nil
}

// MerkleLayer computes one layer of a Poseidon2 Merkle tree.
// Input nodes must have even count.
func MerkleLayer(nodes []Fr256) ([]Fr256, error) {
	n := len(nodes)
	if n == 0 || n%2 != 0 {
		return nil, ErrZKInvalidArg
	}

	// Split into left and right halves
	half := n / 2
	left := nodes[:half]
	right := nodes[half:]

	return Poseidon2Hash(left, right)
}

// MerkleRoot computes Merkle root from leaves.
// Leaves are padded to power of 2 if needed.
func MerkleRoot(leaves []Fr256) (Fr256, error) {
	n := len(leaves)
	if n == 0 {
		return Fr256{}, ErrZKInvalidArg
	}

	gpu := getGPU()
	var out Fr256

	ret := C.lux_gpu_merkle_root(
		gpu,
		(*C.LuxFr256)(unsafe.Pointer(&out)),
		(*C.LuxFr256)(unsafe.Pointer(&leaves[0])),
		C.size_t(n),
	)

	if err := zkError(ret); err != nil {
		return Fr256{}, err
	}
	return out, nil
}

// MerkleTree builds complete Merkle tree and returns all internal nodes.
// Returns n-1 internal nodes for n leaves.
func MerkleTree(leaves []Fr256) ([]Fr256, error) {
	n := len(leaves)
	if n == 0 {
		return nil, ErrZKInvalidArg
	}
	if n == 1 {
		return []Fr256{leaves[0]}, nil
	}

	// Pad to power of 2
	pow2 := 1
	for pow2 < n {
		pow2 <<= 1
	}

	padded := make([]Fr256, pow2)
	copy(padded, leaves)

	// Build tree bottom-up
	tree := make([]Fr256, 0, pow2-1)
	current := padded

	for len(current) > 1 {
		layer, err := MerkleLayer(current)
		if err != nil {
			return nil, err
		}
		tree = append(tree, layer...)
		current = layer
	}

	return tree, nil
}

// BatchCommitment computes batch Poseidon2 commitments.
// commitment[i] = Poseidon2(Poseidon2(values[i], blindings[i]), salts[i])
func BatchCommitment(values, blindings, salts []Fr256) ([]Fr256, error) {
	n := len(values)
	if n != len(blindings) || n != len(salts) {
		return nil, ErrZKInvalidArg
	}
	if n == 0 {
		return nil, nil
	}

	gpu := getGPU()
	out := make([]Fr256, n)

	ret := C.lux_gpu_commitment(
		gpu,
		(*C.LuxFr256)(unsafe.Pointer(&out[0])),
		(*C.LuxFr256)(unsafe.Pointer(&values[0])),
		(*C.LuxFr256)(unsafe.Pointer(&blindings[0])),
		(*C.LuxFr256)(unsafe.Pointer(&salts[0])),
		C.size_t(n),
	)

	if err := zkError(ret); err != nil {
		return nil, err
	}
	return out, nil
}

// BatchNullifier computes batch Poseidon2 nullifiers.
// nullifier[i] = Poseidon2(Poseidon2(keys[i], commitments[i]), indices[i])
func BatchNullifier(keys, commitments, indices []Fr256) ([]Fr256, error) {
	n := len(keys)
	if n != len(commitments) || n != len(indices) {
		return nil, ErrZKInvalidArg
	}
	if n == 0 {
		return nil, nil
	}

	// Use commitment with keys as values, commitments as blindings, indices as salts
	return BatchCommitment(keys, commitments, indices)
}

// Sync waits for all GPU operations to complete.
func Sync() error {
	gpu := getGPU()
	ret := C.lux_gpu_sync(gpu)
	return zkError(ret)
}

// GetMemoryInfo returns total and free GPU memory in bytes.
func GetMemoryInfo() (total, free uint64) {
	gpu := getGPU()
	return uint64(C.lux_gpu_memory_total(gpu)), uint64(C.lux_gpu_memory_free(gpu))
}
