//go:build !cgo

// Package gpu provides ZK operations stub when CGO is disabled.
package gpu

import "errors"

var ErrZKNotAvailable = errors.New("ZK operations require CGO")

// Error codes
var (
	ErrZKInvalidArg = errors.New("zk: invalid argument")
	ErrZKSize       = errors.New("zk: invalid size")
	ErrZKAlloc      = errors.New("zk: allocation failed")
	ErrZKNotImpl    = errors.New("zk: not implemented")
	ErrZKNoDevice   = errors.New("zk: no GPU device available")
)

// Fr256 represents a 256-bit field element (BN254 scalar field).
type Fr256 [4]uint64

// ZKGPUAvailable returns false when CGO is disabled.
func ZKGPUAvailable() bool { return false }

// ZKGetBackend returns "CPU" when CGO is disabled.
func ZKGetBackend() string { return "CPU" }

// ZKGetDeviceName returns "CPU" when CGO is disabled.
func ZKGetDeviceName() string { return "CPU" }

// ZKGetThreshold returns threshold for operation type.
func ZKGetThreshold(opType int) int {
	switch opType {
	case 1:
		return 64 // Poseidon2
	case 2:
		return 128 // Merkle
	case 3:
		return 256 // MSM
	case 4:
		return 128 // Commitment
	case 5:
		return 512 // FRI
	default:
		return 64
	}
}

// Poseidon2Hash computes batch Poseidon2 hashes.
func Poseidon2Hash(left, right []Fr256) ([]Fr256, error) {
	return nil, ErrZKNotAvailable
}

// MerkleLayer computes one layer of Merkle tree.
func MerkleLayer(nodes []Fr256) ([]Fr256, error) {
	return nil, ErrZKNotAvailable
}

// MerkleRoot computes Merkle root from leaves.
func MerkleRoot(leaves []Fr256) (Fr256, error) {
	return Fr256{}, ErrZKNotAvailable
}

// MerkleTree builds complete Merkle tree.
func MerkleTree(leaves []Fr256) ([]Fr256, error) {
	return nil, ErrZKNotAvailable
}

// BatchCommitment computes batch commitments.
func BatchCommitment(values, blindings, salts []Fr256) ([]Fr256, error) {
	return nil, ErrZKNotAvailable
}

// BatchNullifier computes batch nullifiers.
func BatchNullifier(keys, commitments, indices []Fr256) ([]Fr256, error) {
	return nil, ErrZKNotAvailable
}

// Sync waits for all GPU operations to complete.
func Sync() error {
	return nil
}

// GetMemoryInfo returns total and free GPU memory in bytes.
func GetMemoryInfo() (total, free uint64) {
	return 0, 0
}
