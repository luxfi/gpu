// Package gpu provides GPU-accelerated ecrecover for batch tx signature verification.
package gpu

// Signature holds the components needed for secp256k1 ecrecover.
type Signature struct {
	R       [32]byte // r component (big-endian, left-padded)
	S       [32]byte // s component (big-endian, left-padded)
	V       uint8    // recovery id (0 or 1)
	MsgHash [32]byte // Keccak256 signing hash
}

// EcrecoverResult holds the output of a single ecrecover operation.
type EcrecoverResult struct {
	Address [20]byte // recovered Ethereum address
	Valid   bool     // true if recovery succeeded
}
