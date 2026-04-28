//go:build !cgo

package gpu

import "errors"

// BatchEcrecover recovers Ethereum addresses for a batch of signatures.
//
// Without CGO the GPU/native backend is unavailable. Empty and nil inputs
// short-circuit (callers use this to skip empty blocks without paying the
// cost of an FFI hop). A non-empty input returns ErrEcrecoverNoCGO so the
// caller can fall back to a pure-Go ecrecover path.
func BatchEcrecover(sigs []Signature) ([]EcrecoverResult, error) {
	if len(sigs) == 0 {
		return nil, nil
	}
	return nil, ErrEcrecoverNoCGO
}

// ErrEcrecoverNoCGO is returned by BatchEcrecover on builds without CGO.
var ErrEcrecoverNoCGO = errors.New("gpu: BatchEcrecover requires CGO")
