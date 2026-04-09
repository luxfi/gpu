//go:build !cgo

package gpu

import "errors"

// BatchEcrecover is not available without CGO.
func BatchEcrecover(_ []Signature) ([]EcrecoverResult, error) {
	return nil, errors.New("gpu: ecrecover requires CGO")
}
