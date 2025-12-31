// +build !cgo

package gpu

// This file provides a clear error when trying to build without CGO

func init() {
	panic("MLX requires CGO. Build with CGO_ENABLED=1")
}