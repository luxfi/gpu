//go:build cgo

package gpu

/*
#cgo pkg-config: lux-gpu
#include <lux/gpu.h>
#include <string.h>
*/
import "C"
import "unsafe"

// BatchEcrecover recovers Ethereum addresses from a batch of signatures
// using GPU-accelerated secp256k1 ecrecover.
func BatchEcrecover(sigs []Signature) ([]EcrecoverResult, error) {
	n := len(sigs)
	if n == 0 {
		return nil, nil
	}

	gpu := getGPU()

	// Pack into C structs
	inputs := make([]C.LuxEcrecoverInput, n)
	outputs := make([]C.LuxEcrecoverOutput, n)

	for i := range sigs {
		C.memcpy(unsafe.Pointer(&inputs[i].r[0]), unsafe.Pointer(&sigs[i].R[0]), 32)
		C.memcpy(unsafe.Pointer(&inputs[i].s[0]), unsafe.Pointer(&sigs[i].S[0]), 32)
		inputs[i].v = C.uint8_t(sigs[i].V)
		C.memcpy(unsafe.Pointer(&inputs[i].msg_hash[0]), unsafe.Pointer(&sigs[i].MsgHash[0]), 32)
	}

	ret := C.lux_gpu_ecrecover_batch(
		gpu,
		&inputs[0],
		&outputs[0],
		C.size_t(n),
	)
	if err := zkError(ret); err != nil {
		return nil, err
	}

	results := make([]EcrecoverResult, n)
	for i := range outputs {
		copy(results[i].Address[:], C.GoBytes(unsafe.Pointer(&outputs[i].address[0]), 20))
		results[i].Valid = outputs[i].valid != 0
	}
	return results, nil
}
