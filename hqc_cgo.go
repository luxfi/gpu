// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo

// HQC (Hamming Quasi-Cyclic) GPU bindings via the luxcpp/gpu native
// kernels (libluxgpu_hqc.a). Symbols come from
// luxcpp/gpu/include/lux/gpu/hqc.h; the canonical CPU implementation
// in luxcpp/gpu/src/hqc_*.cpp is the byte-equal oracle every backend
// (Metal / CUDA / WebGPU) must match.
//
// Layering:
//
//	luxfi/crypto/hqc   <-- Go consumer (this is what callers import)
//	luxfi/gpu (here)   <-- thin C ABI binding, threshold routing
//	luxfi/accel        <-- session / device lifetime
//	luxcpp/gpu/src     <-- canonical CPU + Metal/CUDA dispatch
//
// Build expects the luxfi sibling-repo layout:
//
//	~/work/lux/mlx -> ~/work/luxcpp/gpu (symlink)
//
// so that ${SRCDIR}/../mlx/include resolves to the lux/gpu headers.
// libluxgpu_hqc.a is statically linked from the same build dir.
//
// Note on the `randombytes` symbol: libluxgpu_hqc.a has unresolved
// references to PQClean's `randombytes(uint8_t*, size_t)`. The Lux
// build system intentionally splits the provider out into a separate
// archive (libluxgpu_hqc_standalone_rng.a) so binaries that supply
// their own `randombytes` (luxfi/crypto/hqc's randombytes_shim.c
// under build tag hqc_pqclean, or luxfi/accel/ops/code's
// code_cpu_randombytes.go strong definition) can avoid linking the
// shim and get a clean single definition. Linking the shim in here
// unconditionally collides with those providers. Any consumer that
// imports luxfi/gpu's HQC entry points reaches them through
// luxfi/accel or luxfi/crypto/hqc, both of which provide
// `randombytes`. So we leave the symbol unresolved at the gpu layer
// and let the higher-level package supply it.
//
// Threshold routing: every entry point inspects the batch count and
// HQCGetThreshold(opType). Below threshold callers should keep the
// op on the CPU pure-Go path in crypto/hqc/gpu_nocgo.go to avoid the
// cgo + device dispatch overhead. The threshold is *advisory*; the
// C kernel is correct for any count >= 1.
package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/../mlx/include
#cgo darwin LDFLAGS: -L${SRCDIR}/../mlx/build -lluxgpu_hqc -lc++ -framework Security
#cgo !darwin LDFLAGS: -L${SRCDIR}/../mlx/build -lluxgpu_hqc -lstdc++ -lm

#include <stdint.h>
#include <stddef.h>
#include "lux/gpu/hqc.h"
*/
import "C"

import (
	"errors"
	"unsafe"
)

// ErrHQCNotAvailable is returned when an HQC entry point hits a
// condition the CGO bindings can't satisfy (mode mismatch, buffer
// length, RNG starvation). Same sentinel value the non-CGO stub
// exports so callers can switch on it uniformly.
var ErrHQCNotAvailable = errors.New("HQC operation not available")

// Granular HQC error sentinels — produced by the C kernels and
// translated here. Callers normally only need ErrHQCNotAvailable but
// these allow precise error handling (e.g. surface RNG starvation
// distinctly from buffer length mismatch).
var (
	ErrHQCInvalidArg    = errors.New("hqc: invalid argument")
	ErrHQCInvalidMode   = errors.New("hqc: invalid mode")
	ErrHQCInvalidLength = errors.New("hqc: invalid buffer length")
	ErrHQCOOM           = errors.New("hqc: out of memory")
	ErrHQCDeviceError   = errors.New("hqc: device error")
	ErrHQCNotSupported  = errors.New("hqc: kernel not supported on this backend")
	ErrHQCRNGFailed     = errors.New("hqc: RNG / seed buffer exhausted")
)

// HQCMode selects the parameter set. Bit pattern matches the C
// LuxHQCMode enum.
type HQCMode int

const (
	HQC128 HQCMode = HQCMode(C.LUX_HQC_128)
	HQC192 HQCMode = HQCMode(C.LUX_HQC_192)
	HQC256 HQCMode = HQCMode(C.LUX_HQC_256)
)

// HQCParams describes the byte sizes for one parameter set. Values
// come from the C library so they cannot drift from the canonical
// PQClean reference.
type HQCParams struct {
	PublicKeyBytes    int
	SecretKeyBytes    int
	CiphertextBytes   int
	SharedSecretBytes int
	ParamN            int
	VecNSize64        int
	ParamN1           int
	ParamN2           int
	ParamK            int
}

// HQCParamsFor returns the canonical sizes for mode. Wraps
// lux_hqc_params() in the C header.
func HQCParamsFor(mode HQCMode) (HQCParams, error) {
	var cp C.LuxHQCParams
	rc := C.lux_hqc_params(C.LuxHQCMode(mode), &cp)
	if rc != C.LUX_HQC_OK {
		return HQCParams{}, hqcErrorFromCode(rc)
	}
	return HQCParams{
		PublicKeyBytes:    int(cp.public_key_bytes),
		SecretKeyBytes:    int(cp.secret_key_bytes),
		CiphertextBytes:   int(cp.ciphertext_bytes),
		SharedSecretBytes: int(cp.shared_secret_bytes),
		ParamN:            int(cp.param_n),
		VecNSize64:        int(cp.vec_n_size_64),
		ParamN1:           int(cp.param_n1),
		ParamN2:           int(cp.param_n2),
		ParamK:            int(cp.param_k),
	}, nil
}

// hqcErrorFromCode translates LuxHQCError to a Go sentinel.
func hqcErrorFromCode(rc C.LuxHQCError) error {
	switch rc {
	case C.LUX_HQC_OK:
		return nil
	case C.LUX_HQC_ERROR_INVALID_ARG:
		return ErrHQCInvalidArg
	case C.LUX_HQC_ERROR_INVALID_MODE:
		return ErrHQCInvalidMode
	case C.LUX_HQC_ERROR_INVALID_LENGTH:
		return ErrHQCInvalidLength
	case C.LUX_HQC_ERROR_OOM:
		return ErrHQCOOM
	case C.LUX_HQC_ERROR_DEVICE_ERROR:
		return ErrHQCDeviceError
	case C.LUX_HQC_ERROR_NOT_SUPPORTED:
		return ErrHQCNotSupported
	case C.LUX_HQC_ERROR_RNG_FAILED:
		return ErrHQCRNGFailed
	default:
		return ErrHQCNotAvailable
	}
}

// HQCGPUAvailable reports whether the HQC kernels are reachable.
// The C library is linked statically so this is always true when
// CGO is enabled. (Returns false on the !cgo stub.)
func HQCGPUAvailable() bool { return true }

// HQCGetBackend returns the active HQC backend. Right now every
// platform routes through the canonical CPU oracle in
// luxcpp/gpu/src/hqc_*.cpp; Metal/CUDA shims will surface here once
// they override the lux_hqc_* C symbols.
func HQCGetBackend() string { return "CPU/Native" }

// HQCGetThreshold returns the per-op batch crossover above which the
// GPU/native path is preferred. Below the threshold callers should
// keep the op on the CPU pure-Go path to avoid the cgo + device
// dispatch overhead.
//
//	opType 1  GF(2)^N polynomial multiplication
//	opType 2  Reed-Solomon decode
//	opType 3  Reed-Muller decode
//	opType 4  KEM batch encaps
//	opType 5  KEM batch decaps
func HQCGetThreshold(opType int) int {
	switch opType {
	case 1:
		return 32
	case 2, 3:
		return 64
	case 4, 5:
		return 8
	default:
		return 32
	}
}

// vecN returns the per-slot uint64 count for mode's GF(2)^N polymul.
// Mirrors lux_hqc_params(mode).vec_n_size_64 without the cgo call.
func vecN(mode HQCMode) (int, error) {
	switch mode {
	case HQC128:
		return (17669 + 63) / 64, nil // 277
	case HQC192:
		return (35851 + 63) / 64, nil // 561
	case HQC256:
		return (57637 + 63) / 64, nil // 901
	default:
		return 0, ErrHQCInvalidMode
	}
}

// rsDims returns (PARAM_N1, PARAM_K) for mode's Reed-Solomon code.
func rsDims(mode HQCMode) (int, int, error) {
	switch mode {
	case HQC128:
		return 46, 16, nil
	case HQC192:
		return 56, 24, nil
	case HQC256:
		return 90, 32, nil
	default:
		return 0, 0, ErrHQCInvalidMode
	}
}

// vecN1N2Size64 returns the bit-packed RM codeword size in uint64s.
func vecN1N2Size64(mode HQCMode) (int, error) {
	switch mode {
	case HQC128:
		return (17664 + 63) / 64, nil // 276
	case HQC192:
		return (35840 + 63) / 64, nil // 560
	case HQC256:
		return (57600 + 63) / 64, nil // 900
	default:
		return 0, ErrHQCInvalidMode
	}
}

// GF2PolymulMod computes c(x) = a(x) * b(x) mod (X^n - 1) in GF(2)[X]
// for ONE polynomial. All three buffers must be length VecNSize64
// (per HQCParamsFor(mode)); they must be distinct (no aliasing). The
// kernel is constant-time and identical to PQClean's vect_mul.
func GF2PolymulMod(mode HQCMode, c, a, b []uint64) error {
	n, err := vecN(mode)
	if err != nil {
		return err
	}
	if len(a) != n || len(b) != n || len(c) != n {
		return ErrHQCInvalidLength
	}
	rc := C.lux_hqc_gf2_polymul(
		C.LuxHQCMode(mode),
		(*C.uint64_t)(unsafe.Pointer(&c[0])),
		(*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
	)
	return hqcErrorFromCode(rc)
}

// GF2PolymulBatch computes count independent polymuls in one call.
// Slot i reads a[i*N:(i+1)*N] and b[i*N:(i+1)*N], writes
// c[i*N:(i+1)*N]. Failure short-circuits and returns the slot's
// error code (no partial results are guaranteed).
func GF2PolymulBatch(mode HQCMode, c, a, b []uint64, count int) error {
	if count <= 0 {
		return ErrHQCInvalidArg
	}
	n, err := vecN(mode)
	if err != nil {
		return err
	}
	if len(a) != count*n || len(b) != count*n || len(c) != count*n {
		return ErrHQCInvalidLength
	}
	cMode := C.LuxHQCMode(mode)
	for i := 0; i < count; i++ {
		ai := a[i*n : (i+1)*n]
		bi := b[i*n : (i+1)*n]
		ci := c[i*n : (i+1)*n]
		rc := C.lux_hqc_gf2_polymul(
			cMode,
			(*C.uint64_t)(unsafe.Pointer(&ci[0])),
			(*C.uint64_t)(unsafe.Pointer(&ai[0])),
			(*C.uint64_t)(unsafe.Pointer(&bi[0])),
		)
		if rc != C.LUX_HQC_OK {
			return hqcErrorFromCode(rc)
		}
	}
	return nil
}

// ReedSolomonDecode runs constant-time RS decoding on one codeword.
// cdw is ParamN1 bytes, msg is ParamK bytes.
func ReedSolomonDecode(mode HQCMode, msg, cdw []byte) error {
	n1, k, err := rsDims(mode)
	if err != nil {
		return err
	}
	if len(cdw) != n1 || len(msg) != k {
		return ErrHQCInvalidLength
	}
	rc := C.lux_hqc_reed_solomon_decode(
		C.LuxHQCMode(mode),
		(*C.uint8_t)(unsafe.Pointer(&msg[0])),
		(*C.uint8_t)(unsafe.Pointer(&cdw[0])),
	)
	return hqcErrorFromCode(rc)
}

// ReedMullerDecode runs duplicated RM(1,7) decoding via Fast Hadamard
// Transform. cdw is the bit-packed VEC_N1N2_SIZE_64 uint64 array,
// msg is ParamN1 bytes.
func ReedMullerDecode(mode HQCMode, msg []byte, cdw []uint64) error {
	n1, _, err := rsDims(mode)
	if err != nil {
		return err
	}
	wantN1N2, err := vecN1N2Size64(mode)
	if err != nil {
		return err
	}
	if len(msg) != n1 || len(cdw) != wantN1N2 {
		return ErrHQCInvalidLength
	}
	rc := C.lux_hqc_reed_muller_decode(
		C.LuxHQCMode(mode),
		(*C.uint8_t)(unsafe.Pointer(&msg[0])),
		(*C.uint64_t)(unsafe.Pointer(&cdw[0])),
	)
	return hqcErrorFromCode(rc)
}

// HQCKeypairBatch generates count HQC keypairs deterministically
// from the seed buffer.
//
//	pks   layout count * params.public_key_bytes
//	sks   layout count * params.secret_key_bytes
//	seeds layout count * 48 bytes
func HQCKeypairBatch(mode HQCMode, pks, sks, seeds []byte, count int) error {
	if count <= 0 {
		return ErrHQCInvalidArg
	}
	p, err := HQCParamsFor(mode)
	if err != nil {
		return err
	}
	if len(pks) != count*p.PublicKeyBytes || len(sks) != count*p.SecretKeyBytes || len(seeds) != count*48 {
		return ErrHQCInvalidLength
	}
	rc := C.lux_hqc_keypair_batch(
		C.LuxHQCMode(mode),
		(*C.uint8_t)(unsafe.Pointer(&pks[0])),
		(*C.uint8_t)(unsafe.Pointer(&sks[0])),
		(*C.uint8_t)(unsafe.Pointer(&seeds[0])),
		C.size_t(count),
	)
	return hqcErrorFromCode(rc)
}

// HQCEncapsBatch runs count independent encapsulations.
//
//	cts   layout count * params.ciphertext_bytes
//	sss   layout count * params.shared_secret_bytes
//	pks   layout count * params.public_key_bytes
//	seeds layout count * 32 bytes (PARAM_K + 16 salt, padded to 32)
func HQCEncapsBatch(mode HQCMode, cts, sss, pks, seeds []byte, count int) error {
	if count <= 0 {
		return ErrHQCInvalidArg
	}
	p, err := HQCParamsFor(mode)
	if err != nil {
		return err
	}
	if len(cts) != count*p.CiphertextBytes ||
		len(sss) != count*p.SharedSecretBytes ||
		len(pks) != count*p.PublicKeyBytes ||
		len(seeds) != count*32 {
		return ErrHQCInvalidLength
	}
	rc := C.lux_hqc_encaps_batch(
		C.LuxHQCMode(mode),
		(*C.uint8_t)(unsafe.Pointer(&cts[0])),
		(*C.uint8_t)(unsafe.Pointer(&sss[0])),
		(*C.uint8_t)(unsafe.Pointer(&pks[0])),
		(*C.uint8_t)(unsafe.Pointer(&seeds[0])),
		C.size_t(count),
	)
	return hqcErrorFromCode(rc)
}

// HQCDecapsBatch runs count independent decapsulations.
//
//	sss layout count * params.shared_secret_bytes
//	cts layout count * params.ciphertext_bytes
//	sks layout count * params.secret_key_bytes
//
// Implicit rejection: a tampered ciphertext does NOT error; the
// corresponding sss slot receives a pseudorandom 64 bytes derived
// from (sk, ct) (Hofheinz-Hövelmanns-Kiltz construction).
func HQCDecapsBatch(mode HQCMode, sss, cts, sks []byte, count int) error {
	if count <= 0 {
		return ErrHQCInvalidArg
	}
	p, err := HQCParamsFor(mode)
	if err != nil {
		return err
	}
	if len(sss) != count*p.SharedSecretBytes ||
		len(cts) != count*p.CiphertextBytes ||
		len(sks) != count*p.SecretKeyBytes {
		return ErrHQCInvalidLength
	}
	rc := C.lux_hqc_decaps_batch(
		C.LuxHQCMode(mode),
		(*C.uint8_t)(unsafe.Pointer(&sss[0])),
		(*C.uint8_t)(unsafe.Pointer(&cts[0])),
		(*C.uint8_t)(unsafe.Pointer(&sks[0])),
		C.size_t(count),
	)
	return hqcErrorFromCode(rc)
}
