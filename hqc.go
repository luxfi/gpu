// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo

// Package gpu HQC (Hamming Quasi-Cyclic) bindings — non-CGO stub.
//
// HQC is the NIST PQC Round-4 selected code-based KEM (NIST IR 8528,
// March 2025). Family-disjoint from ML-KEM: a structural break against
// MLWE does NOT compromise HQC and vice-versa.
//
// The CGO build (hqc_cgo.go) binds to luxcpp/gpu's lux_hqc_* C API.
// This file is the compile-without-cgo stub. Every entry point
// returns ErrHQCNotAvailable so the package keeps compiling on hosts
// without a C toolchain. Real CPU and GPU coverage flows from
// crypto/hqc/gpu_{cgo,nocgo}.go and accel/ops/code/.
package gpu

import "errors"

// ErrHQCNotAvailable is returned by every HQC entry point when CGO is
// disabled. Callers are expected to fall back to the pure-Go bit-
// sliced CPU path in crypto/hqc/gpu_nocgo.go.
var ErrHQCNotAvailable = errors.New("HQC operations require CGO")

// HQCMode selects the HQC parameter set. Bit pattern matches
// luxcpp/gpu/include/lux/gpu/hqc.h's LuxHQCMode.
type HQCMode int

const (
	// HQC128 — NIST PQ Category 1 (~AES-128).
	HQC128 HQCMode = 0
	// HQC192 — NIST PQ Category 3 (~AES-192).
	HQC192 HQCMode = 1
	// HQC256 — NIST PQ Category 5 (~AES-256).
	HQC256 HQCMode = 2
)

// HQCParams describes the byte sizes for one parameter set.
type HQCParams struct {
	PublicKeyBytes    int
	SecretKeyBytes    int
	CiphertextBytes   int
	SharedSecretBytes int
	ParamN            int // ring size in bits
	VecNSize64        int // CEIL(ParamN / 64)
	ParamN1           int // Reed-Solomon code length (bytes)
	ParamN2           int // Reed-Muller code length per RS symbol (bits)
	ParamK            int // message length (bytes)
}

// HQCParamsFor returns the canonical sizes for mode. Mirrors the
// values served by lux_hqc_params() in the C API. Constants are
// NIST-fixed and must not drift.
func HQCParamsFor(mode HQCMode) (HQCParams, error) {
	switch mode {
	case HQC128:
		return HQCParams{
			PublicKeyBytes:    2249,
			SecretKeyBytes:    2305,
			CiphertextBytes:   4433,
			SharedSecretBytes: 64,
			ParamN:            17669,
			VecNSize64:        (17669 + 63) / 64,
			ParamN1:           46,
			ParamN2:           384,
			ParamK:            16,
		}, nil
	case HQC192:
		return HQCParams{
			PublicKeyBytes:    4522,
			SecretKeyBytes:    4586,
			CiphertextBytes:   8978,
			SharedSecretBytes: 64,
			ParamN:            35851,
			VecNSize64:        (35851 + 63) / 64,
			ParamN1:           56,
			ParamN2:           640,
			ParamK:            24,
		}, nil
	case HQC256:
		return HQCParams{
			PublicKeyBytes:    7245,
			SecretKeyBytes:    7317,
			CiphertextBytes:   14421,
			SharedSecretBytes: 64,
			ParamN:            57637,
			VecNSize64:        (57637 + 63) / 64,
			ParamN1:           90,
			ParamN2:           640,
			ParamK:            32,
		}, nil
	default:
		return HQCParams{}, ErrHQCNotAvailable
	}
}

// HQCGPUAvailable reports whether GPU-backed HQC kernels are
// available (always false without CGO).
func HQCGPUAvailable() bool { return false }

// HQCGetBackend returns the active HQC backend ("CPU" without CGO).
func HQCGetBackend() string { return "CPU" }

// HQCGetThreshold returns the per-op batch crossover above which the
// GPU path is preferred. Below the threshold a single-shot CPU path
// is faster (avoids the cgo + device dispatch overhead).
//
//	opType 1  GF(2)^N polynomial multiplication
//	opType 2  Reed-Solomon decode
//	opType 3  Reed-Muller decode
//	opType 4  KEM batch encaps
//	opType 5  KEM batch decaps
func HQCGetThreshold(opType int) int {
	switch opType {
	case 1:
		return 32 // GF2Polymul: dispatch wins above 32 polys/batch
	case 2:
		return 64 // RS decode: small inner kernel, lots of slots needed
	case 3:
		return 64 // RM decode: ditto
	case 4:
		return 8 // KEM encaps: ~30 us per item, dispatch amortises early
	case 5:
		return 8 // KEM decaps
	default:
		return 32
	}
}

// GF2PolymulMod computes c(x) = a(x) * b(x) mod (X^n - 1) in GF(2)[X]
// for one polynomial. All three buffers MUST be length VecNSize64
// (per HQCParamsFor(mode)). Without CGO this returns ErrHQCNotAvailable.
func GF2PolymulMod(_ HQCMode, _, _, _ []uint64) error {
	return ErrHQCNotAvailable
}

// GF2PolymulBatch computes count independent polymuls in one call.
// Slot i reads a[i*N:(i+1)*N] and b[i*N:(i+1)*N], writes c[i*N:(i+1)*N].
func GF2PolymulBatch(_ HQCMode, _, _, _ []uint64, _ int) error {
	return ErrHQCNotAvailable
}

// ReedSolomonDecode runs constant-time RS decoding on one codeword.
// In bytes: cdw is ParamN1 bytes, msg is ParamK bytes.
func ReedSolomonDecode(_ HQCMode, _, _ []byte) error {
	return ErrHQCNotAvailable
}

// ReedMullerDecode runs duplicated RM(1,7) decoding via Fast Hadamard
// Transform. cdw is the bit-packed input (VEC_N1N2_SIZE_64 uint64s),
// msg is ParamN1 bytes.
func ReedMullerDecode(_ HQCMode, _ []byte, _ []uint64) error {
	return ErrHQCNotAvailable
}

// HQCKeypairBatch generates count HQC keypairs from a contiguous
// seed buffer. See HQCParamsFor for size requirements.
func HQCKeypairBatch(_ HQCMode, _, _, _ []byte, _ int) error {
	return ErrHQCNotAvailable
}

// HQCEncapsBatch runs count encapsulations against count public keys.
func HQCEncapsBatch(_ HQCMode, _, _, _, _ []byte, _ int) error {
	return ErrHQCNotAvailable
}

// HQCDecapsBatch runs count decapsulations. Implicit rejection: a
// tampered ciphertext yields a pseudorandom 64-byte secret, never
// surfaces as an error.
func HQCDecapsBatch(_ HQCMode, _, _, _ []byte, _ int) error {
	return ErrHQCNotAvailable
}
