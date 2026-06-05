//go:build cgo

// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Cgo wrappers exercised by the end-to-end test files. Each wrapper calls a
// public lux_gpu C API function and returns a Go-friendly result. Test files
// (xxx_e2e_test.go) drive these wrappers — cgo is forbidden inside _test.go,
// so the wrappers live here.
//
// All wrappers are synchronous and copy buffers across the cgo boundary
// before returning, so callers do not need to keep Go slices pinned. The
// underlying C functions consume only pointers + lengths and produce
// completed results before returning, which is exactly the contract that
// allows the cgo runtime to relax pointer rules (see cmd/cgo's "Passing
// pointers" semantics, 1.21+).

package gpu

/*
#cgo pkg-config: lux-gpu
#include <lux/gpu.h>
#include <lux/gpu/crypto.h>
#include <stdlib.h>
#include <string.h>

// Weakly-linked signers exported by libluxgpu (LUX_GPU_LINK_CRYPTO=ON).
// We declare them with __attribute__((weak)) so the binary still links when
// the gpu library is built without the crypto bridge — at runtime we check
// the function pointer is non-null before invoking.
int ed25519_keygen(const unsigned char seed[32],
                   unsigned char sk[32], unsigned char pk[32])
    __attribute__((weak));
int ed25519_sign(const unsigned char sk[32], const unsigned char* msg,
                 size_t msg_len, unsigned char sig[64]) __attribute__((weak));
int ed25519_verify(const unsigned char pk[32], const unsigned char* msg,
                   size_t msg_len, const unsigned char sig[64])
    __attribute__((weak));

int sr25519_sign(const unsigned char sk[32], const unsigned char* msg,
                 size_t msg_len, unsigned char sig[64]) __attribute__((weak));
int sr25519_verify(const unsigned char pk[32], const unsigned char* msg,
                   size_t msg_len, const unsigned char sig[64])
    __attribute__((weak));

int mldsa_keygen(int mode, const unsigned char* seed,
                 unsigned char* pk, unsigned char* sk) __attribute__((weak));
int mldsa_sign(int mode, const unsigned char* sk,
               const unsigned char* msg, size_t msg_len,
               unsigned char* sig, size_t* sig_len) __attribute__((weak));
int mldsa_verify(int mode, const unsigned char* pk,
                 const unsigned char* msg, size_t msg_len,
                 const unsigned char* sig, size_t sig_len)
    __attribute__((weak));

int mlkem_keygen(int mode, const unsigned char seed[32],
                 unsigned char* pk, unsigned char* sk) __attribute__((weak));
int mlkem_encap(int mode, const unsigned char* pk,
                unsigned char* ct, unsigned char ss[32])
    __attribute__((weak));
int mlkem_decap(int mode, const unsigned char* sk,
                const unsigned char* ct, unsigned char ss[32])
    __attribute__((weak));

int slhdsa_keygen(int mode, const unsigned char seed[32],
                  unsigned char* pk, unsigned char* sk) __attribute__((weak));
int slhdsa_sign(int mode, const unsigned char* sk,
                const unsigned char* msg, size_t msg_len,
                unsigned char* sig, size_t* sig_len) __attribute__((weak));
int slhdsa_verify(int mode, const unsigned char* pk,
                  const unsigned char* msg, size_t msg_len,
                  const unsigned char* sig, size_t sig_len)
    __attribute__((weak));

// Helpers that return whether the weak symbol resolved at link/load time.
static int have_ed25519(void)  { return ed25519_keygen != 0 && ed25519_sign != 0; }
static int have_sr25519(void)  { return sr25519_sign  != 0 && sr25519_verify != 0; }
static int have_mldsa(void)    { return mldsa_keygen  != 0 && mldsa_sign != 0; }
static int have_mlkem(void)    { return mlkem_keygen  != 0 && mlkem_encap != 0 && mlkem_decap != 0; }
static int have_slhdsa(void)   { return slhdsa_keygen != 0 && slhdsa_sign != 0; }

// Indirect call wrappers — cgo can't take the address of weak symbols, but
// it CAN call them through a regular C function call.
static int call_ed25519_keygen(const unsigned char* seed, unsigned char* sk, unsigned char* pk) {
    return ed25519_keygen(seed, sk, pk);
}
static int call_ed25519_sign(const unsigned char* sk, const unsigned char* msg, size_t msg_len, unsigned char* sig) {
    return ed25519_sign(sk, msg, msg_len, sig);
}
static int call_sr25519_sign(const unsigned char* sk, const unsigned char* msg, size_t msg_len, unsigned char* sig) {
    return sr25519_sign(sk, msg, msg_len, sig);
}
static int call_sr25519_verify(const unsigned char* pk, const unsigned char* msg, size_t msg_len, const unsigned char* sig) {
    return sr25519_verify(pk, msg, msg_len, sig);
}
static int call_mldsa_keygen(int mode, const unsigned char* seed, unsigned char* pk, unsigned char* sk) {
    return mldsa_keygen(mode, seed, pk, sk);
}
static int call_mldsa_sign(int mode, const unsigned char* sk, const unsigned char* msg, size_t msg_len, unsigned char* sig, size_t* sig_len) {
    return mldsa_sign(mode, sk, msg, msg_len, sig, sig_len);
}
static int call_mlkem_keygen(int mode, const unsigned char* seed, unsigned char* pk, unsigned char* sk) {
    return mlkem_keygen(mode, seed, pk, sk);
}
static int call_mlkem_encap(int mode, const unsigned char* pk, unsigned char* ct, unsigned char* ss) {
    return mlkem_encap(mode, pk, ct, ss);
}
static int call_slhdsa_keygen(int mode, const unsigned char* seed, unsigned char* pk, unsigned char* sk) {
    return slhdsa_keygen(mode, seed, pk, sk);
}
static int call_slhdsa_sign(int mode, const unsigned char* sk, const unsigned char* msg, size_t msg_len, unsigned char* sig, size_t* sig_len) {
    return slhdsa_sign(mode, sk, msg, msg_len, sig, sig_len);
}
*/
import "C"

import (
	"errors"
	"unsafe"
)

// =============================================================================
// Hash wrappers
// =============================================================================

// keccak256BatchCgo calls lux_gpu_keccak256_batch over a flattened input
// buffer and per-message lengths. inputsConcat must be the concatenation of
// all messages in order; outputs are 32 bytes per message, contiguous.
func keccak256BatchCgo(inputsConcat []byte, lens []int) ([]byte, LuxErr) {
	n := len(lens)
	if n == 0 {
		return nil, errLuxOK
	}
	// Allocate scratch with a non-zero base address even when inputsConcat
	// is empty (e.g. all messages are length 0) — the C side reads only
	// past the lens-derived offsets so the pointer doesn't need to alias
	// real data when len=0.
	inBuf := inputsConcat
	if len(inBuf) == 0 {
		inBuf = []byte{0}
	}
	out := make([]byte, n*32)
	clens := make([]C.size_t, n)
	for i, l := range lens {
		clens[i] = C.size_t(l)
	}
	rc := C.lux_gpu_keccak256_batch(
		getGPU(),
		(*C.uint8_t)(unsafe.Pointer(&inBuf[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
		(*C.size_t)(unsafe.Pointer(&clens[0])),
		C.size_t(n),
	)
	return out, errFromLux(rc)
}

// blake3HashCgo calls lux_blake3_hash with a flattened input.
func blake3HashCgo(inputsConcat []byte, lens []int) ([]byte, LuxErr) {
	n := len(lens)
	if n == 0 {
		return nil, errLuxOK
	}
	inBuf := inputsConcat
	if len(inBuf) == 0 {
		inBuf = []byte{0}
	}
	out := make([]byte, n*32)
	clens := make([]C.size_t, n)
	for i, l := range lens {
		clens[i] = C.size_t(l)
	}
	rc := C.lux_blake3_hash(
		getGPU(),
		(*C.uint8_t)(unsafe.Pointer(&inBuf[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
		(*C.size_t)(unsafe.Pointer(&clens[0])),
		C.size_t(n),
	)
	return out, errFromLux(rc)
}

// poseidon2HashCgo calls lux_poseidon2_hash. The C API takes inputs as
// concatenated rate-sized blocks of u64-quad limbs; outputs are one u64 quad
// per input. We expose the lower-level call so tests can drive arbitrary
// rates (the typical rate is 2 for the BN254-Fr Poseidon2).
func poseidon2HashCgo(inputs []uint64, rate int, numHashes int) ([]uint64, LuxErr) {
	if numHashes == 0 {
		return nil, errLuxOK
	}
	out := make([]uint64, numHashes*4)
	rc := C.lux_poseidon2_hash(
		getGPU(),
		(*C.uint64_t)(unsafe.Pointer(&inputs[0])),
		(*C.uint64_t)(unsafe.Pointer(&out[0])),
		C.size_t(rate),
		C.size_t(numHashes),
	)
	return out, errFromLux(rc)
}

// =============================================================================
// Tensor wrappers (exercise the C tensor surface independently of the Array
// wrapper in gpu_cgo.go so test code can verify numerical results).
// =============================================================================

type cTensor struct {
	handle *C.LuxTensor
	shape  []int64
	dtype  C.LuxDtype
}

// tensorZeros wraps lux_tensor_zeros and returns a value-type holder. The
// caller is responsible for calling tensorDestroy on the returned tensor.
func tensorZeros(shape []int64, dtype C.LuxDtype) *cTensor {
	cshape := make([]C.int64_t, len(shape))
	for i, v := range shape {
		cshape[i] = C.int64_t(v)
	}
	h := C.lux_tensor_zeros(getGPU(), &cshape[0], C.int(len(shape)), dtype)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), shape...), dtype: dtype}
}

func tensorOnes(shape []int64, dtype C.LuxDtype) *cTensor {
	cshape := make([]C.int64_t, len(shape))
	for i, v := range shape {
		cshape[i] = C.int64_t(v)
	}
	h := C.lux_tensor_ones(getGPU(), &cshape[0], C.int(len(shape)), dtype)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), shape...), dtype: dtype}
}

func tensorFull(shape []int64, value float64, dtype C.LuxDtype) *cTensor {
	cshape := make([]C.int64_t, len(shape))
	for i, v := range shape {
		cshape[i] = C.int64_t(v)
	}
	h := C.lux_tensor_full(getGPU(), &cshape[0], C.int(len(shape)), dtype, C.double(value))
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), shape...), dtype: dtype}
}

// tensorFromFloat32 wraps lux_tensor_from_data with an LUX_FLOAT32 dtype.
func tensorFromFloat32(data []float32, shape []int64) *cTensor {
	cshape := make([]C.int64_t, len(shape))
	for i, v := range shape {
		cshape[i] = C.int64_t(v)
	}
	h := C.lux_tensor_from_data(getGPU(),
		unsafe.Pointer(&data[0]),
		&cshape[0], C.int(len(shape)),
		C.LUX_FLOAT32)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), shape...), dtype: C.LUX_FLOAT32}
}

func tensorAdd(a, b *cTensor) *cTensor {
	h := C.lux_tensor_add(getGPU(), a.handle, b.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), a.shape...), dtype: a.dtype}
}

func tensorSub(a, b *cTensor) *cTensor {
	h := C.lux_tensor_sub(getGPU(), a.handle, b.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), a.shape...), dtype: a.dtype}
}

func tensorMul(a, b *cTensor) *cTensor {
	h := C.lux_tensor_mul(getGPU(), a.handle, b.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), a.shape...), dtype: a.dtype}
}

func tensorDiv(a, b *cTensor) *cTensor {
	h := C.lux_tensor_div(getGPU(), a.handle, b.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), a.shape...), dtype: a.dtype}
}

func tensorMatMul(a, b *cTensor) *cTensor {
	h := C.lux_tensor_matmul(getGPU(), a.handle, b.handle)
	if h == nil {
		return nil
	}
	// Output shape: drop a's last dim, append b's last dim.
	out := append([]int64(nil), a.shape[:len(a.shape)-1]...)
	out = append(out, b.shape[len(b.shape)-1])
	return &cTensor{handle: h, shape: out, dtype: a.dtype}
}

// tensorSum reduces along all dims when axes is nil/empty.
func tensorSum(t *cTensor, axes []int) *cTensor {
	var h *C.LuxTensor
	if len(axes) == 0 {
		h = C.lux_tensor_sum(getGPU(), t.handle, nil, 0)
	} else {
		caxes := make([]C.int, len(axes))
		for i, a := range axes {
			caxes[i] = C.int(a)
		}
		h = C.lux_tensor_sum(getGPU(), t.handle, &caxes[0], C.int(len(axes)))
	}
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: []int64{1}, dtype: t.dtype}
}

func tensorMean(t *cTensor, axes []int) *cTensor {
	var h *C.LuxTensor
	if len(axes) == 0 {
		h = C.lux_tensor_mean(getGPU(), t.handle, nil, 0)
	} else {
		caxes := make([]C.int, len(axes))
		for i, a := range axes {
			caxes[i] = C.int(a)
		}
		h = C.lux_tensor_mean(getGPU(), t.handle, &caxes[0], C.int(len(axes)))
	}
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: []int64{1}, dtype: t.dtype}
}

func tensorReduceSum(t *cTensor) float32 {
	return float32(C.lux_tensor_reduce_sum(getGPU(), t.handle))
}
func tensorReduceMax(t *cTensor) float32 {
	return float32(C.lux_tensor_reduce_max(getGPU(), t.handle))
}
func tensorReduceMin(t *cTensor) float32 {
	return float32(C.lux_tensor_reduce_min(getGPU(), t.handle))
}
func tensorReduceMean(t *cTensor) float32 {
	return float32(C.lux_tensor_reduce_mean(getGPU(), t.handle))
}

func tensorNdim(t *cTensor) int             { return int(C.lux_tensor_ndim(t.handle)) }
func tensorShape(t *cTensor, dim int) int64 { return int64(C.lux_tensor_shape(t.handle, C.int(dim))) }
func tensorSize(t *cTensor) int64           { return int64(C.lux_tensor_size(t.handle)) }
func tensorDtype(t *cTensor) C.LuxDtype     { return C.lux_tensor_dtype(t.handle) }

// tensorToHostFloat32 copies the tensor's data back to a Go []float32 of the
// expected size.
func tensorToHostFloat32(t *cTensor, size int) ([]float32, LuxErr) {
	out := make([]float32, size)
	rc := C.lux_tensor_to_host(t.handle,
		unsafe.Pointer(&out[0]),
		C.size_t(size*4))
	return out, errFromLux(rc)
}

func tensorDestroy(t *cTensor) {
	if t != nil && t.handle != nil {
		C.lux_tensor_destroy(t.handle)
		t.handle = nil
	}
}

// =============================================================================
// Unary tensor ops
// =============================================================================

func tensorNeg(t *cTensor) *cTensor {
	h := C.lux_tensor_neg(getGPU(), t.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}
func tensorExp(t *cTensor) *cTensor {
	h := C.lux_tensor_exp(getGPU(), t.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}
func tensorLog(t *cTensor) *cTensor {
	h := C.lux_tensor_log(getGPU(), t.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}
func tensorSqrt(t *cTensor) *cTensor {
	h := C.lux_tensor_sqrt(getGPU(), t.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}
func tensorAbs(t *cTensor) *cTensor {
	h := C.lux_tensor_abs(getGPU(), t.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}
func tensorTanh(t *cTensor) *cTensor {
	h := C.lux_tensor_tanh(getGPU(), t.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}
func tensorSigmoid(t *cTensor) *cTensor {
	h := C.lux_tensor_sigmoid(getGPU(), t.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}
func tensorReLU(t *cTensor) *cTensor {
	h := C.lux_tensor_relu(getGPU(), t.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}
func tensorGELU(t *cTensor) *cTensor {
	h := C.lux_tensor_gelu(getGPU(), t.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}
func tensorTranspose(t *cTensor) *cTensor {
	h := C.lux_tensor_transpose(getGPU(), t.handle)
	if h == nil {
		return nil
	}
	out := make([]int64, len(t.shape))
	for i, d := range t.shape {
		out[len(t.shape)-1-i] = d
	}
	return &cTensor{handle: h, shape: out, dtype: t.dtype}
}
func tensorCopy(t *cTensor) *cTensor {
	h := C.lux_tensor_copy(getGPU(), t.handle)
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}

// tensorSoftmax exercises lux_tensor_softmax along the given axis.
func tensorSoftmax(t *cTensor, axis int) *cTensor {
	h := C.lux_tensor_softmax(getGPU(), t.handle, C.int(axis))
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}

func tensorLogSoftmax(t *cTensor, axis int) *cTensor {
	h := C.lux_tensor_log_softmax(getGPU(), t.handle, C.int(axis))
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}

// tensorLayerNorm wraps lux_tensor_layer_norm. gamma and beta are
// per-feature scale/shift tensors of shape [features].
func tensorLayerNorm(t, gamma, beta *cTensor, eps float32) *cTensor {
	h := C.lux_tensor_layer_norm(getGPU(), t.handle, gamma.handle, beta.handle, C.float(eps))
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}

// tensorRMSNorm wraps lux_tensor_rms_norm.
func tensorRMSNorm(t, weight *cTensor, eps float32) *cTensor {
	h := C.lux_tensor_rms_norm(getGPU(), t.handle, weight.handle, C.float(eps))
	if h == nil {
		return nil
	}
	return &cTensor{handle: h, shape: append([]int64(nil), t.shape...), dtype: t.dtype}
}

// =============================================================================
// MSM + KZG wrappers
// =============================================================================

// msmBN254 calls lux_msm with BN254 G1 affine points and 256-bit scalars.
// scalars must be n * 4 u64 words (little-endian limbs); points must be
// n * 12 u64 words (x[4] || y[4] || infinity_pad[4]).
func msmBN254(scalars []uint64, points []uint64, n int) ([]uint64, LuxErr) {
	// Output is one projective G1 point: 12 u64 words (x[4] || y[4] || z[4]).
	out := make([]uint64, 12)
	rc := C.lux_msm(
		getGPU(),
		unsafe.Pointer(&scalars[0]),
		unsafe.Pointer(&points[0]),
		unsafe.Pointer(&out[0]),
		C.size_t(n),
		C.LUX_CURVE_BN254,
	)
	return out, errFromLux(rc)
}

// =============================================================================
// FHE wrappers
// =============================================================================

// nttForward / nttInverse drive the modular NTT primitives in-place.
func nttForward(data []uint64, modulus uint64) LuxErr {
	rc := C.lux_ntt_forward(getGPU(),
		(*C.uint64_t)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
		C.uint64_t(modulus),
	)
	return errFromLux(rc)
}
func nttInverse(data []uint64, modulus uint64) LuxErr {
	rc := C.lux_ntt_inverse(getGPU(),
		(*C.uint64_t)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
		C.uint64_t(modulus),
	)
	return errFromLux(rc)
}

// polyMul performs negacyclic polynomial multiplication mod (X^n + 1) mod q.
func polyMul(a, b []uint64, modulus uint64) ([]uint64, LuxErr) {
	if len(a) != len(b) {
		return nil, errFromLux(C.LUX_ERROR_INVALID_ARGUMENT)
	}
	out := make([]uint64, len(a))
	rc := C.lux_poly_mul(getGPU(),
		(*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&out[0])),
		C.size_t(len(a)),
		C.uint64_t(modulus),
	)
	return out, errFromLux(rc)
}

// tfheBootstrap is a direct wrapper over lux_tfhe_bootstrap.
func tfheBootstrap(lweIn []uint64, bsk []uint64, testPoly []uint64,
	nLwe, N, k, l, baseLog uint32, q uint64) ([]uint64, LuxErr) {
	outLen := int(k)*int(N) + 1
	out := make([]uint64, outLen)
	rc := C.lux_tfhe_bootstrap(getGPU(),
		(*C.uint64_t)(unsafe.Pointer(&lweIn[0])),
		(*C.uint64_t)(unsafe.Pointer(&out[0])),
		(*C.uint64_t)(unsafe.Pointer(&bsk[0])),
		(*C.uint64_t)(unsafe.Pointer(&testPoly[0])),
		C.uint32_t(nLwe),
		C.uint32_t(N),
		C.uint32_t(k),
		C.uint32_t(l),
		C.uint32_t(baseLog),
		C.uint64_t(q),
	)
	return out, errFromLux(rc)
}

// tfheKeyswitch wraps lux_tfhe_keyswitch.
func tfheKeyswitch(lweIn []uint64, ksk []uint64,
	nIn, nOut, l, baseLog uint32, q uint64) ([]uint64, LuxErr) {
	out := make([]uint64, nOut+1)
	rc := C.lux_tfhe_keyswitch(getGPU(),
		(*C.uint64_t)(unsafe.Pointer(&lweIn[0])),
		(*C.uint64_t)(unsafe.Pointer(&out[0])),
		(*C.uint64_t)(unsafe.Pointer(&ksk[0])),
		C.uint32_t(nIn),
		C.uint32_t(nOut),
		C.uint32_t(l),
		C.uint32_t(baseLog),
		C.uint64_t(q),
	)
	return out, errFromLux(rc)
}

// blindRotate wraps lux_blind_rotate, operating in-place on acc.
func blindRotate(acc, bsk, lweA []uint64,
	nLwe, N, k, l, baseLog uint32, q uint64) LuxErr {
	rc := C.lux_blind_rotate(getGPU(),
		(*C.uint64_t)(unsafe.Pointer(&acc[0])),
		(*C.uint64_t)(unsafe.Pointer(&bsk[0])),
		(*C.uint64_t)(unsafe.Pointer(&lweA[0])),
		C.uint32_t(nLwe),
		C.uint32_t(N),
		C.uint32_t(k),
		C.uint32_t(l),
		C.uint32_t(baseLog),
		C.uint64_t(q),
	)
	return errFromLux(rc)
}

// =============================================================================
// FHE helpers (constant-time inspectors)
// =============================================================================

func fheIsValidN(N uint32) bool { return bool(C.lux_fhe_is_valid_N(C.uint32_t(N))) }

func fheIsValidGadget(l, baseLog uint32) bool {
	return bool(C.lux_fhe_is_valid_gadget(C.uint32_t(l), C.uint32_t(baseLog)))
}

func fheIsValidPBS(nLwe, N, k, l, baseLog uint32, q uint64) bool {
	return bool(C.lux_fhe_is_valid_pbs(
		C.uint32_t(nLwe), C.uint32_t(N), C.uint32_t(k),
		C.uint32_t(l), C.uint32_t(baseLog), C.uint64_t(q)))
}

func fheBskWords(nLwe, N, k, l uint32) int {
	return int(C.lux_fhe_bsk_words(C.uint32_t(nLwe), C.uint32_t(N), C.uint32_t(k), C.uint32_t(l)))
}

func fheKskWords(nIn, nOut, l, baseLog uint32) int {
	return int(C.lux_fhe_ksk_words(C.uint32_t(nIn), C.uint32_t(nOut), C.uint32_t(l), C.uint32_t(baseLog)))
}

func fheLweOutWords(N, k uint32) int {
	return int(C.lux_fhe_lwe_out_words(C.uint32_t(N), C.uint32_t(k)))
}

func fheAccWords(N, k uint32) int {
	return int(C.lux_fhe_acc_words(C.uint32_t(N), C.uint32_t(k)))
}

func fheSuggestBaseLog(l uint32, q uint64) uint32 {
	return uint32(C.lux_fhe_suggest_base_log(C.uint32_t(l), C.uint64_t(q)))
}

func fheSignedDecompDigit(value uint64, level, baseLog uint32) int64 {
	return int64(C.lux_fhe_signed_decomp_digit(C.uint64_t(value), C.uint32_t(level), C.uint32_t(baseLog)))
}

func fheComputeATilde(a uint64, N uint32, q uint64) uint32 {
	return uint32(C.lux_fhe_compute_a_tilde(C.uint64_t(a), C.uint32_t(N), C.uint64_t(q)))
}

func fheABIRevision() uint32 { return uint32(C.lux_fhe_abi_revision()) }

// =============================================================================
// Backend query + lifecycle (the surface beyond what gpu_cgo.go exposes).
// =============================================================================

func backendCount() int           { return int(C.lux_backend_count()) }
func backendAvailable(b int) bool { return bool(C.lux_backend_available(C.LuxBackend(b))) }
func backendName(b int) string    { return C.GoString(C.lux_backend_name(C.LuxBackend(b))) }
func deviceCount(b int) int       { return int(C.lux_device_count(C.LuxBackend(b))) }

// gpuErrorString returns whatever lux_gpu_error reports for the current GPU
// context, or empty string when no error is pending.
func gpuErrorString() string {
	s := C.lux_gpu_error(getGPU())
	if s == nil {
		return ""
	}
	return C.GoString(s)
}

// =============================================================================
// Stream + event lifecycle
// =============================================================================

type cStream struct{ h *C.LuxStream }
type cEvent struct{ h *C.LuxEvent }

func streamCreate() *cStream {
	h := C.lux_stream_create(getGPU())
	if h == nil {
		return nil
	}
	return &cStream{h: h}
}
func streamDestroy(s *cStream) { C.lux_stream_destroy(s.h); s.h = nil }
func streamSync(s *cStream) LuxErr {
	return errFromLux(C.lux_stream_sync(s.h))
}

func eventCreate() *cEvent {
	h := C.lux_event_create(getGPU())
	if h == nil {
		return nil
	}
	return &cEvent{h: h}
}
func eventDestroy(e *cEvent)                   { C.lux_event_destroy(e.h); e.h = nil }
func eventRecord(e *cEvent, s *cStream) LuxErr { return errFromLux(C.lux_event_record(e.h, s.h)) }
func eventWait(e *cEvent, s *cStream) LuxErr   { return errFromLux(C.lux_event_wait(e.h, s.h)) }
func eventElapsed(a, b *cEvent) float32        { return float32(C.lux_event_elapsed(a.h, b.h)) }

// =============================================================================
// secp256k1 ecrecover (single-call helper used by the e2e test)
// =============================================================================

func ecrecoverBatchCgo(sigs []Signature) ([]EcrecoverResult, LuxErr) {
	n := len(sigs)
	if n == 0 {
		return nil, errLuxOK
	}
	inputs := make([]C.LuxEcrecoverInput, n)
	outputs := make([]C.LuxEcrecoverOutput, n)
	for i := range sigs {
		C.memcpy(unsafe.Pointer(&inputs[i].r[0]), unsafe.Pointer(&sigs[i].R[0]), 32)
		C.memcpy(unsafe.Pointer(&inputs[i].s[0]), unsafe.Pointer(&sigs[i].S[0]), 32)
		inputs[i].v = C.uint8_t(sigs[i].V)
		C.memcpy(unsafe.Pointer(&inputs[i].msg_hash[0]), unsafe.Pointer(&sigs[i].MsgHash[0]), 32)
	}
	rc := C.lux_gpu_ecrecover_batch(
		getGPU(),
		&inputs[0],
		&outputs[0],
		C.size_t(n),
	)
	if rc != C.LUX_OK {
		return nil, errFromLux(rc)
	}
	results := make([]EcrecoverResult, n)
	for i := range outputs {
		C.memcpy(unsafe.Pointer(&results[i].Address[0]), unsafe.Pointer(&outputs[i].address[0]), 20)
		results[i].Valid = outputs[i].valid != 0
	}
	return results, errLuxOK
}

// =============================================================================
// Ed25519 / sr25519 / mldsa / mlkem / slhdsa verify-batch wrappers
// =============================================================================

// ed25519VerifyBatch calls lux_gpu_ed25519_verify_batch with parallel slices
// of public keys, messages (pre-hashed, 64 bytes each), and signatures.
func ed25519VerifyBatch(pks, msgs, sigs [][]byte) ([]bool, LuxErr) {
	n := len(pks)
	if n == 0 {
		return nil, errLuxOK
	}
	if len(msgs) != n || len(sigs) != n {
		return nil, errFromLux(C.LUX_ERROR_INVALID_ARGUMENT)
	}
	pkPtrs := make([]*C.uint8_t, n)
	msgPtrs := make([]*C.uint8_t, n)
	sigPtrs := make([]*C.uint8_t, n)
	for i := 0; i < n; i++ {
		pkPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&pks[i][0]))
		msgPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&msgs[i][0]))
		sigPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&sigs[i][0]))
	}
	results := make([]C.bool, n)
	rc := C.lux_gpu_ed25519_verify_batch(
		getGPU(),
		(**C.uint8_t)(unsafe.Pointer(&pkPtrs[0])),
		(**C.uint8_t)(unsafe.Pointer(&msgPtrs[0])),
		(**C.uint8_t)(unsafe.Pointer(&sigPtrs[0])),
		(*C.bool)(unsafe.Pointer(&results[0])),
		C.size_t(n),
	)
	if rc != C.LUX_OK {
		return nil, errFromLux(rc)
	}
	out := make([]bool, n)
	for i := 0; i < n; i++ {
		out[i] = bool(results[i])
	}
	return out, errLuxOK
}

func sr25519VerifyBatch(pks, msgs, sigs [][]byte) ([]bool, LuxErr) {
	n := len(pks)
	if n == 0 {
		return nil, errLuxOK
	}
	pkPtrs := make([]*C.uint8_t, n)
	msgPtrs := make([]*C.uint8_t, n)
	sigPtrs := make([]*C.uint8_t, n)
	for i := 0; i < n; i++ {
		pkPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&pks[i][0]))
		msgPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&msgs[i][0]))
		sigPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&sigs[i][0]))
	}
	results := make([]C.bool, n)
	rc := C.lux_gpu_sr25519_verify_batch(
		getGPU(),
		(**C.uint8_t)(unsafe.Pointer(&pkPtrs[0])),
		(**C.uint8_t)(unsafe.Pointer(&msgPtrs[0])),
		(**C.uint8_t)(unsafe.Pointer(&sigPtrs[0])),
		(*C.bool)(unsafe.Pointer(&results[0])),
		C.size_t(n),
	)
	if rc != C.LUX_OK {
		return nil, errFromLux(rc)
	}
	out := make([]bool, n)
	for i := 0; i < n; i++ {
		out[i] = bool(results[i])
	}
	return out, errLuxOK
}

func mldsaVerifyBatch(pks, msgs, sigs [][]byte) ([]bool, LuxErr) {
	n := len(pks)
	if n == 0 {
		return nil, errLuxOK
	}
	pkPtrs := make([]*C.uint8_t, n)
	msgPtrs := make([]*C.uint8_t, n)
	sigPtrs := make([]*C.uint8_t, n)
	for i := 0; i < n; i++ {
		pkPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&pks[i][0]))
		msgPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&msgs[i][0]))
		sigPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&sigs[i][0]))
	}
	results := make([]C.bool, n)
	rc := C.lux_gpu_mldsa_verify_batch(
		getGPU(),
		(**C.uint8_t)(unsafe.Pointer(&pkPtrs[0])),
		(**C.uint8_t)(unsafe.Pointer(&msgPtrs[0])),
		(**C.uint8_t)(unsafe.Pointer(&sigPtrs[0])),
		(*C.bool)(unsafe.Pointer(&results[0])),
		C.size_t(n),
	)
	if rc != C.LUX_OK {
		return nil, errFromLux(rc)
	}
	out := make([]bool, n)
	for i := 0; i < n; i++ {
		out[i] = bool(results[i])
	}
	return out, errLuxOK
}

// slhdsaVerifyBatch wraps the ABI v13 lux_gpu_slhdsa_verify_batch which
// takes per-element message lengths. v12 hardcoded msg_len=32 inside
// the wrapper, which silently broke every caller whose msg was not 32
// bytes (the FIPS 205 / ACVP contract is arbitrary length).
func slhdsaVerifyBatch(pks, msgs, sigs [][]byte) ([]bool, LuxErr) {
	n := len(pks)
	if n == 0 {
		return nil, errLuxOK
	}
	pkPtrs := make([]*C.uint8_t, n)
	msgPtrs := make([]*C.uint8_t, n)
	sigPtrs := make([]*C.uint8_t, n)
	msgLens := make([]C.size_t, n)
	for i := 0; i < n; i++ {
		pkPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&pks[i][0]))
		if len(msgs[i]) > 0 {
			msgPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&msgs[i][0]))
		} else {
			msgPtrs[i] = nil
		}
		sigPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&sigs[i][0]))
		msgLens[i] = C.size_t(len(msgs[i]))
	}
	results := make([]C.bool, n)
	rc := C.lux_gpu_slhdsa_verify_batch(
		getGPU(),
		(**C.uint8_t)(unsafe.Pointer(&pkPtrs[0])),
		(**C.uint8_t)(unsafe.Pointer(&msgPtrs[0])),
		(*C.size_t)(unsafe.Pointer(&msgLens[0])),
		(**C.uint8_t)(unsafe.Pointer(&sigPtrs[0])),
		(*C.bool)(unsafe.Pointer(&results[0])),
		C.size_t(n),
	)
	if rc != C.LUX_OK {
		return nil, errFromLux(rc)
	}
	out := make([]bool, n)
	for i := 0; i < n; i++ {
		out[i] = bool(results[i])
	}
	return out, errLuxOK
}

// mlkemDecapBatch wraps lux_gpu_mlkem_decapsulate_batch. shared_secrets is
// an output array of pointers — caller owns the buffers; this helper
// allocates them and returns the concatenated 32-byte secrets.
func mlkemDecapBatch(sks, cts [][]byte) ([]byte, LuxErr) {
	n := len(sks)
	if n == 0 {
		return nil, errLuxOK
	}
	if len(cts) != n {
		return nil, errFromLux(C.LUX_ERROR_INVALID_ARGUMENT)
	}
	skPtrs := make([]*C.uint8_t, n)
	ctPtrs := make([]*C.uint8_t, n)
	out := make([]byte, n*32)
	ssPtrs := make([]*C.uint8_t, n)
	for i := 0; i < n; i++ {
		skPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&sks[i][0]))
		ctPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&cts[i][0]))
		ssPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&out[i*32]))
	}
	rc := C.lux_gpu_mlkem_decapsulate_batch(
		getGPU(),
		(**C.uint8_t)(unsafe.Pointer(&skPtrs[0])),
		(**C.uint8_t)(unsafe.Pointer(&ctPtrs[0])),
		(**C.uint8_t)(unsafe.Pointer(&ssPtrs[0])),
		C.size_t(n),
	)
	if rc != C.LUX_OK {
		return nil, errFromLux(rc)
	}
	return out, errLuxOK
}

// =============================================================================
// PQ signer / keygen wrappers (weak-linked; nil-checked at runtime)
// =============================================================================

func haveEd25519() bool { return C.have_ed25519() != 0 }
func haveSr25519() bool { return C.have_sr25519() != 0 }
func haveMldsa() bool   { return C.have_mldsa() != 0 }
func haveMlkem() bool   { return C.have_mlkem() != 0 }
func haveSlhdsa() bool  { return C.have_slhdsa() != 0 }

func ed25519KeygenC(seed [32]byte) (sk, pk [32]byte, err error) {
	if !haveEd25519() {
		return sk, pk, errors.New("ed25519 signer not linked")
	}
	rc := C.call_ed25519_keygen(
		(*C.uchar)(unsafe.Pointer(&seed[0])),
		(*C.uchar)(unsafe.Pointer(&sk[0])),
		(*C.uchar)(unsafe.Pointer(&pk[0])),
	)
	if rc != 0 {
		err = errors.New("ed25519_keygen failed")
	}
	return
}

func ed25519SignC(sk [32]byte, msg []byte) (sig [64]byte, err error) {
	if !haveEd25519() {
		return sig, errors.New("ed25519 signer not linked")
	}
	rc := C.call_ed25519_sign(
		(*C.uchar)(unsafe.Pointer(&sk[0])),
		(*C.uchar)(unsafe.Pointer(&msg[0])),
		C.size_t(len(msg)),
		(*C.uchar)(unsafe.Pointer(&sig[0])),
	)
	if rc != 0 {
		err = errors.New("ed25519_sign failed")
	}
	return
}

// sr25519: there is no exposed keygen in the C-ABI (the implementation
// derives pk from sk on each sign). We use sr25519_sign + sr25519_verify
// directly with a random sk and pk drawn from the same seed.
func sr25519SignC(sk [32]byte, msg []byte) (sig [64]byte, err error) {
	if !haveSr25519() {
		return sig, errors.New("sr25519 signer not linked")
	}
	rc := C.call_sr25519_sign(
		(*C.uchar)(unsafe.Pointer(&sk[0])),
		(*C.uchar)(unsafe.Pointer(&msg[0])),
		C.size_t(len(msg)),
		(*C.uchar)(unsafe.Pointer(&sig[0])),
	)
	if rc != 0 {
		err = errors.New("sr25519_sign failed")
	}
	return
}

func sr25519VerifyC(pk [32]byte, msg []byte, sig [64]byte) bool {
	if !haveSr25519() {
		return false
	}
	return C.call_sr25519_verify(
		(*C.uchar)(unsafe.Pointer(&pk[0])),
		(*C.uchar)(unsafe.Pointer(&msg[0])),
		C.size_t(len(msg)),
		(*C.uchar)(unsafe.Pointer(&sig[0])),
	) == 0
}

// mldsaKeygenC: mode is the NIST level (2/3/5). pk and sk buffers must be
// pre-allocated to the algorithm's pk/sk sizes (mode 3 = 1952 / 4032 bytes).
func mldsaKeygenC(mode int, seed [32]byte, pk, sk []byte) error {
	if !haveMldsa() {
		return errors.New("mldsa signer not linked")
	}
	rc := C.call_mldsa_keygen(
		C.int(mode),
		(*C.uchar)(unsafe.Pointer(&seed[0])),
		(*C.uchar)(unsafe.Pointer(&pk[0])),
		(*C.uchar)(unsafe.Pointer(&sk[0])),
	)
	if rc != 0 {
		return errors.New("mldsa_keygen failed")
	}
	return nil
}

func mldsaSignC(mode int, sk, msg, sig []byte) (sigLen int, err error) {
	if !haveMldsa() {
		return 0, errors.New("mldsa signer not linked")
	}
	cSigLen := C.size_t(len(sig))
	rc := C.call_mldsa_sign(
		C.int(mode),
		(*C.uchar)(unsafe.Pointer(&sk[0])),
		(*C.uchar)(unsafe.Pointer(&msg[0])),
		C.size_t(len(msg)),
		(*C.uchar)(unsafe.Pointer(&sig[0])),
		&cSigLen,
	)
	if rc != 0 {
		return 0, errors.New("mldsa_sign failed")
	}
	return int(cSigLen), nil
}

func mlkemKeygenC(mode int, seed [32]byte, pk, sk []byte) error {
	if !haveMlkem() {
		return errors.New("mlkem signer not linked")
	}
	rc := C.call_mlkem_keygen(
		C.int(mode),
		(*C.uchar)(unsafe.Pointer(&seed[0])),
		(*C.uchar)(unsafe.Pointer(&pk[0])),
		(*C.uchar)(unsafe.Pointer(&sk[0])),
	)
	if rc != 0 {
		return errors.New("mlkem_keygen failed")
	}
	return nil
}

func mlkemEncapC(mode int, pk, ct []byte) (ss [32]byte, err error) {
	if !haveMlkem() {
		return ss, errors.New("mlkem signer not linked")
	}
	rc := C.call_mlkem_encap(
		C.int(mode),
		(*C.uchar)(unsafe.Pointer(&pk[0])),
		(*C.uchar)(unsafe.Pointer(&ct[0])),
		(*C.uchar)(unsafe.Pointer(&ss[0])),
	)
	if rc != 0 {
		err = errors.New("mlkem_encap failed")
	}
	return
}

func slhdsaKeygenC(mode int, seed [32]byte, pk, sk []byte) error {
	if !haveSlhdsa() {
		return errors.New("slhdsa signer not linked")
	}
	rc := C.call_slhdsa_keygen(
		C.int(mode),
		(*C.uchar)(unsafe.Pointer(&seed[0])),
		(*C.uchar)(unsafe.Pointer(&pk[0])),
		(*C.uchar)(unsafe.Pointer(&sk[0])),
	)
	if rc != 0 {
		return errors.New("slhdsa_keygen failed")
	}
	return nil
}

func slhdsaSignC(mode int, sk, msg, sig []byte) (sigLen int, err error) {
	if !haveSlhdsa() {
		return 0, errors.New("slhdsa signer not linked")
	}
	cSigLen := C.size_t(len(sig))
	rc := C.call_slhdsa_sign(
		C.int(mode),
		(*C.uchar)(unsafe.Pointer(&sk[0])),
		(*C.uchar)(unsafe.Pointer(&msg[0])),
		C.size_t(len(msg)),
		(*C.uchar)(unsafe.Pointer(&sig[0])),
		&cSigLen,
	)
	if rc != 0 {
		return 0, errors.New("slhdsa_sign failed")
	}
	return int(cSigLen), nil
}

// =============================================================================
// Error helpers
// =============================================================================

// LuxErr is an alias for error returns from cgo wrappers, with a few
// well-known sentinels for the common return codes.
type LuxErr = error

var (
	errLuxOK              error = nil
	errLuxInvalidArg            = errors.New("lux: invalid argument")
	errLuxOOM                   = errors.New("lux: out of memory")
	errLuxBackendNotAvail       = errors.New("lux: backend not available")
	errLuxDeviceNotFound        = errors.New("lux: device not found")
	errLuxKernelFailed          = errors.New("lux: kernel failed")
	errLuxNotSupported          = errors.New("lux: not supported")
	errLuxUnknown               = errors.New("lux: unknown error")
)

func errFromLux(rc C.LuxError) error {
	switch rc {
	case C.LUX_OK:
		return nil
	case C.LUX_ERROR_INVALID_ARGUMENT:
		return errLuxInvalidArg
	case C.LUX_ERROR_OUT_OF_MEMORY:
		return errLuxOOM
	case C.LUX_ERROR_BACKEND_NOT_AVAILABLE:
		return errLuxBackendNotAvail
	case C.LUX_ERROR_DEVICE_NOT_FOUND:
		return errLuxDeviceNotFound
	case C.LUX_ERROR_KERNEL_FAILED:
		return errLuxKernelFailed
	case C.LUX_ERROR_NOT_SUPPORTED:
		return errLuxNotSupported
	default:
		return errLuxUnknown
	}
}

// isNotSupported returns true iff err is the not-supported sentinel from a
// cgo wrapper. Used by tests to skip gracefully on backends that don't
// implement a particular op.
func isNotSupported(err error) bool { return errors.Is(err, errLuxNotSupported) }
