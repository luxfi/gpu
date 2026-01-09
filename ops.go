package gpu

// Dtype represents tensor data types.
type Dtype int

const (
	Float32 Dtype = iota
	Float64
	Float16
	BFloat16
	Int32
	Int64
	Uint32
	Uint64
	Bool
)

// String returns the dtype name.
func (d Dtype) String() string {
	switch d {
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	case Float16:
		return "float16"
	case BFloat16:
		return "bfloat16"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Uint32:
		return "uint32"
	case Uint64:
		return "uint64"
	case Bool:
		return "bool"
	default:
		return "unknown"
	}
}

// Size returns bytes per element.
func (d Dtype) Size() int {
	switch d {
	case Float32, Int32, Uint32:
		return 4
	case Float64, Int64, Uint64:
		return 8
	case Float16, BFloat16:
		return 2
	case Bool:
		return 1
	default:
		return 0
	}
}

// Tensor represents a GPU tensor.
type Tensor struct {
	handle tensorHandle
	shape  []int
	dtype  Dtype
}

type tensorHandle interface {
	data() []byte
	sync() error
	close() error
}

// Shape returns the tensor dimensions.
func (t *Tensor) Shape() []int {
	return t.shape
}

// Dtype returns the element type.
func (t *Tensor) Dtype() Dtype {
	return t.dtype
}

// Size returns total number of elements.
func (t *Tensor) Size() int {
	n := 1
	for _, s := range t.shape {
		n *= s
	}
	return n
}

// TensorOps provides tensor creation and manipulation.
type TensorOps interface {
	// Creation
	Zeros(shape []int, dtype Dtype) (*Tensor, error)
	Ones(shape []int, dtype Dtype) (*Tensor, error)
	Full(shape []int, value float64, dtype Dtype) (*Tensor, error)
	FromSlice(data any, shape []int, dtype Dtype) (*Tensor, error)

	// Arithmetic
	Add(a, b *Tensor) (*Tensor, error)
	Sub(a, b *Tensor) (*Tensor, error)
	Mul(a, b *Tensor) (*Tensor, error)
	Div(a, b *Tensor) (*Tensor, error)
	MatMul(a, b *Tensor) (*Tensor, error)

	// Reductions
	Sum(t *Tensor, axis ...int) (*Tensor, error)
	Mean(t *Tensor, axis ...int) (*Tensor, error)
	Max(t *Tensor, axis ...int) (*Tensor, error)
	Min(t *Tensor, axis ...int) (*Tensor, error)

	// Shape operations
	Reshape(t *Tensor, shape []int) (*Tensor, error)
	Transpose(t *Tensor, axes ...int) (*Tensor, error)
	Concat(tensors []*Tensor, axis int) (*Tensor, error)
}

// CryptoOps provides GPU-accelerated cryptographic operations.
type CryptoOps interface {
	// BLS12-381
	BLSVerify(sig, msg, pubkey []byte) (bool, error)
	BLSVerifyBatch(sigs, msgs, pubkeys [][]byte) ([]bool, error)
	BLSAggregate(sigs [][]byte) ([]byte, error)

	// Poseidon hash
	PoseidonHash(inputs [][]byte) ([][]byte, error)
	PoseidonHashBatch(inputs [][][]byte) ([][]byte, error)

	// MSM (multi-scalar multiplication)
	MSM(scalars, points [][]byte) ([]byte, error)

	// KZG commitments
	KZGCommit(poly []byte) ([]byte, error)
	KZGProve(poly []byte, point []byte) ([]byte, error)
	KZGVerify(commitment, proof, point, value []byte) (bool, error)

	// Shamir secret sharing
	ShamirSplit(secret []byte, n, k int) ([][]byte, error)
	ShamirCombine(shares [][]byte) ([]byte, error)
}

// FHEOps provides GPU-accelerated FHE operations.
type FHEOps interface {
	// NTT (Number Theoretic Transform)
	NTTForward(poly []uint64, modulus uint64) ([]uint64, error)
	NTTInverse(poly []uint64, modulus uint64) ([]uint64, error)
	NTTBatch(polys [][]uint64, modulus uint64) ([][]uint64, error)

	// TFHE operations
	TFHEKeyGen(params TFHEParams) (*TFHEKeys, error)
	TFHEEncrypt(keys *TFHEKeys, bit bool) (*TFHECiphertext, error)
	TFHEDecrypt(keys *TFHEKeys, ct *TFHECiphertext) (bool, error)
	TFHEAnd(keys *TFHEKeys, a, b *TFHECiphertext) (*TFHECiphertext, error)
	TFHEOr(keys *TFHEKeys, a, b *TFHECiphertext) (*TFHECiphertext, error)
	TFHENot(keys *TFHEKeys, a *TFHECiphertext) (*TFHECiphertext, error)
	TFHEBootstrap(keys *TFHEKeys, ct *TFHECiphertext) (*TFHECiphertext, error)

	// CKKS operations
	CKKSEncrypt(keys *CKKSKeys, values []float64) (*CKKSCiphertext, error)
	CKKSDecrypt(keys *CKKSKeys, ct *CKKSCiphertext) ([]float64, error)
	CKKSAdd(a, b *CKKSCiphertext) (*CKKSCiphertext, error)
	CKKSMul(keys *CKKSKeys, a, b *CKKSCiphertext) (*CKKSCiphertext, error)
	CKKSRotate(keys *CKKSKeys, ct *CKKSCiphertext, steps int) (*CKKSCiphertext, error)
}

// MLOps provides GPU-accelerated ML operations.
type MLOps interface {
	// Matrix operations
	GEMM(a, b *Tensor, alpha, beta float32, transA, transB bool) (*Tensor, error)
	BatchMatMul(a, b *Tensor) (*Tensor, error)

	// Attention
	ScaledDotProductAttention(q, k, v *Tensor, mask *Tensor) (*Tensor, error)
	MultiHeadAttention(q, k, v *Tensor, numHeads int) (*Tensor, error)

	// Activations
	ReLU(t *Tensor) (*Tensor, error)
	GELU(t *Tensor) (*Tensor, error)
	Softmax(t *Tensor, axis int) (*Tensor, error)
	LayerNorm(t *Tensor, gamma, beta *Tensor) (*Tensor, error)

	// Convolution
	Conv2D(input, weight *Tensor, stride, padding []int) (*Tensor, error)
	MaxPool2D(input *Tensor, kernelSize, stride []int) (*Tensor, error)

	// Quantization
	Quantize(t *Tensor, bits int) (*Tensor, float32, int32, error)
	Dequantize(t *Tensor, scale float32, zeroPoint int32) (*Tensor, error)
}

// TFHEParams holds TFHE scheme parameters.
type TFHEParams struct {
	N            int // LWE dimension
	K            int // RLWE dimension
	SecurityBits int // Target security level (80, 128, etc)
}

// TFHEKeys holds TFHE key material.
type TFHEKeys struct {
	handle any
}

// TFHECiphertext is an encrypted bit.
type TFHECiphertext struct {
	handle any
}

// CKKSKeys holds CKKS key material.
type CKKSKeys struct {
	handle any
}

// CKKSCiphertext is an encrypted vector.
type CKKSCiphertext struct {
	handle any
}
