# Lux GPU - Unified GPU Acceleration

## Overview

Single package for GPU-accelerated operations with switchable backends:
- **Metal** - Apple Silicon (macOS/iOS)
- **CUDA** - NVIDIA GPUs (Linux/Windows)
- **Dawn** - WebGPU via Dawn/wgpu (cross-platform)
- **CPU** - SIMD-optimized fallback (always available)

## Architecture

```
lux/gpu/
├── gpu.go           # Main API: Session, Backend switching
├── backend.go       # Backend enum and detection
├── device.go        # DeviceInfo
├── tensor.go        # Tensor type and operations
├── errors.go        # Error types
├── ops.go           # Unified compute operations
│
├── crypto/          # Cryptographic accelerations
│   ├── bls.go       # BLS12-381 operations
│   ├── poseidon.go  # Poseidon hash
│   ├── msm.go       # Multi-scalar multiplication
│   └── kzg.go       # KZG commitments
│
├── fhe/             # Fully Homomorphic Encryption
│   ├── tfhe.go      # TFHE gates
│   ├── ckks.go      # CKKS approximate arithmetic
│   └── ntt.go       # Number Theoretic Transform
│
├── ml/              # Machine Learning
│   ├── matmul.go    # Matrix multiplication
│   ├── attention.go # Attention mechanisms
│   └── quantize.go  # Quantization ops
│
└── internal/
    └── capi/        # CGO bindings to liblux_accel
```

## Usage

```go
import "github.com/luxfi/lux/gpu"

// Auto-detect best backend
sess := gpu.DefaultSession()

// Or explicitly select backend
sess, _ := gpu.NewSession(gpu.WithBackend(gpu.Metal))
sess, _ := gpu.NewSession(gpu.WithBackend(gpu.CUDA))
sess, _ := gpu.NewSession(gpu.WithBackend(gpu.Dawn))

// Runtime switching
sess.SetBackend(gpu.Metal)

// Check available backends
for _, b := range gpu.AvailableBackends() {
    fmt.Printf("%s: %v\n", b.Name, b.Available)
}

// Operations
tensor := sess.Zeros([]int{1024, 1024}, gpu.Float32)
result := sess.MatMul(a, b)

// Crypto operations
sess.Crypto().BLSVerifyBatch(sigs, msgs, pks)
sess.Crypto().PoseidonHash(inputs)

// FHE operations
sess.FHE().NTT(poly)
sess.FHE().TFHEBootstrap(ct)
```

## Backend Selection Priority

1. Environment: `LUX_GPU_BACKEND=metal|cuda|dawn|cpu`
2. Explicit: `gpu.WithBackend(gpu.Metal)`
3. Auto-detect (in order):
   - Metal (if macOS + Apple Silicon)
   - CUDA (if NVIDIA GPU + driver)
   - Dawn (if GPU available)
   - CPU (always)

## C++ Library (liblux_accel)

Single unified C library with backend plugins:

```
luxcpp/lux-accel/
├── include/lux/
│   ├── accel.h         # Main C API
│   ├── backend.h       # Backend types
│   ├── tensor.h        # Tensor operations
│   ├── crypto.h        # Crypto ops
│   └── fhe.h           # FHE ops
│
├── src/
│   ├── accel.cpp       # Core implementation
│   ├── backend.cpp     # Backend management
│   ├── metal/          # Metal backend
│   ├── cuda/           # CUDA backend
│   └── dawn/           # Dawn/WebGPU backend
│
└── kernels/            # GPU kernel sources
    ├── metal/*.metal
    ├── cuda/*.cu
    └── wgsl/*.wgsl
```

## Build Configuration

```bash
# Build with all backends (auto-detect)
cmake -B build

# Specific backends
cmake -B build -DLUX_BACKEND_METAL=ON
cmake -B build -DLUX_BACKEND_CUDA=ON
cmake -B build -DLUX_BACKEND_DAWN=ON

# Go package
CGO_ENABLED=1 go build ./gpu/...
```
