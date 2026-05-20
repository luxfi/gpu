//go:build cgo

// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package gpu

import (
	"os"
	"testing"
)

// Pin the backend plugin search path + Metal kernel root so the vtbl-loaded
// backends pick up the freshly-built artifacts in ~/work/luxcpp/gpu/build and
// the metal shader sources alongside them, rather than whichever older copy
// happens to live under /usr/local/lib.
//
// PluginLoader resolves LUX_GPU_BACKEND_PATH at C++ static-initializer time
// (the first lux_gpu_create() triggers construction of the singleton). Go's
// package init() also calls lux_gpu_create(). To race ahead of either, the
// env vars MUST be set before the cgo runtime initializes — which we achieve
// by setting them in this package's own init() block, ordered to run before
// the gpu_cgo.go init() via Go's lexical filename ordering (this file is
// named a_setup_test.go so it sorts first among test files; non-test files
// have their inits run in lex order alongside).
//
// Note: this file does NOT use cgo (cgo in *_test.go is unsupported). It
// uses pure Go os.Setenv, which on Darwin/Linux propagates through the libc
// environ via runtime_setenv, so the C++ initializers of libluxgpu observe
// the values on first call.

const (
	defaultLuxBackendPath  = "/Users/z/work/luxcpp/gpu/build/metal_backend"
	defaultLuxMetalKernels = "/Users/z/work/luxcpp/metal/src/shaders"
)

// applyBackendEnv pushes the backend env vars into the process environment
// if they are unset (idempotent on test re-runs).
func applyBackendEnv() {
	if existing := os.Getenv("LUX_GPU_BACKEND_PATH"); existing == "" {
		_ = os.Setenv("LUX_GPU_BACKEND_PATH", defaultLuxBackendPath)
	} else if !containsPath(existing, defaultLuxBackendPath) {
		_ = os.Setenv("LUX_GPU_BACKEND_PATH",
			defaultLuxBackendPath+":"+existing)
	}
	if os.Getenv("LUX_METAL_KERNEL_PATH") == "" {
		_ = os.Setenv("LUX_METAL_KERNEL_PATH", defaultLuxMetalKernels)
	}
}

// containsPath returns true if elem appears as a colon-separated segment of
// list. Used to keep applyBackendEnv idempotent across t.Setenv reentrancy.
func containsPath(list, elem string) bool {
	start := 0
	for i := 0; i <= len(list); i++ {
		if i == len(list) || list[i] == ':' {
			if list[start:i] == elem {
				return true
			}
			start = i + 1
		}
	}
	return false
}

func init() {
	// init() in a _test.go file is part of the test binary's init graph and
	// runs alongside non-test inits. Lexical file order ensures this runs
	// before gpu_cgo.go's init() under the standard Go toolchain.
	applyBackendEnv()
}

// TestMain re-applies the env (so subprocess-style runners that clear the
// process environ between tests still see our values) and runs the suite.
func TestMain(m *testing.M) {
	applyBackendEnv()
	os.Exit(m.Run())
}
