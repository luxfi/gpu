// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package gpu

import "testing"

// TestSignatureLayout pins the Signature struct field sizes that callers
// rely on when packing R/S/V/MsgHash for batch ecrecover.
func TestSignatureLayout(t *testing.T) {
	var sig Signature
	if got := len(sig.R); got != 32 {
		t.Fatalf("Signature.R length = %d, want 32", got)
	}
	if got := len(sig.S); got != 32 {
		t.Fatalf("Signature.S length = %d, want 32", got)
	}
	if got := len(sig.MsgHash); got != 32 {
		t.Fatalf("Signature.MsgHash length = %d, want 32", got)
	}
	// V is a single recovery byte (0 or 1 after normalization).
	sig.V = 1
	if sig.V != 1 {
		t.Fatalf("Signature.V round-trip failed")
	}
}

// TestEcrecoverResultLayout pins the EcrecoverResult struct so callers
// can rely on Address being a 20-byte Ethereum address and Valid being
// a boolean recovery flag.
func TestEcrecoverResultLayout(t *testing.T) {
	var r EcrecoverResult
	if got := len(r.Address); got != 20 {
		t.Fatalf("EcrecoverResult.Address length = %d, want 20", got)
	}
	r.Valid = true
	if !r.Valid {
		t.Fatalf("EcrecoverResult.Valid round-trip failed")
	}
}

// TestBatchEcrecoverEmpty verifies that an empty batch returns no results
// and no error in both cgo and non-cgo builds. Callers depend on this to
// short-circuit empty blocks without a CGO trip.
func TestBatchEcrecoverEmpty(t *testing.T) {
	results, err := BatchEcrecover(nil)
	if err != nil {
		t.Fatalf("BatchEcrecover(nil) error = %v, want nil", err)
	}
	if results != nil {
		t.Fatalf("BatchEcrecover(nil) results = %v, want nil", results)
	}

	results, err = BatchEcrecover([]Signature{})
	if err != nil {
		t.Fatalf("BatchEcrecover(empty) error = %v, want nil", err)
	}
	if results != nil {
		t.Fatalf("BatchEcrecover(empty) results = %v, want nil", results)
	}
}
