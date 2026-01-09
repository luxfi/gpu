package gpu

import (
	"context"
	"errors"
	"sync"
)

// Session represents a GPU compute session.
// It manages device resources and provides access to compute operations.
type Session struct {
	mu      sync.RWMutex
	handle  sessionHandle
	backend Backend
	device  Device
	closed  bool
}

// sessionHandle is the internal interface implemented by backend-specific code.
type sessionHandle interface {
	backend() Backend
	device() Device
	sync() error
	syncContext(ctx context.Context) error
	close() error

	// Operation namespaces
	tensor() TensorOps
	crypto() CryptoOps
	fhe() FHEOps
	ml() MLOps
}

// SessionOption configures session creation.
type SessionOption func(*sessionConfig)

type sessionConfig struct {
	backend     Backend
	deviceIndex int
}

// WithBackend sets the desired backend.
func WithBackend(b Backend) SessionOption {
	return func(c *sessionConfig) {
		c.backend = b
	}
}

// WithDevice sets the device index for multi-GPU systems.
func WithDevice(index int) SessionOption {
	return func(c *sessionConfig) {
		c.deviceIndex = index
	}
}

var (
	defaultSession     *Session
	defaultSessionOnce sync.Once
	defaultSessionErr  error
)

// DefaultSession returns the global default session.
// It auto-detects the best available backend.
func DefaultSession() (*Session, error) {
	defaultSessionOnce.Do(func() {
		defaultSession, defaultSessionErr = NewSession()
	})
	return defaultSession, defaultSessionErr
}

// NewSession creates a new GPU session with the specified options.
func NewSession(opts ...SessionOption) (*Session, error) {
	cfg := &sessionConfig{backend: Auto}
	for _, opt := range opts {
		opt(cfg)
	}
	return newSession(cfg)
}

// Backend returns the current backend.
func (s *Session) Backend() Backend {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.backend
}

// Device returns the current device info.
func (s *Session) Device() Device {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.device
}

// SetBackend switches to a different backend.
// Returns an error if the backend is not available.
func (s *Session) SetBackend(b Backend) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrSessionClosed
	}

	if b == s.backend {
		return nil
	}

	// Close current handle
	if s.handle != nil {
		if err := s.handle.close(); err != nil {
			return err
		}
	}

	// Create new session with requested backend
	cfg := &sessionConfig{backend: b}
	newSess, err := newSession(cfg)
	if err != nil {
		return err
	}

	s.handle = newSess.handle
	s.backend = newSess.backend
	s.device = newSess.device
	return nil
}

// Sync waits for all pending operations to complete.
func (s *Session) Sync() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return ErrSessionClosed
	}
	return s.handle.sync()
}

// SyncContext waits for pending operations with cancellation support.
func (s *Session) SyncContext(ctx context.Context) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return ErrSessionClosed
	}
	return s.handle.syncContext(ctx)
}

// Close releases session resources.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if s.handle != nil {
		return s.handle.close()
	}
	return nil
}

// Tensor returns the tensor operations interface.
func (s *Session) Tensor() TensorOps {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.handle.tensor()
}

// Crypto returns cryptographic operations.
func (s *Session) Crypto() CryptoOps {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.handle.crypto()
}

// FHE returns FHE operations.
func (s *Session) FHE() FHEOps {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.handle.fhe()
}

// ML returns machine learning operations.
func (s *Session) ML() MLOps {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.handle.ml()
}

// Common errors
var (
	ErrSessionClosed   = errors.New("session closed")
	ErrBackendNotAvail = errors.New("backend not available")
	ErrNoBackends      = errors.New("no GPU backends available")
	ErrOutOfMemory     = errors.New("out of GPU memory")
	ErrInvalidShape    = errors.New("invalid tensor shape")
)
