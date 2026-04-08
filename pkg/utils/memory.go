// Package utils provides utility functions for memory management
package utils

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"sync"
)

var (
	// Memory pools for different sizes to reduce allocations
	bytePool4   = sync.Pool{New: func() interface{} { b := make([]byte, 4); return &b }}
	bytePool8   = sync.Pool{New: func() interface{} { b := make([]byte, 8); return &b }}
	bytePool16  = sync.Pool{New: func() interface{} { b := make([]byte, 16); return &b }}
	bytePool64  = sync.Pool{New: func() interface{} { b := make([]byte, 64); return &b }}
	bytePool256 = sync.Pool{New: func() interface{} { b := make([]byte, 256); return &b }}
	bytePool1K  = sync.Pool{New: func() interface{} { b := make([]byte, 1024); return &b }}
	bytePool4K  = sync.Pool{New: func() interface{} { b := make([]byte, 4096); return &b }}
)

// GetPooledBytes returns a byte slice from the pool
func GetPooledBytes(size int) []byte {
	switch {
	case size <= 4:
		return (*bytePool4.Get().(*[]byte))[:size]
	case size <= 8:
		return (*bytePool8.Get().(*[]byte))[:size]
	case size <= 16:
		return (*bytePool16.Get().(*[]byte))[:size]
	case size <= 64:
		return (*bytePool64.Get().(*[]byte))[:size]
	case size <= 256:
		return (*bytePool256.Get().(*[]byte))[:size]
	case size <= 1024:
		return (*bytePool1K.Get().(*[]byte))[:size]
	case size <= 4096:
		return (*bytePool4K.Get().(*[]byte))[:size]
	default:
		return make([]byte, size)
	}
}

// PutPooledBytes returns a byte slice to the pool
func PutPooledBytes(buf []byte) {
	cap := cap(buf)
	switch {
	case cap == 4:
		bytePool4.Put(&buf)
	case cap == 8:
		bytePool8.Put(&buf)
	case cap == 16:
		bytePool16.Put(&buf)
	case cap == 64:
		bytePool64.Put(&buf)
	case cap == 256:
		bytePool256.Put(&buf)
	case cap == 1024:
		bytePool1K.Put(&buf)
	case cap == 4096:
		bytePool4K.Put(&buf)
	}
}

// MemoryStats holds memory statistics
type MemoryStats struct {
	AllocBytes   uint64
	TotalAlloc   uint64
	SysBytes     uint64
	NumGC        uint32
	NumGoroutine int
}

// GetMemoryStats returns current memory statistics
func GetMemoryStats() MemoryStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	return MemoryStats{
		AllocBytes:   m.Alloc,
		TotalAlloc:   m.TotalAlloc,
		SysBytes:     m.Sys,
		NumGC:        m.NumGC,
		NumGoroutine: runtime.NumGoroutine(),
	}
}

// ForceGC forces a garbage collection
func ForceGC() {
	runtime.GC()
	debug.FreeOSMemory()
}

// SetGCPercent sets the GC target percentage
func SetGCPercent(percent int) int {
	return debug.SetGCPercent(percent)
}

// ByteSize represents a size in bytes with human-readable formatting
type ByteSize int64

func (b ByteSize) String() string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
		TB = GB * 1024
	)
	
	switch {
	case b >= TB:
		return fmt.Sprintf("%.2f TB", float64(b)/TB)
	case b >= GB:
		return fmt.Sprintf("%.2f GB", float64(b)/GB)
	case b >= MB:
		return fmt.Sprintf("%.2f MB", float64(b)/MB)
	case b >= KB:
		return fmt.Sprintf("%.2f KB", float64(b)/KB)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

// MemoryLimiter provides memory usage limiting
type MemoryLimiter struct {
	maxBytes int64
	current  int64
	mu       sync.RWMutex
}

// NewMemoryLimiter creates a new memory limiter
func NewMemoryLimiter(maxBytes int64) *MemoryLimiter {
	return &MemoryLimiter{maxBytes: maxBytes}
}

// Alloc attempts to allocate bytes, returns error if would exceed limit
func (m *MemoryLimiter) Alloc(bytes int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.current+bytes > m.maxBytes {
		return fmt.Errorf("memory limit exceeded: current=%s, request=%s, max=%s",
			ByteSize(m.current), ByteSize(bytes), ByteSize(m.maxBytes))
	}
	
	m.current += bytes
	return nil
}

// Free releases allocated bytes
func (m *MemoryLimiter) Free(bytes int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.current -= bytes
	if m.current < 0 {
		m.current = 0
	}
}

// Current returns current allocation
func (m *MemoryLimiter) Current() int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.current
}

// Max returns the maximum allowed allocation
func (m *MemoryLimiter) Max() int64 {
	return m.maxBytes
}

// Reset resets the current allocation counter
func (m *MemoryLimiter) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.current = 0
}

// RingBuffer is a circular byte buffer for streaming data
type RingBuffer struct {
	buf  []byte
	size int
	r    int
	w    int
	mu   sync.Mutex
}

// NewRingBuffer creates a new ring buffer
func NewRingBuffer(size int) *RingBuffer {
	return &RingBuffer{
		buf:  make([]byte, size),
		size: size,
	}
}

// Write writes data to the buffer
func (r *RingBuffer) Write(p []byte) (n int, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	for _, b := range p {
		r.buf[r.w] = b
		r.w = (r.w + 1) % r.size
		if r.w == r.r {
			r.r = (r.r + 1) % r.size
		}
	}
	return len(p), nil
}

// Read reads data from the buffer
func (r *RingBuffer) Read(p []byte) (n int, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	for i := range p {
		if r.r == r.w {
			return i, nil
		}
		p[i] = r.buf[r.r]
		r.r = (r.r + 1) % r.size
	}
	return len(p), nil
}

// Available returns number of bytes available to read
func (r *RingBuffer) Available() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if r.w >= r.r {
		return r.w - r.r
	}
	return r.size - r.r + r.w
}

// Reset clears the buffer
func (r *RingBuffer) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.r, r.w = 0, 0
}
