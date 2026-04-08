package inference

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// Profiler tracks timing information for inference
type Profiler struct {
	enabled bool
	times   map[string][]time.Duration
	starts  map[string]time.Time
	order   []string
	mu      sync.RWMutex
}

// NewProfiler creates a new profiler
func NewProfiler(enabled bool) *Profiler {
	return &Profiler{
		enabled: enabled,
		times:   make(map[string][]time.Duration),
		starts:  make(map[string]time.Time),
		order:   make([]string, 0),
	}
}

// Start begins timing an operation
func (p *Profiler) Start(name string) {
	if !p.enabled {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	
	if _, exists := p.starts[name]; !exists {
		p.order = append(p.order, name)
	}
	p.starts[name] = time.Now()
}

// End finishes timing an operation
func (p *Profiler) End(name string) time.Duration {
	if !p.enabled {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	
	start, ok := p.starts[name]
	if !ok {
		return 0
	}
	
	elapsed := time.Since(start)
	delete(p.starts, name)
	
	p.times[name] = append(p.times[name], elapsed)
	return elapsed
}

// GetTotal returns total time for an operation
func (p *Profiler) GetTotal(name string) time.Duration {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	var total time.Duration
	for _, d := range p.times[name] {
		total += d
	}
	return total
}

// GetAverage returns average time for an operation
func (p *Profiler) GetAverage(name string) time.Duration {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	if len(p.times[name]) == 0 {
		return 0
	}
	
	var total time.Duration
	for _, d := range p.times[name] {
		total += d
	}
	return total / time.Duration(len(p.times[name]))
}

// GetCount returns number of samples for an operation
func (p *Profiler) GetCount(name string) int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.times[name])
}

// Print outputs profiling results
func (p *Profiler) Print() {
	if !p.enabled {
		return
	}
	
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	fmt.Println("\n=== Profiling Results ===")
	
	// Create sorted list by total time
	type stat struct {
		name  string
		total time.Duration
		avg   time.Duration
		count int
	}
	
	var stats []stat
	for _, name := range p.order {
		if times, ok := p.times[name]; ok && len(times) > 0 {
			var total time.Duration
			for _, t := range times {
				total += t
			}
			stats = append(stats, stat{
				name:  name,
				total: total,
				avg:   total / time.Duration(len(times)),
				count: len(times),
			})
		}
	}
	
	// Sort by total time descending
	sort.Slice(stats, func(i, j int) bool {
		return stats[i].total > stats[j].total
	})
	
	fmt.Printf("%-30s %12s %12s %8s\n", "Operation", "Total", "Average", "Count")
	fmt.Println(string(make([]byte, 70)))
	
	for _, s := range stats {
		fmt.Printf("%-30s %12v %12v %8d\n", s.name, s.total, s.avg, s.count)
	}
	fmt.Println()
}

// Reset clears all profiling data
func (p *Profiler) Reset() {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.times = make(map[string][]time.Duration)
	p.starts = make(map[string]time.Time)
	p.order = p.order[:0]
}

// MemoryStats tracks memory usage
type MemoryStats struct {
	PeakAlloc    int64
	CurrentAlloc int64
	mu           sync.RWMutex
}

// NewMemoryStats creates a new memory tracker
func NewMemoryStats() *MemoryStats {
	return &MemoryStats{}
}

// Alloc records an allocation
func (m *MemoryStats) Alloc(bytes int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.CurrentAlloc += bytes
	if m.CurrentAlloc > m.PeakAlloc {
		m.PeakAlloc = m.CurrentAlloc
	}
}

// Free records a deallocation
func (m *MemoryStats) Free(bytes int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.CurrentAlloc -= bytes
	if m.CurrentAlloc < 0 {
		m.CurrentAlloc = 0
	}
}

// GetCurrent returns current allocation
func (m *MemoryStats) GetCurrent() int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.CurrentAlloc
}

// GetPeak returns peak allocation
func (m *MemoryStats) GetPeak() int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.PeakAlloc
}

// Print outputs memory stats
func (m *MemoryStats) Print() {
	peak := m.GetPeak()
	current := m.GetCurrent()
	
	peakGB := float64(peak) / (1024 * 1024 * 1024)
	currentGB := float64(current) / (1024 * 1024 * 1024)
	
	fmt.Printf("Memory: Current=%.2fGB, Peak=%.2fGB\n", currentGB, peakGB)
}
