// Package inference provides the core Fragment-Streaming inference engine
package inference

import (
	"context"
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"

	"github.com/Alartist40/LeafcutterLLM/pkg/tensor"
)

// Layer represents a neural network layer
type Layer interface {
	Forward(input *tensor.Tensor) (*tensor.Tensor, error)
	Load(state map[string]*tensor.Tensor) error
	Unload() error
	Name() string
}

// KVCache stores key-value pairs for transformer attention
type KVCache struct {
	mu     sync.RWMutex
	keys   [][]*tensor.Tensor
	values [][]*tensor.Tensor
}

// NewKVCache creates a new KV cache for the specified number of layers
func NewKVCache(numLayers int) *KVCache {
	return &KVCache{
		keys:   make([][]*tensor.Tensor, numLayers),
		values: make([][]*tensor.Tensor, numLayers),
	}
}

// Get returns cached KV tensors for a layer
func (c *KVCache) Get(layerIdx int) ([]*tensor.Tensor, []*tensor.Tensor) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.keys[layerIdx], c.values[layerIdx]
}

// Append appends new KV tensors to the cache
func (c *KVCache) Append(layerIdx int, k, v *tensor.Tensor) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.keys[layerIdx] = append(c.keys[layerIdx], k)
	c.values[layerIdx] = append(c.values[layerIdx], v)
}

// Clear clears the cache for a specific layer or all layers
func (c *KVCache) Clear(layerIdx ...int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if len(layerIdx) == 0 {
		// Clear all
		for i := range c.keys {
			c.keys[i] = nil
			c.values[i] = nil
		}
	} else {
		for _, idx := range layerIdx {
			if idx >= 0 && idx < len(c.keys) {
				c.keys[idx] = nil
				c.values[idx] = nil
			}
		}
	}
}

// Config holds inference configuration
type Config struct {
	Device           string        // "cpu", "cuda", "cuda:0", etc.
	DType            tensor.DType  // Float32, Float16
	MaxSeqLen        int
	NumThreads       int
	AnticipatoryAssemblyPipelines      bool          // Enable AnticipatoryAssemblyPipelines of next layer
	Profiling        bool          // Enable timing profiling
	KVCacheEnabled   bool          // Use KV caching
}

// DefaultConfig returns sensible defaults
func DefaultConfig() *Config {
	return &Config{
		Device:         "cpu",
		DType:          tensor.Float16,
		MaxSeqLen:      2048,
		NumThreads:     runtime.NumCPU(),
		AnticipatoryAssemblyPipelines:    true,
		Profiling:      false,
		KVCacheEnabled: true,
	}
}

// Engine executes Fragment-Streaming inference
type Engine struct {
	config      *Config
	layers      []Layer
	layerLoader LayerLoader
	profiler    *Profiler
	kvCache     *KVCache
}

// LayerLoader loads layer weights from disk
type LayerLoader interface {
	LoadLayer(idx int) (map[string]*tensor.Tensor, error)
	GetLayerCount() int
	GetLayerName(idx int) string
}

// NewEngine creates a new inference engine
func NewEngine(config *Config, loader LayerLoader) *Engine {
	if config == nil {
		config = DefaultConfig()
	}
	if config.NumThreads <= 0 {
		config.NumThreads = runtime.NumCPU()
	}

	return &Engine{
		config:      config,
		layers:      make([]Layer, loader.GetLayerCount()),
		layerLoader: loader,
		profiler:    NewProfiler(config.Profiling),
		kvCache:     NewKVCache(loader.GetLayerCount()),
	}
}

// Forward runs inference through all layers
func (e *Engine) Forward(ctx context.Context, input *tensor.Tensor) (*tensor.Tensor, error) {
	e.profiler.Start("total_forward")
	
	numLayers := e.layerLoader.GetLayerCount()
	current := input
	
	// Channel for AnticipatoryAssemblyPipelines next layer
	var prefetchChan chan *prefetchResult
	if e.config.AnticipatoryAssemblyPipelines {
		prefetchChan = make(chan *prefetchResult, 1)
	}

	for i := 0; i < numLayers; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Kick off prefetch for next layer
		if e.config.AnticipatoryAssemblyPipelines && i+1 < numLayers {
			go e.prefetchLayer(i + 1, prefetchChan)
		}

		e.profiler.Start(fmt.Sprintf("layer_%d", i))
		
		// Load current layer
		var layerState map[string]*tensor.Tensor
		var err error
		
		e.profiler.Start(fmt.Sprintf("layer_%d_load", i))
		if e.config.AnticipatoryAssemblyPipelines && i > 0 {
			// Wait for prefetch result
			result := <-prefetchChan
			if result.err != nil {
				return nil, fmt.Errorf("prefetch failed for layer %d: %w", i, result.err)
			}
			layerState = result.state
		} else {
			layerState, err = e.layerLoader.LoadLayer(i)
			if err != nil {
				return nil, fmt.Errorf("failed to load layer %d: %w", i, err)
			}
		}
		e.profiler.End(fmt.Sprintf("layer_%d_load", i))

		// Create layer if not exists
		if e.layers[i] == nil {
			layer, err := e.createLayer(i, layerState)
			if err != nil {
				return nil, fmt.Errorf("failed to create layer %d: %w", i, err)
			}
			e.layers[i] = layer
		}

		// Load state into layer
		if err := e.layers[i].Load(layerState); err != nil {
			return nil, fmt.Errorf("failed to load state into layer %d: %w", i, err)
		}

		// Run forward pass
		e.profiler.Start(fmt.Sprintf("layer_%d_compute", i))
		
		// Handle KV cache for attention layers
		if e.config.KVCacheEnabled && isAttentionLayer(e.layerLoader.GetLayerName(i)) {
			// TODO: Implement KV cache injection
		}
		
		output, err := e.layers[i].Forward(current)
		if err != nil {
			return nil, fmt.Errorf("forward failed at layer %d: %w", i, err)
		}
		e.profiler.End(fmt.Sprintf("layer_%d_compute", i))

		// Unload layer to free memory
		e.profiler.Start(fmt.Sprintf("layer_%d_unload", i))
		if err := e.layers[i].Unload(); err != nil {
			log.Printf("Warning: failed to unload layer %d: %v", i, err)
		}
		e.profiler.End(fmt.Sprintf("layer_%d_unload", i))

		// Force garbage collection periodically
		if i%10 == 0 {
			runtime.GC()
		}

		current = output
		e.profiler.End(fmt.Sprintf("layer_%d", i))
	}

	e.profiler.End("total_forward")
	
	if e.config.Profiling {
		e.profiler.Print()
	}

	return current, nil
}

// ForwardBatch runs batched inference
func (e *Engine) ForwardBatch(ctx context.Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	// For simplicity, process sequentially
	// In production, could use parallel processing for independent sequences
	results := make([]*tensor.Tensor, len(inputs))
	for i, input := range inputs {
		output, err := e.Forward(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("batch item %d failed: %w", i, err)
		}
		results[i] = output
	}
	return results, nil
}

// Generate generates tokens autoregressively
func (e *Engine) Generate(ctx context.Context, inputIDs []int, maxNewTokens int, callback func(int)) ([]int, error) {
	// Convert input IDs to tensor
	inputShape := []int{1, len(inputIDs)}
	inputTensor := tensor.NewTensor(inputShape, tensor.Int64)
	for i, id := range inputIDs {
		// Store as int64
		idx := i * 8
		for j := 0; j < 8; j++ {
			inputTensor.Data[idx+j] = byte(id >> (8 * j))
		}
	}

	generated := make([]int, 0, maxNewTokens)
	
	for i := 0; i < maxNewTokens; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Run forward pass
		output, err := e.Forward(ctx, inputTensor)
		if err != nil {
			return nil, fmt.Errorf("generation step %d failed: %w", i, err)
		}

		// Get logits and sample next token
		nextToken := e.sampleNextToken(output)
		generated = append(generated, nextToken)

		if callback != nil {
			callback(nextToken)
		}

		// Update input for next iteration (KV cache handles past context)
		inputTensor = e.prepareNextInput(nextToken)
	}

	return generated, nil
}

// sampleNextToken samples the next token from logits
func (e *Engine) sampleNextToken(logits *tensor.Tensor) int {
	// Simple greedy sampling - take argmax
	// In production could implement temperature, top-k, top-p sampling
	
	lastDim := logits.Shape[len(logits.Shape)-1]
	offset := logits.Size() - lastDim

	maxIdx := 0
	maxVal := float32(-1e38)
	
	for i := 0; i < lastDim; i++ {
		var val float32
		switch logits.DType {
		case tensor.Float32:
			val = logits.GetFloat32(offset + i)
		case tensor.Float16:
			val = logits.GetFloat16(offset + i)
		}
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	return maxIdx
}

// prepareNextInput prepares the input for the next generation step
func (e *Engine) prepareNextInput(nextToken int) *tensor.Tensor {
	// Just the new token since KV cache handles past context
	shape := []int{1, 1}
	t := tensor.NewTensor(shape, tensor.Int64)
	for j := 0; j < 8; j++ {
		t.Data[j] = byte(nextToken >> (8 * j))
	}
	return t
}

// prefetchResult holds the result of a prefetch operation
type prefetchResult struct {
	state map[string]*tensor.Tensor
	err   error
}

// prefetchLayer loads a layer in the background
func (e *Engine) prefetchLayer(idx int, ch chan *prefetchResult) {
	start := time.Now()
	state, err := e.layerLoader.LoadLayer(idx)
	if err != nil {
		ch <- &prefetchResult{err: err}
		return
	}
	ch <- &prefetchResult{state: state}
	
	if e.config.Profiling {
		log.Printf("Prefetched layer %d in %v", idx, time.Since(start))
	}
}

// createLayer creates a Layer implementation based on the layer index and state
func (e *Engine) createLayer(idx int, state map[string]*tensor.Tensor) (Layer, error) {
	name := e.layerLoader.GetLayerName(idx)
	
	switch {
	case isEmbeddingLayer(name):
		return NewEmbeddingLayer(name), nil
	case isAttentionLayer(name):
		return NewAttentionLayer(name, e.config), nil
	case isFFNLayer(name):
		return NewFFNLayer(name, e.config), nil
	case isLayerNorm(name):
		return NewLayerNorm(name, e.config), nil
	case isLMHead(name):
		return NewLinearLayer(name, e.config), nil
	default:
		// Default to a passthrough/linear layer
		return NewLinearLayer(name, e.config), nil
	}
}

// Layer type detection helpers
func isEmbeddingLayer(name string) bool {
	return contains(name, "embed_tokens") || contains(name, "embeddings") || contains(name, "token")
}

func isAttentionLayer(name string) bool {
	return contains(name, "self_attn") || contains(name, "attention") || contains(name, "attn")
}

func isFFNLayer(name string) bool {
	return contains(name, "mlp") || contains(name, "feed_forward") || contains(name, "fc")
}

func isLayerNorm(name string) bool {
	return contains(name, "norm") || contains(name, "ln_")
}

func isLMHead(name string) bool {
	return contains(name, "lm_head") || contains(name, "lm_head")
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		   (s == substr || 
		    len(s) > len(substr) && 
			(s[:len(substr)] == substr || 
			 s[len(s)-len(substr):] == substr ||
			 containsSubstring(s, substr)))
}

func containsSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// ClearCache clears the KV cache
func (e *Engine) ClearCache() {
	e.kvCache.Clear()
}

// Release frees all resources
func (e *Engine) Release() {
	for _, layer := range e.layers {
		if layer != nil {
			layer.Unload()
		}
	}
	e.layers = nil
	e.ClearCache()
}
