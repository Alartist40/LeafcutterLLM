// Package inference provides the core layer-by-layer inference engine
package inference

import (
	"context"
	"fmt"
	"log"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/xander/airllm-go/pkg/tensor"
)

// Layer represents a neural network layer
type Layer interface {
	Forward(input, pastK, pastV *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error)
	Load(state map[string]*tensor.Tensor) error
	Unload() error
	Name() string
}

// KVCache stores key-value pairs for transformer attention
type KVCache struct {
	mu     sync.RWMutex
	keys   []*tensor.Tensor
	values []*tensor.Tensor
}

// NewKVCache creates a new KV cache for the specified number of layers
func NewKVCache(numLayers int) *KVCache {
	return &KVCache{
		keys:   make([]*tensor.Tensor, numLayers),
		values: make([]*tensor.Tensor, numLayers),
	}
}

// Get returns cached KV tensors for a layer
func (c *KVCache) Get(layerIdx int) (*tensor.Tensor, *tensor.Tensor) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.keys[layerIdx], c.values[layerIdx]
}

// Append appends new KV tensors to the cache by replacing them with the fully concatenated ones
// Actually, since AttentionLayer does the concat, if we store the full tensor, we just Set it.
func (c *KVCache) Set(layerIdx int, k, v *tensor.Tensor) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.keys[layerIdx] = k
	c.values[layerIdx] = v
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
	Prefetching      bool          // Enable prefetching of next layer
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
		Prefetching:    true,
		Profiling:      false,
		KVCacheEnabled: true,
	}
}

// Engine executes layer-by-layer inference
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

// Forward runs inference through all layers.
func (e *Engine) Forward(ctx context.Context, input *tensor.Tensor) (*tensor.Tensor, error) {
	e.profiler.Start("total_forward")

	// Ensure engine.Forward clears this cache properly if a new context starts (new prompt)
	if input.Shape[1] > 1 {
		e.ClearCache()
	}

	numLayers := e.layerLoader.GetLayerCount()
	current := input

	// Buffered channel: at most one prefetch in flight at a time.
	var prefetchChan chan *prefetchResult
	if e.config.Prefetching && numLayers > 1 {
		prefetchChan = make(chan *prefetchResult, 1)
	}

	for i := 0; i < numLayers; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		e.profiler.Start(fmt.Sprintf("layer_%d", i))
		e.profiler.Start(fmt.Sprintf("layer_%d_load", i))

		var layerState map[string]*tensor.Tensor
		var err error

		if e.config.Prefetching && i > 0 && prefetchChan != nil {
			// Consume the prefetch result launched in the previous iteration.
			result := <-prefetchChan
			if result.err != nil {
				return nil, fmt.Errorf("prefetch failed for layer %d: %w", i, result.err)
			}
			layerState = result.state
		} else {
			// Direct load for layer 0 (or when prefetching is off).
			layerState, err = e.layerLoader.LoadLayer(i)
			if err != nil {
				return nil, fmt.Errorf("failed to load layer %d: %w", i, err)
			}
		}
		e.profiler.End(fmt.Sprintf("layer_%d_load", i))

		// Kick off background load of the *next* layer (if there is one).
		if e.config.Prefetching && prefetchChan != nil && i+1 < numLayers {
			go e.prefetchLayer(i+1, prefetchChan)
		}

		// Create layer object if needed.
		if e.layers[i] == nil {
			layer, err := e.createLayer(i, layerState)
			if err != nil {
				return nil, fmt.Errorf("failed to create layer %d: %w", i, err)
			}
			e.layers[i] = layer
		}

		// Load weights into layer.
		if err := e.layers[i].Load(layerState); err != nil {
			return nil, fmt.Errorf("failed to load state into layer %d: %w", i, err)
		}

		e.profiler.Start(fmt.Sprintf("layer_%d_compute", i))
		
		var output, newK, newV *tensor.Tensor
		if e.config.KVCacheEnabled {
			pastK, pastV := e.kvCache.Get(i)
			output, newK, newV, err = e.layers[i].Forward(current, pastK, pastV)
			if err == nil && newK != nil && newV != nil {
			    // The AttentionLayer returned newK and newV.
			    // Wait, if it returned the new slice, we need to concatenate it.
			    // But wait, my implementation of layers.go returned output, newK, newV.
			    // If pastK, pastV are provided, we should concat them here.
			    if pastK != nil && pastV != nil {
			        newK, _ = concatTensorsOnSeqDim(pastK, newK)
			        newV, _ = concatTensorsOnSeqDim(pastV, newV)
			    }
				e.kvCache.Set(i, newK, newV)
			}
		} else {
			output, _, _, err = e.layers[i].Forward(current, nil, nil)
		}
		
		if err != nil {
			return nil, fmt.Errorf("forward failed at layer %d: %w", i, err)
		}
		e.profiler.End(fmt.Sprintf("layer_%d_compute", i))

		e.profiler.Start(fmt.Sprintf("layer_%d_unload", i))
		if err := e.layers[i].Unload(); err != nil {
			log.Printf("Warning: failed to unload layer %d: %v", i, err)
		}
		e.profiler.End(fmt.Sprintf("layer_%d_unload", i))

		// Free OS memory after every unload — more targeted than periodic GC.
		debug.FreeOSMemory()

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
// FIX: Use strings.Contains (stdlib) — removes the buggy hand-rolled contains.
// FIX: isLMHead had a duplicate predicate ("lm_head" || "lm_head");
//      second arm is now "output_layer" for ChatGLM compatibility.
func isEmbeddingLayer(name string) bool {
	return strings.Contains(name, "embed_tokens") ||
		strings.Contains(name, "embeddings") ||
		strings.Contains(name, "word_embeddings")
}

func isAttentionLayer(name string) bool {
	return strings.Contains(name, "self_attn") ||
		strings.Contains(name, "attention") ||
		strings.Contains(name, "attn")
}

func isFFNLayer(name string) bool {
	return strings.Contains(name, "mlp") ||
		strings.Contains(name, "feed_forward") ||
		strings.Contains(name, "ffn")
}

func isLayerNorm(name string) bool {
	return strings.Contains(name, "norm") || strings.Contains(name, "ln_")
}

func isLMHead(name string) bool {
	return strings.Contains(name, "lm_head") || strings.Contains(name, "output_layer")
}

// contains is kept for backward compat but delegates to stdlib.
func contains(s, substr string) bool {
	return strings.Contains(s, substr)
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
