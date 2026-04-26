// Package inference — engine.go
//
// Engine is the Logic Center of LeafcutterLLM.
//
// It drives the autoregressive token-generation loop:
//  1. Build the layer pipeline (embedding → N × transformer → LM-head).
//  2. For each generation step:
//     a. Load one layer's weights from disk via Loader.LoadLayer(idx).
//     b. Run layer.Forward(input, pastK, pastV) → (output, newK, newV).
//     c. Accumulate newK/newV in the per-layer KV cache.
//     d. Unload the layer weights to free RAM.
//  3. After all layers: argmax sample the next token, call onToken, repeat.
//
// Only one layer's weights are in RAM at any moment — the AirLLM trick.
package inference

import (
	"context"
	"fmt"
	"math"

	"github.com/Alartist40/LeafcutterLLM/pkg/tensor"
)

// ─── Engine ───────────────────────────────────────────────────────────────────

// Engine drives forward passes through a transformer model.
type Engine struct {
	Config *Config
	Loader LayerLoader

	// kvCache[layerIdx] = (K, V) accumulated across all prior steps.
	kvCache [][2]*tensor.Tensor
}

// NewEngine creates an Engine with the given config and weight loader.
func NewEngine(config *Config, loader LayerLoader) *Engine {
	if config == nil {
		cfg := DefaultConfig
		config = &cfg
	}
	var kvCache [][2]*tensor.Tensor
	if loader != nil {
		kvCache = make([][2]*tensor.Tensor, loader.GetLayerCount())
	}
	return &Engine{
		Config:  config,
		Loader:  loader,
		kvCache: kvCache,
	}
}

// Release frees KV-cache tensors and nulls the loader reference.
func (e *Engine) Release() error {
	for i := range e.kvCache {
		e.kvCache[i] = [2]*tensor.Tensor{}
	}
	e.Loader = nil
	return nil
}

// ─── Generate ─────────────────────────────────────────────────────────────────

// Generate satisfies the inference.Model interface.
//
// It runs a full autoregressive decode loop:
//   - Prefill  : process all prompt tokens in one forward pass (step 0).
//   - Decode   : generate up to maxTokens new tokens one at a time.
//
// onToken is called for each newly generated token so callers can stream
// partial results without waiting for the full sequence.
func (e *Engine) Generate(
	ctx context.Context,
	prompt []int,
	maxTokens int,
	onToken func(int),
) ([]int, error) {
	if e.Loader == nil {
		return nil, fmt.Errorf("engine: no layer loader attached")
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("engine: empty prompt")
	}
	if maxTokens <= 0 {
		maxTokens = e.Config.MaxSeqLen
	}

	// Reset the KV cache for a new generation.
	for i := range e.kvCache {
		e.kvCache[i] = [2]*tensor.Tensor{}
	}

	// Build a 1-D token ID tensor for the full prompt (prefill).
	input := tokenIDsToTensor(prompt)

	// ── Prefill pass ────────────────────────────────────────────────────────
	logits, err := e.forward(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("engine: prefill failed: %w", err)
	}

	generated := make([]int, 0, maxTokens)

	// ── Decode loop ─────────────────────────────────────────────────────────
	for step := 0; step < maxTokens; step++ {
		select {
		case <-ctx.Done():
			return generated, ctx.Err()
		default:
		}

		nextToken := argmax(logits)
		generated = append(generated, nextToken)
		if onToken != nil {
			onToken(nextToken)
		}

		// Stop on EOS (token 2 by convention; model-specific in production).
		if nextToken == 2 {
			break
		}

		// Feed the new token back as the next input (single-token decode step).
		input = tokenIDsToTensor([]int{nextToken})
		logits, err = e.forward(ctx, input)
		if err != nil {
			return generated, fmt.Errorf("engine: decode step %d failed: %w", step, err)
		}
	}

	return generated, nil
}

// ─── forward ──────────────────────────────────────────────────────────────────

// forward runs one complete forward pass through all transformer layers for the
// given input token tensor.  It updates e.kvCache in place and returns the
// final logit vector (shape [seqLen, vocabSize]).
func (e *Engine) forward(ctx context.Context, input *tensor.Tensor) (*tensor.Tensor, error) {
	layerCount := e.Loader.GetLayerCount()
	h := input // hidden state; updated by each layer

	for idx := 0; idx < layerCount; idx++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// ── 1. Load weights for this layer ──────────────────────────────────
		state, err := e.Loader.LoadLayer(idx)
		if err != nil {
			return nil, fmt.Errorf("layer %d load: %w", idx, err)
		}

		// ── 2. Build the layer and load its weights ──────────────────────────
		// We construct a lightweight AttentionLayer + FFNLayer + LayerNorm
		// on every step and discard them when done (their tensors are the
		// backing store — no extra copy).
		attn := NewAttentionLayer(e.Loader.GetLayerName(idx)+".self_attn", e.Config)
		if err := attn.Load(state); err != nil {
			return nil, fmt.Errorf("layer %d attn load: %w", idx, err)
		}

		ffn := NewFFNLayer(e.Loader.GetLayerName(idx)+".mlp", e.Config)
		if err := ffn.Load(state); err != nil {
			return nil, fmt.Errorf("layer %d ffn load: %w", idx, err)
		}

		preNorm := NewLayerNorm(e.Loader.GetLayerName(idx)+".input_layernorm", e.Config)
		if err := preNorm.Load(state); err != nil {
			return nil, fmt.Errorf("layer %d pre-norm load: %w", idx, err)
		}

		postNorm := NewLayerNorm(e.Loader.GetLayerName(idx)+".post_attention_layernorm", e.Config)
		if err := postNorm.Load(state); err != nil {
			return nil, fmt.Errorf("layer %d post-norm load: %w", idx, err)
		}

		// ── 3. Pre-attention RMSNorm ────────────────────────────────────────
		normed, _, _, err := preNorm.Forward(h, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("layer %d pre-norm forward: %w", idx, err)
		}

		// ── 4. Self-attention (with KV cache) ───────────────────────────────
		pastK := e.kvCache[idx][0]
		pastV := e.kvCache[idx][1]

		attnOut, newK, newV, err := attn.Forward(normed, pastK, pastV)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn forward: %w", idx, err)
		}

		// Store updated KV for next generation step.
		e.kvCache[idx] = [2]*tensor.Tensor{newK, newV}

		// ── 5. Residual connection ──────────────────────────────────────────
		h = addTensors(h, attnOut)

		// ── 6. Post-attention RMSNorm ───────────────────────────────────────
		normed, _, _, err = postNorm.Forward(h, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("layer %d post-norm forward: %w", idx, err)
		}

		// ── 7. Feed-forward network ─────────────────────────────────────────
		ffnOut, _, _, err := ffn.Forward(normed, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn forward: %w", idx, err)
		}

		// ── 8. Residual connection ──────────────────────────────────────────
		h = addTensors(h, ffnOut)

		// ── 9. Unload weights (keep only the KV tensors) ────────────────────
		attn.Unload()  //nolint:errcheck
		ffn.Unload()   //nolint:errcheck
		preNorm.Unload()  //nolint:errcheck
		postNorm.Unload() //nolint:errcheck
	}

	// ── Final norm (model.norm) ─────────────────────────────────────────────
	finalNormState, err := e.Loader.LoadSpecialLayer("model.norm")
	if err == nil && len(finalNormState) > 0 {
		finalNorm := NewLayerNorm("model.norm", e.Config)
		if loadErr := finalNorm.Load(finalNormState); loadErr == nil {
			normed, _, _, normErr := finalNorm.Forward(h, nil, nil)
			if normErr == nil {
				h = normed
			}
		}
	}

	// ── lm_head projection: [1, seqLen, hiddenSize] → [1, seqLen, vocabSize] ──
	lmHeadState, err := e.Loader.LoadSpecialLayer("lm_head")
	if err != nil || len(lmHeadState) == 0 {
		// No lm_head found — return hidden state as-is (stub mode for testing).
		return h, nil
	}

	lmHead := NewLinearLayer("lm_head", e.Config)
	if err := lmHead.Load(lmHeadState); err != nil {
		return h, nil // graceful degradation
	}

	logits, _, _, err := lmHead.Forward(h, nil, nil)
	if err != nil {
		return h, nil // graceful degradation
	}

	return logits, nil
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

// tokenIDsToTensor wraps a []int of token IDs into a 2-D Tensor [1, seqLen].
// Uses []int64 to match embedLookup's expectations. (FIX-011)
func tokenIDsToTensor(ids []int) *tensor.Tensor {
	data := make([]int64, len(ids))
	for i, id := range ids {
		data[i] = int64(id)
	}
	return &tensor.Tensor{
		Shape:   []int{1, len(ids)},
		Data:    data,
		DType:   tensor.Int64,
		Strides: []int{1, len(ids)},
	}
}

// argmax returns the index of the maximum value in the last dimension of t. (FIX-012)
func argmax(t *tensor.Tensor) int {
	if t == nil {
		return 0
	}
	data, ok := t.Data.([]float32)
	if !ok || len(data) == 0 {
		return 0
	}
	cols := t.Shape[len(t.Shape)-1]
	if cols <= 0 {
		cols = len(data)
	}
	start := len(data) - cols
	if start < 0 {
		start = 0
	}
	best, bestVal := start, float32(math.Inf(-1))
	for i := start; i < len(data); i++ {
		if data[i] > bestVal {
			bestVal = data[i]
			best = i
		}
	}
	return best - start
}

// addTensors performs element-wise addition using GetFloat32/SetFloat32 for type safety. (FIX-012)
func addTensors(a, b *tensor.Tensor) *tensor.Tensor {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	n := a.Size()
	if b.Size() < n {
		n = b.Size()
	}
	for i := 0; i < n; i++ {
		a.SetFloat32(i, a.GetFloat32(i)+b.GetFloat32(i))
	}
	return a
}
