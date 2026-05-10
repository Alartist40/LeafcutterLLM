package model

import (
	"github.com/Alartist40/LeafcutterLLM/pkg/inference"
)

type ModelSizeEstimate struct {
	Config               inference.Config
	QuantBits            int
	TotalParams          int64
	WeightsSize          int64
	KVCacheSize          int64
	ActivationsSize      int64
	LayerLoadingOverhead int64
	PeakMemory           int64 // Naive loading
	LeafcutterPeak       int64 // With layer-by-layer
}

// EstimateModelSize calculates expected RAM usage for a model
func EstimateModelSize(cfg inference.Config, quantBits int) ModelSizeEstimate {
	est := ModelSizeEstimate{
		Config:    cfg,
		QuantBits: quantBits,
	}

	// Calculate parameter count
	est.TotalParams = calculateParameterCount(cfg)

	// Calculate base model size (weights only)
	bytesPerParam := float64(quantBits) / 8.0
	est.WeightsSize = int64(float64(est.TotalParams) * bytesPerParam)

	// Add overhead for KV cache
	est.KVCacheSize = calculateKVCacheSize(cfg)

	// Add overhead for activations (temporary buffers)
	est.ActivationsSize = calculateActivationSize(cfg)

	// Single layer size (The layer is loaded in quantized form, then dequantized)
	// We count the quantized size as part of the total weights, but for the 
	// active peak, we need the size of ONE layer at its storage quantization level.
	// LeafcutterLLM streams these, so only one is resident.
	singleLayerParams := calculateSingleLayerParams(cfg)
	est.LayerLoadingOverhead = int64(float64(singleLayerParams) * bytesPerParam)

	// System overhead (buffers, Go runtime, etc.)
	const overhead = 500 * 1024 * 1024 // 500 MB

	// Total peak memory (naive loading would keep all weights)
	est.PeakMemory = est.WeightsSize + est.KVCacheSize +
		est.ActivationsSize + overhead

	// Estimate with LeafcutterLLM's layer-by-layer optimization
	// Peak = Single Layer (Quantized) + KV Cache + Activations + Overhead
	// Embeddings and special layers are also streamed/loaded on-demand or 
	// part of the layer count, so they don't stay in RAM as a whole.
	est.LeafcutterPeak = est.LayerLoadingOverhead + est.KVCacheSize +
		est.ActivationsSize + overhead

	return est
}

func calculateSingleLayerParams(cfg inference.Config) int64 {
	params := int64(0)

	// Attention: Q, K, V, O projections
	params += int64(cfg.HiddenSize * cfg.HiddenSize * 4)

	// FFN: gate, up, down
	params += int64(cfg.HiddenSize * cfg.IntermediateSize * 3)

	// Layer norms (2 per layer)
	params += int64(cfg.HiddenSize * 2)

	return params
}

func calculateParameterCount(cfg inference.Config) int64 {
	params := int64(0)

	// Embedding layer
	params += int64(cfg.VocabSize * cfg.HiddenSize)

	// Each transformer layer
	params += calculateSingleLayerParams(cfg) * int64(cfg.NumHiddenLayers)

	// Final layer norm
	params += int64(cfg.HiddenSize)

	// LM head (hidden → vocab)
	params += int64(cfg.HiddenSize * cfg.VocabSize)

	return params
}

func calculateKVCacheSize(cfg inference.Config) int64 {
	// KV cache = 2 (K+V) * num_layers * num_heads * head_dim * max_seq_len * 4 bytes
	// head_dim = hidden_size / num_heads
	if cfg.NumHeads <= 0 {
		return 0
	}
	headDim := cfg.HiddenSize / cfg.NumHeads
	return int64(2 * cfg.NumHiddenLayers * cfg.NumHeads * headDim * cfg.MaxSeqLen * 4)
}

func calculateActivationSize(cfg inference.Config) int64 {
	// Rough but more principled estimate:
	// 1. Hidden states: 2 * batch * seq * hidden * 4 (float32)
	// 2. Attention scores: batch * heads * seq * seq * 4 (float32)
	// 3. FFN intermediate: batch * seq * intermediate * 4 (float32)
	// We sum the largest concurrent allocations.
	batchSize := 1
	
	hiddenStates := int64(2 * batchSize * cfg.MaxSeqLen * cfg.HiddenSize * 4)
	ffnIntermediate := int64(batchSize * cfg.MaxSeqLen * cfg.IntermediateSize * 4)
	
	// For attention scores, we use a simplified max seq len
	attnScores := int64(batchSize * cfg.NumHeads * cfg.MaxSeqLen * cfg.MaxSeqLen * 4)
	
	// If MaxSeqLen is huge, attnScores dominates. 
	// We'll take the max of (hidden + ffn) and (hidden + attn)
	peak := hiddenStates + ffnIntermediate
	if hiddenStates + attnScores > peak {
		peak = hiddenStates + attnScores
	}
	
	return peak
}

func calculateLayerSize(cfg inference.Config, bits int) int64 {
	// Single layer size
	bytesPerParam := float64(bits) / 8.0

	layerParams := int64(0)
	layerParams += int64(cfg.HiddenSize * cfg.HiddenSize * 4)       // Attention
	layerParams += int64(cfg.HiddenSize * cfg.IntermediateSize * 3) // FFN
	layerParams += int64(cfg.HiddenSize * 2)                       // Norms

	return int64(float64(layerParams) * bytesPerParam)
}
