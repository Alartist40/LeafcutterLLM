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

	// Single layer size (Dequantized to Float32 during inference)
	est.LayerLoadingOverhead = calculateLayerSize(cfg, 32)

	// System overhead (buffers, Go runtime, etc.)
	const overhead = 500 * 1024 * 1024 // 500 MB

	// Total peak memory (naive loading would keep all weights)
	est.PeakMemory = est.WeightsSize + est.KVCacheSize +
		est.ActivationsSize + overhead

	// Estimate with LeafcutterLLM's layer-by-layer optimization
	// Peak = Single Layer (F32) + KV Cache + Activations + Embeddings + Overhead
	embeddingsSize := int64(cfg.VocabSize * cfg.HiddenSize * 4) // F32
	est.LeafcutterPeak = est.LayerLoadingOverhead + est.KVCacheSize +
		est.ActivationsSize + embeddingsSize + overhead

	return est
}

func calculateParameterCount(cfg inference.Config) int64 {
	params := int64(0)

	// Embedding layer
	params += int64(cfg.VocabSize * cfg.HiddenSize)

	// Each transformer layer
	layerParams := int64(0)

	// Attention: Q, K, V, O projections
	layerParams += int64(cfg.HiddenSize * cfg.HiddenSize * 4)

	// FFN: gate, up, down
	layerParams += int64(cfg.HiddenSize * cfg.IntermediateSize * 3)

	// Layer norms (2 per layer)
	layerParams += int64(cfg.HiddenSize * 2)

	// Multiply by number of layers
	params += layerParams * int64(cfg.NumHiddenLayers)

	// Final layer norm
	params += int64(cfg.HiddenSize)

	// LM head (hidden → vocab)
	params += int64(cfg.HiddenSize * cfg.VocabSize)

	return params
}

func calculateKVCacheSize(cfg inference.Config) int64 {
	// KV cache: 2 (K+V) * num_layers * hidden_size * max_seq_len
	// Assuming float32 (4 bytes per value) as that's what we use in engine.go
	return int64(2 * cfg.NumHiddenLayers * cfg.HiddenSize * cfg.MaxSeqLen * 4)
}

func calculateActivationSize(cfg inference.Config) int64 {
	// Temporary activations for attention and FFN
	// Rough estimate: batch_size * seq_len * hidden_size * 4
	batchSize := 1 // Assuming single sequence
	return int64(batchSize * cfg.MaxSeqLen * cfg.HiddenSize * 4 * 4) // float32 overhead
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
