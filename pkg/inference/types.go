// Package inference — types.go
//
// Core configuration and interface types for the LeafcutterLLM inference engine.
package inference

import "github.com/Alartist40/LeafcutterLLM/pkg/tensor"

// ─── Config ───────────────────────────────────────────────────────────────────

// Config holds all hyperparameters and runtime settings for one model.
// Fields are a superset of what HuggingFace config.json provides, plus
// runtime knobs (Device, DType, NumThreads, …) used by cmd/airllm.
type Config struct {
	// ── Architecture ────────────────────────────────────────────────────────
	HiddenSize       int     `json:"hidden_size"`
	NumHiddenLayers  int     `json:"num_hidden_layers"`
	NumHeads          int `json:"num_attention_heads"`
	NumAttentionHeads  int `json:"-"` // alias for cmd/airllm compat; populated from NumHeads
	NumKVHeads       int     `json:"num_key_value_heads"`
	IntermediateSize int     `json:"intermediate_size"`
	MaxSeqLen        int     `json:"max_seq_len"`
	RMSNormEps       float32 `json:"rms_norm_eps"`
	VocabSize        int     `json:"vocab_size"`

	// ── Runtime ─────────────────────────────────────────────────────────────
	Device         string      // "cpu" | "cuda"
	DType          tensor.DType // tensor.Float32 | tensor.Float16
	NumThreads     int          // 0 = auto
	Prefetching    bool         // enable background layer prefetch
	Profiling      bool         // emit per-layer timing logs
	KVCacheEnabled bool         // enable KV cache (always true for autoregressive)
}

// DefaultConfig provides sane defaults matching LLaMA-7B.
var DefaultConfig = Config{
	HiddenSize:        4096,
	NumHiddenLayers:   32,
	NumHeads:          32,
	NumAttentionHeads: 32,
	NumKVHeads:        32,
	IntermediateSize:  11008,
	MaxSeqLen:         4096,
	RMSNormEps:        1e-6,
	VocabSize:         32000,
	Device:            "cpu",
	DType:             tensor.Float32,
	KVCacheEnabled:    true,
}

// ─── LayerLoader ──────────────────────────────────────────────────────────────

// LayerLoader defines the interface for streaming transformer layer weights
// from disk.  Only one layer's weights are held in memory at a time —
// LeafcutterLLM's core memory-efficiency technique.
type LayerLoader interface {
	// LoadLayer returns a map of weight-key → *tensor.Tensor for layer idx.
	LoadLayer(idx int) (map[string]*tensor.Tensor, error)
	// GetLayerCount returns the total number of transformer layers.
	GetLayerCount() int
	// GetLayerName returns the HF weight-key prefix for layer idx,
	// e.g. "model.layers.0".
	GetLayerName(idx int) string
	// LoadSpecialLayer loads top-level weights by name prefix (e.g. "lm_head").
	LoadSpecialLayer(name string) (map[string]*tensor.Tensor, error)
}
