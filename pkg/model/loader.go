// Package model — loader.go
//
// LoadCheckPoint reads a HuggingFace-style config.json and returns a *CheckPoint
// that bundles the model metadata with a RealLayerLoader.
//
// RealLayerLoader implements inference.LayerLoader by streaming individual
// transformer layers from safetensors shards on demand via
// internal/safetensors, so only one layer is resident in RAM at a time
// (LeafcutterLLM's core memory-efficiency technique).
package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/Alartist40/LeafcutterLLM/internal/safetensors"
	"github.com/Alartist40/LeafcutterLLM/pkg/inference"
	"github.com/Alartist40/LeafcutterLLM/pkg/tensor"
)

// ─── CheckPoint ───────────────────────────────────────────────────────────────

// CheckPoint holds metadata and the layer loader for a model checkpoint.
type CheckPoint struct {
	Architecture string
	VocabSize    int
	LayerCount   int
	Config       inference.Config
	LayerLoader  inference.LayerLoader
}

// ─── LoadCheckPoint ───────────────────────────────────────────────────────────

// LoadCheckPoint reads config.json at modelPath and returns a fully initialised
// CheckPoint with a RealLayerLoader wired in.
func LoadCheckPoint(modelPath string) (*CheckPoint, error) {
	configPath := filepath.Join(modelPath, "config.json")
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model config not found at %s", configPath)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config.json: %w", err)
	}

	// Raw JSON that covers LLaMA / Mistral / GPT-2 / Qwen layouts.
	var raw struct {
		Architectures         []string `json:"architectures"`
		Arch                  string   `json:"arch"`
		VocabSize             int      `json:"vocab_size"`
		HiddenSize            int      `json:"hidden_size"`
		NumHiddenLayers       int      `json:"num_hidden_layers"`
		NumAttentionHeads     int      `json:"num_attention_heads"`
		NumKeyValueHeads      int      `json:"num_key_value_heads"`
		IntermediateSize      int      `json:"intermediate_size"`
		MaxPositionEmbeddings int      `json:"max_position_embeddings"`
		RMSNormEps            float32  `json:"rms_norm_eps"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse config.json: %w", err)
	}

	// Resolve architecture string.
	arch := raw.Arch
	if arch == "" && len(raw.Architectures) > 0 {
		arch = raw.Architectures[0]
	}

	// Merge parsed values with DefaultConfig so a sparse config.json still works.
	def := inference.DefaultConfig
	cfg := inference.Config{
		HiddenSize:        orDefault(raw.HiddenSize, def.HiddenSize),
		NumHiddenLayers:   orDefault(raw.NumHiddenLayers, def.NumHiddenLayers),
		NumHeads:          orDefault(raw.NumAttentionHeads, def.NumHeads),
		NumKVHeads:        orDefault(raw.NumKeyValueHeads, def.NumKVHeads),
		IntermediateSize:  orDefault(raw.IntermediateSize, def.IntermediateSize),
		MaxSeqLen:         orDefault(raw.MaxPositionEmbeddings, def.MaxSeqLen),
		VocabSize:         orDefault(raw.VocabSize, def.VocabSize),
		RMSNormEps:        orDefaultF32(raw.RMSNormEps, def.RMSNormEps),
		Device:            "cpu",
		DType:             tensor.Float32,
		KVCacheEnabled:    true,
	}
	cfg.NumAttentionHeads = cfg.NumHeads // keep alias in sync

	loader := newRealLayerLoader(modelPath, cfg.NumHiddenLayers)

	return &CheckPoint{
		Architecture: arch,
		VocabSize:    cfg.VocabSize,
		LayerCount:   cfg.NumHiddenLayers,
		Config:       cfg,
		LayerLoader:  loader,
	}, nil
}

// ─── helpers ──────────────────────────────────────────────────────────────────

func orDefault(val, fallback int) int {
	if val != 0 {
		return val
	}
	return fallback
}

func orDefaultF32(val, fallback float32) float32 {
	if val != 0 {
		return val
	}
	return fallback
}

// ─── RealLayerLoader ──────────────────────────────────────────────────────────

// RealLayerLoader streams transformer layer weights from safetensors shards.
// It supports two on-disk layouts:
//
//  1. Single shard  — <root>/model.safetensors  (all weights in one file)
//  2. Multi-shard   — <root>/model.safetensors.index.json  (weight→shard map)
//
// LoadLayer picks the correct shard for the requested layer index,
// opens it via internal/safetensors, reads only the tensors for that layer,
// and closes the file immediately — keeping RAM usage to one layer at a time.
type RealLayerLoader struct {
	rootPath   string
	layerCount int

	// weightMap is populated from model.safetensors.index.json when present.
	// Maps tensor key prefix → shard filename (e.g. "model.layers.0" → "model-00001-of-00003.safetensors").
	weightMap map[string]string // nil = single-shard mode
}

func newRealLayerLoader(root string, layerCount int) *RealLayerLoader {
	if layerCount <= 0 {
		layerCount = 32 // safe default (LLaMA-7B)
	}
	loader := &RealLayerLoader{
		rootPath:   root,
		layerCount: layerCount,
	}

	// Try to load a weight map for multi-shard models.
	indexPath := filepath.Join(root, "model.safetensors.index.json")
	if wm, err := safetensors.LoadModelIndex(indexPath); err == nil {
		loader.weightMap = wm
	}

	return loader
}

// GetLayerCount returns the number of transformer layers.
func (l *RealLayerLoader) GetLayerCount() int { return l.layerCount }

// GetLayerName returns the HuggingFace weight-key prefix for layer idx.
// e.g. "model.layers.0", "model.layers.31"
func (l *RealLayerLoader) GetLayerName(idx int) string {
	return fmt.Sprintf("model.layers.%d", idx)
}

// LoadLayer loads all safetensors weights whose key starts with
// "model.layers.<idx>" and returns them as a map[key]*tensor.Tensor.
//
// Memory model: the file is opened, the slice is read into RAM, and the
// file is closed before returning.  The caller (engine.go) calls
// layer.Load(state) then layer.Unload() when done, releasing the tensors.
func (l *RealLayerLoader) LoadLayer(idx int) (map[string]*tensor.Tensor, error) {
	prefix := l.GetLayerName(idx) // e.g. "model.layers.7"

	// ── Multi-shard path ──────────────────────────────────────────────────
	if l.weightMap != nil {
		// Collect all unique shard filenames that contain tensors for this layer.
		shardSet := make(map[string]struct{})
		for key, shard := range l.weightMap {
			if len(key) >= len(prefix) && key[:len(prefix)] == prefix {
				shardSet[shard] = struct{}{}
			}
		}

		if len(shardSet) == 0 {
			// No tensors mapped for this layer — return empty (zero weights).
			return map[string]*tensor.Tensor{}, nil
		}

		result := make(map[string]*tensor.Tensor)
		for shardFile := range shardSet {
			shardPath := filepath.Join(l.rootPath, shardFile)
			tensors, err := safetensors.LoadLayer(shardPath, prefix)
			if err != nil {
				return nil, fmt.Errorf("loader: shard %s layer %d: %w", shardFile, idx, err)
			}
			for k, v := range tensors {
				// Upcast Float16 → Float32 so all downstream math uses float32.
				if v != nil && v.DType == tensor.Float16 {
					v = upcastF16toF32(v)
				}
				result[k] = v
			}
		}
		return result, nil
	}

	// ── Single-shard path ─────────────────────────────────────────────────
	shardPath := filepath.Join(l.rootPath, "model.safetensors")
	if _, err := os.Stat(shardPath); os.IsNotExist(err) {
		// No safetensors file at all — return empty so the engine can still
		// run with zero-initialised weights during development.
		return map[string]*tensor.Tensor{}, nil
	}

	tensors, err := safetensors.LoadLayer(shardPath, prefix)
	if err != nil {
		return nil, fmt.Errorf("loader: single-shard layer %d: %w", idx, err)
	}

	// Upcast Float16 → Float32.
	for k, v := range tensors {
		if v != nil && v.DType == tensor.Float16 {
			tensors[k] = upcastF16toF32(v)
		}
	}
	return tensors, nil
}

// LoadSpecialLayer loads a named top-level layer (e.g. "lm_head", "model.norm")
// that is not part of the numbered transformer stack.
func (l *RealLayerLoader) LoadSpecialLayer(name string) (map[string]*tensor.Tensor, error) {
    // Same logic as LoadLayer but uses `name` as the prefix instead of
    // "model.layers.<idx>"

    // Multi-shard path
    if l.weightMap != nil {
        shardSet := make(map[string]struct{})
        for key, shard := range l.weightMap {
            if strings.HasPrefix(key, name) {
                shardSet[shard] = struct{}{}
            }
        }
        if len(shardSet) == 0 {
            return map[string]*tensor.Tensor{}, nil
        }
        result := make(map[string]*tensor.Tensor)
        for shardFile := range shardSet {
            shardPath := filepath.Join(l.rootPath, shardFile)
            tensors, err := safetensors.LoadLayer(shardPath, name)
            if err != nil {
                return nil, fmt.Errorf("loader: special layer %s shard %s: %w", name, shardFile, err)
            }
            for k, v := range tensors {
                if v != nil && v.DType == tensor.Float16 {
                    v = upcastF16toF32(v)
                }
                result[k] = v
            }
        }
        return result, nil
    }

    // Single-shard path
    shardPath := filepath.Join(l.rootPath, "model.safetensors")
    if _, err := os.Stat(shardPath); os.IsNotExist(err) {
        return map[string]*tensor.Tensor{}, nil
    }
    tensors, err := safetensors.LoadLayer(shardPath, name)
    if err != nil {
        return nil, fmt.Errorf("loader: special layer %s: %w", name, err)
    }
    for k, v := range tensors {
        if v != nil && v.DType == tensor.Float16 {
            tensors[k] = upcastF16toF32(v)
        }
    }
    return tensors, nil
}

// ─── Float16 → Float32 upcast ────────────────────────────────────────────────

// upcastF16toF32 converts a Float16 tensor to Float32.
// Safetensors stores weights as BF16 or F16; BLAS/CGO expects F32.
func upcastF16toF32(src *tensor.Tensor) *tensor.Tensor {
	u16 := src.Data.([]uint16)
	f32 := make([]float32, len(u16))
	for i, h := range u16 {
		f32[i] = float16ToFloat32(h)
	}
	dst := tensor.NewTensor(src.Shape, tensor.Float32)
	dst.Data = f32
	return dst
}

// float16ToFloat32 converts an IEEE 754 half-precision float to float32.
func float16ToFloat32(h uint16) float32 {
	// Extract components
	sign := uint32(h>>15) << 31
	exp := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)

	var f uint32
	switch exp {
	case 0: // Subnormal or zero
		if mant == 0 {
			f = sign
		} else {
			// Normalise subnormal
			exp = 1
			for mant&0x400 == 0 {
				mant <<= 1
				exp--
			}
			mant &^= 0x400
			f = sign | ((exp + 127 - 15) << 23) | (mant << 13)
		}
	case 31: // Inf or NaN
		f = sign | 0x7F800000 | (mant << 13)
	default:
		f = sign | ((exp + 127 - 15) << 23) | (mant << 13)
	}

	return *(*float32)(unsafe.Pointer(&f))
}
