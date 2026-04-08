// Package model provides model loading and management
package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/xander/airllm-go/internal/safetensors"
	"github.com/xander/airllm-go/pkg/tensor"
)

// Config represents a HuggingFace model configuration
type Config struct {
	Architectures        []string               `json:"architectures"`
	ModelType            string                 `json:"model_type"`
	VocabSize            int                    `json:"vocab_size"`
	HiddenSize           int                    `json:"hidden_size"`
	NumHiddenLayers      int                    `json:"num_hidden_layers"`
	NumAttentionHeads    int                    `json:"num_attention_heads"`
	NumKeyValueHeads     int                    `json:"num_key_value_heads"`
	IntermediateSize     int                    `json:"intermediate_size"`
	MaxPositionEmbedding int                    `json:"max_position_embeddings"`
	RMSNormEps           float64                `json:"rms_norm_eps"`
	HiddenAct            string                 `json:"hidden_act"`
	RoPETheta            float64                `json:"rope_theta"`
	SlidingWindow        int                    `json:"sliding_window"`
	AttentionDropout     float64                `json:"attention_dropout"`
	BosTokenID           int                    `json:"bos_token_id"`
	EosTokenID           int                    `json:"eos_token_id"`
	PadTokenID           int                    `json:"pad_token_id"`
	TorchDType           string                 `json:"torch_dtype"`
	TransformersVersion  string                 `json:"transformers_version"`
	UseCache             bool                   `json:"use_cache"`
	AutoMap              map[string]interface{} `json:"auto_map,omitempty"`
}

// LoadConfig loads a model configuration from config.json
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(filepath.Join(path, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to read config.json: %w", err)
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config.json: %w", err)
	}

	return &config, nil
}

// DetectArchitecture determines which model class to use
func (c *Config) DetectArchitecture() string {
	if len(c.Architectures) == 0 {
		return "llama"
	}

	arch := c.Architectures[0]
	switch {
	case strings.Contains(arch, "Llama"):
		return "llama"
	case strings.Contains(arch, "Qwen"):
		return "qwen"
	case strings.Contains(arch, "Mistral"):
		return "mistral"
	case strings.Contains(arch, "Mixtral"):
		return "mixtral"
	case strings.Contains(arch, "Baichuan"):
		return "baichuan"
	case strings.Contains(arch, "ChatGLM"):
		return "chatglm"
	case strings.Contains(arch, "InternLM"):
		return "internlm"
	default:
		return "llama" // Default fallback
	}
}

// GetNumKVHeads returns number of key/value heads (for GQA)
func (c *Config) GetNumKVHeads() int {
	if c.NumKeyValueHeads > 0 {
		return c.NumKeyValueHeads
	}
	return c.NumAttentionHeads
}

// LayerNames contains the naming conventions for different model architectures
type LayerNames struct {
	Embed        string
	LayerPrefix  string
	Norm         string
	LMHead       string
	RotaryPosEmb string // Optional, for models that have it
}

var architectureLayerNames = map[string]LayerNames{
	"llama": {
		Embed:       "model.embed_tokens",
		LayerPrefix: "model.layers",
		Norm:        "model.norm",
		LMHead:      "lm_head",
	},
	"qwen": {
		Embed:       "model.embed_tokens",
		LayerPrefix: "model.layers",
		Norm:        "model.norm",
		LMHead:      "lm_head",
	},
	"mistral": {
		Embed:       "model.embed_tokens",
		LayerPrefix: "model.layers",
		Norm:        "model.norm",
		LMHead:      "lm_head",
	},
	"mixtral": {
		Embed:       "model.embed_tokens",
		LayerPrefix: "model.layers",
		Norm:        "model.norm",
		LMHead:      "lm_head",
	},
	"baichuan": {
		Embed:       "model.embed_tokens",
		LayerPrefix: "model.layers",
		Norm:        "model.norm",
		LMHead:      "lm_head",
	},
	"chatglm": {
		Embed:        "transformer.embedding.word_embeddings",
		LayerPrefix:  "transformer.encoder.layers",
		Norm:         "transformer.encoder.final_layernorm",
		LMHead:       "transformer.output_layer",
		RotaryPosEmb: "transformer.rotary_pos_emb",
	},
	"internlm": {
		Embed:       "model.embed_tokens",
		LayerPrefix: "model.layers",
		Norm:        "model.norm",
		LMHead:      "lm_head",
	},
}

// GetLayerNames returns layer naming conventions for the model architecture
func (c *Config) GetLayerNames() LayerNames {
	arch := c.DetectArchitecture()
	if names, ok := architectureLayerNames[arch]; ok {
		return names
	}
	return architectureLayerNames["llama"]
}

// LayerLoader implements the inference.LayerLoader interface
type LayerLoader struct {
	checkpointPath string
	layerNames     []string
	weightMap      map[string]string // tensor name -> file name
	cache          map[string]map[string]*tensor.Tensor
	cacheMu        sync.RWMutex
	maxCacheSize   int
}

// NewLayerLoader creates a new layer loader
func NewLayerLoader(checkpointPath string) (*LayerLoader, error) {
	// Try to load index file
	indexPath := filepath.Join(checkpointPath, "model.safetensors.index.json")
	weightMap, err := safetensors.LoadModelIndex(indexPath)
	if err != nil {
		// No index file, assume single file
		weightMap = nil
	}

	return &LayerLoader{
		checkpointPath: checkpointPath,
		weightMap:      weightMap,
		cache:          make(map[string]map[string]*tensor.Tensor),
		maxCacheSize:   2, // Keep 2 layers cached
	}, nil
}

// SetArchitecture initializes layer names based on config
func (l *LayerLoader) SetArchitecture(layerNames []string) {
	l.layerNames = layerNames
}

// GetLayerCount returns the number of layers
func (l *LayerLoader) GetLayerCount() int {
	return len(l.layerNames)
}

// GetLayerName returns the name of a layer
func (l *LayerLoader) GetLayerName(idx int) string {
	if idx < 0 || idx >= len(l.layerNames) {
		return ""
	}
	return l.layerNames[idx]
}

// LoadLayer loads a single layer's tensors
func (l *LayerLoader) LoadLayer(idx int) (map[string]*tensor.Tensor, error) {
	layerName := l.GetLayerName(idx)
	if layerName == "" {
		return nil, fmt.Errorf("invalid layer index: %d", idx)
	}

	// Check cache first
	l.cacheMu.RLock()
	if cached, ok := l.cache[layerName]; ok {
		l.cacheMu.RUnlock()
		return cached, nil
	}
	l.cacheMu.RUnlock()

	// Determine which file(s) to load from
	var filePath string
	if l.weightMap != nil {
		// Find any tensor in this layer to determine the file
		sampleTensor := ""
		for tensorName := range l.weightMap {
			if strings.HasPrefix(tensorName, layerName) {
				sampleTensor = tensorName
				break
			}
		}
		if sampleTensor == "" {
			return nil, fmt.Errorf("no tensors found for layer %s", layerName)
		}
		filePath = filepath.Join(l.checkpointPath, l.weightMap[sampleTensor])
	} else {
		// Single file assumption
		files, err := filepath.Glob(filepath.Join(l.checkpointPath, "*.safetensors"))
		if err != nil || len(files) == 0 {
			return nil, fmt.Errorf("no safetensors files found")
		}
		filePath = files[0]
	}

	// Load from file
	reader, err := safetensors.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open safetensors file: %w", err)
	}
	defer reader.Close()

	// Get all tensors with this layer prefix
	state, err := reader.GetTensorsWithPrefix(layerName)
	if err != nil {
		return nil, fmt.Errorf("failed to load tensors: %w", err)
	}

	// Cache the result
	l.cacheMu.Lock()
	l.cache[layerName] = state
	// Simple cache eviction
	if len(l.cache) > l.maxCacheSize {
		// Remove oldest entry (simplistic)
		for k := range l.cache {
			if k != layerName {
				delete(l.cache, k)
				break
			}
		}
	}
	l.cacheMu.Unlock()

	return state, nil
}

// ClearCache clears the layer cache
func (l *LayerLoader) ClearCache() {
	l.cacheMu.Lock()
	defer l.cacheMu.Unlock()
	l.cache = make(map[string]map[string]*tensor.Tensor)
}

// CheckPoint represents a loaded model checkpoint
type CheckPoint struct {
	Config       *Config
	LayerLoader  *LayerLoader
	Tokenizer    *Tokenizer
	Architecture string
}

// LoadCheckPoint loads a model checkpoint from a directory
func LoadCheckPoint(path string) (*CheckPoint, error) {
	// Load config
	config, err := LoadConfig(path)
	if err != nil {
		return nil, err
	}

	// Detect architecture
	arch := config.DetectArchitecture()

	// Create layer loader
	loader, err := NewLayerLoader(path)
	if err != nil {
		return nil, err
	}

	// Build layer names list
	layerNames := config.GetLayerNames()
	allLayerNames := make([]string, 0)

	// Embedding
	allLayerNames = append(allLayerNames, layerNames.Embed)

	// Transformer layers
	for i := 0; i < config.NumHiddenLayers; i++ {
		allLayerNames = append(allLayerNames, fmt.Sprintf("%s.%d", layerNames.LayerPrefix, i))
	}

	// Final norm and LM head
	allLayerNames = append(allLayerNames, layerNames.Norm)
	allLayerNames = append(allLayerNames, layerNames.LMHead)

	// Optional: rotary embeddings
	if layerNames.RotaryPosEmb != "" {
		allLayerNames = append([]string{layerNames.RotaryPosEmb}, allLayerNames...)
	}

	loader.SetArchitecture(allLayerNames)

	return &CheckPoint{
		Config:       config,
		LayerLoader:  loader,
		Architecture: arch,
	}, nil
}

// MakeTokenizerPath guesses the tokenizer file path
func (cp *CheckPoint) MakeTokenizerPath() string {
	// Check for common tokenizer files
	candidates := []string{
		"tokenizer.json",
		"tokenizer.model",
		"tokenizer_config.json",
	}

	for _, candidate := range candidates {
		path := filepath.Join(cp.LayerLoader.checkpointPath, candidate)
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	return ""
}
