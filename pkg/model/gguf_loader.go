package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/Alartist40/LeafcutterLLM/internal/gguf"
	"github.com/Alartist40/LeafcutterLLM/pkg/inference"
	"github.com/Alartist40/LeafcutterLLM/pkg/tensor"
)

type GGUFLayerLoader struct {
	ggufFile   *gguf.GGUFFile
	layerCount int
}

func newGGUFLayerLoader(path string) (*GGUFLayerLoader, inference.Config, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, inference.Config{}, err
	}

	cfg := extractConfigFromGGUF(g.Metadata)

	return &GGUFLayerLoader{
		ggufFile:   g,
		layerCount: cfg.NumHiddenLayers,
	}, cfg, nil
}

func extractConfigFromGGUF(metadata map[string]interface{}) inference.Config {
	cfg := inference.DefaultConfig

	// Helper to get int from multiple possible GGUF types
	getInt := func(key string) (int, bool) {
		if val, ok := metadata[key].(uint32); ok {
			return int(val), true
		}
		if val, ok := metadata[key].(uint64); ok {
			return int(val), true
		}
		if val, ok := metadata[key].(int32); ok {
			return int(val), true
		}
		if val, ok := metadata[key].(int64); ok {
			return int(val), true
		}
		return 0, false
	}

	if v, ok := getInt("llama.embedding_length"); ok {
		cfg.HiddenSize = v
	}
	if v, ok := getInt("llama.block_count"); ok {
		cfg.NumHiddenLayers = v
	}
	if v, ok := getInt("llama.attention.head_count"); ok {
		cfg.NumHeads = v
		cfg.NumAttentionHeads = v
	}
	if v, ok := getInt("llama.attention.head_count_kv"); ok {
		cfg.NumKVHeads = v
	}
	if v, ok := getInt("llama.feed_forward_length"); ok {
		cfg.IntermediateSize = v
	}
	if v, ok := getInt("llama.context_length"); ok {
		cfg.MaxSeqLen = v
	}

	// Vocab size is often the length of the tokens array
	if val, ok := metadata["tokenizer.ggml.tokens"].([]interface{}); ok {
		cfg.VocabSize = len(val)
	}

	return cfg
}

func (l *GGUFLayerLoader) GetLayerCount() int { return l.layerCount }

func (l *GGUFLayerLoader) GetLayerName(idx int) string {
	return fmt.Sprintf("model.layers.%d", idx)
}

func (l *GGUFLayerLoader) LoadLayer(idx int) (map[string]*tensor.Tensor, error) {
	prefix := fmt.Sprintf("blk.%d", idx)
	result := make(map[string]*tensor.Tensor)

	// GGUF to Engine mapping
	mappings := map[string]string{
		".attn_q.weight":      ".self_attn.q_proj.weight",
		".attn_k.weight":      ".self_attn.k_proj.weight",
		".attn_v.weight":      ".self_attn.v_proj.weight",
		".attn_output.weight": ".self_attn.o_proj.weight",
		".ffn_gate.weight":    ".mlp.gate_proj.weight",
		".ffn_up.weight":      ".mlp.up_proj.weight",
		".ffn_down.weight":    ".mlp.down_proj.weight",
		".attn_norm.weight":   ".input_layernorm.weight",
		".ffn_norm.weight":    ".post_attention_layernorm.weight",
	}

	for suffix, engineSuffix := range mappings {
		ggufName := prefix + suffix
		engineName := fmt.Sprintf("model.layers.%d%s", idx, engineSuffix)

		t, err := l.loadAndConvert(ggufName)
		if err == nil {
			result[engineName] = t
		}
	}

	return result, nil
}

func (l *GGUFLayerLoader) LoadSpecialLayer(name string) (map[string]*tensor.Tensor, error) {
	result := make(map[string]*tensor.Tensor)

	specialMappings := map[string]string{
		"token_embd.weight":  "model.embed_tokens.weight",
		"output_norm.weight": "model.norm.weight",
		"output.weight":      "lm_head.weight",
	}

	for ggufName, engineName := range specialMappings {
		// Only load if the requested name matches either GGUF or engine name
		if strings.HasPrefix(engineName, name) || strings.HasPrefix(ggufName, name) || name == "model.norm" {
			t, err := l.loadAndConvert(ggufName)
			if err == nil {
				result[engineName] = t
			}
		}
	}

	return result, nil
}

func (l *GGUFLayerLoader) loadAndConvert(name string) (*tensor.Tensor, error) {
	data, err := l.ggufFile.GetTensor(name)
	if err != nil {
		return nil, err
	}

	var tInfo *gguf.GGUFTensor
	for _, t := range l.ggufFile.Tensors {
		if t.Name == name {
			tInfo = &t
			break
		}
	}
	if tInfo == nil {
		return nil, fmt.Errorf("tensor info not found for %s", name)
	}

	return convertGGUFToTensor(data, tInfo)
}

func convertGGUFToTensor(data []byte, tInfo *gguf.GGUFTensor) (*tensor.Tensor, error) {
	shape := make([]int, len(tInfo.Dimensions))
	// GGUF dimensions are [width, height, ...] (reverse of PyTorch/HuggingFace)
	// Our engine expects [height, width] for weights.
	for i := 0; i < len(tInfo.Dimensions); i++ {
		shape[len(tInfo.Dimensions)-1-i] = int(tInfo.Dimensions[i])
	}

	switch tInfo.Type {
	case gguf.GGML_TYPE_F32:
		return tensor.FromBuffer(data, shape, tensor.Float32), nil
	case gguf.GGML_TYPE_F16:
		t := tensor.FromBuffer(data, shape, tensor.Float16)
		return t.ToFloat32(), nil
	case gguf.GGML_TYPE_Q4_0:
		return dequantizeQ4_0(data, shape), nil
	case gguf.GGML_TYPE_Q8_0:
		return dequantizeQ8_0(data, shape), nil
	default:
		return nil, fmt.Errorf("unsupported GGUF tensor type: %d", tInfo.Type)
	}
}

func dequantizeQ4_0(data []byte, shape []int) *tensor.Tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}
	out := tensor.NewTensor(shape, tensor.Float32)
	outData := out.Data.([]float32)

	// Q4_0: block of 32 values
	// 2 bytes f16 scale + 16 bytes q4
	blockSize := 32
	groupSize := 18

	for i := 0; i < size; i += blockSize {
		blockIdx := i / blockSize
		start := blockIdx * groupSize
		if start+groupSize > len(data) {
			break
		}
		blockData := data[start : start+groupSize]

		scale := float16BitsToFloat32(binary.LittleEndian.Uint16(blockData[0:2]))

		for j := 0; j < 16; j++ {
			qs := blockData[2+j]
			q0 := float32(int(qs&0x0F) - 8)
			q1 := float32(int(qs>>4) - 8)

			if i+j < size {
				outData[i+j] = q0 * scale
			}
			if i+j+16 < size {
				outData[i+j+16] = q1 * scale
			}
		}
	}
	return out
}

func dequantizeQ8_0(data []byte, shape []int) *tensor.Tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}
	out := tensor.NewTensor(shape, tensor.Float32)
	outData := out.Data.([]float32)

	// Q8_0: block of 32 values
	// 2 bytes f16 scale + 32 bytes q8
	blockSize := 32
	groupSize := 34

	for i := 0; i < size; i += blockSize {
		blockIdx := i / blockSize
		start := blockIdx * groupSize
		if start+groupSize > len(data) {
			break
		}
		blockData := data[start : start+groupSize]

		scale := float16BitsToFloat32(binary.LittleEndian.Uint16(blockData[0:2]))

		for j := 0; j < 32; j++ {
			if i+j < size {
				outData[i+j] = float32(int8(blockData[2+j])) * scale
			}
		}
	}
	return out
}

func float16BitsToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)
	var f uint32
	switch exp {
	case 0:
		if mant == 0 {
			f = sign
		} else {
			e := uint32(1)
			for mant&0x400 == 0 {
				mant <<= 1
				e++
			}
			mant &^= 0x400
			f = sign | ((127 - 15 - e + 1) << 23) | (mant << 13)
		}
	case 31:
		f = sign | 0x7F800000 | (mant << 13)
	default:
		f = sign | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(f)
}
