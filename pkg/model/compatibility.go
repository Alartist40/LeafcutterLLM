package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Alartist40/LeafcutterLLM/internal/gguf"
	"github.com/Alartist40/LeafcutterLLM/pkg/inference"
	"github.com/Alartist40/LeafcutterLLM/pkg/utils"
)

type CompatibilityLevel int

const (
	Compatible   CompatibilityLevel = iota // ✅ Green
	Marginal                               // ⚠️ Yellow
	Incompatible                           // ❌ Red
)

type CompatibilityReport struct {
	Level          CompatibilityLevel
	CanRun         bool
	Warning        string
	Hardware       *utils.HardwareInfo
	ModelSize      ModelSizeEstimate
	RecommendedRAM int64
	SafetyMargin   float64 // 0.0-1.0
	MemorySavingsX float64 // e.g. 11.0 for 11x savings
}

// CheckCompatibility determines if a model can run on current hardware
func CheckCompatibility(mInfo ModelInfo, quantBits int) (*CompatibilityReport, error) {
	// Detect hardware
	hw, err := utils.DetectHardware()
	if err != nil {
		return nil, fmt.Errorf("failed to detect hardware: %w", err)
	}

	// Load model config (lightweight - just metadata)
	cfg, err := getModelConfig(mInfo.Path, mInfo.Format)
	if err != nil {
		return nil, err
	}

	// Estimate model size
	estimate := EstimateModelSize(cfg, quantBits)

	report := &CompatibilityReport{
		Hardware:  hw,
		ModelSize: estimate,
	}

	// LeafcutterLLM Advantage: How much we save vs naive loading
	if estimate.LeafcutterPeak > 0 {
		report.MemorySavingsX = float64(estimate.PeakMemory) / float64(estimate.LeafcutterPeak)
	}

	// Required RAM for LeafcutterLLM
	requiredRAM := estimate.LeafcutterPeak
	report.RecommendedRAM = requiredRAM

	// Check if model fits
	availableRAM := float64(hw.AvailableRAM)
	if availableRAM == 0 {
		availableRAM = float64(hw.TotalRAM) * 0.8 // Fallback if AvailableRAM is 0
	}
	totalRAM := float64(hw.TotalRAM)

	ratio := float64(requiredRAM) / availableRAM
	report.SafetyMargin = 1.0 - ratio

	if requiredRAM > int64(totalRAM*1.2) {
		// Model way too big even for LeafcutterLLM + small swap
		report.Level = Incompatible
		report.CanRun = false
		report.Warning = fmt.Sprintf(
			"Model requires %.1f GB (LeafcutterLLM peak) but system only has %.1f GB total RAM",
			float64(requiredRAM)/1e9,
			totalRAM/1e9,
		)
	} else if ratio > 0.95 || requiredRAM > int64(totalRAM) {
		// Model fits but very tight, will likely use swap
		report.Level = Marginal
		report.CanRun = true
		report.Warning = fmt.Sprintf(
			"TIGHT BUT POSSIBLE: Model uses %.0f%% of available RAM - expect slow performance due to swap usage",
			ratio*100,
		)
	} else if ratio > 0.7 {
		// Tight but workable
		report.Level = Marginal
		report.CanRun = true
		report.Warning = fmt.Sprintf(
			"Model will use %.0f%% of available RAM - may be slow",
			ratio*100,
		)
	} else {
		// Comfortable margin
		report.Level = Compatible
		report.CanRun = true
		report.Warning = ""
	}

	return report, nil
}

func getModelConfig(path, format string) (inference.Config, error) {
	switch format {
	case "gguf":
		return getGGUFConfig(path)
	case "safetensors":
		return getSafetensorsConfig(path)
	default:
		return inference.Config{}, fmt.Errorf("unknown format: %s", format)
	}
}

func getGGUFConfig(path string) (inference.Config, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return inference.Config{}, err
	}
	defer g.Close()
	return extractConfigFromGGUF(g.Metadata), nil
}

func getSafetensorsConfig(modelPath string) (inference.Config, error) {
	configPath := filepath.Join(modelPath, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return inference.Config{}, err
	}

	var raw struct {
		VocabSize             int     `json:"vocab_size"`
		HiddenSize            int     `json:"hidden_size"`
		NumHiddenLayers       int     `json:"num_hidden_layers"`
		NumAttentionHeads     int     `json:"num_attention_heads"`
		NumKeyValueHeads      int     `json:"num_key_value_heads"`
		IntermediateSize      int     `json:"intermediate_size"`
		MaxPositionEmbeddings int     `json:"max_position_embeddings"`
		RMSNormEps            float32 `json:"rms_norm_eps"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return inference.Config{}, err
	}

	cfg := inference.DefaultConfig
	if raw.HiddenSize != 0 {
		cfg.HiddenSize = raw.HiddenSize
	}
	if raw.NumHiddenLayers != 0 {
		cfg.NumHiddenLayers = raw.NumHiddenLayers
	}
	if raw.NumAttentionHeads != 0 {
		cfg.NumHeads = raw.NumAttentionHeads
	}
	if raw.NumKeyValueHeads != 0 {
		cfg.NumKVHeads = raw.NumKeyValueHeads
	}
	if raw.IntermediateSize != 0 {
		cfg.IntermediateSize = raw.IntermediateSize
	}
	if raw.MaxPositionEmbeddings != 0 {
		cfg.MaxSeqLen = raw.MaxPositionEmbeddings
	}
	if raw.VocabSize != 0 {
		cfg.VocabSize = raw.VocabSize
	}

	return cfg, nil
}

func formatParams(params int64) string {
	if params < 1e6 {
		return fmt.Sprintf("%d", params)
	}
	if params < 1e9 {
		return fmt.Sprintf("%.1fM", float64(params)/1e6)
	}
	return fmt.Sprintf("%.1fB", float64(params)/1e9)
}
