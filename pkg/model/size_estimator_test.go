package model

import (
	"testing"
	"github.com/Alartist40/LeafcutterLLM/pkg/inference"
)

func TestEstimateModelSize(t *testing.T) {
	cfg := inference.Config{
		HiddenSize:       4096,
		NumHiddenLayers:  32,
		NumHeads:         32,
		IntermediateSize: 11008,
		MaxSeqLen:        2048,
		VocabSize:        32000,
	}

	// 7B Q4 model estimate
	est := EstimateModelSize(cfg, 4)

	if est.TotalParams <= 0 {
		t.Errorf("expected positive total params, got %d", est.TotalParams)
	}

	// Naive estimate should be around 3.5GB weights + cache
	if est.PeakMemory < 3*1024*1024*1024 {
		t.Errorf("naive peak memory too low: %d", est.PeakMemory)
	}

	// Leafcutter peak should be around 2.5GB-3GB due to 2GB KV cache
	if est.LeafcutterPeak > 4*1024*1024*1024 {
		t.Errorf("leafcutter peak memory too high: %d", est.LeafcutterPeak)
	}
	
	if est.LeafcutterPeak >= est.PeakMemory {
		t.Errorf("leafcutter peak should be less than naive peak")
	}
}

func TestEstimateModelSizeInvalid(t *testing.T) {
	cfg := inference.Config{
		HiddenSize: 0,
	}
	est := EstimateModelSize(cfg, 4)
	if est.PeakMemory != 0 {
		t.Errorf("expected 0 peak memory for invalid config")
	}
}

func TestCalculateKVCacheSize(t *testing.T) {
	cfg := inference.Config{
		HiddenSize:      4096,
		NumHeads:         32,
		NumHiddenLayers: 32,
		MaxSeqLen:       2048,
	}
	
	// head_dim = 128
	// KV = 2 * 32 * 32 * 128 * 2048 * 4 = 2,147,483,648 bytes = 2GB
	size := calculateKVCacheSize(cfg)
	expected := int64(2147483648)
	if size != expected {
		t.Errorf("expected KV cache size %d, got %d", expected, size)
	}
}
