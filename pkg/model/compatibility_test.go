package model

import (
	"testing"
	"github.com/Alartist40/LeafcutterLLM/pkg/inference"
)

func TestCheckCompatibilityWithConfig(t *testing.T) {
	cfg := inference.Config{
		HiddenSize:       4096,
		NumHiddenLayers:  32,
		NumHeads:         32,
		IntermediateSize: 11008,
		MaxSeqLen:        2048,
		VocabSize:        32000,
	}

	report, err := CheckCompatibilityWithConfig(cfg, 4)
	if err != nil {
		t.Fatalf("CheckCompatibilityWithConfig failed: %v", err)
	}

	if report.Hardware == nil {
		t.Error("expected hardware info in report")
	}

	if report.MemorySavingsX <= 1.0 {
		t.Errorf("expected memory savings > 1.0, got %f", report.MemorySavingsX)
	}
}

func TestCheckCompatibilityInvalidConfig(t *testing.T) {
	cfg := inference.Config{
		NumHiddenLayers: 0,
	}
	_, err := CheckCompatibilityWithConfig(cfg, 4)
	if err == nil {
		t.Error("expected error for invalid config")
	}
}
