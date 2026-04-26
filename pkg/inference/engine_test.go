package inference_test

import (
    "context"
    "fmt"
    "testing"
    "github.com/Alartist40/LeafcutterLLM/pkg/inference"
    "github.com/Alartist40/LeafcutterLLM/pkg/tensor"
)

// stubLoader implements LayerLoader with zero-weight layers.
type stubLoader struct {
    layerCount int
}

func (s *stubLoader) GetLayerCount() int { return s.layerCount }
func (s *stubLoader) GetLayerName(idx int) string {
    return fmt.Sprintf("model.layers.%d", idx)
}
func (s *stubLoader) LoadLayer(idx int) (map[string]*tensor.Tensor, error) {
    // Return empty state — layers will use zero weights.
    return map[string]*tensor.Tensor{}, nil
}

func (s *stubLoader) LoadSpecialLayer(name string) (map[string]*tensor.Tensor, error) {
    return map[string]*tensor.Tensor{}, nil
}

func TestEngineNoLoader(t *testing.T) {
    cfg := inference.DefaultConfig
    cfg.NumHiddenLayers = 1
    e := inference.NewEngine(&cfg, nil)

    _, err := e.Generate(context.Background(), []int{1, 2, 3}, 5, nil)
    if err == nil {
        t.Fatal("expected error with nil loader")
    }
}

func TestEngineEmptyPrompt(t *testing.T) {
    cfg := inference.DefaultConfig
    cfg.NumHiddenLayers = 1
    loader := &stubLoader{layerCount: 1}
    e := inference.NewEngine(&cfg, loader)

    _, err := e.Generate(context.Background(), []int{}, 5, nil)
    if err == nil {
        t.Fatal("expected error with empty prompt")
    }
}

func TestEngineCancellation(t *testing.T) {
    cfg := inference.DefaultConfig
    cfg.NumHiddenLayers = 2
    loader := &stubLoader{layerCount: 2}
    e := inference.NewEngine(&cfg, loader)

    ctx, cancel := context.WithCancel(context.Background())
    cancel() // cancel immediately

    _, err := e.Generate(ctx, []int{1}, 100, nil)
    if err == nil {
        t.Fatal("expected context cancellation error")
    }
}
