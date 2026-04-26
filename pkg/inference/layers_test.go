package inference_test

import (
    "testing"
    "github.com/Alartist40/LeafcutterLLM/pkg/inference"
    "github.com/Alartist40/LeafcutterLLM/pkg/tensor"
)

// makeF32Tensor creates a Float32 tensor filled with sequential values.
func makeF32Tensor(shape []int) *tensor.Tensor {
    t := tensor.NewTensor(shape, tensor.Float32)
    for i := 0; i < t.Size(); i++ {
        t.SetFloat32(i, float32(i+1))
    }
    return t
}

func TestLinearLayerForward(t *testing.T) {
    cfg := &inference.Config{HiddenSize: 4, NumHeads: 1}
    layer := inference.NewLinearLayer("test.linear", cfg)

    // weight: [2, 4] (outFeatures=2, inFeatures=4)
    w := tensor.NewTensor([]int{2, 4}, tensor.Float32)
    // Set identity-like weights for predictable output
    w.SetFloat32(0, 1); w.SetFloat32(5, 1) // diagonal
    state := map[string]*tensor.Tensor{"test.linear.weight": w}
    if err := layer.Load(state); err != nil {
        t.Fatalf("Load failed: %v", err)
    }

    // input: [1, 1, 4]
    input := tensor.NewTensor([]int{1, 1, 4}, tensor.Float32)
    for i := 0; i < 4; i++ {
        input.SetFloat32(i, float32(i+1))
    }

    out, newK, newV, err := layer.Forward(input, nil, nil)
    if err != nil {
        t.Fatalf("Forward error: %v", err)
    }
    if newK != nil || newV != nil {
        t.Error("LinearLayer should return nil KV")
    }
    if out == nil {
        t.Fatal("output is nil")
    }
    if out.Shape[len(out.Shape)-1] != 2 {
        t.Fatalf("expected output last dim 2, got %d", out.Shape[len(out.Shape)-1])
    }
}

func TestLayerNormForward(t *testing.T) {
    cfg := &inference.Config{HiddenSize: 4}
    norm := inference.NewLayerNorm("test.norm", cfg)

    // All-ones weight
    w := tensor.NewTensor([]int{4}, tensor.Float32)
    for i := 0; i < 4; i++ {
        w.SetFloat32(i, 1.0)
    }
    state := map[string]*tensor.Tensor{"test.norm.weight": w}
    if err := norm.Load(state); err != nil {
        t.Fatalf("Load failed: %v", err)
    }

    // Input [1, 1, 4]: values [1, 2, 3, 4]
    input := makeF32Tensor([]int{1, 1, 4})

    out, _, _, err := norm.Forward(input, nil, nil)
    if err != nil {
        t.Fatalf("Forward error: %v", err)
    }
    if out == nil {
        t.Fatal("output is nil")
    }
    // After RMS norm, mean of squares should be ~1
    if out.Size() != 4 {
        t.Fatalf("expected size 4, got %d", out.Size())
    }
}

func TestLayerNormNilWeight(t *testing.T) {
    cfg := &inference.Config{HiddenSize: 4}
    norm := inference.NewLayerNorm("test.norm2", cfg)
    // Load nothing (nil weight)
    norm.Load(map[string]*tensor.Tensor{})

    input := makeF32Tensor([]int{1, 1, 4})
    // Should NOT panic with nil weight
    _, _, _, err := norm.Forward(input, nil, nil)
    if err != nil {
        // Acceptable to return an error, but must not panic
        t.Logf("LayerNorm with nil weight returned error (acceptable): %v", err)
    }
}

func TestEmbeddingLayerForward(t *testing.T) {
    emb := inference.NewEmbeddingLayer("model.embed_tokens")
    // vocab_size=10, hidden=4
    w := tensor.NewTensor([]int{10, 4}, tensor.Float32)
    for i := 0; i < 40; i++ {
        w.SetFloat32(i, float32(i))
    }
    state := map[string]*tensor.Tensor{"model.embed_tokens.weight": w}
    if err := emb.Load(state); err != nil {
        t.Fatalf("Load failed: %v", err)
    }

    // Token IDs tensor: [1, 2] = tokens [3, 5]
    ids := &tensor.Tensor{
        Shape: []int{1, 2},
        Data:  []int64{3, 5},
        DType: tensor.Int64,
    }
    out, _, _, err := emb.Forward(ids, nil, nil)
    if err != nil {
        t.Fatalf("Forward error: %v", err)
    }
    if out.Shape[0] != 1 || out.Shape[1] != 2 || out.Shape[2] != 4 {
        t.Fatalf("unexpected shape: %v", out.Shape)
    }
    // Token 3 row: elements 12,13,14,15
    if out.GetFloat32(0) != 12.0 {
        t.Fatalf("expected embedding[3][0]=12, got %f", out.GetFloat32(0))
    }
}
