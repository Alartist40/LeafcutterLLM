package tensor_test

import (
	"math"
	"testing"

	"github.com/Alartist40/LeafcutterLLM/pkg/tensor"
)

func TestNewTensor(t *testing.T) {
	ten := tensor.NewTensor([]int{2, 3}, tensor.Float32)
	if ten == nil {
		t.Fatal("NewTensor returned nil")
	}
	if ten.Size() != 6 {
		t.Fatalf("expected size 6, got %d", ten.Size())
	}
	data, ok := ten.Data.([]float32)
	if !ok || len(data) != 6 {
		t.Fatalf("expected []float32 of len 6, got %T len %d", ten.Data, len(data))
	}
}

func TestNewTensorInvalidDim(t *testing.T) {
	if tensor.NewTensor([]int{0, 3}, tensor.Float32) != nil {
		t.Fatal("expected nil for zero dimension")
	}
	if tensor.NewTensor([]int{-1}, tensor.Float32) != nil {
		t.Fatal("expected nil for negative dimension")
	}
}

func TestGetSetFloat32(t *testing.T) {
	ten := tensor.NewTensor([]int{4}, tensor.Float32)
	ten.SetFloat32(2, 3.14)
	if got := ten.GetFloat32(2); math.Abs(float64(got-3.14)) > 1e-5 {
		t.Fatalf("expected 3.14, got %f", got)
	}
}

func TestClone(t *testing.T) {
	orig := tensor.NewTensor([]int{3}, tensor.Float32)
	orig.SetFloat32(0, 1.0)
	orig.SetFloat32(1, 2.0)
	orig.SetFloat32(2, 3.0)
	clone := orig.Clone()
	clone.SetFloat32(0, 99.0)
	if orig.GetFloat32(0) != 1.0 {
		t.Fatal("Clone modified original — not a deep copy")
	}
}

func TestTranspose2D(t *testing.T) {
	// 2x3 matrix: [[1,2,3],[4,5,6]]
	m := tensor.NewTensor([]int{2, 3}, tensor.Float32)
	for i := 0; i < 6; i++ {
		m.SetFloat32(i, float32(i+1))
	}
	mt, err := m.Transpose(0, 1)
	if err != nil {
		t.Fatalf("Transpose error: %v", err)
	}
	if mt.Shape[0] != 3 || mt.Shape[1] != 2 {
		t.Fatalf("expected shape [3,2], got %v", mt.Shape)
	}
	// mt[0][1] should be original m[1][0] = 4
	if mt.GetFloat32(1) != 4.0 { // flat index 1 = row 0 col 1 in [3,2]
		t.Fatalf("wrong transposed value: %f", mt.GetFloat32(1))
	}
}

func TestTranspose4D(t *testing.T) {
	// [1, 2, 3, 4] tensor — transpose dims 1 and 2: [1,3,2,4]
	m := tensor.NewTensor([]int{1, 2, 3, 4}, tensor.Float32)
	for i := 0; i < m.Size(); i++ {
		m.SetFloat32(i, float32(i))
	}
	mt, err := m.Transpose(1, 2)
	if err != nil {
		t.Fatalf("Transpose4D error: %v", err)
	}
	if mt.Shape[0] != 1 || mt.Shape[1] != 3 || mt.Shape[2] != 2 || mt.Shape[3] != 4 {
		t.Fatalf("wrong output shape: %v", mt.Shape)
	}
	if mt.Size() != m.Size() {
		t.Fatalf("size changed after transpose: %d vs %d", mt.Size(), m.Size())
	}
}

func TestToFloat32FromFloat16(t *testing.T) {
	ten := tensor.NewTensor([]int{2}, tensor.Float16)
	data := ten.Data.([]uint16)
	data[0] = 0x3C00 // float16 representation of 1.0
	data[1] = 0x0000 // float16 representation of 0.0

	f32 := ten.ToFloat32()
	if f32.DType != tensor.Float32 {
		t.Fatalf("expected Float32, got %v", f32.DType)
	}
	if math.Abs(float64(f32.GetFloat32(0)-1.0)) > 1e-4 {
		t.Fatalf("expected 1.0, got %f", f32.GetFloat32(0))
	}
	if f32.GetFloat32(1) != 0.0 {
		t.Fatalf("expected 0.0, got %f", f32.GetFloat32(1))
	}
}
