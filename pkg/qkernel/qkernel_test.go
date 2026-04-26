//go:build cgo

package qkernel_test

import (
    "math"
    "testing"
    "github.com/Alartist40/LeafcutterLLM/pkg/qkernel"
    "github.com/Alartist40/LeafcutterLLM/pkg/tensor"
)

func TestSGEMMIdentity(t *testing.T) {
    // A = [[1,0],[0,1]] (identity 2x2)
    // B = [[1,0],[0,1]] (identity 2x2, stored as [N=2, K=2])
    // Result should be identity

    A := tensor.NewTensor([]int{2, 2}, tensor.Float32)
    A.SetFloat32(0, 1); A.SetFloat32(3, 1)

    B := tensor.NewTensor([]int{2, 2}, tensor.Float32)
    B.SetFloat32(0, 1); B.SetFloat32(3, 1)

    out, err := qkernel.SGEMM(A, B, 1.0, 0.0)
    if err != nil {
        t.Fatalf("SGEMM error: %v", err)
    }
    if out.GetFloat32(0) != 1.0 || out.GetFloat32(3) != 1.0 {
        t.Fatalf("identity × identity ≠ identity: %v", []float32{
            out.GetFloat32(0), out.GetFloat32(1),
            out.GetFloat32(2), out.GetFloat32(3),
        })
    }
}

func TestSGEMMKnownResult(t *testing.T) {
    // A = [1, 2, 3, 4] shaped [2, 2]
    // B = [1, 0, 0, 1] shaped [2, 2] (identity)
    // A @ B^T = A @ I = A
    A := tensor.NewTensor([]int{2, 2}, tensor.Float32)
    A.SetFloat32(0, 1); A.SetFloat32(1, 2)
    A.SetFloat32(2, 3); A.SetFloat32(3, 4)

    B := tensor.NewTensor([]int{2, 2}, tensor.Float32)
    B.SetFloat32(0, 1); B.SetFloat32(3, 1)

    out, err := qkernel.SGEMM(A, B, 1.0, 0.0)
    if err != nil {
        t.Fatalf("SGEMM error: %v", err)
    }
    expected := []float32{1, 2, 3, 4}
    for i, e := range expected {
        if math.Abs(float64(out.GetFloat32(i)-e)) > 1e-4 {
            t.Fatalf("index %d: expected %f got %f", i, e, out.GetFloat32(i))
        }
    }
}
