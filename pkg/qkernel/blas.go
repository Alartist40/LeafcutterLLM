// Package qkernel provides a CGO wrapper around OpenBLAS (e.g., Intel MKL).
//
// This bridges the gap between Go's memory model and hardware-accelerated math.
// Python's PyTorch uses MKL/oneMKL for Linear layers. We use OpenBLAS `cblas_sgemm` to match that behavior.
package qkernel

/*
#cgo CFLAGS: -O3 -march=native
#cgo pkg-config: openblas
#include <cblas.h>
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"unsafe"

	"github.com/Alartist40/LeafcutterLLM/pkg/tensor"
)

// SGEMM performs C = alpha * A * B^T + beta * C using hardware-accelerated BLAS.
//
// Args:
//   A: Input tensor [M, K] or [B, S, K].
//   B: Weight tensor [N, K].
//   alpha: Scaling factor for the product.
//   beta: Scaling factor for the existing output (accumulation).
//
// Architecture Notes:
//   Go slices are row-major. OpenBLAS cblas_sgemm supports CblasRowMajor.
//   We treat B as [N, K] and use CblasTrans to effectively multiply by [K, N].
func SGEMM(A, B *tensor.Tensor, alpha, beta float32) (*tensor.Tensor, error) {
	if A == nil || B == nil {
		return nil, errors.New("A and B tensors cannot be nil")
	}

	if A.DType != tensor.Float32 || B.DType != tensor.Float32 {
		return matmulNaive(A, B, alpha, beta)
	}

	// Calculate M (rows), K (inner), N (output cols)
	M := 1
	for i := 0; i < len(A.Shape)-1; i++ {
		M *= A.Shape[i]
	}
	K := A.Shape[len(A.Shape)-1]

	if len(B.Shape) != 2 {
		return nil, fmt.Errorf("SGEMM requires 2D weight tensor B, got %v", B.Shape)
	}

	N := B.Shape[0]
	KB := B.Shape[1]

	if K != KB {
		return nil, fmt.Errorf("SGEMM dimension mismatch: A.K=%d, B.K=%d", K, KB)
	}

	outShape := make([]int, len(A.Shape))
	copy(outShape, A.Shape)
	outShape[len(outShape)-1] = N

	out := tensor.NewTensor(outShape, tensor.Float32)

	aData, okA := A.Data.([]float32)
	bData, okB := B.Data.([]float32)
	cData, okC := out.Data.([]float32)

	if !okA || !okB || !okC || len(aData) == 0 || len(bData) == 0 || len(cData) == 0 {
		return out, nil
	}

	// Perform hardware-accelerated matrix multiplication
	C.cblas_sgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasTrans,
		C.int(M),
		C.int(N),
		C.int(K),
		C.float(alpha),
		(*C.float)(unsafe.Pointer(&aData[0])),
		C.int(K),
		(*C.float)(unsafe.Pointer(&bData[0])),
		C.int(K),
		C.float(beta),
		(*C.float)(unsafe.Pointer(&cData[0])),
		C.int(N),
	)

	// Prevent GC from reclaiming buffers while C code is running
	runtime.KeepAlive(aData)
	runtime.KeepAlive(bData)
	runtime.KeepAlive(cData)

	return out, nil
}

// matmulNaive is a pure-Go fallback; converts non-Float32 inputs first. (FIX-021)
func matmulNaive(A, B *tensor.Tensor, alpha, beta float32) (*tensor.Tensor, error) {
	if A.DType != tensor.Float32 {
		A = A.ToFloat32()
	}
	if B.DType != tensor.Float32 {
		B = B.ToFloat32()
	}
	M := 1
	for i := 0; i < len(A.Shape)-1; i++ {
		M *= A.Shape[i]
	}
	K := A.Shape[len(A.Shape)-1]
	N := B.Shape[0]
	outShape := make([]int, len(A.Shape))
	copy(outShape, A.Shape)
	outShape[len(outShape)-1] = N
	out := tensor.NewTensor(outShape, tensor.Float32)
	aData := A.Data.([]float32)
	bData := B.Data.([]float32)
	cData := out.Data.([]float32)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var dot float32
			for k := 0; k < K; k++ {
				dot += aData[i*K+k] * bData[j*K+k]
			}
			cData[i*N+j] = alpha*dot + beta*cData[i*N+j]
		}
	}
	return out, nil
}

// Contains implements a simple string containment check for layer naming.
func Contains(s, substr string) bool {
	return strings.Contains(s, substr)
}
