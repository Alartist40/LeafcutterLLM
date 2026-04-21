package qkernel

/*
#cgo pkg-config: openblas
#include <cblas.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/xander/airllm-go/pkg/tensor"
)

// SGEMM performs C = alpha * A * B^T + beta * C using OpenBLAS.
// A is [M x K], B is [N x K] in row-major memory. Output is [M x N].
func SGEMM(A, B *tensor.Tensor, alpha, beta float32) (*tensor.Tensor, error) {
	if A.Dtype != tensor.Float32 || B.Dtype != tensor.Float32 {
		return nil, errors.New("SGEMM requires Float32 tensors")
	}

	M := 1
	for i := 0; i < len(A.Shape)-1; i++ {
		M *= A.Shape[i]
	}
	K := A.Shape[len(A.Shape)-1]

	if len(B.Shape) != 2 {
		return nil, errors.New("SGEMM requires 2D weight tensor B")
	}

	N := B.Shape[0]
	KB := B.Shape[1]

	if K != KB {
		return nil, fmt.Errorf("SGEMM dimension mismatch: K=%d, KB=%d", K, KB)
	}

	outShape := make([]int, len(A.Shape))
	copy(outShape, A.Shape)
	outShape[len(outShape)-1] = N

	out := tensor.NewTensor(outShape, tensor.Float32)

	aData := A.Data.([]float32)
	bData := B.Data.([]float32)
	cData := out.Data.([]float32)

	if len(aData) > 0 && len(bData) > 0 {
		C.cblas_sgemm(
			C.CblasRowMajor,
			C.CblasNoTrans,
			C.CblasTrans, // B^T
			C.int(M),
			C.int(N),
			C.int(K),
			C.float(alpha),
			(*C.float)(unsafe.Pointer(&aData[0])),
			C.int(K),
			(*C.float)(unsafe.Pointer(&bData[0])),
			C.int(K), // lda of B when row-major is K
			C.float(beta),
			(*C.float)(unsafe.Pointer(&cData[0])),
			C.int(N),
		)
	}

	runtime.KeepAlive(aData)
	runtime.KeepAlive(bData)
	runtime.KeepAlive(cData)

	return out, nil
}
