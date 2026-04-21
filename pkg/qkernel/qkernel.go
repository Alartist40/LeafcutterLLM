// Package qkernel provides a CGO wrapper around the 4-bit packed GEMM C kernel.
//
// Safety protocol enforced here:
//   1. All slice capacities are validated before passing to C.
//   2. runtime.KeepAlive is called on every Go object after the C call so the
//      GC cannot free memory while C is still reading it.
//   3. The C kernel itself NULL-guards all pointers.
package qkernel

/*
#cgo CFLAGS: -O3 -march=native -ffast-math
#include "qkernel.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// GroupSize is the number of weights per quantization group (must match qkernel.h).
const GroupSize = C.QKERNEL_GROUP_SIZE

// Q4Tensor holds a 4-bit quantized weight matrix.
// Shape: [OutFeatures, InFeatures] — matches HuggingFace [out, in] convention.
type Q4Tensor struct {
	// Packed holds two 4-bit weights per byte.
	// Layout: [OutFeatures][InFeatures/2] bytes.
	Packed []uint8

	// Scales holds one float32 per group per output row.
	// Layout: [OutFeatures][InFeatures/GroupSize] float32.
	Scales []float32

	// Zeros holds the zero-point per group per output row (optional).
	// If nil the kernel defaults to 8.0 (symmetric NF4 midpoint).
	Zeros []float32

	OutFeatures int
	InFeatures  int
	GrpSize     int
}

// NewQ4Tensor allocates a Q4Tensor for the given weight shape.
func NewQ4Tensor(outFeatures, inFeatures, groupSize int) (*Q4Tensor, error) {
	if outFeatures <= 0 || inFeatures <= 0 || groupSize <= 0 {
		return nil, fmt.Errorf("qkernel: invalid shape [%d, %d] groupSize=%d",
			outFeatures, inFeatures, groupSize)
	}
	numGroups := (inFeatures + groupSize - 1) / groupSize
	packedLen := outFeatures * ((inFeatures + 1) / 2)
	return &Q4Tensor{
		Packed:      make([]uint8, packedLen),
		Scales:      make([]float32, outFeatures*numGroups),
		Zeros:       make([]float32, outFeatures*numGroups),
		OutFeatures: outFeatures,
		InFeatures:  inFeatures,
		GrpSize:     groupSize,
	}, nil
}

// GEMM performs: C[M, N] = A[M, K] × dequant(weight)
//
// A must be a float32 slice of length M*K.
// Returns a newly allocated float32 slice of length M*N.
//
// Safety: validates all slice bounds before calling C.
func (w *Q4Tensor) GEMM(A []float32, M, K int) ([]float32, error) {
	N := w.OutFeatures

	// ── Bounds validation ────────────────────────────────────────────
	requiredA := M * K
	if len(A) < requiredA {
		return nil, fmt.Errorf("qkernel.GEMM: A too short: have %d, need %d", len(A), requiredA)
	}
	if K != w.InFeatures {
		return nil, fmt.Errorf("qkernel.GEMM: K mismatch: A has K=%d, weight has InFeatures=%d", K, w.InFeatures)
	}

	numGroups := (K + w.GrpSize - 1) / w.GrpSize
	requiredPacked := N * ((K + 1) / 2)
	requiredScales := N * numGroups

	if len(w.Packed) < requiredPacked {
		return nil, fmt.Errorf("qkernel.GEMM: Packed too short: have %d, need %d", len(w.Packed), requiredPacked)
	}
	if len(w.Scales) < requiredScales {
		return nil, fmt.Errorf("qkernel.GEMM: Scales too short: have %d, need %d", len(w.Scales), requiredScales)
	}

	// ── Output allocation ────────────────────────────────────────────
	C_out := make([]float32, M*N)

	// ── C call ──────────────────────────────────────────────────────
	var zerosPtr *C.float
	if len(w.Zeros) >= requiredScales {
		zerosPtr = (*C.float)(unsafe.Pointer(&w.Zeros[0]))
	}

	C.q4_gemm(
		(*C.float)(unsafe.Pointer(&C_out[0])),
		(*C.float)(unsafe.Pointer(&A[0])),
		(*C.uint8_t)(unsafe.Pointer(&w.Packed[0])),
		(*C.float)(unsafe.Pointer(&w.Scales[0])),
		zerosPtr,
		C.int(M), C.int(N), C.int(K),
		C.int(w.GrpSize),
	)

	// ── Safety: keep Go objects alive past the C call ────────────────
	runtime.KeepAlive(A)
	runtime.KeepAlive(w.Packed)
	runtime.KeepAlive(w.Scales)
	runtime.KeepAlive(w.Zeros)
	runtime.KeepAlive(C_out)

	return C_out, nil
}

// GEMMBatched is a convenience wrapper for [batch, seq_len, K] × weight.
// Returns a float32 slice shaped [batch*seq_len, N].
func (w *Q4Tensor) GEMMBatched(A []float32, batch, seqLen, K int) ([]float32, error) {
	return w.GEMM(A, batch*seqLen, K)
}

// QuantizeFromF32 packs a float32 weight matrix into Q4 format.
// weights must be laid out [OutFeatures, InFeatures] in row-major order.
// This is a reference (slow) implementation; use bitsandbytes offline for production.
func (w *Q4Tensor) QuantizeFromF32(weights []float32) error {
	if len(weights) < w.OutFeatures*w.InFeatures {
		return fmt.Errorf("qkernel: weights slice too short")
	}
	numGroups := (w.InFeatures + w.GrpSize - 1) / w.GrpSize

	for n := 0; n < w.OutFeatures; n++ {
		for g := 0; g < numGroups; g++ {
			kStart := g * w.GrpSize
			kEnd := kStart + w.GrpSize
			if kEnd > w.InFeatures {
				kEnd = w.InFeatures
			}

			// Find min/max in this group
			minVal := weights[n*w.InFeatures+kStart]
			maxVal := minVal
			for k := kStart; k < kEnd; k++ {
				v := weights[n*w.InFeatures+k]
				if v < minVal {
					minVal = v
				}
				if v > maxVal {
					maxVal = v
				}
			}

			rangeVal := maxVal - minVal
			scale := float32(1.0)
			if rangeVal > 1e-8 {
				scale = rangeVal / 15.0 // 4-bit = 0..15
			}
			zero := -minVal / scale

			w.Scales[n*numGroups+g] = scale
			w.Zeros[n*numGroups+g] = zero

			// Pack nibbles
			for k := kStart; k < kEnd; k++ {
				quant := int((weights[n*w.InFeatures+k]/scale)+zero+0.5)
				if quant < 0 {
					quant = 0
				}
				if quant > 15 {
					quant = 15
				}
				byteIdx := n*((w.InFeatures+1)/2) + k/2
				if k%2 == 0 {
					w.Packed[byteIdx] = (w.Packed[byteIdx] & 0xF0) | uint8(quant)
				} else {
					w.Packed[byteIdx] = (w.Packed[byteIdx] & 0x0F) | (uint8(quant) << 4)
				}
			}
		}
	}
	return nil
}
