// Package compression provides model compression techniques
package compression

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/xander/airllm-go/pkg/tensor"
)

// QuantType represents quantization types
type QuantType int

const (
	QuantNone QuantType = iota
	Quant8Bit
	Quant4Bit
	Quant3Bit
	Quant2Bit
)

// QuantizedTensor holds quantized tensor data
type QuantizedTensor struct {
	Data      []byte    // Quantized values
	Scales    []float32 // Per-block scales
	Zeros     []float32 // Per-block zeros (for symmetric, these are 0)
	Shape     []int     // Original shape
	BlockSize int       // Size of quantization blocks
	QuantType QuantType
	NumBits   int
}

// Quantize8Bit performs block-wise 8-bit quantization
func Quantize8Bit(t *tensor.Tensor, blockSize int) (*QuantizedTensor, error) {
	if blockSize <= 0 {
		blockSize = 256 // Default block size
	}

	// Convert to float32 for quantization
	tF32 := t.ToFloat32()
	size := tF32.Size()

	// Calculate number of blocks
	numBlocks := (size + blockSize - 1) / blockSize

	qt := &QuantizedTensor{
		Data:      make([]byte, size),
		Scales:    make([]float32, numBlocks),
		Shape:     make([]int, len(t.Shape)),
		BlockSize: blockSize,
		QuantType: Quant8Bit,
		NumBits:   8,
	}
	copy(qt.Shape, t.Shape)

	// Quantize block by block
	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > size {
			end = size
		}
		blockLen := end - start

		// Find min/max in block
		minVal := float32(math.MaxFloat32)
		maxVal := float32(-math.MaxFloat32)
		for i := start; i < end; i++ {
			val := tF32.GetFloat32(i)
			if val < minVal {
				minVal = val
			}
			if val > maxVal {
				maxVal = val
			}
		}

		// Calculate scale and zero point
		scale := (maxVal - minVal) / 255.0
		if scale == 0 {
			scale = 1.0 // Avoid division by zero
		}
		zeroPoint := minVal
		qt.Scales[b] = scale
		qt.Zeros = append(qt.Zeros, zeroPoint)

		// Quantize
		for i := 0; i < blockLen; i++ {
			val := tF32.GetFloat32(start + i)
			quantized := uint8((val - zeroPoint) / scale)
			qt.Data[start+i] = quantized
		}
	}

	return qt, nil
}

// Quantize4Bit performs block-wise 4-bit quantization (NF4-like)
func Quantize4Bit(t *tensor.Tensor, blockSize int) (*QuantizedTensor, error) {
	if blockSize <= 0 {
		blockSize = 64
	}

	tF32 := t.ToFloat32()
	size := tF32.Size()

	// For 4-bit, two values pack into one byte
	dataSize := (size + 1) / 2
	numBlocks := (size + blockSize - 1) / blockSize

	qt := &QuantizedTensor{
		Data:      make([]byte, dataSize),
		Scales:    make([]float32, numBlocks),
		Shape:     make([]int, len(t.Shape)),
		BlockSize: blockSize,
		QuantType: Quant4Bit,
		NumBits:   4,
	}
	copy(qt.Shape, t.Shape)

	// NF4 quantization levels (normalized float 4-bit)
	// These approximate a normal distribution
	nf4Values := []float32{-1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414,
		-0.1847734, -0.09105004, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.3379152,
		0.4407098, 0.562617, 0.7229568, 1.0}

	// Quantize block by block
	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > size {
			end = size
		}

		// Find block scale
		maxAbs := float32(0)
		for i := start; i < end; i++ {
			absVal := float32(math.Abs(float64(tF32.GetFloat32(i))))
			if absVal > maxAbs {
				maxAbs = absVal
			}
		}
		qt.Scales[b] = maxAbs

		// Quantize each value
		for i := start; i < end; i++ {
			val := tF32.GetFloat32(i) / maxAbs

			// Find closest NF4 level
			minDiff := float32(math.MaxFloat32)
			bestIdx := 0
			for j, nf4 := range nf4Values {
				diff := float32(math.Abs(float64(val - nf4)))
				if diff < minDiff {
					minDiff = diff
					bestIdx = j
				}
			}

			// Pack into data
			byteIdx := (i - start) / 2
			isHighNibble := (i-start)%2 == 1

			if isHighNibble {
				qt.Data[byteIdx] |= byte(bestIdx << 4)
			} else {
				qt.Data[byteIdx] = byte(bestIdx)
			}
		}
	}

	return qt, nil
}

// Dequantize converts a quantized tensor back to float32
func (qt *QuantizedTensor) Dequantize() (*tensor.Tensor, error) {
	switch qt.QuantType {
	case Quant8Bit:
		return qt.dequantize8Bit()
	case Quant4Bit:
		return qt.dequantize4Bit()
	default:
		return nil, fmt.Errorf("unsupported quantization type")
	}
}

func (qt *QuantizedTensor) dequantize8Bit() (*tensor.Tensor, error) {
	// Calculate total size from shape
	size := 1
	for _, dim := range qt.Shape {
		size *= dim
	}

	result := tensor.NewTensor(qt.Shape, tensor.Float32)

	// Dequantize block by block
	for i := 0; i < size; i++ {
		blockIdx := i / qt.BlockSize
		if blockIdx >= len(qt.Scales) {
			blockIdx = len(qt.Scales) - 1
		}

		scale := qt.Scales[blockIdx]
		zero := float32(0)
		if len(qt.Zeros) > blockIdx {
			zero = qt.Zeros[blockIdx]
		}

		quantized := float32(qt.Data[i])
		val := quantized*scale + zero
		result.SetFloat32(i, val)
	}

	return result, nil
}

func (qt *QuantizedTensor) dequantize4Bit() (*tensor.Tensor, error) {
	// Calculate total size
	size := 1
	for _, dim := range qt.Shape {
		size *= dim
	}

	result := tensor.NewTensor(qt.Shape, tensor.Float32)

	nf4Values := []float32{-1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414,
		-0.1847734, -0.09105004, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.3379152,
		0.4407098, 0.562617, 0.7229568, 1.0}

	// Dequantize block by block
	for i := 0; i < size; i++ {
		blockIdx := i / qt.BlockSize
		if blockIdx >= len(qt.Scales) {
			blockIdx = len(qt.Scales) - 1
		}
		scale := qt.Scales[blockIdx]

		// Extract nibble
		byteIdx := i / 2
		isHighNibble := i%2 == 1

		var idx int
		if isHighNibble {
			idx = int(qt.Data[byteIdx] >> 4)
		} else {
			idx = int(qt.Data[byteIdx] & 0x0F)
		}

		val := nf4Values[idx] * scale
		result.SetFloat32(i, val)
	}

	return result, nil
}

// CompressedLayer represents a layer with quantized weights
type CompressedLayer struct {
	Name            string
	QuantizedWeights map[string]*QuantizedTensor
	OriginalShape    map[string][]int
}

// CompressLayer compresses a layer's weights
func CompressLayer(name string, state map[string]*tensor.Tensor, quantType QuantType, blockSize int) (*CompressedLayer, error) {
	cl := &CompressedLayer{
		Name:             name,
		QuantizedWeights: make(map[string]*QuantizedTensor),
		OriginalShape:    make(map[string][]int),
	}

	for key, tensor := range state {
		// Only quantize weight tensors
		if !hasSuffix(key, ".weight") {
			continue
		}

		cl.OriginalShape[key] = make([]int, len(tensor.Shape))
		copy(cl.OriginalShape[key], tensor.Shape)

		var qt *QuantizedTensor
		var err error

		switch quantType {
		case Quant8Bit:
			qt, err = Quantize8Bit(tensor, blockSize)
		case Quant4Bit:
			qt, err = Quantize4Bit(tensor, blockSize)
		default:
			continue // Skip unknown quantization
		}

		if err != nil {
			return nil, fmt.Errorf("failed to quantize %s: %w", key, err)
		}

		cl.QuantizedWeights[key] = qt
	}

	return cl, nil
}

// DecompressLayer decompresses a layer
func (cl *CompressedLayer) Decompress() (map[string]*tensor.Tensor, error) {
	state := make(map[string]*tensor.Tensor)

	for key, qt := range cl.QuantizedWeights {
		tensor, err := qt.Dequantize()
		if err != nil {
			return nil, fmt.Errorf("failed to dequantize %s: %w", key, err)
		}
		state[key] = tensor
	}

	return state, nil
}

// Serialize serializes a quantized tensor to bytes
func (qt *QuantizedTensor) Serialize() ([]byte, error) {
	// Simple binary format:
	// [QuantType:1][NumBits:1][BlockSize:4][NumDims:1][Dims...][NumBlocks:4][Scales...][Zeros...][DataLen:4][Data...]

	buf := make([]byte, 0, 1024+len(qt.Data))

	// Header
	buf = append(buf, byte(qt.QuantType))
	buf = append(buf, byte(qt.NumBits))
	buf = appendBinaryUint32(buf, uint32(qt.BlockSize))
	buf = append(buf, byte(len(qt.Shape)))
	for _, dim := range qt.Shape {
		buf = appendBinaryUint32(buf, uint32(dim))
	}

	// Scales
	buf = appendBinaryUint32(buf, uint32(len(qt.Scales)))
	for _, scale := range qt.Scales {
		buf = appendBinaryFloat32(buf, scale)
	}

	// Zeros
	buf = appendBinaryUint32(buf, uint32(len(qt.Zeros)))
	for _, zero := range qt.Zeros {
		buf = appendBinaryFloat32(buf, zero)
	}

	// Data
	buf = appendBinaryUint32(buf, uint32(len(qt.Data)))
	buf = append(buf, qt.Data...)

	return buf, nil
}

// DeserializeQuantizedTensor creates a QuantizedTensor from serialized bytes
func DeserializeQuantizedTensor(data []byte) (*QuantizedTensor, error) {
	if len(data) < 6 {
		return nil, fmt.Errorf("data too short")
	}

	offset := 0
	qt := &QuantizedTensor{}

	qt.QuantType = QuantType(data[offset])
	offset++
	qt.NumBits = int(data[offset])
	offset++
	qt.BlockSize = int(readBinaryUint32(data, offset))
	offset += 4

	numDims := int(data[offset])
	offset++

	qt.Shape = make([]int, numDims)
	for i := 0; i < numDims; i++ {
		qt.Shape[i] = int(readBinaryUint32(data, offset))
		offset += 4
	}

	numBlocks := int(readBinaryUint32(data, offset))
	offset += 4

	qt.Scales = make([]float32, numBlocks)
	for i := 0; i < numBlocks; i++ {
		qt.Scales[i] = readBinaryFloat32(data, offset)
		offset += 4
	}

	numZeros := int(readBinaryUint32(data, offset))
	offset += 4

	qt.Zeros = make([]float32, numZeros)
	for i := 0; i < numZeros; i++ {
		qt.Zeros[i] = readBinaryFloat32(data, offset)
		offset += 4
	}

	dataLen := int(readBinaryUint32(data, offset))
	offset += 4

	if offset+dataLen > len(data) {
		return nil, fmt.Errorf("data length mismatch")
	}

	qt.Data = make([]byte, dataLen)
	copy(qt.Data, data[offset:offset+dataLen])

	return qt, nil
}

// Helper functions for binary serialization
func appendBinaryUint32(buf []byte, val uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, val)
	return append(buf, b...)
}

func readBinaryUint32(data []byte, offset int) uint32 {
	return binary.LittleEndian.Uint32(data[offset : offset+4])
}

func appendBinaryFloat32(buf []byte, val float32) []byte {
	return appendBinaryUint32(buf, math.Float32bits(val))
}

func readBinaryFloat32(data []byte, offset int) float32 {
	return math.Float32frombits(readBinaryUint32(data, offset))
}

// hasSuffix checks if string s ends with suffix suffix
func hasSuffix(s, suffix string) bool {
	if len(s) < len(suffix) {
		return false
	}
	return s[len(s)-len(suffix):] == suffix
}
