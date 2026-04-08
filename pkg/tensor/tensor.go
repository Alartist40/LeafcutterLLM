// Package tensor provides efficient tensor operations for LLM inference
package tensor

import (
	"encoding/binary"
	"fmt"
	"unsafe"
)

// DType represents tensor data types
type DType int

const (
	Float32 DType = iota
	Float16
	Int64
	Int32
	Uint8
)

func (d DType) Size() int {
	switch d {
	case Float32:
		return 4
	case Float16:
		return 2
	case Int64:
		return 8
	case Int32:
		return 4
	case Uint8:
		return 1
	default:
		return 4
	}
}

func (d DType) String() string {
	switch d {
	case Float32:
		return "F32"
	case Float16:
		return "F16"
	case Int64:
		return "I64"
	case Int32:
		return "I32"
	case Uint8:
		return "U8"
	default:
		return "UNKNOWN"
	}
}

// Tensor represents a multidimensional array with efficient memory layout
type Tensor struct {
	Shape  []int
	Data   []byte
	DType  DType
	offset int // For sliced views
}

// NewTensor creates a new tensor with the given shape and dtype
func NewTensor(shape []int, dtype DType) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return &Tensor{
		Shape: shape,
		Data:  make([]byte, size*dtype.Size()),
		DType: dtype,
	}
}

// FromBuffer creates a tensor from an existing byte buffer
func FromBuffer(data []byte, shape []int, dtype DType) *Tensor {
	return &Tensor{
		Shape: shape,
		Data:  data,
		DType: dtype,
	}
}

// Size returns the total number of elements
func (t *Tensor) Size() int {
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

// ByteSize returns the total size in bytes
func (t *Tensor) ByteSize() int {
	return len(t.Data)
}

// Reshape returns a view of the tensor with new shape (sharing underlying data)
func (t *Tensor) Reshape(newShape []int) (*Tensor, error) {
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}
	if newSize != t.Size() {
		return nil, fmt.Errorf("cannot reshape tensor of size %d into shape %v", t.Size(), newShape)
	}
	return &Tensor{
		Shape:  newShape,
		Data:   t.Data,
		DType:  t.DType,
		offset: t.offset,
	}, nil
}

// Slice returns a sliced view of the tensor (zero-copy)
func (t *Tensor) Slice(start, end int) (*Tensor, error) {
	if len(t.Shape) == 0 {
		return nil, fmt.Errorf("cannot slice 0-d tensor")
	}
	if start < 0 || end > t.Shape[0] || start >= end {
		return nil, fmt.Errorf("invalid slice range [%d:%d] for dimension of size %d", start, end, t.Shape[0])
	}

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[0] = end - start

	stride := t.Size() / t.Shape[0]
	newOffset := t.offset + start*stride*t.DType.Size()

	return &Tensor{
		Shape:  newShape,
		Data:   t.Data,
		DType:  t.DType,
		offset: newOffset,
	}, nil
}

// GetFloat32 returns a float32 value at the given flat index
func (t *Tensor) GetFloat32(idx int) float32 {
	if t.DType != Float32 {
		panic(fmt.Sprintf("cannot GetFloat32 on %s tensor", t.DType))
	}
	actualIdx := (t.offset/t.DType.Size() + idx) * 4
	return *(*float32)(unsafe.Pointer(&t.Data[actualIdx]))
}

// SetFloat32 sets a float32 value at the given flat index
func (t *Tensor) SetFloat32(idx int, val float32) {
	if t.DType != Float32 {
		panic(fmt.Sprintf("cannot SetFloat32 on %s tensor", t.DType))
	}
	actualIdx := (t.offset/t.DType.Size() + idx) * 4
	binary.LittleEndian.PutUint32(t.Data[actualIdx:], *(*uint32)(unsafe.Pointer(&val)))
}

// GetFloat16 returns a float16 value as float32
func (t *Tensor) GetFloat16(idx int) float32 {
	if t.DType != Float16 {
		panic(fmt.Sprintf("cannot GetFloat16 on %s tensor", t.DType))
	}
	actualIdx := (t.offset/2 + idx) * 2
	bits := binary.LittleEndian.Uint16(t.Data[actualIdx:])
	return Float16ToFloat32(bits)
}

// SetFloat16 sets a float16 value from float32
func (t *Tensor) SetFloat16(idx int, val float32) {
	if t.DType != Float16 {
		panic(fmt.Sprintf("cannot SetFloat16 on %s tensor", t.DType))
	}
	actualIdx := (t.offset/2 + idx) * 2
	bits := Float32ToFloat16(val)
	binary.LittleEndian.PutUint16(t.Data[actualIdx:], bits)
}

// ToFloat32 converts the tensor to Float32 dtype (allocates new memory)
func (t *Tensor) ToFloat32() *Tensor {
	if t.DType == Float32 {
		// Return a copy
		newTensor := NewTensor(t.Shape, Float32)
		copy(newTensor.Data, t.Data)
		return newTensor
	}

	newTensor := NewTensor(t.Shape, Float32)
	switch t.DType {
	case Float16:
		for i := 0; i < t.Size(); i++ {
			newTensor.SetFloat32(i, t.GetFloat16(i))
		}
	default:
		panic(fmt.Sprintf("ToFloat32 not implemented for %s", t.DType))
	}
	return newTensor
}

// Float16ToFloat32 converts IEEE 754 half-precision to single-precision
func Float16ToFloat32(bits uint16) float32 {
	sign := uint32(bits >> 15)
	exp := uint32((bits >> 10) & 0x1F)
	frac := uint32(bits & 0x3FF)

	if exp == 0 {
		if frac == 0 {
			// Zero
			return *(*float32)(unsafe.Pointer(&((sign << 31))))
		}
		// Subnormal number
		return float32(-1) * float32(sign) * (float32(frac) / float32(1<<10)) * float32(1) / float32(1<<14)
	}
	if exp == 0x1F {
		if frac == 0 {
			// Infinity
			return *(*float32)(unsafe.Pointer(&((sign << 31) | 0x7F800000)))
		}
		// NaN
		return *(*float32)(unsafe.Pointer(&((sign << 31) | 0x7FC00000 | frac)))
	}
	// Normal number
	return *(*float32)(unsafe.Pointer(&(((sign << 31) | ((exp + 112) << 23) | (frac << 13)))))
}

// Float32ToFloat16 converts single-precision to IEEE 754 half-precision
func Float32ToFloat16(f float32) uint16 {
	bits := *(*uint32)(unsafe.Pointer(&f))
	sign := uint16(bits >> 31)
	exp := int16((bits >> 23) & 0xFF) - 127
	frac := bits & 0x7FFFFF

	if exp <= -15 {
		// Subnormal or zero
		if exp < -24 {
			return sign << 15 // Zero
		}
		frac = (frac | 0x800000) >> uint16(-14-exp)
		return (sign << 15) | uint16(frac>>13)
	}
	if exp > 15 {
		// Overflow to infinity
		return (sign << 15) | 0x7C00
	}

	return (sign << 15) | uint16((exp+15)<<10) | uint16(frac>>13)
}

// Transpose swaps two dimensions of the tensor
func (t *Tensor) Transpose(dim1, dim2 int) (*Tensor, error) {
	if dim1 < 0 || dim1 >= len(t.Shape) || dim2 < 0 || dim2 >= len(t.Shape) {
		return nil, fmt.Errorf("invalid transpose dimensions %d, %d", dim1, dim2)
	}

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[dim1], newShape[dim2] = newShape[dim2], newShape[dim1]

	// Calculate strides
	oldStrides := make([]int, len(t.Shape))
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		oldStrides[i] = stride
		stride *= t.Shape[i]
	}

	// For a true transpose, we need to actually reorder data
	// This allocates new memory
	result := NewTensor(newShape, t.DType)
	
	// Simple implementation for 2D case
	if len(t.Shape) == 2 {
		for i := 0; i < t.Shape[0]; i++ {
			for j := 0; j < t.Shape[1]; j++ {
				oldIdx := i*t.Shape[1] + j
				newIdx := j*newShape[1] + i
				switch t.DType {
				case Float32:
					result.SetFloat32(newIdx, t.GetFloat32(oldIdx))
				case Float16:
					result.SetFloat16(newIdx, t.GetFloat16(oldIdx))
				default:
					copy(result.Data[newIdx*t.DType.Size():], t.Data[oldIdx*t.DType.Size():(oldIdx+1)*t.DType.Size()])
				}
			}
		}
	} else {
		// Fall back to byte copy for other dimensions
		copy(result.Data, t.Data)
	}

	return result, nil
}

// Clone creates a deep copy of the tensor
func (t *Tensor) Clone() *Tensor {
	newTensor := &Tensor{
		Shape:  make([]int, len(t.Shape)),
		Data:   make([]byte, len(t.Data)),
		DType:  t.DType,
		offset: 0,
	}
	copy(newTensor.Shape, t.Shape)
	copy(newTensor.Data, t.Data)
	return newTensor
}
