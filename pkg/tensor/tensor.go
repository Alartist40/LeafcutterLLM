// Package tensor provides efficient tensor operations for LLM inference
package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
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
	offset int // For sliced views (in bytes)
}

// NewTensor creates a new tensor with the given shape and dtype.
// Panics if any dimension is <= 0.
func NewTensor(shape []int, dtype DType) *Tensor {
	size := 1
	for _, dim := range shape {
		if dim <= 0 {
			panic(fmt.Sprintf("tensor: invalid dimension %d in shape %v", dim, shape))
		}
		size *= dim
	}
	shapeCopy := make([]int, len(shape))
	copy(shapeCopy, shape)
	return &Tensor{
		Shape: shapeCopy,
		Data:  make([]byte, size*dtype.Size()),
		DType: dtype,
	}
}

// FromBuffer creates a tensor from an existing byte buffer
func FromBuffer(data []byte, shape []int, dtype DType) *Tensor {
	shapeCopy := make([]int, len(shape))
	copy(shapeCopy, shape)
	return &Tensor{
		Shape: shapeCopy,
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

// Reshape returns a view with new shape (sharing underlying data)
func (t *Tensor) Reshape(newShape []int) (*Tensor, error) {
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}
	if newSize != t.Size() {
		return nil, fmt.Errorf("cannot reshape tensor of size %d into shape %v", t.Size(), newShape)
	}
	sc := make([]int, len(newShape))
	copy(sc, newShape)
	return &Tensor{Shape: sc, Data: t.Data, DType: t.DType, offset: t.offset}, nil
}

// Slice returns a sliced view (zero-copy) along the first dimension
func (t *Tensor) Slice(start, end int) (*Tensor, error) {
	if len(t.Shape) == 0 {
		return nil, fmt.Errorf("cannot slice 0-d tensor")
	}
	if start < 0 || end > t.Shape[0] || start >= end {
		return nil, fmt.Errorf("invalid slice range [%d:%d] for dimension %d", start, end, t.Shape[0])
	}
	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[0] = end - start
	stride := t.Size() / t.Shape[0]
	return &Tensor{
		Shape:  newShape,
		Data:   t.Data,
		DType:  t.DType,
		offset: t.offset + start*stride*t.DType.Size(),
	}, nil
}

// elementOffset returns the byte position of element idx in t.Data
func (t *Tensor) elementOffset(idx int) int {
	return t.offset + idx*t.DType.Size()
}

// GetFloat32 returns a float32 value at the given flat index
func (t *Tensor) GetFloat32(idx int) float32 {
	if t.DType != Float32 {
		panic(fmt.Sprintf("cannot GetFloat32 on %s tensor", t.DType))
	}
	off := t.elementOffset(idx)
	return *(*float32)(unsafe.Pointer(&t.Data[off]))
}

// SetFloat32 sets a float32 value at the given flat index
func (t *Tensor) SetFloat32(idx int, val float32) {
	if t.DType != Float32 {
		panic(fmt.Sprintf("cannot SetFloat32 on %s tensor", t.DType))
	}
	off := t.elementOffset(idx)
	binary.LittleEndian.PutUint32(t.Data[off:], math.Float32bits(val))
}

// GetFloat16 returns a float16 value decoded as float32
func (t *Tensor) GetFloat16(idx int) float32 {
	if t.DType != Float16 {
		panic(fmt.Sprintf("cannot GetFloat16 on %s tensor", t.DType))
	}
	off := t.elementOffset(idx)
	bits := binary.LittleEndian.Uint16(t.Data[off:])
	return Float16ToFloat32(bits)
}

// SetFloat16 encodes a float32 as float16 and stores it
func (t *Tensor) SetFloat16(idx int, val float32) {
	if t.DType != Float16 {
		panic(fmt.Sprintf("cannot SetFloat16 on %s tensor", t.DType))
	}
	off := t.elementOffset(idx)
	binary.LittleEndian.PutUint16(t.Data[off:], Float32ToFloat16(val))
}

// GetInt64 returns an int64 value at the given flat index
func (t *Tensor) GetInt64(idx int) int64 {
	if t.DType != Int64 {
		panic(fmt.Sprintf("cannot GetInt64 on %s tensor", t.DType))
	}
	off := t.elementOffset(idx)
	return int64(binary.LittleEndian.Uint64(t.Data[off:]))
}

// ToFloat32 converts the tensor to Float32 dtype (allocates new memory)
func (t *Tensor) ToFloat32() *Tensor {
	if t.DType == Float32 {
		newTensor := NewTensor(t.Shape, Float32)
		copy(newTensor.Data, t.Data[t.offset:])
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

// Float16ToFloat32 converts IEEE 754 half-precision to single-precision.
// FIX: Prior subnormal path used float multiplication (sign * frac) which
// produced wrong results. Now uses correct IEEE 754 bit construction.
func Float16ToFloat32(bits uint16) float32 {
	sign := uint32(bits>>15) << 31
	exp := uint32((bits >> 10) & 0x1F)
	frac := uint32(bits & 0x3FF)

	var result uint32
	switch {
	case exp == 0 && frac == 0:
		result = sign // signed zero
	case exp == 0:
		// Subnormal: renormalize
		e := uint32(1)
		f := frac
		for f&0x400 == 0 {
			f <<= 1
			e++
		}
		f &^= 0x400 // remove implicit leading 1
		// f16 subnormal exp = -14; f32 bias = 127
		// new_exp = 127 + (-14) - (e-1) = 114 - e + 1 = 115 - e
		result = sign | ((115-e+127-1)<<23) | (f << 13)
	case exp == 0x1F && frac == 0:
		result = sign | 0x7F800000 // infinity
	case exp == 0x1F:
		result = sign | 0x7FC00000 | (frac << 13) // NaN
	default:
		// Normal: rebias from f16 (bias=15) to f32 (bias=127): add 112
		result = sign | ((exp+112)<<23) | (frac << 13)
	}
	return math.Float32frombits(result)
}

// Float32ToFloat16 converts single-precision to IEEE 754 half-precision
func Float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>31) << 15
	exp := int32((bits>>23)&0xFF) - 127
	frac := bits & 0x7FFFFF

	switch {
	case exp > 15:
		return sign | 0x7C00 // overflow to infinity
	case exp < -24:
		return sign // underflow to zero
	case exp < -14:
		// Subnormal f16
		frac = (frac | 0x800000) >> uint32(-14-exp)
		return sign | uint16(frac>>13)
	default:
		return sign | uint16((exp+15)<<10) | uint16(frac>>13)
	}
}

// Transpose swaps two dimensions.
// FIX: Previous N-D path did a raw byte copy (wrong). Now does correct element
// permutation for any number of dimensions.
func (t *Tensor) Transpose(dim1, dim2 int) (*Tensor, error) {
	ndim := len(t.Shape)
	if dim1 < 0 || dim1 >= ndim || dim2 < 0 || dim2 >= ndim {
		return nil, fmt.Errorf("invalid transpose dimensions %d, %d for %d-D tensor", dim1, dim2, ndim)
	}
	if dim1 == dim2 {
		return t.Clone(), nil
	}

	newShape := make([]int, ndim)
	copy(newShape, t.Shape)
	newShape[dim1], newShape[dim2] = newShape[dim2], newShape[dim1]

	result := NewTensor(newShape, t.DType)
	total := t.Size()
	elemSize := t.DType.Size()

	// Build strides for old and new layouts
	oldStrides := make([]int, ndim)
	newStrides := make([]int, ndim)
	oldStrides[ndim-1] = 1
	newStrides[ndim-1] = 1
	for i := ndim - 2; i >= 0; i-- {
		oldStrides[i] = oldStrides[i+1] * t.Shape[i+1]
		newStrides[i] = newStrides[i+1] * newShape[i+1]
	}

	coords := make([]int, ndim)
	for flatOld := 0; flatOld < total; flatOld++ {
		rem := flatOld
		for d := 0; d < ndim; d++ {
			coords[d] = rem / oldStrides[d]
			rem %= oldStrides[d]
		}
		coords[dim1], coords[dim2] = coords[dim2], coords[dim1]
		flatNew := 0
		for d := 0; d < ndim; d++ {
			flatNew += coords[d] * newStrides[d]
		}
		srcOff := t.elementOffset(flatOld)
		copy(result.Data[flatNew*elemSize:], t.Data[srcOff:srcOff+elemSize])
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
	copy(newTensor.Data, t.Data[t.offset:])
	return newTensor
}
