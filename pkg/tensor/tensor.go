// Package tensor provides core tensor data structures and operations
package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
)

// DType represents the data type of a tensor
type DType int

const (
	Float32 DType = iota
	Float16
	Int64
	Int32
	Uint8
)

// Tensor holds raw data and metadata
type Tensor struct {
	Shape   []int
	Data    interface{} // supports multiple types
	Strides []int
	DType   DType
}

// NewTensor allocates a new Tensor with given shape and type
func NewTensor(shape []int, dt DType) *Tensor {
	if len(shape) == 0 {
		return nil
	}

	size := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil
		}
		size *= dim
	}

	var data interface{}
	switch dt {
	case Float32:
		data = make([]float32, size)
	case Float16:
		data = make([]uint16, size)
	case Int64:
		data = make([]int64, size)
	case Int32:
		data = make([]int32, size)
	case Uint8:
		data = make([]uint8, size)
	default:
		data = make([]byte, size)
	}

	return &Tensor{
		Shape:   shape,
		Data:    data,
		Strides: shape,
		DType:   dt,
	}
}

// FromBuffer creates a tensor from an existing byte buffer by converting it to the correct type
func FromBuffer(buffer []byte, shape []int, dt DType) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	t := &Tensor{
		Shape:   shape,
		DType:   dt,
		Strides: shape,
	}

	switch dt {
	case Float32:
		data := make([]float32, size)
		for i := 0; i < size; i++ {
			if (i+1)*4 <= len(buffer) {
				data[i] = math.Float32frombits(binary.LittleEndian.Uint32(buffer[i*4 : (i+1)*4]))
			}
		}
		t.Data = data
	case Float16:
		data := make([]uint16, size)
		for i := 0; i < size; i++ {
			if (i+1)*2 <= len(buffer) {
				data[i] = binary.LittleEndian.Uint16(buffer[i*2 : (i+1)*2])
			}
		}
		t.Data = data
	case Uint8:
		data := make([]uint8, len(buffer))
		copy(data, buffer)
		t.Data = data
	default:
		t.Data = buffer
	}

	return t
}

// Clone creates a deep copy of the tensor including all data. (FIX-003)
func (t *Tensor) Clone() *Tensor {
	if t == nil {
		return nil
	}
	clone := &Tensor{
		Shape: make([]int, len(t.Shape)),
		DType: t.DType,
	}
	copy(clone.Shape, t.Shape)
	if t.Strides != nil {
		clone.Strides = make([]int, len(t.Strides))
		copy(clone.Strides, t.Strides)
	}
	switch src := t.Data.(type) {
	case []float32:
		dst := make([]float32, len(src)); copy(dst, src); clone.Data = dst
	case []uint16:
		dst := make([]uint16, len(src)); copy(dst, src); clone.Data = dst
	case []int64:
		dst := make([]int64, len(src)); copy(dst, src); clone.Data = dst
	case []int32:
		dst := make([]int32, len(src)); copy(dst, src); clone.Data = dst
	case []int:
		dst := make([]int, len(src)); copy(dst, src); clone.Data = dst
	case []uint8:
		dst := make([]uint8, len(src)); copy(dst, src); clone.Data = dst
	default:
		clone.Data = t.Data
	}
	return clone
}

// GetFloat32 retrieves a float32 value at index i
func (t *Tensor) GetFloat32(i int) float32 {
	if data, ok := t.Data.([]float32); ok {
		if i < len(data) {
			return data[i]
		}
	}
	return 0
}

// SetFloat32 sets a float32 value at index i
func (t *Tensor) SetFloat32(i int, v float32) {
	if data, ok := t.Data.([]float32); ok {
		if i < len(data) {
			data[i] = v
		}
	}
}

// GetInt64 retrieves element i as int64. Returns 0 if out of bounds or wrong type. (FIX-005)
func (t *Tensor) GetInt64(i int) int64 {
	if data, ok := t.Data.([]int64); ok && i < len(data) {
		return data[i]
	}
	return 0
}

// Size returns total number of elements. (FIX-006 — nil safety)
func (t *Tensor) Size() int {
	if t == nil || len(t.Shape) == 0 {
		return 0
	}
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

// Reshape returns a new Tensor with the specified shape
func (t *Tensor) Reshape(shape []int) *Tensor {
	return &Tensor{
		Shape:   shape,
		Data:    t.Data,
		Strides: shape,
		DType:   t.DType,
	}
}

// Transpose swaps dimensions dim1 and dim2, returning a new tensor. (FIX-002)
func (t *Tensor) Transpose(dim1, dim2 int) (*Tensor, error) {
	ndim := len(t.Shape)
	if dim1 < 0 || dim1 >= ndim || dim2 < 0 || dim2 >= ndim {
		return nil, fmt.Errorf("transpose: invalid dims %d,%d for %d-D tensor", dim1, dim2, ndim)
	}
	if dim1 == dim2 {
		return t.Clone(), nil
	}

	newShape := make([]int, ndim)
	copy(newShape, t.Shape)
	newShape[dim1], newShape[dim2] = newShape[dim2], newShape[dim1]

	result := NewTensor(newShape, t.DType)
	if result == nil {
		return nil, fmt.Errorf("transpose: failed to allocate result tensor")
	}
	total := t.Size()

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
		switch src := t.Data.(type) {
		case []float32:
			result.Data.([]float32)[flatNew] = src[flatOld]
		case []uint16:
			result.Data.([]uint16)[flatNew] = src[flatOld]
		case []int32:
			result.Data.([]int32)[flatNew] = src[flatOld]
		case []int64:
			result.Data.([]int64)[flatNew] = src[flatOld]
		case []uint8:
			result.Data.([]uint8)[flatNew] = src[flatOld]
		}
	}
	return result, nil
}

// ToFloat32 converts the tensor to Float32. Returns self if already Float32. (FIX-004)
func (t *Tensor) ToFloat32() *Tensor {
	if t == nil {
		return nil
	}
	if t.DType == Float32 {
		return t
	}
	size := t.Size()
	out := NewTensor(t.Shape, Float32)
	if out == nil {
		return t
	}
	outData := out.Data.([]float32)
	switch src := t.Data.(type) {
	case []uint16: // Float16 stored as uint16 bits
		for i, bits := range src {
			outData[i] = float16BitsToFloat32(bits)
		}
	case []int32:
		for i, v := range src {
			outData[i] = float32(v)
		}
	case []int64:
		for i, v := range src {
			outData[i] = float32(v)
		}
	case []int:
		for i, v := range src {
			outData[i] = float32(v)
		}
	case []uint8:
		for i, v := range src {
			outData[i] = float32(v)
		}
	default:
		_ = size
	}
	return out
}

// float16BitsToFloat32 converts IEEE 754 half-precision bits to float32.
func float16BitsToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp  := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)
	var f uint32
	switch exp {
	case 0:
		if mant == 0 {
			f = sign
		} else {
			e := uint32(1)
			for mant&0x400 == 0 {
				mant <<= 1
				e++
			}
			mant &^= 0x400
			f = sign | ((127 - 15 - e + 1) << 23) | (mant << 13)
		}
	case 31:
		f = sign | 0x7F800000 | (mant << 13)
	default:
		f = sign | ((exp+127-15)<<23) | (mant << 13)
	}
	return math.Float32frombits(f)
}