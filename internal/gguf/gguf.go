package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	GGUF_MAGIC   = 0x46554747 // "GGUF" in little-endian is "GGUF"
	GGUF_VERSION = 3
)

type GGUFHeader struct {
	Magic         uint32
	Version       uint32
	TensorCount   uint64
	MetadataCount uint64
}

type GGUFMetadata struct {
	Key   string
	Type  uint32
	Value interface{}
}

type GGUFTensor struct {
	Name       string
	Dimensions []uint64
	Type       uint32
	Offset     uint64
}

type GGUFFile struct {
	Header   GGUFHeader
	Metadata map[string]interface{}
	Tensors  []GGUFTensor
	file     *os.File
	dataPos  int64
}

// Open reads a GGUF file and parses its structure
func Open(path string) (*GGUFFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	g := &GGUFFile{
		file:     f,
		Metadata: make(map[string]interface{}),
	}

	// Read header
	if err := binary.Read(f, binary.LittleEndian, &g.Header); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	if g.Header.Magic != GGUF_MAGIC {
		f.Close()
		return nil, fmt.Errorf("invalid GGUF magic: 0x%x", g.Header.Magic)
	}

	if g.Header.Version != GGUF_VERSION {
		f.Close()
		return nil, fmt.Errorf("unsupported GGUF version: %d", g.Header.Version)
	}

	// Read metadata
	for i := uint64(0); i < g.Header.MetadataCount; i++ {
		key, value, err := readMetadata(f)
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to read metadata %d: %w", i, err)
		}
		g.Metadata[key] = value
	}

	// Read tensor info
	g.Tensors = make([]GGUFTensor, g.Header.TensorCount)
	for i := uint64(0); i < g.Header.TensorCount; i++ {
		tensor, err := readTensorInfo(f)
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to read tensor %d: %w", i, err)
		}
		g.Tensors[i] = tensor
	}

	// After tensors, there is padding to reach the data section
	currentPos, _ := f.Seek(0, io.SeekCurrent)
	alignment := uint64(32) // Default alignment
	if val, ok := g.Metadata["general.alignment"].(uint32); ok {
		alignment = uint64(val)
	}

	padding := (alignment - (uint64(currentPos) % alignment)) % alignment
	g.dataPos, err = f.Seek(int64(padding), io.SeekCurrent)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to seek to data section: %w", err)
	}

	return g, nil
}

// GetTensor reads tensor data by name
func (g *GGUFFile) GetTensor(name string) ([]byte, error) {
	for _, t := range g.Tensors {
		if t.Name == name {
			// Calculate size based on dimensions and type
			size := calculateTensorSize(t.Dimensions, t.Type)

			// Seek to tensor data (Offset is relative to the start of the data section)
			if _, err := g.file.Seek(g.dataPos+int64(t.Offset), io.SeekStart); err != nil {
				return nil, fmt.Errorf("failed to seek to tensor %s data: %w", name, err)
			}

			// Read tensor data
			data := make([]byte, size)
			if _, err := io.ReadFull(g.file, data); err != nil {
				return nil, fmt.Errorf("failed to read tensor %s data: %w", name, err)
			}

			return data, nil
		}
	}

	return nil, fmt.Errorf("tensor not found: %s", name)
}

// Close closes the GGUF file
func (g *GGUFFile) Close() error {
	return g.file.Close()
}

// GGUF Type Enums
const (
	TypeUint8   uint32 = 0
	TypeInt8    uint32 = 1
	TypeUint16  uint32 = 2
	TypeInt16   uint32 = 3
	TypeUint32  uint32 = 4
	TypeInt32   uint32 = 5
	TypeFloat32 uint32 = 6
	TypeBool    uint32 = 7
	TypeString  uint32 = 8
	TypeArray   uint32 = 9
	TypeUint64  uint32 = 10
	TypeInt64   uint32 = 11
	TypeFloat64 uint32 = 12
)

func readString(r io.Reader) (string, error) {
	var len uint64
	if err := binary.Read(r, binary.LittleEndian, &len); err != nil {
		return "", err
	}
	buf := make([]byte, len)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readMetadata(r io.Reader) (string, interface{}, error) {
	key, err := readString(r)
	if err != nil {
		return "", nil, err
	}

	var mType uint32
	if err := binary.Read(r, binary.LittleEndian, &mType); err != nil {
		return "", nil, err
	}

	val, err := readValue(r, mType)
	return key, val, err
}

func readValue(r io.Reader, mType uint32) (interface{}, error) {
	switch mType {
	case TypeUint8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeInt8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeUint16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeInt16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeFloat32:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeUint64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeInt64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeFloat64:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeBool:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	case TypeString:
		return readString(r)
	case TypeArray:
		var itemType uint32
		if err := binary.Read(r, binary.LittleEndian, &itemType); err != nil {
			return nil, err
		}
		var len uint64
		if err := binary.Read(r, binary.LittleEndian, &len); err != nil {
			return nil, err
		}
		arr := make([]interface{}, len)
		for i := uint64(0); i < len; i++ {
			v, err := readValue(r, itemType)
			if err != nil {
				return nil, err
			}
			arr[i] = v
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unknown metadata type: %d", mType)
	}
}

func readTensorInfo(r io.Reader) (GGUFTensor, error) {
	name, err := readString(r)
	if err != nil {
		return GGUFTensor{}, err
	}

	var nDims uint32
	if err := binary.Read(r, binary.LittleEndian, &nDims); err != nil {
		return GGUFTensor{}, err
	}

	dims := make([]uint64, nDims)
	for i := uint32(0); i < nDims; i++ {
		if err := binary.Read(r, binary.LittleEndian, &dims[i]); err != nil {
			return GGUFTensor{}, err
		}
	}

	var tType uint32
	if err := binary.Read(r, binary.LittleEndian, &tType); err != nil {
		return GGUFTensor{}, err
	}

	var offset uint64
	if err := binary.Read(r, binary.LittleEndian, &offset); err != nil {
		return GGUFTensor{}, err
	}

	return GGUFTensor{
		Name:       name,
		Dimensions: dims,
		Type:       tType,
		Offset:     offset,
	}, nil
}

// GGUF Tensor Types (simplified)
const (
	GGML_TYPE_F32  = 0
	GGML_TYPE_F16  = 1
	GGML_TYPE_Q4_0 = 2
	GGML_TYPE_Q4_1 = 3
	GGML_TYPE_Q5_0 = 6
	GGML_TYPE_Q5_1 = 7
	GGML_TYPE_Q8_0 = 8
	GGML_TYPE_Q8_1 = 9
)

func calculateTensorSize(dims []uint64, typ uint32) int64 {
	count := uint64(1)
	for _, d := range dims {
		count *= d
	}

	switch typ {
	case GGML_TYPE_F32:
		return int64(count * 4)
	case GGML_TYPE_F16:
		return int64(count * 2)
	case GGML_TYPE_Q4_0:
		// 32 values per block, block size is 2 (f16 scale) + 16 (q4 data) = 18 bytes
		return int64((count + 31) / 32 * 18)
	case GGML_TYPE_Q4_1:
		// 32 values per block, block size is 2 (f16 scale) + 2 (f16 bias) + 16 (q4 data) = 20 bytes
		return int64((count + 31) / 32 * 20)
	case GGML_TYPE_Q5_0:
		// 32 values per block, block size is 2 (f16 scale) + 4 (qs) + 16 (q5 data) = 22 bytes
		return int64((count + 31) / 32 * 22)
	case GGML_TYPE_Q8_0:
		// 32 values per block, block size is 2 (f16 scale) + 32 (q8 data) = 34 bytes
		return int64((count + 31) / 32 * 34)
	default:
		// Fallback or error
		return int64(count) // Incorrect for quantized but better than 0
	}
}
