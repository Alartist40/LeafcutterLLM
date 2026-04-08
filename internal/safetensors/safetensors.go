// Package safetensors provides parsing for Hugging Face safetensors format
package safetensors

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"

	"github.com/xander/airllm-go/pkg/tensor"
)

const (
	HeaderLengthSize = 8
)

// Header represents the JSON header of a safetensors file
type Header struct {
	Tensors map[string]TensorInfo
}

// TensorInfo contains metadata for a tensor in the file
type TensorInfo struct {
	DType  string  `json:"dtype"`
	Shape  []int   `json:"shape"`
	Data   Offsets `json:"data_offsets"`
}

// Offsets represents byte offsets in the file
type Offsets [2]int64

// Reader provides safe concurrent access to safetensors files
type Reader struct {
	path      string
	file      *os.File
	headerLen uint64
	header    Header
	mu        sync.RWMutex
}

// Open opens a safetensors file for reading
func Open(path string) (*Reader, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}

	r := &Reader{
		path: path,
		file: file,
	}

	if err := r.readHeader(); err != nil {
		file.Close()
		return nil, err
	}

	return r, nil
}

// Close closes the safetensors file
func (r *Reader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.file != nil {
		err := r.file.Close()
		r.file = nil
		return err
	}
	return nil
}

// readHeader reads and parses the JSON header
func (r *Reader) readHeader() error {
	// Read header length (8 bytes, little endian)
	var headerLenBytes [8]byte
	if _, err := r.file.Read(headerLenBytes[:]); err != nil {
		return fmt.Errorf("failed to read header length: %w", err)
	}

	// Parse as little endian uint64
	headerLen := uint64(0)
	for i := 0; i < 8; i++ {
		headerLen |= uint64(headerLenBytes[i]) << (8 * i)
	}
	r.headerLen = headerLen

	// Read header JSON
	headerBytes := make([]byte, headerLen)
	if _, err := r.file.Read(headerBytes); err != nil {
		return fmt.Errorf("failed to read header: %w", err)
	}

	// Parse JSON
	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return fmt.Errorf("failed to parse header JSON: %w", err)
	}

	r.header.Tensors = make(map[string]TensorInfo)
	for name, data := range rawHeader {
		// Skip __metadata__ key
		if name == "__metadata__" {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(data, &info); err != nil {
			return fmt.Errorf("failed to parse tensor info for %s: %w", name, err)
		}
		r.header.Tensors[name] = info
	}

	return nil
}

// GetTensor reads a single tensor from the file
func (r *Reader) GetTensor(name string) (*tensor.Tensor, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	info, ok := r.header.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %s not found", name)
	}

	dtype := parseDType(info.DType)
	if dtype == -1 {
		return nil, fmt.Errorf("unsupported dtype: %s", info.DType)
	}

	// Calculate actual file offset
	dataStart := int64(HeaderLengthSize) + int64(r.headerLen) + info.Data[0]
	dataSize := int(info.Data[1] - info.Data[0])

	// Read tensor data
	data := make([]byte, dataSize)
	if _, err := r.file.ReadAt(data, dataStart); err != nil {
		return nil, fmt.Errorf("failed to read tensor data: %w", err)
	}

	return tensor.FromBuffer(data, info.Shape, dtype), nil
}

// GetTensorsWithPrefix returns all tensors whose names start with the given prefix
func (r *Reader) GetTensorsWithPrefix(prefix string) (map[string]*tensor.Tensor, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string]*tensor.Tensor)
	var mu sync.Mutex
	var wg sync.WaitGroup
	errChan := make(chan error, len(r.header.Tensors))

	for name, info := range r.header.Tensors {
		if len(name) >= len(prefix) && name[:len(prefix)] == prefix {
			wg.Add(1)
			go func(n string, ti TensorInfo) {
				defer wg.Done()

				dtype := parseDType(ti.DType)
				if dtype == -1 {
					errChan <- fmt.Errorf("unsupported dtype: %s", ti.DType)
					return
				}

				dataStart := int64(HeaderLengthSize) + int64(r.headerLen) + ti.Data[0]
				dataSize := int(ti.Data[1] - ti.Data[0])

				data := make([]byte, dataSize)
				if _, err := r.file.ReadAt(data, dataStart); err != nil {
					errChan <- fmt.Errorf("failed to read tensor %s: %w", n, err)
					return
				}

				t := tensor.FromBuffer(data, ti.Shape, dtype)
				mu.Lock()
				result[n] = t
				mu.Unlock()
			}(name, info)
		}
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		if err != nil {
			return nil, err
		}
	}

	return result, nil
}

// ListTensors returns a sorted list of all tensor names
func (r *Reader) ListTensors() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.header.Tensors))
	for name := range r.header.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// GetLayerNames returns unique layer prefixes (e.g., "model.layers.0.")
func (r *Reader) GetLayerNames() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	layerSet := make(map[string]bool)
	for name := range r.header.Tensors {
		// Extract layer prefix
		for i := len(name) - 1; i >= 0; i-- {
			if name[i] == '.' {
				layerSet[name[:i+1]] = true
				break
			}
		}
	}

	layers := make([]string, 0, len(layerSet))
	for layer := range layerSet {
		layers = append(layers, layer)
	}
	sort.Strings(layers)
	return layers
}

// parseDType converts string dtype to tensor.DType
func parseDType(s string) tensor.DType {
	switch s {
	case "F32":
		return tensor.Float32
	case "F16":
		return tensor.Float16
	case "I64":
		return tensor.Int64
	case "I32":
		return tensor.Int32
	case "U8":
		return tensor.Uint8
	default:
		return -1
	}
}

// LoadLayer loads tensors belonging to a specific layer
func LoadLayer(path, layerName string) (map[string]*tensor.Tensor, error) {
	reader, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	return reader.GetTensorsWithPrefix(layerName)
}

// FileInfo holds metadata about a safetensors file
type FileInfo struct {
	Path       string
	TensorCount int
	TotalSize  int64
}

// ScanDirectory scans for safetensors files and returns info about them
func ScanDirectory(dir string) ([]FileInfo, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var files []FileInfo
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if len(name) > 11 && name[len(name)-11:] == ".safetensors" {
			info, err := entry.Info()
			if err != nil {
				continue
			}
			files = append(files, FileInfo{
				Path:      dir + "/" + name,
				TotalSize: info.Size(),
			})
		}
	}
	return files, nil
}

// LoadModelIndex loads model.safetensors.index.json
func LoadModelIndex(path string) (map[string]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, err
	}

	return index.WeightMap, nil
}
