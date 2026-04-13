package inference

import (
	"fmt"
	"math"
	"sync"

	"github.com/Alartist40/LeafcutterLLM/pkg/tensor"
)

// LinearLayer implements a fully connected layer
type LinearLayer struct {
	name   string
	config *Config
	weight *tensor.Tensor
	bias   *tensor.Tensor
	mu     sync.RWMutex
}

// NewLinearLayer creates a new linear layer
func NewLinearLayer(name string, config *Config) *LinearLayer {
	return &LinearLayer{
		name:   name,
		config: config,
	}
}

// Name returns the layer name
func (l *LinearLayer) Name() string {
	return l.name
}

// Load loads the layer weights
func (l *LinearLayer) Load(state map[string]*tensor.Tensor) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Look for weight
	weightKey := l.name + ".weight"
	if w, ok := state[weightKey]; ok {
		l.weight = w
	} else {
		// Try without prefix
		for k, v := range state {
			if len(k) >= 7 && k[len(k)-7:] == ".weight" {
				l.weight = v
				break
			}
		}
	}

	// Look for bias
	biasKey := l.name + ".bias"
	if b, ok := state[biasKey]; ok {
		l.bias = b
	}

	return nil
}

// Unload clears the weights
func (l *LinearLayer) Unload() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.weight = nil
	l.bias = nil
	return nil
}

// Forward performs linear transformation: y = xW^T + b
func (l *LinearLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	l.mu.RLock()
	weight := l.weight
	bias := l.bias
	l.mu.RUnlock()

	if weight == nil {
		return nil, fmt.Errorf("weight not loaded for layer %s", l.name)
	}

	// Simple matrix multiplication
	// Assuming input shape [batch, seq_len, in_features] and weight [out_features, in_features]
	// or input [batch, in_features] and weight [out_features, in_features]
	
	output, err := matmul(input, weight)
	if err != nil {
		return nil, fmt.Errorf("matmul failed: %w", err)
	}

	if bias != nil {
		output, err = addBias(output, bias)
		if err != nil {
			return nil, fmt.Errorf("add bias failed: %w", err)
		}
	}

	return output, nil
}

// EmbeddingLayer implements token embeddings
type EmbeddingLayer struct {
	name   string
	config *Config
	weight *tensor.Tensor
	mu     sync.RWMutex
}

// NewEmbeddingLayer creates a new embedding layer
func NewEmbeddingLayer(name string) *EmbeddingLayer {
	return &EmbeddingLayer{name: name}
}

// Name returns the layer name
func (l *EmbeddingLayer) Name() string {
	return l.name
}

// Load loads the embedding weights
func (l *EmbeddingLayer) Load(state map[string]*tensor.Tensor) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	for k, v := range state {
		if len(k) >= 7 && k[len(k)-7:] == ".weight" {
			l.weight = v
			return nil
		}
	}

	return fmt.Errorf("embedding weight not found")
}

// Unload clears the weights
func (l *EmbeddingLayer) Unload() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.weight = nil
	return nil
}

// Forward performs embedding lookup
func (l *EmbeddingLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	l.mu.RLock()
	weight := l.weight
	l.mu.RUnlock()

	if weight == nil {
		return nil, fmt.Errorf("embedding weight not loaded")
	}

	// Input should be int64 token IDs
	// Weight shape: [vocab_size, hidden_size]
	// Output shape: [batch, seq_len, hidden_size]

	return embedLookup(input, weight)
}

// LayerNorm implements RMS/Layer normalization
type LayerNorm struct {
	name      string
	config    *Config
	weight    *tensor.Tensor
	epsilon   float32
	isRMSNorm bool
	mu        sync.RWMutex
}

// NewLayerNorm creates a new layer norm
func NewLayerNorm(name string, config *Config) *LayerNorm {
	return &LayerNorm{
		name:    name,
		config:  config,
		epsilon: 1e-6,
		isRMSNorm: true, // Most modern LLMs use RMSNorm
	}
}

// SetRMSNorm configures whether to use RMSNorm (true) or standard LayerNorm (false)
func (l *LayerNorm) SetRMSNorm(isRMSNorm bool) {
	l.isRMSNorm = isRMSNorm
}

// Name returns the layer name
func (l *LayerNorm) Name() string {
	return l.name
}

// Load loads normalization weights
func (l *LayerNorm) Load(state map[string]*tensor.Tensor) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	for k, v := range state {
		if len(k) >= 7 && k[len(k)-7:] == ".weight" {
			l.weight = v
			return nil
		}
	}

	return nil // Some layer norms don't have weights
}

// Unload clears the weights
func (l *LayerNorm) Unload() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.weight = nil
	return nil
}

// Forward applies normalization
func (l *LayerNorm) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	l.mu.RLock()
	weight := l.weight
	l.mu.RUnlock()

	if l.isRMSNorm {
		return rmsNorm(input, weight, l.epsilon)
	}
	return layerNorm(input, weight, l.epsilon)
}

// AttentionLayer implements self-attention
type AttentionLayer struct {
	name       string
	config     *Config
	qWeight    *tensor.Tensor
	kWeight    *tensor.Tensor
	vWeight    *tensor.Tensor
	oWeight    *tensor.Tensor
	qBias      *tensor.Tensor
	kBias      *tensor.Tensor
	vBias      *tensor.Tensor
	oBias      *tensor.Tensor
	numHeads   int
	headDim    int
	mu         sync.RWMutex
}

// NewAttentionLayer creates a new attention layer
func NewAttentionLayer(name string, config *Config) *AttentionLayer {
	return &AttentionLayer{
		name:     name,
		config:   config,
		numHeads: 32,  // Default, will be inferred from weights
		headDim:  128, // Default
	}
}

// Name returns the layer name
func (l *AttentionLayer) Name() string {
	return l.name
}

// Load loads attention weights
func (l *AttentionLayer) Load(state map[string]*tensor.Tensor) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	for k, v := range state {
		switch {
		case contains(k, "q_proj") || contains(k, "query"):
			if contains(k, "weight") {
				l.qWeight = v
				// Infer num heads from weight size
				if len(v.Shape) >= 2 {
					l.headDim = v.Shape[1] / l.numHeads
				}
			} else if contains(k, "bias") {
				l.qBias = v
			}
		case contains(k, "k_proj") || contains(k, "key"):
			if contains(k, "weight") {
				l.kWeight = v
			} else if contains(k, "bias") {
				l.kBias = v
			}
		case contains(k, "v_proj") || contains(k, "value"):
			if contains(k, "weight") {
				l.vWeight = v
			} else if contains(k, "bias") {
				l.vBias = v
			}
		case contains(k, "o_proj") || contains(k, "output") || contains(k, "proj"):
			if contains(k, "weight") {
				l.oWeight = v
			} else if contains(k, "bias") {
				l.oBias = v
			}
		}
	}

	return nil
}

// Unload clears the weights
func (l *AttentionLayer) Unload() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.qWeight, l.kWeight, l.vWeight, l.oWeight = nil, nil, nil, nil
	l.qBias, l.kBias, l.vBias, l.oBias = nil, nil, nil, nil
	return nil
}

// Forward performs self-attention
func (l *AttentionLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	l.mu.RLock()
	qWeight, kWeight, vWeight, oWeight := l.qWeight, l.kWeight, l.vWeight, l.oWeight
	qBias, kBias, vBias, oBias := l.qBias, l.kBias, l.vBias, l.oBias
	numHeads := l.numHeads
	headDim := l.headDim
	l.mu.RUnlock()

	if qWeight == nil || kWeight == nil || vWeight == nil || oWeight == nil {
		return nil, fmt.Errorf("attention weights not properly loaded")
	}

	// Linear projections
	q, err := matmul(input, qWeight)
	if err != nil {
		return nil, err
	}
	k, err := matmul(input, kWeight)
	if err != nil {
		return nil, err
	}
	v, err := matmul(input, vWeight)
	if err != nil {
		return nil, err
	}

	if qBias != nil {
		q, _ = addBias(q, qBias)
	}
	if kBias != nil {
		k, _ = addBias(k, kBias)
	}
	if vBias != nil {
		v, _ = addBias(v, vBias)
	}

	// Reshape for multi-head attention: [batch, seq, heads * head_dim] -> [batch, heads, seq, head_dim]
	batchSize := input.Shape[0]
	seqLen := input.Shape[1]

	q = reshape4D(q, batchSize, seqLen, numHeads, headDim)
	k = reshape4D(k, batchSize, seqLen, numHeads, headDim)
	v = reshape4D(v, batchSize, seqLen, numHeads, headDim)

	// Transpose for attention: [batch, heads, seq, head_dim]
	q, _ = q.Transpose(1, 2)
	k, _ = k.Transpose(1, 2)
	v, _ = v.Transpose(1, 2)

	// Scaled dot-product attention
	attnOutput, err := scaledDotProductAttention(q, k, v, nil)
	if err != nil {
		return nil, err
	}

	// Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, heads * head_dim]
	attnOutput, _ = attnOutput.Transpose(1, 2)
	attnOutput = reshape3D(attnOutput, batchSize, seqLen, numHeads*headDim)

	// Output projection
	output, err := matmul(attnOutput, oWeight)
	if err != nil {
		return nil, err
	}

	if oBias != nil {
		output, _ = addBias(output, oBias)
	}

	return output, nil
}

// FFNLayer implements the feed-forward network
type FFNLayer struct {
	name     string
	config   *Config
	gateWeight *tensor.Tensor
	upWeight   *tensor.Tensor
	downWeight *tensor.Tensor
	gateBias   *tensor.Tensor
	upBias     *tensor.Tensor
	downBias   *tensor.Tensor
	mu         sync.RWMutex
}

// NewFFNLayer creates a new FFN layer
func NewFFNLayer(name string, config *Config) *FFNLayer {
	return &FFNLayer{
		name:   name,
		config: config,
	}
}

// Name returns the layer name
func (l *FFNLayer) Name() string {
	return l.name
}

// Load loads FFN weights
func (l *FFNLayer) Load(state map[string]*tensor.Tensor) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	for k, v := range state {
		switch {
		case contains(k, "gate_proj") || contains(k, "w1") || contains(k, "fc1"):
			if contains(k, "weight") {
				l.gateWeight = v
			} else if contains(k, "bias") {
				l.gateBias = v
			}
		case contains(k, "up_proj") || contains(k, "w2") || contains(k, "fc2"):
			if contains(k, "weight") {
				l.upWeight = v
			} else if contains(k, "bias") {
				l.upBias = v
			}
		case contains(k, "down_proj") || contains(k, "w3") || contains(k, "fc3"):
			if contains(k, "weight") {
				l.downWeight = v
			} else if contains(k, "bias") {
				l.downBias = v
			}
		}
	}

	return nil
}

// Unload clears the weights
func (l *FFNLayer) Unload() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.gateWeight, l.upWeight, l.downWeight = nil, nil, nil
	l.gateBias, l.upBias, l.downBias = nil, nil, nil
	return nil
}

// Forward performs SwiGLU FFN: x = x * gate(x) then up * down
func (l *FFNLayer) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	l.mu.RLock()
	gateWeight, upWeight, downWeight := l.gateWeight, l.upWeight, l.downWeight
	gateBias, upBias, downBias := l.gateBias, l.upBias, l.downBias
	l.mu.RUnlock()

	if upWeight == nil || downWeight == nil {
		return nil, fmt.Errorf("FFN weights not properly loaded")
	}

	var hidden *tensor.Tensor
	var err error

	if gateWeight != nil {
		// SwiGLU activation: gate(x) * up(x)
		gate, err := matmul(input, gateWeight)
		if err != nil {
			return nil, err
		}
		if gateBias != nil {
			gate, _ = addBias(gate, gateBias)
		}
		gate = silu(gate)

		up, err := matmul(input, upWeight)
		if err != nil {
			return nil, err
		}
		if upBias != nil {
			up, _ = addBias(up, upBias)
		}

		hidden, err = mul(gate, up)
		if err != nil {
			return nil, err
		}
	} else {
		// Standard FFN
		hidden, err = matmul(input, upWeight)
		if err != nil {
			return nil, err
		}
		if upBias != nil {
			hidden, _ = addBias(hidden, upBias)
		}
		hidden = gelu(hidden)
	}

	// Down projection
	output, err := matmul(hidden, downWeight)
	if err != nil {
		return nil, err
	}
	if downBias != nil {
		output, _ = addBias(output, downBias)
	}

	return output, nil
}

// Helper functions for tensor operations

func matmul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	// Simplified 2D matmul: [M, K] x [K, N] = [M, N]
	// or batched: [B, M, K] x [K, N] = [B, M, N]
	
	if len(a.Shape) < 2 || len(b.Shape) < 2 {
		return nil, fmt.Errorf("matmul requires at least 2D tensors")
	}

	// Get the actual 2D shapes
	var m, k1, k2, n int
	batch := 1

	if len(a.Shape) == 2 {
		m = a.Shape[0]
		k1 = a.Shape[1]
	} else {
		// Batched
		batch = 1
		for i := 0; i < len(a.Shape)-2; i++ {
			batch *= a.Shape[i]
		}
		m = a.Shape[len(a.Shape)-2]
		k1 = a.Shape[len(a.Shape)-1]
	}

	k2 = b.Shape[len(b.Shape)-2]
	n = b.Shape[len(b.Shape)-1]

	if k1 != k2 {
		return nil, fmt.Errorf("matmul dimension mismatch: %d vs %d", k1, k2)
	}

	// Create output tensor
	var outShape []int
	if len(a.Shape) == 2 {
		outShape = []int{m, n}
	} else {
		outShape = make([]int, len(a.Shape))
		copy(outShape, a.Shape[:len(a.Shape)-1])
		outShape[len(outShape)-1] = n
	}

	output := tensor.NewTensor(outShape, tensor.Float32)

	// Naive implementation - in production would use optimized BLAS
	aF32 := a.ToFloat32()
	bF32 := b.ToFloat32()

	for bi := 0; bi < batch; bi++ {
		for mi := 0; mi < m; mi++ {
			for ni := 0; ni < n; ni++ {
				sum := float32(0)
				for ki := 0; ki < k1; ki++ {
					aIdx := bi*m*k1 + mi*k1 + ki
					bIdx := ki*n + ni
					sum += aF32.GetFloat32(aIdx) * bF32.GetFloat32(bIdx)
				}
				outIdx := bi*m*n + mi*n + ni
				output.SetFloat32(outIdx, sum)
			}
		}
	}

	if a.DType == tensor.Float16 || b.DType == tensor.Float16 {
		// Convert back to float16 if inputs were float16
		// For now just return float32
	}

	return output, nil
}

func addBias(input, bias *tensor.Tensor) (*tensor.Tensor, error) {
	// Add bias to last dimension
	result := input.Clone()
	
	if len(bias.Shape) == 1 {
		// Broadcast over all but last dimension
		lastDim := bias.Shape[0]
		numElements := input.Size() / lastDim
		
		for i := 0; i < numElements; i++ {
			for j := 0; j < lastDim; j++ {
				idx := i*lastDim + j
				var val float32
				switch input.DType {
				case tensor.Float32:
					val = input.GetFloat32(idx)
				case tensor.Float16:
					val = input.GetFloat16(idx)
				}
				var biasVal float32
				switch bias.DType {
				case tensor.Float32:
					biasVal = bias.GetFloat32(j)
				case tensor.Float16:
					biasVal = bias.GetFloat16(j)
				}
				result.SetFloat32(idx, val+biasVal)
			}
		}
	}

	return result, nil
}

func embedLookup(input *tensor.Tensor, weight *tensor.Tensor) (*tensor.Tensor, error) {
	// input: [batch, seq_len] int64
	// weight: [vocab_size, hidden_dim]
	// output: [batch, seq_len, hidden_dim]

	return nil, fmt.Errorf("embedLookup not fully implemented")
}

func rmsNorm(input, weight *tensor.Tensor, epsilon float32) (*tensor.Tensor, error) {
	// RMS normalization: x / sqrt(mean(x^2) + epsilon) * weight
	
	lastDim := input.Shape[len(input.Shape)-1]
	numGroups := input.Size() / lastDim

	result := tensor.NewTensor(input.Shape, input.DType)
	inputF32 := input.ToFloat32()

	for g := 0; g < numGroups; g++ {
		// Calculate RMS
		var sumSquares float32
		for i := 0; i < lastDim; i++ {
			idx := g*lastDim + i
			val := inputF32.GetFloat32(idx)
			sumSquares += val * val
		}
		rms := float32(math.Sqrt(float64(sumSquares/float32(lastDim) + epsilon)))

		// Normalize and multiply by weight
		for i := 0; i < lastDim; i++ {
			idx := g*lastDim + i
			val := inputF32.GetFloat32(idx)
			normalized := val / rms
			if weight != nil {
				var w float32
				switch weight.DType {
				case tensor.Float32:
					w = weight.GetFloat32(i)
				case tensor.Float16:
					w = weight.GetFloat16(i)
				}
				normalized *= w
			}
			result.SetFloat32(idx, normalized)
		}
	}

	return result, nil
}

func layerNorm(input, weight *tensor.Tensor, epsilon float32) (*tensor.Tensor, error) {
	// Similar to RMS norm but also subtracts mean
	return input, nil // Placeholder
}

func scaledDotProductAttention(q, k, v, mask *tensor.Tensor) (*tensor.Tensor, error) {
	// q, k, v: [batch, heads, seq, head_dim]
	// output: [batch, heads, seq, head_dim]

	// For now, return simplified version
	return v, nil // Placeholder
}

func reshape4D(t *tensor.Tensor, b, s, h, d int) *tensor.Tensor {
	return tensor.FromBuffer(t.Data, []int{b, s, h, d}, t.DType)
}

func reshape3D(t *tensor.Tensor, b, s, d int) *tensor.Tensor {
	return tensor.FromBuffer(t.Data, []int{b, s, d}, t.DType)
}

func silu(x *tensor.Tensor) *tensor.Tensor {
	// SiLU(x) = x * sigmoid(x)
	result := tensor.NewTensor(x.Shape, x.DType)
	inputF32 := x.ToFloat32()

	for i := 0; i < x.Size(); i++ {
		val := inputF32.GetFloat32(i)
		sigmoid := float32(1 / (1 + math.Exp(-float64(val))))
		result.SetFloat32(i, val*sigmoid)
	}

	return result
}

func mul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	result := tensor.NewTensor(a.Shape, a.DType)
	
	for i := 0; i < a.Size(); i++ {
		var av, bv float32
		switch a.DType {
		case tensor.Float32:
			av = a.GetFloat32(i)
		case tensor.Float16:
			av = a.GetFloat16(i)
		}
		switch b.DType {
		case tensor.Float32:
			bv = b.GetFloat32(i)
		case tensor.Float16:
			bv = b.GetFloat16(i)
		}
		result.SetFloat32(i, av*bv)
	}

	return result, nil
}

func gelu(x *tensor.Tensor) *tensor.Tensor {
	// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	result := tensor.NewTensor(x.Shape, x.DType)
	inputF32 := x.ToFloat32()

	sqrt2OverPi := float32(math.Sqrt(2.0 / math.Pi))

	for i := 0; i < x.Size(); i++ {
		val := inputF32.GetFloat32(i)
		inner := sqrt2OverPi * (val + 0.044715*val*val*val)
		result.SetFloat32(i, 0.5*val*(1+float32(math.Tanh(float64(inner)))))
	}

	return result
}
