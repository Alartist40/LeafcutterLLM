package inference

import (
	"fmt"
	"math"
	"sync"

	"github.com/Alartist40/LeafcutterLLM/pkg/qkernel"
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

func NewLinearLayer(name string, config *Config) *LinearLayer {
	return &LinearLayer{name: name, config: config}
}

func (l *LinearLayer) Name() string { return l.name }

func (l *LinearLayer) Load(state map[string]*tensor.Tensor) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	weightKey := l.name + ".weight"
	if w, ok := state[weightKey]; ok {
		l.weight = w
	} else {
		for k, v := range state {
			if contains(k, l.name) && contains(k, ".weight") {
				l.weight = v
				break
			}
		}
	}
	biasKey := l.name + ".bias"
	if b, ok := state[biasKey]; ok {
		l.bias = b
	}
	return nil
}

func (l *LinearLayer) Unload() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.weight = nil
	l.bias = nil
	return nil
}

func (l *LinearLayer) Forward(input, pastK, pastV *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	l.mu.RLock()
	weight, bias := l.weight, l.bias
	l.mu.RUnlock()
	if weight == nil {
		return nil, nil, nil, fmt.Errorf("weight not loaded for layer %s", l.name)
	}

	output, err := matmulTransposed(input, weight)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("matmul failed: %w", err)
	}
	if bias != nil {
		if output, err = addBias(output, bias); err != nil {
			return nil, nil, nil, fmt.Errorf("add bias failed: %w", err)
		}
	}
	return output, nil, nil, nil
}

// EmbeddingLayer implements token embeddings
type EmbeddingLayer struct {
	name   string
	weight *tensor.Tensor
	mu     sync.RWMutex
}

func NewEmbeddingLayer(name string) *EmbeddingLayer {
	return &EmbeddingLayer{name: name}
}

func (l *EmbeddingLayer) Name() string { return l.name }

func (l *EmbeddingLayer) Load(state map[string]*tensor.Tensor) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	for k, v := range state {
		if contains(k, l.name) && contains(k, ".weight") {
			l.weight = v
			return nil
		}
	}
	return fmt.Errorf("embedding weight not found in state for layer %s", l.name)
}

func (l *EmbeddingLayer) Unload() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.weight = nil
	return nil
}

func (l *EmbeddingLayer) Forward(input, pastK, pastV *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	l.mu.RLock()
	weight := l.weight
	l.mu.RUnlock()
	if weight == nil {
		return nil, nil, nil, fmt.Errorf("embedding weight not loaded")
	}
	out, err := embedLookup(input, weight)
	return out, nil, nil, err
}

// LayerNorm implements RMS/Layer normalization
type LayerNorm struct {
	name      string
	config    *Config
	weight    *tensor.Tensor
	bias      *tensor.Tensor
	epsilon   float32
	isRMSNorm bool
	mu        sync.RWMutex
}

func NewLayerNorm(name string, config *Config) *LayerNorm {
	return &LayerNorm{
		name:      name,
		config:    config,
		epsilon:   1e-6,
		isRMSNorm: true,
	}
}

func (l *LayerNorm) SetRMSNorm(v bool)    { l.isRMSNorm = v }
func (l *LayerNorm) Name() string          { return l.name }

func (l *LayerNorm) Load(state map[string]*tensor.Tensor) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	for k, v := range state {
		if contains(k, l.name) {
			if contains(k, ".weight") {
				l.weight = v
			}
			if contains(k, ".bias") {
				l.bias = v
			}
		}
	}
	return nil
}

func (l *LayerNorm) Unload() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.weight = nil
	l.bias = nil
	return nil
}

func (l *LayerNorm) Forward(input, pastK, pastV *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	l.mu.RLock()
	w, b := l.weight, l.bias
	l.mu.RUnlock()
	if l.isRMSNorm {
		out, err := rmsNorm(input, w, l.epsilon)
		return out, nil, nil, err
	}
	out, err := layerNorm(input, w, b, l.epsilon)
	return out, nil, nil, err
}

// AttentionLayer implements multi-head self-attention
type AttentionLayer struct {
	name                  string
	config                *Config
	qWeight, kWeight      *tensor.Tensor
	vWeight, oWeight      *tensor.Tensor
	qBias, kBias          *tensor.Tensor
	vBias, oBias          *tensor.Tensor
	numHeads, numKVHeads  int
	headDim               int
	mu                    sync.RWMutex
}

func NewAttentionLayer(name string, config *Config) *AttentionLayer {
	return &AttentionLayer{
		name:       name,
		config:     config,
		numHeads:   32,
		numKVHeads: 32,
		headDim:    128,
	}
}

func (l *AttentionLayer) Name() string { return l.name }

func (l *AttentionLayer) Load(state map[string]*tensor.Tensor) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	for k, v := range state {
		if !contains(k, l.name) {
			continue
		}
		switch {
		case contains(k, "q_proj") || contains(k, "query"):
			if contains(k, "weight") {
				l.qWeight = v
				if len(v.Shape) >= 2 && l.numHeads > 0 {
					l.headDim = v.Shape[0] / l.numHeads
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
		case contains(k, "o_proj") || contains(k, "out_proj") || contains(k, "output_layer"):
			if contains(k, "weight") {
				l.oWeight = v
			} else if contains(k, "bias") {
				l.oBias = v
			}
		}
	}
	return nil
}

func (l *AttentionLayer) Unload() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.qWeight, l.kWeight, l.vWeight, l.oWeight = nil, nil, nil, nil
	l.qBias, l.kBias, l.vBias, l.oBias = nil, nil, nil, nil
	return nil
}

func (l *AttentionLayer) Forward(input, pastK, pastV *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	l.mu.RLock()
	qW, kW, vW, oW := l.qWeight, l.kWeight, l.vWeight, l.oWeight
	qB, kB, vB, oB := l.qBias, l.kBias, l.vBias, l.oBias
	numHeads, headDim := l.numHeads, l.headDim
	l.mu.RUnlock()

	if qW == nil || kW == nil || vW == nil || oW == nil {
		return nil, nil, nil, fmt.Errorf("attention weights not fully loaded for %s", l.name)
	}

	q, err := matmulTransposed(input, qW)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("matmul failed: %w", err)
	}
	k, err := matmulTransposed(input, kW)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("matmul failed: %w", err)
	}
	v, err := matmulTransposed(input, vW)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("matmul failed: %w", err)
	}

	if qB != nil {
		if q, err = addBias(q, qB); err != nil {
			return nil, nil, nil, fmt.Errorf("add bias failed: %w", err)
		}
	}
	if kB != nil {
		if k, err = addBias(k, kB); err != nil {
			return nil, nil, nil, fmt.Errorf("add bias failed: %w", err)
		}
	}
	if vB != nil {
		if v, err = addBias(v, vB); err != nil {
			return nil, nil, nil, fmt.Errorf("add bias failed: %w", err)
		}
	}

	batchSize := input.Shape[0]
	seqLen := input.Shape[1]

	// [B, S, H*D] -> [B, S, H, D]
	q = reshape4D(q, batchSize, seqLen, numHeads, headDim)
	k = reshape4D(k, batchSize, seqLen, numHeads, headDim)
	v = reshape4D(v, batchSize, seqLen, numHeads, headDim)

	// -> [B, H, S, D]
	if q, err = q.Transpose(1, 2); err != nil {
		return nil, nil, nil, err
	}
	if k, err = k.Transpose(1, 2); err != nil {
		return nil, nil, nil, err
	}
	if v, err = v.Transpose(1, 2); err != nil {
		return nil, nil, nil, err
	}

	// KV Caching (FIX-013)
	var newK, newV *tensor.Tensor
	if pastK != nil && pastV != nil {
		newK, err = concatTensorsOnSeqDim(pastK, k)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("KV concat K: %w", err)
		}
		newV, err = concatTensorsOnSeqDim(pastV, v)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("KV concat V: %w", err)
		}
		k = newK
		v = newV
	} else {
		newK = k
		newV = v
	}

	attnOut, err := scaledDotProductAttention(q, k, v, numHeads, headDim)
	if err != nil {
		return nil, nil, nil, err
	}

	// [B, H, S, D] -> [B, S, H, D]
	if attnOut, err = attnOut.Transpose(1, 2); err != nil {
		return nil, nil, nil, err
	}
	attnOut = reshape3D(attnOut, batchSize, seqLen, numHeads*headDim)

	output, err := matmulTransposed(attnOut, oW)
	if err != nil {
		return nil, nil, nil, err
	}
	if oB != nil {
		if output, err = addBias(output, oB); err != nil {
			return nil, nil, nil, err
		}
	}
	return output, newK, newV, nil
}

// FFNLayer implements feed-forward network
type FFNLayer struct {
	name                          string
	config                        *Config
	gateWeight, upWeight          *tensor.Tensor
	downWeight                    *tensor.Tensor
	gateBias, upBias, downBias    *tensor.Tensor
	mu                            sync.RWMutex
}

func NewFFNLayer(name string, config *Config) *FFNLayer {
	return &FFNLayer{name: name, config: config}
}

func (l *FFNLayer) Name() string { return l.name }

func (l *FFNLayer) Load(state map[string]*tensor.Tensor) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	for k, v := range state {
		if !contains(k, l.name) {
			continue
		}
		switch {
		case contains(k, "gate_proj") || contains(k, "w1") || contains(k, "fc1"):
			if contains(k, "weight") {
				l.gateWeight = v
			} else if contains(k, "bias") {
				l.gateBias = v
			}
		case contains(k, "up_proj") || contains(k, "w3"):
			if contains(k, "weight") {
				l.upWeight = v
			} else if contains(k, "bias") {
				l.upBias = v
			}
		case contains(k, "down_proj") || contains(k, "w2") || contains(k, "fc2"):
			if contains(k, "weight") {
				l.downWeight = v
			} else if contains(k, "bias") {
				l.downBias = v
			}
		}
	}
	return nil
}

func (l *FFNLayer) Unload() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.gateWeight, l.upWeight, l.downWeight = nil, nil, nil
	l.gateBias, l.upBias, l.downBias = nil, nil, nil
	return nil
}

func (l *FFNLayer) Forward(input, pastK, pastV *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	l.mu.RLock()
	gW, uW, dW := l.gateWeight, l.upWeight, l.downWeight
	gB, uB, dB := l.gateBias, l.upBias, l.downBias
	l.mu.RUnlock()

	if uW == nil || dW == nil {
		return nil, nil, nil, fmt.Errorf("FFN weights not fully loaded for %s", l.name)
	}

	var hidden *tensor.Tensor
	var err error

	if gW != nil {
		gate, err := matmulTransposed(input, gW)
		if err != nil {
			return nil, nil, nil, err
		}
		if gB != nil {
			if gate, err = addBias(gate, gB); err != nil {
				return nil, nil, nil, err
			}
		}
		gate = silu(gate)
		up, err := matmulTransposed(input, uW)
		if err != nil {
			return nil, nil, nil, err
		}
		if uB != nil {
			if up, err = addBias(up, uB); err != nil {
				return nil, nil, nil, err
			}
		}
		hidden, err = mulElemwise(gate, up)
		if err != nil {
			return nil, nil, nil, err
		}
	} else {
		hidden, err = matmulTransposed(input, uW)
		if err != nil {
			return nil, nil, nil, err
		}
		if uB != nil {
			if hidden, err = addBias(hidden, uB); err != nil {
				return nil, nil, nil, err
			}
		}
		hidden = gelu(hidden)
	}

	output, err := matmulTransposed(hidden, dW)
	if err != nil {
		return nil, nil, nil, err
	}
	if dB != nil {
		if output, err = addBias(output, dB); err != nil {
			return nil, nil, nil, err
		}
	}
	return output, nil, nil, nil
}

// ─── Math Helpers ────────────────────────────────────────────────────────────────

func matmulTransposed(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	return qkernel.SGEMM(A, B, 1.0, 0.0)
}

func addBias(A, bias *tensor.Tensor) (*tensor.Tensor, error) {
	aData := A.Data.([]float32)
	bData := bias.Data.([]float32)
	M := A.Size() / bias.Size()
	N := bias.Size()
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			aData[i*N+j] += bData[j]
		}
	}
	return A, nil
}

func embedLookup(input *tensor.Tensor, weight *tensor.Tensor) (*tensor.Tensor, error) {
	if input == nil || weight == nil {
		return nil, fmt.Errorf("embedLookup: nil input or weight")
	}
	batchSize := input.Shape[0]
	seqLen := input.Shape[1]
	hiddenSize := weight.Shape[1]
	out := tensor.NewTensor([]int{batchSize, seqLen, hiddenSize}, tensor.Float32)
	outData := out.Data.([]float32)
	weightData := weight.Data.([]float32)
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			var tokenID int
			switch d := input.Data.(type) {
			case []int64:
				tokenID = int(d[b*seqLen+s])
			case []int32:
				tokenID = int(d[b*seqLen+s])
			case []int:
				tokenID = d[b*seqLen+s]
			default:
				continue
			}
			if tokenID < 0 || tokenID >= weight.Shape[0] {
				continue
			}
			srcStart := tokenID * hiddenSize
			dstStart := (b*seqLen + s) * hiddenSize
			copy(outData[dstStart:dstStart+hiddenSize], weightData[srcStart:srcStart+hiddenSize])
		}
	}
	return out, nil
}

func rmsNorm(input, weight *tensor.Tensor, epsilon float32) (*tensor.Tensor, error) {
	if input == nil {
		return nil, fmt.Errorf("rmsNorm: nil input")
	}
	data, ok := input.Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("rmsNorm: input is not Float32")
	}
	N := input.Shape[len(input.Shape)-1]
	M := input.Size() / N
	var wData []float32
	if weight != nil {
		if wd, ok2 := weight.Data.([]float32); ok2 {
			wData = wd
		}
	}
	outF32 := tensor.NewTensor(input.Shape, tensor.Float32)
	outData := outF32.Data.([]float32)
	for i := 0; i < M; i++ {
		row := data[i*N : (i+1)*N]
		var sumSq float32
		for _, v := range row {
			sumSq += v * v
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(N) + epsilon)))
		for j := 0; j < N; j++ {
			norm := row[j] / rms
			if wData != nil && j < len(wData) {
				norm *= wData[j]
			}
			outData[i*N+j] = norm
		}
	}
	return outF32, nil
}

func layerNorm(input, weight, bias *tensor.Tensor, epsilon float32) (*tensor.Tensor, error) {
	if input == nil {
		return nil, fmt.Errorf("layerNorm: nil input")
	}
	data, ok := input.Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("layerNorm: input is not Float32")
	}
	N := input.Shape[len(input.Shape)-1]
	M := input.Size() / N
	var wData, bData []float32
	if weight != nil {
		if wd, ok2 := weight.Data.([]float32); ok2 {
			wData = wd
		}
	}
	if bias != nil {
		if bd, ok2 := bias.Data.([]float32); ok2 {
			bData = bd
		}
	}
	outF32 := tensor.NewTensor(input.Shape, tensor.Float32)
	outData := outF32.Data.([]float32)
	for i := 0; i < M; i++ {
		row := data[i*N : (i+1)*N]
		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= float32(N)
		var variance float32
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(N)
		std := float32(math.Sqrt(float64(variance + epsilon)))
		for j := 0; j < N; j++ {
			norm := (row[j] - mean) / std
			if wData != nil && j < len(wData) {
				norm *= wData[j]
			}
			if bData != nil && j < len(bData) {
				norm += bData[j]
			}
			outData[i*N+j] = norm
		}
	}
	return outF32, nil
}

func scaledDotProductAttention(q, k, v *tensor.Tensor, numHeads, headDim int) (*tensor.Tensor, error) {
	// FIX-016: ensure all inputs are Float32.
	if q.DType != tensor.Float32 {
		q = q.ToFloat32()
	}
	if k.DType != tensor.Float32 {
		k = k.ToFloat32()
	}
	if v.DType != tensor.Float32 {
		v = v.ToFloat32()
	}
	qData := q.Data.([]float32)
	kData := k.Data.([]float32)
	vData := v.Data.([]float32)

	b := q.Shape[0]
	h := q.Shape[1]
	sq := q.Shape[2]
	sk := k.Shape[2]
	d := headDim

	out := tensor.NewTensor([]int{b, h, sq, d}, tensor.Float32)
	outData := out.Data.([]float32)

	scale := float32(1.0 / math.Sqrt(float64(d)))

	// For each batch and head
	for bi := 0; bi < b; bi++ {
		for hi := 0; hi < h; hi++ {
			qOff := (bi*h + hi) * sq * d
			kOff := (bi*h + hi) * sk * d
			vOff := (bi*h + hi) * sk * d
			outOff := (bi*h + hi) * sq * d

			// For each query token
			for i := 0; i < sq; i++ {
				scores := make([]float32, sk)
				var maxScore float32 = float32(math.Inf(-1))

				// Q * K^T
				for j := 0; j < sk; j++ {
					var sum float32
					for m := 0; m < d; m++ {
						sum += qData[qOff+i*d+m] * kData[kOff+j*d+m]
					}
					sum *= scale
					scores[j] = sum
					if sum > maxScore {
						maxScore = sum
					}
				}

				// Softmax
				var expSum float32
				for j := 0; j < sk; j++ {
					exp := float32(math.Exp(float64(scores[j] - maxScore)))
					scores[j] = exp
					expSum += exp
				}
				for j := 0; j < sk; j++ {
					scores[j] /= expSum
				}

				// Scores * V
				for m := 0; m < d; m++ {
					var sum float32
					for j := 0; j < sk; j++ {
						sum += scores[j] * vData[vOff+j*d+m]
					}
					outData[outOff+i*d+m] = sum
				}
			}
		}
	}
	return out, nil
}

func silu(x *tensor.Tensor) *tensor.Tensor {
	data := x.Data.([]float32)
	for i, v := range data {
		data[i] = v * (1.0 / (1.0 + float32(math.Exp(float64(-v)))))
	}
	return x
}

func gelu(x *tensor.Tensor) *tensor.Tensor {
	data := x.Data.([]float32)
	for i, v := range data {
		data[i] = 0.5 * v * (1.0 + float32(math.Erf(float64(v)/math.Sqrt(2.0))))
	}
	return x
}

func mulElemwise(A, B *tensor.Tensor) (*tensor.Tensor, error) {
	if A == nil || B == nil {
		return nil, fmt.Errorf("mulElemwise: nil input")
	}
	if A.DType != tensor.Float32 {
		A = A.ToFloat32()
	}
	if B.DType != tensor.Float32 {
		B = B.ToFloat32()
	}
	aData := A.Data.([]float32)
	bData := B.Data.([]float32)
	if len(aData) != len(bData) {
		return nil, fmt.Errorf("mulElemwise: size mismatch %d vs %d", len(aData), len(bData))
	}
	out := tensor.NewTensor(A.Shape, tensor.Float32)
	outData := out.Data.([]float32)
	for i := range aData {
		outData[i] = aData[i] * bData[i]
	}
	return out, nil
}

func concatTensorsOnSeqDim(past, current *tensor.Tensor) (*tensor.Tensor, error) {
	if past == nil {
		return current, nil
	}
	if current == nil {
		return past, nil
	}
	if len(past.Shape) != 4 || len(current.Shape) != 4 {
		return nil, fmt.Errorf("concatTensorsOnSeqDim: expected 4D tensors, got %v and %v", past.Shape, current.Shape)
	}
	b := past.Shape[0]
	h := past.Shape[1]
	s1 := past.Shape[2]
	s2 := current.Shape[2]
	d := past.Shape[3]
	totalS := s1 + s2
	out := tensor.NewTensor([]int{b, h, totalS, d}, tensor.Float32)
	outData := out.Data.([]float32)
	pData := past.Data.([]float32)
	cData := current.Data.([]float32)
	for i := 0; i < b*h; i++ {
		copy(outData[i*totalS*d:i*totalS*d+s1*d], pData[i*s1*d:(i+1)*s1*d])
		copy(outData[i*totalS*d+s1*d:(i+1)*totalS*d], cData[i*s2*d:(i+1)*s2*d])
	}
	return out, nil
}

func contains(s, substr string) bool {
	return qkernel.Contains(s, substr)
}

func reshape4D(t *tensor.Tensor, b, s, h, d int) *tensor.Tensor {
	return t.Reshape([]int{b, s, h, d})
}

func reshape3D(t *tensor.Tensor, b, s, d int) *tensor.Tensor {
	return t.Reshape([]int{b, s, d})
}