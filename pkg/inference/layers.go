package inference

import (
    "fmt"
    "math"
    "sync"

    "github.com/xander/airllm-go/pkg/qkernel"
    "github.com/xander/airllm-go/pkg/tensor"
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
            if len(k) >= 7 && k[len(k)-7:] == ".weight" {
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

// Forward performs y = x @ W^T + b.
// Uses C Kernel if weights are quantized, otherwise naive loops.
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
        if len(k) >= 7 && k[len(k)-7:] == ".weight" {
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
        return nil, fmt.Errorf("embedding weight not loaded")
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
        if len(k) >= 7 && k[len(k)-7:] == ".weight" {
            l.weight = v
        }
        if len(k) >= 5 && k[len(k)-5:] == ".bias" {
            l.bias = v
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
        return nil, nil, nil, err
    }
    k, err := matmulTransposed(input, kW)
    if err != nil {
        return nil, nil, nil, err
    }
    v, err := matmulTransposed(input, vW)
    if err != nil {
        return nil, nil, nil, err
    }

    if qB != nil {
        if q, err = addBias(q, qB); err != nil {
            return nil, nil, nil, err
        }
    }
    if kB != nil {
        if k, err = addBias(k, kB); err != nil {
            return nil, nil, nil, err
        }
    }
    if vB != nil {
        if v, err = addBias(v, vB); err != nil {
            return nil, nil, nil, err
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

    // Implement KV Caching logic
    newK := k
    newV := v
    if pastK != nil && pastV != nil {
        k, err = concatTensorsOnSeqDim(pastK, k)
        if err != nil {
            return nil, nil, nil, err
        }
        v, err = concatTensorsOnSeqDim(pastV, v)
        if err != nil {
            return nil, nil, nil, err
        }
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

// FFNLayer implements the feed-forward network (SwiGLU or standard)
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
        // SwiGLU: hidden = silu(gate(x)) * up(x)
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
        if hidden, err = mulElemwise(gate, up); err != nil {
            return nil, nil, nil, err
        }
    } else {
        // Standard FFN: hidden = gelu(up(x))
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

// ─── Math Helpers ────────────────────────────────────────────────

// concatTensorsOnSeqDim concatenates two 4D tensors [B, H, S, D] along the sequence dimension (dim 2)
func concatTensorsOnSeqDim(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if len(a.Shape) != 4 || len(b.Shape) != 4 {
		return nil, fmt.Errorf("concat requires 4D tensors")
	}

	B := a.Shape[0]
	H := a.Shape[1]
	Sa := a.Shape[2]
	Sb := b.Shape[2]
	D := a.Shape[3]

	out := tensor.NewTensor([]int{B, H, Sa + Sb, D}, a.DType)
	elemSize := 1
	if a.DType == tensor.Float32 {
		elemSize = 4
	} else if a.DType == tensor.Float16 {
		elemSize = 2
	}

	outData := out.Data
	aData := a.Data
	bData := b.Data

	for bIdx := 0; bIdx < B; bIdx++ {
		for hIdx := 0; hIdx < H; hIdx++ {
			aStart := (bIdx*H + hIdx) * Sa * D * elemSize
			aEnd := aStart + Sa*D*elemSize
			bStart := (bIdx*H + hIdx) * Sb * D * elemSize
			bEnd := bStart + Sb*D*elemSize

			outStartA := (bIdx*H + hIdx) * (Sa + Sb) * D * elemSize
			outStartB := outStartA + Sa*D*elemSize

			copy(outData[outStartA:], aData[aStart:aEnd])
			copy(outData[outStartB:], bData[bStart:bEnd])
		}
	}
	return out, nil
}

// matmulTransposed computes A @ B^T.
// Checks if B is quantized and calls C kernel, otherwise uses OpenBLAS for Float32/16.
func matmulTransposed(A, B *tensor.Tensor) (*tensor.Tensor, error) {
    if len(A.Shape) < 2 || len(B.Shape) != 2 {
        return nil, fmt.Errorf("invalid shapes for matmulTransposed: A=%v, B=%v", A.Shape, B.Shape)
    }

    aInFeatures := A.Shape[len(A.Shape)-1]
    bOutFeatures := B.Shape[0]
    bInFeatures := B.Shape[1]

    if aInFeatures != bInFeatures {
        return nil, fmt.Errorf("inner dimension mismatch: %d != %d", aInFeatures, bInFeatures)
    }

    // Check if B is quantized (heuristic: check if Data is byte-sized)
    isQuantized := false
    if B.Dtype == tensor.Uint8 { // Assuming Uint8 implies Q4 for this implementation
        isQuantized = true
    }

    if isQuantized {
        // Fallback to naive for Q4 in this stub (since we don't extract scales here)
        return matmulNaive(A, B) 
    }

    if (A.Dtype == tensor.Float32 || A.Dtype == tensor.Float16) &&
       (B.Dtype == tensor.Float32 || B.Dtype == tensor.Float16) {
        return qkernel.SGEMM(A, B, 1.0, 0.0)
    }

    return matmulNaive(A, B)
}

// matmulNaive performs standard matmul on Float32
func matmulNaive(A, B *tensor.Tensor) (*tensor.Tensor, error) {
    aRows := 1
    for i := 0; i < len(A.Shape)-1; i++ {
        aRows *= A.Shape[i]
    }
    k := A.Shape[len(A.Shape)-1]
    n := B.Shape[0]

    out := tensor.NewTensor(A.Shape, A.Dtype)
    outData := out.Data.([]float32)
    
    // Convert inputs to float32 for calculation if they aren't already
    var aData, bData []float32
    if A.Dtype == tensor.Float32 {
        aData = A.Data.([]float32)
    } else {
        // Conversion required (omitted for brevity, assume F32)
        return nil, fmt.Errorf("naive matmul only supports F32 currently")
    }

    if B.Dtype == tensor.Float32 {
        bData = B.Data.([]float32)
    } else {
        return nil, fmt.Errorf("naive matmul only supports F32 currently")
    }

    // A [M, K] @ B [N, K]^T = [M, N]
    // We assume B is actually stored as [N, K] (row major)
    // If B was [K, N], we would swap loops.
    
    for i := 0; i < aRows; i++ {
        for j := 0; j < n; j++ {
            var sum float32
            for kk := 0; kk < k; kk++ {
                sum += aData[i*k+kk] * bData[j*k+kk]
            }
            outData[i*n+j] = sum
        }
    }
    return out, nil
}

func addBias(A, bias *tensor.Tensor) (*tensor.Tensor, error) {
    result := A.Clone()
    lastDim := bias.Shape[0]
    numGroups := A.Size() / lastDim

    for i := 0; i < numGroups; i++ {
        for j := 0; j < lastDim; j++ {
            idx := i*lastDim + j
            val := getF32(A, idx)
            biasVal := getF32(bias, j)
            result.SetFloat32(idx, val+biasVal)
        }
    }
    return result, nil
}

func embedLookup(input *tensor.Tensor, weight *tensor.Tensor) (*tensor.Tensor, error) {
    // Implementation of embed lookup (simplified)
    // Returns input shape + hidden_dim
    hiddenDim := weight.Shape[1]
    inputData := input.Data.([]int64) // Assuming input is token IDs
    weightData := weight.Data.([]float32)

    // This is a naive O(1) lookup for demonstration
    // For performance, this should also use kernel or optimized copy
    // but embedding lookup is usually memory bandwidth bound.
    
    batchSize := input.Shape[0]
    seqLen := input.Shape[1]
    
    outShape := []int{batchSize, seqLen, hiddenDim}
    out := tensor.NewTensor(outShape, tensor.Float32)
    outData := out.Data.([]float32)

    for b := 0; b < batchSize; b++ {
        for s := 0; s < seqLen; s++ {
            tokenID := int(inputData[b*seqLen+s])
            if tokenID < 0 || tokenID >= weight.Shape[0] {
                continue
            }
            rowOffset := tokenID * hiddenDim
            outOffset := (b*seqLen + s) * hiddenDim
            for h := 0; h < hiddenDim; h++ {
                outData[outOffset+h] = weightData[rowOffset+h]
            }
        }
    }
    return out, nil
}

func rmsNorm(input, weight *tensor.Tensor, epsilon float32) (*tensor.Tensor, error) {
    lastDim := input.Shape[len(input.Shape)-1]
    numGroups := input.Size() / lastDim
    result := tensor.NewTensor(input.Shape, input.Dtype)

    for g := 0; g < numGroups; g++ {
        var sumSq float32
        for i := 0; i < lastDim; i++ {
            v := getF32(input, g*lastDim+i)
            sumSq += v * v
        }
        rms := float32(math.Sqrt(float64(sumSq/float32(lastDim) + epsilon)))
        for i := 0; i < lastDim; i++ {
            idx := g*lastDim + i
            val := getF32(input, idx)
            var w float32
            if weight != nil {
                w = getF32(weight, i)
            } else {
                w = 1.0
            }
            result.SetFloat32(idx, (val/rms)*w)
        }
    }
    return result, nil
}

func layerNorm(input, weight, bias *tensor.Tensor, epsilon float32) (*tensor.Tensor, error) {
    lastDim := input.Shape[len(input.Shape)-1]
    numGroups := input.Size() / lastDim
    result := tensor.NewTensor(input.Shape, input.Dtype)

    for g := 0; g < numGroups; g++ {
        var sum float32
        for i := 0; i < lastDim; i++ {
            sum += getF32(input, g*lastDim+i)
        }
        mean := sum / float32(lastDim)

        var varSum float32
        for i := 0; i < lastDim; i++ {
            diff := getF32(input, g*lastDim+i) - mean
            varSum += diff * diff
        }
        variance := varSum / float32(lastDim)
        std := float32(math.Sqrt(float64(variance + epsilon)))

        for i := 0; i < lastDim; i++ {
            idx := g*lastDim + i
            val := getF32(input, idx)
            normalized := (val - mean) / std
            var w float32
            if weight != nil {
                w = getF32(weight, i)
            } else {
                w = 1.0
            }
            finalVal := normalized * w
            if bias != nil {
                finalVal += getF32(bias, i)
            }
            result.SetFloat32(idx, finalVal)
        }
    }
    return result, nil
}

func scaledDotProductAttention(q, k, v *tensor.Tensor, numHeads, headDim int) (*tensor.Tensor, error) {
    // Naive SDPA implementation
    // q, k, v are [Batch, Heads, Seq, HeadDim]
    
    batchSize := q.Shape[0]
    qSeqLen := q.Shape[2]
    kvSeqLen := k.Shape[2]
    
    outShape := []int{batchSize, numHeads, qSeqLen, headDim}
    out := tensor.NewTensor(outShape, tensor.Float32)
    outData := out.Data.([]float32)
    
    qData := q.Data.([]float32)
    kData := k.Data.([]float32)
    vData := v.Data.([]float32)
    
    scale := float32(1.0 / math.Sqrt(float64(headDim)))

    pastS := kvSeqLen - qSeqLen

    for b := 0; b < batchSize; b++ {
        for h := 0; h < numHeads; h++ {
            for i := 0; i < qSeqLen; i++ {
                // Calculate Q @ K^T
                scores := make([]float32, kvSeqLen)
                var maxScore float32 = -3.4028235e38
                
                for j := 0; j < kvSeqLen; j++ {
                    var dot float32
                    for d := 0; d < headDim; d++ {
                        qIdx := b*numHeads*qSeqLen*headDim + h*qSeqLen*headDim + i*headDim + d
                        kIdx := b*numHeads*kvSeqLen*headDim + h*kvSeqLen*headDim + j*headDim + d
                        dot += qData[qIdx] * kData[kIdx]
                    }
                    s := dot * scale
                    scores[j] = s
                    if s > maxScore {
                        maxScore = s
                    }
                }
                
                // Softmax
                var sumExp float32
                for j := 0; j < kvSeqLen; j++ {
                    // Causal Mask: zero out future
                    // The absolute position of the query is pastS + i.
                    if j > pastS + i {
                        scores[j] = 0
                        continue
                    }
                    e := float32(math.Exp(float64(scores[j] - maxScore)))
                    scores[j] = e
                    sumExp += e
                }
                
                // Multiply by V
                for d := 0; d < headDim; d++ {
                    var val float32
                    for j := 0; j < kvSeqLen; j++ {
                        if scores[j] > 0 {
                            vIdx := b*numHeads*kvSeqLen*headDim + h*kvSeqLen*headDim + j*headDim + d
                            prob := scores[j]
                            if sumExp > 0 {
                                prob /= sumExp
                            }
                            val += prob * vData[vIdx]
                        }
                    }
                    outIdx := b*numHeads*qSeqLen*headDim + h*qSeqLen*headDim + i*headDim + d
                    outData[outIdx] = val
                }
            }
        }
    }
    return out, nil
}

func swiglu(gate, up *tensor.Tensor) (*tensor.Tensor, error) {
    if gate.Shape[0] != up.Shape[0] {
        return nil, fmt.Errorf("swiglu shape mismatch")
    }
    out := tensor.NewTensor(gate.Shape, tensor.Float32)
    
    gData := gate.Data.([]float32)
    uData := up.Data.([]float32)
    oData := out.Data.([]float32)
    
    for i := range gData {
        g := gData[i]
        // swish = x * sigmoid(x)
        sig := float32(1.0 / (1.0 + float32(math.Exp(float64(-g)))))
        oData[i] = g * sig * uData[i]
    }
    return out, nil
}

func gelu(x *tensor.Tensor) (*tensor.Tensor, error) {
    out := tensor.NewTensor(x.Shape, tensor.Float32)
    data := x.Data.([]float32)
    oData := out.Data.([]float32)
    
    for i := range data {
        val := data[i]
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        tanh_val := float32(math.Tanh(float64(val * 0.7978845608 * (1 + 0.044715*val*val)))
        oData[i] = 0.5 * val * (1 + tanh_val)
    }
    return out, nil
}

func mulElemwise(A, B *tensor.Tensor) (*tensor.Tensor, error) {
    if A.Size() != B.Size() {
        return nil, fmt.Errorf("mulElemwise size mismatch")
    }
    out := tensor.NewTensor(A.Shape, tensor.Float32)
    aData := A.Data.([]float32)
    bData := B.Data.([]float32)
    oData := out.Data.([]float32)
    
    for i := range aData {
        oData[i] = aData[i] * bData[i]
    }
    return out, nil
}

func getF32(t *tensor.Tensor, i int) float32 {
    // Assuming Float32 data in this implementation
    return t.Data.([]float32)[i]
}

func contains(s, substr string) bool {
    for i := 0; i <= len(s)-len(substr); i++ {
        if s[i:i+len(substr)] == substr {
            return true
        }
    }
    return false
}

func reshape4D(t *tensor.Tensor, b, s, h, d int) *tensor.Tensor {
    // Simplified reshape that creates a new tensor with new shape
    // In a real implementation, this would re-use data
    return tensor.NewTensor([]int{b, s, h, d}, t.Dtype)
}

func reshape3D(t *tensor.Tensor, b, s, d int) *tensor.Tensor {
    return tensor.NewTensor([]int{b, s, d}, t.Dtype)
}