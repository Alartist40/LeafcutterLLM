// Package inference — speculative.go
//
// SpeculativeEngine runs two models concurrently:
//   - Draft model: small, fast — speculatively generates K tokens.
//   - Target model: large, accurate — verifies the draft in a single batched pass.
//
// This exploits Go's goroutine model to overlap I/O and compute in a way that
// is fundamentally impossible in Python (GIL prevents true CPU parallelism).
//
// Verification algorithm (simplified Speculative Decoding, Chen et al. 2023):
//   1. Draft generates tokens [d_1 … d_K].
//   2. Target evaluates all K tokens in one forward pass (batch).
//   3. For each position i, if target's argmax == d_i → accept; else → reject
//      and resample from the corrected distribution, discarding d_{i+1..K}.
//   4. Accepted tokens are flushed to the caller via a channel.

package inference

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
)

// ─── Token type ──────────────────────────────────────────────────────────────

// Token is a vocabulary index.
type Token = int

// ─── Interfaces ──────────────────────────────────────────────────────────────

// Model is implemented by any engine (small or large) that can produce logits.
type Model interface {
	// GenerateDraft produces up to draftLen speculative tokens starting from context.
	// Returns the tokens and per-token float32 probability distributions (length vocab).
	GenerateDraft(ctx context.Context, context []Token, draftLen int) (tokens []Token, probs [][]float32, err error)

	// VerifyBatch scores a batch of tokens in a single forward pass.
	// context is the confirmed prefix; candidates is the draft sequence.
	// Returns per-position logit distributions shaped [len(candidates), vocabSize].
	VerifyBatch(ctx context.Context, context []Token, candidates []Token) (logits [][]float32, err error)

	// VocabSize returns the vocabulary size.
	VocabSize() int
}

// ─── Wire types between goroutines ───────────────────────────────────────────

type draftResult struct {
	context    []Token
	tokens     []Token
	probs      [][]float32 // draft model probability per token
	err        error
}

type verifyResult struct {
	accepted []Token // verified tokens
	bonus    Token   // one resampled token appended after first rejection
	err      error
}

// ─── Config ──────────────────────────────────────────────────────────────────

// SpecConfig holds hyperparameters for speculative decoding.
type SpecConfig struct {
	// DraftLen is how many tokens the draft model speculatively generates.
	DraftLen int

	// MaxTokens is the total number of tokens to generate.
	MaxTokens int

	// Temperature for sampling (0 = greedy).
	Temperature float32

	// DraftQueueDepth is the channel buffer depth between draft and target.
	DraftQueueDepth int
}

func DefaultSpecConfig() SpecConfig {
	return SpecConfig{
		DraftLen:        5,
		MaxTokens:       256,
		Temperature:     1.0,
		DraftQueueDepth: 2,
	}
}

// ─── SpeculativeEngine ───────────────────────────────────────────────────────

// SpeculativeEngine runs a draft model and a target model concurrently,
// using Go channels to pipeline token generation and verification.
type SpeculativeEngine struct {
	draft  Model
	target Model
	cfg    SpecConfig
	mu     sync.Mutex
}

// NewSpeculativeEngine creates a new SpeculativeEngine.
// draft should be a small/fast model; target should be the large/accurate model.
func NewSpeculativeEngine(draft, target Model, cfg SpecConfig) (*SpeculativeEngine, error) {
	if draft == nil || target == nil {
		return nil, errors.New("speculative: draft and target models must not be nil")
	}
	if draft.VocabSize() != target.VocabSize() {
		return nil, fmt.Errorf("speculative: vocab size mismatch: draft=%d target=%d",
			draft.VocabSize(), target.VocabSize())
	}
	if cfg.DraftLen <= 0 {
		cfg.DraftLen = 5
	}
	if cfg.DraftQueueDepth <= 0 {
		cfg.DraftQueueDepth = 2
	}
	return &SpeculativeEngine{draft: draft, target: target, cfg: cfg}, nil
}

// Generate produces up to cfg.MaxTokens tokens, writing each accepted token
// to the onToken callback as soon as it is verified.
//
// Concurrency model:
//   - runDraftLoop goroutine: continually calls draft.GenerateDraft and sends
//     results into draftCh.
//   - runTargetLoop (main goroutine): reads from draftCh, calls target.VerifyBatch,
//     and flushes accepted tokens.
//
// The two loops communicate via buffered channels, allowing the draft model to
// run ahead while the target model is verifying the previous batch.
func (se *SpeculativeEngine) Generate(
	ctx context.Context,
	prompt []Token,
	onToken func(Token),
) ([]Token, error) {
	se.mu.Lock()
	defer se.mu.Unlock()

	cfg := se.cfg

	confirmed := make([]Token, len(prompt))
	copy(confirmed, prompt)

	totalGenerated := 0

	// cancelDraft must always be called before every return.
	// We use a wrapper so relaunching can replace it safely.
	var currentCancel context.CancelFunc
	startDraft := func(budget int) chan draftResult {
		if currentCancel != nil {
			currentCancel()
		}
		ch := make(chan draftResult, cfg.DraftQueueDepth)
		dCtx, cancel := context.WithCancel(ctx)
		currentCancel = cancel
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			se.runDraftLoop(dCtx, confirmed, cfg.DraftLen, budget, ch, &wg)
		}()
		return ch
	}
	defer func() {
		if currentCancel != nil {
			currentCancel()
		}
	}()

	draftCh := startDraft(cfg.MaxTokens)
	var allGenerated []Token

	for totalGenerated < cfg.MaxTokens {
		select {
		case <-ctx.Done():
			return allGenerated, ctx.Err()

		case dr, ok := <-draftCh:
			if !ok {
				return allGenerated, nil
			}
			if dr.err != nil {
				return allGenerated, fmt.Errorf("draft error: %w", dr.err)
			}

			accepted, bonus, err := se.verify(ctx, dr.context, dr.tokens, dr.probs)
			if err != nil {
				return allGenerated, fmt.Errorf("target verify error: %w", err)
			}

			for _, tok := range accepted {
				if onToken != nil {
					onToken(tok)
				}
				allGenerated = append(allGenerated, tok)
				confirmed = append(confirmed, tok)
				totalGenerated++
				if totalGenerated >= cfg.MaxTokens {
					break
				}
			}

			if bonus >= 0 && totalGenerated < cfg.MaxTokens {
				if onToken != nil {
					onToken(bonus)
				}
				allGenerated = append(allGenerated, bonus)
				confirmed = append(confirmed, bonus)
				totalGenerated++
			}

			if totalGenerated >= cfg.MaxTokens {
				return allGenerated, nil
			}

			// Relaunch draft from updated confirmed context.
			draftCh = startDraft(cfg.MaxTokens - totalGenerated)
		}
	}

	return allGenerated, nil
}

// runDraftLoop continuously generates draft batches and sends them on ch.
// It closes ch when the context is cancelled or the budget is exhausted.
func (se *SpeculativeEngine) runDraftLoop(
	ctx context.Context,
	initialContext []Token,
	draftLen, budget int,
	ch chan<- draftResult,
	wg *sync.WaitGroup,
) {
	defer wg.Done()
	defer close(ch)

	localCtx := make([]Token, len(initialContext))
	copy(localCtx, initialContext)

	remaining := budget
	for remaining > 0 {
		select {
		case <-ctx.Done():
			return
		default:
		}

		k := draftLen
		if k > remaining {
			k = remaining
		}

		tokens, probs, err := se.draft.GenerateDraft(ctx, localCtx, k)
		dr := draftResult{
			context: localCtx,
			tokens:  tokens,
			probs:   probs,
			err:     err,
		}

		select {
		case <-ctx.Done():
			return
		case ch <- dr:
		}

		if err != nil {
			return
		}

		// Optimistically extend local context (will be corrected by target).
		localCtx = append(localCtx, tokens...)
		remaining -= len(tokens)
	}
}

// verify runs the target model and implements the speculative decoding
// acceptance/rejection criterion.
//
// Returns accepted tokens and a bonus token (−1 if not applicable).
func (se *SpeculativeEngine) verify(
	ctx context.Context,
	prefix, draftTokens []Token,
	draftProbs [][]float32,
) (accepted []Token, bonus Token, err error) {
	if len(draftTokens) == 0 {
		return nil, -1, nil
	}

	targetLogits, err := se.target.VerifyBatch(ctx, prefix, draftTokens)
	if err != nil {
		return nil, -1, err
	}

	bonus = -1

	for i, dt := range draftTokens {
		if i >= len(targetLogits) {
			break
		}

		targetProbs := softmaxSlice(targetLogits[i], se.cfg.Temperature)
		p_target := targetProbs[dt]

		var p_draft float32 = 1.0
		if i < len(draftProbs) && dt < len(draftProbs[i]) {
			p_draft = draftProbs[i][dt]
		}

		// Acceptance ratio r = min(1, p_target / p_draft)
		ratio := float32(1.0)
		if p_draft > 1e-9 {
			ratio = p_target / p_draft
			if ratio > 1.0 {
				ratio = 1.0
			}
		}

		if rand.Float32() <= ratio {
			accepted = append(accepted, dt)
		} else {
			// Rejection: resample from corrected distribution p_target - p_draft
			corrected := make([]float32, len(targetProbs))
			for j := range corrected {
				var pd float32
				if i < len(draftProbs) && j < len(draftProbs[i]) {
					pd = draftProbs[i][j]
				}
				v := targetProbs[j] - pd
				if v < 0 {
					v = 0
				}
				corrected[j] = v
			}
			bonus = sampleFromProbs(corrected)
			return accepted, bonus, nil
		}
	}

	// All tokens accepted — sample one bonus token from target's last position.
	if len(targetLogits) > 0 {
		last := softmaxSlice(targetLogits[len(targetLogits)-1], se.cfg.Temperature)
		bonus = sampleFromProbs(last)
	}
	return accepted, bonus, nil
}

// ─── Sampling helpers ────────────────────────────────────────────────────────

// softmaxSlice applies temperature-scaled softmax to logits, returning probabilities.
func softmaxSlice(logits []float32, temp float32) []float32 {
	if temp <= 0 {
		temp = 1.0
	}
	probs := make([]float32, len(logits))
	var maxV float32 = -1e38
	for _, v := range logits {
		if v > maxV {
			maxV = v
		}
	}
	var sum float32
	for i, v := range logits {
		e := float32(math.Exp(float64((v - maxV) / temp)))
		probs[i] = e
		sum += e
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	}
	return probs
}

// sampleFromProbs draws a token index from an unnormalized distribution.
func sampleFromProbs(probs []float32) Token {
	var total float32
	for _, p := range probs {
		total += p
	}
	if total <= 0 {
		return 0
	}
	r := rand.Float32() * total
	var cumulative float32
	for i, p := range probs {
		cumulative += p
		if r <= cumulative {
			return i
		}
	}
	return len(probs) - 1
}

// ─── Stats ───────────────────────────────────────────────────────────────────

// SpecStats accumulates token acceptance statistics across calls.
type SpecStats struct {
	mu            sync.Mutex
	TotalDraft    int
	TotalAccepted int
}

func (s *SpecStats) Record(draft, accepted int) {
	s.mu.Lock()
	s.TotalDraft += draft
	s.TotalAccepted += accepted
	s.mu.Unlock()
}

func (s *SpecStats) AcceptanceRate() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.TotalDraft == 0 {
		return 0
	}
	return float64(s.TotalAccepted) / float64(s.TotalDraft)
}

// ─── Logging helper ──────────────────────────────────────────────────────────

func specLog(format string, args ...interface{}) {
	log.Printf("[speculative] "+format, args...)
}

var _ = specLog // suppress unused warning
