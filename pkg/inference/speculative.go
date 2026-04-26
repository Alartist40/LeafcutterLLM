// Package inference — speculative.go
//
// SpeculativeEngine runs two models concurrently:
//   - Draft model: small, fast — speculatively generates K tokens.
//   - Target model: large, accurate — verifies the draft in a single batched pass.
//
// This exploits Go's goroutine model to overlap I/O and compute.
package inference

import (
	"context"
	"errors"
	"sync"
)

type Token = int

type Model interface {
	Generate(ctx context.Context, prompt []Token, maxTokens int, onToken func(Token)) ([]Token, error)
}

type verifyResult struct {
	accepted []Token // verified tokens
	bonus    Token   // one resampled token appended after first rejection
	err      error
}

// ─── Config ──────────────────────────────────────────────────────────────

type SpecConfig struct {
	DraftLen        int
	MaxTokens       int
	Temperature     float32
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

type SpeculativeEngine struct {
	draft  Model
	target Model
	cfg    SpecConfig
	mu     sync.Mutex
}

func NewSpeculativeEngine(draft, target Model, cfg SpecConfig) (*SpeculativeEngine, error) {
	if draft == nil || target == nil {
		return nil, errors.New("speculative: draft and target models must not be nil")
	}

	return &SpeculativeEngine{
		draft:  draft,
		target: target,
		cfg:    cfg,
	}, nil
}

func (se *SpeculativeEngine) Generate(
	ctx context.Context,
	prompt []Token,
	onToken func(Token),
) ([]Token, error) {
	// FIX-019: Removed se.mu.Lock()/Unlock() — draft/target models are immutable after construction.

	tokens := make([]Token, len(prompt))
	copy(tokens, prompt)

	draftCh := make(chan []Token, se.cfg.DraftQueueDepth)
	verifyCh := make(chan verifyResult, se.cfg.DraftQueueDepth)

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup

	// ─── Draft Loop (Goroutine A) ─────────────────────────────────────────────
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(draftCh)

		current := tokens
		for len(current) < se.cfg.MaxTokens {
			select {
			case <-childCtx.Done():
				return
			default:
				drafted, err := se.draft.Generate(childCtx, current, se.cfg.DraftLen, nil)
				if err != nil {
					return
				}
				if len(drafted) <= len(current) {
					return
				}
				newTokens := drafted[len(current):]

				select {
				case draftCh <- newTokens:
					current = append(current, newTokens...)
				case <-childCtx.Done():
					return
				}
			}
		}
	}()

	// ─── Verification Loop (Goroutine B) ────────────────────────────────────────
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(verifyCh)

		current := tokens
		for {
			select {
			case <-childCtx.Done():
				return
			case drafted, ok := <-draftCh:
				if !ok {
					return
				}

				result := se.verifyBatch(childCtx, current, drafted)
				select {
				case verifyCh <- result:
					if result.err != nil {
						return
					}
					current = append(current, result.accepted...)
					if result.bonus > 0 { // FIX-018
						current = append(current, result.bonus)
					}
				case <-childCtx.Done():
					return
				}
			}
		}
	}()

	// ─── Main Aggregator ────────────────────────────────────────────────────────
	for {
		select {
		case <-ctx.Done():
			return tokens, ctx.Err()
		case res, ok := <-verifyCh:
			if !ok {
				return tokens, nil
			}
			if res.err != nil {
				return tokens, res.err
			}

			for _, t := range res.accepted {
				tokens = append(tokens, t)
				if onToken != nil {
					onToken(t)
				}
			}
			if res.bonus > 0 { // FIX-018
				tokens = append(tokens, res.bonus)
				if onToken != nil {
					onToken(res.bonus)
				}
			}

			if len(tokens) >= se.cfg.MaxTokens {
				return tokens, nil
			}
		}
	}
}

func (se *SpeculativeEngine) verifyBatch(ctx context.Context, confirmed, drafted []Token) verifyResult {
	if se.draft == nil {
		return verifyResult{err: errors.New("draft model is nil, skipping verification")}
	}

	if _, isReal := se.draft.(*Engine); isReal {
		// Prevent the real engine from being used as a draft model
		return verifyResult{err: errors.New("draft model cannot be the Real Engine, skipping verification")}
	}

	targetOutput, err := se.target.Generate(ctx, confirmed, len(drafted)+1, nil)
	if err != nil {
		return verifyResult{err: err}
	}
	
	if len(targetOutput) <= len(confirmed) {
		return verifyResult{err: errors.New("target model failed to generate tokens")}
	}
	
	newTargetTokens := targetOutput[len(confirmed):]
	
	acceptedCount := 0
	for i := 0; i < len(drafted) && i < len(newTargetTokens); i++ {
		// In a real implementation with specLog, we would compute corrected probabilities here.
		// For now, exact matching simulates the verification and scoring logic.
		if drafted[i] == newTargetTokens[i] {
			acceptedCount++
		} else {
			break
		}
	}
	
	// Bonus Logic: calculate a corrected distribution (Target - Draft) and sample bonus.
	// We simulate the bonus resampling by taking the first target token that diverged.
	bonusIdx := acceptedCount
	if bonusIdx >= len(newTargetTokens) {
		bonusIdx = len(newTargetTokens) - 1
	}
	bonus := newTargetTokens[bonusIdx]

	return verifyResult{
		accepted: drafted[:acceptedCount],
		bonus:    bonus,
	}
}
