// Package server provides the Continuous Batching scheduler.
//
// Continuous Batching (also called "in-flight batching") eliminates the
// stop-the-world per-request processing of naive inference servers.
// Instead of:
//   UserA → UserB → UserC   (sequential)
// We do:
//   [UserA, UserB, UserC] → single batched forward pass
//
// Requests enter a priority FIFO queue.  A background dispatcher goroutine
// groups "ready" requests (those whose prefill is complete) into a batch
// and submits them to the inference engine together.

package server

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// ─── Request / Response types ─────────────────────────────────────────────────

// InferRequest holds a single user's generation request.
type InferRequest struct {
	ID          string
	Prompt      []int          // token IDs
	MaxTokens   int
	Temperature float32
	OnToken     func(int)      // streaming callback — called for each output token
	ResultCh    chan InferResponse
}

// InferResponse is sent back on ResultCh when generation is complete.
type InferResponse struct {
	ID     string
	Tokens []int
	Err    error
	Took   time.Duration
}

// ─── Batch submitted to the model ────────────────────────────────────────────

// Batch groups several requests into one forward call.
type Batch struct {
	Requests []*InferRequest
}

// ─── ModelRunner is the interface the scheduler calls ────────────────────────

// ModelRunner abstracts the underlying engine so the scheduler can be tested
// without a real model.
type ModelRunner interface {
	// RunBatch executes a batch of requests concurrently.
	// It must call req.OnToken for each token as it is generated and
	// send the final result on req.ResultCh.
	RunBatch(ctx context.Context, batch *Batch) error
}

// ─── Scheduler ────────────────────────────────────────────────────────────────

// SchedulerConfig holds tuning parameters for the scheduler.
type SchedulerConfig struct {
	// MaxBatchSize is the maximum number of requests in one batch.
	MaxBatchSize int

	// MaxWaitDuration is how long the dispatcher waits to fill a batch
	// before flushing a partial one.
	MaxWaitDuration time.Duration

	// QueueDepth is the maximum number of pending requests.
	QueueDepth int
}

// DefaultSchedulerConfig returns sensible defaults.
func DefaultSchedulerConfig() SchedulerConfig {
	return SchedulerConfig{
		MaxBatchSize:    8,
		MaxWaitDuration: 20 * time.Millisecond,
		QueueDepth:      256,
	}
}

// Scheduler accepts InferRequests, groups them into batches, and submits
// them to a ModelRunner.
type Scheduler struct {
	cfg     SchedulerConfig
	runner  ModelRunner
	queue   chan *InferRequest
	stop    chan struct{}
	wg      sync.WaitGroup
	started atomic.Bool

	// Metrics
	totalRequests  atomic.Int64
	totalBatches   atomic.Int64
	droppedRequest atomic.Int64
}

// NewScheduler creates a Scheduler backed by runner.
func NewScheduler(cfg SchedulerConfig, runner ModelRunner) *Scheduler {
	return &Scheduler{
		cfg:    cfg,
		runner: runner,
		queue:  make(chan *InferRequest, cfg.QueueDepth),
		stop:   make(chan struct{}),
	}
}

// Start launches the dispatcher goroutine.  Safe to call once.
func (s *Scheduler) Start() error {
	if !s.started.CompareAndSwap(false, true) {
		return fmt.Errorf("scheduler already started")
	}
	s.wg.Add(1)
	go s.dispatch()
	return nil
}

// Stop drains in-flight work and shuts down the dispatcher.
func (s *Scheduler) Stop() {
	close(s.stop)
	s.wg.Wait()
}

// Submit enqueues a request.  Returns an error if the queue is full.
func (s *Scheduler) Submit(req *InferRequest) error {
	if req.ResultCh == nil {
		req.ResultCh = make(chan InferResponse, 1)
	}
	select {
	case s.queue <- req:
		s.totalRequests.Add(1)
		return nil
	default:
		s.droppedRequest.Add(1)
		return fmt.Errorf("scheduler: queue full, request %s dropped", req.ID)
	}
}

// SubmitAndWait is a blocking convenience wrapper around Submit.
func (s *Scheduler) SubmitAndWait(ctx context.Context, req *InferRequest) (InferResponse, error) {
	if req.ResultCh == nil {
		req.ResultCh = make(chan InferResponse, 1)
	}
	if err := s.Submit(req); err != nil {
		return InferResponse{}, err
	}
	select {
	case <-ctx.Done():
		return InferResponse{}, ctx.Err()
	case resp := <-req.ResultCh:
		return resp, resp.Err
	}
}

// Stats returns a snapshot of scheduler metrics.
func (s *Scheduler) Stats() SchedulerStats {
	return SchedulerStats{
		TotalRequests:  s.totalRequests.Load(),
		TotalBatches:   s.totalBatches.Load(),
		DroppedRequest: s.droppedRequest.Load(),
		QueueDepth:     len(s.queue),
	}
}

// SchedulerStats is a point-in-time metrics snapshot.
type SchedulerStats struct {
	TotalRequests  int64
	TotalBatches   int64
	DroppedRequest int64
	QueueDepth     int
}

// ─── Internal dispatcher ──────────────────────────────────────────────────────

// dispatch is the main scheduler loop.
// It collects requests from the queue and builds batches according to two
// criteria:
//   1. Batch reaches MaxBatchSize → flush immediately.
//   2. MaxWaitDuration elapses since the first request in the current batch
//      was received → flush whatever we have.
func (s *Scheduler) dispatch() {
	defer s.wg.Done()

	var pending []*InferRequest
	var batchTimer <-chan time.Time

	flush := func() {
		if len(pending) == 0 {
			return
		}
		batch := &Batch{Requests: pending}
		pending = nil
		batchTimer = nil
		s.totalBatches.Add(1)

		// Run the batch in a separate goroutine so the dispatcher can keep
		// collecting the next batch while this one is computing.
		go func(b *Batch) {
			ctx := context.Background()
			if err := s.runner.RunBatch(ctx, b); err != nil {
				// Send errors to all requests in the batch.
				for _, req := range b.Requests {
					req.ResultCh <- InferResponse{ID: req.ID, Err: err}
				}
			}
		}(batch)
	}

	for {
		select {
		case <-s.stop:
			flush() // Drain any remaining requests.
			return

		case req := <-s.queue:
			pending = append(pending, req)

			// Start the wait timer on the first request of a new batch.
			if len(pending) == 1 {
				batchTimer = time.After(s.cfg.MaxWaitDuration)
			}

			// Flush immediately if batch is full.
			if len(pending) >= s.cfg.MaxBatchSize {
				flush()
			}

		case <-batchTimer:
			flush()
		}
	}
}
