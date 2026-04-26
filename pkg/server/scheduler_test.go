package server_test

import (
    "context"
	"fmt"
    "sync/atomic"
    "testing"
    "time"
    "github.com/Alartist40/LeafcutterLLM/pkg/server"
)

type countingRunner struct {
    count atomic.Int64
}

func (r *countingRunner) RunBatch(ctx context.Context, batch *server.Batch) error {
    r.count.Add(int64(len(batch.Requests)))
    for _, req := range batch.Requests {
        req.ResultCh <- server.InferResponse{ID: req.ID, Tokens: req.Prompt}
    }
    return nil
}

func TestSchedulerBasic(t *testing.T) {
    runner := &countingRunner{}
    cfg := server.SchedulerConfig{
        MaxBatchSize:    4,
        MaxWaitDuration: 10 * time.Millisecond,
        QueueDepth:      16,
    }
    sched := server.NewScheduler(cfg, runner)
    if err := sched.Start(); err != nil {
        t.Fatalf("Start failed: %v", err)
    }
    defer sched.Stop()

    const N = 8
    results := make([]server.InferResponse, N)
    errs := make([]error, N)

    ctx := context.Background()
    done := make(chan struct{}, N)

    for i := 0; i < N; i++ {
        i := i
        go func() {
            req := &server.InferRequest{
                ID:       fmt.Sprintf("req-%d", i),
                Prompt:   []int{i},
                ResultCh: make(chan server.InferResponse, 1),
            }
            resp, err := sched.SubmitAndWait(ctx, req)
            results[i] = resp
            errs[i] = err
            done <- struct{}{}
        }()
    }

    for i := 0; i < N; i++ {
        select {
        case <-done:
        case <-time.After(5 * time.Second):
            t.Fatalf("timed out waiting for response %d", i)
        }
    }

    for i, err := range errs {
        if err != nil {
            t.Errorf("request %d error: %v", i, err)
        }
    }

    if got := runner.count.Load(); got != int64(N) {
        t.Errorf("expected %d requests processed, got %d", N, got)
    }
}

func TestSchedulerQueueFull(t *testing.T) {
    runner := &countingRunner{}
    cfg := server.SchedulerConfig{
        MaxBatchSize:    1,
        MaxWaitDuration: 1 * time.Hour, // never flush automatically
        QueueDepth:      2,
    }
    sched := server.NewScheduler(cfg, runner)
    sched.Start()
    defer sched.Stop()

    // Fill queue
    for i := 0; i < 2; i++ {
        req := &server.InferRequest{ID: fmt.Sprintf("r%d", i), ResultCh: make(chan server.InferResponse, 1)}
        sched.Submit(req)
    }

    // This one should be dropped
    overflow := &server.InferRequest{ID: "overflow", ResultCh: make(chan server.InferResponse, 1)}
    if err := sched.Submit(overflow); err == nil {
        t.Error("expected error when queue full")
    }
}
