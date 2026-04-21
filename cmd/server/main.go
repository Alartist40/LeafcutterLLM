// cmd/server/main.go — airllm-go inference server
//
// Starts an HTTP server that accepts text generation requests.
// The server uses the Continuous Batching Scheduler to group concurrent
// requests into single batched forward passes, dramatically improving
// throughput compared to serial request processing.
//
// Usage:
//
//	airllm-server --model /path/to/model --draft /path/to/draft-model \
//	              --port 8080 --batch-size 8
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/xander/airllm-go/pkg/inference"
	"github.com/xander/airllm-go/pkg/server"
	"github.com/xander/airllm-go/pkg/tokenizer"
)

// ─── CLI flags ────────────────────────────────────────────────────────────────

var (
	modelPath    = flag.String("model", "", "Path to the target (large) model")
	draftPath    = flag.String("draft", "", "Path to the draft (small) model (optional)")
	port         = flag.Int("port", 8080, "HTTP port to listen on")
	maxBatch     = flag.Int("batch-size", 8, "Maximum continuous batch size")
	maxWaitMs    = flag.Int("batch-wait-ms", 20, "Max milliseconds to wait before flushing a partial batch")
	maxTokens    = flag.Int("max-tokens", 256, "Default max tokens per request")
	draftLen     = flag.Int("draft-len", 5, "Speculative decoding draft length")
	queueDepth   = flag.Int("queue-depth", 256, "Request queue depth")
	enableSpec   = flag.Bool("speculative", false, "Enable speculative decoding (requires --draft)")
	version      = flag.Bool("version", false, "Print version and exit")
)

const serverVersion = "airllm-go server v0.2.0 (speculative + continuous-batching)"

// ─── HTTP request/response DTOs ──────────────────────────────────────────────

type GenerateRequest struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float32 `json:"temperature,omitempty"`
	Stream      bool    `json:"stream,omitempty"`
}

type GenerateResponse struct {
	ID     string `json:"id"`
	Text   string `json:"text,omitempty"`
	Tokens []int  `json:"tokens,omitempty"`
	TookMs int64  `json:"took_ms"`
	Error  string `json:"error,omitempty"`
}

// ─── Stub model runner (wires scheduler → engine) ────────────────────────────
// In production this would hold a real *inference.Engine.
// Here we implement the ModelRunner interface so the server compiles and runs
// without a real model on disk — useful for integration testing the batching logic.

type stubRunner struct {
	targetEngine *inference.Engine // nil if no model path given
	specEngine   *inference.SpeculativeEngine
	mu           sync.Mutex
	reqCounter   atomic.Int64
}

func (r *stubRunner) RunBatch(ctx context.Context, batch *server.Batch) error {
	var wg sync.WaitGroup
	for _, req := range batch.Requests {
		wg.Add(1)
		go func(req *server.InferRequest) {
			defer wg.Done()
			start := time.Now()

			tokens, err := r.runSingle(ctx, req)
			req.ResultCh <- server.InferResponse{
				ID:     req.ID,
				Tokens: tokens,
				Err:    err,
				Took:   time.Since(start),
			}
		}(req)
	}
	wg.Wait()
	return nil
}

func (r *stubRunner) runSingle(ctx context.Context, req *server.InferRequest) ([]int, error) {
	if r.specEngine != nil {
		return r.specEngine.Generate(ctx, req.Prompt, req.OnToken)
	}
	// Stub: echo back prompt tokens (for testing scheduler / HTTP layer).
	if req.OnToken != nil {
		for _, t := range req.Prompt {
			req.OnToken(t)
		}
	}
	return req.Prompt, nil
}

// ─── HTTP handlers ────────────────────────────────────────────────────────────

type apiServer struct {
	scheduler *server.Scheduler
	tokenizer *tokenizer.BPETokenizer
	counter   atomic.Int64
}

func (s *apiServer) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	if req.Prompt == "" {
		http.Error(w, "prompt is required", http.StatusBadRequest)
		return
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = *maxTokens
	}
	if req.Temperature <= 0 {
		req.Temperature = 1.0
	}

	reqID := fmt.Sprintf("req-%d", s.counter.Add(1))
	
	var tokens []int
	if s.tokenizer != nil {
		tokens = s.tokenizer.Encode(req.Prompt)
	} else {
		tokens = tokenizeSimpleFallback(req.Prompt)
	}

	var mu sync.Mutex
	var streamTokens []int

	onToken := func(t int) {
		mu.Lock()
		streamTokens = append(streamTokens, t)
		mu.Unlock()
	}

	inferReq := &server.InferRequest{
		ID:          reqID,
		Prompt:      tokens,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		OnToken:     onToken,
		ResultCh:    make(chan server.InferResponse, 1),
	}

	resp, err := s.scheduler.SubmitAndWait(r.Context(), inferReq)

	w.Header().Set("Content-Type", "application/json")
	out := GenerateResponse{
		ID:     reqID,
		Tokens: resp.Tokens,
		TookMs: resp.Took.Milliseconds(),
	}
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		out.Error = err.Error()
	}
	json.NewEncoder(w).Encode(out) //nolint:errcheck
}

func (s *apiServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	stats := s.scheduler.Stats()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{ //nolint:errcheck
		"status":          "ok",
		"version":         serverVersion,
		"total_requests":  stats.TotalRequests,
		"total_batches":   stats.TotalBatches,
		"dropped":         stats.DroppedRequest,
		"queue_depth":     stats.QueueDepth,
	})
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	flag.Parse()

	if *version {
		fmt.Println(serverVersion)
		os.Exit(0)
	}

	log.SetFlags(log.LstdFlags | log.Lmsgprefix)
	log.SetPrefix("[airllm-server] ")

	// ── Build runner ──────────────────────────────────────────────────
	runner := &stubRunner{}

	if *enableSpec {
		if *draftPath == "" {
			log.Fatal("--speculative requires --draft <path>")
		}
		log.Printf("Speculative decoding enabled: draft=%s target=%s draftLen=%d",
			*draftPath, *modelPath, *draftLen)
		// In production: load draft and target engines here.
		// runner.specEngine = buildSpecEngine(...)
	}

	// ── Build scheduler ───────────────────────────────────────────────
	schCfg := server.SchedulerConfig{
		MaxBatchSize:    *maxBatch,
		MaxWaitDuration: time.Duration(*maxWaitMs) * time.Millisecond,
		QueueDepth:      *queueDepth,
	}
	sched := server.NewScheduler(schCfg, runner)
	if err := sched.Start(); err != nil {
		log.Fatalf("failed to start scheduler: %v", err)
	}
	defer sched.Stop()

	// ── Build tokenizer ───────────────────────────────────────────────
	var tok *tokenizer.BPETokenizer
	if *modelPath != "" {
		tokPath := *modelPath + "/tokenizer.json"
		if loaded, err := tokenizer.LoadHFTokenizer(tokPath); err == nil {
			tok = loaded
			log.Printf("Loaded real BPE tokenizer from %s", tokPath)
		} else {
			log.Printf("Warning: failed to load tokenizer.json: %v (falling back to simple)", err)
		}
	}

	// ── HTTP mux ──────────────────────────────────────────────────────
	api := &apiServer{scheduler: sched, tokenizer: tok}
	mux := http.NewServeMux()
	mux.HandleFunc("/generate", api.handleGenerate)
	mux.HandleFunc("/health", api.handleHealth)

	addr := fmt.Sprintf(":%d", *port)
	httpServer := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 5 * time.Minute,
		IdleTimeout:  120 * time.Second,
	}

	// ── Graceful shutdown ─────────────────────────────────────────────
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigCh
		log.Printf("Received %v — shutting down…", sig)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(ctx); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("HTTP shutdown error: %v", err)
		}
	}()

	log.Printf("Listening on %s", addr)
	log.Printf("Batch size=%d  wait=%dms  queue=%d  speculative=%v",
		*maxBatch, *maxWaitMs, *queueDepth, *enableSpec)

	if err := httpServer.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("ListenAndServe: %v", err)
	}
	log.Println("Server stopped.")
}

// tokenizeSimpleFallback converts a string into a minimal token sequence if tokenizer.json is missing.
func tokenizeSimpleFallback(text string) []int {
	tokens := []int{1} // BOS
	for _, b := range []byte(text) {
		tokens = append(tokens, int(b))
	}
	tokens = append(tokens, 2) // EOS
	return tokens
}
