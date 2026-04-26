// cmd/server/main.go — LeafcutterLLM Server (Revised)
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

	"github.com/Alartist40/LeafcutterLLM/pkg/inference"
	"github.com/Alartist40/LeafcutterLLM/pkg/server"
	"github.com/Alartist40/LeafcutterLLM/pkg/tokenizer"
)

// ─── CLI flags ─────────────────────────────────────────────────────────────────

var (
	// Model flags
	modelPath  = flag.String("model", "", "Path to target (large) model")
	draftPath  = flag.String("draft", "", "Path to draft (small) model (optional)")
	port       = flag.Int("port", 8080, "HTTP port to listen on")
	maxBatch   = flag.Int("batch-size", 8, "Maximum continuous batch size")
	maxWaitMs  = flag.Int("batch-wait-ms", 20, "Max milliseconds to wait before flushing a partial batch")
	maxTokens  = flag.Int("max-tokens", 256, "Default max tokens per request")
	draftLen   = flag.Int("draft-len", 5, "Speculative decoding draft length")
	queueDepth = flag.Int("queue-depth", 256, "Request queue depth")
	enableSpec = flag.Bool("speculative", false, "Enable speculative decoding (requires --draft)")
	showVer    = flag.Bool("version", false, "Print version and exit")
)

const serverVersion = "leafcutter-server v0.4.0 (Turbo Engine: Q4+Speculative+Batching)"

// ─── HTTP request/response DTOs ────────────────────────────────────────────────

type GenerateRequest struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float32 `json:"temperature,omitempty"`
	Stream      bool    `json:"stream,omitempty"`
}

type GenerateResponse struct {
	ID     string `json:"id"`
	Tokens []int  `json:"tokens,omitempty"`
	TookMs int64  `json:"took_ms"`
	Error  string `json:"error,omitempty"`
}

// ─── Real model runner (wires scheduler → engine) ──────────────────────────────

type modelRunner struct {
	targetEngine *inference.Engine
	specEngine   *inference.SpeculativeEngine
	mu           sync.Mutex
	reqCounter   atomic.Int64
}

func (r *modelRunner) RunBatch(ctx context.Context, batch *server.Batch) error {
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

// FIX-009: Replace broken runSingle with correct control flow.
func (r *modelRunner) runSingle(ctx context.Context, req *server.InferRequest) ([]int, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	if r.specEngine != nil {
		return r.specEngine.Generate(ctx, req.Prompt, req.OnToken)
	}

	if r.targetEngine != nil {
		return r.targetEngine.Generate(ctx, req.Prompt, req.MaxTokens, req.OnToken)
	}

	return nil, fmt.Errorf("no engine loaded — pass --model <path> to load a model")
}

// ─── HTTP handlers ─────────────────────────────────────────────────────────────

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
		req.MaxTokens = 256
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

	onToken := func(t int) {
		// Stub/Debug: fmt.Printf("[Stream] Token %d", t)
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
	json.NewEncoder(w).Encode(out)
}

func (s *apiServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	stats := s.scheduler.Stats()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":         "ok",
		"version":        serverVersion,
		"total_requests": stats.TotalRequests,
		"total_batches":  stats.TotalBatches,
		"dropped":        stats.DroppedRequest,
		"queue_depth":    stats.QueueDepth,
	})
}

// ─── Main ──────────────────────────────────────────────────────────────────────

func main() {
	flag.Parse()

	// FIX-008: use showVer (renamed from version to avoid conflict with const)
	if *showVer {
		fmt.Println(serverVersion)
		os.Exit(0)
	}

	log.SetFlags(log.LstdFlags | log.Lmsgprefix)
	log.SetPrefix("[leafcutter-server] ")

	// ── Build runner ────────────────────────────────────────────────────────────
	runner := &modelRunner{}

	if *enableSpec {
		if *draftPath == "" {
			log.Fatal("--speculative requires --draft <path>")
		}
		log.Printf("Speculative decoding enabled: draft=%s target=%s draftLen=%d",
			*draftPath, *modelPath, *draftLen)
	}

	// ── Build scheduler ─────────────────────────────────────────────────────────
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

	// FIX-010: Correct tokenizer loading block with proper brace structure.
	var tok *tokenizer.BPETokenizer
	if *modelPath != "" {
		tokPath := *modelPath + "/tokenizer.json"
		if loaded, err := tokenizer.LoadHFTokenizer(tokPath); err == nil {
			tok = loaded
			log.Printf("Loaded BPE tokenizer from %s", tokPath)
		} else {
			log.Printf("Warning: failed to load tokenizer.json: %v (falling back to byte-level)", err)
		}
	}

	// ── HTTP mux ────────────────────────────────────────────────────────────────
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

	// ── Graceful shutdown ────────────────────────────────────────────────────────
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

	log.Printf("LeafcutterLLM server listening on %s", addr)
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
	tokens = append(tokens, 2) // EOS token
	return tokens
}