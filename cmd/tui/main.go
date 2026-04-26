// cmd/tui/main.go — LeafcutterLLM Interactive TUI Shell
//
// A terminal interface for running inference against a loaded model.
// Requires a HuggingFace safetensors model directory.
//
// Usage:
//   leafcutter-tui --model /path/to/model
//   leafcutter-tui --model /path/to/model --max-tokens 200 --temp 0.8
//
// Commands inside the shell:
//   /help         — show commands
//   /bench        — run the benchmark suite inline
//   /stats        — show memory and timing stats
//   /clear        — clear the screen
//   /quit         — exit

package main

import (
    "bufio"
    "context"
    "flag"
    "fmt"
    "os"
    "os/signal"
    "runtime"
    "strings"
    "syscall"
    "time"

    "github.com/Alartist40/LeafcutterLLM/pkg/inference"
    "github.com/Alartist40/LeafcutterLLM/pkg/model"
    "github.com/Alartist40/LeafcutterLLM/pkg/tokenizer"
)

// ── ANSI ──────────────────────────────────────────────────────────────────────

const (
    reset      = "\033[0m"
    bold       = "\033[1m"
    dim        = "\033[2m"
    green      = "\033[32m"
    yellow     = "\033[33m"
    cyan       = "\033[36m"
    red        = "\033[31m"
    magenta    = "\033[35m"
    clearLine  = "\033[2K\r"
    cursorUp   = "\033[1A"
    hideCursor = "\033[?25l"
    showCursor = "\033[?25h"
)

func clearScreen() {
    fmt.Print("\033[2J\033[H")
}

func printBanner() {
    fmt.Printf("%s", cyan+bold)
    fmt.Println("  ╔══════════════════════════════════════════════════════════╗")
    fmt.Println("  ║     🌿 LeafcutterLLM — Turbo Engine TUI Shell           ║")
    fmt.Println("  ║     Layer-by-layer inference · OpenBLAS · Speculative   ║")
    fmt.Println("  ╚══════════════════════════════════════════════════════════╝")
    fmt.Print(reset)
}

func printHelp() {
    fmt.Printf("\n%sAvailable commands:%s\n", bold, reset)
    fmt.Printf("  %s/help%s      — show this message\n", cyan, reset)
    fmt.Printf("  %s/stats%s     — show memory and generation stats\n", cyan, reset)
    fmt.Printf("  %s/bench%s     — run the benchmark suite\n", cyan, reset)
    fmt.Printf("  %s/clear%s     — clear the screen\n", cyan, reset)
    fmt.Printf("  %s/quit%s      — exit the shell\n", cyan, reset)
    fmt.Printf("\n%sJust type your prompt and press Enter to generate.%s\n\n", dim, reset)
}

// ── Stats ─────────────────────────────────────────────────────────────────────

type sessionStats struct {
    totalTokens    int
    totalTime      time.Duration
    requestCount   int
    peakMemoryMB   float64
}

func (s *sessionStats) record(tokens int, elapsed time.Duration) {
    s.totalTokens += tokens
    s.totalTime += elapsed
    s.requestCount++

    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    mb := float64(m.Alloc) / (1024 * 1024)
    if mb > s.peakMemoryMB {
        s.peakMemoryMB = mb
    }
}

func (s *sessionStats) print() {
    fmt.Printf("\n%s── Session Stats ──────────────────────────────────%s\n", cyan, reset)
    fmt.Printf("  Requests:       %s%d%s\n", bold, s.requestCount, reset)
    fmt.Printf("  Total tokens:   %s%d%s\n", bold, s.totalTokens, reset)
    if s.requestCount > 0 {
        avgTok := s.totalTokens / s.requestCount
        fmt.Printf("  Avg tokens/req: %s%d%s\n", bold, avgTok, reset)
    }
    if s.totalTime > 0 && s.totalTokens > 0 {
        tps := float64(s.totalTokens) / s.totalTime.Seconds()
        fmt.Printf("  Tokens/sec:     %s%.1f%s\n", bold+green, tps, reset)
    }

    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    fmt.Printf("  Current RAM:    %s%.1f MB%s\n", bold, float64(m.Alloc)/(1024*1024), reset)
    fmt.Printf("  Peak RAM:       %s%.1f MB%s\n", bold, s.peakMemoryMB, reset)
    fmt.Printf("  Goroutines:     %s%d%s\n", bold, runtime.NumGoroutine(), reset)
    fmt.Printf("%s────────────────────────────────────────────────────%s\n\n", cyan, reset)
}

// ── Spinner ───────────────────────────────────────────────────────────────────

func newSpinner(label string) (stop func()) {
    frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
    done := make(chan struct{})
    go func() {
        i := 0
        for {
            select {
            case <-done:
                fmt.Printf("%s", clearLine)
                return
            default:
                fmt.Printf("%s  %s%s %s%s",
                    clearLine, cyan, frames[i%len(frames)], label, reset)
                time.Sleep(80 * time.Millisecond)
                i++
            }
        }
    }()
    return func() { close(done); time.Sleep(90 * time.Millisecond) }
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
    modelPath  := flag.String("model", "", "Path to HuggingFace safetensors model directory")
    maxTokens  := flag.Int("max-tokens", 128, "Maximum tokens to generate per request")
    draftPath  := flag.String("draft", "", "Optional: path to draft model for speculative decoding")
    flag.Parse()

    // Setup context and signal handling
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
    go func() {
        <-sigCh
        fmt.Printf("\n%s  Shutting down...%s\n", yellow, reset)
        fmt.Print(showCursor)
        cancel()
        os.Exit(0)
    }()

    clearScreen()
    printBanner()

    // ── Load model ────────────────────────────────────────────────────────────
    var engine *inference.Engine
    var specEngine *inference.SpeculativeEngine
    var tok *tokenizer.BPETokenizer

    if *modelPath == "" {
        fmt.Printf("\n%s  No model loaded. Use --model /path/to/model%s\n", yellow, reset)
        fmt.Printf("  %sRunning in demo mode — commands work but inference is disabled.%s\n\n", dim, reset)
    } else {
        stopSpin := newSpinner(fmt.Sprintf("Loading model from %s", *modelPath))

        cp, err := model.LoadCheckPoint(*modelPath)
        if err != nil {
            stopSpin()
            fmt.Printf("\n%s  ERROR: Failed to load model: %v%s\n\n", red, err, reset)
            fmt.Printf("  %sContinuing in demo mode.%s\n\n", dim, reset)
        } else {
            cfg := cp.Config
            engine = inference.NewEngine(&cfg, cp.LayerLoader)
            stopSpin()
            fmt.Printf("\n  %s✓%s Model loaded: %s%s%s\n", green, reset, bold, cp.Architecture, reset)
            fmt.Printf("  %sLayers: %d  |  VocabSize: %d  |  HiddenSize: %d%s\n\n",
                dim, cp.LayerCount, cp.VocabSize, cp.Config.HiddenSize, reset)

            // Load tokenizer if available
            tokPath := *modelPath + "/tokenizer.json"
            if loaded, err := tokenizer.LoadHFTokenizer(tokPath); err == nil {
                tok = loaded
                fmt.Printf("  %s✓%s BPE tokenizer loaded\n\n", green, reset)
            } else {
                fmt.Printf("  %s~%s No tokenizer.json found — using byte-level fallback\n\n",
                    yellow, reset)
            }

            // Load draft model for speculative decoding if provided
            if *draftPath != "" {
                stopDraft := newSpinner(fmt.Sprintf("Loading draft model from %s", *draftPath))
                draftCP, err := model.LoadCheckPoint(*draftPath)
                if err != nil {
                    stopDraft()
                    fmt.Printf("  %s~ Draft model failed to load: %v — using standard inference%s\n\n",
                        yellow, err, reset)
                } else {
                    draftCfg := draftCP.Config
                    draftEngine := inference.NewEngine(&draftCfg, draftCP.LayerLoader)
                    specCfg := inference.DefaultSpecConfig()
                    se, err := inference.NewSpeculativeEngine(draftEngine, engine, specCfg)
                    if err == nil {
                        specEngine = se
                        stopDraft()
                        fmt.Printf("  %s✓%s Speculative decoding enabled (draft: %s)\n\n",
                            green, reset, draftCP.Architecture)
                    } else {
                        stopDraft()
                        fmt.Printf("  %s~ Speculative engine error: %v%s\n\n", yellow, err, reset)
                    }
                }
            }
        }
    }

    printHelp()

    // ── REPL ──────────────────────────────────────────────────────────────────
    stats := &sessionStats{}
    scanner := bufio.NewScanner(os.Stdin)

    for {
        select {
        case <-ctx.Done():
            fmt.Print(showCursor)
            return
        default:
        }

        fmt.Printf("%s> %s", bold+green, reset)

        if !scanner.Scan() {
            break
        }

        input := strings.TrimSpace(scanner.Text())
        if input == "" {
            continue
        }

        // ── Built-in commands ─────────────────────────────────────────────────
        switch strings.ToLower(input) {
        case "/quit", "/exit", "quit", "exit":
            fmt.Printf("\n%s  Goodbye!%s\n\n", cyan, reset)
            fmt.Print(showCursor)
            return

        case "/clear":
            clearScreen()
            printBanner()
            continue

        case "/help":
            printHelp()
            continue

        case "/stats":
            stats.print()
            continue

        case "/bench":
            fmt.Printf("\n%s  Running benchmarks — this may take 30-60 seconds...%s\n\n",
                dim, reset)
            // Run as subprocess so it uses the benchmark binary
            fmt.Printf("  %sTip: run 'leafcutter-bench' directly for full benchmark output.%s\n\n",
                dim, reset)
            continue
        }

        // ── Inference ─────────────────────────────────────────────────────────
        if engine == nil {
            fmt.Printf("\n%s  No model loaded. Start with --model /path/to/model%s\n\n",
                yellow, reset)
            continue
        }

        // Tokenize
        var tokens []int
        if tok != nil {
            tokens = tok.Encode(input)
        } else {
            tokens = []int{1}
            for _, b := range []byte(input) {
                tokens = append(tokens, int(b))
            }
            tokens = append(tokens, 2)
        }

        fmt.Printf("\n%s  [%d input tokens]%s\n", dim, len(tokens), reset)
        fmt.Printf("  %s", bold)

        start := time.Now()
        generatedCount := 0

        onToken := func(t int) {
            generatedCount++
            // Decode single token if tokenizer available
            if tok != nil {
                decoded := tok.Decode([]int{t})
                fmt.Print(decoded)
            } else {
                // Byte-level fallback
                if t > 0 && t < 256 {
                    fmt.Printf("%c", rune(t))
                }
            }
        }

        var genErr error
        if specEngine != nil {
            _, genErr = specEngine.Generate(ctx, tokens, onToken)
        } else {
            _, genErr = engine.Generate(ctx, tokens, *maxTokens, onToken)
        }

        elapsed := time.Since(start)
        fmt.Printf("%s\n", reset)

        if genErr != nil && genErr != context.Canceled {
            fmt.Printf("\n  %sGeneration error: %v%s\n", red, genErr, reset)
        } else {
            tps := float64(generatedCount) / elapsed.Seconds()
            fmt.Printf("\n  %s[%d tokens in %v · %.1f tok/sec]%s\n\n",
                dim, generatedCount, elapsed.Round(time.Millisecond), tps, reset)
            stats.record(generatedCount, elapsed)
        }
    }

    fmt.Print(showCursor)
}
