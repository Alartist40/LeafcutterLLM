// Package main provides the airllm CLI
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/xander/airllm-go/pkg/inference"
	"github.com/xander/airllm-go/pkg/model"
	"github.com/xander/airllm-go/pkg/tensor"
)

var (
	// Model flags
	modelPath   = flag.String("model", "", "Path to model checkpoint directory")
	modelID     = flag.String("model-id", ".", "HuggingFace model ID or local path")
	
	// Generation flags
	prompt      = flag.String("prompt", "What is the capital of France?", "Input prompt")
	maxTokens   = flag.Int("max-tokens", 100, "Maximum number of tokens to generate")
	temperature = flag.Float64("temperature", 0.8, "Sampling temperature")
	
	// Performance flags
	device      = flag.String("device", "cpu", "Device to use: cpu, cuda")
	numThreads  = flag.Int("threads", 0, "Number of threads (0 = auto)")
	prefetching = flag.Bool("prefetch", true, "Enable layer prefetching")
	profiling   = flag.Bool("profile", false, "Enable profiling output")
	
	// Memory flags
	maxSeqLen   = flag.Int("max-seq-len", 2048, "Maximum sequence length")
	dtype       = flag.String("dtype", "float16", "Data type: float32, float16")
	
	// Quantization flags
	compression = flag.String("compression", "", "Compression: 4bit, 8bit (empty = none)")
	
	// Special modes
	interactive = flag.Bool("interactive", false, "Run in interactive mode")
	version     = flag.Bool("version", false, "Show version")
)

const versionStr = "airllm-go v1.0.0"

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "High-performance LLM inference for large models on limited memory\n\n")
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  %s -model /path/to/llama-7b -prompt \"Hello\"\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -model /path/to/llama-70b -interactive\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -model /path/to/model -compression 4bit -profile\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nOptions:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	if *version {
		fmt.Println(versionStr)
		fmt.Println("  A Go reimplementation of AirLLM for high-performance inference")
		fmt.Println("  Run 70B+ parameter models on 4GB+ of RAM")
		os.Exit(0)
	}

	// Determine model path
	if *modelPath == "" {
		if flag.NArg() > 0 {
			*modelPath = flag.Arg(0)
		} else {
			*modelPath = *modelID
		}
	}

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Error: Model path required\n\n")
		flag.Usage()
		os.Exit(1)
	}

	// Setup context for cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Println("\nReceived interrupt signal, shutting down...")
		cancel()
	}()

	// Run the requested mode
	if *interactive {
		runInteractive(ctx)
	} else {
		runSingle(ctx)
	}
}

func runSingle(ctx context.Context) {
	fmt.Printf("Loading model from: %s\n", *modelPath)
	startTime := time.Now()

	// Load checkpoint
	checkpoint, err := model.LoadCheckPoint(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load checkpoint: %v", err)
	}

	fmt.Printf("Model architecture: %s\n", checkpoint.Architecture)
	fmt.Printf("Hidden size: %d\n", checkpoint.Config.HiddenSize)
	fmt.Printf("Layers: %d\n", checkpoint.Config.NumHiddenLayers)
	fmt.Printf("Attention heads: %d\n", checkpoint.Config.NumAttentionHeads)

	// Build config for inference
	cfg := buildConfig()

	// Create inference engine
	engine := inference.NewEngine(cfg, checkpoint.LayerLoader)
	
	fmt.Printf("Model loaded in %v\n\n", time.Since(startTime))

	// Tokenize input (simplified - just split by space for demo's sake)
	// In production, would use a proper tokenizer
	tokens := tokenizeSimple(*prompt)
	fmt.Printf("Input: %s\n", *prompt)
	fmt.Printf("Tokens: %v\n\n", tokens)

	// Generate
	fmt.Println("Generating...")
	genStart := time.Now()

	result, err := engine.Generate(ctx, tokens, *maxTokens, func(token int) {
		fmt.Printf("%d ", token)
	})

	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	fmt.Printf("\n\nGenerated %d tokens in %v\n", len(result), time.Since(genStart))

	// Cleanup
	engine.Release()
}

func runInteractive(ctx context.Context) {
	fmt.Println(versionStr)
	fmt.Println("Loading model...")

	// Load checkpoint
	checkpoint, err := model.LoadCheckPoint(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load checkpoint: %v", err)
	}

	fmt.Printf("Model: %s\n", checkpoint.Architecture)
	fmt.Printf("Type 'quit' or 'exit' to stop\n\n")

	// Build config
	cfg := buildConfig()

	// Create inference engine
	engine := inference.NewEngine(cfg, checkpoint.LayerLoader)
	defer engine.Release()

	// Interactive loop
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		fmt.Print("> ")
		
		// Read user input
		var input string
		if _, err := fmt.Scanln(&input); err != nil {
			fmt.Println()
			continue
		}

		if strings.ToLower(input) == "quit" || strings.ToLower(input) == "exit" {
			fmt.Println("Goodbye!")
			return
		}

		// Tokenize and generate
		tokens := tokenizeSimple(input)
		
		fmt.Print("Assistant: ")
		start := time.Now()

		result, err := engine.Generate(ctx, tokens, *maxTokens, func(token int) {
			// In a real implementation, would detokenize here
			// For now, just print token ID
			if token < 256 {
				fmt.Printf("%c", rune(token))
			}
		})

		if err != nil {
			fmt.Printf("\nGeneration error: %v\n", err)
			continue
		}

		fmt.Printf(" (%d tokens in %v)\n\n", len(result), time.Since(start))
	}
}

func buildConfig() *inference.Config {
	var dtype tensor.DType
	switch *dtype {
	case "float32":
		dtype = tensor.Float32
	case "float16":
		dtype = tensor.Float16
	default:
		dtype = tensor.Float16
	}

	return &inference.Config{
		Device:         *device,
		DType:          dtype,
		MaxSeqLen:      *maxSeqLen,
		NumThreads:     *numThreads,
		Prefetching:    *prefetching,
		Profiling:      *profiling,
		KVCacheEnabled: true,
	}
}

// Simple tokenization for demo purposes
func tokenizeSimple(text string) []int {
	// This is a very naive tokenizer - in production use a proper BPE tokenizer
	// For now, convert characters to token IDs for testing
	tokens := []int{1} // BOS token
	
	// Simple word-level tokenization
	words := strings.Fields(text)
	for i, word := range words {
		// Convert first letter to token (just for demo)
		if len(word) > 0 {
			tokens = append(tokens, int(word[0]))
		}
		
		// Add space token between words
		if i < len(words)-1 {
			tokens = append(tokens, 259) // Space token ID
		}
	}
	
	tokens = append(tokens, 2) // EOS token
	return tokens
}
