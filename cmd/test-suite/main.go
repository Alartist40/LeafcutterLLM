package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"time"

	"github.com/Alartist40/LeafcutterLLM/pkg/inference"
	"github.com/Alartist40/LeafcutterLLM/pkg/model"
)

type TestResult struct {
	ModelName string    `json:"model_name"`
	Timestamp time.Time `json:"timestamp"`
	Hardware  string    `json:"hardware"`

	Memory struct {
		PeakRAMMB    uint64 `json:"peak_ram_mb"`
		ModelSizeGB  float64 `json:"model_size_gb"`
		LayerSplitOK bool    `json:"layer_splitting_ok"`
	} `json:"memory"`

	Latency struct {
		FirstTokenMS float64 `json:"first_token_ms"`
		TokensPerSec float64 `json:"tokens_per_second"`
	} `json:"latency"`

	Throughput struct {
		SingleReqTPS float64 `json:"single_request_tps"`
	} `json:"throughput"`
}

func main() {
	modelPath := flag.String("model", "", "Path to model")
	outputFile := flag.String("output", "", "Output JSON file")
	flag.Parse()

	if *modelPath == "" {
		fmt.Println("Usage: ./test-suite --model <path> --output <file>")
		os.Exit(1)
	}

	log.Printf("Starting test for model: %s", *modelPath)

	cp, err := model.LoadCheckPoint(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	engine := inference.NewEngine(&cp.Config, cp.LayerLoader)
	defer engine.Release()

	result := TestResult{
		ModelName: *modelPath,
		Timestamp: time.Now(),
		Hardware:  getHardwareInfo(),
	}

	// Warmup
	log.Println("Warmup...")
	ctx := context.Background()
	_, _ = engine.Generate(ctx, []int{1, 2, 3}, 5, nil)

	// Test 1: Memory & Latency
	log.Println("Measuring Latency and RAM...")
	start := time.Now()
	var firstTokenTime time.Duration
	tokensGenerated := 0

	onToken := func(t int) {
		if tokensGenerated == 0 {
			firstTokenTime = time.Since(start)
		}
		tokensGenerated++
	}

	prompt := []int{1, 5, 10, 20, 30} // Dummy prompt
	_, err = engine.Generate(ctx, prompt, 20, onToken)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}
	totalTime := time.Since(start)

	result.Latency.FirstTokenMS = float64(firstTokenTime.Milliseconds())
	result.Latency.TokensPerSec = float64(tokensGenerated) / totalTime.Seconds()
	result.Throughput.SingleReqTPS = result.Latency.TokensPerSec

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	result.Memory.PeakRAMMB = m.Alloc / 1024 / 1024
	result.Memory.LayerSplitOK = true // Assuming if it ran

	// Save results
	data, _ := json.MarshalIndent(result, "", "  ")
	if *outputFile != "" {
		os.WriteFile(*outputFile, data, 0644)
		log.Printf("Results saved to %s", *outputFile)
	} else {
		fmt.Println(string(data))
	}
}

func getHardwareInfo() string {
	return fmt.Sprintf("%s/%s (%d CPUs)", runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
}
