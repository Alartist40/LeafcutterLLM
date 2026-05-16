// cmd/test-kquant/main.go
// LeafcutterLLM K-Quant Verification Tool
//
// This program verifies that Leafcutter can successfully load a K-quant GGUF model.
// Run it after building Leafcutter to confirm K-quant support is working:
//
//	go run ./cmd/test-kquant/ <path-to-model.gguf>
//
// Expected output for a valid K-quant model:
//	Model: llama, Layers=32, VocabSize=151936
//	Layer 0 loaded: 9 tensors
//	✅ K-quant dequantization WORKS!
//
// TEAM NOTE: Keep this file. Use it to verify K-quant support after any changes
// to pkg/model/gguf_loader.go or internal/gguf/gguf.go.

package main

import (
	"fmt"
	"os"

	"github.com/Alartist40/LeafcutterLLM/pkg/model"
)

func main() {
	path := "/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}

	cp, err := model.LoadCheckPoint(path)
	if err != nil {
		fmt.Println("❌ ERROR:", err)
		os.Exit(1)
	}

	fmt.Printf("Architecture: %s\n", cp.Architecture)
	fmt.Printf("Layers:       %d\n", cp.LayerCount)
	fmt.Printf("VocabSize:    %d\n", cp.VocabSize)

	layer0, err := cp.LayerLoader.LoadLayer(0)
	if err != nil {
		fmt.Println("❌ ERROR loading layer 0:", err)
		os.Exit(1)
	}
	fmt.Printf("Layer 0 tensors: %d\n", len(layer0))
	for name, t := range layer0 {
		fmt.Printf("  %s: shape=%v dtype=%d\n", name, t.Shape, t.DType)
		break
	}
	fmt.Println("✅ K-quant dequantization WORKS! Model loads successfully.")
}
