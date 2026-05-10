package main

import (
	"fmt"
	"github.com/Alartist40/LeafcutterLLM/pkg/model"
)

func printCompatibilityReport(report *model.CompatibilityReport) {
	fmt.Println("\n═══════════════════════════════════════════════════")
	fmt.Println("       Hardware Compatibility Check")
	fmt.Println("═══════════════════════════════════════════════════\n")

	// Hardware info
	fmt.Printf("💻 System Information:\n")
	fmt.Printf("   CPU Cores:       %d\n", report.Hardware.NumCPU)
	fmt.Printf("   Architecture:    %s\n", report.Hardware.CPUArch)
	fmt.Printf("   Total RAM:       %s\n", formatSize(report.Hardware.TotalRAM))
	fmt.Printf("   Available RAM:   %s\n", formatSize(report.Hardware.AvailableRAM))
	fmt.Printf("   OpenBLAS:        %v\n\n", report.Hardware.HasOpenBLAS)

	// Model info
	fmt.Printf("📊 Model Requirements:\n")
	fmt.Printf("   Parameters:      %s\n", formatParams(report.ModelSize.TotalParams))
	fmt.Printf("   Quantization:    Q%d\n", report.ModelSize.QuantBits)
	fmt.Printf("   Total Weights:   %s (on-disk)\n\n", formatSize(report.ModelSize.WeightsSize))

	// LeafcutterLLM Advantage
	fmt.Printf("🌿 LeafcutterLLM Advantage:\n")
	fmt.Printf("   Traditional Engine: %s needed ❌\n", formatSize(report.ModelSize.PeakMemory))
	fmt.Printf("   LeafcutterLLM:      %s needed ✅\n", formatSize(report.ModelSize.LeafcutterPeak))
	fmt.Printf("   Memory Savings:     %.1fx reduction!\n\n", report.MemorySavingsX)

	// Compatibility verdict
	switch report.Level {
	case model.Compatible:
		fmt.Printf("✅ COMPATIBLE\n")
		if report.SafetyMargin > 0 {
			fmt.Printf("   Model will run comfortably with %.0f%% safety margin\n",
				report.SafetyMargin*100)
		} else {
			fmt.Printf("   Model will run comfortably\n")
		}
	case model.Marginal:
		fmt.Printf("⚠️  MARGINAL\n")
		fmt.Printf("   %s\n", report.Warning)
	case model.Incompatible:
		fmt.Printf("❌ INCOMPATIBLE\n")
		fmt.Printf("   %s\n", report.Warning)
		fmt.Printf("\n💡 Suggestions:\n")
		fmt.Printf("   1. Reduce KV cache size (lower max tokens)\n")
		fmt.Printf("   2. Use higher quantization (Q3 or Q2)\n")
		fmt.Printf("   3. Enable more system swap space\n")
	}

	fmt.Println("\n═══════════════════════════════════════════════════\n")
}

func formatParams(params int64) string {
	if params < 1e6 {
		return fmt.Sprintf("%d", params)
	}
	if params < 1e9 {
		return fmt.Sprintf("%.1fM", float64(params)/1e6)
	}
	return fmt.Sprintf("%.1fB", float64(params)/1e9)
}
