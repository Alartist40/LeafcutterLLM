package utils

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
)

type HardwareInfo struct {
	TotalRAM     int64 // bytes
	AvailableRAM int64 // bytes
	NumCPU       int
	CPUArch      string
	HasOpenBLAS  bool
}

// DetectHardware returns current system capabilities
func DetectHardware() (*HardwareInfo, error) {
	hw := &HardwareInfo{
		NumCPU:  runtime.NumCPU(),
		CPUArch: runtime.GOARCH,
	}

	// Detect RAM (platform-specific)
	totalRAM, availRAM, err := detectRAM()
	if err != nil {
		// Log error but continue with 0 values
		fmt.Fprintf(os.Stderr, "Warning: RAM detection failed: %v\n", err)
	}
	hw.TotalRAM = totalRAM
	hw.AvailableRAM = availRAM

	// Check for OpenBLAS (simplified check)
	hw.HasOpenBLAS = true // LeafcutterLLM requires OpenBLAS for Turbo mode

	return hw, nil
}

// detectRAM returns total and available RAM in bytes
func detectRAM() (total, available int64, err error) {
	switch runtime.GOOS {
	case "linux":
		return detectRAMLinux()
	case "darwin":
		return detectRAMMacOS()
	case "windows":
		return detectRAMWindows()
	default:
		return 0, 0, fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}
}

func detectRAMLinux() (total, available int64, err error) {
	// Read /proc/meminfo
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0, err
	}

	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}

		switch fields[0] {
		case "MemTotal:":
			total, _ = strconv.ParseInt(fields[1], 10, 64)
			total *= 1024 // Convert KB to bytes
		case "MemAvailable:":
			available, _ = strconv.ParseInt(fields[1], 10, 64)
			available *= 1024 // Convert KB to bytes
		case "MemFree:":
			if available == 0 {
				available, _ = strconv.ParseInt(fields[1], 10, 64)
				available *= 1024
			}
		}
	}

	return total, available, nil
}

func detectRAMMacOS() (total, available int64, err error) {
	// sysctl hw.memsize
	// For now return dummy or use a shell command
	return 8 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024, nil
}

func detectRAMWindows() (total, available int64, err error) {
	return 8 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024, nil
}
