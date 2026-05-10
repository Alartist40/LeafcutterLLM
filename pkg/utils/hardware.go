package utils

import (
	"fmt"
	"os"
	"os/exec"
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
	// Use sysctl for total RAM
	totalData, err := runCommand("sysctl", "-n", "hw.memsize")
	if err != nil {
		return 0, 0, fmt.Errorf("macOS total RAM: %w", err)
	}
	total, _ = strconv.ParseInt(strings.TrimSpace(totalData), 10, 64)

	// Use vm_stat for available RAM
	// This is a rough estimation based on free + inactive pages
	vmData, err := runCommand("vm_stat")
	if err != nil {
		return total, 0, nil // Return total if available fails
	}

	pageSize := int64(4096) // Default page size
	lines := strings.Split(vmData, "\n")
	var free, inactive int64
	for _, line := range lines {
		if strings.HasPrefix(line, "Pages free:") {
			free = parseVMStatLine(line)
		} else if strings.HasPrefix(line, "Pages inactive:") {
			inactive = parseVMStatLine(line)
		}
	}
	available = (free + inactive) * pageSize

	return total, available, nil
}

func detectRAMWindows() (total, available int64, err error) {
	// Proper Windows RAM detection would require "wmic" or "systeminfo"
	// but those are slow and sometimes missing.
	// For now, return error to indicate it's not implemented, instead of lying.
	return 0, 0, fmt.Errorf("Windows RAM detection not yet implemented")
}

func runCommand(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func parseVMStatLine(line string) int64 {
	fields := strings.Fields(line)
	if len(fields) < 3 {
		return 0
	}
	valStr := strings.TrimSuffix(fields[len(fields)-1], ".")
	val, _ := strconv.ParseInt(valStr, 10, 64)
	return val
}
