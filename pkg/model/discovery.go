package model

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ModelInfo describes a discovered model
type ModelInfo struct {
	Name       string
	Path       string
	Format     string // "safetensors" or "gguf"
	SizeBytes  int64
	Compatible bool // Set by hardware checker
}

// DiscoverModels scans the models/ directory and returns all found models
func DiscoverModels(modelsDir string) ([]ModelInfo, error) {
	var discovered []ModelInfo

	// Ensure models directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("models directory not found: %s", modelsDir)
	}

	// Walk the directory
	err := filepath.Walk(modelsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip the root directory
		if path == modelsDir {
			return nil
		}

		// Check for GGUF files
		if !info.IsDir() && strings.HasSuffix(path, ".gguf") {
			discovered = append(discovered, ModelInfo{
				Name:      strings.TrimSuffix(filepath.Base(path), ".gguf"),
				Path:      path,
				Format:    "gguf",
				SizeBytes: info.Size(),
			})
			return nil
		}

		// Check for safetensors directories (has config.json)
		if info.IsDir() {
			configPath := filepath.Join(path, "config.json")
			if _, err := os.Stat(configPath); err == nil {
				size, _ := getDirSize(path)
				discovered = append(discovered, ModelInfo{
					Name:      filepath.Base(path),
					Path:      path,
					Format:    "safetensors",
					SizeBytes: size,
				})
				return filepath.SkipDir // Don't recurse into this directory
			}
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to scan models directory: %w", err)
	}

	return discovered, nil
}

// getDirSize calculates total size of a directory
func getDirSize(path string) (int64, error) {
	var size int64
	err := filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size, err
}
