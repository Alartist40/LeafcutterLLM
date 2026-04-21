package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Tokenizer defines the interface for text tokenization.
type Tokenizer interface {
	Encode(text string) []int
	Decode(tokens []int) string
}

// BPETokenizer implements a basic Byte-Pair Encoding tokenizer
// compatible with HuggingFace tokenizer.json format.
type BPETokenizer struct {
	vocab      map[string]int
	invVocab   map[int]string
	merges     map[string]int // maps "word1 word2" -> rank
	unkTokenID int
}

type hfTokenizerModel struct {
	Model struct {
		Type   string            `json:"type"`
		Vocab  map[string]int    `json:"vocab"`
		Merges []string          `json:"merges"`
	} `json:"model"`
}

// LoadHFTokenizer loads a HuggingFace tokenizer.json file.
func LoadHFTokenizer(path string) (*BPETokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer file: %w", err)
	}

	var hf hfTokenizerModel
	if err := json.Unmarshal(data, &hf); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer JSON: %w", err)
	}

	tok := &BPETokenizer{
		vocab:    hf.Model.Vocab,
		invVocab: make(map[int]string, len(hf.Model.Vocab)),
		merges:   make(map[string]int, len(hf.Model.Merges)),
	}

	for word, id := range tok.vocab {
		tok.invVocab[id] = word
	}

	for i, merge := range hf.Model.Merges {
		tok.merges[merge] = i
	}

	if id, ok := tok.vocab["<unk>"]; ok {
		tok.unkTokenID = id
	}

	return tok, nil
}

// Encode converts text into token IDs using BPE.
func (t *BPETokenizer) Encode(text string) []int {
	// A real implementation requires pre-tokenization (regex split) and byte-level mapping.
	// This is a simplified greedy matching approach for demonstration.
	var tokens []int
	words := strings.Fields(text) // Very basic pre-tokenization

	for _, word := range words {
		// Replace spaces with standard BPE space marker if applicable (e.g., Llama ' ')
		word = " " + word
		
		// Split into characters
		var chars []string
		for _, r := range word {
			chars = append(chars, string(r))
		}

		// Perform BPE merges
		for {
			if len(chars) < 2 {
				break
			}
			bestRank := -1
			bestIdx := -1

			for i := 0; i < len(chars)-1; i++ {
				pair := chars[i] + " " + chars[i+1]
				if rank, ok := t.merges[pair]; ok {
					if bestRank == -1 || rank < bestRank {
						bestRank = rank
						bestIdx = i
					}
				}
			}

			if bestIdx == -1 {
				break // No more merges possible
			}

			// Merge chars[bestIdx] and chars[bestIdx+1]
			chars[bestIdx] = chars[bestIdx] + chars[bestIdx+1]
			chars = append(chars[:bestIdx+1], chars[bestIdx+2:]...)
		}

		for _, char := range chars {
			if id, ok := t.vocab[char]; ok {
				tokens = append(tokens, id)
			} else {
				tokens = append(tokens, t.unkTokenID)
			}
		}
	}
	return tokens
}

// Decode converts token IDs back into a string.
func (t *BPETokenizer) Decode(tokens []int) string {
	var sb strings.Builder
	for _, id := range tokens {
		if word, ok := t.invVocab[id]; ok {
			// Convert standard BPE space marker back to space
			word = strings.ReplaceAll(word, " ", " ")
			sb.WriteString(word)
		} else {
			sb.WriteString("")
		}
	}
	return strings.TrimSpace(sb.String())
}
