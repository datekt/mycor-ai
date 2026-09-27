package engine

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

func LoadBrain(path string) bool {
	return loadBrain(path) == nil
}

func loadBrain(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var loaded BrainModel
	if err := json.Unmarshal(data, &loaded); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	if len(loaded.Vocabulary) == 0 {
		return errors.New("empty vocabulary")
	}
	if loaded.Weights == nil {
		loaded.Weights = make(map[string][]float64)
	}

	seen := make(map[string]bool, len(loaded.Vocabulary))
	for _, w := range loaded.Vocabulary {
		if w == "" {
			return errors.New("empty word in vocabulary")
		}
		if seen[w] {
			return fmt.Errorf("duplicate word in vocabulary: %q", w)
		}
		seen[w] = true
	}

	if loaded.Vocabulary[0] != unkToken {
		loaded.Vocabulary = append([]string{unkToken}, loaded.Vocabulary...)
		for k, v := range loaded.Weights {
			extended := make([]float64, len(v)+1)
			copy(extended[1:], v)
			extended[0] = 0
			loaded.Weights[k] = extended
		}
		if loaded.Velocity != nil {
			for k, v := range loaded.Velocity {
				extended := make([]float64, len(v)+1)
				copy(extended[1:], v)
				loaded.Velocity[k] = extended
			}
		}
	}

	vLen := len(loaded.Vocabulary)

	for k, v := range loaded.Weights {
		if len(v) < vLen {
			extended := make([]float64, vLen)
			copy(extended, v)
			for i := len(v); i < vLen; i++ {
				extended[i] = initWeight()
			}
			loaded.Weights[k] = extended
		} else if len(v) > vLen {
			loaded.Weights[k] = v[:vLen]
		}
	}

	newVelocity := make(map[string][]float64, len(loaded.Weights))
	for k := range loaded.Weights {
		if loaded.Velocity != nil {
			if v, ok := loaded.Velocity[k]; ok {
				if len(v) < vLen {
					ext := make([]float64, vLen)
					copy(ext, v)
					newVelocity[k] = ext
				} else if len(v) > vLen {
					newVelocity[k] = v[:vLen]
				} else {
					newVelocity[k] = v
				}
				continue
			}
		}
		newVelocity[k] = make([]float64, vLen)
	}

	Vocabulary = loaded.Vocabulary
	Weights = loaded.Weights
	Velocity = newVelocity
	WordToIdx = make(map[string]int, len(Vocabulary))
	for i, word := range Vocabulary {
		WordToIdx[word] = i
	}

	BackupVocabulary = nil
	BackupWordToIdx = nil
	BackupWeights = nil
	BackupVelocity = nil
	recentContexts = nil
	sessionHistory = nil
	lastThoughts = ""
	return nil
}

func SaveBrain(path string) error {
	model := BrainModel{
		Version:    modelVersion,
		Vocabulary: Vocabulary,
		Weights:    Weights,
		Velocity:   Velocity,
	}
	data, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		return err
	}

	dir := filepath.Dir(path)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}
