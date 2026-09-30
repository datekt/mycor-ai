/*
Package engine implements the MYCOR language model: a pure-Go N-gram
architecture with an adaptive context window, dynamic backoff, Top-K and
Top-P sampling, thinking mode, idle daydreams, and persistent weights.

The package keeps global mutable state representing a single process-wide
brain. Call InitEngine to reset it, LoadBrain to load from disk, and
Train or GenerateResponse to interact with it.

Weight and velocity maps are intentionally unexported. External packages
query them through ContextWeightSize, which is the only read accessor the
UI needs.
*/
package engine

import (
	"math/rand"

	"mycor/internal/config"
)

const (
	unkToken            = "<unk>"
	modelVersion        = 5
	MinVocabForThinking = 10
	thinkingSteps       = 4
	maxRecentContexts   = 20
	maxSessionHistory   = 100
)

var (
	Vocabulary []string
	WordToIdx  map[string]int
	weights    map[string][]float64
	velocity   map[string][]float64

	BackupVocabulary []string
	BackupWordToIdx  map[string]int
	BackupWeights    map[string][]float64
	BackupVelocity   map[string][]float64

	recentContexts []string
	sessionHistory []ChatMessage
	lastThoughts   string
)

func InitEngine() {
	Vocabulary = []string{unkToken}
	WordToIdx = map[string]int{unkToken: 0}
	weights = make(map[string][]float64)
	velocity = make(map[string][]float64)

	BackupVocabulary = nil
	BackupWordToIdx = nil
	BackupWeights = nil
	BackupVelocity = nil

	recentContexts = nil
	sessionHistory = nil
	lastThoughts = ""
}

func initWeight() float64 {
	return (rand.Float64() - 0.5) * 0.1
}

func ContextWeightSize(ctxKey string) (int, bool) {
	v, ok := weights[ctxKey]
	if !ok {
		return 0, false
	}
	return len(v), true
}

func currentContextSize() int {
	n := config.ContextSize
	if n < config.MinContextSize {
		return config.MinContextSize
	}
	if n > config.MaxContextSize {
		return config.MaxContextSize
	}
	return n
}
