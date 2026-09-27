/*
Package engine implements the MYCOR language model: a pure-Go N-gram
architecture with a two-word context window, dynamic backoff, thinking
mode, idle daydreams, and persistent weights.

The package keeps global mutable state representing a single process-wide
brain. Call InitEngine to reset it, LoadBrain to load from disk, and
Train or GenerateResponse to interact with it.
*/
package engine

import (
	"math/rand"
)

const (
	unkToken            = "<unk>"
	modelVersion        = 4
	MinVocabForThinking = 10
	thinkingSteps       = 4
	maxRecentContexts   = 20
	maxSessionHistory   = 100
)

var (
	Vocabulary []string
	WordToIdx  map[string]int
	Weights    map[string][]float64
	Velocity   map[string][]float64

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
	Weights = make(map[string][]float64)
	Velocity = make(map[string][]float64)

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
