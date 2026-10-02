/*
Package engine implements the MYCOR language model: a pure-Go N-gram
architecture with an adaptive context window, dynamic backoff, Top-K and
Top-P sampling, thinking mode, idle daydreams, and persistent weights.

The model lives in a Brain value instead of package-level variables. A Brain
is safe for concurrent use:

  - mu guards every field. Readers (generation, statistics) take RLock;
    writers (training, reset, load) take Lock.
  - trainMu serialises training against itself, so two concurrent /api/train
    requests cannot interleave their weight updates. It is held for the whole
    training call, but the state lock is released between individual steps.

That split is what keeps the server responsive: a long bulk import holds
trainMu for minutes but only ever locks mu for one training step at a time, so
chat requests interleave freely instead of queueing behind the import.

Weight vectors are sparse (map[int]float64 indexed by vocabulary id). Adding a
word to the vocabulary therefore costs O(1) instead of rewriting every context
vector, and an undo only has to snapshot the contexts a step actually touched.
*/
package engine

import (
	"math/rand"
	"sync"
	"time"

	"mycor/internal/config"
)

const (
	unkToken     = "<unk>"
	modelVersion = 6
	unkIndex     = 0
)

// Brain is a single MYCOR model. The zero value is not usable; call NewBrain.
type Brain struct {
	mu      sync.RWMutex
	trainMu sync.Mutex
	rng     *rand.Rand

	vocab     []string
	wordToIdx map[string]int

	// Sparse weight vectors keyed by vocabulary index. Index 0 (<unk>) is
	// never present: it is reserved for out-of-vocabulary contexts.
	weights  map[string]sparseVector
	velocity map[string]sparseVector

	undo *undoJournal

	recentContexts []string
	sessionHistory []ChatMessage
	lastThoughts   string
}

// NewBrain returns an initialised, empty brain.
func NewBrain() *Brain {
	b := &Brain{}
	b.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	b.reset()
	return b
}

// reset clears all learned state. The caller must hold b.mu for writing.
func (b *Brain) reset() {
	b.vocab = []string{unkToken}
	b.wordToIdx = map[string]int{unkToken: unkIndex}
	b.weights = make(map[string]sparseVector)
	b.velocity = make(map[string]sparseVector)
	b.undo = nil
	b.recentContexts = nil
	b.sessionHistory = nil
	b.lastThoughts = ""
}

// Reset discards the entire brain, returning it to its initial state.
func (b *Brain) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	b.reset()
}

// initWeight returns a small random weight used to seed a fresh entry. The
// caller must hold b.mu.
func (b *Brain) initWeight() float64 {
	return (b.rng.Float64() - 0.5) * 0.1
}

// contextSize returns the active context width, clamped to the configured
// bounds. It reads config rather than trusting it to be sane.
func contextSize(cfg config.Config) int {
	n := cfg.ContextSize
	if n < config.MinContextSize {
		return config.MinContextSize
	}
	if n > config.MaxContextSize {
		return config.MaxContextSize
	}
	return n
}

// ThinkingEnabled reports whether the vocabulary is large enough for the
// thinking pass to produce anything meaningful.
func (b *Brain) ThinkingEnabled() bool {
	cfg := config.Get()
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.vocab) >= cfg.MinVocabThinking
}

// MinVocabForThinking exposes the configured threshold for the UI.
func MinVocabForThinking() int { return config.Get().MinVocabThinking }
