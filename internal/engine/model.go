package engine

import (
	"fmt"

	"mycor/internal/config"
)

// ChatMessage is one entry of the conversational history.
type ChatMessage struct {
	Role string `json:"role"`
	Text string `json:"text"`
}

// Stats is a read-only snapshot of the brain for the UI.
type Stats struct {
	Vocabulary     int      `json:"vocabulary"`
	Parameters     int      `json:"parameters"`
	Thinking       bool     `json:"thinking"`
	MinThinking    int      `json:"minThinking"`
	ContextSize    int      `json:"contextSize"`
	RecentContexts []string `json:"recentContexts"`
}

// Stats returns a consistent snapshot of the model's counters.
func (b *Brain) Stats() Stats {
	cfg := config.Get()
	b.mu.RLock()
	defer b.mu.RUnlock()
	return Stats{
		Vocabulary:     len(b.vocab),
		Parameters:     b.countParametersLocked(),
		Thinking:       len(b.vocab) >= cfg.MinVocabThinking,
		MinThinking:    cfg.MinVocabThinking,
		ContextSize:    contextSize(cfg),
		RecentContexts: append([]string(nil), b.recentContexts...),
	}
}

// VocabularyLen returns the number of known words, including <unk>.
func (b *Brain) VocabularyLen() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.vocab)
}

// ContextWeightSize reports how many sparse entries a context holds. It is the
// only weight accessor the UI needs.
func (b *Brain) ContextWeightSize(ctxKey string) (int, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	w, ok := b.weights[ctxKey]
	if !ok {
		return 0, false
	}
	return len(w), true
}

// CountParameters returns the total number of stored weights.
func (b *Brain) CountParameters() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.countParametersLocked()
}

// countParametersLocked sums sparse vector lengths. The caller must hold b.mu.
func (b *Brain) countParametersLocked() int {
	total := 0
	for _, w := range b.weights {
		total += len(w)
	}
	return total
}

// RegisterWord adds a word to the vocabulary and returns its index.
//
// Because weights are sparse, a new word needs no changes to the existing
// vectors: that was the quadratic cost of the old dense layout. The caller
// must hold b.mu for writing.
func (b *Brain) RegisterWord(word string) int {
	if word == "" {
		return unkIndex
	}
	if idx, ok := b.wordToIdx[word]; ok {
		return idx
	}
	idx := len(b.vocab)
	b.vocab = append(b.vocab, word)
	b.wordToIdx[word] = idx
	if b.undo != nil {
		b.undo.words = append(b.undo.words, word)
	}
	return idx
}

// RegisterWords registers every word of the slice in one pass.
func (b *Brain) RegisterWords(words []string) {
	for _, w := range words {
		b.RegisterWord(w)
	}
}

// undoJournal records just enough state to reverse a single Train call.
//
// The previous implementation snapshotted every weight vector on every
// training step, which is O(vocabulary x contexts) per call and dominated the
// profile for all but the smallest models. Here we record only the contexts a
// step actually touches, and only the words the step actually introduces.
type undoJournal struct {
	weights  map[string]sparseVector // pre-step copy, nil value means "was absent"
	velocity map[string]sparseVector
	words    []string // words registered during the step
	started  bool
}

// beginUndo starts a fresh journal. The caller must hold b.mu for writing.
func (b *Brain) beginUndo() {
	b.undo = &undoJournal{
		weights:  make(map[string]sparseVector),
		velocity: make(map[string]sparseVector),
	}
}

// recordContext snapshots a context the first time it is touched, so repeated
// updates within one training call do not re-copy it.
func (b *Brain) recordContext(key string) {
	if b.undo == nil {
		return
	}
	if _, seen := b.undo.weights[key]; !seen {
		b.undo.weights[key] = cloneSparse(b.weights[key])
	}
	if _, seen := b.undo.velocity[key]; !seen {
		b.undo.velocity[key] = cloneSparse(b.velocity[key])
	}
}

// cloneSparse copies a sparse vector. A nil input yields nil, which the undo
// path interprets as "this context did not exist".
func cloneSparse(src sparseVector) sparseVector {
	if src == nil {
		return nil
	}
	dst := make(sparseVector, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

// UndoLastTrain reverts the most recent Train call. It reports whether a
// training step was actually reverted.
func (b *Brain) UndoLastTrain() bool {
	b.trainMu.Lock()
	defer b.trainMu.Unlock()

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.undo == nil || !b.undo.started {
		return false
	}
	j := b.undo
	b.undo = nil

	for key, prev := range j.weights {
		if prev == nil {
			delete(b.weights, key)
			continue
		}
		b.weights[key] = prev
	}
	for key, prev := range j.velocity {
		if prev == nil {
			delete(b.velocity, key)
			continue
		}
		b.velocity[key] = prev
	}
	// Drop the words this step introduced; <unk> is never removed.
	for _, w := range j.words {
		delete(b.wordToIdx, w)
	}
	if n := len(b.vocab) - len(j.words); n > 0 {
		b.vocab = b.vocab[:n]
	}
	return true
}

// ensureSparse returns the weight vector for key, creating it if needed.
// The caller must hold b.mu for writing.
func (b *Brain) ensureSparse(key string) sparseVector {
	b.recordContext(key)
	w, ok := b.weights[key]
	if !ok {
		w = make(sparseVector)
		b.weights[key] = w
	}
	v, ok := b.velocity[key]
	if !ok {
		v = make(sparseVector)
		b.velocity[key] = v
	}
	return w
}

// entry returns the weight and velocity vectors for key, seeding a random
// weight when the entry is created. The caller must hold b.mu for writing.
func (b *Brain) entry(key string, idx int) (w, v float64, ok bool) {
	vec := b.ensureSparse(key)
	wv, exists := vec[idx]
	if !exists {
		wv = b.initWeight()
		vec[idx] = wv
	}
	vv, exists := b.velocity[key][idx]
	if !exists {
		vv = 0
		b.velocity[key][idx] = vv
	}
	return wv, vv, true
}

// String renders a short debug description.
func (b *Brain) String() string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return fmt.Sprintf("Brain{vocab:%d params:%d contexts:%d}",
		len(b.vocab), b.countParametersLocked(), len(b.weights))
}
