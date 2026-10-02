package engine

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// magicHeader prefixes every brain file written by this version. Older builds
// sniffed the format by looking at the first byte and assuming JSON, which
// mis-detected any file that merely started with whitespace or a brace and
// failed outright on a UTF-8 BOM. The explicit marker removes the guesswork
// while staying compatible with files written by previous versions.
var magicHeader = []byte("MYCOR-BRAIN\x00")

// BrainModel is the on-disk representation of a trained brain.
type BrainModel struct {
	Version    int                     `json:"version"`
	Vocabulary []string                `json:"vocabulary"`
	Weights    map[string]sparseVector `json:"weights"`
	Velocity   map[string]sparseVector `json:"velocity,omitempty"`
}

// sparseVector is a weight vector keyed by vocabulary index. It unmarshals
// from either the sparse object form or the legacy dense array form written by
// MYCOR v3/v4, so older brains still load.
type sparseVector map[int]float64

// UnmarshalJSON accepts both `{"3": 0.5}` and `[0.0, 0.1, 0.5]`.
func (v *sparseVector) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		*v = nil
		return nil
	}

	if trimmed[0] == '[' {
		var dense []float64
		if err := json.Unmarshal(trimmed, &dense); err != nil {
			return err
		}
		out := make(sparseVector, len(dense))
		for i, f := range dense {
			out[i] = f
		}
		*v = out
		return nil
	}

	var raw map[string]float64
	if err := json.Unmarshal(trimmed, &raw); err != nil {
		return err
	}
	out := make(sparseVector, len(raw))
	for key, f := range raw {
		var idx int
		if _, err := fmt.Sscanf(key, "%d", &idx); err != nil {
			return fmt.Errorf("bad weight index %q: %w", key, err)
		}
		out[idx] = f
	}
	*v = out
	return nil
}

// dense expands a sparse vector into a dense slice, used when the caller needs
// positional access.
func (v sparseVector) dense(size int) []float64 {
	out := make([]float64, size)
	for i, f := range v {
		if i >= 0 && i < size {
			out[i] = f
		}
	}
	return out
}

// Load reads a brain from disk, returning an error describing the problem.
//
// On failure the receiver is left untouched: the new state is fully validated
// before anything is published.
func (b *Brain) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	model, err := decodeBrain(data)
	if err != nil {
		return fmt.Errorf("decode brain %s: %w", path, err)
	}
	return b.applyModel(model)
}

// Save writes the brain to disk atomically.
func (b *Brain) Save(path string) error {
	b.mu.RLock()
	model := BrainModel{
		Version:    modelVersion,
		Vocabulary: append([]string(nil), b.vocab...),
		Weights:    cloneWeights(b.weights),
		Velocity:   cloneWeights(b.velocity),
	}
	b.mu.RUnlock()

	data, err := encodeBrain(&model)
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

// encodeBrain prefixes the gob payload with the magic header.
func encodeBrain(model *BrainModel) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(model); err != nil {
		return nil, fmt.Errorf("encode brain: %w", err)
	}
	out := make([]byte, 0, len(magicHeader)+buf.Len())
	out = append(out, magicHeader...)
	return append(out, buf.Bytes()...), nil
}

// decodeBrain detects the encoding without guessing from the first byte.
//
// Precedence: the magic header wins; otherwise leading whitespace, a UTF-8 BOM
// and a leading '{' are stripped and the document is treated as JSON; anything
// else is decoded as gob.
func decodeBrain(data []byte) (*BrainModel, error) {
	if len(data) == 0 {
		return nil, errors.New("empty file")
	}
	if bytes.HasPrefix(data, magicHeader) {
		var model BrainModel
		if err := gob.NewDecoder(bytes.NewReader(data[len(magicHeader):])).Decode(&model); err != nil {
			return nil, fmt.Errorf("invalid gob: %w", err)
		}
		return &model, nil
	}

	trimmed := bytes.TrimSpace(data)
	trimmed = bytes.TrimPrefix(trimmed, []byte("\xef\xbb\xbf"))
	trimmed = bytes.TrimSpace(trimmed)
	if len(trimmed) > 0 && trimmed[0] == '{' {
		var model BrainModel
		if err := json.Unmarshal(trimmed, &model); err != nil {
			return nil, fmt.Errorf("invalid JSON: %w", err)
		}
		return &model, nil
	}

	var model BrainModel
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&model); err != nil {
		return nil, fmt.Errorf("unrecognised brain format: %w", err)
	}
	return &model, nil
}

// applyModel validates a decoded brain and installs it.
func (b *Brain) applyModel(model *BrainModel) error {
	if model == nil {
		return errors.New("nil brain model")
	}
	vocab, err := normalizeVocabulary(model.Vocabulary)
	if err != nil {
		return err
	}
	if model.Weights == nil {
		model.Weights = map[string]sparseVector{}
	}

	// Drop entries pointing outside the vocabulary and the reserved <unk>
	// slot, then guarantee a velocity vector for every context.
	weights := make(map[string]sparseVector, len(model.Weights))
	for key, vec := range model.Weights {
		weights[key] = sanitizeVector(vec, len(vocab))
	}
	velocity := make(map[string]sparseVector, len(model.Weights))
	for key, vec := range model.Velocity {
		velocity[key] = sanitizeVector(vec, len(vocab))
	}
	for key := range weights {
		if _, ok := velocity[key]; !ok {
			velocity[key] = sparseVector{}
		}
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	b.vocab = vocab
	b.wordToIdx = make(map[string]int, len(vocab))
	for i, w := range vocab {
		b.wordToIdx[w] = i
	}
	b.weights = weights
	b.velocity = velocity
	b.undo = nil
	b.recentContexts = nil
	b.sessionHistory = nil
	b.lastThoughts = ""
	return nil
}

// normalizeVocabulary validates a stored vocabulary and guarantees that <unk>
// is present exactly once at index 0.
//
// The previous loader only prepended <unk> when it was missing from the front,
// so a brain that happened to store <unk> later in the list ended up with the
// token twice and a word->index map pointing at the wrong entry. Moving the
// token (and rejecting genuine duplicates) fixes both the duplicate and the
// resulting index corruption.
func normalizeVocabulary(vocab []string) ([]string, error) {
	if len(vocab) == 0 {
		return nil, errors.New("empty vocabulary")
	}
	seen := make(map[string]bool, len(vocab))
	unkSeen := false
	for _, w := range vocab {
		if w == "" {
			return nil, errors.New("empty word in vocabulary")
		}
		if w == unkToken {
			if unkSeen {
				return nil, fmt.Errorf("duplicate word in vocabulary: %q", w)
			}
			unkSeen = true
			continue
		}
		if seen[w] {
			return nil, fmt.Errorf("duplicate word in vocabulary: %q", w)
		}
		seen[w] = true
	}

	out := make([]string, 0, len(vocab)+1)
	out = append(out, unkToken)
	out = append(out, vocab...)
	if unkSeen {
		// Drop the later occurrence now that <unk> occupies index 0.
		filtered := out[:1]
		for _, w := range out[1:] {
			if w != unkToken {
				filtered = append(filtered, w)
			}
		}
		out = filtered
	}
	return out, nil
}

// sanitizeVector drops out-of-range indices and the reserved <unk> slot.
func sanitizeVector(vec sparseVector, size int) sparseVector {
	out := make(sparseVector, len(vec))
	for idx, f := range vec {
		if idx == unkIndex || idx < 0 || idx >= size {
			continue
		}
		if f != f { // NaN
			continue
		}
		out[idx] = f
	}
	return out
}

func cloneWeights(src map[string]sparseVector) map[string]sparseVector {
	out := make(map[string]sparseVector, len(src))
	for k, v := range src {
		out[k] = cloneSparse(v)
	}
	return out
}
