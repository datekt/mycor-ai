package engine

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"mycor/internal/config"
)

func newTestBrain(t *testing.T) *Brain {
	t.Helper()
	config.Reset()
	t.Cleanup(config.Reset)
	return NewBrain()
}

func trainThinkingCorpus(b *Brain, iterations int) {
	pairs := [][2]string{
		{"один", "два"},
		{"три", "четыре"},
		{"пять", "шесть"},
		{"семь", "восемь"},
		{"девять", "десять"},
	}
	for i := 0; i < iterations; i++ {
		for _, p := range pairs {
			b.Train(p[0], p[1])
		}
	}
}

func TestTokenize(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []string
	}{
		{"simple", "hello world", []string{"hello", "world"}},
		{"punct", "Привет, мир!", []string{"привет", ",", "мир", "!"}},
		{"extra spaces", "  a   b  ", []string{"a", "b"}},
		{"empty", "", nil},
		{"russian em dash", "привет — мир", []string{"привет", "—", "мир"}},
		{"russian ellipsis", "думаю… молчу", []string{"думаю", "…", "молчу"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Tokenize(c.in)
			if len(got) != len(c.want) {
				t.Fatalf("got %v, want %v", got, c.want)
			}
			for i := range got {
				if got[i] != c.want[i] {
					t.Errorf("idx %d: got %q, want %q", i, got[i], c.want[i])
				}
			}
		})
	}
}

func TestJoinWords(t *testing.T) {
	cases := []struct {
		name string
		in   []string
		want string
	}{
		{"simple", []string{"hello", "world"}, "hello world"},
		{"punct", []string{"привет", ",", "мир", "!"}, "привет, мир!"},
		{"emoticon", []string{"привет", "мир", ":)"}, "привет мир :)"},
		{"em dash", []string{"привет", "—", "мир"}, "привет — мир"},
		{"empty", nil, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := JoinWords(c.in); got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

func TestRegisterWord(t *testing.T) {
	b := newTestBrain(t)
	b.mu.Lock()
	defer b.mu.Unlock()

	if idx := b.RegisterWord("hello"); idx != 1 {
		t.Errorf("first registered word should be index 1, got %d", idx)
	}
	if idx := b.RegisterWord("hello"); idx != 1 {
		t.Errorf("duplicate should return same index, got %d", idx)
	}
	if idx := b.RegisterWord("world"); idx != 2 {
		t.Errorf("second word should be index 2, got %d", idx)
	}
	if len(b.vocab) != 3 {
		t.Errorf("vocab len = %d, want 3", len(b.vocab))
	}
}

func TestRegisterWordEmpty(t *testing.T) {
	b := newTestBrain(t)
	b.mu.Lock()
	defer b.mu.Unlock()

	if idx := b.RegisterWord(""); idx != unkIndex {
		t.Errorf("empty word should return unk index 0, got %d", idx)
	}
	if len(b.vocab) != 1 {
		t.Errorf("empty word should not extend vocabulary, got %d", len(b.vocab))
	}
}

// Registering words must stay O(1) per word: with the old dense vectors each new
// word rewrote every existing context vector, making a vocabulary build
// quadratic. This test asserts the observable consequence of the sparse layout.
func TestRegisterWordDoesNotTouchExistingContexts(t *testing.T) {
	b := newTestBrain(t)
	b.Train("alpha beta", "gamma")

	b.mu.Lock()
	before := b.weights[unkToken+" "+unkToken]
	if len(before) == 0 {
		b.mu.Unlock()
		t.Fatal("setup failed: expected a trained context")
	}
	for i := 0; i < 200; i++ {
		b.RegisterWord("word" + strings.Repeat("x", i%7) + string(rune('a'+i%26)) + itoa(i))
	}
	after := b.weights[unkToken+" "+unkToken]
	// A new word must not materialise entries in unrelated contexts.
	if len(after) != len(before) {
		t.Errorf("context grew from %d to %d entries after registering words",
			len(before), len(after))
	}
	b.mu.Unlock()
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [12]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(buf[pos:])
}

func TestContextWeightSize(t *testing.T) {
	b := newTestBrain(t)
	b.Train("a b", "c")

	size, ok := b.ContextWeightSize("a b")
	if !ok {
		t.Fatal("expected a b context to be present")
	}
	if size == 0 {
		t.Error("expected a non-empty sparse vector")
	}
	if _, ok := b.ContextWeightSize("missing key"); ok {
		t.Error("missing context should return false")
	}
}

func TestCountParameters(t *testing.T) {
	b := newTestBrain(t)
	if b.CountParameters() != 0 {
		t.Error("fresh brain should have no parameters")
	}
	b.Train("привет", "мир")
	if b.CountParameters() == 0 {
		t.Error("expected non-zero parameters after training")
	}
}
func TestBackoffChainDeduplicates(t *testing.T) {
	chain := backoffChain([]string{"x", unkToken})
	seen := make(map[string]bool)
	for _, k := range chain {
		if seen[k] {
			t.Errorf("duplicate in backoff chain: %q", k)
		}
		seen[k] = true
	}
	if len(chain) != 2 {
		t.Errorf("chain len = %d, want 2", len(chain))
	}
}

func TestBackoffChainFullLadder(t *testing.T) {
	chain := backoffChain([]string{"a", "b"})
	if len(chain) != 4 {
		t.Fatalf("chain len = %d, want 4", len(chain))
	}
	want := []string{"a b", unkToken + " b", "a " + unkToken, unkToken + " " + unkToken}
	for i := range want {
		if chain[i] != want[i] {
			t.Errorf("chain[%d] = %q, want %q", i, chain[i], want[i])
		}
	}
}

func TestBackoffChainThreeGram(t *testing.T) {
	chain := backoffChain([]string{"a", "b", "c"})
	if len(chain) != 6 {
		t.Fatalf("chain len = %d, want 6", len(chain))
	}
	if chain[0] != "a b c" {
		t.Errorf("first entry should be full context, got %q", chain[0])
	}
}

func TestBackoffChainUnigram(t *testing.T) {
	chain := backoffChain([]string{"a"})
	if len(chain) != 2 {
		t.Fatalf("chain len = %d, want 2", len(chain))
	}
	if chain[0] != "a" {
		t.Errorf("chain[0] = %q, want a", chain[0])
	}
	if chain[1] != unkToken {
		t.Errorf("chain[1] = %q, want %q", chain[1], unkToken)
	}
}

func TestMakeContextKey(t *testing.T) {
	if got := MakeContextKey(); got != "" {
		t.Errorf("empty key = %q, want empty", got)
	}
	if got := MakeContextKey("a", "b"); got != "a b" {
		t.Errorf("key = %q, want %q", got, "a b")
	}
}

// The backoff ladder must find a trained context even when some or all of the
// query words are unknown.
func TestDynamicBackoffReachesUnkContext(t *testing.T) {
	b := newTestBrain(t)
	for i := 0; i < 10; i++ {
		b.Train("a b", "c")
	}

	cfg := config.Get()
	cases := [][]string{{"a", "b"}, {"zzz", "a"}, {"a", "zzz"}, {"zzz", "yyy"}}
	for _, ctx := range cases {
		b.mu.Lock()
		_, ok := b.sampleNextIndexLocked(ctx, nil, cfg)
		b.mu.Unlock()
		if !ok {
			t.Errorf("expected backoff to find a context for %v", ctx)
		}
	}
}

func TestEmptyBrainSamplesNothing(t *testing.T) {
	b := newTestBrain(t)
	b.mu.Lock()
	defer b.mu.Unlock()
	if _, ok := b.sampleNextIndexLocked([]string{"a", "b"}, nil, config.Get()); ok {
		t.Fatal("empty brain should have no matching context")
	}
}

func TestSoftmaxSparseSumsToOne(t *testing.T) {
	b := newTestBrain(t)
	probs := b.softmaxSparse(sparseVector{1: 1, 2: 2, 3: 3}, 1, nil, 0)
	if len(probs) != 3 {
		t.Fatalf("len = %d, want 3", len(probs))
	}
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("sum = %f, want 1", sum)
	}
	if !(probs[1] < probs[2] && probs[2] < probs[3]) {
		t.Errorf("monotonicity broken: %v", probs)
	}
}

func TestSoftmaxSparsePenalty(t *testing.T) {
	b := newTestBrain(t)
	logits := sparseVector{1: 1, 2: 2, 3: 3}
	base := b.softmaxSparse(logits, 1, nil, 0)
	penalised := b.softmaxSparse(logits, 1, map[int]bool{3: true}, 2.0)
	if penalised[3] >= base[3] {
		t.Errorf("penalized prob should drop: %f vs %f", penalised[3], base[3])
	}
}

func TestSoftmaxSparseUniformOnDegenerateInput(t *testing.T) {
	b := newTestBrain(t)
	negInf := math.Inf(-1)
	logits := sparseVector{0: negInf, 1: negInf, 2: negInf}
	probs := b.softmaxSparse(logits, 1, nil, 0)
	expected := 1.0 / 3
	sum := 0.0
	for _, p := range probs {
		sum += p
		if math.Abs(p-expected) > 1e-9 {
			t.Errorf("prob = %v, want %v", p, expected)
		}
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("sum = %f, want 1", sum)
	}
}

func TestSoftmaxSparseEmpty(t *testing.T) {
	b := newTestBrain(t)
	if probs := b.softmaxSparse(nil, 1, nil, 0); probs != nil {
		t.Errorf("empty logits should yield nil, got %v", probs)
	}
}
func TestSelectTopKPicksLargest(t *testing.T) {
	probs := map[int]float64{0: 0.1, 1: 0.4, 2: 0.3, 3: 0.2}
	got := selectTopK(probs, 2)
	if len(got) != 2 {
		t.Fatalf("expected 2 candidates, got %d", len(got))
	}
	for _, idx := range got {
		if idx != 1 && idx != 2 {
			t.Errorf("unexpected candidate %d, want the two largest", idx)
		}
	}
}

func TestSelectTopKZeroKeepsAll(t *testing.T) {
	probs := map[int]float64{0: 0.1, 1: 0.4, 2: 0.3, 3: 0.2}
	if got := selectTopK(probs, 0); len(got) != len(probs) {
		t.Errorf("k=0 should keep all, got %d", len(got))
	}
}

func TestSelectTopKAtLeastDictionarySize(t *testing.T) {
	probs := map[int]float64{0: 0.1, 1: 0.4}
	if got := selectTopK(probs, 10); len(got) != 2 {
		t.Errorf("k >= len should keep all, got %d", len(got))
	}
}

// selectTopK is a bounded selection: it must never return more than k entries
// nor allocate proportionally to the vocabulary.
func TestSelectTopKBoundedByK(t *testing.T) {
	probs := make(map[int]float64, 5000)
	for i := 0; i < 5000; i++ {
		probs[i] = float64(i) / 5000
	}
	got := selectTopK(probs, 5)
	if len(got) != 5 {
		t.Fatalf("got %d candidates, want 5", len(got))
	}
	min := math.MaxFloat64
	for _, idx := range got {
		if probs[idx] < min {
			min = probs[idx]
		}
	}
	// Every entry strictly larger than the smallest selected one must be part
	// of the selection.
	for i, p := range probs {
		if p > min && !contains(got, i) {
			t.Errorf("missed entry %d with p=%v, above the selection floor %v", i, p, min)
		}
	}
}

func contains(haystack []int, needle int) bool {
	for _, v := range haystack {
		if v == needle {
			return true
		}
	}
	return false
}

func TestNarrowTopPCutsNucleus(t *testing.T) {
	probs := map[int]float64{0: 0.6, 1: 0.2, 2: 0.15, 3: 0.05}
	candidates := selectTopK(probs, len(probs))
	got := narrowTopP(candidates, probs, 0.5)
	if len(got) != 1 {
		t.Errorf("expected 1 candidate, got %d", len(got))
	}
	if len(got) == 1 && got[0] != 0 {
		t.Errorf("expected the most probable index, got %d", got[0])
	}
}

func TestNarrowTopPZeroKeepsArgmax(t *testing.T) {
	probs := map[int]float64{0: 0.1, 1: 0.4, 2: 0.3, 3: 0.2}
	got := narrowTopP(selectTopK(probs, len(probs)), probs, 0)
	if len(got) != 1 || got[0] != 1 {
		t.Errorf("p=0 should keep argmax 1, got %v", got)
	}
}

func TestNarrowTopPOneKeepsAll(t *testing.T) {
	probs := map[int]float64{0: 0.1, 1: 0.4, 2: 0.3, 3: 0.2}
	got := narrowTopP(selectTopK(probs, len(probs)), probs, 1.0)
	if len(got) != len(probs) {
		t.Errorf("p=1 should keep all, got %d", len(got))
	}
}

func TestNarrowTopPEmpty(t *testing.T) {
	if got := narrowTopP(nil, map[int]float64{}, 0.9); len(got) != 0 {
		t.Errorf("empty candidates should stay empty, got %v", got)
	}
}

func TestRenormalize(t *testing.T) {
	probs := map[int]float64{0: 2, 1: 2}
	renormalize(probs)
	if math.Abs(probs[0]-0.5) > 1e-9 || math.Abs(probs[1]-0.5) > 1e-9 {
		t.Errorf("probs = %v, want 0.5 each", probs)
	}
}
func TestTrainTeachesBackoffContexts(t *testing.T) {
	b := newTestBrain(t)
	b.Train("a b", "c")
	for _, key := range []string{"a b", unkToken + " b", "a " + unkToken, unkToken + " " + unkToken} {
		if _, ok := b.ContextWeightSize(key); !ok {
			t.Errorf("expected context %q to be trained", key)
		}
	}
}

func TestTrainUpdatesVocabulary(t *testing.T) {
	b := newTestBrain(t)
	b.Train("привет", "мир")
	if got := b.VocabularyLen(); got != 3 {
		t.Fatalf("vocab len = %d, want 3", got)
	}
}

func TestTrainEmptyInputIsNoop(t *testing.T) {
	b := newTestBrain(t)
	if loss := b.Train("", "мир"); loss != 0 {
		t.Errorf("empty prompt should produce zero loss, got %v", loss)
	}
	if loss := b.Train("привет", ""); loss != 0 {
		t.Errorf("empty target should produce zero loss, got %v", loss)
	}
	if b.VocabularyLen() != 1 {
		t.Error("invalid training should not extend the vocabulary")
	}
}

// Undo must restore the exact pre-training state, including the vocabulary,
// which is what the journal-based approach has to get right.
func TestUndoLastTrain(t *testing.T) {
	b := newTestBrain(t)
	b.Train("привет", "мир")
	if b.VocabularyLen() < 3 {
		t.Fatal("setup failed")
	}
	if !b.UndoLastTrain() {
		t.Fatal("undo should succeed")
	}
	if b.VocabularyLen() != 1 {
		t.Errorf("after undo vocab len = %d, want 1", b.VocabularyLen())
	}
	if b.CountParameters() != 0 {
		t.Errorf("after undo params = %d, want 0", b.CountParameters())
	}
	if b.UndoLastTrain() {
		t.Error("second undo should fail")
	}
}

func TestUndoRestoresPreviousWeights(t *testing.T) {
	b := newTestBrain(t)
	b.Train("a", "b")
	b.Train("c", "d")
	before := snapshotWeights(b)
	b.Train("a", "c")
	if !b.UndoLastTrain() {
		t.Fatal("undo failed")
	}
	after := snapshotWeights(b)
	if len(before) != len(after) {
		t.Fatalf("context count changed: %d -> %d", len(before), len(after))
	}
	for key, vec := range before {
		if len(after[key]) != len(vec) {
			t.Errorf("context %q: %d entries, want %d", key, len(after[key]), len(vec))
			continue
		}
		for idx, w := range vec {
			if math.Abs(after[key][idx]-w) > 1e-12 {
				t.Errorf("context %q index %d: %v, want %v", key, idx, after[key][idx], w)
			}
		}
	}
}

// The journal must copy only the contexts a step touches. If it copied every
// context, an unrelated training step would be silently reverted too.
func TestUndoOnlyRestoresTouchedContexts(t *testing.T) {
	b := newTestBrain(t)
	// "alpha beta" is unique to the first step: later training never puts it
	// into its own backoff chain.
	b.Train("alpha beta", "gamma")

	b.mu.RLock()
	untouchedBefore := cloneSparse(b.weights["alpha beta"])
	b.mu.RUnlock()
	if len(untouchedBefore) == 0 {
		t.Fatal("setup failed: expected a trained context")
	}

	b.Train("m n", "o")
	if !b.UndoLastTrain() {
		t.Fatal("undo failed")
	}

	b.mu.RLock()
	got := cloneSparse(b.weights["alpha beta"])
	b.mu.RUnlock()

	if len(got) != len(untouchedBefore) {
		t.Fatalf("untouched context changed size: %d -> %d",
			len(untouchedBefore), len(got))
	}
	for idx, w := range untouchedBefore {
		if math.Abs(got[idx]-w) > 1e-12 {
			t.Errorf("untouched context index %d changed: %v -> %v", idx, w, got[idx])
		}
	}
}

func TestTrainBatchDoesNotCreateUndoPoint(t *testing.T) {
	b := newTestBrain(t)
	b.TrainBatch("a", "b")
	if b.UndoLastTrain() {
		t.Error("TrainBatch should not create an undo point")
	}
}

func snapshotWeights(b *Brain) map[string]sparseVector {
	b.mu.RLock()
	defer b.mu.RUnlock()
	out := make(map[string]sparseVector, len(b.weights))
	for k, v := range b.weights {
		out[k] = cloneSparse(v)
	}
	return out
}
func TestSaveLoadRoundTrip(t *testing.T) {
	b := newTestBrain(t)
	b.Train("привет", "мир")
	b.Train("мир", "как дела")

	dir := t.TempDir()
	path := filepath.Join(dir, "brain.gob")
	if err := b.Save(path); err != nil {
		t.Fatalf("save failed: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat failed: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("saved file is empty")
	}

	vocabBefore := b.VocabularyLen()
	paramsBefore := b.CountParameters()

	restored := NewBrain()
	if err := restored.Load(path); err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if restored.VocabularyLen() != vocabBefore {
		t.Errorf("vocab = %d, want %d", restored.VocabularyLen(), vocabBefore)
	}
	if restored.CountParameters() != paramsBefore {
		t.Errorf("params = %d, want %d", restored.CountParameters(), paramsBefore)
	}
}

// The saved file must carry the magic marker rather than relying on a caller
// guessing the format from the first byte.
func TestSaveWritesMagicHeader(t *testing.T) {
	b := newTestBrain(t)
	b.Train("a", "b")

	path := filepath.Join(t.TempDir(), "brain.gob")
	if err := b.Save(path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(data), string(magicHeader)) {
		t.Error("saved brain is missing the magic header")
	}
}

func TestLoadLegacyDenseJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "brain.json")
	data := `{"version":4,"vocabulary":["<unk>","hello","world"],"weights":{"hello world":[0,0.5,0.3]}}`
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}

	b := newTestBrain(t)
	if err := b.Load(path); err != nil {
		t.Fatalf("legacy JSON load failed: %v", err)
	}
	if b.VocabularyLen() != 3 {
		t.Errorf("vocab = %d, want 3", b.VocabularyLen())
	}
	b.mu.RLock()
	vec := cloneSparse(b.weights["hello world"])
	b.mu.RUnlock()
	// The dense legacy array stored one slot per vocabulary entry; slot 0 is
	// the reserved <unk> and is dropped, leaving two stored weights.
	if len(vec) != 2 {
		t.Fatalf("weight entries = %d, want 2", len(vec))
	}
	// Index 0 is the reserved <unk> slot and is never stored.
	if _, ok := vec[unkIndex]; ok {
		t.Error("<unk> index should not be stored in a weight vector")
	}
	if math.Abs(vec[1]-0.5) > 1e-9 {
		t.Errorf("vec[1] = %v, want 0.5", vec[1])
	}
	if math.Abs(vec[2]-0.3) > 1e-9 {
		t.Errorf("vec[2] = %v, want 0.3", vec[2])
	}
}

// A JSON file with leading whitespace used to be mis-detected as gob, because
// the format probe looked at the first byte only.
func TestLoadJSONWithLeadingWhitespace(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "padded.json")
	data := "\n\t  {\"version\":5,\"vocabulary\":[\"<unk>\",\"a\"],\"weights\":{}}"
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}
	b := newTestBrain(t)
	if err := b.Load(path); err != nil {
		t.Fatalf("padded JSON should load: %v", err)
	}
	if b.VocabularyLen() != 2 {
		t.Errorf("vocab = %d, want 2", b.VocabularyLen())
	}
}
func TestLoadV3FixtureAddsUnkAtIndexZero(t *testing.T) {
	b := newTestBrain(t)
	path := filepath.Join("..", "..", "testdata", "brain_v3.json")
	if err := b.Load(path); err != nil {
		t.Fatalf("expected v3 fixture to load and migrate: %v", err)
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.vocab[0] != unkToken {
		t.Errorf("unk should be at index 0, got %q", b.vocab[0])
	}
	if len(b.vocab) != 5 {
		t.Errorf("vocab len = %d, want 5", len(b.vocab))
	}
	// Every index must round-trip through the word->index map correctly.
	for i, w := range b.vocab {
		if got := b.wordToIdx[w]; got != i {
			t.Errorf("wordToIdx[%q] = %d, want %d", w, got, i)
		}
	}
}

// A vocabulary that stores <unk> somewhere other than the front used to be
// given a second <unk>, which corrupted every index after it.
func TestLoadVocabularyWithUnkInTheMiddle(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "midunk.json")
	data := `{"version":5,"vocabulary":["hello","<unk>","world"],"weights":{"hello <unk>":[0.1,0.2,0.3]}}`
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}

	b := newTestBrain(t)
	if err := b.Load(path); err != nil {
		t.Fatalf("load failed: %v", err)
	}
	b.mu.RLock()
	defer b.mu.RUnlock()

	seen := map[string]int{}
	for i, w := range b.vocab {
		if prev, dup := seen[w]; dup {
			t.Fatalf("duplicate %q at %d and %d", w, prev, i)
		}
		seen[w] = i
	}
	if b.vocab[0] != unkToken {
		t.Errorf("unk should be moved to index 0, got %q", b.vocab[0])
	}
	if b.wordToIdx["hello"] != 1 || b.wordToIdx["world"] != 2 {
		t.Errorf("indices corrupted: hello=%d world=%d",
			b.wordToIdx["hello"], b.wordToIdx["world"])
	}
}

func TestLoadRejectsDuplicateVocabulary(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "dup.json")
	data := `{"version":4,"vocabulary":["<unk>","a","a"],"weights":{}}`
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}
	b := newTestBrain(t)
	if err := b.Load(path); err == nil {
		t.Error("expected duplicate vocabulary to be rejected")
	}
}

func TestLoadRejectsEmptyVocabulary(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.json")
	if err := os.WriteFile(path, []byte(`{"version":5,"vocabulary":[]}`), 0644); err != nil {
		t.Fatal(err)
	}
	b := newTestBrain(t)
	if err := b.Load(path); err == nil {
		t.Error("expected empty vocabulary to be rejected")
	}
}

// A failed load must leave the brain untouched so a corrupt file cannot
// destroy the working model.
func TestLoadFailureLeavesStateUntouched(t *testing.T) {
	b := newTestBrain(t)
	b.Train("привет", "мир")
	before := b.VocabularyLen()

	for _, fixture := range []string{"brain_corrupt.json", "brain_duplicate.json"} {
		path := filepath.Join("..", "..", "testdata", fixture)
		if err := b.Load(path); err == nil {
			t.Errorf("expected %s to fail to load", fixture)
		}
		if b.VocabularyLen() != before {
			t.Errorf("after %s vocab = %d, want %d", fixture, b.VocabularyLen(), before)
		}
		b.mu.RLock()
		if b.vocab[0] != unkToken {
			t.Errorf("after %s unk is no longer first", fixture)
		}
		b.mu.RUnlock()
	}
}

func TestLoadV4Fixture(t *testing.T) {
	b := newTestBrain(t)
	path := filepath.Join("..", "..", "testdata", "brain_v4.json")
	if err := b.Load(path); err != nil {
		t.Fatalf("expected v4 fixture to load: %v", err)
	}
	if b.VocabularyLen() != 5 {
		t.Errorf("vocab = %d, want 5", b.VocabularyLen())
	}
}

func TestLoadMissingFile(t *testing.T) {
	b := newTestBrain(t)
	if err := b.Load(filepath.Join(t.TempDir(), "nope.gob")); err == nil {
		t.Error("expected an error for a missing brain file")
	}
}

func TestDecodeBrainRejectsEmpty(t *testing.T) {
	if _, err := decodeBrain(nil); err == nil {
		t.Error("expected an error for empty input")
	}
}

func TestDecodeBrainRejectsGarbage(t *testing.T) {
	if _, err := decodeBrain([]byte("not a brain at all")); err == nil {
		t.Error("expected an error for unrecognised content")
	}
}
func TestGenerateResponseNoTrain(t *testing.T) {
	b := newTestBrain(t)
	if out := b.GenerateResponse("привет", 5); out != "" {
		t.Errorf("untrained brain should return empty string, got %q", out)
	}
	if b.LastThoughts() != "" {
		t.Errorf("untrained brain should have no thoughts, got %q", b.LastThoughts())
	}
}

func TestGenerateResponseAfterTrain(t *testing.T) {
	b := newTestBrain(t)
	for i := 0; i < 10; i++ {
		b.Train("привет мир", "как дела")
	}
	if out := b.GenerateResponse("привет мир", 5); out == "" {
		t.Error("expected non-empty output")
	}
}

func TestGenerateResponseNeverEmitsUnk(t *testing.T) {
	b := newTestBrain(t)
	b.Train("привет мир", "как дела")
	for attempt := 0; attempt < 20; attempt++ {
		out := b.GenerateResponse("совершенносекретноеслово", 5)
		if strings.Contains(out, unkToken) {
			t.Fatalf("output should not contain %q, got %q", unkToken, out)
		}
	}
}

func TestGenerateResponseZeroLength(t *testing.T) {
	b := newTestBrain(t)
	b.Train("a", "b")
	if out := b.GenerateResponse("a", 0); out != "" {
		t.Errorf("maxLen 0 should produce empty output, got %q", out)
	}
}

func TestRecentContexts(t *testing.T) {
	b := newTestBrain(t)
	if got := b.RecentContexts(); len(got) != 0 {
		t.Errorf("expected empty, got %v", got)
	}
	b.Train("a", "b")
	b.Train("c", "d")
	got := b.RecentContexts()
	if len(got) == 0 {
		t.Fatal("expected non-empty recent contexts")
	}
	for i := 1; i < len(got); i++ {
		for j := 0; j < i; j++ {
			if got[i] == got[j] {
				t.Errorf("duplicate context in recent list: %q", got[i])
			}
		}
	}
}

// RecentContexts returns a copy, so a caller cannot corrupt internal state.
func TestRecentContextsReturnsCopy(t *testing.T) {
	b := newTestBrain(t)
	b.Train("a", "b")
	got := b.RecentContexts()
	if len(got) == 0 {
		t.Fatal("expected contexts")
	}
	got[0] = "tampered"
	if b.RecentContexts()[0] == "tampered" {
		t.Error("RecentContexts exposed internal state")
	}
}

func TestSessionHistory(t *testing.T) {
	b := newTestBrain(t)
	b.AppendHistory("user", "привет")
	b.AppendHistory("ai", "мир")
	if got := b.SessionHistory(); len(got) != 2 {
		t.Fatalf("session history len = %d, want 2", len(got))
	}
	b.Reset()
	if got := b.SessionHistory(); len(got) != 0 {
		t.Error("session history should reset")
	}
}

func TestAppendHistoryIgnoresEmptyText(t *testing.T) {
	b := newTestBrain(t)
	b.AppendHistory("user", "")
	if got := b.SessionHistory(); len(got) != 0 {
		t.Errorf("empty message should not be stored, got %v", got)
	}
}

func TestSessionHistoryRespectsCap(t *testing.T) {
	b := newTestBrain(t)
	limit := config.Get().MaxSessionHistory
	for i := 0; i < limit+25; i++ {
		b.AppendHistory("user", "message")
	}
	if got := b.SessionHistory(); len(got) != limit {
		t.Errorf("history len = %d, want %d", len(got), limit)
	}
}

func TestThinkRequiresVocabulary(t *testing.T) {
	b := newTestBrain(t)
	b.Train("a", "b")
	if got := b.Think("a"); len(got) != 0 {
		t.Errorf("think should be empty with small vocab, got %v", got)
	}
	if b.LastThoughts() != "" {
		t.Errorf("LastThoughts should be empty, got %q", b.LastThoughts())
	}
}

func TestThinkAfterEnoughTraining(t *testing.T) {
	b := newTestBrain(t)
	trainThinkingCorpus(b, 10)
	if b.VocabularyLen() < MinVocabForThinking() {
		t.Fatalf("setup failed: vocab = %d", b.VocabularyLen())
	}
	for attempt := 0; attempt < 20; attempt++ {
		if got := b.Think("один"); len(got) > 0 {
			return
		}
	}
	t.Fatal("expected thoughts after retries")
}

// Thinking steps moved from a compile-time constant into the configuration.
func TestThinkingStepsAreConfigurable(t *testing.T) {
	b := newTestBrain(t)
	trainThinkingCorpus(b, 10)

	if _, err := config.SetField("thinkingSteps", 1, false); err != nil {
		t.Fatal(err)
	}
	for attempt := 0; attempt < 20; attempt++ {
		if got := b.Think("один"); len(got) > 1 {
			t.Fatalf("expected at most 1 thought, got %d", len(got))
		}
	}

	if _, err := config.SetField("thinkingSteps", 8, false); err != nil {
		t.Fatal(err)
	}
	if got := config.Get().ThinkingSteps; got != 8 {
		t.Errorf("ThinkingSteps = %d, want 8", got)
	}
}
func TestThinkingResetOnReset(t *testing.T) {
	b := newTestBrain(t)
	trainThinkingCorpus(b, 10)
	for attempt := 0; attempt < 20; attempt++ {
		b.GenerateResponse("один", 3)
		if b.LastThoughts() != "" {
			break
		}
	}
	b.Reset()
	if b.LastThoughts() != "" {
		t.Errorf("thoughts should reset, got %q", b.LastThoughts())
	}
}

func TestGenerateIdleThoughtEmptyForSmallVocab(t *testing.T) {
	b := newTestBrain(t)
	if got := b.GenerateIdleThought(); got != "" {
		t.Errorf("expected empty idle thought with small vocab, got %q", got)
	}
}

func TestGenerateIdleThoughtNeverSeedsUnk(t *testing.T) {
	b := newTestBrain(t)
	trainThinkingCorpus(b, 10)
	for attempt := 0; attempt < 50; attempt++ {
		if got := b.GenerateIdleThought(); strings.Contains(got, unkToken) {
			t.Fatalf("iteration %d produced %q containing %q", attempt, got, unkToken)
		}
	}
}

func TestStatsSnapshot(t *testing.T) {
	b := newTestBrain(t)
	b.Train("a b", "c")
	stats := b.Stats()
	if stats.Vocabulary != 4 {
		t.Errorf("vocabulary = %d, want 4", stats.Vocabulary)
	}
	if stats.Parameters == 0 {
		t.Error("expected parameters > 0")
	}
	if stats.MinThinking != config.Get().MinVocabThinking {
		t.Errorf("minThinking = %d, want %d", stats.MinThinking, config.Get().MinVocabThinking)
	}
}

// Concurrent chat, training and statistics access must be race-free. Run with
// -race to get the full benefit.
func TestConcurrentChatAndTraining(t *testing.T) {
	b := newTestBrain(t)

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 30; j++ {
				b.Train("word"+itoa(i), "word"+itoa(j))
			}
		}(i)
	}
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 30; j++ {
				b.GenerateResponse("привет", 5)
				_ = b.Stats()
				_ = b.RecentContexts()
				b.AppendHistory("user", "hi")
			}
		}()
	}
	wg.Wait()
}

func TestConcurrentUndoIsSafe(t *testing.T) {
	b := newTestBrain(t)
	b.Train("a", "b")

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			b.Train("x", "y")
		}
	}()
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				b.UndoLastTrain()
			}
		}()
	}
	wg.Wait()
}

// BenchmarkVocabularyBuild demonstrates the effect of the sparse layout: the
// old dense vectors made registration quadratic in the number of contexts.
func BenchmarkVocabularyBuild(b *testing.B) {
	br := NewBrain()
	for i := 0; i < 400; i++ {
		br.Train("ctx"+itoa(i), "a b")
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		br.mu.Lock()
		br.RegisterWord("new" + itoa(i))
		br.mu.Unlock()
	}
}
