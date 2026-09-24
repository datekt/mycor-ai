package engine

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

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
	InitEngine()
	if RegisterWord("hello") != 1 {
		t.Errorf("first registered word should be index 1")
	}
	if RegisterWord("hello") != 1 {
		t.Errorf("duplicate should return same index")
	}
	if RegisterWord("world") != 2 {
		t.Errorf("second word should be index 2")
	}
	if len(Vocabulary) != 3 {
		t.Errorf("vocab len = %d, want 3", len(Vocabulary))
	}
}

func TestSoftmaxBase(t *testing.T) {
	logits := []float64{1, 2, 3}
	probs := softmaxBase(logits, 1, nil, 0)
	if len(probs) != 3 {
		t.Fatalf("len = %d", len(probs))
	}
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("sum = %f, want 1", sum)
	}
	if !(probs[0] < probs[1] && probs[1] < probs[2]) {
		t.Errorf("monotonicity broken: %v", probs)
	}
}

func TestSoftmaxRepetitionPenalty(t *testing.T) {
	logits := []float64{1, 2, 3}
	base := softmaxBase(logits, 1, nil, 0)
	used := map[int]bool{2: true}
	pen := softmaxBase(logits, 1, used, 2.0)
	if pen[2] >= base[2] {
		t.Errorf("penalized prob should drop: %f vs %f", pen[2], base[2])
	}
}

func TestTrainUpdatesWeights(t *testing.T) {
	InitEngine()
	Train("привет", "мир")
	if len(Vocabulary) != 3 {
		t.Fatalf("vocab len = %d, want 3", len(Vocabulary))
	}
	if CountParameters() == 0 {
		t.Error("expected non-zero parameters after training")
	}
}

func TestUndoLastTrain(t *testing.T) {
	InitEngine()
	Train("привет", "мир")
	if len(Vocabulary) < 3 {
		t.Fatal("setup failed")
	}
	if !UndoLastTrain() {
		t.Fatal("undo should succeed")
	}
	if len(Vocabulary) != 1 {
		t.Errorf("after undo vocab len = %d, want 1", len(Vocabulary))
	}
	if UndoLastTrain() {
		t.Error("second undo should fail")
	}
}

func TestUndoRestoresWeights(t *testing.T) {
	InitEngine()
	Train("a", "b")
	Train("c", "d")
	paramsBefore := CountParameters()
	Train("a", "c")
	if !UndoLastTrain() {
		t.Fatal("undo failed")
	}
	if CountParameters() != paramsBefore {
		t.Errorf("params = %d, want %d", CountParameters(), paramsBefore)
	}
}

func TestTrainBatchDoesNotBackup(t *testing.T) {
	InitEngine()
	TrainBatch("a", "b")
	if BackupVocabulary != nil {
		t.Error("TrainBatch should not create a backup")
	}
}

func TestSaveLoadBrain(t *testing.T) {
	InitEngine()
	Train("привет", "мир")
	Train("мир", "как дела")

	dir := t.TempDir()
	path := filepath.Join(dir, "brain.json")
	SaveBrain(path)

	vocabBefore := len(Vocabulary)
	paramsBefore := CountParameters()

	InitEngine()
	if !LoadBrain(path) {
		t.Fatal("load failed")
	}
	if len(Vocabulary) != vocabBefore {
		t.Errorf("vocab = %d, want %d", len(Vocabulary), vocabBefore)
	}
	if CountParameters() != paramsBefore {
		t.Errorf("params = %d, want %d", CountParameters(), paramsBefore)
	}
}

func TestLoadBrainInvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(path, []byte("{broken"), 0644); err != nil {
		t.Fatal(err)
	}
	if LoadBrain(path) {
		t.Error("expected failure on bad JSON")
	}
}

func TestLoadBrainEmptyVocabulary(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "brain.json")
	if err := os.WriteFile(path, []byte(`{"vocabulary":[],"weights":{}}`), 0644); err != nil {
		t.Fatal(err)
	}
	if LoadBrain(path) {
		t.Error("expected failure on empty vocabulary")
	}
}

func TestLoadBrainAddsUnk(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "brain.json")
	data := `{"vocabulary":["hello","world"],"weights":{}}`
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}
	if !LoadBrain(path) {
		t.Fatal("load failed")
	}
	if Vocabulary[0] != "<unk>" {
		t.Errorf("first word = %q, want <unk>", Vocabulary[0])
	}
	if len(Vocabulary) != 3 {
		t.Errorf("vocab len = %d, want 3", len(Vocabulary))
	}
}

func TestLoadBrainRealignsWeights(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "brain.json")
	data := `{"vocabulary":["<unk>","a","b"],"weights":{"<unk> a":[0.1,0.2]}}`
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}
	if !LoadBrain(path) {
		t.Fatal("load failed")
	}
	w := Weights["<unk> a"]
	if len(w) != 3 {
		t.Errorf("weight len = %d, want 3", len(w))
	}
}

func TestGenerateResponseNoTrain(t *testing.T) {
	InitEngine()
	out := GenerateResponse("привет", 5)
	_ = out
}

func TestGenerateResponseAfterTrain(t *testing.T) {
	InitEngine()
	for i := 0; i < 10; i++ {
		Train("привет мир", "как дела")
	}
	out := GenerateResponse("привет мир", 5)
	if out == "" {
		t.Error("expected non-empty output")
	}
}

func TestGenerateResponseUnknownWord(t *testing.T) {
	InitEngine()
	Train("привет мир", "как дела")
	out := GenerateResponse("совершенносекретноеслово", 3)
	_ = out
}

func TestRecentContexts(t *testing.T) {
	InitEngine()
	if got := RecentContexts(); len(got) != 0 {
		t.Errorf("expected empty, got %v", got)
	}
	Train("a", "b")
	Train("c", "d")
	got := RecentContexts()
	if len(got) == 0 {
		t.Error("expected non-empty recent contexts")
	}
	for i := 1; i < len(got); i++ {
		for j := 0; j < i; j++ {
			if got[i] == got[j] {
				t.Errorf("duplicate context in recent list: %q", got[i])
			}
		}
	}
}
