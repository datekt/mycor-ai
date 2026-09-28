package engine

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func trainThinkingCorpus(t *testing.T, iterations int) {
	t.Helper()
	pairs := [][2]string{
		{"один", "два"},
		{"три", "четыре"},
		{"пять", "шесть"},
		{"семь", "восемь"},
		{"девять", "десять"},
	}
	for i := 0; i < iterations; i++ {
		for _, p := range pairs {
			Train(p[0], p[1])
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

func TestTokenizeEmoticons(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"привет :)", []string{"привет", ":)"}},
		{"hello :D world", []string{"hello", ":d", "world"}},
		{":) :) :)", []string{":)", ":)", ":)"}},
		{"hi <3", []string{"hi", "<3"}},
		{"o_o what", []string{"o_o", "what"}},
		{"good :-)", []string{"good", ":-)"}},
		{"sad :(", []string{"sad", ":("}},
		{"crying :'(", []string{"crying", ":'("}},
	}
	for _, c := range cases {
		got := Tokenize(c.in)
		if len(got) != len(c.want) {
			t.Fatalf("Tokenize(%q) = %v, want %v", c.in, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("Tokenize(%q)[%d] = %q, want %q", c.in, i, got[i], c.want[i])
			}
		}
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

func TestRegisterWordEmpty(t *testing.T) {
	InitEngine()
	if RegisterWord("") != 0 {
		t.Errorf("empty word should return unk index 0")
	}
	if len(Vocabulary) != 1 {
		t.Errorf("empty word should not extend vocabulary")
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

func TestSoftmaxBaseUniformOnDegenerateInput(t *testing.T) {
	logits := []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1)}
	probs := softmaxBase(logits, 1, nil, 0)
	if len(probs) != len(logits) {
		t.Fatalf("len = %d, want %d", len(probs), len(logits))
	}
	expected := 1.0 / float64(len(logits))
	sum := 0.0
	for _, p := range probs {
		sum += p
		if math.Abs(p-expected) > 1e-9 {
			t.Errorf("prob = %v, want %v", p, expected)
		}
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("sum = %v, want 1", sum)
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

func TestBackoffChainDeduplicates(t *testing.T) {
	chain := backoffChain("x", unkToken)
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
	chain := backoffChain("a", "b")
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

func TestDynamicBackoffEmptyEngine(t *testing.T) {
	InitEngine()
	_, ok := sampleNextIndex("a", "b", nil, 1.0)
	if ok {
		t.Fatal("empty engine should have no matching context")
	}
}

func TestDynamicBackoffUsesUnkPrefix(t *testing.T) {
	InitEngine()
	for i := 0; i < 10; i++ {
		Train("a b", "c")
	}
	_, ok := sampleNextIndex("zzz", "a", nil, 1.0)
	if !ok {
		t.Fatal("expected backoff to find 'unk a' context")
	}
}

func TestDynamicBackoffUsesUnkSuffix(t *testing.T) {
	InitEngine()
	for i := 0; i < 10; i++ {
		Train("a b", "c")
	}
	_, ok := sampleNextIndex("a", "zzz", nil, 1.0)
	if !ok {
		t.Fatal("expected backoff to find 'a unk' context")
	}
}

func TestDynamicBackoffFallsBackToUnkUnk(t *testing.T) {
	InitEngine()
	for i := 0; i < 10; i++ {
		Train("a b", "c")
	}
	_, ok := sampleNextIndex("zzz", "yyy", nil, 1.0)
	if !ok {
		t.Fatal("expected backoff to reach unigram context")
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

func TestTrainTeachesBackoffContexts(t *testing.T) {
	InitEngine()
	Train("a b", "c")
	for _, key := range []string{"a b", unkToken + " b", "a " + unkToken, unkToken + " " + unkToken} {
		if _, ok := weights[key]; !ok {
			t.Errorf("expected context %q to be trained", key)
		}
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
	if err := SaveBrain(path); err != nil {
		t.Fatalf("save failed: %v", err)
	}

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

func TestSaveLoadBrainPersistsVelocity(t *testing.T) {
	InitEngine()
	Train("a", "b")
	Train("a", "b")

	dir := t.TempDir()
	path := filepath.Join(dir, "brain.json")
	if err := SaveBrain(path); err != nil {
		t.Fatal(err)
	}

	InitEngine()
	if !LoadBrain(path) {
		t.Fatal("load failed")
	}
	if len(velocity) == 0 {
		t.Error("velocity should be restored")
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

func TestLoadBrainCorruptFixture(t *testing.T) {
	InitEngine()
	path := filepath.Join("..", "..", "testdata", "brain_corrupt.json")
	if LoadBrain(path) {
		t.Fatal("expected LoadBrain to fail on corrupt fixture")
	}
	if len(Vocabulary) != 1 || Vocabulary[0] != unkToken {
		t.Errorf("state should stay untouched on failure, got %v", Vocabulary)
	}
}

func TestLoadBrainDuplicateFixture(t *testing.T) {
	InitEngine()
	path := filepath.Join("..", "..", "testdata", "brain_duplicate.json")
	if LoadBrain(path) {
		t.Fatal("expected LoadBrain to fail on duplicate fixture")
	}
	if len(Vocabulary) != 1 || Vocabulary[0] != unkToken {
		t.Errorf("state should stay untouched on failure, got %v", Vocabulary)
	}
}

func TestLoadBrainV3Fixture(t *testing.T) {
	InitEngine()
	path := filepath.Join("..", "..", "testdata", "brain_v3.json")
	if !LoadBrain(path) {
		t.Fatal("expected v3 fixture to load and migrate")
	}
	if Vocabulary[0] != unkToken {
		t.Errorf("unk should be inserted at index 0, got %q", Vocabulary[0])
	}
	if len(Vocabulary) != 5 {
		t.Errorf("vocab len = %d, want 5", len(Vocabulary))
	}
	for k, w := range weights {
		if len(w) != len(Vocabulary) {
			t.Errorf("context %q weight len = %d, want %d", k, len(w), len(Vocabulary))
		}
	}
}

func TestLoadBrainV4Fixture(t *testing.T) {
	InitEngine()
	path := filepath.Join("..", "..", "testdata", "brain_v4.json")
	if !LoadBrain(path) {
		t.Fatal("expected v4 fixture to load")
	}
	if len(Vocabulary) != 5 {
		t.Errorf("vocab len = %d, want 5", len(Vocabulary))
	}
	if _, ok := weights["hello world"]; !ok {
		t.Error("expected hello world context in weights")
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

func TestLoadBrainDuplicateWord(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "brain.json")
	data := `{"vocabulary":["<unk>","a","a"],"weights":{}}`
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}
	if LoadBrain(path) {
		t.Error("expected failure on duplicate word")
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
	w := weights["<unk> a"]
	if len(w) != 3 {
		t.Errorf("weight len = %d, want 3", len(w))
	}
	if len(velocity["<unk> a"]) != 3 {
		t.Errorf("velocity len = %d, want 3", len(velocity["<unk> a"]))
	}
}

func TestGenerateResponseNoTrain(t *testing.T) {
	InitEngine()
	out := GenerateResponse("привет", 5)
	if out != "" {
		t.Errorf("untrained engine should return empty string, got %q", out)
	}
	if lastThoughts != "" {
		t.Errorf("untrained engine should have no thoughts, got %q", lastThoughts)
	}
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
	if strings.Contains(out, unkToken) {
		t.Errorf("output should not contain %q, got %q", unkToken, out)
	}
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

func TestSessionHistoryReset(t *testing.T) {
	InitEngine()
	AppendHistory("user", "привет")
	AppendHistory("ai", "мир")
	if len(SessionHistory()) != 2 {
		t.Fatalf("session history len = %d, want 2", len(SessionHistory()))
	}
	InitEngine()
	if len(SessionHistory()) != 0 {
		t.Error("session history should reset on InitEngine")
	}
}

func TestSessionHistorySurvivesLoadBrain(t *testing.T) {
	InitEngine()
	Train("a", "b")
	AppendHistory("user", "a")
	AppendHistory("ai", "b")

	dir := t.TempDir()
	path := filepath.Join(dir, "brain.json")
	if err := SaveBrain(path); err != nil {
		t.Fatal(err)
	}

	InitEngine()
	if !LoadBrain(path) {
		t.Fatal("load failed")
	}
	if len(SessionHistory()) != 0 {
		t.Error("session history should reset after LoadBrain")
	}
	if len(Vocabulary) < 3 {
		t.Error("vocabulary should be restored after LoadBrain")
	}
}

func TestThinkRequiresVocabulary(t *testing.T) {
	InitEngine()
	Train("a", "b")
	if got := Think("a"); len(got) != 0 {
		t.Errorf("think should be empty with small vocab, got %v", got)
	}
	if LastThoughts() != "" {
		t.Errorf("LastThoughts should be empty, got %q", LastThoughts())
	}
}

func TestThinkAfterEnoughTraining(t *testing.T) {
	InitEngine()
	trainThinkingCorpus(t, 10)
	if len(Vocabulary) < MinVocabForThinking {
		t.Fatalf("setup failed: vocab = %d", len(Vocabulary))
	}

	var got []string
	for attempt := 0; attempt < 20; attempt++ {
		got = Think("один")
		if len(got) > 0 {
			break
		}
	}
	if len(got) == 0 {
		t.Fatal("expected thoughts after retries")
	}
}

func TestGenerateResponseWithThinking(t *testing.T) {
	InitEngine()
	trainThinkingCorpus(t, 10)

	out := GenerateResponse("один", 5)
	if out == "" {
		t.Error("expected non-empty output")
	}

	var thoughts string
	for attempt := 0; attempt < 20; attempt++ {
		GenerateResponse("один", 5)
		if thoughts = LastThoughts(); thoughts != "" {
			break
		}
	}
	if thoughts == "" {
		t.Fatal("expected non-empty thoughts after retries")
	}
}

func TestThinkingResetOnInit(t *testing.T) {
	InitEngine()
	trainThinkingCorpus(t, 10)

	var populated bool
	for attempt := 0; attempt < 20; attempt++ {
		GenerateResponse("один", 3)
		if LastThoughts() != "" {
			populated = true
			break
		}
	}
	if !populated {
		t.Fatal("expected thoughts after retries")
	}

	InitEngine()
	if LastThoughts() != "" {
		t.Errorf("thoughts should reset, got %q", LastThoughts())
	}
}

func TestGenerateIdleThoughtEmpty(t *testing.T) {
	InitEngine()
	if got := GenerateIdleThought(); got != "" {
		t.Errorf("expected empty idle thought with small vocab, got %q", got)
	}
}

func TestGenerateIdleThoughtWithVocab(t *testing.T) {
	InitEngine()
	trainThinkingCorpus(t, 10)
	if len(Vocabulary) < MinVocabForThinking {
		t.Fatalf("setup failed: vocab = %d", len(Vocabulary))
	}
	got := GenerateIdleThought()
	if strings.Contains(got, unkToken) {
		t.Errorf("idle thought must never contain %q, got %q", unkToken, got)
	}
}

func TestGenerateIdleThoughtNeverSeedsUnk(t *testing.T) {
	InitEngine()
	trainThinkingCorpus(t, 10)
	for attempt := 0; attempt < 50; attempt++ {
		got := GenerateIdleThought()
		if strings.Contains(got, unkToken) {
			t.Fatalf("iteration %d produced %q containing %q", attempt, got, unkToken)
		}
	}
}

func TestContextWeightSize(t *testing.T) {
	InitEngine()
	Train("a b", "c")
	size, ok := ContextWeightSize("a b")
	if !ok {
		t.Fatal("expected a b context to be present")
	}
	if size != len(Vocabulary) {
		t.Errorf("size = %d, want %d", size, len(Vocabulary))
	}
	if _, ok := ContextWeightSize("missing key"); ok {
		t.Error("missing context should return false")
	}
}
