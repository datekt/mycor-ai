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
		if _, ok := Weights[key]; !ok {
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
	if len(Velocity) == 0 {
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
	w := Weights["<unk> a"]
	if len(w) != 3 {
		t.Errorf("weight len = %d, want 3", len(w))
	}
	if len(Velocity["<unk> a"]) != 3 {
		t.Errorf("velocity len = %d, want 3", len(Velocity["<unk> a"]))
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
	pairs := [][2]string{
		{"один", "два"},
		{"три", "четыре"},
		{"пять", "шесть"},
		{"семь", "восемь"},
		{"девять", "десять"},
	}
	for i := 0; i < 5; i++ {
		for _, p := range pairs {
			Train(p[0], p[1])
		}
	}
	if len(Vocabulary) < MinVocabForThinking {
		t.Fatalf("setup failed: vocab = %d", len(Vocabulary))
	}
	thoughts := Think("один")
	if len(thoughts) == 0 {
		t.Error("expected thoughts with large vocab")
	}
}

func TestGenerateResponseWithThinking(t *testing.T) {
	InitEngine()
	pairs := [][2]string{
		{"один", "два"},
		{"три", "четыре"},
		{"пять", "шесть"},
		{"семь", "восемь"},
		{"девять", "десять"},
	}
	for i := 0; i < 5; i++ {
		for _, p := range pairs {
			Train(p[0], p[1])
		}
	}
	out := GenerateResponse("один", 5)
	if out == "" {
		t.Error("expected non-empty output")
	}
	if LastThoughts() == "" {
		t.Error("expected non-empty thoughts after large training")
	}
}

func TestThinkingResetOnInit(t *testing.T) {
	InitEngine()
	pairs := [][2]string{
		{"один", "два"},
		{"три", "четыре"},
		{"пять", "шесть"},
		{"семь", "восемь"},
		{"девять", "десять"},
	}
	for i := 0; i < 5; i++ {
		for _, p := range pairs {
			Train(p[0], p[1])
		}
	}
	GenerateResponse("один", 3)
	if LastThoughts() == "" {
		t.Fatal("expected thoughts to be populated")
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
	pairs := [][2]string{
		{"один", "два"},
		{"три", "четыре"},
		{"пять", "шесть"},
		{"семь", "восемь"},
		{"девять", "десять"},
	}
	for i := 0; i < 5; i++ {
		for _, p := range pairs {
			Train(p[0], p[1])
		}
	}
	if len(Vocabulary) < MinVocabForThinking {
		t.Fatalf("setup failed: vocab = %d", len(Vocabulary))
	}
	got := GenerateIdleThought()
	_ = got
}
