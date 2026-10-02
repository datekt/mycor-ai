package importer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"mycor/internal/config"
	"mycor/internal/engine"
)

func newBrain(t *testing.T) *engine.Brain {
	t.Helper()
	config.Reset()
	t.Cleanup(config.Reset)
	return engine.NewBrain()
}

func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestImportTxtFile(t *testing.T) {
	dir := t.TempDir()
	src := writeFile(t, dir, "input.txt", "привет мир\nкак дела\nвсё хорошо\n")

	b := newBrain(t)
	brain := filepath.Join(dir, "brain.gob")
	result, err := ImportTxtFile(b, src, brain, nil)
	if err != nil {
		t.Fatalf("import failed: %v", err)
	}
	if _, err := os.Stat(brain); err != nil {
		t.Fatalf("brain file not created: %v", err)
	}
	if result.Trained != 2 {
		t.Errorf("trained = %d, want 2", result.Trained)
	}
	if b.VocabularyLen() < 4 {
		t.Errorf("vocab too small: %d", b.VocabularyLen())
	}
}

func TestImportTxtFileMissing(t *testing.T) {
	dir := t.TempDir()
	b := newBrain(t)
	if _, err := ImportTxtFile(b, filepath.Join(dir, "nope.txt"), filepath.Join(dir, "brain.gob"), nil); err == nil {
		t.Error("expected error for missing file")
	}
}

func TestImportTxtFileEmpty(t *testing.T) {
	dir := t.TempDir()
	src := writeFile(t, dir, "empty.txt", "\n\n\n")
	b := newBrain(t)
	if _, err := ImportTxtFile(b, src, filepath.Join(dir, "brain.gob"), nil); err != nil {
		t.Fatalf("import failed: %v", err)
	}
	if b.VocabularyLen() != 1 {
		t.Error("an empty file should teach nothing")
	}
}

func TestImportTxtFileSingleLine(t *testing.T) {
	dir := t.TempDir()
	src := writeFile(t, dir, "one.txt", "одинокая строка\n")
	b := newBrain(t)
	result, err := ImportTxtFile(b, src, filepath.Join(dir, "brain.gob"), nil)
	if err != nil {
		t.Fatalf("import failed: %v", err)
	}
	if result.Trained != 0 {
		t.Errorf("a single line cannot form a pair, trained = %d", result.Trained)
	}
}

// Reading uses bufio.Reader rather than Scanner so multi-megabyte lines do not
// hit the 64KB token limit.
func TestImportTxtFileLongLine(t *testing.T) {
	dir := t.TempDir()
	content := strings.Repeat("a", 2*1024*1024) + "\n" + strings.Repeat("b", 2*1024*1024) + "\n"
	src := writeFile(t, dir, "long.txt", content)

	b := newBrain(t)
	if _, err := ImportTxtFile(b, src, filepath.Join(dir, "brain.gob"), nil); err != nil {
		t.Fatalf("long-line import failed: %v", err)
	}
	if b.VocabularyLen() < 3 {
		t.Errorf("vocab too small after long-line import: %d", b.VocabularyLen())
	}
}

func TestImportPersistsBrain(t *testing.T) {
	dir := t.TempDir()
	src := writeFile(t, dir, "input.txt", "a b\nc d\ne f\n")
	brainPath := filepath.Join(dir, "brain.gob")

	b := newBrain(t)
	if _, err := ImportTxtFile(b, src, brainPath, nil); err != nil {
		t.Fatalf("import failed: %v", err)
	}
	vocab := b.VocabularyLen()

	restored := engine.NewBrain()
	if err := restored.Load(brainPath); err != nil {
		t.Fatalf("reload failed: %v", err)
	}
	if restored.VocabularyLen() != vocab {
		t.Errorf("vocab after reload = %d, want %d", restored.VocabularyLen(), vocab)
	}
}

// The brain must be saved once at the end, not after every line.
func TestImportReportsProgress(t *testing.T) {
	dir := t.TempDir()
	src := writeFile(t, dir, "progress.txt", "a b\nc d\ne f\ng h\n")

	b := newBrain(t)
	var seen []int
	if _, err := ImportTxtFile(b, src, "", func(processed, total int) {
		seen = append(seen, processed)
		if total != 3 {
			t.Errorf("total = %d, want 3", total)
		}
	}); err != nil {
		t.Fatal(err)
	}
	if len(seen) == 0 {
		t.Error("expected progress callbacks")
	}
	if last := seen[len(seen)-1]; last != 3 {
		t.Errorf("final progress = %d, want 3", last)
	}
}

func TestImportEnablesThinking(t *testing.T) {
	dir := t.TempDir()
	src := writeFile(t, dir, "book.txt", "один два\nтри четыре\nпять шесть\nсемь восемь\nдевять десять\n")

	b := newBrain(t)
	if _, err := ImportTxtFile(b, src, filepath.Join(dir, "brain.gob"), nil); err != nil {
		t.Fatalf("import failed: %v", err)
	}
	if b.VocabularyLen() < engine.MinVocabForThinking() {
		t.Fatalf("vocab = %d, want >= %d", b.VocabularyLen(), engine.MinVocabForThinking())
	}
	for attempt := 0; attempt < 20; attempt++ {
		if len(b.Think("один")) > 0 {
			return
		}
	}
	t.Fatal("expected non-empty thoughts after retries")
}

func TestImportSampleFixture(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join("..", "..", "testdata", "sample.txt")
	brain := filepath.Join(dir, "brain.gob")

	b := newBrain(t)
	result, err := ImportTxtFile(b, src, brain, nil)
	if err != nil {
		t.Fatalf("import failed: %v", err)
	}
	if b.VocabularyLen() < engine.MinVocabForThinking() {
		t.Errorf("vocab = %d, want >= %d", b.VocabularyLen(), engine.MinVocabForThinking())
	}
	if result.Parameters <= 0 {
		t.Error("expected learned parameters")
	}
	if _, err := os.Stat(brain); err != nil {
		t.Fatalf("brain file not created: %v", err)
	}
}

// An import must not leave an undo point behind.
func TestImportIsNotUndoable(t *testing.T) {
	dir := t.TempDir()
	src := writeFile(t, dir, "undo.txt", "a b\nc d\n")

	b := newBrain(t)
	if _, err := ImportTxtFile(b, src, "", nil); err != nil {
		t.Fatal(err)
	}
	if b.UndoLastTrain() {
		t.Error("a batch import must not create an undo point")
	}
}
