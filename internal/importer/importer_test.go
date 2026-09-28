package importer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"mycor/internal/engine"
)

func TestImportTxtFile(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "input.txt")
	content := "привет мир\nкак дела\nвсё хорошо\n"
	if err := os.WriteFile(src, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	brain := filepath.Join(dir, "brain.json")
	engine.InitEngine()
	if err := ImportTxtFile(src, brain); err != nil {
		t.Fatalf("import failed: %v", err)
	}

	if _, err := os.Stat(brain); err != nil {
		t.Fatalf("brain file not created: %v", err)
	}
	if len(engine.Vocabulary) < 4 {
		t.Errorf("vocab too small: %d", len(engine.Vocabulary))
	}
}

func TestImportTxtFileMissing(t *testing.T) {
	dir := t.TempDir()
	engine.InitEngine()
	err := ImportTxtFile(filepath.Join(dir, "nope.txt"), filepath.Join(dir, "brain.json"))
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestImportTxtFileEmpty(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "empty.txt")
	if err := os.WriteFile(src, []byte("\n\n\n"), 0644); err != nil {
		t.Fatal(err)
	}
	brain := filepath.Join(dir, "brain.json")
	engine.InitEngine()
	if err := ImportTxtFile(src, brain); err != nil {
		t.Fatalf("import failed: %v", err)
	}
	if _, err := os.Stat(brain); err != nil {
		t.Fatalf("brain file not created: %v", err)
	}
}

func TestImportTxtFileSingleLine(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "one.txt")
	if err := os.WriteFile(src, []byte("одинокая строка\n"), 0644); err != nil {
		t.Fatal(err)
	}
	brain := filepath.Join(dir, "brain.json")
	engine.InitEngine()
	if err := ImportTxtFile(src, brain); err != nil {
		t.Fatalf("import failed: %v", err)
	}
}

func TestImportTxtFileLongLine(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "long.txt")
	first := strings.Repeat("a", 2*1024*1024)
	second := strings.Repeat("b", 2*1024*1024)
	content := first + "\n" + second + "\n"
	if err := os.WriteFile(src, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	brain := filepath.Join(dir, "brain.json")
	engine.InitEngine()
	if err := ImportTxtFile(src, brain); err != nil {
		t.Fatalf("long-line import failed: %v", err)
	}
	if len(engine.Vocabulary) < 3 {
		t.Errorf("vocab too small after long-line import: %d", len(engine.Vocabulary))
	}
}

func TestImportTxtFilePersistsBrain(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "input.txt")
	if err := os.WriteFile(src, []byte("a b\nc d\ne f\n"), 0644); err != nil {
		t.Fatal(err)
	}
	brain := filepath.Join(dir, "brain.json")
	engine.InitEngine()
	if err := ImportTxtFile(src, brain); err != nil {
		t.Fatalf("import failed: %v", err)
	}
	vocab := len(engine.Vocabulary)

	engine.InitEngine()
	if !engine.LoadBrain(brain) {
		t.Fatal("reload failed")
	}
	if len(engine.Vocabulary) != vocab {
		t.Errorf("vocab after reload = %d, want %d", len(engine.Vocabulary), vocab)
	}
}

func TestImportEnablesThinking(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "book.txt")
	content := "один два\nтри четыре\nпять шесть\nсемь восемь\nдевять десять\n"
	if err := os.WriteFile(src, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	brain := filepath.Join(dir, "brain.json")
	engine.InitEngine()
	if err := ImportTxtFile(src, brain); err != nil {
		t.Fatalf("import failed: %v", err)
	}
	if len(engine.Vocabulary) < engine.MinVocabForThinking {
		t.Fatalf("vocab = %d, want >= %d", len(engine.Vocabulary), engine.MinVocabForThinking)
	}

	var thoughts []string
	for attempt := 0; attempt < 20; attempt++ {
		thoughts = engine.Think("один")
		if len(thoughts) > 0 {
			break
		}
	}
	if len(thoughts) == 0 {
		t.Fatal("expected non-empty thoughts after retries")
	}
}

func TestImportSampleFixture(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join("..", "..", "testdata", "sample.txt")
	brain := filepath.Join(dir, "brain.json")
	engine.InitEngine()
	if err := ImportTxtFile(src, brain); err != nil {
		t.Fatalf("import failed: %v", err)
	}
	if len(engine.Vocabulary) < engine.MinVocabForThinking {
		t.Errorf("vocab = %d, want >= %d", len(engine.Vocabulary), engine.MinVocabForThinking)
	}
	if _, err := os.Stat(brain); err != nil {
		t.Fatalf("brain file not created: %v", err)
	}
}
