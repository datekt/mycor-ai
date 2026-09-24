package utils

import (
	"os"
	"path/filepath"
	"testing"

	"mycor/engine"
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
