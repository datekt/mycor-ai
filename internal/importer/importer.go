/*
Package importer provides bulk training from plain text files.

Each pair of consecutive non-empty lines is treated as a (prompt, target)
training example and passed to the engine with TrainBatch. No undo point is
created, so /undo cannot reverse a batch import.

Input is read through bufio.Reader with ReadString so that arbitrarily long
lines are supported without hitting bufio.Scanner's token-size limit.

Progress is reported through a Progress callback rather than written straight to
stdout, which keeps the importer testable and lets the web layer surface
progress without spamming the server console.
*/
package importer

import (
	"bufio"
	"errors"
	"io"
	"os"
	"strings"

	"mycor/internal/engine"
)

// Result summarises an import run.
type Result struct {
	Lines       int
	Trained     int
	Vocabulary  int
	Parameters  int
	AverageLoss float64
}

// Progress is called periodically with the number of examples processed so far.
type Progress func(processed, total int)

// ImportTxtFile trains the brain from a plain text file.
//
// The brain is saved once at the end (and through the engine's own debounced
// saver when one is configured), never per line: writing the whole brain to
// disk after every one of thousands of examples was the dominant cost of large
// imports.
func ImportTxtFile(b *engine.Brain, filePath string, brainPath string, progress Progress) (Result, error) {
	var result Result

	file, err := os.Open(filePath)
	if err != nil {
		return result, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	var lines []string
	for {
		line, err := reader.ReadString('\n')
		if len(line) > 0 {
			text := strings.TrimSpace(line)
			if text != "" {
				lines = append(lines, text)
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return result, err
		}
	}

	result.Lines = len(lines)
	if len(lines) < 2 {
		return result, nil
	}

	total := len(lines) - 1
	totalLoss := 0.0
	for i := 0; i < total; i++ {
		loss := b.TrainBatch(lines[i], lines[i+1])
		totalLoss += loss
		result.Trained++
		if progress != nil && i%500 == 0 {
			progress(i, total)
		}
	}

	if brainPath != "" {
		if err := b.Save(brainPath); err != nil {
			return result, err
		}
	}
	if result.Trained > 0 {
		result.AverageLoss = totalLoss / float64(result.Trained)
	}
	result.Vocabulary = b.VocabularyLen()
	result.Parameters = b.CountParameters()
	if progress != nil {
		progress(total, total)
	}
	return result, nil
}
