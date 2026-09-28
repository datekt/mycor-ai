/*
Package importer provides bulk training from plain text files.

Each pair of consecutive non-empty lines is treated as a (prompt, target)
training example and passed to the engine with TrainBatch. No backup is
created, so /undo cannot reverse a batch import.

Input is read through bufio.Reader with ReadString so that arbitrarily long
lines are supported without hitting bufio.Scanner's token-size limit.
*/
package importer

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"

	"mycor/internal/engine"
)

func ImportTxtFile(filePath string, brainPath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
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
			if err == io.EOF {
				break
			}
			return err
		}
	}

	fmt.Printf("Reading %d lines. Starting batch scan...\n", len(lines))

	totalLoss := 0.0
	trained := 0
	for i := 0; i < len(lines)-1; i++ {
		loss := engine.TrainBatch(lines[i], lines[i+1])
		totalLoss += loss
		trained++
		if i%100 == 0 && i > 0 {
			fmt.Printf("Processed %d lines...\n", i)
		}
	}

	if err := engine.SaveBrain(brainPath); err != nil {
		return err
	}

	if trained > 0 {
		fmt.Printf("Average loss: %.4f\n", totalLoss/float64(trained))
	}
	fmt.Printf("Vocabulary size: %d (thinking mode: %v)\n",
		len(engine.Vocabulary),
		len(engine.Vocabulary) >= engine.MinVocabForThinking)
	return nil
}
