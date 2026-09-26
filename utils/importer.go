package utils

import (
	"bufio"
	"fmt"
	"mycor/engine"
	"os"
	"strings"
)

func ImportTxtFile(filePath string, brainPath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var lines []string
	for scanner.Scan() {
		text := strings.TrimSpace(scanner.Text())
		if text != "" {
			lines = append(lines, text)
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}

	fmt.Printf("📖 Найдено %d строк. Начинаю пакетное сканирование...\n", len(lines))

	totalLoss := 0.0
	trained := 0
	for i := 0; i < len(lines)-1; i++ {
		loss := engine.TrainBatch(lines[i], lines[i+1])
		totalLoss += loss
		trained++
		if i%100 == 0 && i > 0 {
			fmt.Printf("⚡ Обработано %d строк...\n", i)
		}
	}

	if err := engine.SaveBrain(brainPath); err != nil {
		return err
	}

	if trained > 0 {
		fmt.Printf("📉 Средняя ошибка: %.4f\n", totalLoss/float64(trained))
	}
	fmt.Printf("💭 Слов в словаре: %d (режим размышления: %v)\n",
		len(engine.Vocabulary),
		len(engine.Vocabulary) >= engine.MinVocabForThinking)
	return nil
}
