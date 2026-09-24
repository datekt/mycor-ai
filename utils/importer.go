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

	engine.SaveBackup()
	for i := 0; i < len(lines)-1; i++ {
		engine.TrainBatch(lines[i], lines[i+1])
		if i%100 == 0 && i > 0 {
			fmt.Printf("⚡ Обработано %d строк...\n", i)
		}
	}

	engine.SaveBrain(brainPath)
	return nil
}
