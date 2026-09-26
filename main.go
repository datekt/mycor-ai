package main

import (
	"bufio"
	"fmt"
	"mycor/config"
	"mycor/engine"
	"mycor/utils"
	"os"
	"strconv"
	"strings"
)

const brainFile = "history.json"

func printHelp() {
	fmt.Println("\n🎛️ ПАНЕЛЬ УПРАВЛЕНИЯ MYCOR AI v4.3:")
	fmt.Println("  /help              - Показать этот список команд")
	fmt.Println("  /stats             - Посмотреть топ связей в синапсах")
	fmt.Println("  /history           - Показать историю текущей сессии")
	fmt.Println("  /undo              - Отменить последнее обучение")
	fmt.Println("  /reset             - Стереть память и начать заново")
	fmt.Println("  /temp [0.1-1.5]    - Изменить креативность (сейчас:", config.Temperature, ")")
	fmt.Println("  /lr [0.01-1.0]     - Изменить скорость обучения (сейчас:", config.LearningRate, ")")
	fmt.Println("  /momentum [0-0.99] - Инерция градиента (сейчас:", config.Momentum, ")")
	fmt.Println("  /import [файл]     - Пакетное обучение из TXT (например: /import book.txt)")
	fmt.Println("  /self              - Режим автономного разговора ИИ с самим собой")
	fmt.Println("  exit               - Выйти из программы")
	fmt.Println("\n💭 Режим размышления включается автоматически при словаре ≥ 10 слов.")
}

func saveOrWarn() {
	if err := engine.SaveBrain(brainFile); err != nil {
		fmt.Println("❌ Ошибка сохранения памяти:", err)
	}
}

func main() {
	engine.InitEngine()
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("🧠 Проверка архитектуры MYCOR v4.3 Ultra...")
	if engine.LoadBrain(brainFile) {
		fmt.Printf("💾 Мозг загружен! Слов в базе: %d | Синапсов: %d параметров\n", len(engine.Vocabulary), engine.CountParameters())
		if len(engine.Vocabulary) >= engine.MinVocabForThinking {
			fmt.Println("💭 Словарь достаточен для режима размышления.")
		}
		fmt.Println("🆕 История сессии пуста. Обучение сохранено на диск.")
	} else {
		fmt.Println("👶 Мозг пуст. Готов к первому ручному или пакетному обучению.")
	}

	printHelp()

	for {
		fmt.Print("\nВы: ")
		userInput, err := reader.ReadString('\n')
		if err != nil {
			saveOrWarn()
			break
		}
		userInput = strings.TrimSpace(userInput)

		if userInput == "exit" {
			saveOrWarn()
			break
		}

		if strings.HasPrefix(userInput, "/") {
			parts := strings.Fields(userInput)
			switch parts[0] {
			case "/help":
				printHelp()
			case "/history":
				history := engine.SessionHistory()
				if len(history) == 0 {
					fmt.Println("📭 История текущей сессии пуста (она сбрасывается при перезапуске).")
				} else {
					fmt.Printf("📜 История сессии (%d сообщений):\n", len(history))
					for _, m := range history {
						fmt.Printf("  [%s] %s\n", m.Role, m.Text)
					}
				}
			case "/undo":
				if engine.UndoLastTrain() {
					fmt.Println("↩️ Предыдущий урок успешно стерт из памяти.")
					saveOrWarn()
				} else {
					fmt.Println("❌ Нечего отменять.")
				}
			case "/reset":
				if err := os.Remove(brainFile); err != nil && !os.IsNotExist(err) {
					fmt.Println("❌ Не удалось удалить файл памяти:", err)
					break
				}
				engine.InitEngine()
				fmt.Println("🧹 Память полностью стерта. Перезагрузка ИИ успешна.")
			case "/temp":
				if len(parts) < 2 {
					fmt.Println("Укажите значение, например: /temp 0.7")
					break
				}
				val, err := strconv.ParseFloat(parts[1], 64)
				if err != nil || val < 0.1 || val > 1.5 {
					fmt.Println("❌ Недопустимое значение. Диапазон: 0.1 - 1.5")
					break
				}
				config.Temperature = val
				fmt.Println("🔥 Новая креативность установлена:", val)
			case "/lr":
				if len(parts) < 2 {
					fmt.Println("Укажите значение, например: /lr 0.5")
					break
				}
				val, err := strconv.ParseFloat(parts[1], 64)
				if err != nil || val < 0.01 || val > 1.0 {
					fmt.Println("❌ Недопустимое значение. Диапазон: 0.01 - 1.0")
					break
				}
				config.LearningRate = val
				fmt.Println("⚡ Скорость обучения установлена:", val)
			case "/momentum":
				if len(parts) < 2 {
					fmt.Println("Укажите значение, например: /momentum 0.9")
					break
				}
				val, err := strconv.ParseFloat(parts[1], 64)
				if err != nil || val < 0.0 || val > 0.99 {
					fmt.Println("❌ Недопустимое значение. Диапазон: 0.0 - 0.99")
					break
				}
				config.Momentum = val
				fmt.Println("🌀 Инерция градиента установлена:", val)
			case "/import":
				if len(parts) > 1 {
					fmt.Println("⏳ Чтение файла...")
					err := utils.ImportTxtFile(parts[1], brainFile)
					if err != nil {
						fmt.Println("❌ Ошибка импорта:", err)
					} else {
						fmt.Printf("🎉 Обучение завершено! Новых синапсов: %d\n", engine.CountParameters())
					}
				} else {
					fmt.Println("Укажите файл, например: /import book.txt")
				}
			case "/stats":
				fmt.Printf("📊 Статистика: Слова в словаре = %d, Всего параметров весов = %d\n", len(engine.Vocabulary), engine.CountParameters())
				if len(engine.Vocabulary) >= engine.MinVocabForThinking {
					fmt.Println("💭 Режим размышления: АКТИВЕН")
				} else {
					fmt.Printf("💭 Режим размышления: выключен (нужно ≥ %d слов, сейчас %d)\n", engine.MinVocabForThinking, len(engine.Vocabulary))
				}
				recent := engine.RecentContexts()
				fmt.Println("Последние активные контекстные зоны:")
				if len(recent) == 0 {
					fmt.Println(" - Пока нет активности.")
				} else {
					start := 0
					if len(recent) > 3 {
						start = len(recent) - 3
					}
					for _, k := range recent[start:] {
						if v, ok := engine.Weights[k]; ok {
							fmt.Printf(" - Синапс [%s] -> %d связей\n", k, len(v))
						}
					}
				}
			case "/self":
				fmt.Println("🤖 Разговор ИИ с зеркалом. Нажмите Ctrl+C для выхода.")
				currentPrompt := "Привет"
				for i := 0; i < 10; i++ {
					reply := engine.GenerateResponse(currentPrompt, 8)
					if thoughts := engine.LastThoughts(); thoughts != "" {
						fmt.Printf("💭 Мысли: %s\n", thoughts)
					}
					if reply == "" || reply == "..." {
						reply = "что ты думаешь ?"
					}
					fmt.Printf("ИИ-Зеркало: %s\n", reply)
					currentPrompt = reply
				}
			}
			continue
		}

		if userInput == "" {
			continue
		}

		engine.AppendHistory("Вы", userInput)

		aiOutput := engine.GenerateResponse(userInput, 15)

		if thoughts := engine.LastThoughts(); thoughts != "" {
			fmt.Printf("💭 Мысли MYCOR: %s\n", thoughts)
		}

		fmt.Printf("ИИ MYCOR: %s\n", aiOutput)
		engine.AppendHistory("ИИ", aiOutput)

		fmt.Print("Как нужно было ответить? (Оставьте пустым для пропуска): ")
		correctAnswer, _ := reader.ReadString('\n')
		correctAnswer = strings.TrimSpace(correctAnswer)

		if correctAnswer != "" {
			engine.AppendHistory("Учитель", correctAnswer)
			loss := engine.Train(userInput, correctAnswer)
			saveOrWarn()
			fmt.Printf("📉 Урок усвоен! Ошибка сети (Loss): %.4f | Параметров: %d\n", loss, engine.CountParameters())
		}
	}
}
