package cli

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"mycor/internal/config"
	"mycor/internal/engine"
	"mycor/internal/i18n"
	"mycor/internal/importer"
)

func handleInput(line string, ir *inputReader) {
	if line == "exit" {
		saveOrWarn()
		os.Exit(0)
	}

	if strings.HasPrefix(line, "/") {
		dispatchCommand(line, ir)
		return
	}

	if line == "" {
		return
	}

	chatTurn(line, ir)
}

func dispatchCommand(line string, ir *inputReader) {
	parts := strings.Fields(line)
	if len(parts) == 0 {
		return
	}
	m := i18n.M()

	switch parts[0] {
	case "/help":
		printHelp()
	case "/history":
		cmdHistory()
	case "/undo":
		cmdUndo()
	case "/reset":
		cmdReset()
	case "/temp":
		cmdTemp(parts)
	case "/lr":
		cmdLR(parts)
	case "/momentum":
		cmdMomentum(parts)
	case "/import":
		cmdImport(parts)
	case "/stats":
		cmdStats()
	case "/self":
		cmdSelf()
	case "/lang":
		cmdLang(parts)
	case "/idle":
		cmdIdle(parts)
	default:
		fmt.Println(m.LangUsage)
	}
}

func cmdHistory() {
	m := i18n.M()
	history := engine.SessionHistory()
	if len(history) == 0 {
		fmt.Println(m.HistoryEmpty)
		return
	}
	fmt.Printf(m.HistoryHeader+"\n", len(history))
	for _, msg := range history {
		fmt.Printf("  [%s] %s\n", msg.Role, msg.Text)
	}
}

func cmdUndo() {
	m := i18n.M()
	if engine.UndoLastTrain() {
		fmt.Println(m.UndoOK)
		saveOrWarn()
		return
	}
	fmt.Println(m.UndoFail)
}

func cmdReset() {
	m := i18n.M()
	if err := os.Remove(brainFile); err != nil && !os.IsNotExist(err) {
		fmt.Printf(m.ResetFail+"\n", err)
		return
	}
	engine.InitEngine()
	fmt.Println(m.ResetOK)
}

func cmdTemp(parts []string) {
	m := i18n.M()
	if len(parts) < 2 {
		fmt.Println(m.TempRange)
		return
	}
	val, err := strconv.ParseFloat(parts[1], 64)
	if err != nil || val < 0.1 || val > 1.5 {
		fmt.Println(m.TempRange)
		return
	}
	config.Temperature = val
	fmt.Printf(m.TempOK+"\n", val)
}

func cmdLR(parts []string) {
	m := i18n.M()
	if len(parts) < 2 {
		fmt.Println(m.LRRange)
		return
	}
	val, err := strconv.ParseFloat(parts[1], 64)
	if err != nil || val < 0.01 || val > 1.0 {
		fmt.Println(m.LRRange)
		return
	}
	config.LearningRate = val
	fmt.Printf(m.LROK+"\n", val)
}

func cmdMomentum(parts []string) {
	m := i18n.M()
	if len(parts) < 2 {
		fmt.Println(m.MomentumRange)
		return
	}
	val, err := strconv.ParseFloat(parts[1], 64)
	if err != nil || val < 0.0 || val > 0.99 {
		fmt.Println(m.MomentumRange)
		return
	}
	config.Momentum = val
	fmt.Printf(m.MomentumOK+"\n", val)
}

func cmdImport(parts []string) {
	m := i18n.M()
	if len(parts) < 2 {
		fmt.Println(m.ImportUsage)
		return
	}
	fmt.Println(m.ImportReading)
	if err := importer.ImportTxtFile(parts[1], brainFile); err != nil {
		fmt.Printf(m.ImportFail+"\n", err)
		return
	}
	fmt.Printf(m.ImportOK+"\n", engine.CountParameters())
}

func cmdStats() {
	m := i18n.M()
	fmt.Printf(m.StatsHeader+"\n", len(engine.Vocabulary), engine.CountParameters())
	if len(engine.Vocabulary) >= engine.MinVocabForThinking {
		fmt.Println(m.StatsThinking)
	} else {
		fmt.Printf(m.StatsNoThinking+"\n", engine.MinVocabForThinking, len(engine.Vocabulary))
	}

	recent := engine.RecentContexts()
	fmt.Println(m.StatsRecent)
	if len(recent) == 0 {
		fmt.Println(m.StatsNoRecent)
		return
	}

	start := 0
	if len(recent) > 3 {
		start = len(recent) - 3
	}
	for _, k := range recent[start:] {
		if v, ok := engine.Weights[k]; ok {
			fmt.Printf(m.Synapse+"\n", k, len(v))
		}
	}
}

func cmdSelf() {
	m := i18n.M()
	fmt.Println(m.SelfHeader)
	currentPrompt := m.SelfSeed
	for i := 0; i < 10; i++ {
		reply := engine.GenerateResponse(currentPrompt, 8)
		if thoughts := engine.LastThoughts(); thoughts != "" {
			fmt.Printf("%s%s\n", m.Thoughts, thoughts)
		}
		if reply == "" || reply == "..." {
			reply = m.SelfSeed
		}
		fmt.Printf(m.SelfMirror+"\n", reply)
		currentPrompt = reply
	}
}

func cmdLang(parts []string) {
	m := i18n.M()
	if len(parts) < 2 {
		fmt.Println(m.LangUsage)
		return
	}
	switch parts[1] {
	case "en":
		i18n.Set(i18n.EN)
		fmt.Println(i18n.M().LangChanged)
	case "ru":
		i18n.Set(i18n.RU)
		fmt.Println(i18n.M().LangChanged)
	default:
		fmt.Println(m.LangUsage)
	}
}

func cmdIdle(parts []string) {
	m := i18n.M()
	if len(parts) < 2 {
		fmt.Println(m.IdleUsage)
		return
	}
	switch parts[1] {
	case "on":
		config.IdleEnabled = true
		fmt.Println(m.IdleOn)
	case "off":
		config.IdleEnabled = false
		fmt.Println(m.IdleOff)
	default:
		fmt.Println(m.IdleUsage)
	}
}

func chatTurn(line string, ir *inputReader) {
	m := i18n.M()

	engine.AppendHistory(m.YouRole, line)

	aiOutput := engine.GenerateResponse(line, 15)

	if thoughts := engine.LastThoughts(); thoughts != "" {
		fmt.Printf("%s%s\n", m.Thoughts, thoughts)
	}

	fmt.Printf("%s%s\n", m.AI, aiOutput)
	engine.AppendHistory(m.AIRole, aiOutput)

	fmt.Print(m.CorrectAnswer)
	correctAnswer, ok := <-ir.ch
	if !ok {
		return
	}

	if correctAnswer == "" {
		return
	}

	engine.AppendHistory(m.TeacherRole, correctAnswer)
	loss := engine.Train(line, correctAnswer)
	saveOrWarn()
	fmt.Printf(m.LossLine+"\n", loss, engine.CountParameters())
}
