/*
Package cli implements the MYCOR interactive terminal.

The CLI owns the REPL loop, the language selection prompt, the help panel,
and the idle ticker. It talks to the engine only through the public API
exposed by mycor/internal/engine and never touches weights directly.
*/
package cli

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"time"

	"mycor/internal/engine"
	"mycor/internal/i18n"
)

const brainFile = "history.json"

func Run() {
	selectLanguage()
	m := i18n.M()

	engine.InitEngine()

	fmt.Println("MYCOR AI v4.4")
	if engine.LoadBrain(brainFile) {
		fmt.Println(m.BrainLoaded)
		fmt.Printf(m.BrainLoadedInfo+"\n", len(engine.Vocabulary), engine.CountParameters())
		if len(engine.Vocabulary) >= engine.MinVocabForThinking {
			fmt.Println(m.ThinkingEnabled)
		}
		fmt.Println(m.SessionReset)
	} else {
		fmt.Println(m.BrainEmpty)
	}

	printHelp()

	ir := newInputReader()
	lastActivity := time.Now()
	ticker := time.NewTicker(idleTickInterval)
	defer ticker.Stop()

	for {
		fmt.Print(i18n.M().YouPrompt)

		select {
		case line, ok := <-ir.ch:
			if !ok {
				saveOrWarn()
				return
			}
			handleInput(line, ir)
			lastActivity = time.Now()
		case <-ticker.C:
			if shouldDaydream(lastActivity) {
				emitDaydream()
				lastActivity = time.Now()
			}
		}
	}
}

func selectLanguage() {
	fmt.Println("Select language / Выберите язык:")
	fmt.Println("  1. English")
	fmt.Println("  2. Русский")
	fmt.Print("Choice [1]: ")

	reader := bufio.NewReader(os.Stdin)
	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)

	if line == "2" {
		i18n.Set(i18n.RU)
		return
	}
	i18n.Set(i18n.EN)
}

func saveOrWarn() {
	if err := engine.SaveBrain(brainFile); err != nil {
		fmt.Printf(i18n.M().SaveFail+"\n", err)
	}
}

func printHelp() {
	m := i18n.M()
	fmt.Println(m.HelpHeader)
	fmt.Println(m.HelpHelp)
	fmt.Println(m.HelpStats)
	fmt.Println(m.HelpHistory)
	fmt.Println(m.HelpUndo)
	fmt.Println(m.HelpReset)
	fmt.Println(m.HelpTemp)
	fmt.Println(m.HelpLR)
	fmt.Println(m.HelpMomentum)
	fmt.Println(m.HelpImport)
	fmt.Println(m.HelpSelf)
	fmt.Println(m.HelpLang)
	fmt.Println(m.HelpIdle)
	fmt.Println(m.HelpExit)
	fmt.Printf(m.HelpThinkingNote+"\n", engine.MinVocabForThinking)
}
