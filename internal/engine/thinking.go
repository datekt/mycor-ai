package engine

import (
	"mycor/internal/config"
)

func think(prompt string) []string {
	if len(Vocabulary) < MinVocabForThinking {
		return nil
	}
	words := Tokenize(prompt)
	if len(words) == 0 {
		return nil
	}

	contextSize := currentContextSize()
	history := make([]string, len(words))
	copy(history, words)

	thoughts := make([]string, 0, thinkingSteps)
	usedWords := map[int]bool{0: true}

	for i := 0; i < thinkingSteps; i++ {
		ctx := buildContext(history, contextSize)
		idx, ok := sampleNextIndex(ctx, usedWords, config.Temperature)
		if !ok {
			break
		}
		nextWord := Vocabulary[idx]
		if nextWord == unkToken {
			break
		}
		thoughts = append(thoughts, nextWord)
		usedWords[idx] = true
		history = append(history, nextWord)
		if nextWord == "." || nextWord == "?" || nextWord == "!" {
			break
		}
	}
	return thoughts
}

func Think(prompt string) []string {
	return think(prompt)
}

func LastThoughts() string {
	return lastThoughts
}
