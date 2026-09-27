package engine

import (
	"mycor/internal/config"
)

func initialContext(words []string) (string, string) {
	var w1, w2 string
	if len(words) >= 2 {
		w1 = words[len(words)-2]
		w2 = words[len(words)-1]
	} else {
		w1 = unkToken
		w2 = words[len(words)-1]
	}
	if _, ok := WordToIdx[w1]; !ok {
		w1 = unkToken
	}
	if _, ok := WordToIdx[w2]; !ok {
		w2 = unkToken
	}
	return w1, w2
}

func think(prompt string) []string {
	if len(Vocabulary) < MinVocabForThinking {
		return nil
	}
	words := Tokenize(prompt)
	if len(words) == 0 {
		return nil
	}

	w1, w2 := initialContext(words)
	thoughts := make([]string, 0, thinkingSteps)
	usedWords := map[int]bool{0: true}

	for i := 0; i < thinkingSteps; i++ {
		idx, ok := sampleNextIndex(w1, w2, usedWords, config.Temperature)
		if !ok {
			break
		}
		nextWord := Vocabulary[idx]
		if nextWord == unkToken {
			break
		}
		thoughts = append(thoughts, nextWord)
		usedWords[idx] = true
		if nextWord == "." || nextWord == "?" || nextWord == "!" {
			break
		}
		w1, w2 = w2, nextWord
	}
	return thoughts
}

func Think(prompt string) []string {
	return think(prompt)
}

func LastThoughts() string {
	return lastThoughts
}
