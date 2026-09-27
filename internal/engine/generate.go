package engine

import (
	"math/rand"

	"mycor/internal/config"
)

func GenerateResponse(prompt string, maxLen int) string {
	thoughts := think(prompt)
	lastThoughts = JoinWords(thoughts)

	effectivePrompt := prompt
	if len(thoughts) > 0 {
		effectivePrompt = prompt + " " + lastThoughts
	}

	return generate(effectivePrompt, maxLen)
}

func generate(prompt string, maxLen int) string {
	if maxLen <= 0 {
		return ""
	}
	words := Tokenize(prompt)
	if len(words) == 0 {
		return "..."
	}

	w1, w2 := initialContext(words)

	result := make([]string, 0, maxLen)
	usedWords := make(map[int]bool)

	for i := 0; i < maxLen; i++ {
		idx, ok := sampleNextIndex(w1, w2, usedWords, config.Temperature)
		if !ok {
			if len(Vocabulary) > 1 {
				pick := rand.Intn(len(Vocabulary)-1) + 1
				nextWord := Vocabulary[pick]
				if nextWord == unkToken {
					continue
				}
				result = append(result, nextWord)
				usedWords[pick] = true
				w1, w2 = w2, nextWord
				continue
			}
			break
		}

		nextWord := Vocabulary[idx]
		if nextWord == unkToken {
			continue
		}
		result = append(result, nextWord)
		usedWords[idx] = true

		if nextWord == "." || nextWord == "?" || nextWord == "!" {
			break
		}
		w1, w2 = w2, nextWord
	}

	return JoinWords(result)
}

func GenerateIdleThought() string {
	if len(Vocabulary) < MinVocabForThinking {
		return ""
	}
	seeds := RecentContexts()
	var prompt string
	if len(seeds) > 0 {
		prompt = seeds[len(seeds)-1]
	} else {
		prompt = Vocabulary[rand.Intn(len(Vocabulary))]
	}
	thoughts := think(prompt)
	return JoinWords(thoughts)
}
