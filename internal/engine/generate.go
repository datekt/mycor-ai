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

	contextSize := currentContextSize()
	history := make([]string, len(words))
	copy(history, words)

	result := make([]string, 0, maxLen)
	usedWords := make(map[int]bool)

	for i := 0; i < maxLen; i++ {
		ctx := buildContext(history, contextSize)
		idx, ok := sampleNextIndex(ctx, usedWords, config.Temperature)
		if !ok {
			if len(Vocabulary) > 1 {
				pick := rand.Intn(len(Vocabulary)-1) + 1
				nextWord := Vocabulary[pick]
				if nextWord == unkToken {
					continue
				}
				result = append(result, nextWord)
				usedWords[pick] = true
				history = append(history, nextWord)
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
		history = append(history, nextWord)

		if nextWord == "." || nextWord == "?" || nextWord == "!" {
			break
		}
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
	} else if len(Vocabulary) > 1 {
		prompt = Vocabulary[rand.Intn(len(Vocabulary)-1)+1]
	} else {
		return ""
	}
	thoughts := think(prompt)
	return JoinWords(thoughts)
}
