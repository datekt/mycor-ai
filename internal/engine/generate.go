package engine

import "mycor/internal/config"

// GenerateResponse produces a reply, recording the model's reasoning first and
// folding that reasoning into the effective prompt.
//
// A single lock acquisition covers the whole generation: the reasoning pass and
// the reply both mutate lastThoughts and the recent-context list, and holding
// the write lock keeps them atomic with respect to each other. Generation is
// bounded to a handful of tokens, so this is far cheaper than the old design
// where a global mutex was held across an entire bulk import.
func (b *Brain) GenerateResponse(prompt string, maxLen int) string {
	b.mu.Lock()
	defer b.mu.Unlock()

	cfg := config.Get()

	thoughts := b.thinkLocked(prompt, cfg)
	b.lastThoughts = JoinWords(thoughts)

	effectivePrompt := prompt
	if b.lastThoughts != "" {
		effectivePrompt = prompt + " " + b.lastThoughts
	}
	return b.generateLocked(effectivePrompt, maxLen, cfg)
}

// generateLocked samples up to maxLen words. The caller must hold b.mu.
func (b *Brain) generateLocked(prompt string, maxLen int, cfg config.Config) string {
	if maxLen <= 0 {
		return ""
	}
	words := Tokenize(prompt)
	if len(words) == 0 {
		return "..."
	}

	csize := contextSize(cfg)
	history := append([]string(nil), words...)

	result := make([]string, 0, maxLen)
	usedWords := make(map[int]bool)

	for i := 0; i < maxLen; i++ {
		ctx := buildContext(history, csize)
		idx, ok := b.sampleNextIndexLocked(ctx, usedWords, cfg)
		if !ok {
			// Nothing learned for this context: fall back to a random word so
			// an under-trained brain still produces visible output.
			pick, found := b.randomUnusedIndex()
			if !found {
				break
			}
			nextWord := b.vocab[pick]
			if nextWord == unkToken {
				continue
			}
			result = append(result, nextWord)
			usedWords[pick] = true
			history = append(history, nextWord)
			continue
		}

		nextWord := b.vocab[idx]
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

// GenerateIdleThought returns a spontaneous thought when the user has been
// idle, seeded from the most recently used context.
func (b *Brain) GenerateIdleThought() string {
	b.mu.Lock()
	defer b.mu.Unlock()

	cfg := config.Get()
	if len(b.vocab) < cfg.MinVocabThinking {
		return ""
	}

	var prompt string
	if n := len(b.recentContexts); n > 0 {
		prompt = b.recentContexts[n-1]
	} else {
		idx, ok := b.randomUnusedIndex()
		if !ok {
			return ""
		}
		prompt = b.vocab[idx]
	}
	return JoinWords(b.thinkLocked(prompt, cfg))
}
