package engine

import "mycor/internal/config"

// thinkLocked produces up to cfg.ThinkingSteps sampled words for a prompt,
// which are shown to the user as the model's visible reasoning.
//
// The caller must hold b.mu for writing, since sampling records the context it
// used in the recent list.
func (b *Brain) thinkLocked(prompt string, cfg config.Config) []string {
	if len(b.vocab) < cfg.MinVocabThinking {
		return nil
	}
	words := Tokenize(prompt)
	if len(words) == 0 {
		return nil
	}

	csize := contextSize(cfg)
	history := append([]string(nil), words...)

	thoughts := make([]string, 0, cfg.ThinkingSteps)
	usedWords := map[int]bool{unkIndex: true}

	for i := 0; i < cfg.ThinkingSteps; i++ {
		ctx := buildContext(history, csize)
		idx, ok := b.sampleNextIndexLocked(ctx, usedWords, cfg)
		if !ok {
			break
		}
		nextWord := b.vocab[idx]
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

// Think returns the model's reasoning for a prompt without generating a reply.
func (b *Brain) Think(prompt string) []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.thinkLocked(prompt, config.Get())
}

// LastThoughts returns the reasoning produced by the most recent generation.
func (b *Brain) LastThoughts() string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.lastThoughts
}
