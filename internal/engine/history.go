package engine

import "mycor/internal/config"

// maxRecent falls back to a bounded recent-context list length.
func maxRecent() int { return config.Get().MaxRecentContexts }

// maxHistory returns the session history cap.
func maxHistory() int { return config.Get().MaxSessionHistory }

// touchContextLocked records that a context was just used. The caller must hold
// b.mu for writing.
func (b *Brain) touchContextLocked(key string) {
	if key == "" {
		return
	}
	for i, k := range b.recentContexts {
		if k == key {
			b.recentContexts = append(b.recentContexts[:i], b.recentContexts[i+1:]...)
			break
		}
	}
	b.recentContexts = append(b.recentContexts, key)
	if limit := maxRecent(); limit > 0 && len(b.recentContexts) > limit {
		b.recentContexts = b.recentContexts[len(b.recentContexts)-limit:]
	}
}

// RecentContexts returns a copy of the most recently used contexts.
func (b *Brain) RecentContexts() []string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return append([]string(nil), b.recentContexts...)
}

// SessionHistory returns a copy of the conversation history.
func (b *Brain) SessionHistory() []ChatMessage {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return append([]ChatMessage(nil), b.sessionHistory...)
}

// AppendHistory adds a message to the session history, trimming it to the
// configured cap.
func (b *Brain) AppendHistory(role, text string) {
	if text == "" {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.sessionHistory = append(b.sessionHistory, ChatMessage{Role: role, Text: text})
	if limit := maxHistory(); limit > 0 && len(b.sessionHistory) > limit {
		b.sessionHistory = b.sessionHistory[len(b.sessionHistory)-limit:]
	}
}

// ClearHistory empties the conversation history without touching the weights.
func (b *Brain) ClearHistory() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.sessionHistory = nil
}
