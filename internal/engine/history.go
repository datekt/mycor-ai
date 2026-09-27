package engine

func touchContext(key string) {
	for i, k := range recentContexts {
		if k == key {
			recentContexts = append(recentContexts[:i], recentContexts[i+1:]...)
			break
		}
	}
	recentContexts = append(recentContexts, key)
	if len(recentContexts) > maxRecentContexts {
		recentContexts = recentContexts[len(recentContexts)-maxRecentContexts:]
	}
}

func RecentContexts() []string {
	out := make([]string, len(recentContexts))
	copy(out, recentContexts)
	return out
}

func SessionHistory() []ChatMessage {
	out := make([]ChatMessage, len(sessionHistory))
	copy(out, sessionHistory)
	return out
}

func AppendHistory(role, text string) {
	if text == "" {
		return
	}
	sessionHistory = append(sessionHistory, ChatMessage{Role: role, Text: text})
	if len(sessionHistory) > maxSessionHistory {
		sessionHistory = sessionHistory[len(sessionHistory)-maxSessionHistory:]
	}
}
