package engine

type BrainModel struct {
	Version    int                  `json:"version"`
	Vocabulary []string             `json:"vocabulary"`
	Weights    map[string][]float64 `json:"weights"`
	Velocity   map[string][]float64 `json:"velocity,omitempty"`
}

type ChatMessage struct {
	Role string `json:"role"`
	Text string `json:"text"`
}

func RegisterWord(word string) int {
	if word == "" {
		return 0
	}
	if idx, exists := WordToIdx[word]; exists {
		return idx
	}
	Vocabulary = append(Vocabulary, word)
	idx := len(Vocabulary) - 1
	WordToIdx[word] = idx

	for k, w := range Weights {
		if len(w) < len(Vocabulary) {
			extended := make([]float64, len(Vocabulary))
			copy(extended, w)
			for i := len(w); i < len(Vocabulary); i++ {
				extended[i] = initWeight()
			}
			Weights[k] = extended
		}
	}
	for k, v := range Velocity {
		if len(v) < len(Vocabulary) {
			extended := make([]float64, len(Vocabulary))
			copy(extended, v)
			Velocity[k] = extended
		}
	}
	return idx
}

func SaveBackup() {
	BackupWeights = make(map[string][]float64, len(Weights))
	for k, v := range Weights {
		cp := make([]float64, len(v))
		copy(cp, v)
		BackupWeights[k] = cp
	}
	BackupVelocity = make(map[string][]float64, len(Velocity))
	for k, v := range Velocity {
		cp := make([]float64, len(v))
		copy(cp, v)
		BackupVelocity[k] = cp
	}
	BackupVocabulary = make([]string, len(Vocabulary))
	copy(BackupVocabulary, Vocabulary)
	BackupWordToIdx = make(map[string]int, len(WordToIdx))
	for k, v := range WordToIdx {
		BackupWordToIdx[k] = v
	}
}

func UndoLastTrain() bool {
	if BackupVocabulary == nil {
		return false
	}
	Vocabulary = BackupVocabulary
	WordToIdx = BackupWordToIdx
	Weights = BackupWeights
	Velocity = BackupVelocity
	BackupVocabulary = nil
	BackupWordToIdx = nil
	BackupWeights = nil
	BackupVelocity = nil
	return true
}

func CountParameters() int {
	total := 0
	for _, v := range Weights {
		total += len(v)
	}
	return total
}
