package engine

import (
	"encoding/json"
	"math"
	"math/rand"
	"mycor/config"
	"os"
	"strings"
)

type BrainModel struct {
	Vocabulary []string             `json:"vocabulary"`
	Weights    map[string][]float64 `json:"weights"`
}

var (
	Vocabulary []string
	WordToIdx  map[string]int
	Weights    map[string][]float64

	BackupVocabulary []string
	BackupWordToIdx  map[string]int
	BackupWeights    map[string][]float64

	recentContexts []string
)

const maxRecentContexts = 20

func InitEngine() {
	WordToIdx = make(map[string]int)
	Weights = make(map[string][]float64)
	BackupWeights = nil
	BackupWordToIdx = nil
	BackupVocabulary = nil
	Vocabulary = []string{"<unk>"}
	WordToIdx["<unk>"] = 0
	recentContexts = nil
}

func Tokenize(text string) []string {
	text = strings.ToLower(text)
	replacer := strings.NewReplacer(",", " , ", ".", " . ", "?", " ? ", "!", " ! ")
	text = replacer.Replace(text)
	rawWords := strings.Fields(text)
	var words []string
	for _, w := range rawWords {
		w = strings.TrimSpace(w)
		if w != "" {
			words = append(words, w)
		}
	}
	return words
}

func isPunct(w string) bool {
	switch w {
	case ".", ",", "?", "!", ":", ";":
		return true
	}
	return false
}

func JoinWords(words []string) string {
	var sb strings.Builder
	for i, w := range words {
		if i > 0 && !isPunct(w) {
			sb.WriteByte(' ')
		}
		sb.WriteString(w)
	}
	return sb.String()
}

func RegisterWord(word string) int {
	if idx, exists := WordToIdx[word]; exists {
		return idx
	}
	Vocabulary = append(Vocabulary, word)
	idx := len(Vocabulary) - 1
	WordToIdx[word] = idx

	for k := range Weights {
		Weights[k] = append(Weights[k], rand.Float64()*0.1)
	}
	return idx
}

func MakeContextKey(w1, w2 string) string {
	return w1 + " " + w2
}

func touchContext(key string) {
	for i, k := range recentContexts {
		if k == key {
			recentContexts = append(recentContexts[:i], recentContexts[i+1:]...)
			break
		}
	}
	recentContexts = append(recentContexts, key)
	if len(recentContexts) > maxRecentContexts {
		recentContexts = recentContexts[1:]
	}
}

func RecentContexts() []string {
	out := make([]string, len(recentContexts))
	copy(out, recentContexts)
	return out
}

func SaveBackup() {
	BackupWeights = make(map[string][]float64, len(Weights))
	for k, v := range Weights {
		newSlice := make([]float64, len(v))
		copy(newSlice, v)
		BackupWeights[k] = newSlice
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
	BackupVocabulary = nil
	BackupWordToIdx = nil
	BackupWeights = nil
	return true
}

func softmaxBase(logits []float64, temperature float64, usedWords map[int]bool, penalty float64) []float64 {
	if len(logits) == 0 {
		return nil
	}
	if temperature <= 0 {
		temperature = 1.0
	}

	scaled := make([]float64, len(logits))
	max := -math.MaxFloat64

	for i, v := range logits {
		s := v / temperature
		if usedWords != nil && usedWords[i] && penalty > 0 {
			if s > 0 {
				s /= penalty
			} else {
				s *= penalty
			}
		}
		scaled[i] = s
		if s > max {
			max = s
		}
	}

	sum := 0.0
	for i, v := range scaled {
		scaled[i] = math.Exp(v - max)
		sum += scaled[i]
	}
	if sum == 0 {
		return scaled
	}
	for i := range scaled {
		scaled[i] /= sum
	}
	return scaled
}

func GenerateResponse(prompt string, maxLen int) string {
	words := Tokenize(prompt)
	if len(words) == 0 {
		return "..."
	}

	var w1, w2 string
	if len(words) >= 2 {
		w1 = words[len(words)-2]
		w2 = words[len(words)-1]
	} else {
		w1 = "<unk>"
		w2 = words[len(words)-1]
	}

	if _, ok := WordToIdx[w1]; !ok {
		w1 = "<unk>"
	}
	if _, ok := WordToIdx[w2]; !ok {
		w2 = "<unk>"
	}

	var result []string
	usedWords := make(map[int]bool)

	for i := 0; i < maxLen; i++ {
		ctxKey := MakeContextKey(w1, w2)
		logits, exists := Weights[ctxKey]

		if !exists || len(logits) == 0 {
			ctxKey = MakeContextKey("<unk>", w2)
			logits, exists = Weights[ctxKey]
		}

		if !exists || len(logits) == 0 {
			if len(Vocabulary) > 1 {
				nextWord := Vocabulary[rand.Intn(len(Vocabulary)-1)+1]
				result = append(result, nextWord)
				usedWords[WordToIdx[nextWord]] = true
				w1, w2 = w2, nextWord
				continue
			}
			break
		}

		touchContext(ctxKey)

		probs := softmaxBase(logits, config.Temperature, usedWords, config.RepetitionPenalty)

		r := rand.Float64()
		cumulative := 0.0
		nextIdx := 0
		for idx, p := range probs {
			cumulative += p
			if r <= cumulative {
				nextIdx = idx
				break
			}
		}

		if nextIdx >= len(Vocabulary) {
			break
		}

		nextWord := Vocabulary[nextIdx]
		result = append(result, nextWord)
		usedWords[nextIdx] = true

		if nextWord == "." || nextWord == "?" || nextWord == "!" {
			break
		}
		w1, w2 = w2, nextWord
	}

	return JoinWords(result)
}

func Train(prompt, target string) float64 {
	SaveBackup()
	return train(prompt, target)
}

func TrainBatch(prompt, target string) float64 {
	return train(prompt, target)
}

func train(prompt, target string) float64 {
	promptWords := Tokenize(prompt)
	targetWords := Tokenize(target)

	if len(promptWords) == 0 || len(targetWords) == 0 {
		return 0.0
	}

	for _, w := range promptWords {
		RegisterWord(w)
	}
	for _, w := range targetWords {
		RegisterWord(w)
	}

	var w1, w2 string
	if len(promptWords) >= 2 {
		w1 = promptWords[len(promptWords)-2]
		w2 = promptWords[len(promptWords)-1]
	} else {
		w1 = "<unk>"
		w2 = promptWords[len(promptWords)-1]
	}

	totalLoss := 0.0
	steps := 0.0

	for _, nextWord := range targetWords {
		targetIdx := WordToIdx[nextWord]
		ctxKey := MakeContextKey(w1, w2)
		touchContext(ctxKey)

		if _, exists := Weights[ctxKey]; !exists {
			Weights[ctxKey] = make([]float64, len(Vocabulary))
			for i := range Weights[ctxKey] {
				Weights[ctxKey][i] = rand.Float64() * 0.1
			}
		} else if len(Weights[ctxKey]) < len(Vocabulary) {
			missing := len(Vocabulary) - len(Weights[ctxKey])
			for i := 0; i < missing; i++ {
				Weights[ctxKey] = append(Weights[ctxKey], rand.Float64()*0.1)
			}
		}

		probs := softmaxBase(Weights[ctxKey], 1.0, nil, 0)

		if targetIdx < len(probs) && probs[targetIdx] > 0 {
			totalLoss += -math.Log(probs[targetIdx])
			steps++
		}

		for i := 0; i < len(Vocabulary) && i < len(Weights[ctxKey]); i++ {
			targetDelta := 0.0
			if i == targetIdx {
				targetDelta = 1.0
			}
			gradient := targetDelta - probs[i]
			Weights[ctxKey][i] += config.LearningRate * gradient
		}
		w1, w2 = w2, nextWord
	}

	if steps == 0 {
		return 0.0
	}
	return totalLoss / steps
}

func CountParameters() int {
	total := 0
	for _, v := range Weights {
		total += len(v)
	}
	return total
}

func LoadBrain(path string) bool {
	file, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var loaded BrainModel
	if json.Unmarshal(file, &loaded) != nil {
		return false
	}
	if len(loaded.Vocabulary) == 0 {
		return false
	}
	if loaded.Weights == nil {
		loaded.Weights = make(map[string][]float64)
	}

	if loaded.Vocabulary[0] != "<unk>" {
		loaded.Vocabulary = append([]string{"<unk>"}, loaded.Vocabulary...)
		for k, v := range loaded.Weights {
			extended := make([]float64, len(v)+1)
			copy(extended[1:], v)
			extended[0] = rand.Float64() * 0.1
			loaded.Weights[k] = extended
		}
	}

	vLen := len(loaded.Vocabulary)
	for k, v := range loaded.Weights {
		if len(v) < vLen {
			extended := make([]float64, vLen)
			copy(extended, v)
			for i := len(v); i < vLen; i++ {
				extended[i] = rand.Float64() * 0.1
			}
			loaded.Weights[k] = extended
		} else if len(v) > vLen {
			loaded.Weights[k] = v[:vLen]
		}
	}

	Vocabulary = loaded.Vocabulary
	Weights = loaded.Weights
	WordToIdx = make(map[string]int, len(Vocabulary))
	for i, word := range Vocabulary {
		WordToIdx[word] = i
	}

	BackupVocabulary = nil
	BackupWordToIdx = nil
	BackupWeights = nil
	recentContexts = nil
	return true
}

func SaveBrain(path string) {
	loaded := BrainModel{Vocabulary: Vocabulary, Weights: Weights}
	data, _ := json.MarshalIndent(loaded, "", "  ")
	_ = os.WriteFile(path, data, 0644)
}
