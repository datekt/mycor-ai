package engine

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"mycor/config"
	"os"
	"path/filepath"
	"strings"
)

const (
	unkToken     = "<unk>"
	modelVersion = 2
)

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

var (
	Vocabulary []string
	WordToIdx  map[string]int
	Weights    map[string][]float64
	Velocity   map[string][]float64

	BackupVocabulary []string
	BackupWordToIdx  map[string]int
	BackupWeights    map[string][]float64
	BackupVelocity   map[string][]float64

	recentContexts []string
	sessionHistory []ChatMessage
)

const maxRecentContexts = 20
const maxSessionHistory = 100

func InitEngine() {
	Vocabulary = []string{unkToken}
	WordToIdx = map[string]int{unkToken: 0}
	Weights = make(map[string][]float64)
	Velocity = make(map[string][]float64)

	BackupVocabulary = nil
	BackupWordToIdx = nil
	BackupWeights = nil
	BackupVelocity = nil

	recentContexts = nil
	sessionHistory = nil
}

func Tokenize(text string) []string {
	text = strings.ToLower(text)
	replacer := strings.NewReplacer(
		",", " , ",
		".", " . ",
		"?", " ? ",
		"!", " ! ",
		":", " : ",
		";", " ; ",
	)
	text = replacer.Replace(text)
	rawWords := strings.Fields(text)
	words := make([]string, 0, len(rawWords))
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

func initWeight() float64 {
	return (rand.Float64() - 0.5) * 0.1
}

func RegisterWord(word string) int {
	if idx, exists := WordToIdx[word]; exists {
		return idx
	}
	Vocabulary = append(Vocabulary, word)
	idx := len(Vocabulary) - 1
	WordToIdx[word] = idx

	for k := range Weights {
		Weights[k] = append(Weights[k], initWeight())
	}
	for k := range Velocity {
		Velocity[k] = append(Velocity[k], 0)
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

func softmaxBase(logits []float64, temperature float64, usedWords map[int]bool, penalty float64) []float64 {
	if len(logits) == 0 {
		return nil
	}
	if temperature <= 0 {
		temperature = 1.0
	}

	scaled := make([]float64, len(logits))
	maxVal := -math.MaxFloat64
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
		if s > maxVal {
			maxVal = s
		}
	}

	sum := 0.0
	for i, v := range scaled {
		scaled[i] = math.Exp(v - maxVal)
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

func ensureWeights(ctxKey string) ([]float64, []float64) {
	vLen := len(Vocabulary)

	w, ok := Weights[ctxKey]
	if !ok {
		w = make([]float64, vLen)
		for i := range w {
			w[i] = initWeight()
		}
	}
	if len(w) < vLen {
		for i := len(w); i < vLen; i++ {
			w = append(w, initWeight())
		}
	} else if len(w) > vLen {
		w = w[:vLen]
	}
	Weights[ctxKey] = w

	vel, vok := Velocity[ctxKey]
	if !vok {
		vel = make([]float64, vLen)
	}
	if len(vel) < vLen {
		for i := len(vel); i < vLen; i++ {
			vel = append(vel, 0)
		}
	} else if len(vel) > vLen {
		vel = vel[:vLen]
	}
	Velocity[ctxKey] = vel

	return w, vel
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
		w1 = unkToken
		w2 = words[len(words)-1]
	}

	if _, ok := WordToIdx[w1]; !ok {
		w1 = unkToken
	}
	if _, ok := WordToIdx[w2]; !ok {
		w2 = unkToken
	}

	result := make([]string, 0, maxLen)
	usedWords := make(map[int]bool)

	for i := 0; i < maxLen; i++ {
		ctxKey := MakeContextKey(w1, w2)
		logits, exists := Weights[ctxKey]

		if !exists || len(logits) == 0 {
			ctxKey = MakeContextKey(unkToken, w2)
			logits, exists = Weights[ctxKey]
		}
		if !exists || len(logits) == 0 {
			ctxKey = MakeContextKey(w1, unkToken)
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
		nextIdx := len(probs) - 1
		for idx, p := range probs {
			cumulative += p
			if r <= cumulative {
				nextIdx = idx
				break
			}
		}

		if nextIdx >= len(Vocabulary) || nextIdx < 0 {
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
		w1 = unkToken
		w2 = promptWords[len(promptWords)-1]
	}

	totalLoss := 0.0
	steps := 0.0
	lr := config.LearningRate
	decay := 1.0 - lr*config.WeightDecay
	momentum := config.Momentum
	if momentum < 0 {
		momentum = 0
	}
	if momentum > 0.99 {
		momentum = 0.99
	}

	for _, nextWord := range targetWords {
		targetIdx, ok := WordToIdx[nextWord]
		if !ok {
			continue
		}
		ctxKey := MakeContextKey(w1, w2)
		touchContext(ctxKey)

		weights, vel := ensureWeights(ctxKey)
		probs := softmaxBase(weights, 1.0, nil, 0)

		if targetIdx < len(probs) && probs[targetIdx] > 0 {
			totalLoss += -math.Log(probs[targetIdx])
			steps++
		}

		limit := len(Vocabulary)
		if len(weights) < limit {
			limit = len(weights)
		}
		for i := 0; i < limit; i++ {
			targetDelta := 0.0
			if i == targetIdx {
				targetDelta = 1.0
			}
			grad := targetDelta - probs[i]
			vel[i] = momentum*vel[i] + (1.0-momentum)*grad
			weights[i] = weights[i]*decay + lr*vel[i]
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
	return loadBrain(path) == nil
}

func loadBrain(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var loaded BrainModel
	if err := json.Unmarshal(data, &loaded); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	if len(loaded.Vocabulary) == 0 {
		return errors.New("empty vocabulary")
	}
	if loaded.Weights == nil {
		loaded.Weights = make(map[string][]float64)
	}

	seen := make(map[string]bool, len(loaded.Vocabulary))
	for _, w := range loaded.Vocabulary {
		if w == "" {
			return errors.New("empty word in vocabulary")
		}
		if seen[w] {
			return fmt.Errorf("duplicate word in vocabulary: %q", w)
		}
		seen[w] = true
	}

	if loaded.Vocabulary[0] != unkToken {
		loaded.Vocabulary = append([]string{unkToken}, loaded.Vocabulary...)
		for k, v := range loaded.Weights {
			extended := make([]float64, len(v)+1)
			copy(extended[1:], v)
			extended[0] = initWeight()
			loaded.Weights[k] = extended
		}
		if loaded.Velocity != nil {
			for k, v := range loaded.Velocity {
				extended := make([]float64, len(v)+1)
				copy(extended[1:], v)
				loaded.Velocity[k] = extended
			}
		}
	}

	vLen := len(loaded.Vocabulary)

	for k, v := range loaded.Weights {
		if len(v) < vLen {
			extended := make([]float64, vLen)
			copy(extended, v)
			for i := len(v); i < vLen; i++ {
				extended[i] = initWeight()
			}
			loaded.Weights[k] = extended
		} else if len(v) > vLen {
			loaded.Weights[k] = v[:vLen]
		}
	}

	newVelocity := make(map[string][]float64, len(loaded.Weights))
	for k := range loaded.Weights {
		if loaded.Velocity != nil {
			if v, ok := loaded.Velocity[k]; ok {
				if len(v) < vLen {
					ext := make([]float64, vLen)
					copy(ext, v)
					newVelocity[k] = ext
				} else if len(v) > vLen {
					newVelocity[k] = v[:vLen]
				} else {
					newVelocity[k] = v
				}
				continue
			}
		}
		newVelocity[k] = make([]float64, vLen)
	}

	Vocabulary = loaded.Vocabulary
	Weights = loaded.Weights
	Velocity = newVelocity
	WordToIdx = make(map[string]int, len(Vocabulary))
	for i, word := range Vocabulary {
		WordToIdx[word] = i
	}

	BackupVocabulary = nil
	BackupWordToIdx = nil
	BackupWeights = nil
	BackupVelocity = nil
	recentContexts = nil
	sessionHistory = nil
	return nil
}

func SaveBrain(path string) error {
	model := BrainModel{
		Version:    modelVersion,
		Vocabulary: Vocabulary,
		Weights:    Weights,
		Velocity:   Velocity,
	}
	data, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		return err
	}

	dir := filepath.Dir(path)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}
