package engine

import (
	"math"
	"math/rand"

	"mycor/internal/config"
)

func MakeContextKey(w1, w2 string) string {
	return w1 + " " + w2
}

func backoffChain(w1, w2 string) []string {
	seen := make(map[string]bool, 4)
	chain := make([]string, 0, 4)
	add := func(key string) {
		if !seen[key] {
			seen[key] = true
			chain = append(chain, key)
		}
	}
	add(MakeContextKey(w1, w2))
	if w1 != unkToken {
		add(MakeContextKey(unkToken, w2))
	}
	if w2 != unkToken {
		add(MakeContextKey(w1, unkToken))
	}
	add(MakeContextKey(unkToken, unkToken))
	return chain
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
	if sum <= 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		uniform := make([]float64, len(scaled))
		share := 1.0 / float64(len(scaled))
		for i := range uniform {
			uniform[i] = share
		}
		return uniform
	}
	for i := range scaled {
		scaled[i] /= sum
	}
	return scaled
}

func ensureWeights(ctxKey string) ([]float64, []float64) {
	vLen := len(Vocabulary)

	w, ok := weights[ctxKey]
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
	weights[ctxKey] = w

	vel, vok := velocity[ctxKey]
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
	velocity[ctxKey] = vel

	return w, vel
}

func sampleNextIndex(w1, w2 string, usedWords map[int]bool, temperature float64) (int, bool) {
	chain := backoffChain(w1, w2)

	var (
		logits []float64
		ctxKey string
		found  bool
	)
	for _, key := range chain {
		if v, ok := weights[key]; ok && len(v) > 0 {
			logits = v
			ctxKey = key
			found = true
			break
		}
	}
	if !found {
		return 0, false
	}

	touchContext(ctxKey)
	probs := softmaxBase(logits, temperature, usedWords, config.RepetitionPenalty)
	if len(probs) == 0 {
		return 0, false
	}

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

	if nextIdx < 0 || nextIdx >= len(Vocabulary) {
		return 0, false
	}
	return nextIdx, true
}
