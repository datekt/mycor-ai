package engine

import (
	"math"
	"math/rand"
	"sort"

	"mycor/internal/config"
)

func MakeContextKey(words ...string) string {
	if len(words) == 0 {
		return ""
	}
	out := words[0]
	for i := 1; i < len(words); i++ {
		out += " " + words[i]
	}
	return out
}

func resolveContext(words []string) []string {
	out := make([]string, len(words))
	for i, w := range words {
		if _, ok := WordToIdx[w]; ok {
			out[i] = w
		} else {
			out[i] = unkToken
		}
	}
	return out
}

func backoffChain(words []string) []string {
	if len(words) == 0 {
		return nil
	}
	n := len(words)
	seen := make(map[string]bool, 2*n+1)
	chain := make([]string, 0, 2*n+1)

	add := func(ctx []string) {
		key := MakeContextKey(ctx...)
		if !seen[key] {
			seen[key] = true
			chain = append(chain, key)
		}
	}

	add(words)

	for level := n - 1; level >= 1; level-- {
		keepLast := make([]string, n)
		for i := 0; i < n-level; i++ {
			keepLast[i] = unkToken
		}
		copy(keepLast[n-level:], words[n-level:])
		add(keepLast)

		keepFirst := make([]string, n)
		copy(keepFirst, words[:level])
		for i := level; i < n; i++ {
			keepFirst[i] = unkToken
		}
		add(keepFirst)
	}

	allUnk := make([]string, n)
	for i := range allUnk {
		allUnk[i] = unkToken
	}
	add(allUnk)

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

type probPair struct {
	idx int
	p   float64
}

func applyTopK(probs []float64, k int) []float64 {
	if k <= 0 || k >= len(probs) {
		return probs
	}
	pairs := make([]probPair, 0, len(probs))
	for i, p := range probs {
		if p > 0 {
			pairs = append(pairs, probPair{i, p})
		}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].p > pairs[j].p
	})
	if k > len(pairs) {
		k = len(pairs)
	}
	out := make([]float64, len(probs))
	for i := 0; i < k; i++ {
		out[pairs[i].idx] = pairs[i].p
	}
	return out
}

func applyTopP(probs []float64, p float64) []float64 {
	if p <= 0 {
		out := make([]float64, len(probs))
		best := -1
		bestP := 0.0
		for i, v := range probs {
			if v > bestP {
				best = i
				bestP = v
			}
		}
		if best >= 0 {
			out[best] = probs[best]
		}
		return out
	}
	if p >= 1 {
		return probs
	}
	pairs := make([]probPair, 0, len(probs))
	for i, v := range probs {
		if v > 0 {
			pairs = append(pairs, probPair{i, v})
		}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].p > pairs[j].p
	})
	cumulative := 0.0
	cutoff := len(pairs)
	for i, pr := range pairs {
		cumulative += pr.p
		if cumulative >= p {
			cutoff = i + 1
			break
		}
	}
	out := make([]float64, len(probs))
	for i := 0; i < cutoff; i++ {
		out[pairs[i].idx] = pairs[i].p
	}
	return out
}

func renormalize(probs []float64) {
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if sum <= 0 {
		return
	}
	for i := range probs {
		probs[i] /= sum
	}
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

func sampleNextIndex(rawContext []string, usedWords map[int]bool, temperature float64) (int, bool) {
	context := resolveContext(rawContext)
	chain := backoffChain(context)

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

	probs = applyTopK(probs, config.TopK)
	probs = applyTopP(probs, config.TopP)
	renormalize(probs)

	r := rand.Float64()
	cumulative := 0.0
	nextIdx := -1
	for idx, p := range probs {
		if p <= 0 {
			continue
		}
		cumulative += p
		if r <= cumulative {
			nextIdx = idx
			break
		}
	}
	if nextIdx < 0 {
		for idx := len(probs) - 1; idx >= 0; idx-- {
			if probs[idx] > 0 {
				nextIdx = idx
				break
			}
		}
	}
	if nextIdx < 0 || nextIdx >= len(Vocabulary) {
		return 0, false
	}
	return nextIdx, true
}
