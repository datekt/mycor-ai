package engine

import (
	"math"
	"strings"
)

// MakeContextKey joins words into a context key. It is a pure function.
func MakeContextKey(words ...string) string {
	if len(words) == 0 {
		return ""
	}
	var sb strings.Builder
	for i, w := range words {
		if i > 0 {
			sb.WriteByte(' ')
		}
		sb.WriteString(w)
	}
	return sb.String()
}

// resolveContextLocked maps unknown words to <unk>. Caller must hold b.mu.
func (b *Brain) resolveContextLocked(words []string) []string {
	out := make([]string, len(words))
	for i, w := range words {
		if _, ok := b.wordToIdx[w]; ok {
			out[i] = w
		} else {
			out[i] = unkToken
		}
	}
	return out
}

// backoffChain returns the full ladder of contexts to train or query, from the
// most specific to the all-<unk> fallback. It is a pure function.
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

// softmaxSparse converts a sparse logit vector into probabilities. Absent
// entries have probability zero, which is exactly what makes the sparse layout
// safe: missing mass never enters the distribution.
//
// The caller must hold b.mu.
func (b *Brain) softmaxSparse(logits map[int]float64, temperature float64, used map[int]bool, penalty float64) map[int]float64 {
	if len(logits) == 0 {
		return nil
	}
	if temperature <= 0 {
		temperature = 1.0
	}

	scaled := make(map[int]float64, len(logits))
	maxVal := math.Inf(-1)
	for i, v := range logits {
		s := v / temperature
		if used != nil && used[i] && penalty > 0 {
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
		e := math.Exp(v - maxVal)
		if e != e { // NaN
			e = 0
		}
		scaled[i] = e
		sum += e
	}
	if sum <= 0 || math.IsInf(sum, 0) {
		share := 1.0 / float64(len(scaled))
		uniform := make(map[int]float64, len(scaled))
		for i := range scaled {
			uniform[i] = share
		}
		return uniform
	}
	for i := range scaled {
		scaled[i] /= sum
	}
	return scaled
}

// renormalize rescales a sparse probability map in place to sum to one.
func renormalize(probs map[int]float64) {
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
