package engine

import (
	"container/heap"

	"mycor/internal/config"
)

// probPair is one candidate index/probability pair.
type probPair struct {
	idx int
	p   float64
}

// minHeap keeps the k largest probabilities, exposing the smallest of them at
// the root so it can be evicted in O(log k).
type minHeap []probPair

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].p < h[j].p }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) {
	*h = append(*h, x.(probPair))
}
func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// selectTopK returns the indices of the k largest probabilities without
// sorting the whole dictionary.
//
// The previous implementation collected every non-zero entry into a slice and
// ran a full sort.Slice over it on every generated token: O(n log n) plus an
// allocation proportional to the vocabulary. A bounded heap keeps only k
// candidates live, so the cost is O(n log k) with O(k) memory.
func selectTopK(probs map[int]float64, k int) []int {
	if k <= 0 || k >= len(probs) {
		out := make([]int, 0, len(probs))
		for idx := range probs {
			out = append(out, idx)
		}
		return out
	}

	h := &minHeap{}
	heap.Init(h)
	for idx, p := range probs {
		if p <= 0 {
			continue
		}
		if h.Len() < k {
			heap.Push(h, probPair{idx, p})
			continue
		}
		if (*h)[0].p < p {
			(*h)[0] = probPair{idx, p}
			heap.Fix(h, 0)
		}
	}
	out := make([]int, h.Len())
	for i, pair := range *h {
		out[i] = pair.idx
	}
	return out
}

// narrowTopP trims an already-narrowed candidate list to the nucleus: the
// shortest prefix of the descending-sorted candidates whose cumulative mass
// reaches p.
//
// Running Top-P after Top-K is what makes this cheap. The candidate list is
// bounded by the Top-K setting (tens of entries), so the sort is negligible,
// whereas the old code sorted the entire vocabulary before discarding almost
// all of it. A p of zero degenerates to the single most probable entry.
func narrowTopP(candidates []int, probs map[int]float64, p float64) []int {
	if len(candidates) == 0 {
		return candidates
	}
	sortByProbDesc(candidates, probs)

	if p <= 0 {
		return candidates[:1]
	}
	if p >= 1 {
		return candidates
	}
	cumulative := 0.0
	for i, idx := range candidates {
		cumulative += probs[idx]
		if cumulative >= p {
			return candidates[:i+1]
		}
	}
	return candidates
}

// sortByProbDesc sorts a short index slice by descending probability. It uses
// insertion sort, which beats sort.Slice here because the slices are tiny and
// the comparator needs a map lookup anyway.
func sortByProbDesc(idxs []int, probs map[int]float64) {
	for i := 1; i < len(idxs); i++ {
		cur := idxs[i]
		j := i - 1
		for j >= 0 && probs[idxs[j]] < probs[cur] {
			idxs[j+1] = idxs[j]
			j--
		}
		idxs[j+1] = cur
	}
}

// sampleNextIndexLocked picks the next vocabulary index for a context using
// Top-K, Top-P and the repetition penalty. The caller must hold b.mu.
func (b *Brain) sampleNextIndexLocked(rawContext []string, used map[int]bool, cfg config.Config) (int, bool) {
	chain := backoffChain(b.resolveContextLocked(rawContext))

	var (
		logits map[int]float64
		ctxKey string
		found  bool
	)
	for _, key := range chain {
		if v, ok := b.weights[key]; ok && len(v) > 0 {
			logits = v
			ctxKey = key
			found = true
			break
		}
	}
	if !found {
		return 0, false
	}

	b.touchContextLocked(ctxKey)

	probs := b.softmaxSparse(logits, cfg.Temperature, used, cfg.RepetitionPenalty)
	if len(probs) == 0 {
		return 0, false
	}

	candidates := narrowTopP(selectTopK(probs, cfg.TopK), probs, cfg.TopP)
	if len(candidates) == 0 {
		return 0, false
	}

	total := 0.0
	for _, idx := range candidates {
		total += probs[idx]
	}
	if total <= 0 {
		return 0, false
	}

	r := b.rng.Float64() * total
	cumulative := 0.0
	for _, idx := range candidates {
		cumulative += probs[idx]
		if r <= cumulative {
			return idx, true
		}
	}
	return candidates[len(candidates)-1], true
}

// randomUnusedIndex picks a random index other than <unk>, used when a context
// has no weights at all. The caller must hold b.mu.
func (b *Brain) randomUnusedIndex() (int, bool) {
	if len(b.vocab) <= 1 {
		return 0, false
	}
	idx := b.rng.Intn(len(b.vocab)-1) + 1
	if idx != unkIndex {
		return idx, true
	}
	return 1, len(b.vocab) > 1
}
