package engine

import (
	"math"

	"mycor/internal/config"
)

// Train performs one learning step and makes it undoable via UndoLastTrain.
func (b *Brain) Train(prompt, target string) float64 {
	b.trainMu.Lock()
	defer b.trainMu.Unlock()

	b.mu.Lock()
	b.beginUndo()
	b.undo.started = true
	b.mu.Unlock()

	return b.train(prompt, target)
}

// TrainBatch performs one learning step without creating an undo point, so a
// bulk import cannot be reversed by mistake. It still serialises against
// concurrent training through trainMu.
func (b *Brain) TrainBatch(prompt, target string) float64 {
	b.trainMu.Lock()
	defer b.trainMu.Unlock()
	return b.train(prompt, target)
}

// train updates the weights for one (prompt, target) pair.
//
// trainMu must be held. The state lock is taken and released around each
// individual token update, so a long-running import does not monopolise the
// brain and chat requests can still be served between steps.
func (b *Brain) train(prompt, target string) float64 {
	cfg := config.Get()

	promptWords := Tokenize(prompt)
	targetWords := Tokenize(target)
	if len(promptWords) == 0 || len(targetWords) == 0 {
		return 0.0
	}

	b.mu.Lock()
	b.RegisterWords(promptWords)
	b.RegisterWords(targetWords)
	b.mu.Unlock()

	lr := cfg.LearningRate
	decay := 1.0 - lr*cfg.WeightDecay
	momentum := cfg.Momentum
	if momentum < 0 {
		momentum = 0
	}
	if momentum > config.MaxMomentum {
		momentum = config.MaxMomentum
	}

	csize := contextSize(cfg)
	history := make([]string, 0, len(promptWords)+len(targetWords))
	history = append(history, promptWords...)
	history = append(history, targetWords...)

	totalLoss := 0.0
	steps := 0.0

	for i := len(promptWords); i < len(history); i++ {
		targetWord := history[i]

		b.mu.Lock()
		ctx := buildContext(history[:i], csize)
		targetIdx, known := b.wordToIdx[targetWord]
		chain := backoffChain(b.resolveContextLocked(ctx))

		var stepLoss float64
		var counted bool
		if known {
			for _, key := range chain {
				l, c := b.trainStepLocked(key, targetIdx, lr, decay, momentum)
				if c {
					stepLoss += l
					counted = true
				}
			}
		}
		b.mu.Unlock()

		if counted {
			totalLoss += stepLoss
			steps++
		}
	}

	if steps == 0 {
		return 0.0
	}
	return totalLoss / steps
}

// trainStepLocked updates one context toward targetIdx and returns the loss
// contribution. The caller must hold b.mu for writing.
func (b *Brain) trainStepLocked(ctxKey string, targetIdx int, lr, decay, momentum float64) (float64, bool) {
	b.touchContextLocked(ctxKey)

	vec := b.ensureSparse(ctxKey)
	vel := b.velocity[ctxKey]

	probs := b.softmaxSparse(vec, 1.0, nil, 0)

	loss, counted := 0.0, false
	if p := probs[targetIdx]; p > 0 {
		loss = -math.Log(p)
		counted = true
	}

	// The update only touches entries that already carry a weight, plus the
	// target itself. Sparseness is preserved: absent entries stay absent
	// rather than being materialised with a random weight.
	//
	// A missing target has probability zero already, which is exactly the
	// gradient a fresh entry needs (grad = 1 - 0 = 1).
	if _, ok := vec[targetIdx]; !ok {
		vec[targetIdx] = b.initWeight()
	}
	vel[targetIdx] = momentum*vel[targetIdx] + (1.0-momentum)*(1.0-probs[targetIdx])
	vec[targetIdx] = vec[targetIdx]*decay + lr*vel[targetIdx]

	for idx, p := range probs {
		if idx == targetIdx {
			continue
		}
		if _, ok := vec[idx]; !ok {
			continue
		}
		grad := -p
		vel[idx] = momentum*vel[idx] + (1.0-momentum)*grad
		vec[idx] = vec[idx]*decay + lr*vel[idx]
	}
	return loss, counted
}

// buildContext returns the last n words of history, left-padded with <unk>.
// It is a pure function and needs no lock.
func buildContext(history []string, n int) []string {
	ctx := make([]string, n)
	start := len(history) - n
	for i := 0; i < n; i++ {
		pos := start + i
		if pos < 0 {
			ctx[i] = unkToken
		} else {
			ctx[i] = history[pos]
		}
	}
	return ctx
}
