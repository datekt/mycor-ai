package engine

import (
	"math"

	"mycor/internal/config"
)

func Train(prompt, target string) float64 {
	SaveBackup()
	return train(prompt, target)
}

func TrainBatch(prompt, target string) float64 {
	return train(prompt, target)
}

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

	trainStep := func(ctxKey string, targetWord string) {
		targetIdx, ok := WordToIdx[targetWord]
		if !ok {
			return
		}
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
	}

	contextSize := currentContextSize()
	history := make([]string, 0, len(promptWords)+len(targetWords))
	history = append(history, promptWords...)
	history = append(history, targetWords...)

	for i := len(promptWords); i < len(history); i++ {
		targetWord := history[i]
		ctx := buildContext(history[:i], contextSize)
		for _, key := range backoffChain(resolveContext(ctx)) {
			trainStep(key, targetWord)
		}
	}

	if steps == 0 {
		return 0.0
	}
	return totalLoss / steps
}
