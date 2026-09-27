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

	if len(promptWords) >= 2 {
		trainStep(MakeContextKey(unkToken, promptWords[0]), promptWords[1])
	}

	w1, w2 := initialContext(promptWords)
	for _, nextWord := range targetWords {
		for _, ctx := range backoffChain(w1, w2) {
			trainStep(ctx, nextWord)
		}
		w1, w2 = w2, nextWord
	}

	if steps == 0 {
		return 0.0
	}
	return totalLoss / steps
}
