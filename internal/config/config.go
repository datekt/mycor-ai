/*
Package config holds runtime tunables for MYCOR.

Values are plain package-level variables with no accessors. The CLI writes
to them directly when the user runs /temp, /lr, /momentum, or /idle. The
engine reads them on every training and generation step, so changes take
effect immediately without restarting. Reset restores every tunable to its
documented default, which /reset relies on.
*/
package config

const (
	DefaultLearningRate      = 0.5
	DefaultTemperature       = 0.7
	DefaultRepetitionPenalty = 1.2
	DefaultMomentum          = 0.9
	DefaultWeightDecay       = 0.0001
	DefaultIdleTimeoutSec    = 45
	DefaultIdleEnabled       = true
)

var (
	LearningRate      = DefaultLearningRate
	Temperature       = DefaultTemperature
	RepetitionPenalty = DefaultRepetitionPenalty
	Momentum          = DefaultMomentum
	WeightDecay       = DefaultWeightDecay
	IdleTimeoutSec    = DefaultIdleTimeoutSec
	IdleEnabled       = DefaultIdleEnabled
)

func Reset() {
	LearningRate = DefaultLearningRate
	Temperature = DefaultTemperature
	RepetitionPenalty = DefaultRepetitionPenalty
	Momentum = DefaultMomentum
	WeightDecay = DefaultWeightDecay
	IdleTimeoutSec = DefaultIdleTimeoutSec
	IdleEnabled = DefaultIdleEnabled
}
