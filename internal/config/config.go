/*
Package config holds runtime tunables for MYCOR.

Values are plain package-level variables with no accessors. The web server
writes to them directly through its JSON API when the user moves a slider.
The engine reads them on every training and generation step, so changes
take effect immediately without restarting. Reset restores every tunable
to its documented default.
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
	DefaultTopK              = 40
	DefaultTopP              = 0.9
	DefaultContextSize       = 2
	MinContextSize           = 1
	MaxContextSize           = 5
)

var (
	LearningRate      = DefaultLearningRate
	Temperature       = DefaultTemperature
	RepetitionPenalty = DefaultRepetitionPenalty
	Momentum          = DefaultMomentum
	WeightDecay       = DefaultWeightDecay
	IdleTimeoutSec    = DefaultIdleTimeoutSec
	IdleEnabled       = DefaultIdleEnabled
	TopK              = DefaultTopK
	TopP              = DefaultTopP
	ContextSize       = DefaultContextSize
)

func Reset() {
	LearningRate = DefaultLearningRate
	Temperature = DefaultTemperature
	RepetitionPenalty = DefaultRepetitionPenalty
	Momentum = DefaultMomentum
	WeightDecay = DefaultWeightDecay
	IdleTimeoutSec = DefaultIdleTimeoutSec
	IdleEnabled = DefaultIdleEnabled
	TopK = DefaultTopK
	TopP = DefaultTopP
	ContextSize = DefaultContextSize
}
