/*
Package config holds runtime tunables for MYCOR.

Values are plain package-level variables with no accessors. The CLI writes
to them directly when the user runs /temp, /lr, /momentum, or /idle. The
engine reads them on every training and generation step, so changes take
effect immediately without restarting.
*/
package config

var (
	LearningRate      = 0.5
	Temperature       = 0.7
	RepetitionPenalty = 1.2
	Momentum          = 0.9
	WeightDecay       = 0.0001
	IdleTimeoutSec    = 45
	IdleEnabled       = true
)
