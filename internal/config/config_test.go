package config

import "testing"

func TestDefaults(t *testing.T) {
	if LearningRate <= 0 || LearningRate > 1 {
		t.Errorf("LearningRate default out of range: %v", LearningRate)
	}
	if Temperature < 0.1 || Temperature > 1.5 {
		t.Errorf("Temperature default out of range: %v", Temperature)
	}
	if RepetitionPenalty < 1.0 {
		t.Errorf("RepetitionPenalty should be >= 1.0, got %v", RepetitionPenalty)
	}
	if Momentum < 0 || Momentum > 0.99 {
		t.Errorf("Momentum default out of range: %v", Momentum)
	}
	if WeightDecay < 0 {
		t.Errorf("WeightDecay should be non-negative, got %v", WeightDecay)
	}
	if IdleTimeoutSec <= 0 {
		t.Errorf("IdleTimeoutSec should be positive, got %v", IdleTimeoutSec)
	}
	if !IdleEnabled {
		t.Error("IdleEnabled should default to true")
	}
}

func TestMutable(t *testing.T) {
	original := Temperature
	Temperature = 0.3
	if Temperature != 0.3 {
		t.Errorf("Temperature not mutable, got %v", Temperature)
	}
	Temperature = original
}

func TestReset(t *testing.T) {
	Temperature = 0.1
	LearningRate = 0.99
	Momentum = 0.0
	IdleEnabled = false
	IdleTimeoutSec = 1

	Reset()

	if Temperature != DefaultTemperature {
		t.Errorf("Temperature = %v, want %v", Temperature, DefaultTemperature)
	}
	if LearningRate != DefaultLearningRate {
		t.Errorf("LearningRate = %v, want %v", LearningRate, DefaultLearningRate)
	}
	if Momentum != DefaultMomentum {
		t.Errorf("Momentum = %v, want %v", Momentum, DefaultMomentum)
	}
	if IdleEnabled != DefaultIdleEnabled {
		t.Errorf("IdleEnabled = %v, want %v", IdleEnabled, DefaultIdleEnabled)
	}
	if IdleTimeoutSec != DefaultIdleTimeoutSec {
		t.Errorf("IdleTimeoutSec = %v, want %v", IdleTimeoutSec, DefaultIdleTimeoutSec)
	}
}
