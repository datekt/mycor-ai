/*
Package config holds runtime tunables for MYCOR.

Every tunable lives in a Config value behind a package-level RWMutex instead of
a bare variable. Readers take a cheap snapshot through Get; writers go through
Set, which validates and clamps before publishing the new value. This lets the
web server mutate settings from slider requests while the engine reads them on
every training and generation step without a data race.

Previously hidden constants (thinking steps, vocabulary thresholds, history
limits) now live here as well, so nothing about the model's behaviour is baked
into the engine as a compile-time constant.

The active configuration can be persisted to and restored from a JSON file so
that it survives restarts.
*/
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Documented defaults.
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
	DefaultThinkingSteps     = 4
	DefaultMinVocabThinking  = 10
	DefaultMaxRecentContexts = 20
	DefaultMaxSessionHistory = 100
)

// Accepted ranges for user supplied values.
const (
	MinContextSize  = 1
	MaxContextSize  = 5
	MinTemperature  = 0.1
	MaxTemperature  = 1.5
	MinLearningRate = 0.01
	MaxLearningRate = 1.0
	MaxMomentum     = 0.99
	MaxTopK         = 1000
	MinThinkingStep = 1
	MaxThinkingStep = 16
	MinRepetition   = 1.0
	MaxRepetition   = 4.0
	MaxWeightDecay  = 1.0
)

// Config is an immutable snapshot of every runtime tunable.
type Config struct {
	LearningRate      float64 `json:"learningRate"`
	Temperature       float64 `json:"temperature"`
	RepetitionPenalty float64 `json:"repetitionPenalty"`
	Momentum          float64 `json:"momentum"`
	WeightDecay       float64 `json:"weightDecay"`
	IdleTimeoutSec    int     `json:"idleTimeoutSec"`
	IdleEnabled       bool    `json:"idleEnabled"`
	TopK              int     `json:"topK"`
	TopP              float64 `json:"topP"`
	ContextSize       int     `json:"contextSize"`
	ThinkingSteps     int     `json:"thinkingSteps"`
	MinVocabThinking  int     `json:"minVocabThinking"`
	MaxRecentContexts int     `json:"maxRecentContexts"`
	MaxSessionHistory int     `json:"maxSessionHistory"`
}

// Defaults returns a Config populated with every documented default.
func Defaults() Config {
	return Config{
		LearningRate:      DefaultLearningRate,
		Temperature:       DefaultTemperature,
		RepetitionPenalty: DefaultRepetitionPenalty,
		Momentum:          DefaultMomentum,
		WeightDecay:       DefaultWeightDecay,
		IdleTimeoutSec:    DefaultIdleTimeoutSec,
		IdleEnabled:       DefaultIdleEnabled,
		TopK:              DefaultTopK,
		TopP:              DefaultTopP,
		ContextSize:       DefaultContextSize,
		ThinkingSteps:     DefaultThinkingSteps,
		MinVocabThinking:  DefaultMinVocabThinking,
		MaxRecentContexts: DefaultMaxRecentContexts,
		MaxSessionHistory: DefaultMaxSessionHistory,
	}
}

var (
	mu      sync.RWMutex
	current = Defaults()
)

// Get returns a copy of the active configuration. Cheap enough to call on
// every generation step.
func Get() Config {
	mu.RLock()
	defer mu.RUnlock()
	return current
}

// Set validates and publishes a new configuration. Values outside their
// documented range are clamped rather than rejected, and the clamped result is
// returned so callers can echo the effective values back to the UI.
func Set(c Config) Config {
	mu.Lock()
	defer mu.Unlock()
	current = normalize(c)
	return current
}

// SetField applies a single named tunable, validating and clamping it. Unknown
// names report an error so that a typo in an API payload is not silently
// swallowed. flag is only consulted for boolean settings.
func SetField(name string, value float64, flag bool) (Config, error) {
	mu.Lock()
	defer mu.Unlock()
	switch name {
	case "learningRate":
		current.LearningRate = value
	case "temperature":
		current.Temperature = value
	case "momentum":
		current.Momentum = value
	case "topP":
		current.TopP = value
	case "weightDecay":
		current.WeightDecay = value
	case "repetitionPenalty":
		current.RepetitionPenalty = value
	case "topK":
		current.TopK = int(value)
	case "contextSize":
		current.ContextSize = int(value)
	case "idleTimeoutSec":
		current.IdleTimeoutSec = int(value)
	case "thinkingSteps":
		current.ThinkingSteps = int(value)
	case "idleEnabled":
		current.IdleEnabled = flag
	default:
		return current, fmt.Errorf("unknown setting %q", name)
	}
	current = normalize(current)
	return current, nil
}

// Reset restores every tunable to its documented default.
func Reset() {
	mu.Lock()
	defer mu.Unlock()
	current = Defaults()
}

// Save writes the active configuration as indented JSON to path, creating the
// parent directory when needed. The write is atomic: a temporary file is
// created next to the target and renamed into place.
func Save(path string) error {
	data, err := json.MarshalIndent(Get(), "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')

	dir := filepath.Dir(path)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// Load reads a configuration previously written by Save. Missing fields keep
// their current value, so a partial document upgrades gracefully. A missing
// file is not an error: the caller simply keeps the running configuration.
func Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if len(data) == 0 {
		return nil
	}

	mu.Lock()
	defer mu.Unlock()
	merged := current
	if err := json.Unmarshal(data, &merged); err != nil {
		return fmt.Errorf("parse config: %w", err)
	}
	current = normalize(merged)
	return nil
}

func normalize(c Config) Config {
	if isBlank(c) {
		return Defaults()
	}
	c.LearningRate = clamp(c.LearningRate, MinLearningRate, MaxLearningRate)
	c.Temperature = clamp(c.Temperature, MinTemperature, MaxTemperature)
	c.RepetitionPenalty = clamp(c.RepetitionPenalty, MinRepetition, MaxRepetition)
	c.Momentum = clamp(c.Momentum, 0, MaxMomentum)
	c.WeightDecay = clamp(c.WeightDecay, 0, MaxWeightDecay)
	c.TopK = int(clamp(float64(c.TopK), 0, MaxTopK))
	c.TopP = clamp(c.TopP, 0, 1)
	c.ContextSize = int(clamp(float64(c.ContextSize), MinContextSize, MaxContextSize))
	c.ThinkingSteps = int(clamp(float64(c.ThinkingSteps), MinThinkingStep, MaxThinkingStep))
	c.MinVocabThinking = int(clamp(float64(c.MinVocabThinking), 0, 100000))
	c.MaxRecentContexts = int(clamp(float64(c.MaxRecentContexts), 1, 10000))
	c.MaxSessionHistory = int(clamp(float64(c.MaxSessionHistory), 1, 100000))
	if c.IdleTimeoutSec < 1 || c.IdleTimeoutSec > 3600 {
		c.IdleTimeoutSec = DefaultIdleTimeoutSec
	}
	return c
}

// isBlank reports whether a decoded document carries no usable information at
// all, which happens when a partial JSON object was fed to Unmarshal.
func isBlank(c Config) bool {
	return c.LearningRate == 0 && c.Temperature == 0 && c.TopK == 0 && c.ContextSize == 0
}

func clamp(v, lo, hi float64) float64 {
	// NaN fails every comparison, so it is caught first and mapped to the
	// lower bound rather than propagating into the model.
	if v != v {
		return lo
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
