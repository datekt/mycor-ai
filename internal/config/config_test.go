package config

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestDefaults(t *testing.T) {
	Reset()
	c := Get()
	if c.LearningRate <= 0 || c.LearningRate > 1 {
		t.Errorf("LearningRate default out of range: %v", c.LearningRate)
	}
	if c.Temperature < 0.1 || c.Temperature > 1.5 {
		t.Errorf("Temperature default out of range: %v", c.Temperature)
	}
	if c.RepetitionPenalty < 1.0 {
		t.Errorf("RepetitionPenalty should be >= 1.0, got %v", c.RepetitionPenalty)
	}
	if c.Momentum < 0 || c.Momentum > 0.99 {
		t.Errorf("Momentum default out of range: %v", c.Momentum)
	}
	if c.WeightDecay < 0 {
		t.Errorf("WeightDecay should be non-negative, got %v", c.WeightDecay)
	}
	if c.IdleTimeoutSec <= 0 {
		t.Errorf("IdleTimeoutSec should be positive, got %v", c.IdleTimeoutSec)
	}
	if !c.IdleEnabled {
		t.Error("IdleEnabled should default to true")
	}
	if c.ThinkingSteps != DefaultThinkingSteps {
		t.Errorf("ThinkingSteps = %d, want %d", c.ThinkingSteps, DefaultThinkingSteps)
	}
	if c.MinVocabThinking != DefaultMinVocabThinking {
		t.Errorf("MinVocabThinking = %d, want %d", c.MinVocabThinking, DefaultMinVocabThinking)
	}
	if c.MaxRecentContexts <= 0 || c.MaxSessionHistory <= 0 {
		t.Errorf("history limits should be positive: %+v", c)
	}
}

// Get returns a copy: mutating it must not affect the live configuration.
func TestGetReturnsCopy(t *testing.T) {
	Reset()
	c := Get()
	c.Temperature = 1.49
	if Get().Temperature != DefaultTemperature {
		t.Error("mutating the snapshot changed the live configuration")
	}
}

func TestSetField(t *testing.T) {
	Reset()
	if _, err := SetField("temperature", 0.3, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := Get().Temperature; got != 0.3 {
		t.Errorf("Temperature = %v, want 0.3", got)
	}
	if _, err := SetField("idleEnabled", 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if Get().IdleEnabled {
		t.Error("IdleEnabled should be false")
	}
}

func TestSetFieldRejectsUnknown(t *testing.T) {
	Reset()
	if _, err := SetField("nonsense", 1, false); err == nil {
		t.Error("expected an error for an unknown setting")
	}
}

func TestSetFieldClamps(t *testing.T) {
	Reset()
	if _, err := SetField("temperature", 99, false); err != nil {
		t.Fatal(err)
	}
	if got := Get().Temperature; got != MaxTemperature {
		t.Errorf("Temperature = %v, want clamp to %v", got, MaxTemperature)
	}
	if _, err := SetField("contextSize", -5, false); err != nil {
		t.Fatal(err)
	}
	if got := Get().ContextSize; got != MinContextSize {
		t.Errorf("ContextSize = %v, want clamp to %v", got, MinContextSize)
	}
}

func TestReset(t *testing.T) {
	Reset()
	if _, err := SetField("temperature", 0.1, false); err != nil {
		t.Fatal(err)
	}
	if _, err := SetField("learningRate", 0.99, false); err != nil {
		t.Fatal(err)
	}
	if _, err := SetField("idleEnabled", 0, false); err != nil {
		t.Fatal(err)
	}
	Reset()

	c := Get()
	if c.Temperature != DefaultTemperature {
		t.Errorf("Temperature = %v, want %v", c.Temperature, DefaultTemperature)
	}
	if c.LearningRate != DefaultLearningRate {
		t.Errorf("LearningRate = %v, want %v", c.LearningRate, DefaultLearningRate)
	}
	if c.IdleEnabled != DefaultIdleEnabled {
		t.Errorf("IdleEnabled = %v, want %v", c.IdleEnabled, DefaultIdleEnabled)
	}
}

// The configuration must survive a round trip through disk, which is what
// makes slider settings persist between runs.
func TestSaveLoadRoundTrip(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "config.json")
	if _, err := SetField("temperature", 1.11, false); err != nil {
		t.Fatal(err)
	}
	if _, err := SetField("topK", 7, false); err != nil {
		t.Fatal(err)
	}
	if err := Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	Reset()
	if Get().Temperature != DefaultTemperature {
		t.Fatal("reset did not clear the value")
	}
	if err := Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}
	c := Get()
	if c.Temperature != 1.11 {
		t.Errorf("Temperature = %v, want 1.11", c.Temperature)
	}
	if c.TopK != 7 {
		t.Errorf("TopK = %v, want 7", c.TopK)
	}
}

func TestLoadMissingFileIsNoop(t *testing.T) {
	Reset()
	if err := Load(filepath.Join(t.TempDir(), "absent.json")); err != nil {
		t.Errorf("missing config should not be an error: %v", err)
	}
}

func TestLoadPartialDocument(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	path := filepath.Join(t.TempDir(), "partial.json")
	if err := os.WriteFile(path, []byte(`{"temperature":1.4}`), 0644); err != nil {
		t.Fatal(err)
	}
	if err := Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}
	c := Get()
	if c.Temperature != 1.4 {
		t.Errorf("Temperature = %v, want 1.4", c.Temperature)
	}
	// Untouched fields keep the previous value.
	if c.TopK != DefaultTopK {
		t.Errorf("TopK = %v, want default %v", c.TopK, DefaultTopK)
	}
}

func TestLoadRejectsCorruptFile(t *testing.T) {
	Reset()
	path := filepath.Join(t.TempDir(), "broken.json")
	if err := os.WriteFile(path, []byte(`{not json`), 0644); err != nil {
		t.Fatal(err)
	}
	if err := Load(path); err == nil {
		t.Error("expected an error for a corrupt config file")
	}
}

func TestNormalizeRejectsNaN(t *testing.T) {
	Reset()
	if _, err := SetField("temperature", math.NaN(), false); err != nil {
		t.Fatal(err)
	}
	if got := Get().Temperature; math.IsNaN(got) {
		t.Error("NaN temperature should have been clamped to a real number")
	}
}

// Concurrent readers and writers must not race.
func TestConcurrentAccess(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				if j%3 == 0 {
					if _, err := SetField("temperature", float64(j%100)/100+0.1, false); err != nil {
						t.Error(err)
						return
					}
					continue
				}
				_ = Get()
			}
		}()
	}
	wg.Wait()
}

func TestJSONShape(t *testing.T) {
	Reset()
	data, err := json.Marshal(Get())
	if err != nil {
		t.Fatal(err)
	}
	var round map[string]interface{}
	if err := json.Unmarshal(data, &round); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"temperature", "learningRate", "topK", "contextSize", "thinkingSteps"} {
		if _, ok := round[key]; !ok {
			t.Errorf("serialised config is missing %q", key)
		}
	}
}
