package cli

import (
	"fmt"
	"time"

	"mycor/internal/config"
	"mycor/internal/engine"
	"mycor/internal/i18n"
)

const idleTickInterval = 5 * time.Second

func shouldDaydream(lastActivity time.Time) bool {
	if !config.IdleEnabled {
		return false
	}
	return time.Since(lastActivity) >= time.Duration(config.IdleTimeoutSec)*time.Second
}

func emitDaydream() {
	thought := engine.GenerateIdleThought()
	if thought == "" {
		return
	}
	fmt.Printf("\n%s %s\n", i18n.M().IdleThought, thought)
}
