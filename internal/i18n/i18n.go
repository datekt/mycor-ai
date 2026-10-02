/*
Package i18n tracks the active user interface language.

The message catalogues used to live here, written in Go and formatted with
Printf. They were only ever consumed by the terminal REPL, which no longer
exists; the browser client now carries its own translations in
internal/web/static/app.js and remembers the choice in localStorage.

What remains is the small piece of server state the API still needs: the
current language, reported in /api/stats and settable through /api/lang.

Access is guarded by a mutex because handlers run concurrently.
*/
package i18n

import "sync"

// Lang is a supported interface language.
type Lang string

// Supported languages.
const (
	EN Lang = "en"
	RU Lang = "ru"
)

// Valid reports whether l is a supported language.
func Valid(l Lang) bool { return l == EN || l == RU }

var (
	mu      sync.RWMutex
	current = EN
)

// Init sets the initial language, ignoring unsupported values.
func Init(l Lang) { Set(l) }

// Set changes the active language, ignoring unsupported values.
func Set(l Lang) {
	if !Valid(l) {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	current = l
}

// Current returns the active language.
func Current() Lang {
	mu.RLock()
	defer mu.RUnlock()
	return current
}
