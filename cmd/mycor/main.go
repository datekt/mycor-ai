/*
Command mycor is the MYCOR AI entry point.

MYCOR is a pure-Go, zero-dependency N-gram language model with an adaptive
context window, dynamic backoff, Top-K and Top-P sampling, thinking mode,
idle daydreams, and persistent weights. v5.0 launches a local HTTP server
with an embedded single-page interface and opens the user's default browser.
*/
package main

import (
	"mycor/internal/web"
)

func main() {
	web.Run()
}
