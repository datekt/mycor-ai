/*
Command mycor is the MYCOR AI entry point.

MYCOR is a pure-Go, zero-dependency N-gram language model with an adaptive
context window, dynamic backoff, Top-K and Top-P sampling, thinking mode,
idle daydreams, and persistent weights. It launches a local HTTP server with an
embedded single-page interface and opens the user's default browser.

The process installs a signal handler so that closing the window or pressing
Ctrl+C shuts the server down gracefully: in-flight requests are given a moment
to finish and the brain is flushed to disk before exit.
*/
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"mycor/internal/web"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := web.Run(ctx); err != nil {
		log.Fatalf("mycor: %v", err)
	}
}
