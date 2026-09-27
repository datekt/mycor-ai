/*
Command mycor is the MYCOR AI entry point.

MYCOR is a pure-Go, zero-dependency N-gram language model with a two-word
context window, dynamic backoff, thinking mode, idle daydreams, and
persistent weights. Run it interactively in the terminal.
*/
package main

import (
	"mycor/internal/cli"
)

func main() {
	cli.Run()
}
