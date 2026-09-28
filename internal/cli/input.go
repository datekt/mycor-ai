package cli

import (
	"bufio"
	"strings"
)

type inputReader struct {
	ch chan string
}

func newInputReader(reader *bufio.Reader) *inputReader {
	ir := &inputReader{ch: make(chan string, 10)}
	go func() {
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				close(ir.ch)
				return
			}
			ir.ch <- strings.TrimSpace(line)
		}
	}()
	return ir
}
