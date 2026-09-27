package cli

import (
	"bufio"
	"os"
	"strings"
)

type inputReader struct {
	ch chan string
}

func newInputReader() *inputReader {
	ir := &inputReader{ch: make(chan string, 10)}
	go func() {
		reader := bufio.NewReader(os.Stdin)
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
