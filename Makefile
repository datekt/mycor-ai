.PHONY: build build-windows build-linux build-macos build-all run test race cover lint check vet fmt clean tidy

BIN_DIR := bin
MAIN    := ./cmd/mycor

ifeq ($(OS),Windows_NT)
BINARY := $(BIN_DIR)/mycor.exe
else
BINARY := $(BIN_DIR)/mycor
endif

build:
	go build -trimpath -ldflags="-s -w" -o $(BINARY) $(MAIN)

build-windows:
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o $(BIN_DIR)/mycor-windows-amd64.exe $(MAIN)

build-linux:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o $(BIN_DIR)/mycor-linux-amd64 $(MAIN)
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -trimpath -ldflags="-s -w" -o $(BIN_DIR)/mycor-linux-arm64 $(MAIN)

build-macos:
	CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o $(BIN_DIR)/mycor-macos-amd64 $(MAIN)
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -trimpath -ldflags="-s -w" -o $(BIN_DIR)/mycor-macos-arm64 $(MAIN)

build-all: build-windows build-linux build-macos

run:
	go run $(MAIN)

test:
	go test ./... -v

race:
	go test ./... -race

cover:
	go test ./... -coverprofile=cover.out
	go tool cover -html=cover.out

check:
	@echo "Checking gofmt..."
	@unformatted=$$(gofmt -s -l .); \
	if [ -n "$$unformatted" ]; then \
		echo "Unformatted files:"; \
		echo "$$unformatted"; \
		exit 1; \
	fi
	@echo "Running go vet..."
	go vet ./...

lint:
	golangci-lint run

vet:
	go vet ./...

fmt:
	gofmt -s -w .
	goimports -w .

tidy:
	go mod tidy

clean:
	rm -rf $(BIN_DIR)/ cover.out