.PHONY: build run test race cover lint check vet fmt clean tidy

BINARY := bin/mycor
MAIN   := ./cmd/mycor

build:
	go build -o $(BINARY) $(MAIN)

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
	rm -rf bin/ cover.out