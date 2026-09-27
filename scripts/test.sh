#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

go vet ./...
go test ./... -race -coverprofile=cover.out
go tool cover -func=cover.out | tail -n 1