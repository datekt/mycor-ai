#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p bin
go build -trimpath -ldflags="-s -w" -o bin/mycor ./cmd/mycor

echo "Built bin/mycor"