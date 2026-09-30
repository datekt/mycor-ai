#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p bin

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        OUT="bin/mycor.exe"
        ;;
    *)
        OUT="bin/mycor"
        ;;
esac

go build -trimpath -ldflags="-s -w" -o "$OUT" ./cmd/mycor

echo "Built $OUT"