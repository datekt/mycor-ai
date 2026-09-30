$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..")

New-Item -ItemType Directory -Force -Path "bin" | Out-Null
go build -trimpath -ldflags="-s -w" -o bin/mycor.exe ./cmd/mycor

Write-Host "Built bin/mycor.exe"
Pop-Location