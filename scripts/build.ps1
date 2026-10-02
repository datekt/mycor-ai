$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

# One target per invocation: cross-compiling all platforms in a single run can
# outlive the calling shell, which kills the remaining builds.
$target = $args[0]
$goos = $args[1]
$goarch = $args[2]
if (-not $target) {
    Write-Host "usage: build-one.ps1 <output-name> <goos> <goarch>"
    exit 2
}

New-Item -ItemType Directory -Force -Path "bin" | Out-Null
$env:CGO_ENABLED = "0"
$env:GOOS = $goos
$env:GOARCH = $goarch

$goArgs = @("build", "-trimpath", "-ldflags", "-s -w", "-o", "bin/$target", "./cmd/mycor")
& go @goArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL $target"
    exit 1
}
Write-Host "OK   $target"