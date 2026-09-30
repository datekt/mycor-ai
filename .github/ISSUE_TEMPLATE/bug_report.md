---
name: Bug report
about: Something is broken
title: "[BUG] "
labels: [bug]
assignees: []
---

## Description

A clear and concise description of what the bug is.

## Steps to reproduce

1. Run `mycor.exe` (or `.\scripts\build.ps1` followed by `.\bin\mycor.exe` if building from source)
2. Type `...`
3. See error

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened. If the browser console shows an error, paste it here. If the terminal that launched `mycor.exe` printed something, include that too.

## Environment

- OS: [e.g. Windows 11, Ubuntu 24.04]
- Go version (if built from source): [output of `go version`]
- MYCOR version: [e.g. 5.0]
- Vocabulary size: [shown in the sidebar under "Statistics"]
- Brain path: [the `Brain file:` line printed by the executable on startup, e.g. `C:\Users\you\AppData\Roaming\MYCOR\history.json`]

## History file

If the bug depends on your trained brain, attach `history.json` (or a minimal reproducible version of it). The file lives in the user configuration directory:

- **Windows:** `%AppData%\MYCOR\history.json`
- **Linux:** `~/.config/MYCOR/history.json`
- **macOS:** `~/Library/Application Support/MYCOR/history.json`

The exact path is printed in the terminal on every launch.