# Contributing to MYCOR

Thanks for your interest in improving MYCOR. This document explains how to
contribute code, docs, and bug reports.

## Getting started

1. Fork the repository and clone your fork:

       git clone https://github.com/<your-user>/MYCOR.git
       cd MYCOR

2. Make sure you have Go 1.27.1 or later:

       go version

3. Fetch dependencies and build:

       make build

4. Run the test suite:

       make test

## Project layout

- `cmd/mycor/` — main entry point
- `internal/cli/` — REPL, commands, idle ticker, input reader
- `internal/config/` — runtime configuration
- `internal/engine/` — tokenizer, model, training, generation, backoff, thinking, persistence
- `internal/i18n/` — English and Russian message packs
- `internal/importer/` — bulk TXT import
- `testdata/` — fixtures for tests
- `docs/` — long-form documentation

## Code style

- Go style is enforced by `gofmt`, `goimports` and `golangci-lint`.
- Run `make fmt` before committing.
- Do not add line comments (`//`). Use package-level and function-level
  documentation comments (`/* ... */`) or none at all. This is a hard rule.
- Prefer short functions with a single responsibility.
- Every new package needs a package-level documentation comment.

## Tests

- Every behavioral change must come with tests.
- Use `testdata/` for fixtures instead of inlining JSON in test files.
- Run `make race` before opening a PR.

## Commits

Use conventional commit prefixes:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code change that neither fixes a bug nor adds a feature
- `test:` adding or fixing tests
- `chore:` build, dependencies, tooling

## Pull requests

1. Create a feature branch: `git checkout -b feat/short-description`.
2. Keep the diff focused. One logical change per PR.
3. Fill in the PR template.
4. Make sure CI is green.
5. Request a review from a maintainer.

## Reporting bugs

Use the bug report template in `.github/ISSUE_TEMPLATE/bug_report.md`.
Include your OS, Go version, MYCOR version, and `history.json` if the bug
depends on the trained state.

## Security

Do not open public issues for security problems. See `.github/SECURITY.md`.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License, the same license that covers the project.