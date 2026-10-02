# Support

## Documentation

- `README.md` — overview, quick start, commands
- `docs/ARCHITECTURE.md` — how the engine works inside
- `docs/BACKOFF.md` — Dynamic Backoff explained
- `docs/THINKING.md` — Thinking Mode and Idle Thinking

## Where to ask

- **GitHub Discussions** — general questions, help with training, showcase
  your own trained `brain.gob`.
- **GitHub Issues** — bug reports and feature requests only. Use the
  provided templates.

## Before opening an issue

1. Make sure you are on the latest `main`:

       git pull
       make build

2. Run the test suite and include the output:

       make test

3. Include your environment:

       go version
       go env GOOS GOARCH

4. If the problem is related to your trained brain, attach `brain.gob`.
   Small files (under 1 MB) are fine to attach directly. For larger ones,
   use a gist or a file-sharing link.

## Response times

This is a volunteer-driven project. We try to respond within a few days,
but there is no guaranteed SLA. Clear, well-scoped reports get faster
answers.