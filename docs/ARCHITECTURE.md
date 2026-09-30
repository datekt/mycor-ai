# Architecture

MYCOR is a pure-Go, zero-dependency language model with a Markov-style
N-gram architecture. This document describes how the engine is put together
and where each responsibility lives.

> **v5.0 update:** the entry point moved from a terminal REPL to a local
> HTTP server with an embedded web UI. The persistence layer switched
> from JSON to `encoding/gob` (with transparent JSON fallback), and the
> fixed bigram context became adaptive N-gram. The trained brain now lives
> in the operating system's per-user configuration directory instead of
> next to the executable. The core architecture — global brain, dynamic
> backoff, thinking mode, snapshot undo — is unchanged.

## High-level picture

    +-------------+     +-------------+     +-------------+
    |   cmd/      | --> | internal/   | --> | internal/   |
    |   mycor/    |     |   web/      |     |   engine/   |
    +-------------+     +-------------+     +-------------+
                              |                   |
                              v                   v
                        +-------------+     +-------------+
                        | internal/   |     | internal/   |
                        |   i18n/     |     |  importer/  |
                        +-------------+     +-------------+
                              |
                              v
                        +-------------+
                        | internal/   |
                        |  config/    |
                        +-------------+

`cmd/mycor/main.go` calls `web.Run()`. The web layer is the only entry
point in v5.0. `internal/cli/` still exists but is not built by default
and is considered legacy.

## Package responsibilities

### `internal/web`

The HTTP server, the embedded single-page interface, and the platform
path resolution for the trained brain. Contains:

| File                | Responsibility                                       |
| ------------------- | ---------------------------------------------------- |
| `server.go`         | `Run`, route registration, JSON handlers             |
| `paths.go`          | `resolveBrainPath`, user config directory lookup     |
| `static/index.html` | The entire UI, embedded via `//go:embed`             |

`Run` resolves the brain path from the OS user configuration directory,
loads the brain, binds to `127.0.0.1:0` (OS-assigned port), prints the
URL, opens the user's default browser, and serves requests until
interrupted. Every handler locks a single package-level mutex before
touching engine state, so the engine itself can remain non-thread-safe.

The brain path is stored in a package-level variable so that all handlers
address the same file. On startup it is resolved by `resolveBrainPath`,
which calls `os.UserConfigDir` and creates a `MYCOR` subdirectory if
needed. If the config directory cannot be resolved, `Run` falls back to
`history.json` in the current working directory and prints a warning.

### `internal/engine`

The heart of the project. Subdivided into focused files:

| File              | Responsibility                                                |
| ----------------- | ------------------------------------------------------------- |
| `engine.go`       | Public façade, `InitEngine`, `ContextWeightSize`               |
| `model.go`        | Global state, `BrainModel`, `RegisterWord`, backup/undo        |
| `tokenizer.go`    | `Tokenize`, `JoinWords`, emoticon matching, punctuation        |
| `backoff.go`      | `backoffChain`, `sampleNextIndex`, `softmaxBase`, Top-K/Top-P  |
| `train.go`        | `Train`, `TrainBatch`, `train`, `buildContext`                 |
| `generate.go`     | `generate`, `GenerateResponse`, `GenerateIdleThought`          |
| `thinking.go`     | `think`, `Think`, `LastThoughts`                               |
| `history.go`      | `AppendHistory`, `SessionHistory`, `RecentContexts`, touching  |
| `persistence.go`  | `SaveBrain`, `LoadBrain`, gob/JSON decoding, validation        |

The package keeps global mutable state. This is deliberate: the model is a
single, process-wide brain, and the web layer treats it as such. Tests call
`InitEngine()` between cases to isolate state.

### `internal/cli` (legacy)

The v4.x terminal REPL. It still compiles and still calls the same public
engine API, but it is not referenced by `cmd/mycor/main.go` and its
version strings and command surface are frozen at the v4.x shape. It is
retained so that the REPL could be revived as a separate `cmd/mycor-cli`
binary if needed; it should not be considered part of the v5.0 release.

### `internal/i18n`

Two message packs (`en`, `ru`) behind a `Messages` struct. `Set(Lang)`
selects the active pack. `M()` returns the current pack. The web layer
uses `Current()` only to report the current language in `/api/stats`; all
user-visible strings in the browser are hardcoded in `index.html`.

### `internal/importer`

Bulk TXT trainer. Reads the file line by line, pairs consecutive non-empty
lines, and calls `engine.TrainBatch` for each pair. No backup is created,
so the **Undo** button cannot reverse a batch import — this is intentional.

### `internal/config`

A flat bag of tunables. No functions, no methods beyond `Reset`. The web
layer writes to them directly when the user moves a slider or clicks a
button. The engine reads them on every training and generation step.

## Data flow

### Training

    user input (prompt, target) via POST /api/train
        |
        v
    Tokenize(prompt), Tokenize(target)
        |
        v
    RegisterWord for every new token
        |
        v
    history := promptWords ++ targetWords
    for i in len(promptWords)..len(history):
        ctx := buildContext(history[:i], ContextSize)
        for key in backoffChain(resolveContext(ctx)):
            trainStep(key, history[i])
        |
        v
    trainStep:
        ensureWeights(ctxKey)
        softmaxBase(weights, 1.0, nil, 0)
        gradient = one-hot - probs
        velocity = m*v + (1-m)*grad
        weights  = w*decay + lr*velocity
        |
        v
    return mean cross-entropy loss

### Generation

    prompt via POST /api/chat
        |
        v
    Tokenize
        |
        v
    think(prompt) if vocab >= MinVocabForThinking
        |
        v
    for i in 0..maxLen:
        ctx := buildContext(history, ContextSize)
        sampleNextIndex(ctx, used, Temperature)
        if no context found:
            random vocabulary pick
        else:
            softmax -> Top-K -> Top-P -> renormalize -> sample
        append word, shift window
        stop on . ? !
        |
        v
    JoinWords

### Persistence

`SaveBrain` serializes the vocabulary, weights, and velocity with
`encoding/gob`. The stream is written to `<path>.tmp` and then atomically
renamed. Files are smaller and load faster than the v4.x JSON format.

`LoadBrain` calls `decodeBrain`, which inspects the first byte:

- `{` — treated as legacy JSON, parsed with `encoding/json`.
- Anything else — treated as a gob stream.

Both paths populate the same `BrainModel`. After decoding, validation runs:

1. Vocabulary is non-empty.
2. No empty strings.
3. No duplicate words.
4. If `<unk>` is missing at index 0, it is inserted and every weight
   vector is shifted right with a zero at index 0.
5. Every weight vector is resized to `len(Vocabulary)`.
6. Every velocity vector is resized to match its weight vector, or
   initialized to zeros.

Failures leave the previous in-memory state untouched.

The brain path itself is resolved by `internal/web/paths.go` via
`os.UserConfigDir()`. The engine never decides where to store the file —
it just writes to whatever path the caller passes to `SaveBrain` and reads
from whatever path the caller passes to `LoadBrain`.

## State lifecycle

    web.Run
        |
        v
    resolveBrainPath -> brainFile (package variable in internal/web)
        |
        v
    InitEngine
        |
        v
    LoadBrain(brainFile) (optional)
        |
        v
    +--> Train / TrainBatch --> SaveBackup --> train --> mutation
    |                                                 |
    |                                                 v
    |                                             SaveBrain(brainFile)
    |
    +--> UndoLastTrain --> restore from backup

The backup is a shallow copy of the current vocabulary, index map, weights
and velocity. Only the latest `Train` call can be undone. `TrainBatch`
deliberately does not create a backup, so a batch import cannot be rolled
back with a single click.

`InitEngine` clears everything except the on-disk brain:

- vocabulary -> `["<unk>"]`
- weights and velocity -> empty maps
- backup -> nil
- recent contexts -> empty
- session history -> empty
- last thoughts -> empty

## Numerical notes

- Softmax is computed with a max-shift for numerical stability and falls
  back to a uniform distribution if the scaled logits sum to zero, NaN,
  or infinity.
- Repetition penalty divides positive logits by the penalty factor and
  multiplies negative logits, so the direction of the penalty is always
  correct regardless of sign.
- Top-K keeps the `K` highest-probability tokens and zeroes the rest.
- Top-P keeps the smallest prefix of the sorted distribution whose
  cumulative probability reaches `P`. `P = 0` keeps only the argmax,
  `P = 1` keeps everything.
- Momentum is clamped to `[0, 0.99]` before use.
- Weight decay is applied multiplicatively: `w = w * (1 - lr * decay)`.

## Concurrency

The engine is not thread-safe. The web layer wraps every handler in a
package-level `sync.Mutex`, so all engine mutations and reads happen
under one lock. The HTTP server itself runs on many goroutines, but they
queue up on the mutex before entering the engine.

The idle polling endpoint (`GET /api/idle`) shares the same lock and the
same `lastActivity` timestamp as chat and training. This is intentional:
idle thoughts never race with user input.

The `brainFile` variable is written once during `Run`, before the HTTP
server starts listening, and is only read afterwards. No lock is required
for it.

## What is intentionally missing

- No context cancellation. Training and generation are fast enough.
- No streaming output. Answers are returned whole.
- No file locking on the brain file. Single-user desktop assumption.
- No authentication. The server binds to `127.0.0.1` on a random port.
- No plugin system. The engine is small enough to modify directly.