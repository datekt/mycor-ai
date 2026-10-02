# Architecture

MYCOR is a pure-Go, zero-dependency language model with a Markov-style
N-gram architecture. This document describes how the engine is put together
and where each responsibility lives.

> **v6.0 update:** the architecture moved from package-level globals to a
> `Brain` value with fine-grained locking, and the global request mutex that
> made the server single-threaded is gone. Weight vectors are sparse, so adding
> a word is O(1) instead of rewriting every context; Top-K/Top-P use bounded
> selection instead of sorting the whole vocabulary; undo journals only the
> contexts a step touched; and persistence is debounced rather than written
> synchronously on every training step. The brain file gained an explicit magic
> header, and the legacy terminal REPL was removed.

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

`cmd/mycor/main.go` calls `web.Run(ctx)`. The web layer is the only entry
point. The terminal REPL that shipped through v4.x was removed in v6.0: nothing
referenced it, and keeping a second, untested code path over the engine API
invited exactly the drift this release set out to remove.

## Package responsibilities

### `internal/web`

The HTTP server, the embedded single-page interface, and the platform
path resolution for the trained brain. Contains:

| File                   | Responsibility                                          |
| ---------------------- | ------------------------------------------------------- |
| `server.go`            | `Server`, `Run`, routing, shutdown, persistence         |
| `handlers.go`          | JSON handlers                                          |
| `saver.go`             | Debounced background persistence                       |
| `paths.go`             | data paths, import path sandboxing                     |
| `server_test.go`       | Handler, security and concurrency tests                |
| `static/index.html`    | UI markup, embedded via `//go:embed`                   |
| `static/app.js`        | UI logic, client-side i18n                             |

`Run` resolves the data paths from the OS user configuration directory,
loads the brain and the configuration, binds to `127.0.0.1:0` (OS-assigned
port), prints the URL, opens the user's default browser, and serves until the
context is cancelled.

**There is no package-level mutex.** This is the central design change of
v6.0. Previously every handler took one global mutex before touching engine
state, which serialised the whole application: a single long import blocked
every chat request, and the process was effectively single-threaded. Now:

- The brain guards its own state with a `sync.RWMutex` (`internal/engine`).
- The idle clock uses its own small mutex.
- Handlers touch only what they need and hold no lock across I/O.

The result is that chat, statistics and configuration stay responsive while an
import is running. Persistence is likewise asynchronous: handlers call
`saver.mark()`, and a background debouncer writes at most every few seconds.
`/api/save` forces an immediate write, and shutdown flushes.

Request bodies are capped (`maxBodyBytes`), import paths are validated against
an allow-list of directories, and shutdown drains in-flight requests before
persisting.

### `internal/engine`

The heart of the project. Subdivided into focused files:

| File              | Responsibility                                                |
| ----------------- | ------------------------------------------------------------- |
| `engine.go`       | `Brain` type, locking policy, `Stats`, context width          |
| `model.go`        | Vocabulary, sparse weights, undo journal, `CountParameters`  |
| `tokenizer.go`    | `Tokenize`, `JoinWords`, emoticon matching, punctuation        |
| `backoff.go`      | `MakeContextKey`, `backoffChain`, `softmaxSparse`, renormalise|
| `sampling.go`     | Bounded Top-K selection, Top-P narrowing, `sampleNextIndex`   |
| `train.go`        | `Train`, `TrainBatch`, `trainStep`, `buildContext`            |
| `generate.go`     | `GenerateResponse`, `GenerateIdleThought`                     |
| `thinking.go`     | `thinkLocked`, `Think`, `LastThoughts`                        |
| `history.go`      | `AppendHistory`, `SessionHistory`, `RecentContexts`, touching  |
| `persistence.go`  | `Save`, `Load`, format detection, validation                  |

**No package-level mutable state.** All model state lives in a `Brain` value
created by `NewBrain`, and every exported method is safe for concurrent use.
Tests create their own brain instead of calling `InitEngine` between cases, so
they can also run in parallel.

Two locks, with distinct jobs:

- `mu` (`sync.RWMutex`) guards the model state. Readers take `RLock`; writers
  take `Lock`. Generation holds it for the whole reply — a reply is a handful of
  tokens, so this is cheap.
- `trainMu` serialises training against itself. A training call holds `trainMu`
  for its entire duration but acquires and releases `mu` around each individual
  step, so a multi-minute import interleaves with chat instead of blocking it.

Weight vectors are sparse (`map[int]float64`, keyed by vocabulary index) with
index 0 reserved for `<unk>`. Two consequences:

- Adding a word to the vocabulary is O(1). The dense layout had to rewrite
  every existing context vector on each new word, making vocabulary growth
  quadratic.
- An undo only has to snapshot the contexts a step actually touched, instead of
  copying the entire model on every training call.

### `internal/importer`

Bulk TXT trainer. Reads the file line by line, pairs consecutive non-empty
lines, and calls `Brain.TrainBatch` for each pair. No undo point is created, so
the **Undo** button cannot reverse a batch import — this is intentional.

Progress is reported through a callback rather than printed, and the brain is
saved once at the end instead of after every line.

### `internal/config`

A `Config` struct behind a package-level `RWMutex`. Readers call `Get` for a
cheap snapshot; writers go through `Set` or `SetField`, which validate and clamp
before publishing. The engine reads the snapshot on every step, so slider
changes still take effect immediately, without a data race.

`Save`/`Load` persist the configuration as JSON next to the brain, which is why
slider positions survive a restart. Previously every tunable was a bare package
variable reset to its default on each launch.

Tunables that used to be hidden constants in the engine — thinking steps, the
vocabulary threshold for thinking, the history limits — are configured here too.

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
    think(prompt) if vocab >= MinVocabThinking
        |
        v
    for i in 0..maxLen:
        ctx := buildContext(history, ContextSize)
        sampleNextIndex(ctx, used, Temperature)
        if no context found:
            random vocabulary pick
        else:
            softmax -> Top-K (bounded heap) -> Top-P -> sample
        append word, shift window
        stop on . ? !
        |
        v
    JoinWords

### Persistence

`Save` serializes the vocabulary, weights, and velocity with `encoding/gob`,
prefixed with a `MYCOR-BRAIN\0` magic header. The stream is written to
`<path>.tmp` and then atomically renamed.

`Load` calls `decodeBrain`, which identifies the format explicitly rather than
sniffing the first byte:

1. The magic header — a gob stream written by v6.0.
2. Otherwise leading whitespace and a UTF-8 BOM are stripped; a leading `{`
   means legacy JSON, which also accepts the old dense weight arrays.
3. Anything else is decoded as gob, for streams written before the header
   existed.

The old probe tested `data[0] == '{'` directly, so a JSON file beginning with a
space or a newline was misread as gob and refused to load.

Both paths populate the same `BrainModel`. After decoding, validation runs:

1. Vocabulary is non-empty.
2. No empty strings.
3. No duplicate words.
4. `<unk>` occupies index 0 exactly once. If it appears elsewhere in the stored
   vocabulary it is moved to the front rather than prepended a second time —
   previously such a file loaded with two `<unk>` entries and a wordв†’index map
   pointing at the wrong slot for every word after it.
5. Weight indices outside the vocabulary, index 0, and NaN values are dropped.
6. Every context gets a velocity vector.

Failures leave the previous in-memory state untouched: the new model is fully
validated before anything is installed.

The brain path itself is resolved by `internal/web/paths.go` via
`os.UserConfigDir()`. The engine never decides where to store the file — it just
writes to whatever path the caller passes to `Save` and reads from whatever path
the caller passes to `Load`.

## State lifecycle

    web.Run(ctx)
        |
        v
    resolveDataPaths -> brainFile, configFile
        |
        v
    config.Load(configFile)
    brain.Load(brainFile)  (optional; starts empty if absent)
        |
        v
    saver.start(ctx)  ---> background: mark() -> debounced persist
        |
        +--> Train --> beginUndo() --> per-step locked mutation
        |                                       |
        |                                       v
        |                                  saver.mark()
        |
        +--> UndoLastTrain --> restore journalled contexts and vocabulary
        |
        +--> Reset --> brain.Reset() + config.Reset() + persist

Undo uses an undo journal: `Train` records a copy of each context vector the
first time that step touches it, plus any words the step registered. Reverting
restores those entries and drops the new words. Copying the whole model, as the
previous backup did, cost O(vocabulary * contexts) on every training call; the
journal costs O(entries actually modified).

Only the latest `Train` call can be undone. `TrainBatch` deliberately does not
create an undo point, so a batch import cannot be rolled back with one click.

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

Every exported engine method is safe for concurrent use. State lives in a
`Brain` value guarded by a `sync.RWMutex`; training additionally serialises
against itself with a second mutex so two training requests cannot interleave.

The lock is deliberately fine-grained. Training holds `trainMu` for the whole
call but takes and releases the state lock once per token step, so a long import
does not lock out chat. The idle clock has its own lock. Handlers hold no lock
across file I/O.

`TestChatRespondsDuringImport` in `internal/web` is the regression test for
this: it starts an import and asserts the server keeps answering other
requests while it runs. `TestConcurrentChatAndTraining` exercises the engine
directly; run the suite with `-race` to check it.

## What is intentionally missing

- No streaming output. Answers are returned whole.
- No file locking on the brain file. Single-user desktop assumption.
- No cross-process locking; two instances would race on the brain file.
- No authentication. The server binds to `127.0.0.1` on a random port.
- No plugin system. The engine is small enough to modify directly.
