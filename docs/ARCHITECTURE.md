# Architecture

MYCOR is a pure-Go, zero-dependency language model with a Markov-style
N-gram architecture. This document describes how the engine is put together
and where each responsibility lives.

## High-level picture

    +----------+     +----------+     +----------+
    |  cmd/    | --> | internal/| --> | internal/|
    |  mycor/  |     |  cli/    |     |  engine/ |
    +----------+     +----------+     +----------+
                            |                |
                            v                v
                       +----------+    +----------+
                       | internal/|    | internal/|
                       |  i18n/   |    |  importer|
                       +----------+    +----------+
                            |
                            v
                       +----------+
                       | internal/|
                       |  config/ |
                       +----------+

The CLI is the only entry point in v4.4. Every layer above sits under
`internal/` and is therefore private to the module.

## Package responsibilities

### `internal/engine`

The heart of the project. Subdivided into focused files:

| File              | Responsibility                                                |
| ----------------- | ------------------------------------------------------------- |
| `engine.go`       | Public façade, `InitEngine`, re-exports of the API             |
| `model.go`        | Global state, `BrainModel`, `RegisterWord`, backup/undo        |
| `tokenizer.go`    | `Tokenize`, `JoinWords`, emoticon matching, punctuation        |
| `backoff.go`      | `backoffChain`, `sampleNextIndex`, `softmaxBase`, weight sync  |
| `train.go`        | `Train`, `TrainBatch`, `train`, `CountParameters`              |
| `generate.go`     | `generate`, `GenerateResponse`, `GenerateIdleThought`          |
| `thinking.go`     | `think`, `Think`, `LastThoughts`, `initialContext`             |
| `history.go`      | `AppendHistory`, `SessionHistory`, `RecentContexts`, touching  |
| `persistence.go`  | `SaveBrain`, `LoadBrain`, migration, validation                |

The package keeps global mutable state. This is deliberate: the model is a
single, process-wide brain, and the CLI treats it as such. Tests call
`InitEngine()` between cases to isolate state.

### `internal/cli`

The REPL layer. Contains:

| File          | Responsibility                                       |
| ------------- | ---------------------------------------------------- |
| `repl.go`     | Main loop, select between stdin and idle ticker      |
| `commands.go` | Dispatch for `/help`, `/stats`, `/import`, `/self`…  |
| `idle.go`     | Idle timer and daydream output                       |
| `input.go`    | Non-blocking stdin reader through a channel          |

The CLI never touches weights directly. All interactions go through the
public engine API.

### `internal/i18n`

Two message packs (`en`, `ru`) behind a `Messages` struct. `Init(Lang)`
selects the active pack. `M()` returns the current pack. The CLI never
references `enMessages` or `ruMessages` directly.

### `internal/importer`

Bulk TXT trainer. Reads the file line by line, pairs consecutive non-empty
lines, and calls `engine.TrainBatch` for each pair. No backup is created,
so `/undo` cannot reverse a batch import — this is intentional.

### `internal/config`

A flat bag of tunables. No functions, no methods. The CLI writes to it
directly when the user runs `/temp`, `/lr`, `/momentum`, `/idle`.

## Data flow

### Training

    user input (prompt, target)
        |
        v
    Tokenize(prompt), Tokenize(target)
        |
        v
    RegisterWord for every new token
        |
        v
    for each context (w1, w2) -> next:
        ensureWeights(ctx)
        softmaxBase(weights, 1.0, nil, 0)
        gradient = one-hot - probs
        velocity = m*v + (1-m)*grad
        weights  = w*decay + lr*velocity
        |
        v
    return mean cross-entropy loss

### Generation

    prompt
        |
        v
    Tokenize
        |
        v
    think(prompt) if vocab >= MinVocabForThinking
        |
        v
    for i in 0..maxLen:
        backoffChain(w1, w2)
        pick first context with trained weights
        softmaxBase(weights, Temperature, used, RepetitionPenalty)
        sample index
        append word, shift window
        stop on . ? !
        |
        v
    JoinWords

### Persistence

`SaveBrain` serializes the vocabulary, weights, and velocity into a single
JSON object with a `version` field. Writes go to `<path>.tmp`, followed by
an atomic `os.Rename`. If anything fails, the `.tmp` file is removed.

`LoadBrain` validates:

1. JSON parses.
2. Vocabulary is non-empty.
3. No empty strings.
4. No duplicate words.
5. If `<unk>` is missing at index 0, it is inserted and every weight
   vector is shifted right with a zero at index 0.
6. Every weight vector is resized to `len(Vocabulary)`.

Failures leave the previous in-memory state untouched.

## State lifecycle

    InitEngine
        |
        v
    LoadBrain (optional)
        |
        v
    +--> Train / TrainBatch --> SaveBackup --> train --> mutation
    |                                                 |
    |                                                 v
    |                                             SaveBrain
    |
    +--> UndoLastTrain --> restore from backup

The backup is a shallow copy of the current vocabulary, index map, weights
and velocity. Only the latest `Train` call can be undone.

`InitEngine` clears everything except the on-disk brain:

- vocabulary -> `["<unk>"]`
- weights and velocity -> empty maps
- backup -> nil
- recent contexts -> empty
- session history -> empty
- last thoughts -> empty

## Numerical notes

- Softmax is computed with a max-shift for numerical stability.
- Repetition penalty divides positive logits by the penalty factor and
  multiplies negative logits, so the direction of the penalty is always
  correct regardless of sign.
- Momentum is clamped to `[0, 0.99]` before use.
- Weight decay is applied multiplicatively: `w = w * (1 - lr * decay)`.

## Concurrency

The engine is not thread-safe. The CLI reads stdin through a single
goroutine and feeds it into the main loop via a channel, but all engine
calls happen on the main goroutine. The idle ticker is a `time.Ticker`
selected against the same channel, so engine state is never touched from
two goroutines at once.

## What is intentionally missing

- No context cancellation. Training and generation are fast enough.
- No streaming output. Answers are printed whole.
- No file locking on `history.json`. Single-user desktop assumption.
- No plugin system. The engine is small enough to modify directly.