# Thinking Mode and Idle Thinking

MYCOR has two related mechanisms that give the model an internal voice:
**Thinking Mode** (activates during normal replies) and **Idle Thinking**
(activates when the user is away). This document explains both.

> **v5.0 update:** the context window is now adaptive. Both thinking modes
> use `buildContext(history, config.ContextSize)` instead of the old
> `initialContext` helper. The idle ticker described below is a CLI detail
> from v4.x; in v5.0 the browser polls `/api/idle` instead. Everything else
> about the thinking pipeline is unchanged.

## Thinking Mode

### Concept

Before producing a final answer, the model generates a short chain of
tokens using the same weights. That chain is not shown as the answer —
it is prepended to the prompt, and the answer is generated from the
extended context. In effect, the answer is a continuation of the thought.

This mimics, in a very small way, the "chain-of-thought" idea: the model
plays out a few alternative continuations internally, then commits to one.

### Activation

    MinVocabForThinking = 10

While the vocabulary has fewer than ten unique tokens, thinking is disabled.
The reason is that with a tiny vocabulary the sampler produces the same
two or three words over and over, and thinking just wastes tokens.

Once the vocabulary reaches ten words, thinking kicks in automatically.

### The pipeline

    prompt
      |
      v
    Tokenize
      |
      v
    history := tokens(prompt)
      |
      v
    for i in 0..thinkingSteps:
        ctx := buildContext(history, ContextSize)
        sampleNextIndex(ctx, used, Temperature)
        append to thoughts
        history = append(history, nextWord)
        stop on . ? !
      |
      v
    lastThoughts = JoinWords(thoughts)
      |
      v
    effectivePrompt = prompt + " " + lastThoughts
      |
      v
    generate(effectivePrompt, maxLen)

`thinkingSteps` is currently 4. Every step uses Dynamic Backoff, so
thinking works even when the exact window context is unknown.

### Visibility

The browser UI prints:

    MYCOR thoughts: ...

right before the answer. This lets you see the inner monologue and diagnose
why the model chose a particular continuation. The sidebar's **Thinking**
row shows whether thinking is active.

### Repetition penalty inside thinking

Thinking gets its own `usedWords` map, and the final generation pass gets
a fresh one. This means a word can appear once in the thoughts and once in
the answer without penalty. That is intentional: the thoughts are a draft,
not the answer itself.

### Reset

`InitEngine` clears `lastThoughts`. `LoadBrain` also clears it, so a fresh
session never inherits the previous session's inner voice.

## Idle Thinking

### Concept

When the user is away, the model does not sit silent. After a configurable
timeout it starts producing short "daydreams" — continuations of the most
recently active context — and appends them to the chat.

It is a small, meditative feature that shows what the network has actually
learned. It also exposes degenerate attractor states: if the same output
appears every forty-five seconds, your training data is too narrow.

### Activation

    IdleEnabled    = true
    IdleTimeoutSec = 45

Controlled at runtime through the **Idle** button in the sidebar, or via
`POST /api/config {"idleEnabled": true}`.

### How it is triggered in v5.0

The browser polls `GET /api/idle` every two seconds. On each request the
server checks:

1. Is `config.IdleEnabled` true?
2. Has the time since the last user request exceeded `IdleTimeoutSec`?

If both are true, the server generates a thought, resets the internal
activity timestamp, and returns the string. The browser appends it to the
chat as an "Idle thought". If either check fails, the response is empty
and nothing is displayed.

This keeps the console clean during the early training phase and prevents
the model from emitting a daydream immediately after a message.

### The pipeline

    seeds := RecentContexts()
    if len(seeds) > 0:
        prompt = seeds[last]
    else:
        prompt = Vocabulary[random from 1:]
    thoughts := think(prompt)
    return JoinWords(thoughts)

The seed comes from `RecentContexts`, which is a ring buffer of the last
twenty touched context keys. The most recent one is the context the model
was in when the user last typed something. That makes the daydream a
natural continuation rather than a random emission.

If there is no history yet (fresh start, no training this session), the
engine picks a random vocabulary word as the seed. Index 0 (`<unk>`) is
explicitly excluded.

### Why it produces nothing sometimes

`think` returns nothing if:

- The vocabulary is below `MinVocabForThinking`.
- The seed tokenizes to an empty slice.
- No context along the backoff chain has trained weights.

In these cases the API returns an empty thought and the browser waits for
the next poll.

### Interaction with user input

In the v5.0 browser client, `/api/idle` is polled on a timer while the
user can type at any moment. The activity timestamp is updated by
`/api/chat` and `/api/train`, so idle thoughts never interrupt an active
conversation. If a daydream fires while the user is mid-typing, it is
still appended above the input field and does not move focus.

## Testing

`internal/engine/engine_test.go` covers:

- `TestThinkRequiresVocabulary` — thinking is skipped below the threshold.
- `TestThinkAfterEnoughTraining` — thoughts are produced above the threshold.
- `TestThinkingResetOnInit` — `InitEngine` clears `lastThoughts`.
- `TestGenerateIdleThoughtEmpty` — idle returns empty with a small vocab.
- `TestGenerateIdleThoughtWithVocab` — idle produces something above the threshold.
- `TestGenerateIdleThoughtNeverSeedsUnk` — fifty idle cycles assert `<unk>`
  never appears in the output.

## Configuration cheat sheet

| Parameter             | Default | Where             | Effect                              |
| --------------------- | ------- | ----------------- | ----------------------------------- |
| `MinVocabForThinking` | 10      | engine (const)    | Threshold for both thinking modes   |
| `thinkingSteps`       | 4       | engine (const)    | Length of the internal chain        |
| `IdleTimeoutSec`      | 45      | `internal/config` | Silence before daydreams start      |
| `IdleEnabled`         | true    | `internal/config` | Master switch for idle mode         |
| `Temperature`         | 0.7     | `internal/config` | Sampling temperature for thoughts   |
| `ContextSize`         | 2       | `internal/config` | Window length used by both modes    |

Everything except the two engine constants can be changed at runtime from
the sidebar.