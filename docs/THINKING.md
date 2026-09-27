# Thinking Mode and Idle Thinking

MYCOR has two related mechanisms that give the model an internal voice:
**Thinking Mode** (activates during normal replies) and **Idle Thinking**
(activates when the user is away). This document explains both.

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
    initialContext(words) -> (w1, w2)
      |
      v
    for i in 0..thinkingSteps:
        sampleNextIndex(w1, w2, used, Temperature)
        append to thoughts
        stop on . ? !
        shift window
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
thinking works even when the exact bigram context is unknown.

### Visibility

The CLI prints:

    MYCOR thoughts: ...

right before the answer. This lets you see the inner monologue and diagnose
why the model chose a particular continuation. The `/stats` command shows
whether thinking is active.

### Repetition penalty inside thinking

Thinking reuses the same `usedWords` map as the final generation pass —
no, more precisely: thinking gets its own `usedWords` map, and the final
generation pass gets a fresh one. This means a word can appear once in
the thoughts and once in the answer without penalty. That is intentional:
the thoughts are a draft, not the answer itself.

### Reset

`InitEngine` clears `lastThoughts`. `LoadBrain` also clears it, so a fresh
session never inherits the previous session's inner voice.

## Idle Thinking

### Concept

If you walk away from the console, the model does not sit silent. After a
configurable timeout it starts producing short "daydreams" — continuations
of the most recently active context — and prints them to the console.

It is a small, meditative feature that shows what the network has actually
learned. It also exposes degenerate attractor states: if the same output
appears every forty-five seconds, your training data is too narrow.

### Activation

    IdleEnabled    = true
    IdleTimeoutSec = 45

Controlled at runtime with:

- `/idle on`
- `/idle off`

The ticker fires every five seconds. On each tick the CLI checks:

1. Is idle mode enabled?
2. Has the user been silent for at least `IdleTimeoutSec` seconds?

If both are true, the engine produces a thought and the inactivity timer
is reset. So daydreams appear roughly every forty-five seconds while you
are away.

### The pipeline

    seeds := RecentContexts()
    if len(seeds) > 0:
        prompt = seeds[last]
    else:
        prompt = Vocabulary[random]
    thoughts := think(prompt)
    return JoinWords(thoughts)

The seed comes from `RecentContexts`, which is the ring buffer of the last
twenty touched context keys. The most recent one is the context the model
was in when the user last typed something. That makes the daydream a
natural continuation rather than a random emission.

If there is no history yet (fresh start, no training this session), the
engine picks a random vocabulary word as the seed.

### Why it produces nothing sometimes

`think` returns nothing if:

- The vocabulary is below `MinVocabForThinking`.
- The seed tokenizes to an empty slice.
- No context along the backoff chain has trained weights.

In these cases the CLI prints nothing and waits for the next tick. This
keeps the console clean during the early training phase.

### Interaction with the input reader

The CLI reads stdin in a background goroutine and pushes lines into a
channel. The main loop selects between:

- A line from the channel.
- A tick from the idle timer.

This means idle thoughts never block user input. As soon as you press
Enter, the select unblocks and the line is processed. The inactivity timer
is reset after every user action, so you will not see a daydream right
after typing.

## Testing

`internal/engine/engine_test.go` covers:

- `TestThinkRequiresVocabulary` — thinking is skipped below the threshold.
- `TestThinkAfterEnoughTraining` — thoughts are produced above the threshold.
- `TestThinkingResetOnInit` — `InitEngine` clears `lastThoughts`.
- `TestGenerateIdleThoughtEmpty` — idle returns empty with a small vocab.
- `TestGenerateIdleThoughtWithVocab` — idle produces something above the threshold.

## Configuration cheat sheet

| Parameter            | Default | Where              | Effect                                |
| -------------------- | ------- | ------------------ | ------------------------------------- |
| `MinVocabForThinking`| 10      | engine (const)     | Threshold for both thinking modes     |
| `thinkingSteps`      | 4       | engine (const)     | Length of the internal chain          |
| `IdleTimeoutSec`     | 45      | `internal/config`  | Silence before daydreams start        |
| `IdleEnabled`        | true    | `internal/config`  | Master switch for idle mode           |
| `Temperature`        | 0.7     | `internal/config`  | Sampling temperature for thoughts     |

Everything except the two engine constants can be changed at runtime.