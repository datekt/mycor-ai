# Dynamic Backoff

Dynamic Backoff is the mechanism that prevents the model from getting stuck
when it encounters a context it has never seen during training.

> **v5.0 update:** the context window is no longer fixed at two words. It is
> configurable at runtime through `config.ContextSize` (1 to 5). The backoff
> ladder is now built adaptively for any window length. This document
> describes the general mechanism; the N=2 case is a special case that
> matches the behaviour of v4.x.

## Why it exists

MYCOR predicts the next token from a fixed-width window of previous tokens:

    (w1, w2, ..., wN) -> next

With the default `ContextSize = 2` this is a trigram model. If the exact
context `(w1, w2, ..., wN)` was never trained, a naive implementation
returns `<unk>` or nothing at all. In practice this happens constantly:

- The user types a word the model has never seen.
- Two known words appear together for the first time.
- The vocabulary grew after training but some contexts still refer to
  old indices.
- The window contains more words than the model has ever seen together.

Dynamic Backoff solves this by trying progressively shorter contexts until
something useful is found.

## The ladder

For a window of length N, the engine builds an ordered chain of context
keys in this order:

1. The full N-gram `(w1, w2, ..., wN)`.
2. For each level `k` from `N-1` down to `1`:
   - `keepLast k`: only the last `k` words survive; all earlier slots
     become `<unk>`.
   - `keepFirst k`: only the first `k` words survive; all later slots
     become `<unk>`.
3. The fully-masked context `(<unk>, ..., <unk>)`.

The chain is deduplicated: if two entries produce the same key, the second
one is dropped. This matters most when the input already contains `<unk>`,
or when N is small.

The first key that exists in the weights map with a non-empty vector wins.
If none exist, the caller decides what to do:

- In `generate`, the engine falls back to a random vocabulary pick.
- In `think`, the engine returns no thoughts and stops early.

### Chain size

| N | Chain length |
| - | ------------ |
| 1 | 2            |
| 2 | 4            |
| 3 | 6            |
| 4 | 8            |
| 5 | 10           |

The formula is `2N` for `N >= 2`, because the full context and the fully
masked context bookend `2 * (N-1)` intermediate masks. For `N = 1` there
are no intermediate levels, so the chain has exactly two entries: the
word itself and `<unk>` (unless the word already is `<unk>`, in which
case the chain deduplicates down to one entry).

## Example for N = 2

Suppose the training data was:

    кошка села на окно
    кошка села на стол

The engine learns:

    ("кошка", "села") -> "на"
    ("села", "на")    -> "окно" | "стол"
    ("на", "окно")    -> "."

Now the user types:

    собака села на

The word `собака` is unknown. Backoff chain:

1. `("собака", "села")` — not in weights.
2. `("<unk>", "села")` — not in weights (we only saw `"кошка"`).
3. `("собака", "<unk>")` — not in weights.
4. `("<unk>", "<unk>")` — not in weights either (the model has not been
   trained with unknown prefixes).

The engine falls through to the random fallback in `generate`, so the
answer is a random word from the vocabulary.

Now suppose the user types:

    кошка прыгнула на

Backoff chain:

1. `("кошка", "прыгнула")` — not in weights.
2. `("<unk>", "прыгнула")` — not in weights.
3. `("кошка", "<unk>")` — not in weights.
4. `("<unk>", "<unk>")` — not in weights.

Still nothing. Backoff only helps when at least one of the two words has
been seen with unknown neighbours. To make that happen, train the model
on sentences that share words but differ in neighbours:

    кошка села на окно
    собака села на стол
    кошка прыгнула на стол

Now `("кошка", "<unk>")` exists from the first and third sentences, and
`("<unk>", "села")` exists from the first and second. Backoff starts
paying off.

## Example for N = 3

With `ContextSize = 3` and a prompt ending in three known words

    кошка села на

the chain has six entries:

1. `("кошка", "села", "на")` — full trigram window.
2. `("<unk>", "села", "на")` — keepLast 2.
3. `("кошка", "села", "<unk>")` — keepFirst 2.
4. `("<unk>", "<unk>", "на")` — keepLast 1.
5. `("кошка", "<unk>", "<unk>")` — keepFirst 1.
6. `("<unk>", "<unk>", "<unk>")` — the unigram wildcard.

The first entry that has trained weights wins. Notice that the intermediate
levels are *not* the same as the N=2 ladder: `(w1, w2, <unk>)` is a new
context that only exists when training also used a window of three. This
is why changing `ContextSize` on a trained brain changes the answers — the
model may still fall back to shorter contexts, but it will not have direct
weights for the intermediate levels.

## Where it lives

`internal/engine/backoff.go`:

    func backoffChain(words []string) []string

`sampleNextIndex` consumes the chain:

    chain := backoffChain(resolveContext(rawContext))
    for _, key := range chain {
        if v, ok := weights[key]; ok && len(v) > 0 {
            logits = v
            break
        }
    }

The chain is rebuilt on every step, so it always reflects the current
vocabulary, weight state, and `ContextSize`.

## Interaction with `<unk>`

`<unk>` is not a real token. It never appears in generated output — the
`generate` loop explicitly skips it. Its only role is to serve as a
wildcard context that the model can learn from:

- During training, `buildContext` pads the left edge of the window with
  `<unk>` when there are not enough previous words yet.
- During training, `trainStep` is called once per entry in the backoff
  chain, so the model learns the full context *and* every masked variant
  that shares the same target.
- During generation, `<unk>` fills the slot of any word the model has
  never seen, or any slot that has been dropped by a backoff level.

## Testing

`internal/engine/engine_test.go` covers:

- `TestBackoffChainDeduplicates` — no duplicate keys when inputs are `<unk>`.
- `TestBackoffChainFullLadder` — N=2 produces the four expected entries
  in the expected order.
- `TestBackoffChainThreeGram` — N=3 produces six entries with the full
  context first.
- `TestBackoffChainUnigram` — N=1 produces `[word, <unk>]`.
- `TestBackoffChainUnigramUnk` — N=1 with `<unk>` produces exactly one entry.
- `TestBackoffChainFiveGram` — N=5 produces ten entries, ending in five
  `<unk>` tokens.
- `TestDynamicBackoffUsesUnkPrefix` — unknown first word falls back.
- `TestDynamicBackoffUsesUnkSuffix` — unknown second word falls back.
- `TestDynamicBackoffFallsBackToUnkUnk` — no context at all still reaches
  the unigram wildcard.
- `TestTrainTeachesBackoffContexts` — every entry in the chain gets a
  weight vector after one training step.

## Tuning

There is only one knob: `config.ContextSize`, adjustable at runtime from
the sidebar's **Context Size** slider (1 to 5). Larger windows produce more
specific predictions when the training data is rich, and fall back more
often when it is not. Smaller windows generalize faster but produce
shallower continuations.

Changing `ContextSize` on an existing brain does not retrain it. The old
weights remain valid for their original context keys, and the model
simply uses whatever entries in the new chain happen to exist. If you want
to fully retrain for a new window size, use **Reset** and start over, or
**Import** the original corpus again.