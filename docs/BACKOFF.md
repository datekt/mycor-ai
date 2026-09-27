# Dynamic Backoff

Dynamic Backoff is the mechanism that prevents the model from getting stuck
when it encounters a context it has never seen during training.

## Why it exists

MYCOR predicts the next token from a two-word context:

    (w1, w2) -> next

This is a trigram model. If the exact pair `(w1, w2)` was never trained,
a naive implementation returns `<unk>` or nothing at all. In practice this
happens constantly:

- The user types a word the model has never seen.
- Two known words appear together for the first time.
- The vocabulary grew after training but some contexts still refer to
  old indices.

Dynamic Backoff solves this by trying progressively shorter contexts until
something useful is found.

## The ladder

For every prediction, the engine builds an ordered chain of context keys:

1. `(w1, w2)` — the full trigram context.
2. `(<unk>, w2)` — drop the first word.
3. `(w1, <unk>)` — drop the second word.
4. `(<unk>, <unk>)` — the unigram context.

The chain is deduplicated: if `w1` is already `<unk>`, step 1 and step 2
are the same key, and step 2 is skipped. Same for `w2` and step 3.

The first key that exists in the weights map with a non-empty vector wins.
If none exist, the caller decides what to do:

- In `generate`, the engine falls back to a random vocabulary pick.
- In `think`, the engine returns no thoughts and stops early.

## Example

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

## Where it lives

`internal/engine/backoff.go`:

    func backoffChain(w1, w2 string) []string

`sampleNextIndex` consumes the chain:

    chain := backoffChain(w1, w2)
    for _, key := range chain {
        if v, ok := Weights[key]; ok && len(v) > 0 {
            logits = v
            break
        }
    }

The chain is rebuilt on every step, so it always reflects the current
vocabulary and weight state.

## Interaction with `<unk>`

`<unk>` is not a real token. It never appears in generated output — the
`generate` loop explicitly skips it. Its only role is to serve as a
wildcard context that the model can learn from:

- During training, when the prompt has fewer than two words,
  `initialContext` substitutes `<unk>` for the missing slot.
- During training with a longer prompt, the first bigram is taught as
  `("<unk>", first_word) -> second_word`, so the model learns to start
  sentences from an unknown prefix.
- During backoff, `<unk>` fills the slot of any word the model has never
  seen.

## Testing

`internal/engine/engine_test.go` covers:

- `TestBackoffChainDeduplicates` — no duplicate keys when inputs are `<unk>`.
- `TestDynamicBackoffUsesUnkPrefix` — unknown first word falls back.
- `TestDynamicBackoffUsesUnkSuffix` — unknown second word falls back.
- `TestDynamicBackoffFallsBackToUnkUnk` — no context at all returns false.

## Tuning

There is nothing to tune. The ladder is fixed at four levels. If you want
a different behaviour, edit `backoffChain` — it is fifteen lines.