<p align="center">
  <img src="assets/mycor-banner.jpeg" alt="MYCOR AI Banner" width="100%">
</p>

# 🧠 MYCOR (Mycor) — v4.5

> **My Core. Zero Dependencies. Pure Go. Dynamic Context. Persistent Memory. Thinking. Daydreaming.**

**MYCOR** is a completely "empty" local language model written in pure **Go 1.27.1**, built from scratch without any third-party frameworks (PyTorch, TensorFlow), Python, or heavy C libraries (CGO).

In an era of terabyte-scale corporate black boxes, **MYCOR** returns control to the developer. It is a digital sandbox and canvas whose character you shape entirely yourself.

---

## 🎯 Project Philosophy

On first launch, MYCOR is a "blank slate" (Tabula Rasa). To any of your requests, the model will output chaotic nonsense. It has no pre-loaded knowledge, censorship, or bias.

**It learns only from you.** By chatting with it in the console and correcting its answers, you literally build its "brain" (a dynamic weight matrix of N-grams) from scratch. Grow your own, albeit simple, but truly *personal* neural network right on your local hardware.

---

## 🆕 What's new in v4.5

v4.5 is a hardening release. No new features — the same engine, the same commands, the same brain format. Everything below is a bug fix, an encapsulation improvement, or a test-coverage upgrade found during a static and architectural review of v4.4.

### Correctness

- 🧮 **`softmaxBase` no longer returns garbage on degenerate input.** If the logits collapse to `-Inf`, `NaN`, or sum to exactly zero, the function now returns a uniform distribution instead of an unnormalized slice. Generation and training stay numerically safe.
- 🎲 **Idle thoughts never seed on `<unk>`.** `GenerateIdleThought` used to call `rand.Intn(len(Vocabulary))` and could pick index 0, feeding the technical `<unk>` token into the thought chain. The sampling range is now `[1, len(Vocabulary))`, and there is a regression test that runs fifty idle cycles asserting the token never appears.
- ⏱️ **Idle daydreams fire closer to the configured timeout.** The ticker interval was reduced from 5 s to 1 s. With a 45‑second `IdleTimeoutSec`, the first dream now appears at ~46 s instead of up to ~49 s.
- 🔤 **Tokenizer is Unicode-aware.** Beyond ASCII `. , ? ! : ;`, the tokenizer now splits on the em dash `—`, en dash `–`, ellipsis `…`, and Russian/typographic quotes `« » “ ” ‘ ’`. Words like `думаю…` no longer hide punctuation inside a single token.
- 🌐 **`LangSelect` and `LangChoice` are actually rendered.** The startup language prompt now pulls both strings from the active i18n pack instead of hardcoding English text.

### Robustness

- 📥 **`/import` handles arbitrarily long lines.** The importer switched from `bufio.Scanner` (which silently fails with `ErrTooLong` past its 1 MB buffer) to `bufio.Reader.ReadString`, so multi‑megabyte lines without spaces now train correctly instead of being dropped.
- 🧹 **`/reset` restores configuration defaults.** Previously it wiped the brain but left the user's `/temp`, `/lr`, `/momentum`, and `/idle` overrides in memory. `config.Reset()` now returns every tunable to its `Default*` constant.
- 🛑 **`exit` and slash commands are intercepted during the teacher prompt.** After the model answers, `chatTurn` waits for a correction. If the user typed `exit` or `/stats` instead, those inputs used to be silently swallowed as a training target. They are now dispatched the same way as at the main prompt.

### Architecture and encapsulation

- 🔒 **`weights` and `velocity` are unexported.** They used to be package-level `Weights`/`Velocity` maps visible to the entire module, which let the CLI read them directly in `/stats`. The engine now exposes a single read accessor, `ContextWeightSize(ctxKey) (int, bool)`, and the CLI goes through it. The internal maps are only reachable from inside `internal/engine`.
- 🎛️ **`config` package gains `Reset()` and `Default*` constants.** The engine, CLI, and `/reset` command all share one source of truth for defaults instead of duplicating literals.
- 📖 **Single shared `bufio.Reader` across the CLI.** `selectLanguage` and `newInputReader` used to create two separate readers over `os.Stdin`, which could lose buffered bytes typed during the language prompt. `Run` now creates one reader and hands it to both.

### Tooling and tests

- ⬆️ **`go.mod` says `go 1.27.1`.** It used to say `1.24`, contradicting the README. The declared toolchain and the documented one are now in sync.
- 🧹 **`gosimple` removed from `.golangci.yml`.** The linter was merged into `staticcheck` and its separate entry produced a deprecation warning on every run.
- 🧪 **All four `testdata/` fixtures are now used.** `brain_corrupt.json`, `brain_duplicate.json`, `brain_v3.json`, and `brain_v4.json` are read by dedicated engine tests that verify corruption rejection, duplicate detection, v3→v4 migration with `<unk>` insertion, and clean v4 round-trip loading. `testdata/sample.txt` is exercised by an importer test.
- ✅ **Two assertion-free tests were rewritten.** `TestGenerateResponseNoTrain` and `TestGenerateIdleThoughtWithVocab` previously computed a value and discarded it with `_ =`. They now assert non-empty output, absence of `<unk>`, and reset behavior.
- 🧪 **New regression tests.** `TestSoftmaxBaseUniformOnDegenerateInput`, `TestGenerateIdleThoughtNeverSeedsUnk`, `TestContextWeightSize`, `TestImportTxtFileLongLine`, `TestImportSampleFixture`, and `TestReset` in the config package.
- 📌 **`modelVersion` stays at 4.** The brain file format is unchanged; v4.4 brains load into v4.5 without migration.

---

## ✨ Key Features of v4.5

- 📦 **Zero Dependencies:** Uses exclusively the Go standard library.
- 🚀 **N-Gram Context Window:** Sliding context window of two words for coherent speech.
- 🪜 **Dynamic Backoff:** Trigrams → Bigrams → Unigrams. No more empty answers on rare phrases.
- 🎭 **Emoticon-aware tokenizer:** Smileys and text-emoticons are single tokens.
- 🧠 **Dynamic vector vocabulary:** New words extend the weight matrix on the fly.
- 💭 **Thinking Mode:** A latent chain of thoughts before the answer (activates at ≥10 words).
- 💤 **Idle Thinking:** The AI daydreams on its own when you are away.
- 🌐 **Bilingual UI:** English / Russian, switchable at runtime.
- 💾 **Persistent memory:** The trained brain is saved to `history.json` and restored on restart. Session history is reset each launch — it lives only in RAM.
- 📊 **Engineering console:** Commands for monitoring loss, temperature, learning rate, and momentum.
- 📖 **Batch import:** Instant auto-training from a text file, now with unlimited line length.
- 🌀 **Momentum Optimizer:** Weight updates with inertia (EMA gradient).
- 🧮 **L2 Weight Decay:** Light regularization against weight blow-up.
- 🛡️ **Atomic save:** Brain is written to a temp file and then renamed.
- 🔒 **Strict validation:** Duplicate words, weight length, and JSON integrity are checked on load.
- 🧪 **Test coverage:** Full suite of unit tests for the engine, importer, thinking mode, dynamic backoff, idle thinking, and configuration reset.

## 🚀 Quick Start

1. Clone the repository and enter the project folder:

       git clone <url>
       cd mycor

2. Run the model (the trailing dot is required to build the modules):

       go run .

3. **Pick a language** (English or Русский) at the startup prompt and press Enter.

4. Start chatting! Type a phrase, look at the answer, then type the correct answer to teach it. Press Enter with no text to skip the training step.

5. Once the vocabulary reaches 10+ words, a `MYCOR thoughts:` line will appear before each answer — this is the network's inner monologue before the final reply.

6. Step away for 45 seconds — and the AI will start whispering its daydreams into the console.

---

## 🌐 Language Switching

At startup:

    Select language / Выберите язык:
      1. English
      2. Русский
    Choice [1]:

Press `1` (or Enter) for English, `2` for Russian.

During a session:

- `/lang en` — switch to English.
- `/lang ru` — переключиться на русский.

All subsequent messages, help, and history role names update immediately.

---

## 🕹️ Control Panel Commands

- `/help` — Show the full command list.
- `/stats` — Current synapse count, thinking-mode status, and recent active contexts.
- `/history` — View the dialogue history of the current session.
- `/undo` — Undo the last lesson.
- `/reset` — Completely erase `history.json`, restart the brain, and restore configuration defaults.
- `/temp [0.1 - 1.5]` — Creativity (Softmax temperature).
- `/lr [0.01 - 1.0]` — Learning rate.
- `/momentum [0.0 - 0.99]` — Gradient inertia (default 0.9).
- `/import [path]` — Batch training from a text file.
- `/self` — Autonomous AI talking to its mirror (with thoughts shown).
- `/lang [en|ru]` — Switch the interface language.
- `/idle [on|off]` — Enable / disable idle thinking (daydreams).
- `exit` — Safe exit with weights saved.

`exit` and slash commands are also accepted while the program is waiting for a teacher correction, so you never lose an input to the training loop by accident.

---

## 🧠 Memory: What Is Saved and What Is Reset

**Saved to `history.json` and survives restart:**
- Vocabulary (`vocabulary`).
- Weight matrix (`weights`).
- Optimizer state (`velocity`, momentum).

**Reset on every launch:**
- Current session dialogue history.
- Recent active contexts list.
- Last "thoughts" of the network (`lastThoughts`).
- Backup copy for `/undo`.

**Reset only by `/reset`:**
- Runtime configuration (`/temp`, `/lr`, `/momentum`, `/idle`) — back to `Default*` values.
- The `history.json` file on disk.
- The in-memory brain.

This separation lets you accumulate knowledge between sessions without dragging random garbage from previous conversations.

---

## 💭 How Thinking Mode Works

1. **Activation threshold:** `MinVocabForThinking = 10`. While the vocabulary has fewer than 10 unique words, the model answers instantly.
2. **Internal generation:** When active, the engine performs `thinkingSteps = 4` sampling steps from the same weight matrix but does not show them to the user during generation.
3. **Answer context:** The resulting "thought" becomes a prefix to the prompt, and the final answer is generated from the extended context. In other words, the answer is a continuation of the thought.
4. **Transparency:** The `MYCOR thoughts:` line is printed after the answer so you can observe the inner logic. `/stats` shows the current mode status.

This approach produces a "two-stage" output: the model first plays out several alternative continuations, then builds its answer on top of them.

---

## 🪜 How Dynamic Backoff Works

For each prediction, the engine tries the following contexts in order:

1. The full trigram context `(w1, w2)`.
2. Bigram with unknown first word: `(<unk>, w2)`.
3. Bigram with unknown second word: `(w1, <unk>)`.
4. The unigram context `(<unk>, <unk>)`.

The first context with trained weights wins. If none of them exist, the engine falls back to a random vocabulary pick (in `generate`) or returns nothing (in `think`). This is why the model no longer produces `<unk>` on rare phrases — it smoothly descends the context ladder, keeping the speech coherent.

The chain is rebuilt on every step from the current vocabulary and weight state, and it is deduplicated: if `w1` is already `<unk>`, the `<unk> w2` level is skipped.

---

## 🎭 Special Tokens for Punctuation and Emoticons

The tokenizer recognizes whole emoticons as single tokens:

`:-)`, `:)`, `;-)`, `;)`, `:-D`, `:D`, `:-P`, `:P`, `:3`, `:/`, `:|`, `:'(`, `:'-(`, `:*`, `^^`, `<3`, `o_o`, `-_-`

Punctuation is split into separate tokens, both ASCII (`. , ? ! : ;`) and Unicode typography (`— – … « » “ ” ‘ ’`). This lets the model learn to attach punctuation to words correctly, and prevents Russian em dashes or ellipses from being glued onto adjacent tokens.

This means your trained network can pick up your habit of ending messages with `:)` or starting them with `^^`, and reproduce it in its own output — a small but important part of what "character" means.

---

## 💤 How Idle Thinking Works

Idle thinking is a background daydream mode. It works like this:

1. **Timer:** After each of your messages, an inactivity timer is reset. If no key is pressed for `IdleTimeoutSec` seconds (default 45), the idle ticker fires.
2. **Seed:** The engine takes the most recently touched context (from `RecentContexts`). If none exists yet, it picks a random word from `Vocabulary[1:]` — index 0 is skipped so `<unk>` is never used as a seed.
3. **Daydream:** It runs the same `think()` pipeline used for thinking mode and prints the result as `Idle thought: ...`.
4. **Timer reset:** After printing, the timer resets, so the AI keeps producing new daydreams every 45 seconds until you return and type something.

Toggle at runtime:

- `/idle on` — enable daydreams.
- `/idle off` — disable.

Idle thinking is a cute way to observe what the network has actually learned — and to spot strange attractor states in your training data.

---

## 🛠️ How It Works Under the Hood

The mathematics of MYCOR take a little over 350 lines of code in the `engine/` folder. The architecture is built on Markov transitions of context chains `[word1 + word2] -> word3`.

Training uses:
- **Softmax with numerical stabilization** (shift by max, uniform fallback on degenerate input).
- **Cross-entropy** as the loss function.
- **SGD with inertia (EMA gradient):** `v = m*v + (1-m)*grad; w = w*decay + lr*v`.
- **Weight Decay:** `WeightDecay` parameter in the config.

Generation uses:
- **Softmax temperature** to control creativity.
- **Repetition Penalty** to fight looping.
- **Thinking Mode** for inner monologue before the answer.
- **Dynamic Backoff** for context descent.
- **Idle Thinking** for autonomous daydreams.

Since the local `history.json` file is in `.gitignore`, the repository is published **absolutely clean**. Each user creates their own unique digital mind.

### 💡 Lifehack: Training via a Large AI (DeepSeek / Claude)

You can generate a character in seconds. Run the model once so a base `history.json` file is created. Pass this file to DeepSeek and ask:

> *"Here is the weight file `history.json` of a Go context model MYCOR. Change the numbers inside this matrix so that the model learns to answer in the style of [character/style]. Return the modified JSON in full."*

Replace your file with the returned text and run the project.

---

## 🗺️ Roadmap

- [x] v1.0 — Basic character-level model.
- [x] v2.0 — Switch to word-level tokens.
- [x] v3.0 — Sliding N-gram context window.
- [x] v4.0 — Command processor, loss calculation, and batch `/import`.
- [x] v4.1 — Engine tests, fixes for `undo`, `softmax`, `import`, `history.json` validation.
- [x] v4.2 — Momentum optimizer, weight decay, atomic save, persistent memory and session history, strict model validation, updated command panel.
- [x] v4.3 — Thinking Mode (≥10 words), transparent thought output, fixes for `RegisterWord` and `reset`, filtered `<unk>` from answers, Go 1.27.1.
- [x] v4.4 — Special tokens for punctuation and emoticons, Dynamic Backoff, Idle Thinking, bilingual interface.
- [x] v4.5 — Hardening release: encapsulation of `weights`/`velocity`, Unicode tokenizer, safe `softmaxBase`, `/reset` config reset, unlimited-line `/import`, `exit`/slash interception during the teacher prompt, shared stdin reader, all `testdata/` fixtures under test, Go version synced to `go.mod`, `gosimple` removed.
- [ ] v5.0 (In development) — Local Web UI:
  - Interactive chat in the browser via Go HTML templates.
  - Visual sliders for Temperature and Learning Rate.
  - Drag-and-Drop form for uploading text books.
  - Real-time Loss charts.

## 📄 License

The project is distributed under the free **MIT** license. You are free to modify, use, and develop this code.