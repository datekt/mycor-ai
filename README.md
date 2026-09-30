<p align="center">
  <img src="assets/mycor-banner.jpeg" alt="MYCOR AI Banner" width="100%">
</p>

# 🧠 MYCOR — v5.0

> **My Core. Zero Dependencies. Pure Go. Web UI in one .exe. Top-K / Top-P. Adaptive N-gram. Binary weights.**

**MYCOR** is a completely "empty" local language model written in pure **Go 1.27.1**, built from scratch without any third-party frameworks, Python, or CGO. Starting with v5.0 it ships as a single self-contained executable that opens a local web interface in your default browser.

In an era of terabyte-scale corporate black boxes, **MYCOR** returns control to the developer. It is a digital sandbox and canvas whose character you shape entirely yourself.

---

## 🎯 Project Philosophy

On first launch, MYCOR is a "blank slate" (Tabula Rasa). To any of your requests, the model will output chaotic nonsense. It has no pre-loaded knowledge, censorship, or bias.

**It learns only from you.** By chatting with it and correcting its answers, you literally build its "brain" (a dynamic weight matrix of N-grams) from scratch. Grow your own, albeit simple, but truly *personal* neural network right on your local hardware.

---

## 🆕 What's new in v5.0

v5.0 turns MYCOR from a terminal REPL into a self-contained desktop application with a browser UI, and upgrades the sampler and the persistence layer.

### Desktop application

- 🖥️ **Single `.exe`, no Wails, no Electron.** `cmd/mycor/main.go` calls `internal/web.Run()`, which binds `net/http` to `127.0.0.1` on an OS-assigned port and opens the user's default browser via `rundll32` / `open` / `xdg-open`.
- 📦 **Embedded UI.** The entire single-page frontend lives in `internal/web/static/index.html` and is compiled into the binary with `//go:embed`.
- 🎛️ **Live sliders.** Temperature, Top-K, Top-P, Learning Rate, Momentum, and Context Size are adjustable from the sidebar and applied to the engine on the next training or generation step.
- 📊 **Stats panel** with vocabulary size, parameter count, thinking-mode indicator, and recent active contexts.
- 🗂️ **Clean executable folder.** The trained brain is stored in the OS user configuration directory (see below), not next to the `.exe`.

### Smarter sampling

- 🔝 **Top-K filtering.** Keep only the K most likely tokens before sampling.
- 🌡️ **Top-P (nucleus) sampling.** Keep the smallest set of tokens whose cumulative probability exceeds P.
- 🔢 **Repetition penalty** is applied before Top-K/Top-P, so the filter operates on the already-adjusted distribution.

### Adaptive N-gram context

- 📐 **Context Size is now runtime-configurable** (1 to 5 words). `backoffChain` was rewritten for arbitrary N: for a window of length N it emits the full context, then every `keepLast`/`keepFirst` mask at level N-1 down to 1, then the fully-masked `<unk>` context.
- 🔁 **Backward compatible.** For N=2 the chain is identical to v4.x — existing trained brains behave the same.

### Faster persistence

- ⚡ **`encoding/gob` weights.** `SaveBrain` writes a gob stream to `<path>.tmp` and atomically renames it. Files are smaller and load noticeably faster than JSON.
- 🔄 **Transparent fallback.** `decodeBrain` inspects the first byte: `{` means old JSON, anything else is treated as gob. All v3/v4 brains load without migration.
- 🧪 **`modelVersion` bumped to 5**, but the brain format is unchanged on disk apart from the encoding.

### Kept from v4.5

- Hardened `softmaxBase` with uniform fallback on degenerate input.
- Unicode-aware tokenizer (em dash, en dash, ellipsis, typographic quotes).
- Idle thoughts never seed on `<unk>`.
- `ContextWeightSize` as the only read accessor for the weight map.
- `/reset` restores configuration defaults.
- Arbitrarily long lines in the TXT importer.

---

## ✨ Key Features

- 📦 **Zero Dependencies:** Uses exclusively the Go standard library.
- 🖥️ **Single binary:** `go build` produces one `.exe` containing the engine and the UI.
- 🚀 **Adaptive N-gram context:** Window of 1 to 5 previous words.
- 🪜 **Dynamic Backoff:** Full context → masked variants → `<unk>` unigram.
- 🔝 **Top-K and Top-P sampling** on top of temperature and repetition penalty.
- 🎭 **Emoticon-aware tokenizer.**
- 🧠 **Dynamic vector vocabulary:** New words extend the weight matrix on the fly.
- 💭 **Thinking Mode:** A latent chain of thoughts before the answer (activates at ≥10 words).
- 💤 **Idle Thinking:** The AI daydreams on its own when you are away.
- 🌐 **Bilingual UI:** English / Russian, switchable at runtime.
- 💾 **Persistent memory:** Binary brain file restored on restart.
- 📖 **Batch import:** Auto-training from a text file with unlimited line length.
- 🌀 **Momentum optimizer** with L2 weight decay.
- 🛡️ **Atomic save:** Temp file + rename.
- 🔒 **Strict validation:** Duplicate words, weight length, and file integrity checked on load.
- 🧪 **Test coverage:** Engine, importer, thinking mode, dynamic backoff, idle thinking, configuration reset, gob round-trip, JSON fallback.

---

## 🚀 Quick Start

### Option A — Download the release

Grab the `.exe` for your platform from GitHub Releases, put it in any folder, run it. The browser opens at `http://127.0.0.1:XXXXX/`.

### Option B — Build from source

    git clone https://github.com/datekt/MYCOR.git
    cd MYCOR

On Windows:

    .\scripts\build.ps1

On Linux/macOS:

    ./scripts/build.sh

Or directly:

    go build -trimpath -ldflags="-s -w" -o bin/mycor.exe ./cmd/mycor

### First conversation

1. The browser tab opens. The sidebar shows `Vocabulary: 1`, `Thinking: OFF`.
2. Type a phrase in the input box and press Enter. The model will answer with something chaotic — it is empty.
3. In the "Teach correction" field, type the correct answer and press **Train**. Loss is printed in the chat log.
4. Repeat until the vocabulary reaches 10 words. The sidebar's `Thinking` row flips to `ON`.
5. Step away for 45 seconds. Idle thoughts start appearing in the chat.

---

## 🕹️ Interface

### Sidebar

| Control | Range | Effect |
| --- | --- | --- |
| Temperature | 0.1 – 1.5 | Softmax temperature |
| Top-K | 0 – 200 | Keep K most likely tokens (0 = off) |
| Top-P | 0 – 1.0 | Nucleus sampling threshold (0 = off) |
| Learning Rate | 0.01 – 1.0 | SGD step size |
| Momentum | 0.0 – 0.99 | EMA gradient inertia |
| Context Size | 1 – 5 | N-gram window length |

### Actions

- **Undo** — revert the last training step.
- **Reset** — erase the brain and restore all defaults.
- **Idle** — toggle idle thinking.
- **Import** — batch-train from a TXT file by path.
- **EN / RU** — switch interface language.

### HTTP API

All endpoints return JSON. The UI is a thin client over them, so you can script MYCOR from anything that speaks HTTP.

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| POST | `/api/chat` | `{"message": "..."}` | `{"reply": "...", "thoughts": "..."}` |
| POST | `/api/train` | `{"prompt": "...", "target": "..."}` | `{"loss": 0.42, "hasLoss": true}` |
| GET | `/api/stats` | — | vocabulary, parameters, config, recent contexts, brain path |
| POST | `/api/config` | any subset of config keys | `{"status": "ok"}` |
| POST | `/api/undo` | — | `{"ok": true}` |
| POST | `/api/reset` | — | `{"status": "ok"}` |
| POST | `/api/import` | `{"path": "book.txt"}` | vocabulary and parameter counts |
| POST | `/api/lang` | `{"lang": "en"}` | `{"lang": "en"}` |
| GET | `/api/history` | — | session messages |
| POST | `/api/save` | — | `{"status": "ok"}` |
| GET | `/api/idle` | — | `{"thought": "..."}` or empty |

---

## 🧠 Memory: Where the Brain Lives

The trained brain is stored in the operating system's per-user configuration directory, so the folder where you keep `mycor.exe` stays clean:

- **Windows:** `%AppData%\MYCOR\history.json`
- **Linux:** `~/.config/MYCOR/history.json`
- **macOS:** `~/Library/Application Support/MYCOR/history.json`

The exact path is printed in the terminal on every launch, in the line `Brain file: ...`. If the config directory cannot be resolved (rare, e.g. in a stripped-down container without `HOME`), the program falls back to `history.json` next to the executable.

**Saved to that file and survives restart:**
- Vocabulary.
- Weight matrix.
- Optimizer state (velocity).

**Reset on every launch:**
- Session dialogue history.
- Recent active contexts list.
- Last "thoughts" of the network.
- Backup copy for Undo.

**Reset only by the Reset button:**
- Runtime configuration.
- The brain file on disk (a fresh empty brain is written in its place).
- The in-memory brain.

To back up or share your trained character, copy `history.json` out of the folder above. To load someone else's brain, drop their file into the same folder before launching MYCOR — the app will pick it up on startup.

---

## 💭 How Thinking Mode Works

1. **Activation threshold:** `MinVocabForThinking = 10`.
2. **Internal generation:** The engine performs `thinkingSteps = 4` sampling steps from the same weight matrix.
3. **Answer context:** The resulting thought becomes a prefix to the prompt, and the final answer is generated from the extended context.
4. **Transparency:** The thought line is shown in the chat above the answer.

---

## 🪜 How Dynamic Backoff Works

For a context window of length N, the engine tries contexts in this order:

1. The full N-gram `(w1, ..., wN)`.
2. `keepLast k` for k = N-1 down to 1: only the last k words survive, the rest become `<unk>`.
3. `keepFirst k` for k = N-1 down to 1: only the first k words survive.
4. The fully-masked `<unk> ... <unk>` context.

The first context with trained weights wins. If none exist, generation falls back to a random vocabulary pick and thinking stops early. The chain is rebuilt every step from the current vocabulary and weight state, and deduplicated.

---

## 🎭 Special Tokens for Punctuation and Emoticons

Whole emoticons are single tokens:

`:-)`, `:)`, `;-)`, `;)`, `:-D`, `:D`, `:-P`, `:P`, `:3`, `:/`, `:|`, `:'(`, `:'-(`, `:*`, `^^`, `<3`, `o_o`, `-_-`

Punctuation is split into separate tokens, both ASCII (`. , ? ! : ;`) and Unicode typography (`— – … « » “ ” ‘ ’`).

---

## 💤 How Idle Thinking Works

1. **Timer:** After each message, an inactivity timer is reset. If no request arrives for `IdleTimeoutSec` seconds (default 45), the idle endpoint produces a thought.
2. **Seed:** The most recently touched context is used. If none exists, a random word from `Vocabulary[1:]`.
3. **Daydream:** The same thinking pipeline runs and the result is appended to the chat.

Toggle at runtime with the **Idle** button.

---

## 🛠️ How It Works Under the Hood

The mathematics of MYCOR take a little over 400 lines of code in the `internal/engine/` folder.

Training uses:
- **Softmax** with numerical stabilization (max shift, uniform fallback).
- **Cross-entropy** loss.
- **SGD with momentum:** `v = m*v + (1-m)*grad; w = w*decay + lr*v`.
- **L2 weight decay.**

Generation uses:
- **Temperature**, **Top-K**, **Top-P**, **Repetition Penalty**.
- **Thinking Mode** for inner monologue.
- **Dynamic Backoff** for context descent.
- **Idle Thinking** for autonomous daydreams.

---

## 🗺️ Roadmap

- [x] v1.0 — Character-level model.
- [x] v2.0 — Word-level tokens.
- [x] v3.0 — Sliding N-gram context window.
- [x] v4.0 — Command processor, loss calculation, batch import.
- [x] v4.1 — Engine tests, fixes for undo, softmax, import, validation.
- [x] v4.2 — Momentum, weight decay, atomic save, persistent memory.
- [x] v4.3 — Thinking Mode, transparent thoughts, `<unk>` filtering.
- [x] v4.4 — Punctuation/emoticon tokens, Dynamic Backoff, Idle Thinking, bilingual UI.
- [x] v4.5 — Hardening: encapsulation, Unicode tokenizer, safe softmax, config reset.
- [x] v5.0 — Desktop edition: embedded web UI, Top-K/Top-P, adaptive N-gram, gob weights, brain in user config dir.
- [ ] v5.1 (planned) — Drag-and-drop TXT import, loss chart, named brains, streaming answers.

## 📄 License

MIT. See `LICENCE`.