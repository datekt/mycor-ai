<p align="center">
  <img src="assets/mycor-banner.jpeg" alt="MYCOR AI Banner" width="100%">
</p>

# 🧠 MYCOR (Mycor) — v4.4

> **My Core. Zero Dependencies. Pure Go. Dynamic Context. Persistent Memory. Thinking. Daydreaming.**

**MYCOR** is a completely "empty" local language model written in pure **Go 1.27.1**, built from scratch without any third-party frameworks (PyTorch, TensorFlow), Python, or heavy C libraries (CGO).

In an era of terabyte-scale corporate black boxes, **MYCOR** returns control to the developer. It is a digital sandbox and canvas whose character you shape entirely yourself.

---

## 🎯 Project Philosophy

On first launch, MYCOR is a "blank slate" (Tabula Rasa). To any of your requests, the model will output chaotic nonsense. It has no pre-loaded knowledge, censorship, or bias.

**It learns only from you.** By chatting with it in the console and correcting its answers, you literally build its "brain" (a dynamic weight matrix of N-grams) from scratch. Grow your own, albeit simple, but truly *personal* neural network right on your local hardware.

---

## 🆕 What's new in v4.4

- 🎭 **Special tokens for punctuation and emoticons:** The tokenizer now recognizes whole emoticons (`:)`, `:D`, `<3`, `^^`, `o_o`, `-_-`, `;-)`, etc.) as single tokens. The model can now learn your style of speech, your exclamations, your smileys and parentheses. Character imitation becomes far more expressive.
- 🪜 **Dynamic Backoff:** A purely technical engine improvement. If the trigram context `(w1, w2)` has no trained weights, the engine now smoothly descends to bigram contexts (`<unk> w2`, `w1 <unk>`) and finally to the unigram context (`<unk> <unk>`) instead of returning `<unk>` or nothing. Speech remains coherent even on rare phrases.
- 💤 **Idle Thinking ("daydreams" in the background):** If you step away from the computer (default 45 seconds of inactivity), the AI starts quietly muttering its own thoughts to the console. It continues the last active context on its own. Toggle with `/idle on|off`.
- 🌐 **Bilingual interface:** At startup, the program asks you to choose a language (English / Русский) by pressing 1, 2, or Enter. During a session, you can switch at any time with `/lang en` or `/lang ru`. All engine internals, variable names, and error messages are now entirely in English.
- 🧹 **Critical bug fixes:**
  - `RegisterWord` now ignores empty strings — the weights matrix and vocabulary can no longer desynchronize.
  - `backoffChain` no longer produces duplicate context keys when one of the arguments is already `<unk>`.
  - Idle-thought generation correctly resets the inactivity timer after every user action.
- ⬆️ **Version bump:** `modelVersion = 4`.

---

## ✨ Key Features of v4.4

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
- 📖 **Batch import:** Instant auto-training from a text file.
- 🌀 **Momentum Optimizer:** Weight updates with inertia (EMA gradient).
- 🧮 **L2 Weight Decay:** Light regularization against weight blow-up.
- 🛡️ **Atomic save:** Brain is written to a temp file and then renamed.
- 🔒 **Strict validation:** Duplicate words, weight length, and JSON integrity are checked on load.
- 🧪 **Test coverage:** Full suite of unit tests for the engine, importer, thinking mode, dynamic backoff, and idle thinking.

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
- `/reset` — Completely erase `history.json` and start from scratch.
- `/temp [0.1 - 1.5]` — Creativity (Softmax temperature).
- `/lr [0.01 - 1.0]` — Learning rate.
- `/momentum [0.0 - 0.99]` — Gradient inertia (default 0.9).
- `/import [path]` — Batch training from a text file.
- `/self` — Autonomous AI talking to its mirror (with thoughts shown).
- `/lang [en|ru]` — Switch the interface language.
- `/idle [on|off]` — Enable / disable idle thinking (daydreams).
- `exit` — Safe exit with weights saved.

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

This separation lets you accumulate knowledge between sessions without dragging random garbage from previous conversations.

---

## 💭 How Thinking Mode Works

1. **Activation threshold:** `MinVocabForThinking = 10`. While the vocabulary has fewer than 10 unique words, the model answers instantly, like v4.2.
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

---

## 🎭 Special Tokens for Punctuation and Emoticons

The tokenizer recognizes whole emoticons as single tokens:

`:-)`, `:)`, `;-)`, `;)`, `:-D`, `:D`, `:-P`, `:P`, `:3`, `:/`, `:|`, `:'(`, `:'-(`, `:*`, `^^`, `<3`, `o_o`, `-_-`

Punctuation (`.`, `,`, `?`, `!`, `:`, `;`) is still split into separate tokens, which lets the model learn to attach punctuation to words correctly.

This means your trained network can pick up your habit of ending messages with `:)` or starting them with `^^`, and reproduce it in its own output — a small but important part of what "character" means.

---

## 💤 How Idle Thinking Works

Idle thinking is a background daydream mode. It works like this:

1. **Timer:** After each of your messages, an inactivity timer is reset. If no key is pressed for `IdleTimeoutSec` seconds (default 45), the idle ticker fires.
2. **Seed:** The engine takes the most recently touched context (from `RecentContexts`). If none exists yet, it picks a random word from the vocabulary.
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
- **Softmax with numerical stabilization** (shift by max).
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
- [x] v4.4 — Special tokens for punctuation and emoticons, Dynamic Backoff, Idle Thinking, bilingual interface, critical bug fixes.
- [ ] v5.0 (In development) — Local Web UI:
  - Interactive chat in the browser via Go HTML templates.
  - Visual sliders for Temperature and Learning Rate.
  - Drag-and-Drop form for uploading text books.
  - Real-time Loss charts.

## 📄 License

The project is distributed under the free **MIT** license. You are free to modify, use, and develop this code.