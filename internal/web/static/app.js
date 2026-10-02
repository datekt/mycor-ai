"use strict";

/*
MYCOR single-page client.

Notes on the fixes this script implements:

  - Slider updates are debounced. Dragging a range input fires an input event
    per pixel, which previously produced a burst of POSTs and made the server
    rewrite configuration hundreds of times per gesture.
  - Translations live in the client. Previously only the Go side carried
    messages, so the UI stayed in English until a message was rendered.
  - Reset clears the chat, the teacher panel and every slider. The old code
    reset the server but left stale slider values and messages on screen.
  - Undo re-synchronises with the server history, because undoing a training
    step must remove the corresponding conversation turn.
  - The session is restored from /api/history on load, so a browser reload no
    longer loses the conversation.
  - The chosen language is remembered in localStorage.
*/

const $ = (sel) => document.querySelector(sel);
const messages = $("#messages");

let lastPrompt = "";
let pendingTrain = false;
let idleEnabled = true;
let busy = false;
let suppressPush = false;

// ---------------------------------------------------------------------------
// Client-side i18n
// ---------------------------------------------------------------------------

const I18N = {
  en: {
    tagline: "Top-K/Top-P · Adaptive N-gram · Sparse Weights",
    statistics: "Statistics", sampling: "Sampling", learning: "Learning",
    actions: "Actions", temperature: "Temperature", topK: "Top-K",
    topP: "Top-P", learningRate: "Learning Rate", momentum: "Momentum",
    contextSize: "Context Size", thinkingSteps: "Thinking Steps",
    undo: "Undo", reset: "Reset", idle: "Idle", import: "Import",
    teachCorrection: "Teach correction:", train: "Train", skip: "Skip",
    send: "Send",
    correctAnswerPlaceholder: "Type the correct answer, or leave empty to skip",
    messagePlaceholder: "Type a message and press Enter...",
    vocabulary: "Vocabulary", parameters: "Parameters", thinking: "Thinking",
    context: "Context", words: "words", on: "ON", off: "OFF",
    welcome: "Welcome to MYCOR AI. Chat, then teach it the correct answer to grow its brain.",
    sessionRestored: "Session restored.",
    undone: "Last training undone.",
    nothingToUndo: "Nothing to undo.",
    resetDone: "Brain erased and configuration reset.",
    resetConfirm: "Erase the brain and reset configuration?",
    idleOn: "Idle thinking enabled.",
    idleOff: "Idle thinking disabled.",
    idleThought: "Idle thought: {text}",
    thoughts: "MYCOR thoughts: {text}",
    noAnswer: "(no answer)",
    loss: "Loss: {value}",
    importPrompt: "Path to a .txt file inside your home, Documents, Downloads or Desktop folder:",
    importDone: "Imported {trained} examples. Vocabulary: {vocabulary}, Parameters: {parameters}.",
    importFail: "Import failed: {error}",
    errorPrefix: "Error: {error}",
    langSwitched: "Language switched to English.",
    serverGone: "Connection to the server was lost.",
    roleYou: "You", roleAI: "AI", roleTeacher: "Teacher",
  },
  ru: {
    tagline: "Top-K/Top-P · Адаптивная N-граммная модель · Разрежённые веса",
    statistics: "Статистика", sampling: "Сэмплирование", learning: "Обучение",
    actions: "Действия", temperature: "Температура", topK: "Top-K",
    topP: "Top-P", learningRate: "Скорость обучения", momentum: "Инерция",
    contextSize: "Размер контекста", thinkingSteps: "Шаги размышлений",
    undo: "Отменить", reset: "Сброс", idle: "Простой", import: "Импорт",
    teachCorrection: "Правильный ответ:", train: "Обучить", skip: "Пропустить",
    send: "Отправить",
    correctAnswerPlaceholder: "Введите правильный ответ или оставьте пустым",
    messagePlaceholder: "Введите сообщение и нажмите Enter...",
    vocabulary: "Словарь", parameters: "Параметры", thinking: "Размышления",
    context: "Контекст", words: "слов", on: "ВКЛ", off: "ВЫКЛ",
    welcome: "Добро пожаловать в MYCOR AI. Поговорите с моделью, а затем научите её правильному ответу.",
    sessionRestored: "Сессия восстановлена.",
    undone: "Последнее обучение отменено.",
    nothingToUndo: "Нечего отменять.",
    resetDone: "Мозг стёрт, настройки сброшены.",
    resetConfirm: "Стереть мозг и сбросить настройки?",
    idleOn: "Размышления в простое включены.",
    idleOff: "Размышления в простое выключены.",
    idleThought: "Мысль в простое: {text}",
    thoughts: "Мысли MYCOR: {text}",
    noAnswer: "(нет ответа)",
    loss: "Потеря: {value}",
    importPrompt: "Путь к .txt файлу в домашней папке, Документах, Загрузках или на рабочем столе:",
    importDone: "Импортировано примеров: {trained}. Словарь: {vocabulary}, Параметры: {parameters}.",
    importFail: "Ошибка импорта: {error}",
    errorPrefix: "Ошибка: {error}",
    langSwitched: "Язык изменён на русский.",
    serverGone: "Соединение с сервером потеряно.",
    roleYou: "Вы", roleAI: "ИИ", roleTeacher: "Учитель",
  },
};

const LANG_KEY = "mycor.lang";
let lang = "en";

function t(key, vars) {
  const table = I18N[lang] || I18N.en;
  let text = table[key];
  if (text === undefined) text = I18N.en[key] !== undefined ? I18N.en[key] : key;
  if (vars) {
    for (const [name, value] of Object.entries(vars)) {
      text = text.split("{" + name + "}").join(String(value));
    }
  }
  return text;
}

function applyLanguage(next) {
  lang = I18N[next] ? next : "en";
  document.documentElement.lang = lang;
  document.title = "MYCOR AI";
  $("#tagline").textContent = t("tagline");

  for (const el of document.querySelectorAll("[data-i18n]")) {
    el.textContent = t(el.dataset.i18n);
  }
  for (const el of document.querySelectorAll("[data-i18n-placeholder]")) {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  }
  localStorage.setItem(LANG_KEY, lang);
}

// ---------------------------------------------------------------------------
// Messaging helpers
// ---------------------------------------------------------------------------

function appendMsg(cls, role, text) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  if (role) {
    const r = document.createElement("div");
    r.className = "role";
    r.textContent = role;
    div.appendChild(r);
  }
  const body = document.createElement("div");
  body.textContent = text;
  div.appendChild(body);
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function clearMessages() {
  messages.innerHTML = "";
}

function setBusy(value) {
  busy = value;
  $("#sidebar").classList.toggle("busy", value);
  $("#inputRow").classList.toggle("busy", value);
}

// roleLabel maps a server-side history role onto the translated label.
function roleLabel(role) {
  switch (role) {
    case "user": return t("roleYou");
    case "ai": return t("roleAI");
    case "teacher": return t("roleTeacher");
    default: return role;
  }
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function api(path, body, method) {
  const opts = { method: method || (body ? "POST" : "GET") };
  if (body) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

// renderHistory repaints the chat from the authoritative server history, which
// is how Undo and Reset bring the transcript back in sync.
function renderHistory(history) {
  clearMessages();
  for (const msg of history) {
    appendMsg(roleClass(msg.role), roleLabel(msg.role), msg.text);
  }
}

function roleClass(role) {
  switch (role) {
    case "user": return "user";
    case "ai": return "ai";
    default: return "system";
  }
}

async function syncHistory() {
  try {
    renderHistory(await api("/api/history"));
  } catch (e) {
    appendMsg("system", "", t("errorPrefix", { error: e.message }));
  }
}

// ---------------------------------------------------------------------------
// Chat and training
// ---------------------------------------------------------------------------

async function sendMessage() {
  const input = $("#input");
  const text = input.value.trim();
  if (!text || busy) return;
  input.value = "";
  appendMsg("user", t("roleYou"), text);
  lastPrompt = text;

  setBusy(true);
  try {
    const data = await api("/api/chat", { message: text });
    if (data.thoughts) {
      appendMsg("thoughts", "", t("thoughts", { text: data.thoughts }));
    }
    appendMsg("ai", t("roleAI"), data.reply || t("noAnswer"));
    pendingTrain = true;
    $("#trainPanel").classList.add("visible");
    $("#trainInput").focus();
  } catch (e) {
    appendMsg("system", "", t("errorPrefix", { error: e.message }));
  } finally {
    setBusy(false);
  }
}

async function submitTrain(skip) {
  if (!pendingTrain || busy) return;
  const input = $("#trainInput");
  const target = skip ? "" : input.value.trim();
  pendingTrain = false;
  input.value = "";
  $("#trainPanel").classList.remove("visible");
  if (!target) return;

  setBusy(true);
  try {
    const data = await api("/api/train", { prompt: lastPrompt, target });
    if (data.hasLoss) {
      appendMsg("system", "", t("loss", { value: data.loss.toFixed(4) }));
    }
    await refreshStats();
  } catch (e) {
    appendMsg("system", "", t("errorPrefix", { error: e.message }));
  } finally {
    setBusy(false);
  }
}

// ---------------------------------------------------------------------------
// Statistics and configuration
// ---------------------------------------------------------------------------

async function refreshStats() {
  let s;
  try {
    s = await api("/api/stats");
  } catch (e) {
    console.error(e);
    return;
  }
  const el = $("#stats");
  el.innerHTML = "";
  const rows = [
    [t("vocabulary"), s.vocabulary],
    [t("parameters"), s.parameters],
    [t("thinking"), s.thinking ? t("on") : t("off")],
    [t("context"), s.contextSize + " " + t("words")],
  ];
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    row.className = "stat-row";
    const left = document.createElement("span");
    left.textContent = label;
    const right = document.createElement("span");
    right.textContent = String(value);
    row.append(left, right);
    el.appendChild(row);
  }
}
// ---------------------------------------------------------------------------
// Sliders
// ---------------------------------------------------------------------------

/*
debounce(fn, ms) delays a call until `ms` have passed without a new invocation.

A range input emits an input event for every pixel of a drag. Sending a request
per event flooded the server and made the sliders feel sticky; now the gesture
produces exactly one request.
*/
function debounce(fn, ms) {
  let timer = null;
  return function (...args) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      fn.apply(this, args);
    }, ms);
  };
}

/*
bindSlider wires a range input to a configuration key.

The label updates instantly while dragging, but the request is debounced, so a
whole gesture results in one POST. The server echoes the effective value, which
lets the control snap back when the server clamps the request.
*/
function bindSlider(id, labelId, key, fmt, transform) {
  const el = $(id);
  const lab = $(labelId);

  const send = debounce(async (v) => {
    if (suppressPush) return;
    try {
      const effective = await api("/api/config", { [key]: transform(v) });
      if (effective && typeof effective[key] === "number") {
        el.value = effective[key];
        lab.textContent = fmt(parseFloat(el.value));
      }
    } catch (e) {
      console.error(e);
    }
  }, SLIDER_DEBOUNCE_MS);

  el.addEventListener("input", () => {
    const v = parseFloat(el.value);
    lab.textContent = fmt(v);
    send(v);
  });
  return { el, lab, fmt, key };
}

const SLIDER_DEBOUNCE_MS = 200;

const sliders = [
  bindSlider("#temp", "#tempVal", "temperature", (v) => v.toFixed(2), (v) => v),
  bindSlider("#topk", "#topkVal", "topK", (v) => Math.round(v).toString(), (v) => Math.round(v)),
  bindSlider("#topp", "#toppVal", "topP", (v) => v.toFixed(2), (v) => v),
  bindSlider("#lr", "#lrVal", "learningRate", (v) => v.toFixed(2), (v) => v),
  bindSlider("#mom", "#momVal", "momentum", (v) => v.toFixed(2), (v) => v),
  bindSlider("#ctx", "#ctxVal", "contextSize", (v) => Math.round(v).toString(), (v) => Math.round(v)),
  bindSlider("#think", "#thinkVal", "thinkingSteps", (v) => Math.round(v).toString(), (v) => Math.round(v)),
];

// applyToSliders pulls the persisted settings back into the UI. suppressPush
// stops the programmatic assignment from echoing a change back to the server.
function applyToSliders(cfg) {
  suppressPush = true;
  for (const { el, lab, fmt, key } of sliders) {
    const value = cfg[key];
    if (typeof value !== "number") continue;
    el.value = value;
    lab.textContent = fmt(value);
  }
  suppressPush = false;
}

// resetSliders puts every control back to its default. The server resets to the
// same values, so both sides agree after a reset.
function resetSliders() {
  suppressPush = true;
  for (const { el, lab, fmt } of sliders) {
    el.value = el.defaultValue;
    lab.textContent = fmt(parseFloat(el.value));
  }
  suppressPush = false;
}
// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

// Undo reverts a training step, which also removes the teacher turn from the
// session history. Re-reading the history keeps the transcript consistent
// instead of leaving an orphaned correction on screen.
async function undoTraining() {
  try {
    const r = await api("/api/undo", {});
    await syncHistory();
    appendMsg("system", "", r.ok ? t("undone") : t("nothingToUndo"));
    await refreshStats();
  } catch (e) {
    appendMsg("system", "", t("errorPrefix", { error: e.message }));
  }
}

// Reset wipes the brain server-side, so the client must drop every trace of the
// old session: messages, the teacher panel, the pending prompt and the sliders.
async function resetEverything() {
  if (!confirm(t("resetConfirm"))) return;
  try {
    await api("/api/reset", {});
    clearMessages();
    resetSliders();
    pendingTrain = false;
    lastPrompt = "";
    $("#trainPanel").classList.remove("visible");
    $("#trainInput").value = "";
    $("#input").value = "";
    idleEnabled = true;
    appendMsg("system", "", t("resetDone"));
    appendMsg("system", "", t("welcome"));
    await refreshStats();
  } catch (e) {
    appendMsg("system", "", t("errorPrefix", { error: e.message }));
  }
}

async function toggleIdle() {
  idleEnabled = !idleEnabled;
  try {
    await api("/api/config", { idleEnabled });
    appendMsg("system", "", idleEnabled ? t("idleOn") : t("idleOff"));
  } catch (e) {
    idleEnabled = !idleEnabled;
    appendMsg("system", "", t("errorPrefix", { error: e.message }));
  }
}

async function runImport() {
  const path = prompt(t("importPrompt"));
  if (!path) return;
  setBusy(true);
  try {
    const r = await api("/api/import", { path });
    appendMsg("system", "", t("importDone", {
      trained: r.trained,
      vocabulary: r.vocabulary,
      parameters: r.parameters,
    }));
    await refreshStats();
  } catch (e) {
    appendMsg("system", "", t("importFail", { error: e.message }));
  } finally {
    setBusy(false);
  }
}

async function switchLanguage() {
  const next = lang === "ru" ? "en" : "ru";
  try {
    await api("/api/lang", { lang: next });
    applyLanguage(next);
    // Re-render everything that contains translated text.
    await syncHistory();
    await refreshStats();
    appendMsg("system", "", t("langSwitched"));
  } catch (e) {
    appendMsg("system", "", t("errorPrefix", { error: e.message }));
  }
}

async function pollIdle() {
  try {
    const r = await api("/api/idle");
    if (r.thought) {
      appendMsg("thoughts", "", t("idleThought", { text: r.thought }));
    }
  } catch (e) {
    // The server may be shutting down; stop the poll rather than spam errors.
    return;
  }
  setTimeout(pollIdle, 2000);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

$("#send").addEventListener("click", sendMessage);
$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
$("#trainSend").addEventListener("click", () => submitTrain(false));
$("#trainSkip").addEventListener("click", () => submitTrain(true));
$("#trainInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    submitTrain(false);
  }
  if (e.key === "Escape") {
    e.preventDefault();
    submitTrain(true);
  }
});
$("#undoBtn").addEventListener("click", undoTraining);
$("#resetBtn").addEventListener("click", resetEverything);
$("#idleBtn").addEventListener("click", toggleIdle);
$("#importBtn").addEventListener("click", runImport);
$("#langBtn").addEventListener("click", switchLanguage);

// restoreSession repaints a previous conversation so a reload does not lose it.
async function restoreSession() {
  let history = [];
  try {
    history = await api("/api/history");
  } catch (e) {
    appendMsg("system", "", t("errorPrefix", { error: e.message }));
    return;
  }
  if (Array.isArray(history) && history.length > 0) {
    renderHistory(history);
    appendMsg("system", "", t("sessionRestored"));
  } else {
    appendMsg("system", "", t("welcome"));
  }
}

async function main() {
  applyLanguage(localStorage.getItem(LANG_KEY) || "en");

  let cfg;
  try {
    cfg = await api("/api/config");
  } catch (e) {
    appendMsg("system", "", t("serverGone"));
    return;
  }
  idleEnabled = cfg.idleEnabled;

  applyToSliders(cfg);
  await restoreSession();
  await refreshStats();
  pollIdle();
  setInterval(refreshStats, 5000);
}

main();