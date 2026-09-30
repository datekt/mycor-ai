/*
Package web serves the MYCOR AI single-page interface from an embedded
filesystem and exposes a small JSON API for chat, training, statistics,
configuration and administrative actions.

The server binds to 127.0.0.1 on an OS-assigned port and opens the user's
default browser on startup, turning the binary into a self-contained
desktop application without any external dependencies.

The trained brain lives in the operating system's per-user configuration
directory (see paths.go) so that the executable folder stays clean and
the brain survives updates to the binary.
*/
package web

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"net"
	"net/http"
	"os/exec"
	"runtime"
	"sync"
	"time"

	"mycor/internal/config"
	"mycor/internal/engine"
	"mycor/internal/i18n"
	"mycor/internal/importer"
)

//go:embed static/*
var staticFS embed.FS

var (
	mu           sync.Mutex
	lastActivity time.Time
	brainFile    string
)

type chatRequest struct {
	Message string `json:"message"`
}

type trainRequest struct {
	Prompt string `json:"prompt"`
	Target string `json:"target"`
}

type chatResponse struct {
	Reply    string  `json:"reply"`
	Thoughts string  `json:"thoughts"`
	Loss     float64 `json:"loss"`
	HasLoss  bool    `json:"hasLoss"`
}

type statsResponse struct {
	Vocabulary     int      `json:"vocabulary"`
	Parameters     int      `json:"parameters"`
	Thinking       bool     `json:"thinking"`
	MinThinking    int      `json:"minThinking"`
	Temperature    float64  `json:"temperature"`
	LearningRate   float64  `json:"learningRate"`
	Momentum       float64  `json:"momentum"`
	TopK           int      `json:"topK"`
	TopP           float64  `json:"topP"`
	ContextSize    int      `json:"contextSize"`
	IdleEnabled    bool     `json:"idleEnabled"`
	IdleTimeoutSec int      `json:"idleTimeoutSec"`
	Lang           string   `json:"lang"`
	RecentContexts []string `json:"recentContexts"`
	BrainPath      string   `json:"brainPath"`
}

type configRequest struct {
	Temperature    *float64 `json:"temperature,omitempty"`
	LearningRate   *float64 `json:"learningRate,omitempty"`
	Momentum       *float64 `json:"momentum,omitempty"`
	TopK           *int     `json:"topK,omitempty"`
	TopP           *float64 `json:"topP,omitempty"`
	ContextSize    *int     `json:"contextSize,omitempty"`
	IdleEnabled    *bool    `json:"idleEnabled,omitempty"`
	IdleTimeoutSec *int     `json:"idleTimeoutSec,omitempty"`
}

type importRequest struct {
	Path string `json:"path"`
}

type langRequest struct {
	Lang string `json:"lang"`
}

func Run() {
	path, err := resolveBrainPath()
	if err != nil {
		fmt.Println("Warning: cannot resolve user config directory, using current folder:", err)
		path = "history.json"
	}
	brainFile = path
	fmt.Println("Brain file:", brainFile)

	engine.InitEngine()
	engine.LoadBrain(brainFile)
	lastActivity = time.Now()

	sub, err := fs.Sub(staticFS, "static")
	if err != nil {
		panic(err)
	}

	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(sub)))
	mux.HandleFunc("/api/chat", handleChat)
	mux.HandleFunc("/api/train", handleTrain)
	mux.HandleFunc("/api/stats", handleStats)
	mux.HandleFunc("/api/config", handleConfig)
	mux.HandleFunc("/api/undo", handleUndo)
	mux.HandleFunc("/api/reset", handleReset)
	mux.HandleFunc("/api/import", handleImport)
	mux.HandleFunc("/api/lang", handleLang)
	mux.HandleFunc("/api/history", handleHistory)
	mux.HandleFunc("/api/save", handleSave)
	mux.HandleFunc("/api/idle", handleIdle)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	addr := listener.Addr().String()
	url := "http://" + addr + "/"
	fmt.Printf("MYCOR AI v5.0 running at %s\n", url)
	go openBrowser(url)

	server := &http.Server{
		Handler:      mux,
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 60 * time.Second,
	}
	if err := server.Serve(listener); err != nil {
		fmt.Println("Server stopped:", err)
	}
}

func openBrowser(url string) {
	time.Sleep(300 * time.Millisecond)
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	_ = cmd.Start()
}

func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	mu.Lock()
	defer mu.Unlock()

	lastActivity = time.Now()
	engine.AppendHistory("user", req.Message)
	reply := engine.GenerateResponse(req.Message, 15)
	engine.AppendHistory("ai", reply)

	writeJSON(w, chatResponse{
		Reply:    reply,
		Thoughts: engine.LastThoughts(),
	})
}

func handleTrain(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req trainRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	mu.Lock()
	defer mu.Unlock()

	lastActivity = time.Now()
	if req.Target == "" {
		writeJSON(w, chatResponse{})
		return
	}
	engine.AppendHistory("teacher", req.Target)
	loss := engine.Train(req.Prompt, req.Target)
	_ = engine.SaveBrain(brainFile)
	writeJSON(w, chatResponse{
		Loss:    loss,
		HasLoss: true,
	})
}

func handleStats(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()

	lang := "en"
	if i18n.Current() == i18n.RU {
		lang = "ru"
	}
	writeJSON(w, statsResponse{
		Vocabulary:     len(engine.Vocabulary),
		Parameters:     engine.CountParameters(),
		Thinking:       len(engine.Vocabulary) >= engine.MinVocabForThinking,
		MinThinking:    engine.MinVocabForThinking,
		Temperature:    config.Temperature,
		LearningRate:   config.LearningRate,
		Momentum:       config.Momentum,
		TopK:           config.TopK,
		TopP:           config.TopP,
		ContextSize:    config.ContextSize,
		IdleEnabled:    config.IdleEnabled,
		IdleTimeoutSec: config.IdleTimeoutSec,
		Lang:           lang,
		RecentContexts: engine.RecentContexts(),
		BrainPath:      brainFile,
	})
}

func handleConfig(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req configRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	mu.Lock()
	defer mu.Unlock()

	if req.Temperature != nil && *req.Temperature >= 0.1 && *req.Temperature <= 1.5 {
		config.Temperature = *req.Temperature
	}
	if req.LearningRate != nil && *req.LearningRate >= 0.01 && *req.LearningRate <= 1.0 {
		config.LearningRate = *req.LearningRate
	}
	if req.Momentum != nil && *req.Momentum >= 0.0 && *req.Momentum <= 0.99 {
		config.Momentum = *req.Momentum
	}
	if req.TopK != nil && *req.TopK >= 0 && *req.TopK <= 1000 {
		config.TopK = *req.TopK
	}
	if req.TopP != nil && *req.TopP >= 0.0 && *req.TopP <= 1.0 {
		config.TopP = *req.TopP
	}
	if req.ContextSize != nil && *req.ContextSize >= config.MinContextSize && *req.ContextSize <= config.MaxContextSize {
		config.ContextSize = *req.ContextSize
	}
	if req.IdleEnabled != nil {
		config.IdleEnabled = *req.IdleEnabled
	}
	if req.IdleTimeoutSec != nil && *req.IdleTimeoutSec > 0 {
		config.IdleTimeoutSec = *req.IdleTimeoutSec
	}
	writeJSON(w, map[string]string{"status": "ok"})
}

func handleUndo(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()
	ok := engine.UndoLastTrain()
	if ok {
		_ = engine.SaveBrain(brainFile)
	}
	writeJSON(w, map[string]bool{"ok": ok})
}

func handleReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()
	engine.InitEngine()
	config.Reset()
	_ = engine.SaveBrain(brainFile)
	writeJSON(w, map[string]string{"status": "ok"})
}

func handleImport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req importRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	mu.Lock()
	defer mu.Unlock()
	if err := importer.ImportTxtFile(req.Path, brainFile); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, map[string]interface{}{
		"status":     "ok",
		"vocabulary": len(engine.Vocabulary),
		"parameters": engine.CountParameters(),
	})
}

func handleLang(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req langRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	mu.Lock()
	defer mu.Unlock()
	switch req.Lang {
	case "ru":
		i18n.Set(i18n.RU)
	case "en":
		i18n.Set(i18n.EN)
	}
	writeJSON(w, map[string]string{"lang": req.Lang})
}

func handleHistory(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()
	writeJSON(w, engine.SessionHistory())
}

func handleSave(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()
	if err := engine.SaveBrain(brainFile); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, map[string]string{"status": "ok"})
}

func handleIdle(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()
	if !config.IdleEnabled {
		writeJSON(w, map[string]string{"thought": ""})
		return
	}
	if time.Since(lastActivity) < time.Duration(config.IdleTimeoutSec)*time.Second {
		writeJSON(w, map[string]string{"thought": ""})
		return
	}
	thought := engine.GenerateIdleThought()
	if thought != "" {
		lastActivity = time.Now()
	}
	writeJSON(w, map[string]string{"thought": thought})
}
