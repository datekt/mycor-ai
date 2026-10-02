package web

import (
	"log"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"mycor/internal/config"
	"mycor/internal/engine"
	"mycor/internal/i18n"
	"mycor/internal/importer"
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
	engine.Stats
	Temperature float64 `json:"temperature"`
	// embedded Stats already carries LearningRate/Momentum/TopK/TopP via
	// config, so only the sampling knobs missing from it are listed here.
	IdleEnabled    bool   `json:"idleEnabled"`
	IdleTimeoutSec int    `json:"idleTimeoutSec"`
	Lang           string `json:"lang"`
	BrainPath      string `json:"brainPath"`
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
	ThinkingSteps  *int     `json:"thinkingSteps,omitempty"`
}

type importRequest struct {
	Path string `json:"path"`
}

type langRequest struct {
	Lang string `json:"lang"`
}

// handleChat answers a user message.
//
// It never takes a server-wide lock: the brain serialises itself, so a chat
// request stays responsive even while a large import is running.
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := decodeJSON(w, r, &req); err != nil {
		return
	}
	if strings.TrimSpace(req.Message) == "" {
		writeError(w, http.StatusBadRequest, "empty message")
		return
	}

	s.markActive()
	s.brain.AppendHistory("user", req.Message)
	reply := s.brain.GenerateResponse(req.Message, 15)
	s.brain.AppendHistory("ai", reply)

	writeJSON(w, chatResponse{
		Reply:    reply,
		Thoughts: s.brain.LastThoughts(),
	})
}

// handleTrain applies one correction.
//
// The brain is no longer written to disk inside the request. Writing the whole
// model on every single training step made the UI freeze; instead the change
// is marked dirty and the background saver flushes it, while /api/save forces
// an immediate write.
func (s *Server) handleTrain(w http.ResponseWriter, r *http.Request) {
	var req trainRequest
	if err := decodeJSON(w, r, &req); err != nil {
		return
	}
	if strings.TrimSpace(req.Target) == "" {
		writeError(w, http.StatusBadRequest, "empty target")
		return
	}

	s.markActive()
	s.brain.AppendHistory("teacher", req.Target)
	loss := s.brain.Train(req.Prompt, req.Target)
	s.saver.mark()

	writeJSON(w, chatResponse{
		Loss:    loss,
		HasLoss: true,
	})
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "GET required")
		return
	}
	cfg := config.Get()
	writeJSON(w, statsResponse{
		Stats:          s.brain.Stats(),
		Temperature:    cfg.Temperature,
		IdleEnabled:    cfg.IdleEnabled,
		IdleTimeoutSec: cfg.IdleTimeoutSec,
		Lang:           lang(),
		BrainPath:      s.brainFile,
	})
}

// handleConfig validates and applies configuration changes, echoing the
// effective values back so the client can correct out-of-range sliders.
// GET returns the active configuration so the client can restore its controls
// on load; POST applies changes and echoes the effective values back.
func (s *Server) handleConfig(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		writeJSON(w, config.Get())
		return
	}

	var req configRequest
	if err := decodeJSON(w, r, &req); err != nil {
		return
	}

	updates := []struct {
		name  string
		value *float64
		flag  *bool
	}{
		{"temperature", req.Temperature, nil},
		{"learningRate", req.LearningRate, nil},
		{"momentum", req.Momentum, nil},
		{"topK", f64i(req.TopK), nil},
		{"topP", req.TopP, nil},
		{"contextSize", f64i(req.ContextSize), nil},
		{"idleTimeoutSec", f64i(req.IdleTimeoutSec), nil},
		{"thinkingSteps", f64i(req.ThinkingSteps), nil},
		{"idleEnabled", nil, req.IdleEnabled},
	}
	for _, u := range updates {
		if u.value == nil && u.flag == nil {
			continue
		}
		value := 0.0
		if u.value != nil {
			value = *u.value
		}
		flag := u.flag != nil && *u.flag
		if _, err := config.SetField(u.name, value, flag); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
	}

	s.saver.mark()
	writeJSON(w, config.Get())
}

func (s *Server) handleUndo(w http.ResponseWriter, r *http.Request) {
	s.markActive()
	ok := s.brain.UndoLastTrain()
	if ok {
		s.saver.mark()
	}
	writeJSON(w, map[string]bool{"ok": ok})
}

// handleReset erases the brain and restores default configuration. Both the
// server state and the files on disk are reset so a restart cannot resurrect
// the old model.
func (s *Server) handleReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodDelete {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	s.brain.Reset()
	config.Reset()
	s.markActive()
	s.persist()
	writeJSON(w, map[string]string{"status": "ok"})
}

// handleImport trains the brain from a text file chosen by the user.
//
// The previous handler accepted any absolute path from the request body, which
// let a request that reached the server read and parse arbitrary files on
// disk. The path is now validated against an explicit allow-list of roots, and
// the file must be a regular .txt file.
func (s *Server) handleImport(w http.ResponseWriter, r *http.Request) {
	var req importRequest
	if err := decodeJSON(w, r, &req); err != nil {
		return
	}
	path, err := resolveImportPath(req.Path)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	s.markActive()
	result, err := importer.ImportTxtFile(s.brain, path, "", func(processed, total int) {
		log.Printf("import %s: %d/%d (%.0f%%)", filepath.Base(path), processed, total,
			100*float64(processed)/float64(max(total, 1)))
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	s.saver.mark()

	writeJSON(w, importResponse{
		Status:      "ok",
		Lines:       result.Lines,
		Trained:     result.Trained,
		AverageLoss: result.AverageLoss,
		Vocabulary:  result.Vocabulary,
		Parameters:  result.Parameters,
	})
}

type importResponse struct {
	Status      string  `json:"status"`
	Lines       int     `json:"lines"`
	Trained     int     `json:"trained"`
	AverageLoss float64 `json:"averageLoss"`
	Vocabulary  int     `json:"vocabulary"`
	Parameters  int     `json:"parameters"`
}

// handleLang switches the UI language.
func (s *Server) handleLang(w http.ResponseWriter, r *http.Request) {
	var req langRequest
	if err := decodeJSON(w, r, &req); err != nil {
		return
	}
	switch i18n.Lang(req.Lang) {
	case i18n.RU:
		i18n.Set(i18n.RU)
	case i18n.EN:
		i18n.Set(i18n.EN)
	default:
		writeError(w, http.StatusBadRequest, "unsupported language: "+req.Lang)
		return
	}
	writeJSON(w, map[string]string{"lang": lang()})
}

// handleHistory returns the conversation so the frontend can restore a session
// after a reload.
func (s *Server) handleHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "GET required")
		return
	}
	writeJSON(w, s.brain.SessionHistory())
}

// handleSave forces an immediate write to disk.
func (s *Server) handleSave(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	if err := s.brain.Save(s.brainFile); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := config.Save(s.configFile); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, map[string]string{"status": "ok"})
}

// handleIdle returns a spontaneous thought once the user has been idle long
// enough, otherwise an empty string.
func (s *Server) handleIdle(w http.ResponseWriter, r *http.Request) {
	cfg := config.Get()
	if !cfg.IdleEnabled || s.idleFor() < time.Duration(cfg.IdleTimeoutSec)*time.Second {
		writeJSON(w, map[string]string{"thought": ""})
		return
	}
	thought := s.brain.GenerateIdleThought()
	if thought != "" {
		s.markActive()
	}
	writeJSON(w, map[string]string{"thought": thought})
}

// f64i converts an int pointer to a float64 pointer for uniform handling.
func f64i(p *int) *float64 {
	if p == nil {
		return nil
	}
	v := float64(*p)
	return &v
}
