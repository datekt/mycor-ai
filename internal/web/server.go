/*
Package web serves the MYCOR AI single-page interface from an embedded
filesystem and exposes a small JSON API for chat, training, statistics,
configuration and administrative actions.

The server binds to 127.0.0.1 on an OS-assigned port and opens the user's
default browser on startup, turning the binary into a self-contained desktop
application without any external dependencies.

Concurrency: there is no package-level mutex. Each handler touches exactly the
state it needs, the brain protects itself with its own RWMutex, and the idle
tracker uses its own small lock. A bulk import therefore no longer blocks
chat, because /api/import no longer holds a global lock across the whole file.

The trained brain and the user configuration live in the operating system's
per-user configuration directory (see paths.go) so that the executable folder
stays clean and both survive updates to the binary.
*/
package web

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net"
	"net/http"
	"os/exec"
	"runtime"
	"sync"
	"time"

	"mycor/internal/config"
	"mycor/internal/engine"
	"mycor/internal/i18n"
)

//go:embed static/*
var staticFS embed.FS

// Server owns all mutable server state.
type Server struct {
	brain      *engine.Brain
	brainFile  string
	configFile string
	static     fs.FS

	// importRoots is the allow-list of directories /api/import may read from.
	// It is a field rather than a package-level value so tests can point it at
	// their own temp directory instead of depending on where the platform
	// happens to place it.
	importRoots []string

	idleMu       sync.Mutex
	lastActivity time.Time

	saver *saveDebouncer
}

// NewServer builds a server bound to a brain file and a config file.
func NewServer(brainFile, configFile string) (*Server, error) {
	sub, err := fs.Sub(staticFS, "static")
	if err != nil {
		return nil, fmt.Errorf("mount static assets: %w", err)
	}
	s := &Server{
		brain:        engine.NewBrain(),
		brainFile:    brainFile,
		configFile:   configFile,
		static:       sub,
		importRoots:  importRoots(),
		lastActivity: time.Now(),
	}
	s.saver = newSaveDebouncer(2*time.Second, 15*time.Second, s.persist)
	return s, nil
}

// Brain exposes the model, mainly for tests.
func (s *Server) Brain() *engine.Brain { return s.brain }

// Run starts the server and blocks until ctx is cancelled, then shuts down
// gracefully.
func Run(ctx context.Context) error {
	brainPath, cfgPath, err := resolveDataPaths()
	if err != nil {
		return err
	}
	srv, err := NewServer(brainPath, cfgPath)
	if err != nil {
		return err
	}
	return srv.Run(ctx, brainPath)
}

// Run serves until ctx is done.
func (s *Server) Run(ctx context.Context, brainPath string) error {
	if err := config.Load(s.configFile); err != nil {
		log.Printf("config: %v (continuing with defaults)", err)
	}
	if err := s.brain.Load(brainPath); err != nil && !errors.Is(err, fs.ErrNotExist) {
		log.Printf("brain: %v (starting empty)", err)
	}
	log.Printf("Brain file: %s", brainPath)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	url := "http://" + listener.Addr().String() + "/"
	log.Printf("MYCOR AI v6.0 running at %s", url)

	server := &http.Server{
		Handler:           s.routes(),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      120 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 16,
	}
	s.saver.start(ctx)

	go openBrowser(url)

	errCh := make(chan error, 1)
	go func() {
		if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
			return
		}
		errCh <- nil
	}()

	select {
	case err := <-errCh:
		s.shutdown(server)
		return err
	case <-ctx.Done():
		log.Println("Shutting down...")
		s.shutdown(server)
		return nil
	}
}

// shutdown stops accepting connections, lets in-flight requests finish, then
// persists the brain and the configuration.
func (s *Server) shutdown(server *http.Server) {
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
	}
	s.saver.stop()
	s.persist()
}

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(s.static)))
	mux.HandleFunc("/api/chat", s.handleChat)
	mux.HandleFunc("/api/train", s.handleTrain)
	mux.HandleFunc("/api/stats", s.handleStats)
	mux.HandleFunc("/api/config", s.handleConfig)
	mux.HandleFunc("/api/undo", s.handleUndo)
	mux.HandleFunc("/api/reset", s.handleReset)
	mux.HandleFunc("/api/import", s.handleImport)
	mux.HandleFunc("/api/lang", s.handleLang)
	mux.HandleFunc("/api/history", s.handleHistory)
	mux.HandleFunc("/api/save", s.handleSave)
	mux.HandleFunc("/api/idle", s.handleIdle)
	mux.HandleFunc("/api/health", s.handleHealth)
	return s.withBodyLimit(mux)
}

// maxBodyBytes caps every JSON request body. Without it a client could stream
// an unbounded payload into the decoder and exhaust server memory.
const maxBodyBytes = 64 << 10

func (s *Server) withBodyLimit(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
		}
		next.ServeHTTP(w, r)
	})
}

// decodeJSON reads a size-capped JSON body into dst.
func decodeJSON(w http.ResponseWriter, r *http.Request, dst interface{}) error {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return errors.New("method not allowed")
	}
	dec := json.NewDecoder(io.LimitReader(r.Body, maxBodyBytes))
	if err := dec.Decode(dst); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return err
	}
	return nil
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
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("write response: %v", err)
	}
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(map[string]string{"error": msg}); err != nil {
		log.Printf("write error response: %v", err)
	}
}

// markActive records user activity for the idle ticker.
func (s *Server) markActive() {
	s.idleMu.Lock()
	s.lastActivity = time.Now()
	s.idleMu.Unlock()
}

// idleFor reports how long the user has been inactive.
func (s *Server) idleFor() time.Duration {
	s.idleMu.Lock()
	defer s.idleMu.Unlock()
	return time.Since(s.lastActivity)
}

// persist writes the brain and the configuration, logging failures rather
// than swallowing them.
func (s *Server) persist() {
	if err := s.brain.Save(s.brainFile); err != nil {
		log.Printf("save brain: %v", err)
		return
	}
	if err := config.Save(s.configFile); err != nil {
		log.Printf("save config: %v", err)
	}
}

// lang reports the active UI language.
func lang() string {
	if i18n.Current() == i18n.RU {
		return "ru"
	}
	return "en"
}

// handleHealth is a cheap liveness probe used by the frontend to tell whether
// the server is still reachable.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "GET required")
		return
	}
	writeJSON(w, map[string]string{"status": "ok"})
}
