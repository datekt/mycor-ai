package web

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"mycor/internal/config"
	"mycor/internal/i18n"
)

// newTestServer builds a server backed by throw-away files.
func newTestServer(t *testing.T) (*Server, http.Handler) {
	t.Helper()
	dir := t.TempDir()

	srv, err := NewServer(filepath.Join(dir, "brain.gob"), filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	config.Reset()
	t.Cleanup(config.Reset)
	i18n.Set(i18n.EN)
	return srv, srv.routes()
}

// makeIdle rewinds the activity clock so the idle endpoint considers the user
// inactive.
func makeIdle(srv *Server) {
	srv.idleMu.Lock()
	srv.lastActivity = time.Now().Add(-2 * time.Hour)
	srv.idleMu.Unlock()
}

func postJSON(t *testing.T, h http.Handler, path string, body interface{}) *httptest.ResponseRecorder {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			t.Fatal(err)
		}
	}
	req := httptest.NewRequest(http.MethodPost, path, &buf)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func getJSON(t *testing.T, h http.Handler, path string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func decodeBody(t *testing.T, rec *httptest.ResponseRecorder, dst interface{}) {
	t.Helper()
	if err := json.Unmarshal(rec.Body.Bytes(), dst); err != nil {
		t.Fatalf("decode response %q: %v", rec.Body.String(), err)
	}
}

func TestHealth(t *testing.T) {
	_, h := newTestServer(t)
	rec := getJSON(t, h, "/api/health")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var body map[string]string
	decodeBody(t, rec, &body)
	if body["status"] != "ok" {
		t.Errorf("status = %q, want ok", body["status"])
	}
}

func TestChatRequiresPost(t *testing.T) {
	_, h := newTestServer(t)
	rec := getJSON(t, h, "/api/chat")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

func TestChatRejectsEmptyMessage(t *testing.T) {
	_, h := newTestServer(t)
	rec := postJSON(t, h, "/api/chat", map[string]string{"message": "   "})
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestChatRejectsMalformedBody(t *testing.T) {
	_, h := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"message":`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

// A huge payload must be refused instead of being buffered in full.
func TestRequestBodySizeIsCapped(t *testing.T) {
	_, h := newTestServer(t)
	huge := strings.Repeat("x", maxBodyBytes*2)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"message":"`+huge+`"}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code == http.StatusOK {
		t.Error("an oversized body should be rejected")
	}
}

func TestChatAndTrainFlow(t *testing.T) {
	srv, h := newTestServer(t)

	rec := postJSON(t, h, "/api/chat", map[string]string{"message": "привет"})
	if rec.Code != http.StatusOK {
		t.Fatalf("chat status = %d: %s", rec.Code, rec.Body.String())
	}

	rec = postJSON(t, h, "/api/train", map[string]string{"prompt": "привет", "target": "мир"})
	if rec.Code != http.StatusOK {
		t.Fatalf("train status = %d: %s", rec.Code, rec.Body.String())
	}
	var trained chatResponse
	decodeBody(t, rec, &trained)
	if !trained.HasLoss {
		t.Error("expected hasLoss to be true")
	}

	if srv.Brain().VocabularyLen() < 3 {
		t.Errorf("vocabulary = %d, want >= 3", srv.Brain().VocabularyLen())
	}
}

func TestTrainRejectsEmptyTarget(t *testing.T) {
	_, h := newTestServer(t)
	rec := postJSON(t, h, "/api/train", map[string]string{"prompt": "a", "target": ""})
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestUndoEndpoint(t *testing.T) {
	_, h := newTestServer(t)
	postJSON(t, h, "/api/train", map[string]string{"prompt": "a", "target": "b"})

	rec := postJSON(t, h, "/api/undo", map[string]string{})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var body map[string]bool
	decodeBody(t, rec, &body)
	if !body["ok"] {
		t.Error("expected the training step to be undone")
	}

	rec = postJSON(t, h, "/api/undo", map[string]string{})
	decodeBody(t, rec, &body)
	if body["ok"] {
		t.Error("second undo should report failure")
	}
}

func TestStatsEndpoint(t *testing.T) {
	_, h := newTestServer(t)
	postJSON(t, h, "/api/train", map[string]string{"prompt": "a", "target": "b"})

	rec := getJSON(t, h, "/api/stats")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var stats statsResponse
	decodeBody(t, rec, &stats)
	if stats.Vocabulary < 3 {
		t.Errorf("vocabulary = %d, want >= 3", stats.Vocabulary)
	}
	if stats.Parameters == 0 {
		t.Error("expected parameters > 0")
	}
	if stats.Lang == "" {
		t.Error("expected a language in stats")
	}
	if stats.BrainPath == "" {
		t.Error("expected the brain path in stats")
	}
}

func TestConfigEndpointUpdatesAndEchoes(t *testing.T) {
	_, h := newTestServer(t)
	rec := postJSON(t, h, "/api/config", map[string]float64{"temperature": 1.25})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	var echo config.Config
	decodeBody(t, rec, &echo)
	if echo.Temperature != 1.25 {
		t.Errorf("temperature = %v, want 1.25", echo.Temperature)
	}
	if config.Get().Temperature != 1.25 {
		t.Error("configuration was not applied")
	}
}

func TestConfigEndpointGetReturnsActiveConfig(t *testing.T) {
	_, h := newTestServer(t)
	postJSON(t, h, "/api/config", map[string]float64{"temperature": 1.2})

	rec := getJSON(t, h, "/api/config")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var cfg config.Config
	decodeBody(t, rec, &cfg)
	if cfg.Temperature != 1.2 {
		t.Errorf("temperature = %v, want 1.2", cfg.Temperature)
	}
}

func TestConfigEndpointClampsOutOfRange(t *testing.T) {
	_, h := newTestServer(t)
	rec := postJSON(t, h, "/api/config", map[string]float64{"temperature": 99})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if got := config.Get().Temperature; got != config.MaxTemperature {
		t.Errorf("temperature = %v, want clamp to %v", got, config.MaxTemperature)
	}
}

func TestConfigEndpointTogglesIdle(t *testing.T) {
	_, h := newTestServer(t)
	rec := postJSON(t, h, "/api/config", map[string]interface{}{"idleEnabled": false})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if config.Get().IdleEnabled {
		t.Error("idle should be disabled")
	}
}

func TestResetClearsBrainAndConfig(t *testing.T) {
	srv, h := newTestServer(t)
	postJSON(t, h, "/api/chat", map[string]string{"message": "a"})
	postJSON(t, h, "/api/train", map[string]string{"prompt": "a", "target": "b"})
	postJSON(t, h, "/api/config", map[string]float64{"temperature": 1.3})

	rec := postJSON(t, h, "/api/reset", map[string]string{})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if srv.Brain().VocabularyLen() != 1 {
		t.Errorf("vocabulary = %d, want 1 after reset", srv.Brain().VocabularyLen())
	}
	if len(srv.Brain().SessionHistory()) != 0 {
		t.Error("history should be cleared by reset")
	}
	if config.Get().Temperature != config.DefaultTemperature {
		t.Error("configuration should be reset to defaults")
	}
}

func TestHistoryEndpoint(t *testing.T) {
	_, h := newTestServer(t)
	// Train first so the model actually produces a reply to record.
	postJSON(t, h, "/api/train", map[string]string{"prompt": "привет мир", "target": "как дела"})
	postJSON(t, h, "/api/chat", map[string]string{"message": "привет"})

	rec := getJSON(t, h, "/api/history")
	var history []struct {
		Role string `json:"role"`
		Text string `json:"text"`
	}
	decodeBody(t, rec, &history)
	if len(history) < 2 {
		t.Fatalf("expected user and ai entries, got %d", len(history))
	}
	// The teacher entry from training precedes the chat turn, so look for the
	// roles instead of assuming a fixed order.
	var sawUser, sawAI bool
	for _, msg := range history {
		switch msg.Role {
		case "user":
			sawUser = true
			if msg.Text != "привет" {
				t.Errorf("user text = %q, want %q", msg.Text, "привет")
			}
		case "ai":
			sawAI = true
		}
	}
	if !sawUser {
		t.Error("history is missing the user turn")
	}
	if !sawAI {
		t.Error("history is missing the ai turn")
	}
}

func TestSaveEndpointWritesFiles(t *testing.T) {
	srv, h := newTestServer(t)
	postJSON(t, h, "/api/train", map[string]string{"prompt": "a", "target": "b"})

	rec := postJSON(t, h, "/api/save", map[string]string{})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if _, err := os.Stat(srv.brainFile); err != nil {
		t.Errorf("brain file not written: %v", err)
	}
	if _, err := os.Stat(srv.configFile); err != nil {
		t.Errorf("config file not written: %v", err)
	}
}

func TestSaveEndpointSurfacesErrors(t *testing.T) {
	srv, h := newTestServer(t)
	// Block the brain path with a regular file so the save genuinely fails and
	// the handler has to report it instead of swallowing the error.
	dir := t.TempDir()
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("not a directory"), 0644); err != nil {
		t.Fatal(err)
	}
	srv.brainFile = filepath.Join(blocker, "brain.gob")

	rec := postJSON(t, h, "/api/save", map[string]string{})
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", rec.Code)
	}
	var body map[string]string
	decodeBody(t, rec, &body)
	if body["error"] == "" {
		t.Error("expected an error message in the response body")
	}
}

func TestLangEndpoint(t *testing.T) {
	_, h := newTestServer(t)
	t.Cleanup(func() { i18n.Set(i18n.EN) })

	rec := postJSON(t, h, "/api/lang", map[string]string{"lang": "ru"})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if i18n.Current() != i18n.RU {
		t.Error("language should be ru")
	}

	rec = postJSON(t, h, "/api/lang", map[string]string{"lang": "de"})
	if rec.Code != http.StatusBadRequest {
		t.Errorf("unsupported language status = %d, want 400", rec.Code)
	}
	if i18n.Current() != i18n.RU {
		t.Error("an unsupported language must not change the current one")
	}
}

func TestIdleEndpointQuietWhileActive(t *testing.T) {
	_, h := newTestServer(t)
	rec := getJSON(t, h, "/api/idle")
	var body map[string]string
	decodeBody(t, rec, &body)
	if body["thought"] != "" {
		t.Errorf("expected no thought right after activity, got %q", body["thought"])
	}
}

func TestIdleEndpointDisabled(t *testing.T) {
	srv, h := newTestServer(t)
	postJSON(t, h, "/api/config", map[string]interface{}{"idleEnabled": false})
	makeIdle(srv)

	rec := getJSON(t, h, "/api/idle")
	var body map[string]string
	decodeBody(t, rec, &body)
	if body["thought"] != "" {
		t.Errorf("expected no thought while idle is disabled, got %q", body["thought"])
	}
}

/*
Import security.

/api/import used to pass whatever path the client sent straight to os.Open, so
any request that could reach the server could read arbitrary files. These tests
pin the allow-list behaviour.
*/

func TestResolveImportPathRejectsEmpty(t *testing.T) {
	if _, err := resolveImportPath("  ", importRoots()); err == nil {
		t.Error("expected an error for an empty path")
	}
}

func TestResolveImportPathRejectsNonTxt(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "secrets.env")
	if err := os.WriteFile(path, []byte("TOKEN=1"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := resolveImportPath(path, importRoots()); err != ErrImportNotText {
		t.Errorf("err = %v, want ErrImportNotText", err)
	}
}

func TestResolveImportPathRejectsTraversal(t *testing.T) {
	// Climbing all the way out of the user home directory must be refused even
	// though the target file need not exist: the check happens before open.
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home directory available")
	}
	traversal := filepath.Join(home, "..", "..", "..", "..", "escape.txt")
	if _, err := resolveImportPath(traversal, importRoots()); err != ErrImportNotAllowed {
		t.Errorf("err = %v, want ErrImportNotAllowed", err)
	}
}

func TestResolveImportPathRejectsAbsoluteOutsideRoot(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home directory available")
	}
	volume := filepath.VolumeName(home)
	if volume == "" {
		t.Skip("no volume to test against")
	}
	outside := filepath.Join(volume+string(filepath.Separator), "windows style.txt")
	if _, err := resolveImportPath(outside, importRoots()); err != ErrImportNotAllowed {
		t.Errorf("err = %v, want ErrImportNotAllowed", err)
	}
}

func TestResolveImportPathRejectsOutsideRoot(t *testing.T) {
	dir := t.TempDir()
	// The test binary's own directory is outside every import root.
	exe, err := os.Executable()
	if err == nil {
		if _, err := resolveImportPath(exe, importRoots()); err == nil {
			t.Error("a path outside the permitted roots should be rejected")
		}
	}
	_ = dir
}

func TestResolveImportPathRejectsDirectory(t *testing.T) {
	dir := t.TempDir()
	sub := filepath.Join(dir, "folder.txt")
	if err := os.Mkdir(sub, 0755); err != nil {
		t.Fatal(err)
	}
	// The temp dir is not an allowed root, so a directory there is rejected on
	// the root check first; either way it must not be readable.
	if _, err := resolveImportPath(sub, importRoots()); err == nil {
		t.Error("a directory should not be importable")
	}
}

func TestResolveImportPathRejectsNUL(t *testing.T) {
	if _, err := resolveImportPath("ok\x00.txt", importRoots()); err == nil {
		t.Error("a NUL byte in the path should be rejected")
	}
}

func TestWithinAnyRootIsComponentAware(t *testing.T) {
	root := filepath.Join(t.TempDir(), "Documents")
	inside := filepath.Join(root, "notes.txt")
	if !withinAnyRoot(inside, []string{root}) {
		t.Error("a file inside the root should be accepted")
	}
	// A sibling whose name merely starts with the root must not match.
	sibling := filepath.Join(filepath.Dir(root), "DocumentsEvil", "notes.txt")
	if withinAnyRoot(sibling, []string{root}) {
		t.Error("a prefix-sharing sibling directory must not count as inside")
	}
}

func TestImportEndpointRejectsOutsidePath(t *testing.T) {
	_, h := newTestServer(t)
	rec := postJSON(t, h, "/api/import", map[string]string{"path": "../../etc/passwd.txt"})
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestImportEndpointRequiresPost(t *testing.T) {
	_, h := newTestServer(t)
	rec := getJSON(t, h, "/api/import")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

// A long import must not stop the server answering other endpoints. This is the
// regression test for the old design, where one global mutex was held for the
// whole file and chat requests queued behind it.
func TestChatRespondsDuringImport(t *testing.T) {
	srv, h := newTestServer(t)

	dir := t.TempDir()
	// t.TempDir() is platform-dependent: on Linux it lives under /tmp, outside
	// every default import root, so the sandbox rejected the fixture and the
	// test never exercised concurrency at all. Point the allow-list at the
	// fixture directory so this test covers concurrency, not sandbox policy.
	srv.importRoots = []string{dir}

	var sb strings.Builder
	for i := 0; i < 600; i++ {
		sb.WriteString("слово")
		sb.WriteString(itoa(i))
		sb.WriteString(" партнёр\n")
		sb.WriteString("ответ")
		sb.WriteString(itoa(i))
		sb.WriteByte('\n')
	}
	path := filepath.Join(dir, "big.txt")
	if err := os.WriteFile(path, []byte(sb.String()), 0644); err != nil {
		t.Fatal(err)
	}

	importStatus := make(chan int, 1)
	importDone := make(chan struct{})
	go func() {
		defer close(importDone)
		rec := postJSON(t, h, "/api/import", map[string]string{"path": path})
		importStatus <- rec.Code
	}()

	// While the import runs, stats and chat must still answer promptly.
	deadline := time.After(3 * time.Second)
	for i := 0; i < 5; i++ {
		select {
		case <-deadline:
			t.Fatal("server stopped answering during an import")
		default:
		}
		rec := getJSON(t, h, "/api/stats")
		if rec.Code != http.StatusOK {
			t.Fatalf("stats status = %d during import", rec.Code)
		}
		time.Sleep(10 * time.Millisecond)
	}

	<-importDone
	// Assert the import actually succeeded. Without this, a path rejected by
	// the sandbox made the parameter check below fail for the wrong reason.
	if code := <-importStatus; code != http.StatusOK {
		t.Fatalf("import status = %d, want 200", code)
	}
	if srv.Brain().CountParameters() == 0 {
		t.Error("import produced no parameters")
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [12]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(buf[pos:])
}

func TestConcurrentRequests(t *testing.T) {
	_, h := newTestServer(t)

	done := make(chan struct{})
	for i := 0; i < 8; i++ {
		go func(i int) {
			defer func() { done <- struct{}{} }()
			for j := 0; j < 10; j++ {
				postJSON(t, h, "/api/train", map[string]string{
					"prompt": "a" + itoa(i),
					"target": "b" + itoa(j),
				})
				postJSON(t, h, "/api/chat", map[string]string{"message": "привет"})
				getJSON(t, h, "/api/stats")
			}
		}(i)
	}
	for i := 0; i < 8; i++ {
		select {
		case <-done:
		case <-time.After(30 * time.Second):
			t.Fatal("concurrent requests did not finish")
		}
	}
}

/*
Save debouncing.

Training used to write the whole brain to disk inside every /api/train
request. These tests pin the coalescing behaviour that replaced it.
*/

func TestDebouncerCoalescesBurstIntoOneWrite(t *testing.T) {
	var mu sync.Mutex
	writes := 0
	d := newSaveDebouncer(30*time.Millisecond, 5*time.Second, func() {
		mu.Lock()
		writes++
		mu.Unlock()
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.start(ctx)

	for i := 0; i < 50; i++ {
		d.mark()
		time.Sleep(time.Millisecond)
	}
	time.Sleep(150 * time.Millisecond)

	mu.Lock()
	got := writes
	mu.Unlock()
	if got != 1 {
		t.Errorf("writes = %d, want 1 for a single burst", got)
	}
}

func TestDebouncerWritesWithoutStart(t *testing.T) {
	done := make(chan struct{})
	d := newSaveDebouncer(20*time.Millisecond, time.Second, func() { close(done) })
	d.mark()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Error("mark() should schedule a write even before start()")
	}
}

func TestDebouncerStopFlushesPendingWrite(t *testing.T) {
	var mu sync.Mutex
	writes := 0
	d := newSaveDebouncer(time.Hour, time.Hour, func() {
		mu.Lock()
		writes++
		mu.Unlock()
	})
	d.mark()
	d.stop()

	mu.Lock()
	got := writes
	mu.Unlock()
	if got != 1 {
		t.Errorf("writes = %d, want 1 after stop flushed the pending write", got)
	}
}

func TestDebouncerStopIsIdempotent(t *testing.T) {
	var mu sync.Mutex
	writes := 0
	d := newSaveDebouncer(time.Hour, time.Hour, func() {
		mu.Lock()
		writes++
		mu.Unlock()
	})
	d.mark()
	d.stop()
	d.stop()

	mu.Lock()
	got := writes
	mu.Unlock()
	if got != 1 {
		t.Errorf("writes = %d, want exactly 1 from repeated stops", got)
	}
}

// Even under a continuous stream of marks the brain must reach disk within the
// maximum interval.
//
// The test waits for the signals rather than counting writes in a fixed
// window: under load, goroutine scheduling can stretch the interval, and the
// assertion here is that the flush happens at all, not how fast.
func TestDebouncerMaxIntervalWrites(t *testing.T) {
	writes := make(chan struct{}, 16)
	d := newSaveDebouncer(time.Hour, 50*time.Millisecond, func() {
		select {
		case writes <- struct{}{}:
		default:
		}
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.start(ctx)

	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
			}
			d.mark()
			time.Sleep(time.Millisecond)
		}
	}()
	defer close(stop)

	// Two flushes prove the max interval keeps re-arming.
	for i := 0; i < 2; i++ {
		select {
		case <-writes:
		case <-time.After(10 * time.Second):
			t.Fatalf("only %d flush(es) after %d expected", i, i+1)
		}
	}
}

// Training must not synchronously hit the disk; the debouncer owns the write.
func TestTrainDoesNotWriteSynchronously(t *testing.T) {
	srv, h := newTestServer(t)
	postJSON(t, h, "/api/train", map[string]string{"prompt": "a", "target": "b"})

	if _, err := os.Stat(srv.brainFile); err == nil {
		t.Error("training should not write the brain synchronously")
	}
}
