package web

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	appDirName     = "MYCOR"
	brainFileName  = "brain.gob"
	configFileName = "config.json"
)

/*
resolveDataPaths returns the absolute paths of the brain and configuration
files inside the operating system's per-user configuration directory:

  - Windows: %AppData%\MYCOR\
  - Linux:   $XDG_CONFIG_HOME/MYCOR/  or  ~/.config/MYCOR/
  - macOS:   ~/Library/Application Support/MYCOR/

The MYCOR subdirectory is created if it does not exist.
*/
func resolveDataPaths() (brainPath, configPath string, err error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", "", fmt.Errorf("resolve user config dir: %w", err)
	}
	appDir := filepath.Join(dir, appDirName)
	if err := os.MkdirAll(appDir, 0755); err != nil {
		return "", "", fmt.Errorf("create app dir: %w", err)
	}
	return filepath.Join(appDir, brainFileName), filepath.Join(appDir, configFileName), nil
}

/*
importRoots lists the directories an import may read from.

/api/import used to accept any path the client sent, so a request that reached
the server could read and parse arbitrary files on the machine. Restricting
imports to the user's own documents, downloads and desktop covers the actual
use case (training on a text file the user has) while removing the arbitrary
file read.
*/
func importRoots() []string {
	var roots []string
	if home, err := os.UserHomeDir(); err == nil {
		for _, sub := range []string{"Documents", "Downloads", "Desktop"} {
			roots = append(roots, filepath.Join(home, sub))
		}
		roots = append(roots, home)
	}
	if cwd, err := os.Getwd(); err == nil {
		roots = append(roots, cwd)
	}
	return roots
}

var (
	// ErrImportNotAllowed is returned when a path is outside the allow-list.
	ErrImportNotAllowed = errors.New("path is outside the permitted import directories")
	// ErrImportNotText is returned for files that are not plain .txt.
	ErrImportNotText = errors.New("only .txt files can be imported")
	// ErrImportNotRegular is returned for directories, devices or sockets.
	ErrImportNotRegular = errors.New("not a regular file")
)

// resolveImportPath validates a user supplied import path and returns the
// cleaned absolute path.
//
// Checks applied, in order: non-empty, no NUL byte, absolute-able, .txt
// extension, symlink-free resolution, located inside an allowed root, and a
// regular file.
func resolveImportPath(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", errors.New("empty path")
	}
	if strings.ContainsRune(raw, 0) {
		return "", errors.New("path contains a NUL byte")
	}

	abs, err := filepath.Abs(raw)
	if err != nil {
		return "", fmt.Errorf("invalid path: %w", err)
	}
	abs = filepath.Clean(abs)

	if !strings.EqualFold(filepath.Ext(abs), ".txt") {
		return "", ErrImportNotText
	}

	// Resolve symlinks so a link inside an allowed directory cannot point at a
	// file outside of it.
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		abs = resolved
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("resolve path: %w", err)
	}

	if !withinAnyRoot(abs, importRoots()) {
		return "", ErrImportNotAllowed
	}

	info, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("open file: %w", err)
	}
	if !info.Mode().IsRegular() {
		return "", ErrImportNotRegular
	}
	return abs, nil
}

// withinAnyRoot reports whether target is inside at least one root. The
// comparison is path-component aware, so "C:\Users\me\DocumentsEvil" is not
// considered to be inside "C:\Users\me\Documents".
func withinAnyRoot(target string, roots []string) bool {
	if len(roots) == 0 {
		return false
	}
	for _, root := range roots {
		cleanRoot, err := filepath.Abs(root)
		if err != nil {
			continue
		}
		cleanRoot = filepath.Clean(cleanRoot)
		rel, err := filepath.Rel(cleanRoot, target)
		if err != nil {
			continue
		}
		if rel == "." || (!strings.HasPrefix(rel, "..") && !filepath.IsAbs(rel)) {
			return true
		}
	}
	return false
}
