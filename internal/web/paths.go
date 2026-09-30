package web

import (
	"os"
	"path/filepath"
)

const appDirName = "MYCOR"
const brainFileName = "history.json"

/*
resolveBrainPath returns the absolute path to the trained brain inside the
operating system's per-user configuration directory:

  - Windows: %AppData%\MYCOR\history.json
  - Linux:   $XDG_CONFIG_HOME/MYCOR/history.json  or  ~/.config/MYCOR/history.json
  - macOS:   ~/Library/Application Support/MYCOR/history.json

The MYCOR subdirectory is created if it does not exist. If the config
directory cannot be resolved, the caller falls back to the current working
directory so that the program still runs.
*/
func resolveBrainPath() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	appDir := filepath.Join(dir, appDirName)
	if err := os.MkdirAll(appDir, 0755); err != nil {
		return "", err
	}
	return filepath.Join(appDir, brainFileName), nil
}
