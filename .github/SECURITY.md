# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 5.0.x   | :white_check_mark: |
| 4.5.x   | :x:                |
| < 4.5   | :x:                |

## Reporting a Vulnerability

Do not open public issues for security problems.

Use GitHub's private vulnerability reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Provide a description, reproduction steps, and the affected version.

We will respond within 5 business days with an assessment and a plan.
If the report is confirmed, we will publish a fix and credit the reporter
in the release notes (unless you prefer to stay anonymous).

## Scope

MYCOR runs entirely locally, binds only to 127.0.0.1 on an OS-assigned
port, and has zero third-party dependencies. The most likely attack
surfaces are:

- Malicious `brain.gob` files placed in the user configuration directory
  (`%AppData%\MYCOR\` on Windows, `~/.config/MYCOR/` on Linux/macOS,
  `~/Library/Application Support/MYCOR/` on macOS), crafted to exhaust
  memory or crash the loader.
- Malicious TXT files passed to the import endpoint.
- Path traversal in the import endpoint when a user supplies an arbitrary
  file path to `/api/import`.

Reports about these areas are especially welcome.