# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 4.4.x   | :white_check_mark: |
| 4.3.x   | :x:                |
| < 4.3   | :x:                |

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

MYCOR runs entirely locally and has zero third-party dependencies. The most
likely attack surfaces are:

- Malicious `history.json` files (crafted to exhaust memory or crash the loader)
- Malicious TXT files passed to `/import`
- Path traversal in `/import` or `/reset`

Reports about these areas are especially welcome.