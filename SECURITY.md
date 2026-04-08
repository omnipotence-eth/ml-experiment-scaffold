# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting Vulnerabilities

Report security vulnerabilities via [GitHub private advisory](https://github.com/omnipotence-eth/ml-experiment-scaffold/security/advisories/new).

Do not open public issues for security vulnerabilities.

## Security Model

- No secrets are stored in this repository
- API keys (W&B, HuggingFace) are loaded from environment variables only
- Model weights and training data are gitignored
- Dependencies are scanned weekly via Dependabot
