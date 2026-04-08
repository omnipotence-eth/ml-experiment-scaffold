# Contributing

## Branch Strategy
- `feature/` — new features
- `fix/` — bug fixes
- `experiment/` — ML experiments
- `chore/` — maintenance

## Ship Workflow

Before every commit:

1. `ruff check . --fix && ruff format .`
2. `pytest tests/ -v --tb=short`
3. Audit: type hints, no bare except, no hardcoded secrets, no print()
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Conventional commit: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`

## PR Checklist

- [ ] Tests pass (`make test`)
- [ ] Linter clean (`make lint`)
- [ ] CHANGELOG updated
- [ ] Config changes documented
- [ ] No hardcoded secrets or API keys
