# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-04-08

### Added
- Generic `Registry` class (`src/registry.py`) with decorator-based registration, key lookup, overwrite warnings, and introspection
- `reward_registry` instance — reward functions now register via `@reward_registry.register("name")`
- 10 new registry tests (`tests/test_registry.py`) including integration test with reward registry
- Back-filled `model_card.md` with bible-ai-assistant SFT+ORPO training data and evaluation results

### Changed
- Refactored `src/rewards.py` to use `Registry` instead of inline dict — extensible without modifying the build function

## [0.2.0] — 2026-04-08

### Added
- MIT LICENSE file
- Project CLAUDE.md with architecture, constraints, and code standards
- GitHub issue templates (bug report, feature request)
- GitHub pull request template with checklist
- `uv.lock` for reproducible CI installs

### Changed
- CI: upgraded `actions/checkout` v4→v5, `astral-sh/setup-uv` v5→v6 (Node.js 24)
- Pre-commit: ruff hook v0.4.8→v0.15.6
- `test_compare.py`: moved `import pytest` to top of file (was at bottom with noqa)
- `train.py`: moved `import wandb` to module-level (was duplicated inside function)
- `vram.py`: replaced bare `except Exception: pass` with `log.debug(exc_info=True)`

## [0.1.0] — 2026-04-08

### Added
- Config system with YAML inheritance (`_base` key)
- Platform detection: Windows, Blackwell sm_120, Flash Attention
- Model factory: LLM (Unsloth/HF), vision (timm/torchvision), tabular (custom nn.Module)
- Unified training entrypoint with `--dry-run`, `--resume`, `--seed`, `--config`
- SFT, ORPO, GRPO/DAPO, vision, and tabular training methods
- GRPO reward functions: correctness (numeric extraction) and format (thinking + boxed)
- lm-eval wrapper with vLLM backend support and safe batch_size defaults
- Training callbacks: VRAMTracker, TrainingHealthMonitor, LocalMetricsLogger
- Utility scripts: validate_data, compare (markdown delta table), run_seeds (3-seed averaging)
- Tier 1 optimizations baked in: fused AdamW, data loader tuning, non-reentrant gradient checkpointing
- Tier 2: Unsloth fp8 loading support via TorchAO
- Tier 3: torch.compile opt-in flag (apply after LoRA)
- Tier 4: DeepSpeed ZeRO-2/3 configs for large models
- CI workflow (ruff + pytest), pre-commit hooks, dependabot
- Makefile with all standard targets

[Unreleased]: https://github.com/omnipotence-eth/ml-experiment-scaffold/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/omnipotence-eth/ml-experiment-scaffold/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/omnipotence-eth/ml-experiment-scaffold/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/omnipotence-eth/ml-experiment-scaffold/releases/tag/v0.1.0
