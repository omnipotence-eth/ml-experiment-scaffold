# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/omnipotence-eth/ml-experiment-scaffold/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/omnipotence-eth/ml-experiment-scaffold/releases/tag/v0.1.0
