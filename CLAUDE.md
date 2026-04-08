# ML Experiment Scaffold — Claude Code Instructions

## Project Overview

GitHub template repo for reproducible ML experiments on a single GPU (RTX 5070 Ti, 16GB VRAM, Blackwell sm_120). Supports LLM fine-tuning (SFT, ORPO, GRPO), vision models, and tabular NNs.

## Architecture

```
configs/*.yaml → src/config.py → src/train.py → src/models.py + src/data.py
                                              → src/callbacks.py
                                              → src/eval.py (lm-eval)
```

- **Config system**: YAML with `_base` inheritance. `configs/base.yaml` has all defaults.
- **Model factory**: `src/models.py` — LLM (Unsloth→HF fallback), vision (timm→torchvision), tabular (dynamic import)
- **Training**: `src/train.py` dispatches to `_train_sft/_train_orpo/_train_grpo/_train_vision`
- **Callbacks**: `src/callbacks.py` — VRAMTracker, TrainingHealthMonitor, LocalMetricsLogger
- **Rewards**: `src/rewards.py` — GRPO correctness + format reward functions with registry

## Key Constraints

- `bf16=True` always — never fp16 on Blackwell
- `max_seq_length: 2048` — 4096 OOMs on 16GB
- GRPO: `loss_type="dapo"`, `beta=0.0` — no KL penalty, no reference model
- lm-eval: `--batch_size 4` — never `auto` (OOMs on generate_until tasks)
- torch.compile must apply AFTER LoRA (module name `_orig_mod.` prefix mismatch)
- `fp8=True` in TRL configs raises TypeError — use Unsloth `load_in_fp8=True` instead

## Commands

```bash
make test                              # Unit tests (no GPU)
make lint                              # Ruff check + format
make dry-run CONFIG=configs/grpo.yaml  # 10 steps, no W&B, VRAM logged
make train CONFIG=configs/grpo.yaml    # Full training
make eval                              # lm-eval on checkpoints
make compare                           # Markdown delta table
make train-seeds                       # 3-seed mean±std
```

## Testing

- 48 tests in `tests/` — all run without GPU
- Callbacks use lazy `_get_trainer_callback_cls()` to avoid hard torch dependency
- Run: `conda run -n mlenv pytest tests/ -v --tb=short`

## Code Standards

- `from __future__ import annotations` in every module
- `logging.getLogger(__name__)` — never `print()`
- `logger.info("msg %s", value)` — never f-strings in log calls
- Specific exceptions only — no bare `except:`
- Type hints on all public functions
- Ruff for lint/format, line length 100

## Config Inheritance

Child configs use `_base: base.yaml`. Deep merge: child overrides win for leaf values, dicts merge recursively. Only one level of inheritance.

## Performance Tiers

1. **Free speed** (always on): fused AdamW, tuned data loading, non-reentrant grad checkpointing
2. **Unsloth fp8**: `model.load_in_fp8: true` — 60% less VRAM
3. **torch.compile**: `compile: true` — for runs >2000 steps
4. **DeepSpeed**: `deepspeed_config: "ds_configs/zero2_offload.json"` — 13B+ models
