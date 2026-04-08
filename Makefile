# ==============================================================================
# ML Experiment Scaffold — Makefile
# Usage: make <target> [CONFIG=configs/grpo.yaml]
# ==============================================================================

.DEFAULT_GOAL := help
CONDA_RUN := conda run -n mlenv
CONFIG ?= configs/sft.yaml

.PHONY: help test validate-data dry-run baseline train resume eval compare train-seeds lint clean clean-checkpoints

help:
	@echo ""
	@echo "ML Experiment Scaffold — available targets"
	@echo "─────────────────────────────────────────────"
	@echo "  test              Run unit tests (no GPU needed)"
	@echo "  validate-data     Check dataset schema against config"
	@echo "  dry-run           10 steps, no W&B, log VRAM"
	@echo "  baseline          Evaluate base model on standard tasks"
	@echo "  train             Full training run"
	@echo "  resume            Resume from last checkpoint"
	@echo "  eval              Evaluate trained model"
	@echo "  compare           Delta table: baseline vs experiment"
	@echo "  train-seeds       Train+eval across seeds 42/0/1"
	@echo "  lint              Ruff check + format"
	@echo "  clean             Remove caches and results"
	@echo "  clean-checkpoints Delete all checkpoints (interactive)"
	@echo ""
	@echo "  Override config:  make train CONFIG=configs/grpo.yaml"
	@echo ""

# ── Quality ─────────────────────────────────────────────────────────────────
test:
	$(CONDA_RUN) pytest tests/ -v --tb=short

lint:
	$(CONDA_RUN) ruff check src/ scripts/ tests/ --fix && $(CONDA_RUN) ruff format src/ scripts/ tests/

# ── Data ────────────────────────────────────────────────────────────────────
validate-data:
	$(CONDA_RUN) python -m scripts.validate_data --config $(CONFIG)

# ── Training ────────────────────────────────────────────────────────────────
dry-run:
	$(CONDA_RUN) python -m src.train --config $(CONFIG) --dry-run

train:
	$(CONDA_RUN) python -m src.train --config $(CONFIG)

resume:
	$(CONDA_RUN) python -m src.train --config $(CONFIG) --resume

# ── Evaluation ──────────────────────────────────────────────────────────────
baseline:
	$(CONDA_RUN) python -m src.eval --model $$(python -c "import yaml; print(yaml.safe_load(open('$(CONFIG)'))['model']['name'])") --tasks arc_easy,hellaswag

eval:
	$(CONDA_RUN) python -m src.eval --model checkpoints --tasks arc_easy,hellaswag

compare:
	$(CONDA_RUN) python -m scripts.compare results/baseline results/final --format markdown

train-seeds:
	$(CONDA_RUN) python -m scripts.run_seeds --seeds 42 0 1 --config $(CONFIG)

# ── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	rm -rf __pycache__ tests/__pycache__ .pytest_cache results/ logs/

clean-checkpoints:
	@read -p "Delete all checkpoints? (yes/no): " c && [ "$$c" = "yes" ] && rm -rf checkpoints/ || echo "Aborted."
