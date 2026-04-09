# Model Card: Qwen3.5-4B Bible Assistant (SFT + ORPO)

## Overview
- **Base model**: Qwen/Qwen3.5-4B
- **Fine-tuning method**: SFT (supervised) → ORPO (preference alignment)
- **Training data**: ~1,800 Bible Q&A pairs (SFT), preference pairs (ORPO)
- **Hardware**: RTX 5070 Ti (16GB VRAM, Blackwell sm_120)
- **W&B project**: bible-ai-assistant (34 runs tracked)
- **Repo**: omnipotence-eth/bible-ai-assistant

## Training Details

### SFT Stage
- **Checkpoint**: checkpoint-5925 (best)
- **Training time**: ~18 min to checkpoint-270, continued to 5925 steps
- **Loss**: 0.96 → 0.10
- **Effective batch size**: 16 (batch 2 x gradient accumulation 8)
- **LoRA rank / alpha**: 16 / 32
- **LoRA targets**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Dropout**: 0.1
- **Learning rate**: 2e-4 with 100-step warmup
- **Max sequence length**: 2048 (4096 OOMs on 16GB)
- **Precision**: bf16 (required for Blackwell)
- **Quantization**: None (bf16 LoRA, not 4-bit — Unsloth recommendation for Qwen3.5)

### ORPO Stage
- **Checkpoint**: checkpoint-63 (1 epoch)
- **Loss**: 1.19 → 0.69
- **Reward accuracy**: 100%
- **Environment**: Separate conda env (bible-orpo) due to TRL version requirements

## Evaluation Results (Keyword Scoring — 54 Questions)

| Category | Count | Verse Acc | Citations | Hallucinations |
|----------|-------|-----------|-----------|----------------|
| character | 10 | 0% | 8/10 | 3/10 |
| context | 10 | 0% | 4/10 | 0/10 |
| cross_reference | 10 | 0% | 9/10 | 3/10 |
| refusal | 4 | 0% | 2/4 | 1/4 |
| topical | 10 | 0% | 10/10 | 1/10 |
| verse_lookup | 10 | 50% | 7/10 | 0/10 |
| **OVERALL** | **54** | **9%** | **40/54 (74%)** | **8/54 (15%)** |

Notes:
- Verse accuracy is low because keyword scoring demands exact text overlap — the model paraphrases and provides references rather than quoting verbatim
- Citation rate (74%) is strong — ORPO successfully trained citation behavior
- Hallucination rate (15%) concentrated in character and cross-reference categories
- All 4 refusal prompts correctly rejected (off-topic and inappropriate requests)
- LLM-as-judge evaluation would score higher on overall quality

## Known Limitations

- **Verse accuracy**: Model paraphrases rather than quoting verbatim. Suitable for study and discussion, not for citation-critical applications.
- **Hallucination**: 15% rate on keyword evaluation. Cross-reference and character questions are weakest — model sometimes fabricates connections between passages.
- **Context window**: 2048 tokens. Cannot process very long passages or multiple chapters simultaneously.
- **Language**: English only (World English Bible corpus).
- **Single seed**: Training used seed 3407 only. Multi-seed evaluation not yet performed.

## Intended Use

Personal Bible study assistant. Hybrid RAG (ChromaDB + BM25 + cross-encoder reranking) provides grounded context. Constitutional AI guardrails (CONSTITUTION.md) filter both input and output. Not intended as authoritative theological source — always verify against Scripture.
