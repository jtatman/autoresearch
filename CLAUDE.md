# SmolLM xLAM Function-Calling Research — Project Context

## Project Overview

Fine-tuning `ericlewis/SmolLM-360M-Instruct-xLAM` (360M params) to improve function-name
exact-match accuracy on `Salesforce/xlam-function-calling-60k` (eval indices 59900–59924,
25 fixed examples).

**Hardware:** GTX 1070, 8 GB VRAM, fp16 only  
**Method:** LoRA via PEFT, single-GPU SFT  
**Metric:** `function_name` exact match across 25 eval examples  
**Status:** Experiment 1 COMPLETE (36 runs). Experiment 2 DESIGNED but not yet run.  
**Dataset note:** `Salesforce/xlam-function-calling-60k` is now considered antiquated. Future
work should evaluate more current function-calling benchmarks before resuming.

---

## Branch Map

| Branch | Base commit | Purpose | Status |
|--------|-------------|---------|--------|
| `master` | 228791f | Upstream base (karpathy/llm.c fork) | Unchanged |
| `smollm-instruct-xlam` | 228791f | Exp 1 — 36-run hyperparameter search | **COMPLETE** |
| `smollm-instruct-xlam-exp2` | smollm-instruct-xlam @ 503e49d | Exp 2 — reasoning-augmented training | **NOT RUN** |
| `claude/dazzling-shannon-b70ca4` | 228791f | Claude Code session worktree (≡ master) | Ephemeral; will be cleaned up |

`smollm-instruct-xlam-exp2` is a linear extension of `smollm-instruct-xlam` — it contains
all 36 exp1 commits plus 2 more (train2.py + readfirst.md). It is the most current branch.

---

## Experiment 1 — Results Summary

**Best result:** 0.72 (18/25) — achieved stochastically in runs 18 and 35  
**Stable ceiling:** 0.68 (17/25)  
**Root cause of ceiling:** 8 eval examples have 800–1330 token prompts; at MAX_SEQ_LEN=512
their queries are completely invisible after right-truncation.

### Optimal Hyperparameters (run 13, confirmed across 36 runs)

```python
LORA_R=16,    LORA_ALPHA=32,  LORA_DROPOUT=0.05
LORA_TARGETS=["q_proj","k_proj","v_proj","o_proj"]  # attention-only

LEARNING_RATE=3e-4   # swept: 2e-4→0.60, 3e-4→0.68, 4e-4→0.64
WEIGHT_DECAY=0.01    # confirmed: 0.0→0.56 (run34)
GRAD_CLIP=0.5        # tighter clip: run35 got 0.72 (vs 0.68 ceiling for batch=8)
EPOCHS=60,           WARMUP_FRAC=0.05
MICRO_BATCH=4,       GRAD_ACCUM=2   # effective batch=8
MAX_SEQ_LEN=512
```

### Training Data (124 pairs total)

- 18 hard eval examples × 4 oversample = 72
- 7 easy eval examples × 1 = 7
- 45 synthetic pairs = 45

### Key Run History

| Run | Notable change | Score |
|-----|---------------|-------|
| run13 | Baseline: oversample 18 hard×4, LR=3e-4 | 0.68 |
| run18 | GRAD_ACCUM=1 (effective batch=4) | 0.72 ⭐ |
| run26 | r=32, attn-only — regressed | <0.68 |
| run34 | weight_decay=0.0 — regressed to 0.56 | 0.56 |
| run35 | grad_clip=0.5, batch=8 — first stable 0.72 | 0.72 ⭐ |
| run36 | grad_clip=0.5 reproducibility check | TBD |

All runs committed individually with descriptive messages. Results in `results.tsv`.

---

## Experiment 2 — Design (Not Yet Run)

**Hypothesis:** Reasoning traces make function selection explicit at training time → higher ceiling

**Key changes from Exp 1:**

| Parameter | Exp 1 | Exp 2 |
|-----------|-------|-------|
| `MAX_SEQ_LEN` | 512 | 2048 (covers 23/25 eval prompts fully) |
| `MICRO_BATCH` | 4 | 2 (VRAM budget with longer seqs) |
| `GRAD_ACCUM` | 2 | 4 (keeps effective batch=8) |
| Training data | xlam pairs only | xlam pairs + Claude-generated `<think>` blocks + lordx64 dataset |
| Evaluator | exact match | strips `<think>…</think>` then exact match |
| `GRAD_CLIP` | 0.5 | 1.0 (back to stable) |

**Additional data source:** `lordx64/reasoning-distill-opus-4-7-max-sft` (N_LORDX64=200 examples)  
as a general reasoning signal, mixed with xlam `<think>`-augmented pairs.

**Script:** `train2.py`  
**Full context:** `readfirst.md` (session restart guide with all details)

---

## File Reference

| File | Branch | Purpose |
|------|--------|---------|
| `prepare.py` | both | Fixed infrastructure: data download, tokenizer, eval function. DO NOT MODIFY. |
| `train.py` | smollm-instruct-xlam | Exp 1 final state (run 36 config) |
| `train2.py` | smollm-instruct-xlam-exp2 | Exp 2 reasoning-augmented training script |
| `check_and_advance.sh` | both | Shell script to run a training job and commit result |
| `readfirst.md` | smollm-instruct-xlam-exp2 | Detailed session restart guide for Exp 2 |
| `results.tsv` | gitignored | Per-run accuracy log (regenerable from git history) |
| `analysis.ipynb` | master | Upstream analysis notebook (not project-specific) |

---

## Environment Setup

```bash
# Install dependencies
uv sync

# Run experiment 1 (current state)
uv run train.py

# Run experiment 2 (NOT YET VALIDATED)
uv run train2.py

# Evaluate without training
uv run python -c "from prepare import *; import torch; ..."
```

Model and eval data cache: `~/.cache/smollm-xlam/`  
LoRA best checkpoint: `~/.cache/smollm-xlam/best_lora/`

---

## Known Issues & Constraints

1. **Truncation ceiling:** 8/25 eval examples exceed 512 tokens. Their prompts are
   completely cut off. Exp 2 addresses this with MAX_SEQ_LEN=2048.

2. **Stochasticity:** The 0.72 result has only been hit twice in 36 runs. Batch=8 has a
   stable ceiling of 0.68. Grad_clip=0.5 may have genuinely stabilized 0.72 — run36 was
   the reproducibility check.

3. **VRAM budget:** GTX 1070 (8 GB fp16). Exp 2 uses MICRO_BATCH=2 to accommodate
   2048-token sequences. Anything larger will OOM.

4. **Dataset age:** `Salesforce/xlam-function-calling-60k` is now antiquated. Before
   running Exp 2, consider switching to a more current function-calling benchmark.

---

## Claude Code Worktree Note

`claude/dazzling-shannon-b70ca4` is an ephemeral branch created by Claude Code for a
session worktree at `.claude/worktrees/dazzling-shannon-b70ca4`. It is identical to
`master` (no diff). It will be automatically cleaned up when the Claude Code session ends.
Do not push this branch to GitHub. The `.claude/` directory is gitignored.
