# loopme — Agentic Research Experiments

## What This Branch Is

A fresh exploration of autonomous research loop patterns. Prior work (see `CONTEXT.md`) 

The name is intentional. The loop is the thing. Not the data, not the domain — the loop.

---

## Prior Work Summary

- Full context: `CONTEXT.md`

---

# Original files from 

## Active Files

| File | Purpose |
|------|---------|
| `CONTEXT.md` | future directions |
| `CLAUDE.md` | This file — session restart guide |
| `prepare.py` | loop infrastructure | 
| `train.py` | loop entry point |
| `loop_state.json` | json file for keeping track of iteration information |
| `README.md` | original README from master - left only for reference

---

## Constraints

- `uv pip install` only — never `uv add` (breaks torch pin)
- GPU: GTX 1070, 8 GB VRAM, CUDA 12.6, torch 2.7.1+cu126

---

## GitHub

`gh` CLI authenticated as `jtatman`. Remote: `https://github.com/jtatman/autoresearch`.
`.env` is gitignored (GITHUB_TOKEN, ANTHROPIC_API_KEY if needed).
