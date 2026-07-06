# Grand Bible Archetype Research

An autonomous loop that maps cross-tradition archetype connections across the Grand Bible —
a compendium of all major religious texts, apocrypha, and eastern works.

The system uses a semantic search API over 132,362 Qdrant-indexed chunks from 613 aligned
chapter files to discover and record entity co-occurrences within archetype contexts,
building a growing graph of cross-tradition connections with each iteration.

## Quick Start

```bash
# Run the research loop (up to 1000 iterations, auto-exits on exhaustion or timeout)
python train.py

# Summarize what's been found so far
python prepare.py
```

## Architecture

| File | Role |
|------|------|
| `train.py` | Current query config (ENDPOINT + QUERY). Rewritten each thematic run. |
| `prepare.py` | Stable infrastructure: DB, API, novelty detection, BFS loop. Do not modify. |
| `research.db` | SQLite observations store (gitignored — local state). |
| `loop_state.json` | BFS queue + iteration counter for crash recovery (gitignored). |
| `CLAUDE.md` | Full project context and session restart guide. |
| `program.md` | Loop instructions for the LLM governor (Claude). |
| `grand_bible/` | Symlinked source data and pipeline — **read-only, never modify**. |

## Data Source

`~/autoresearch/grand_bible` — the Grand Bible NLP pipeline and API (read-only symlink).
Qdrant running locally on Docker (port 6333). REST API on `http://localhost:8081`.

## Loop Exit Conditions

| Condition | Trigger |
|-----------|---------|
| (a) Exhausted | 10 consecutive iterations with zero new observations |
| (b) Timeout | Any single iteration exceeds 30 seconds |
| (c) Hard stop | 1,000 total iterations reached |
