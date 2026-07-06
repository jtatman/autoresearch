# Grand Bible Archetype Research — Project Context

## What This Is

An autonomous agentic research loop that maps cross-tradition archetype connections across
the Grand Bible — a compendium of every major religious text, apocrypha, Zoroastrian,
and eastern works. The system performs BFS-driven semantic search, recording entity
co-occurrences within archetype contexts into a local SQLite database.

This is Phase 1: a deliberately simple "proof of concept" loop. It will likely exhaust
interesting connections quickly and reveal where more nuanced querying is needed.
That's the point — it will show us what Phase 2 needs to address.

---

## Repository Context

**Origin:** Forked from `karpathy/autoresearch` (March 2026). The original was an
autonomous ML hyperparameter search loop. We have completely repurposed it as a semantic
research loop over religious text. Nothing from the ML experiment remains in the active
code.

**Branch:** `grand_bible_research` — branched from `master` (228791f).

**Prior experiments (archived, not active):**
- `smollm-instruct-xlam` — 36-run LoRA fine-tuning search on SmolLM-360M (COMPLETE)
- `smollm-instruct-xlam-exp2` — reasoning-augmented exp2 design (DESIGNED, NEVER RUN)
  Both superseded: the xlam dataset is now considered antiquated.

---

## Active File Map

| File | Purpose | Editable? |
|------|---------|-----------|
| `train.py` | Current query: `ENDPOINT` + `QUERY` variables only | Yes — only these 2 lines |
| `prepare.py` | Full infrastructure: DB, API, BFS loop, novelty detection | No (bug fixes only) |
| `README.md` | Project overview | Yes |
| `program.md` | Loop instructions for the LLM governor | Yes |
| `CLAUDE.md` | This file — permanent record | Yes — update after each run |
| `research.db` | SQLite observation store (gitignored) | Auto-managed |
| `loop_state.json` | BFS queue + iteration counter (gitignored) | Auto-managed |
| `grand_bible/` | Source data + API — **READ ONLY, NEVER MODIFY** | No |

---

## Data Source: The Grand Bible API

Running at `http://localhost:8081`. Powered by Qdrant (Docker, port 6333) with two
collections over 132,362 chunks from 613 aligned chapter files.

### Endpoints

```
GET /api/health                       — status + counts (concepts:20, cross_entities:270, ...)
GET /api/search?q=<term>[&top_k=10]  — returns: concepts[], cooccurring[], top_passages[]
GET /api/browse/concepts             — 20 archetypes with seed queries and known_names by tradition
GET /api/browse/entities?limit=N     — top N cross-tradition entities with chapter/concept membership
GET /api/browse/variants             — 79 entity variant groups (e.g. Noah/Utnapishtim/Manu cluster)
```

### Search Result Shape

```json
{
  "concepts": ["flood_myth", "sacred_covenant"],   // archetype slugs this entity maps to
  "cooccurring": [{"norm": "moses", "count": 40, "skeleton": "MSS"}, ...],
  "top_passages": [{"chapter": "genesis", "score": 0.629, "text": "..."}]
}
```

### The 20 Archetypes

`flood_myth`, `solar_deity`, `creation_myth`, `dying_rising_god`, `sacred_fire`,
`descent_to_underworld`, `divine_law`, `world_mountain`, `cosmic_tree`, `virgin_birth`,
`afterlife_judgment`, `end_times`, `sacred_sacrifice`, `divine_messenger`,
`sacred_text_revelation`, `trickster`, `primordial_waters`, `sacred_covenant`,
`paradise_garden`, `seven_heavens`

---

## prepare.py — Infrastructure Design

### SQLite Schema

```sql
observations(id, iteration, endpoint, query, co_entity, concept, chapter, score, passage, discovered_at)
  UNIQUE INDEX on (query, co_entity, chapter)  ← fast duplicate check

loop_runs(id, started_at, ended_at, seed_query, total_iterations, new_observations, exit_reason)
```

### Loop Logic

1. On first run: fetch all 20 archetype `queries` from `/api/browse/concepts` → seed BFS queue
   as `[query_string, parent_archetype_slug]` pairs (concept propagates through BFS expansion)
2. Each iteration: dequeue next `[query, concept]`, call `/api/search`, extract co-occurring
   entities from result
3. For each `(query, co_entity, chapter)` triple not in DB → store as new observation;
   enqueue `[co_entity, concept]` for future search (BFS expansion)
4. Checkpoint state to `loop_state.json` every 25 iterations for crash recovery

### Exit Conditions

| Code | Condition |
|------|-----------|
| (a) | 10 consecutive iterations with zero new observations |
| (b) | Single iteration exceeds 30 seconds wall clock |
| (c) | 1,000 total iterations reached |

### Novelty Detection (Phase 1)

Pure set membership: `(query, co_entity, chapter)` triple not in DB → NEW.
The Qdrant cosine scores are stored for record-keeping.

**Phase 2 hook (not yet implemented):** Compare new passage score against existing
observations for the same entity. If score differs by > threshold, it's a genuinely
different semantic context even if the nominal triple exists. This is where a lightweight
LLM call (or local model) would provide value.

---

## train.py — The Tiny Rewritable Script

```python
import prepare

ENDPOINT = "/api/search"   # ← rewrite this
QUERY    = "flood myth"    # ← rewrite this

if __name__ == "__main__":
    prepare.run_loop(ENDPOINT, QUERY)
```

Between thematic runs, only `ENDPOINT` and `QUERY` change. The loop logic never changes.

---

## Run History

| Run | Seed Query | Exit | Iterations | New Obs | Notes |
|-----|-----------|------|-----------|---------|-------|
| test-1 | flood myth | killed (head pipe) | ~50 | ~2,100 | Pre-fix, concept field blank |
| test-2 | flood myth | killed (timeout 20s) | ~12 | ~510 | Post-fix, concepts correct |
| — | — | — | — | — | Real run not yet started |

Delete `research.db` and `loop_state.json` to start fresh. Both gitignored.

---

## GitHub Authentication

`gh` CLI authenticated as `jtatman`. Remote: `https://github.com/jtatman/autoresearch`.

`.env` tokens (gitignored):

| Variable | Account | Purpose |
|----------|---------|---------|
| `GITHUB_TOKEN` | `jtatman` | Primary token for this repo |
| `TATMANTECH_GITHUB_API_TOKEN` | `tatmantech` | Spare — other repos/orgs |
| `TATMANTECH_GITHUB_API_KEY2` | `tatmantech` | Spare — other repos/orgs |

`gh` is v2.4.0 (Ubuntu package, 2022). Functional but old; upgrade from
https://github.com/cli/cli/releases if newer features are needed.

---

## Phase 2 Preview

When Phase 1 exhausts interesting connections (likely quickly — the BFS will saturate
the 270 known cross-tradition entities fast), Phase 2 options:

1. **Direct Qdrant access** — bypass the API; query `chapters_dense` and `chapters_colbert`
   collections directly for more complex relationship queries. Collection details in
   `~/autoresearch/grand_bible/CLAUDE.md`.

2. **Cosine score novelty** — add LLM judgment on score distance (described above).

3. **New archetype seeds** — edit `~/autoresearch/grand_bible/data/concepts.json` and
   re-run the pipeline's step 12–14 to generate new entity clusters. Then resume here.
   Note: this modifies the grand_bible pipeline — coordinate carefully.

4. **Richer queries** — use variant groups and entity browse endpoints to explore
   cross-tradition names for the same archetype figure (e.g. Noah/Utnapishtim/Manu/Ziusudra).

---

## Hardware

GTX 1070, 8 GB VRAM (Pascal, compute 6.1). CUDA 12.6 max. The research loop itself
uses no GPU — it's pure Python + SQLite + HTTP. GPU only relevant for future
embedding or local LLM calls.

---

## Conventions

- `uv pip install` only — never `uv add` (breaks torch pin)
- Commit after each meaningful thematic run
- `grand_bible/` is a read-only symlink — never write to it
- `research.db` and `loop_state.json` are local state — never commit
