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
| (c) | No more cycle seeds — all co-entities have been queried (DB saturated) |

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
| run-1  | flood myth | queue exhausted (Phase 1) | 476 | 16,306 | Full Phase 1 BFS complete |
| run-2  | flood myth | hard stop (MAX_ITERATIONS=1000) | 1000 | 34,205 | Phase 2 at 524/5554 (9.4%) when stopped |
| run-3  | flood myth | fully exhausted | 6121 | 17,774 | Remainder pass (536 vectors) + Phase 1 cycling added; DB saturated at 208,512 obs |

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

## Phase 2: Centroid Burndown (implemented)

When Phase 1 BFS exhausts, `run_loop()` automatically transitions to Phase 2:

1. **Centroid computation** — for each of the 20 archetypes, batch-fetch the 384-dim
   vectors of all seed chunks (from `concept_clusters.json` chunk_ids = Qdrant point IDs),
   compute normalised mean centroid via numpy.

2. **Neighbor search** — query `chapters_dense` with the centroid vector (top 300 per
   archetype). Returns semantically similar chunks not necessarily reached by the API's
   named-entity index.

3. **Entity extraction** — regex-based proper-noun extraction from chunk text
   (`extract_entities()` in prepare.py). Filters sentence starters, skip words.
   Handles "And Noah" → "Noah" correctly.

4. **`unresearched_vectors` table** — stores (entity, chapter, archetype, affinity).
   Loop processes them highest-affinity-first; marks each as `researched=1` after querying.

5. **Seamless transition** — `run_loop()` handles both phases; Phase 2 has a more lenient
   dry-streak threshold (20 vs 10) since sparse-tradition entities yield fewer API hits.

**Qdrant snapshots** (taken before Phase 2 work began):
- `grand_bible/data/snapshots/chapters_dense.snapshot` — 360 MB
- `grand_bible/data/snapshots/chapters_colbert.snapshot` — 20 GB
- Restore with `grand_bible/steps/import_qdrant.py`

## Phase 3: Archetype Candidate Scoring (implemented)

Runs automatically once after full exhaustion. Scores every entity in
`unresearched_vectors` against the existing 20 archetype centroids.

**Score formula:** `min_archetype_dist × log1p(concept_spread) × log1p(obs_freq)`
- `min_archetype_dist` — cosine distance from entity's chunk vector to nearest archetype centroid (high = novel embedding region)
- `concept_spread` — distinct archetypes entity appeared under as a co-occurrence partner in Phase 1/2 (from `observations.co_entity`)
- `obs_freq` — total times entity appeared as co-occurrence (significance weight)

Results stored in `archetype_candidates` table (5,554 candidates scored).

**Top candidates from run-1:**

| Entity | Score | C.Spread | Dist | Nearest Arch |
|--------|-------|----------|------|--------------|
| minerva | 3.75 | 17 | 0.306 | cosmic_tree |
| olympus | 2.48 | 11 | 0.262 | solar_deity |
| joshua | 2.46 | 9 | 0.285 | sacred_covenant |
| athene | 2.39 | 10 | 0.281 | trickster |
| aquila | 2.24 | 8 | 0.302 | virgin_birth |
| ziusudra | 2.16 | 6 | 0.336 | flood_myth |
| uriel | 2.06 | 9 | 0.248 | seven_heavens |
| ravana | 1.99 | 12 | 0.202 | solar_deity |
| isis | 1.55 | 6 | 0.270 | trickster |
| mani | 1.46 | 5 | 0.308 | sacred_fire |
| buddha | 1.35 | 5 | 0.262 | solar_deity |

**Interpretation:** High-scoring candidates with high `dist` and `concept_spread`
are the best new archetype prospects. Entities like `minerva` (dist=0.31,
spread=17) and `mani` (Manichaeism founder, dist=0.31, spread=5) appear in
genuinely novel embedding regions. `mainaka` (mountain that rose from the sea
for Hanuman — world_mountain territory) and `ravana` (multi-tradition demon
king) bridge multiple archetype contexts.

Before adding any as a new archetype: verify the candidate's Qdrant centroid
is meaningfully distant from ALL 20 existing centroids (not just the nearest),
and that it has representable seed chunks across ≥3 traditions.

## Phase 1 Cycling (implemented)

After Phase 2 fully exhausts, `run_loop()` automatically seeds a new Phase 1 BFS cycle
from the top 200 co-entities in the observations table that have never been used as queries
themselves — the richest unexplored nodes in the co-occurrence graph.

- `_get_cycle_seeds(db, n=200)` — selects top unqueried co-entities (length ≥ 3, excludes blanks)
- `seen` is pre-seeded from all past `query` values so BFS doesn't redundantly re-query
  already-exhausted entities; new seeds are explicitly excluded from `seen`
- Dry streaks clear the queue (rather than breaking) so they also trigger the cycling check
- Loop terminates only when `_get_cycle_seeds` returns empty — meaning every co-entity of
  sufficient length has been queried at least once

**Current state:** DB saturated at 208,512 observations across all 20 archetypes.
All 5,554 Phase 2 vectors researched. All co-entities queried. Fully exhausted.

---

## Phase 3 Preview

- **Cosine score novelty** — LLM or local model compares score distance of new observations
  against existing DB entries; detects same entity in genuinely different semantic context.
- **New archetype seeds** — edit `grand_bible/data/concepts.json`, re-run steps 12–14,
  then resume loop. (Modifies grand_bible pipeline — coordinate carefully.)
- **ColBERT reranking** — use `chapters_colbert` for more precise passage retrieval on
  Phase 2 entity candidates before deciding whether to search the API.

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
