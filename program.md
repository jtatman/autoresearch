# Loop Instructions — Grand Bible Archetype Research

You are the loop governor for an autonomous archetype research process over the Grand Bible.

## Your Role

The loop in `prepare.py` is fully autonomous — it runs up to 1,000 iterations without
your intervention, expanding a BFS query graph seeded from 20 archetype archetypes.
Your job is to:

1. **Start a run**: `python train.py` (redirected: `python train.py > run.log 2>&1 &`)
2. **Monitor**: `tail -f run.log` or check `python prepare.py` for a summary
3. **Review exit**: When the loop ends, read the exit reason and observation count
4. **Advance**: If results are interesting, change `ENDPOINT` and/or `QUERY` in `train.py`
   to seed a new thematic angle, then commit and run again
5. **Record**: Update `CLAUDE.md` with what was found and what comes next

## What to Rewrite in train.py

Only these two lines change between thematic runs:

```python
ENDPOINT = "/api/search"   # or /api/browse/entities, /api/browse/variants
QUERY    = "flood myth"    # the seed query; BFS expands from here
```

Good seed queries: archetype slugs (`flood_myth`, `virgin_birth`), entity names
(`noah`, `osiris`, `manu`), cross-tradition events (`dying god resurrection`).

## Novelty Standard

Phase 1: A connection is NEW if the triple `(query, co_entity, chapter)` is not already
in `research.db`. The loop stops after 10 consecutive dry iterations.

Phase 2 (future): Introduce cosine score comparison — two observations of the same entity
in different score ranges indicate genuinely different semantic contexts, even if the
nominal triple already exists. This is where an LLM call or local model would add value.

## API Endpoints

```
GET /api/search?q=<term>[&top_k=10]   — semantic search; returns concepts, cooccurring, top_passages
GET /api/browse/concepts              — all 20 archetype definitions with seed queries
GET /api/browse/entities?limit=N      — top N cross-tradition entities
GET /api/browse/variants              — all 79 entity variant groups
```

## Exit and Recovery

- `loop_state.json` persists BFS queue and iteration count; delete it to restart from scratch
- `research.db` accumulates all observations; delete it to reset the observation store
- Both files are gitignored — they are local run state only
- If the API is down: loop exits on timeout; restart with `python train.py` after API recovers

## NEVER

- Modify `prepare.py` (only bug fixes, via the human)
- Modify anything in `~/autoresearch/grand_bible/`
- Commit `research.db` or `loop_state.json`
- Run `uv add` — use `uv pip install` if a new dep is ever needed
