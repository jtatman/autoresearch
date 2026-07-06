"""
Grand Bible archetype research — infrastructure.

Manages SQLite database, API calls, novelty detection, BFS query expansion,
Qdrant centroid computation, and the autonomous research loop (Phase 1 + 2).

DO NOT MODIFY unless changing core infrastructure. Query parameters live in train.py.
"""

import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE             = "http://localhost:8081"
QDRANT_BASE          = "http://localhost:6333"
QDRANT_COLLECTION    = "chapters_dense"
CONCEPT_CLUSTERS     = Path.home() / "autoresearch/grand_bible/data/concept_clusters.json"

_HERE          = os.path.dirname(os.path.abspath(__file__))
DB_PATH        = os.path.join(_HERE, "research.db")
STATE_PATH     = os.path.join(_HERE, "loop_state.json")

MAX_ITERATIONS    = 1000
LOOP_TIMEOUT      = 30.0   # seconds — single iteration wall clock limit
DRY_STREAK_MAX    = 10     # consecutive zero-new-info iterations before exit (a)
DRY_STREAK_P2     = 20     # more lenient threshold for Phase 2 (sparse entities)
CENTROID_NEIGHBORS = 300   # neighbors to fetch per archetype centroid
QDRANT_BATCH       = 100   # point IDs per Qdrant batch request

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _db():
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-32000")
    return db

def setup_db():
    db = _db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS observations (
            id            INTEGER PRIMARY KEY,
            iteration     INTEGER NOT NULL,
            endpoint      TEXT    NOT NULL,
            query         TEXT    NOT NULL,
            co_entity     TEXT    NOT NULL DEFAULT '',
            concept       TEXT    NOT NULL DEFAULT '',
            chapter       TEXT    NOT NULL DEFAULT '',
            score         REAL,
            passage       TEXT,
            discovered_at REAL    NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_connection
            ON observations(query, co_entity, chapter);
        CREATE INDEX IF NOT EXISTS idx_concept   ON observations(concept);
        CREATE INDEX IF NOT EXISTS idx_co_entity ON observations(co_entity);

        -- Phase 2: entities found via centroid search not yet queried via API
        CREATE TABLE IF NOT EXISTS unresearched_vectors (
            id         INTEGER PRIMARY KEY,
            entity     TEXT    NOT NULL,
            chapter    TEXT    NOT NULL DEFAULT '',
            chunk_id   INTEGER,
            archetype  TEXT    NOT NULL DEFAULT '',
            affinity   REAL,
            passage    TEXT,
            researched INTEGER NOT NULL DEFAULT 0,
            added_at   REAL    NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_uv_entity_chapter
            ON unresearched_vectors(entity, chapter);
        CREATE INDEX IF NOT EXISTS idx_uv_researched
            ON unresearched_vectors(researched, affinity);

        CREATE TABLE IF NOT EXISTS loop_runs (
            id               INTEGER PRIMARY KEY,
            started_at       REAL    NOT NULL,
            ended_at         REAL,
            seed_query       TEXT,
            total_iterations INTEGER DEFAULT 0,
            new_observations INTEGER DEFAULT 0,
            exit_reason      TEXT
        );
    """)
    db.commit()
    return db

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"iteration": 0, "queue": [], "seen": [], "phase2_populated": False}

def _save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

# ---------------------------------------------------------------------------
# Grand Bible REST API
# ---------------------------------------------------------------------------

def _api_get(path, params=None):
    url = API_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=LOOP_TIMEOUT - 2) as r:
        return json.loads(r.read())

def api_search(query, top_k=10):
    return _api_get("/api/search", {"q": query, "top_k": top_k})

def api_concepts():
    return _api_get("/api/browse/concepts")

# ---------------------------------------------------------------------------
# Qdrant direct access
# ---------------------------------------------------------------------------

def _qdrant_post(path, body):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        QDRANT_BASE + path, data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def qdrant_fetch_vectors(point_ids):
    """Batch-fetch 384-dim vectors for a list of Qdrant point IDs."""
    vectors = []
    for i in range(0, len(point_ids), QDRANT_BATCH):
        batch = point_ids[i:i + QDRANT_BATCH]
        result = _qdrant_post(
            f"/collections/{QDRANT_COLLECTION}/points",
            {"ids": batch, "with_vector": True, "with_payload": False}
        )
        for pt in result.get("result", []):
            v = pt.get("vector")
            if v:
                vectors.append(v)
    return vectors

def qdrant_search_centroid(centroid_vector, limit=CENTROID_NEIGHBORS):
    """Find the nearest `limit` chunks to a centroid vector."""
    result = _qdrant_post(
        f"/collections/{QDRANT_COLLECTION}/points/search",
        {"vector": centroid_vector, "limit": limit,
         "with_payload": True, "with_vector": False}
    )
    return result.get("result", [])

# ---------------------------------------------------------------------------
# Entity extraction (Phase 2)
# ---------------------------------------------------------------------------

# Match 1-3 consecutive capitalized words, each 3+ lowercase chars after the capital
_ENTITY_RE = re.compile(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b')
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

_SKIP = frozenset({
    # Function words / pronouns
    'The', 'And', 'But', 'For', 'Nor', 'Yet', 'Not', 'All', 'Any',
    'He', 'She', 'It', 'They', 'We', 'You', 'One', 'Who', 'Whose',
    'His', 'Her', 'Its', 'Our', 'Your', 'Their', 'Thy', 'Thine',
    'This', 'That', 'These', 'Those', 'Such', 'Same', 'Each', 'Both',
    # Religious common nouns (not proper names)
    'Lord', 'God', 'Christ', 'Jesus', 'Holy', 'Spirit', 'Father',
    'King', 'Queen', 'Prince', 'Prophet', 'Angel', 'Heaven', 'Hell',
    'Son', 'Man', 'Men', 'Woman', 'Women', 'People', 'Nation',
    # Temporal / spatial
    'Day', 'Days', 'Night', 'Year', 'Years', 'Time', 'Age', 'Era',
    'Land', 'Earth', 'World', 'Water', 'Fire', 'Light', 'Dark',
    'North', 'South', 'East', 'West',
    # Archaic English
    'Said', 'Unto', 'Thou', 'Thee', 'Hath', 'Thus', 'Also', 'Yea',
    'With', 'Upon', 'Into', 'From', 'After', 'Before', 'Until', 'Then',
    'When', 'Where', 'Which', 'What', 'How', 'Now', 'Here', 'There',
    # Ordinals / quantifiers
    'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Many', 'Some',
    'New', 'Old', 'Great', 'High', 'Most', 'Every', 'Other',
    # Text meta
    'Book', 'Chapter', 'Verse', 'Scripture', 'Bible', 'Text', 'Part',
})

def extract_entities(text):
    """Return lowercase proper-noun phrases from a chunk, filtering sentence starters."""
    sentence_starts = {0}
    for m in _SENTENCE_BOUNDARY.finditer(text):
        sentence_starts.add(m.end())

    found = set()
    for m in _ENTITY_RE.finditer(text):
        words = m.group(1).split()
        if m.start() in sentence_starts:
            # Salvage only if the sentence-starting word is a skip word
            # (e.g. "And Noah" → keep "Noah"; "Manu said…" → skip entirely)
            if words[0] not in _SKIP:
                continue
            words = words[1:]   # strip the conjunction/article
        # Strip any remaining leading skip words
        while words and words[0] in _SKIP:
            words.pop(0)
        if not words:
            continue
        phrase = " ".join(words)
        if len(phrase) < 3:
            continue
        found.add(phrase.lower())
    return found

# ---------------------------------------------------------------------------
# Phase 2: populate unresearched_vectors via centroid search
# ---------------------------------------------------------------------------

def populate_unresearched(db):
    """
    For each of the 20 archetypes:
      1. Load seed chunk IDs from concept_clusters.json
      2. Batch-fetch their vectors from Qdrant chapters_dense
      3. Compute normalised centroid
      4. Search for CENTROID_NEIGHBORS nearest chunks
      5. Extract entity names from chunk text
      6. Store new (entity, chapter, archetype, affinity) rows in unresearched_vectors,
         skipping entities already seen in Phase 1

    Returns count of new rows inserted.
    """
    if not CONCEPT_CLUSTERS.exists():
        print(f"  WARNING: concept_clusters.json not found at {CONCEPT_CLUSTERS}")
        return 0

    with open(CONCEPT_CLUSTERS) as f:
        clusters = json.load(f)

    # Entities already known from Phase 1
    known = set(
        row[0].lower() for row in
        db.execute("SELECT DISTINCT co_entity FROM observations WHERE co_entity != ''")
    )
    # Chapters already well-represented (use for de-emphasis, not hard filter)
    known_chapters = set(
        row[0] for row in
        db.execute("SELECT DISTINCT chapter FROM observations WHERE chapter != ''")
    )

    total_inserted = 0
    inserted_this_run = 0

    print(f"\n  Phase 2 centroid search: {len(clusters)} archetypes, "
          f"{CENTROID_NEIGHBORS} neighbors each")
    print(f"  Filtering against {len(known)} known entities, "
          f"{len(known_chapters)} known chapters\n")

    for archetype in clusters:
        slug     = archetype['slug']
        by_ch    = archetype.get('by_chapter', {})
        chunk_ids = [
            p['chunk_id']
            for passages in by_ch.values()
            for p in passages
            if p.get('chunk_id')
        ]

        if not chunk_ids:
            print(f"  {slug}: no seed chunks, skipping")
            continue

        # Fetch vectors and compute centroid
        try:
            vecs = qdrant_fetch_vectors(chunk_ids)
        except Exception as e:
            print(f"  {slug}: vector fetch failed: {e}")
            continue

        if not vecs:
            continue

        arr      = np.array(vecs, dtype=np.float32)
        centroid = arr.mean(axis=0)
        norm     = np.linalg.norm(centroid)
        if norm == 0:
            continue
        centroid /= norm

        # Search for nearest neighbors
        try:
            hits = qdrant_search_centroid(centroid.tolist())
        except Exception as e:
            print(f"  {slug}: centroid search failed: {e}")
            continue

        new_entities = 0
        for hit in hits:
            payload  = hit.get('payload', {})
            chapter  = payload.get('chapter_name', '')
            text     = payload.get('text', '')
            affinity = hit.get('score', 0.0)
            chunk_id = hit.get('id')

            for entity in extract_entities(text):
                if entity in known:
                    continue
                try:
                    db.execute(
                        """INSERT OR IGNORE INTO unresearched_vectors
                           (entity, chapter, chunk_id, archetype, affinity,
                            passage, added_at)
                           VALUES (?,?,?,?,?,?,?)""",
                        (entity, chapter, chunk_id, slug, affinity,
                         text[:400], time.time())
                    )
                    if db.execute("SELECT changes()").fetchone()[0]:
                        new_entities += 1
                        known.add(entity)   # don't re-add same entity for other archetypes
                except sqlite3.Error:
                    pass

        db.commit()
        total_inserted += new_entities
        print(f"  {slug:30s}  seeds={len(chunk_ids):3d}  "
              f"hits={len(hits):3d}  new_entities={new_entities:4d}")

    print(f"\n  Total unresearched_vectors inserted: {total_inserted}")
    return total_inserted

# ---------------------------------------------------------------------------
# Novelty (Phase 1 — set membership on observation triples)
# ---------------------------------------------------------------------------

def is_novel(db, query, co_entity, chapter):
    return db.execute(
        "SELECT 1 FROM observations WHERE query=? AND co_entity=? AND chapter=?",
        (query, co_entity, chapter)
    ).fetchone() is None

def store(db, iteration, endpoint, query, co_entity, concept, chapter, score, passage):
    try:
        db.execute(
            """INSERT INTO observations
               (iteration, endpoint, query, co_entity, concept, chapter,
                score, passage, discovered_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (iteration, endpoint, query, co_entity, concept, chapter,
             score, (passage or "")[:600], time.time())
        )
        db.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# ---------------------------------------------------------------------------
# Main loop  (Phase 1: BFS → Phase 2: centroid burndown, seamlessly)
# ---------------------------------------------------------------------------

def run_loop(endpoint, seed_query, top_k=10):
    db    = setup_db()
    state = _load_state()

    # ---- Phase 1 seed ----
    if not state["queue"] and not state.get("phase2_populated"):
        try:
            concepts_data = api_concepts()
            seed_pairs    = [[seed_query, ""]]
            for c in concepts_data:
                slug = c.get("slug", "")
                for q in c.get("queries", []):
                    seed_pairs.append([q, slug])
            print(f"Seeded queue with {len(seed_pairs)} queries from {len(concepts_data)} archetypes.")
        except Exception as e:
            print(f"Warning: could not fetch concepts for seeding: {e}")
            seed_pairs = [[seed_query, ""]]
        state["queue"] = seed_pairs
        state["seen"]  = []

    queue      = deque(state["queue"])
    seen       = set(state["seen"])
    iteration  = state["iteration"]
    dry_streak = 0
    phase      = 1 if not state.get("phase2_populated") else 2

    db.execute("INSERT INTO loop_runs (started_at, seed_query) VALUES (?,?)",
               (time.time(), seed_query))
    db.commit()
    run_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    new_total   = 0
    exit_reason = None

    print(f"\n{'='*60}")
    print(f"Grand Bible Research Loop  (Phase {phase})")
    print(f"Endpoint: {endpoint} | Seed: {seed_query!r}")
    print(f"Iteration: {iteration} | Queue: {len(queue)}")
    print(f"{'='*60}\n")

    while iteration < MAX_ITERATIONS:

        # ---- Phase transition: BFS exhausted → populate centroid seeds ----
        if not queue:
            if phase == 1 and not state.get("phase2_populated"):
                print(f"\n{'─'*60}")
                print("Phase 1 BFS complete. Starting Phase 2: centroid search.")
                print(f"{'─'*60}")
                n_inserted = populate_unresearched(db)
                state["phase2_populated"] = True
                phase = 2
                dry_streak = 0   # reset for Phase 2

                rows = db.execute(
                    """SELECT entity, archetype FROM unresearched_vectors
                       WHERE researched=0 ORDER BY affinity DESC LIMIT 5000"""
                ).fetchall()
                for entity, archetype in rows:
                    if entity not in seen:
                        queue.append([entity, archetype])

                print(f"\nPhase 2 queue: {len(queue)} entities loaded.")
                if not queue:
                    exit_reason = "Phase 2 found no new unresearched entities"
                    break
                continue   # re-enter loop with Phase 2 queue

            else:
                break   # both phases done

        # ---- Dequeue ----
        iteration += 1
        t0 = time.time()

        entry          = queue.popleft()
        current_query  = entry[0] if isinstance(entry, list) else entry
        parent_concept = entry[1] if isinstance(entry, list) and len(entry) > 1 else ""

        if current_query in seen:
            iteration -= 1
            continue
        seen.add(current_query)

        tag = "P2" if phase == 2 else "P1"
        print(f"[{iteration:04d}/{tag}] q={current_query!r}", end="  ", flush=True)

        # ---- API call ----
        try:
            result = api_search(current_query, top_k=top_k)
        except urllib.error.URLError as e:
            elapsed = time.time() - t0
            print(f"TIMEOUT ({elapsed:.1f}s): {e}")
            if elapsed >= LOOP_TIMEOUT:
                exit_reason = f"timeout at iteration {iteration} ({elapsed:.1f}s)"
                break
            continue
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        elapsed_api = time.time() - t0

        passages        = result.get("top_passages", [])
        cooccurring     = result.get("cooccurring", [])
        api_concept_list = result.get("concepts", [])
        concept          = api_concept_list[0] if api_concept_list else parent_concept

        # ---- Novelty check & store ----
        new_this = 0

        if cooccurring:
            for passage in passages[:3]:
                chapter = passage.get("chapter", "")
                score   = passage.get("score", 0.0)
                text    = passage.get("text", "")
                for co in cooccurring[:20]:
                    co_norm = co.get("norm", "")
                    if is_novel(db, current_query, co_norm, chapter):
                        if store(db, iteration, endpoint, current_query,
                                 co_norm, concept, chapter, score, text):
                            new_this += 1
                            if co_norm not in seen:
                                queue.append([co_norm, concept])
        else:
            for passage in passages:
                chapter = passage.get("chapter", "")
                score   = passage.get("score", 0.0)
                text    = passage.get("text", "")
                if is_novel(db, current_query, "", chapter):
                    if store(db, iteration, endpoint, current_query,
                             "", concept, chapter, score, text):
                        new_this += 1

        # Mark Phase 2 entity as researched
        if phase == 2:
            db.execute(
                "UPDATE unresearched_vectors SET researched=1 WHERE entity=?",
                (current_query,)
            )
            db.commit()

        elapsed_total = time.time() - t0
        new_total    += new_this

        print(f"new={new_this:3d}  api={elapsed_api:.2f}s  total={elapsed_total:.2f}s")

        # ---- Exit condition b: timeout ----
        if elapsed_total >= LOOP_TIMEOUT:
            exit_reason = f"timeout at iteration {iteration} ({elapsed_total:.1f}s)"
            break

        # ---- Exit condition a: dry streak ----
        streak_limit = DRY_STREAK_P2 if phase == 2 else DRY_STREAK_MAX
        if new_this == 0:
            dry_streak += 1
            if dry_streak >= streak_limit:
                exit_reason = (f"exhausted (Phase {phase}): "
                               f"{dry_streak} consecutive dry iterations")
                break
        else:
            dry_streak = 0

        # Checkpoint every 25 iterations
        if iteration % 25 == 0:
            state.update({"iteration": iteration,
                          "queue": list(queue), "seen": list(seen)})
            _save_state(state)
            pending_p2 = db.execute(
                "SELECT COUNT(*) FROM unresearched_vectors WHERE researched=0"
            ).fetchone()[0]
            print(f"  [ckpt] iter={iteration} new={new_total} "
                  f"queue={len(queue)} p2_pending={pending_p2}")

    # ---- Wrap up ----
    if not exit_reason:
        if iteration >= MAX_ITERATIONS:
            exit_reason = f"hard stop: reached {MAX_ITERATIONS} iterations"
        else:
            exit_reason = f"queue exhausted (Phase {phase})"

    db.execute(
        "UPDATE loop_runs SET ended_at=?, total_iterations=?, "
        "new_observations=?, exit_reason=? WHERE id=?",
        (time.time(), iteration, new_total, exit_reason, run_id)
    )
    db.commit()

    state.update({"iteration": iteration, "queue": list(queue), "seen": list(seen)})
    _save_state(state)

    print(f"\n{'='*60}")
    print(f"Exit: {exit_reason}")
    print(f"Iterations: {iteration} / {MAX_ITERATIONS}")
    print(f"New observations: {new_total}")
    obs_total = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    uv_done   = db.execute(
        "SELECT COUNT(*) FROM unresearched_vectors WHERE researched=1"
    ).fetchone()[0]
    uv_total  = db.execute("SELECT COUNT(*) FROM unresearched_vectors").fetchone()[0]
    print(f"Total observations in DB: {obs_total}")
    print(f"Unresearched vectors: {uv_done}/{uv_total} researched")
    print(f"{'='*60}\n")

    return exit_reason, iteration, new_total


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(top_n=20):
    db = _db()

    print("\n--- Top co-entity connections (observations) ---")
    for co, concept, n in db.execute(
        """SELECT co_entity, concept, COUNT(*) n FROM observations
           WHERE co_entity != '' GROUP BY co_entity, concept
           ORDER BY n DESC LIMIT ?""", (top_n,)
    ):
        print(f"  {co:30s}  [{concept:25s}]  x{n}")

    print("\n--- Archetype coverage ---")
    for concept, n in db.execute(
        """SELECT concept, COUNT(*) n FROM observations
           WHERE concept != '' GROUP BY concept ORDER BY n DESC"""
    ):
        print(f"  {concept:30s}  {n}")

    print("\n--- Unresearched vectors status ---")
    row = db.execute(
        "SELECT COUNT(*), SUM(researched), COUNT(DISTINCT archetype) FROM unresearched_vectors"
    ).fetchone()
    if row and row[0]:
        total, done, archetypes = row
        print(f"  Total: {total}  Done: {done or 0}  Pending: {total - (done or 0)}  "
              f"Archetypes: {archetypes}")
        print("\n  Top pending entities by affinity:")
        for entity, archetype, score in db.execute(
            """SELECT entity, archetype, affinity FROM unresearched_vectors
               WHERE researched=0 ORDER BY affinity DESC LIMIT 10"""
        ):
            print(f"    {entity:35s}  [{archetype:20s}]  {score:.4f}")
    else:
        print("  (not yet populated — run Phase 2 first)")


if __name__ == "__main__":
    summarize()
