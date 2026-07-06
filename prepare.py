"""
Grand Bible archetype research — infrastructure.

Manages SQLite database, API calls, novelty detection, BFS query expansion,
and the autonomous research loop.

DO NOT MODIFY unless changing core infrastructure. Query parameters live in train.py.
"""

import json
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE       = "http://localhost:8081"
_HERE          = os.path.dirname(os.path.abspath(__file__))
DB_PATH        = os.path.join(_HERE, "research.db")
STATE_PATH     = os.path.join(_HERE, "loop_state.json")
MAX_ITERATIONS = 1000
LOOP_TIMEOUT   = 30.0   # seconds — single iteration wall clock limit
DRY_STREAK_MAX = 10     # consecutive zero-new-info iterations before exit (a)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _db():
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-32000")   # 32 MB page cache
    return db

def setup_db():
    db = _db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS observations (
            id           INTEGER PRIMARY KEY,
            iteration    INTEGER NOT NULL,
            endpoint     TEXT    NOT NULL,
            query        TEXT    NOT NULL,
            co_entity    TEXT    NOT NULL DEFAULT '',
            concept      TEXT    NOT NULL DEFAULT '',
            chapter      TEXT    NOT NULL DEFAULT '',
            score        REAL,
            passage      TEXT,
            discovered_at REAL   NOT NULL
        );

        -- Fast duplicate check: same query+co_entity+chapter is the same connection
        CREATE UNIQUE INDEX IF NOT EXISTS idx_connection
            ON observations(query, co_entity, chapter);

        CREATE INDEX IF NOT EXISTS idx_concept   ON observations(concept);
        CREATE INDEX IF NOT EXISTS idx_co_entity ON observations(co_entity);

        CREATE TABLE IF NOT EXISTS loop_runs (
            id                INTEGER PRIMARY KEY,
            started_at        REAL NOT NULL,
            ended_at          REAL,
            seed_query        TEXT,
            total_iterations  INTEGER DEFAULT 0,
            new_observations  INTEGER DEFAULT 0,
            exit_reason       TEXT
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
    return {"iteration": 0, "queue": [], "seen": []}

def _save_state(iteration, queue, seen):
    with open(STATE_PATH, "w") as f:
        json.dump({"iteration": iteration,
                   "queue": list(queue),
                   "seen": list(seen)}, f, indent=2)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _get(path, params=None):
    url = API_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=LOOP_TIMEOUT - 2) as r:
        return json.loads(r.read())

def api_search(query, top_k=10):
    return _get("/api/search", {"q": query, "top_k": top_k})

def api_concepts():
    return _get("/api/browse/concepts")

# ---------------------------------------------------------------------------
# Novelty
#
# Phase 1: pure set membership — (query, co_entity, chapter) not in DB → NEW.
# Phase 2 hook: compare cosine scores of semantically similar observations;
#   if new score differs by > threshold from existing, it's a different context.
# ---------------------------------------------------------------------------

def is_novel(db, query, co_entity, chapter):
    row = db.execute(
        "SELECT 1 FROM observations WHERE query=? AND co_entity=? AND chapter=?",
        (query, co_entity, chapter)
    ).fetchone()
    return row is None

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
# Main loop
# ---------------------------------------------------------------------------

def run_loop(endpoint, seed_query, top_k=10):
    db      = setup_db()
    state   = _load_state()

    # Seed queue: initial query + all archetype seed queries.
    # Queue entries are [query, concept_slug] pairs so archetype context
    # propagates through BFS expansion.
    if not state["queue"]:
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

    # Queue entries: [query_str, concept_slug]
    queue      = deque(state["queue"])
    seen       = set(state["seen"])
    iteration  = state["iteration"]
    dry_streak = 0

    db.execute("INSERT INTO loop_runs (started_at, seed_query) VALUES (?,?)",
               (time.time(), seed_query))
    db.commit()
    run_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    new_total   = 0
    exit_reason = None

    print(f"\n{'='*60}")
    print(f"Grand Bible Research Loop")
    print(f"Endpoint: {endpoint} | Seed: {seed_query!r}")
    print(f"Starting at iteration {iteration} | Queue depth: {len(queue)}")
    print(f"{'='*60}\n")

    while queue and iteration < MAX_ITERATIONS:
        iteration += 1
        t0 = time.time()

        entry         = queue.popleft()
        current_query = entry[0] if isinstance(entry, list) else entry
        parent_concept = entry[1] if isinstance(entry, list) and len(entry) > 1 else ""

        if current_query in seen:
            iteration -= 1   # don't count skipped queries
            continue
        seen.add(current_query)

        print(f"[{iteration:04d}] q={current_query!r}", end="  ", flush=True)

        # --- API call ---
        try:
            result = api_search(current_query, top_k=top_k)
        except urllib.error.URLError as e:
            elapsed = time.time() - t0
            print(f"TIMEOUT/ERROR ({elapsed:.1f}s): {e}")
            if elapsed >= LOOP_TIMEOUT:
                exit_reason = f"timeout at iteration {iteration}: API call took {elapsed:.1f}s"
                break
            continue
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        elapsed_api = time.time() - t0

        passages    = result.get("top_passages", [])
        cooccurring = result.get("cooccurring", [])
        # Prefer concept from API result; fall back to parent archetype from seed
        api_concepts_list = result.get("concepts", [])
        concept = api_concepts_list[0] if api_concepts_list else parent_concept

        # --- Novelty check & store ---
        new_this = 0

        if cooccurring:
            for passage in passages[:3]:      # top 3 passages per query
                chapter = passage.get("chapter", "")
                score   = passage.get("score", 0.0)
                text    = passage.get("text", "")
                for co in cooccurring[:20]:   # top 20 co-occurring entities
                    co_norm = co.get("norm", "")
                    if is_novel(db, current_query, co_norm, chapter):
                        if store(db, iteration, endpoint, current_query,
                                 co_norm, concept, chapter, score, text):
                            new_this += 1
                            # BFS expansion: inherit parent concept when queuing co-entity
                            if co_norm not in seen:
                                queue.append([co_norm, concept])
        else:
            # No co-entities: store raw passage observations
            for passage in passages:
                chapter = passage.get("chapter", "")
                score   = passage.get("score", 0.0)
                text    = passage.get("text", "")
                if is_novel(db, current_query, "", chapter):
                    if store(db, iteration, endpoint, current_query,
                             "", concept, chapter, score, text):
                        new_this += 1

        elapsed_total = time.time() - t0
        new_total    += new_this

        print(f"new={new_this:3d}  api={elapsed_api:.2f}s  total={elapsed_total:.2f}s")

        # --- Exit condition b: iteration wall clock exceeded ---
        if elapsed_total >= LOOP_TIMEOUT:
            exit_reason = f"timeout at iteration {iteration} ({elapsed_total:.1f}s)"
            break

        # --- Exit condition a: consecutive dry runs ---
        if new_this == 0:
            dry_streak += 1
            if dry_streak >= DRY_STREAK_MAX:
                exit_reason = (f"exhausted: {dry_streak} consecutive iterations "
                               f"with no new observations")
                break
        else:
            dry_streak = 0

        # Checkpoint state every 25 iterations
        if iteration % 25 == 0:
            _save_state(iteration, list(queue), list(seen))
            print(f"  [checkpoint] iter={iteration} total_new={new_total} queue={len(queue)}")

    # --- Wrap up ---
    if not exit_reason:
        if iteration >= MAX_ITERATIONS:
            exit_reason = f"hard stop: reached {MAX_ITERATIONS} iterations"
        else:
            exit_reason = "queue exhausted"

    db.execute(
        "UPDATE loop_runs SET ended_at=?, total_iterations=?, new_observations=?, exit_reason=? WHERE id=?",
        (time.time(), iteration, new_total, exit_reason, run_id)
    )
    db.commit()
    _save_state(iteration, queue, seen)

    print(f"\n{'='*60}")
    print(f"Exit: {exit_reason}")
    print(f"Iterations: {iteration} / {MAX_ITERATIONS}")
    print(f"New observations: {new_total}")
    total_in_db = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    print(f"Total in database: {total_in_db}")
    print(f"{'='*60}\n")

    return exit_reason, iteration, new_total


# ---------------------------------------------------------------------------
# Summary query (for reporting)
# ---------------------------------------------------------------------------

def summarize(top_n=20):
    db = _db()
    print("\n--- Top co-entity connections by frequency ---")
    rows = db.execute(
        """SELECT co_entity, concept, COUNT(*) as n
           FROM observations WHERE co_entity != ''
           GROUP BY co_entity, concept
           ORDER BY n DESC LIMIT ?""", (top_n,)
    ).fetchall()
    for co, concept, n in rows:
        print(f"  {co:30s}  [{concept:25s}]  x{n}")

    print("\n--- Concepts by observation count ---")
    rows = db.execute(
        """SELECT concept, COUNT(*) as n FROM observations
           WHERE concept != '' GROUP BY concept ORDER BY n DESC"""
    ).fetchall()
    for concept, n in rows:
        print(f"  {concept:30s}  {n}")


if __name__ == "__main__":
    summarize()
