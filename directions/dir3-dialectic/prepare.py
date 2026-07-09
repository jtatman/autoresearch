"""
Direction 3 — Dialectical Engine :: IMMUTABLE HALF (the box + the judge).

Once locked, the loop may not edit this file. It defines:
  - the two source canons and their shared CORE (what any synthesis must preserve),
  - the APPEND-ONLY antithesis ledger (the falsifier's permanent memory),
  - the latent demands of the problem (the ground the loop mines objections from),
  - the falsifier (rejection-only: reports which demands a position violates, never
    asserts it is true or novel),
  - corroboration() — the keep/revert comparator (survival count, not truth),
  - satisfiability tooling to tell dialectical CLOSURE from discovered IRREDUCIBILITY,
  - the exit conditions.

Prime Directive rule (b): the loop may APPEND antitheses to the ledger; it may never
delete one and may never edit this file. The judge's memory cannot be rewritten by the
defendant.

MODE = mechanical (default): a deterministic constraint model that exercises every control
property of the box. MODE = llm: the same interface backed by an Anthropic-API falsifier
(natural-language positions + objections) — stubbed until wired. The swap is invisible to
iterate.py, exactly like Dir4's DIR4_DATASET.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER_PATH = os.path.join(HERE, "ledger.jsonl")   # append-only, one antithesis per line
MODE = os.environ.get("DIR3_MODE", "mechanical")

# ---------------------------------------------------------------------------
# The problem (immutable): two canons as partial commitments over N propositions.
# They agree on a CORE (preserved by any sublation) and conflict elsewhere.
# ---------------------------------------------------------------------------

N_PROPS = 10
CANON_A = {0: 1, 1: 1, 2: 1, 3: 0}   # e.g. a rationalist stance
CANON_B = {2: 0, 3: 0, 4: 1, 5: 1}   # e.g. an empiricist stance (conflicts on prop 2)


def canon_core() -> dict[int, int]:
    """Propositions on which both canons agree — the common ground a synthesis must keep."""
    return {p: CANON_A[p] for p in CANON_A if p in CANON_B and CANON_A[p] == CANON_B[p]}


CORE = canon_core()   # {3: 0}

# Latent demands of the problem. The loop does not see these as "truth"; it mines them one
# at a time as antitheses (the strongest objection = a demand the current position violates)
# and appends them to the ledger. A clause is satisfied if ANY literal matches the position.
# The last two demand opposite commitments on prop 2 — a built-in antinomy: once both are on
# the ledger, no position can satisfy it, and that irreducibility is itself the result.
LatentDemand = tuple[str, list]
_BASE_DEMANDS: list[LatentDemand] = [
    ("keep the rationalist ground (p0)",            [(0, 1)]),
    ("honor autonomy or experience (p1 or p4)",     [(1, 1), (4, 1)]),
    ("account for the empirical (p5)",              [(5, 1)]),
    ("resolve the mediating tension (p6 or p7)",    [(6, 1), (7, 1)]),
    ("preserve the rationalist commitment on p2",   [(2, 1)]),   # antinomy pole A
]
# SCENARIO=antinomy (default) adds the opposing pole on p2 => the ledger becomes an antinomy
# (irreducibility). SCENARIO=reconcilable omits it => the conflict is sublatable (closure).
# Same box, two dialectical outcomes — the proof that the judge distinguishes them.
SCENARIO = os.environ.get("DIR3_SCENARIO", "antinomy")
LATENT_DEMANDS: list[LatentDemand] = list(_BASE_DEMANDS)
if SCENARIO == "antinomy":
    LATENT_DEMANDS.append(("preserve the empiricist commitment on p2", [(2, 0)]))  # antinomy pole B

MAX_ITERATIONS = 40        # hard safety exit — never run unsupervised past this
CLOSURE_PATIENCE = 5       # iterations with no corroboration gain => dialectical closure


# ---------------------------------------------------------------------------
# Ledger — append-only store of antitheses (the falsifier's memory)
# ---------------------------------------------------------------------------

@dataclass
class Antithesis:
    text: str                       # human-readable objection
    literals: list = field(default_factory=list)   # mechanical clause: [[prop, val], ...]
    source: str = ""
    iteration: int = -1


def load_ledger() -> list[Antithesis]:
    if not os.path.exists(LEDGER_PATH):
        return []
    out = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Antithesis(**json.loads(line)))
    return out


def append_antithesis(a: Antithesis) -> None:
    """Append-only. There is deliberately no delete/update — this enforces rule (b)."""
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": a.text, "literals": a.literals,
                            "source": a.source, "iteration": a.iteration}) + "\n")


# ---------------------------------------------------------------------------
# Mechanical primitives (the "reality" the loop reasons against)
# ---------------------------------------------------------------------------

def clause_satisfied(position, literals) -> bool:
    return any(position[p] == v for p, v in literals)


def preserves_core(position) -> bool:
    return all(position[p] == v for p, v in CORE.items())


def latent_violations(position) -> list[LatentDemand]:
    """Demands the position currently fails — the pool the loop mines the next objection from.
    Immutable reality; the loop's *strategy* for choosing among these lives in iterate.py."""
    return [(text, lits) for text, lits in LATENT_DEMANDS if not clause_satisfied(position, lits)]


def free_props() -> list[int]:
    return [p for p in range(N_PROPS) if p not in CORE]


def max_satisfiable(ledger: list[Antithesis]) -> int:
    """Max number of ledger antitheses any core-preserving position can satisfy. Brute force
    over the free propositions (small by construction). Lets us tell closure from antinomy."""
    if not ledger:
        return 0
    fps = free_props()
    base = [0] * N_PROPS
    for p, v in CORE.items():
        base[p] = v
    best = 0
    for bits in itertools.product((0, 1), repeat=len(fps)):
        cand = base[:]
        for p, b in zip(fps, bits):
            cand[p] = b
        best = max(best, sum(clause_satisfied(cand, a.literals) for a in ledger))
        if best == len(ledger):
            break
    return best


def ledger_is_satisfiable(ledger: list[Antithesis]) -> bool:
    return max_satisfiable(ledger) == len(ledger)


# ---------------------------------------------------------------------------
# The falsifier — REJECTION ONLY. Never asserts a synthesis is true or novel.
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    survived: list = field(default_factory=list)   # antitheses this position withstands
    failed: list = field(default_factory=list)     # antitheses it violates
    new_contradiction: str | None = None           # abandons the canon core => self-contradiction


def falsify(position, ledger: list[Antithesis]) -> Verdict:
    """Test `position` against every antithesis in the ledger and report what it violates.
    Only rejects; makes no positive claim of truth."""
    if MODE != "mechanical":
        raise NotImplementedError("LLM falsifier not wired yet (set DIR3_MODE=mechanical)")
    survived, failed = [], []
    for a in ledger:
        (survived if clause_satisfied(position, a.literals) else failed).append(a.text)
    nc = None if preserves_core(position) else "abandons the canon core"
    return Verdict(survived=survived, failed=failed, new_contradiction=nc)


# ---------------------------------------------------------------------------
# The comparator — corroboration by survived challenge (NOT proof)
# ---------------------------------------------------------------------------

def corroboration(verdict: Verdict) -> float:
    """Survival count. Zero if the position self-contradicts (abandons the core). This is the
    only signal that decides keep/revert. Corroboration is not a truth value."""
    if verdict.new_contradiction is not None:
        return 0.0
    return float(len(verdict.survived))


# ---------------------------------------------------------------------------
# Exit conditions (rule (c): never run forever unsupervised)
# ---------------------------------------------------------------------------

def should_exit(iteration: int, no_gain_streak: int) -> tuple[bool, str]:
    if iteration >= MAX_ITERATIONS:
        return True, "iteration cap"
    if no_gain_streak >= CLOSURE_PATIENCE:
        return True, "dialectical closure (no corroboration gain)"
    return False, ""
