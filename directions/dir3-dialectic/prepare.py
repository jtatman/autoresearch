"""
Direction 3 — Dialectical Engine :: IMMUTABLE HALF.

This file is the box. Once locked, the loop may not edit it. It defines:
  - the source canons the dialectic reasons over,
  - the APPEND-ONLY antithesis ledger (the falsifier's memory),
  - the falsifier protocol (rejection-only; never asserts truth),
  - corroboration() — the keep/revert comparator,
  - the exit conditions.

Prime Directive rule (b): the loop may APPEND antitheses to the ledger; it may never
delete one and may never edit this file. The judge's memory cannot be rewritten by the
defendant.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify once locked)
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER_PATH = os.path.join(HERE, "ledger.jsonl")   # append-only, one antithesis per line
CANONS_DIR = os.path.join(HERE, "canons")          # source positions/corpora to synthesize from

MAX_ITERATIONS = 40        # hard safety exit — never run unsupervised past this
CLOSURE_PATIENCE = 5       # iterations with no corroboration gain => dialectical closure


# ---------------------------------------------------------------------------
# Ledger — append-only store of antitheses (hard negatives for the falsifier)
# ---------------------------------------------------------------------------

@dataclass
class Antithesis:
    text: str                       # the objection / rival position
    source: str = ""                # canon or derivation it came from
    iteration: int = -1             # when it entered the ledger


def load_ledger() -> list[Antithesis]:
    """Read every antithesis ever recorded. Order is preservation order."""
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
    """Append-only. There is deliberately no delete/update. This enforces rule (b)."""
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": a.text, "source": a.source, "iteration": a.iteration}) + "\n")


# ---------------------------------------------------------------------------
# The falsifier — REJECTION ONLY. Never asserts a synthesis is true or novel.
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    survived: list[str] = field(default_factory=list)   # antitheses this synthesis dissolves/withstands
    collapsed_under: str | None = None                  # first antithesis that breaks it, if any
    new_contradiction: str | None = None                # fatal new tension the synthesis itself spawns


def falsify(synthesis: str, ledger: list[Antithesis]) -> Verdict:
    """
    Attempt to BREAK `synthesis` against every antithesis in the ledger.
    Returns which it withstands and the first under which it collapses.

    This is the external judge. It only ever rejects; it makes no positive claim of
    truth. Implementation is an LLM-driven adversary (Anthropic API) prompted to find
    the collapse, plus a check for self-generated contradiction.

    TODO(impl): call the model to (1) test each antithesis, (2) detect new contradiction.
    """
    raise NotImplementedError("falsifier protocol not yet implemented")


# ---------------------------------------------------------------------------
# The comparator — corroboration by survived challenge (NOT proof)
# ---------------------------------------------------------------------------

def corroboration(verdict: Verdict) -> float:
    """
    Score a synthesis by how much of the ledger it survives, penalized to zero if it
    collapses or self-contradicts. This is the keep/revert signal — the only thing that
    decides whether the loop advances.

    Corroboration is a survival count, not a truth value.
    """
    if verdict.collapsed_under is not None or verdict.new_contradiction is not None:
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
