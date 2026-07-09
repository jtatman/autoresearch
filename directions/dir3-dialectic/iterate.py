"""
Direction 3 — Dialectical Engine :: MUTABLE HALF (the "attempt").

Blow this file away and rewrite it each iteration to improve HOW the loop reasons. It may
use ONLY the immutable contract from prepare.py: it appends antitheses (never deletes),
submits a synthesis to the falsifier, and advances on corroboration gain.

One pass = thesis -> antithesis -> synthesis -> falsify -> keep/revert.

Strategy in this (mechanical) generation:
  - antithesis: raise the strongest unraised objection = a latent demand the current
    position violates (rule: the judge only ever judges against objections actually made).
  - synthesis: a CONSERVATIVE sublation — among positions that best satisfy the ledger
    (surviving the newest antithesis first), keep the one closest to the incumbent. Retain
    as much as possible while resolving the new tension.
  - keep only on strict corroboration gain AND surviving the newest antithesis.

Prime Directive: CAN rewrite this file; CANNOT touch prepare.py or the ledger except by
append; SHOULD stop on closure, irreducibility, or the iteration cap.
"""

from __future__ import annotations

import itertools
import json
import os

from prepare import (
    Antithesis, load_ledger, append_antithesis, falsify, corroboration, should_exit,
    ledger_is_satisfiable, clause_satisfied, latent_violations,
    HERE, N_PROPS, CORE, CANON_A, free_props,
)

STATE_PATH = os.path.join(HERE, "loop_state.json")


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"iteration": 0, "position": None, "best_corroboration": 0.0, "no_gain_streak": 0}


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def initial_position() -> tuple:
    """Thesis: start from canon A (which already respects the core)."""
    pos = [0] * N_PROPS
    for p, v in CORE.items():
        pos[p] = v
    for p, v in CANON_A.items():
        pos[p] = v
    return tuple(pos)


# --- the mutable reasoning ------------------------------------------------------

def generate_antithesis(position, ledger, iteration) -> Antithesis | None:
    """Strongest unraised objection: the first latent demand the position violates that is
    not already on the ledger. None => no objection can be raised (a terminal condition)."""
    on_ledger = {a.text for a in ledger}
    for text, lits in latent_violations(position):
        if text not in on_ledger:
            return Antithesis(text=text, literals=lits, source="latent", iteration=iteration)
    return None


def generate_synthesis(ledger, incumbent) -> tuple:
    """Conservative sublation: maximize (survives-newest, total-ledger-survived), then stay
    as close to the incumbent as possible (fewest changed commitments)."""
    fps = free_props()
    newest = ledger[-1] if ledger else None
    base = [0] * N_PROPS
    for p, v in CORE.items():
        base[p] = v
    best, best_key = None, (-1, -1, -(N_PROPS + 1))
    for bits in itertools.product((0, 1), repeat=len(fps)):
        cand = base[:]
        for p, b in zip(fps, bits):
            cand[p] = b
        cand = tuple(cand)
        total = sum(clause_satisfied(cand, a.literals) for a in ledger)
        survives_new = 1 if (newest and clause_satisfied(cand, newest.literals)) else 0
        closeness = -sum(1 for i in range(N_PROPS) if cand[i] != incumbent[i])
        key = (survives_new, total, closeness)
        if key > best_key:
            best_key, best = key, cand
    return best


# --- one keep/revert step -------------------------------------------------------

def _finalize(state, ledger, trigger):
    """No further objection can be raised: diagnose closure vs irreducibility and exit."""
    if ledger_is_satisfiable(ledger):
        reason = "dialectical closure (the position withstands every raisable objection)"
    else:
        reason = "discovered irreducibility (antinomy: the ledger cannot be jointly satisfied)"
    print(f"EXIT [{trigger}]: {reason} at it={state['iteration']} "
          f"(ledger={len(ledger)}, corroboration={state['best_corroboration']:.0f})")
    save_state(state)


def step() -> bool:
    state = load_state()
    ledger = load_ledger()

    stop, reason = should_exit(state["iteration"], state["no_gain_streak"])
    if stop:
        _finalize(state, ledger, reason)
        return False

    position = tuple(state["position"]) if state["position"] else initial_position()

    anti = generate_antithesis(position, ledger, state["iteration"])
    if anti is None:
        _finalize(state, ledger, "no raisable objection")
        return False

    append_antithesis(anti)                      # enters the falsifier's permanent memory
    ledger = load_ledger()

    synthesis = generate_synthesis(ledger, position)
    verdict = falsify(synthesis, ledger)
    score = corroboration(verdict)
    survives_newest = anti.text in verdict.survived

    kept = score > state["best_corroboration"] and survives_newest and verdict.new_contradiction is None
    if kept:
        state.update(position=list(synthesis), best_corroboration=score, no_gain_streak=0)
        tag = "KEEP  "
    else:
        state["no_gain_streak"] += 1
        tag = "REVERT"

    print(f"{tag} it={state['iteration']:02d} | raised: {anti.text!r} | "
          f"corroboration={score:.0f}/{len(ledger)} best={state['best_corroboration']:.0f} "
          f"| survives_newest={survives_newest} no_gain={state['no_gain_streak']}")

    state["iteration"] += 1
    save_state(state)
    return True


if __name__ == "__main__":
    while step():
        pass
