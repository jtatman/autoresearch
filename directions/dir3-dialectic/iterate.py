"""
Direction 3 — Dialectical Engine :: MUTABLE HALF (the "attempt").

Blow this file away and rewrite it each iteration to improve how the loop reasons.
It may use ONLY the immutable contract from prepare.py: it appends antitheses (never
deletes), submits a synthesis to the falsifier, and advances on corroboration gain.

One pass = thesis -> antithesis -> synthesis -> falsify -> keep/revert.

Prime Directive: CAN rewrite this file; CANNOT touch prepare.py or the ledger except by
append; SHOULD stop when prepare.should_exit() says so.
"""

from __future__ import annotations

import json
import os

from prepare import (
    Antithesis, load_ledger, append_antithesis, falsify, corroboration, should_exit,
    HERE,
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


# --- The mutable reasoning, all TODO stubs for now ------------------------------

def generate_antithesis(position: str | None, ledger: list[Antithesis], iteration: int) -> Antithesis:
    """Mine the strongest objection / rival canon against the current position."""
    raise NotImplementedError


def generate_synthesis(position: str | None, ledger: list[Antithesis]) -> str:
    """Produce a position that attempts to sublate the tension the thesis could not."""
    raise NotImplementedError


# --- One keep/revert step -------------------------------------------------------

def step() -> bool:
    """Run one dialectical iteration. Returns True to continue, False to exit."""
    state = load_state()
    ledger = load_ledger()

    stop, reason = should_exit(state["iteration"], state["no_gain_streak"])
    if stop:
        print(f"EXIT: {reason} at iteration {state['iteration']}")
        return False

    # antithesis (append-only — enters the falsifier's permanent memory)
    anti = generate_antithesis(state["position"], ledger, state["iteration"])
    append_antithesis(anti)
    ledger = load_ledger()

    # synthesis -> falsify -> corroboration
    synthesis = generate_synthesis(state["position"], ledger)
    verdict = falsify(synthesis, ledger)
    score = corroboration(verdict)

    # keep/revert: advance only on strict corroboration gain
    if score > state["best_corroboration"]:
        state.update(position=synthesis, best_corroboration=score, no_gain_streak=0)
        print(f"KEEP  it={state['iteration']} corroboration={score:.0f}")
    else:
        state["no_gain_streak"] += 1
        print(f"REVERT it={state['iteration']} corroboration={score:.0f} (best={state['best_corroboration']:.0f})")

    state["iteration"] += 1
    save_state(state)
    return True


if __name__ == "__main__":
    while step():
        pass
