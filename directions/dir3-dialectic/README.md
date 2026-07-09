# Direction 3 — The Dialectical Engine

A loop that discovers philosophical positions the way a person thinks, not the way a paper
generator prints. The **position is the artifact**; a paper, if any, is exhaust.

## The mechanism (Hegel as keep/revert)

- **thesis** — the incumbent position.
- **antithesis** — the strongest objection / rival demand raised against it, appended
  **forever** to the ledger.
- **synthesis** — kept over its predecessor *only if it sublates*: resolves more of the
  accumulated ledger (higher corroboration) **and** survives the newest antithesis, without
  abandoning the canon core.

## The comparator is a falsifier, not a verifier

The judge never asserts "this is true / novel." It only **rejects** — reports which
antitheses a position violates. Trust is not proof; it is **corroboration by survived
challenge** (a survival count). The judge lives in the immutable half, external to the
mutation — *the judge cannot be a thing the defendant is allowed to rewrite* (rule (b)).

## The contract (Prime Directive here)

| File | Mutability | Holds |
|------|-----------|-------|
| `prepare.py` | **immutable** | the two canons + their core, the append-only ledger, the latent demands, `falsify()`, `corroboration()`, satisfiability tooling, exit conditions |
| `iterate.py` | **mutable** | which objection to raise, how to synthesize, one keep/revert step |
| `ledger.jsonl` | append-only | every antithesis ever raised (never deleted — rule (b)) |
| `loop_state.json` | runtime | iteration, position, best corroboration, no-gain streak |

- **CAN** — rewrite `iterate.py`; append antitheses.
- **CANNOT** — edit `prepare.py`, or delete/rewrite a ledger entry.
- **SHOULD** — exit on **dialectical closure**, on **discovered irreducibility** (itself a
  result), or on the iteration cap. Never run unsupervised without one.

## Two terminal outcomes (the interesting part)

- **Dialectical closure** — the position withstands every objection that can still be
  raised, and the ledger is jointly satisfiable. A stable synthesis.
- **Discovered irreducibility** — the ledger becomes an **antinomy**: no position can
  satisfy it (`max_satisfiable < len(ledger)`). That some tension is *essential* is a
  genuine philosophical result, not a failure.

## MODE — mechanical now, LLM later (the swap point)

`DIR3_MODE=mechanical` (default) is a deterministic constraint model: a **position** is
commitments over N propositions; the two **canons** seed it (agreeing on a *core* that must
be preserved, conflicting elsewhere); an **antithesis** is a demand (clause); the
**falsifier** reports which demands a position violates; **synthesis** is a conservative
search (satisfy the newest, then maximize total survived, then stay closest to the
incumbent). It exists to prove the *control properties* — append-only ledger, corroboration
keep/revert, closure-vs-irreducibility diagnosis — deterministically, before any LLM noise.

`DIR3_MODE=llm` is the same interface backed by an Anthropic-API falsifier: positions,
antitheses, and collapse judgments in natural language, over real philosophical canons.
Stubbed until wired (needs a key + the `anthropic` SDK). `iterate.py` never knows which
mode it is in — exactly like Dir4's `DIR4_DATASET`.

## Run

```bash
# from this folder, always via uv run
DIR3_SCENARIO=antinomy     uv run python iterate.py   # -> discovered irreducibility (default)
DIR3_SCENARIO=reconcilable uv run python iterate.py   # -> dialectical closure
```

## What we found (mechanical proof)

Same box, two outcomes — the proof the judge distinguishes them:

| Scenario | Trace | Outcome |
|----------|-------|---------|
| `antinomy` | corroboration 1→2→3 (sublation), then the two poles of p2 (rationalist vs empiricist commitment) both hit the ledger; satisfying either costs the other → REVERT | **irreducibility** (antinomy, `max_satisfiable=3 < 4`) |
| `reconcilable` | corroboration 1→2, every raisable objection satisfied | **closure** (2/2) |

The antinomy run is the striking one: the loop climbs by genuinely sublating tensions, then
founders on an *irreducible* commitment where the two canons cannot both be honored — a
Kantian antinomy shape — and reports that irreducibility as its result. Every control
property fired: append-only ledger (grew 1→4, never shrank), corroboration-gated keep/revert,
survives-newest requirement, and the closure-vs-irreducibility diagnosis.

## Failure modes to watch when the LLM lands

- Falsifier too strong → **nihilism**: nothing survives, no synthesis is ever kept.
- Falsifier too weak → **sophistry**: any clever-sounding synthesis passes.
Keep generator and falsifier in productive tension — the same GAN-balance lesson as Dir4.

## Honest limitations

The mechanical model proves the *plumbing*, not philosophical content — its "positions" are
bit-vectors and its "objections" are clauses. Real dialectical value only arrives with the
LLM mode over actual canons. What carries over is the hard-won structure: an append-only
ledger, a rejection-only judge external to the mutation, corroboration as a survival count,
and a principled distinction between closure and irreducibility.

## Prior art

Forked concept: https://github.com/jtatman/AutoResearchClaw — a paper generator. This
direction abstracts *away* from paper-production toward the thinking itself: the position is
the artifact, and the exit condition is dialectical, not word count.
