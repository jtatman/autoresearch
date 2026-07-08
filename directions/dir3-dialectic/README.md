# Direction 3 — The Dialectical Engine

A loop that discovers philosophical positions the way a person thinks, not the way a
paper generator prints. The **position is the artifact**; the paper (if any) is exhaust.

## The mechanism (Hegel as keep/revert)

- **thesis** — the incumbent position.
- **antithesis** — the strongest objection / rival canon mustered against it. Appended,
  forever, to the **ledger**.
- **synthesis** — kept over its predecessor *only if it sublates*: dissolves more of the
  accumulated ledger than the last synthesis did, **and** survives the newest antithesis,
  without spawning a fatal new contradiction.

## The comparator is a falsifier, not a verifier

The judge never asserts "this is true / novel." It can only **reject** — find the
antithesis under which a synthesis collapses. Trust is not proof; it is **corroboration
by survived challenge**. A synthesis that has withstood N antitheses is corroborated to
degree N, nothing more. Absence of a refutation is never proof of novelty (the buried,
undeveloped position may exist and simply not have surfaced yet) — which is exactly why
this must be a loop and not a one-shot judge. The loop *is* the falsifier's training.

## The three laws (Prime Directive) as applied here

- **CAN** — `iterate.py` regenerates the synthesis and mines new antitheses each pass.
- **CANNOT** — `prepare.py` and the **append-only ledger** are immutable. The loop may
  *add* antitheses; it may never delete one. The judge's memory cannot be rewritten by
  the defendant.
- **SHOULD** — exit on **dialectical closure** (no surviving antithesis raises corroboration),
  on **discovered irreducibility** (the tension is provably permanent — itself a result),
  or on an iteration cap. Never run unsupervised without one of these.

## Failure modes (adversarial balance)

- Falsifier too strong → **nihilism**: nothing survives, paralysis.
- Falsifier too weak → **sophistry**: any clever-sounding synthesis passes.
Keep generator and falsifier in productive tension.

## Files

| File | Mutability | Holds |
|------|-----------|-------|
| `prepare.py` | **immutable** | source canons, append-only ledger, the falsifier protocol, `corroboration()` comparator, exit conditions |
| `iterate.py` | **mutable** | thesis→antithesis→synthesis generation, one keep/revert step |
| `loop_state.json` | runtime | iteration count, current position, corroboration score |

## Prior art

Forked concept: https://github.com/jtatman/AutoResearchClaw — a paper generator. This
direction abstracts *away* from paper-production toward the thinking itself.
