# loopme

A framework for **self-perpetuating improvement loops** under a fixed contract. Forked
from Karpathy's [autoresearch](https://github.com/jtatman/autoresearch) (nanochat), then
abstracted: the loop is the thing — not the data, not the domain.

The original upstream project (an LLM-pretraining loop optimizing `val_bpb`) is archived on
the `master` branch as the reference specimen. This branch studies the *mechanism*.

## The box (Prime Directive — the three laws)

Every loop, whatever the domain or strategy, is one box with a fixed contract:

- **CAN** — freely regenerate the mutable half (`iterate.py`) each pass.
- **CANNOT** — modify the immutable half (`prepare.py`): the constants and, crucially, the
  **comparator** that decides keep/revert. *The judge cannot be a thing the defendant is
  allowed to rewrite.*
- **SHOULD** — exit on defined conditions. **Never run forever unsupervised.**

Any loop *strategy* (ralph loop, mutation loop, N-parallel, …) is legal iff it obeys all
three. They're different ways of searching, not different projects.

## The one operation

    generate a variant → measure it → compare to incumbent → keep or revert

A domain is "loopable" only to the degree its comparator is **cheap and trustworthy**. Code
(`val_bpb`) is the ideal case (cheap + objective) — and therefore deliberately off the table
here. The interesting domains are the ones where you must first *manufacture* a trustworthy
comparator. The design principle we've landed on: **the comparator is a falsifier, external
to the mutation, that never asserts truth and never stops learning to reject.**

## Directions

- **`directions/dir3-dialectic/`** — the dialectical engine. Position-as-artifact; Hegel's
  thesis/antithesis/synthesis as keep/revert; the comparator is a falsifier scored by
  corroboration (survived challenges), backed by an append-only antithesis ledger.
- **`directions/dir4-labeling/`** — adversarial self-labeling. Automates a labeler onto a
  new unlabeled source and fine-tunes a CNN-LSTM; a negative-only discriminator vetoes slop;
  keep/revert measured only on a small immutable human-labeled anchor.

Each direction: `prepare.py` (immutable box + comparator) · `iterate.py` (mutable attempt) ·
`README.md` (its contract). Run with `uv run iterate.py` from inside the folder.

## Constraints

- `uv pip install` only — never `uv add` (breaks the torch pin).
- GPU: GTX 1070, 8 GB VRAM, CUDA 12.6, torch 2.7.1+cu126 (Pascal sm_61 — no bf16, no FA2/FA3).

See `CLAUDE.md` for the full framework spec.
