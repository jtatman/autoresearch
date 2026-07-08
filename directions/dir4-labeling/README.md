# Direction 4 — Adversarial Self-Labeling (CNN-LSTM)

A loop that **automates the labeler** and generalizes onto a *new, unlabeled* source, then
fine-tunes a CNN-LSTM on the labels it produced. This is training, but not GPT and not
Karpathy's framework — the research question is: **how small can a trusted anchor be before
the loop starts lying to itself?**

## Why an anchor exists (the confirmation-collapse trap)

If the loop generates the labels, it cannot score itself against those labels — that's a
mirror, and self-training will happily "improve" straight into slop. So the comparator
must live in something the loop *cannot rewrite*: a small, human-labeled **anchor** slice
of the new source, held in `prepare.py`, never seen during label generation, always the
sole keep/revert signal.

## The discriminator is a falsifier (negative knowledge only)

It does **not** need to know where the bounding box *should* be. It only needs to know
where it *shouldn't* — reject implausible labels against patterns it holds true. A
negative-only critic can only *shrink* the acceptable set; it can never assert a specific
wrong box, which is exactly what makes it safe. The generator proposes 3×–5× label variants
per frame; the discriminator vetoes; survivors that also **agree with each other**
(adversarial consistency) become pseudo-labels.

## The three laws (Prime Directive) as applied here

- **CAN** — `iterate.py` regenerates the labeler + discriminator + fine-tune recipe each pass.
- **CANNOT** — `prepare.py` is immutable: the labeled source, the unlabeled source, the
  fixed frame-extraction, the **anchor**, and `evaluate()`. The judge (anchor accuracy)
  cannot be edited by the loop.
- **SHOULD** — exit on the objective (e.g. ≥89% anchor accuracy, or +40% total gain), on a
  no-improvement / drift-detected streak, or on an iteration cap. Wall-clock cap as a
  safety net. Never run unsupervised without one.

## Failure modes (adversarial balance)

- Discriminator too strong → nothing propagates, loop **stalls**.
- Discriminator too weak → slop propagates, **confirmation collapse**.

## Files

| File | Mutability | Holds |
|------|-----------|-------|
| `prepare.py` | **immutable** | labeled source, unlabeled source, frame extraction, the anchor, `evaluate()` |
| `iterate.py` | **mutable** | generate label variants → discriminator veto → pseudo-labels → fine-tune → eval → keep/revert |
| `loop_state.json` | runtime | iteration, best anchor accuracy, drift signals |

## Note on deps

Frame extraction / vision may need packages beyond the shared `pyproject.toml`. Install
with `uv pip install ...` only — never `uv add` (breaks the torch pin).
