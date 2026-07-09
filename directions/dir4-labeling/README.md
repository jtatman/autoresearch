# Direction 4 — Adversarial Self-Labeling

A loop that **automates the labeler**: it generalizes onto a *new, unlabeled* image source,
generates pseudo-labels for it, and fine-tunes a detector on the labels it produced. The
research question is not "can it train" — it's **how small can the trusted anchor/seed be
before the loop starts lying to itself?**

This is the domain instance of the loopme thesis: a keep/revert loop is only as trustworthy
as its comparator, so the whole design is about manufacturing a comparator the loop *cannot
game*.

---

## The contract (Prime Directive here)

| File | Mutability | Holds |
|------|-----------|-------|
| `prepare.py` | **immutable** | datasets, fixed frame handling, the **anchor**, `evaluate()` (IoU accuracy). The judge. |
| `iterate.py` | **mutable** | generator, discriminator, proposal + acceptance + augmentation + fine-tune recipe. The attempt. |
| `loop_state.json` | runtime | iteration, best anchor accuracy, drift streak |

- **CAN** — rewrite `iterate.py` freely each pass.
- **CANNOT** — touch `prepare.py`, the anchor, or `evaluate()`. *The judge cannot be a thing
  the defendant is allowed to rewrite* — the guard against confirmation collapse.
- **SHOULD** — exit on target accuracy, a drift streak (anchor regressing), or an iteration
  cap. Never run unsupervised without one.

## Two design principles that survived contact with data

1. **The anchor is the only truth signal.** If the loop scores itself on labels it generated,
   that's a mirror and it will "improve" into slop. Keep/revert is measured *only* on a small
   human-labeled hold-out the loop never sees during labeling.
2. **The discriminator is a falsifier, not a verifier.** It never asserts where the box
   *should* be; it only rejects where it clearly *shouldn't*. A negative-only critic can only
   shrink the acceptable set, never assert a specific wrong box — which is what makes it safe.
   It is **learned and co-evolving** (trained each round on real seed boxes vs clearly-wrong
   corruptions vs the generator's own false-accepts), so it stays sharp as the generator
   improves instead of being gamed.

## The pipeline (one round)

```
train D (seed-real vs clearly-wrong vs generator false-accepts)
  → generator proposes via MC-dropout (K stochastic passes = uncertainty)
  → D vetoes implausible proposals (negative-only)
  → surviving samples must AGREE (low scatter) → consensus pseudo-label
  → augment survivors (offset rejection attrition)
  → fine-tune generator
  → evaluate on the immutable anchor → keep or revert
  → stash D's false-accepts as next round's hard negatives
```

Generator = a **frozen pretrained ResNet-18** + a small trainable regression head (grayscale
repeated to 3ch, ImageNet-normalized). MC-dropout on the head both regularizes training and,
at inference, produces the proposal diversity whose agreement is the reliability signal.

## Run

```bash
# from this folder, always via uv run (never bare python — see repo CLAUDE.md)
DIR4_DATASET=synthetic uv run python iterate.py   # numpy-only toy task (default)
DIR4_DATASET=penn      uv run python iterate.py   # Penn-Fudan pedestrians (auto-downloads ~53MB)
```

`DIR4_DATASET` is the documented swap point — `load_labeled/load_unlabeled/anchor/evaluate`
are identical for both worlds, so `iterate.py` never knows which it's in.

---

## What we found (the experimental arc)

Metric is anchor IoU accuracy (IoU ≥ 0.5), the sole immutable keep/revert signal.

| Stage | Result | Lesson |
|------|--------|--------|
| Synthetic, fixed statistical critic | 0.09 → **0.81** | the loop genuinely learns when the generator can extract signal from pseudo-labels |
| Synthetic, learned co-evolving D | 0.09 → **0.70** | co-evolution works but is finicky; hit **both** GAN failure modes live — D-starvation (fixed by class balance) and arms-race drift (caught by the anchor) |
| Real Penn, from-scratch 64px net | warm 0.13, loop **collapses into self-agreement** | timid jitter manufactures false consensus; the anchor refused to certify non-progress |
| Real Penn, **MC-dropout** proposals | 0.13 → **0.27** (~2×) | uncertainty-gated agreement is a *real* reliability signal; jitter and photometric TTA were not |
| Real Penn, **frozen ResNet-18** backbone (lever 2) | warm **0.13 → 0.26** | the generator was the ceiling; pretraining raised it. The trusted seed can be tiny (25 labels ≈ 90 labels) |
| Unfreeze `layer4` | single run 0.30 → 0.33 … | …but a **5-seed noise check** showed this was noise: unfrozen 0.300 ± 0.030 vs frozen 0.313 ± 0.050 — indistinguishable, and unfreezing *damages* the warm start (0.26 → 0.15). **Reverted.** |

**Seed-validated bottom line (Penn, 25 labeled / 115 unlabeled / 30 anchor, 5 seeds):**
- **Lever 2 (pretrained backbone) is the real win** — warm start 0.13 → 0.26.
- **The self-labeling loop adds a small positive gain on frozen features (~+0.05), and it
  holds across seeds** (best ≥ warm in every seed; one seed reached 0.40). Suggestive rather
  than airtight at n=5 — but it's the honest signal that self-labeling *does* add value on
  real data.
- **Unfreezing the backbone is a net wash that adds risk** — dropped, per the simplicity
  criterion.
- **The anchor governed every run.** Frozen or unfrozen, right split or wrong, learned critic
  or fixed — it never kept a model below its best and always halted on collapse.

The methodological punchline: a single run reported a 0.30 → 0.33 "win" that the multi-seed
check exposed as one frame of luck. *A trustworthy comparator is not one run — it's survival
across seeds.* The loopme thesis, applied to our own research process.

## Honest limitations & next levers

- Absolute accuracy is low (~0.30) — 64px grayscale + largest-box-per-image is a plumbing
  proof, not a real detector. The Penn anchor is 30 frames, so resolution is ±1 frame (3.3%).
- Not yet tested: higher-res/RGB frames (a `prepare.py` change), a real multi-object detector
  head, and the **temporal LSTM** — which only becomes meaningful once frames come from video
  rather than i.i.d. images (the synthetic frames and Penn stills are both non-temporal, so
  the LSTM is deliberately omitted rather than faked).

## Deps

`torchvision 0.22.1+cu126` (matches the `torch 2.7.1+cu126` pin) installed via
`uv pip install --no-deps` — never `uv add`. Runs on the GTX 1070 via `uv run`.
