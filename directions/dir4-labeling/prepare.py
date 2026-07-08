"""
Direction 4 — Adversarial Self-Labeling :: IMMUTABLE HALF (the box + the judge).

Once locked, the loop may not edit this file. It defines:
  - the labeled source (seed of "true patterns" the critic learns from),
  - the unlabeled source (frames the loop must label — GT is NEVER exposed here),
  - fixed frame handling,
  - the ANCHOR: a small human-labeled hold-out — the ONLY keep/revert signal,
  - evaluate(): the immutable comparator (IoU accuracy on the anchor),
  - the exit conditions.

Prime Directive rule (b): the loop cannot touch the anchor or evaluate(). The judge
cannot be rewritten by the defendant, or the loop will lie to itself.

Epistemic stance (the acceptance principle): evaluate() never asserts a label is TRUE.
It reports the fraction of anchor frames the model gets "close enough" (IoU >= 0.5). Trust
is a survival rate, not a truth value. We only ever approximate until proven otherwise.

--- Synthetic bootstrap ---------------------------------------------------------
To run today with no downloads, prepare() generates a synthetic detection task: 32x32
grayscale frames each containing one bright rectangle (the "object"). Swap _synthesize()
for real frame extraction (video -> frames) + real annotations to move to plates/TU-DAT;
nothing downstream changes.
"""

from __future__ import annotations

import os
import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify once locked)
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
LABELED_NPZ = os.path.join(DATA_DIR, "labeled.npz")     # frames + GT boxes (seed patterns)
UNLABELED_NPZ = os.path.join(DATA_DIR, "unlabeled.npz")  # frames only — GT withheld by design
ANCHOR_NPZ = os.path.join(DATA_DIR, "anchor.npz")        # frames + GT boxes (the judge's ruler)

FRAME_HW = 32                # frame height == width
IOU_THRESHOLD = 0.5          # "close enough" for the acceptance principle — part of the metric

# Dataset sizes: the unlabeled pool deliberately exceeds the labeled seed, because the loop
# must generalize onto more than it was shown.
N_LABELED = 200
N_UNLABELED = 500
N_ANCHOR = 80
SEED = 42

# Exit conditions (rule (c): never run forever unsupervised)
TARGET_ACCURACY = 0.85       # stop at/above this anchor accuracy
MAX_ITERATIONS = 15          # hard safety cap
DRIFT_PATIENCE = 3           # consecutive non-improving rounds => stop (self-deception guard)


# ---------------------------------------------------------------------------
# Synthetic task generation (stand-in for real frame extraction + labeling)
# ---------------------------------------------------------------------------

def _make_sample(rng: np.random.Generator):
    """One frame: noisy background + one brighter rectangle. Returns (frame, box)."""
    img = rng.normal(0.2, 0.05, (FRAME_HW, FRAME_HW)).astype(np.float32)
    w = int(rng.integers(6, 14))
    h = int(rng.integers(4, 10))
    x0 = int(rng.integers(0, FRAME_HW - w))
    y0 = int(rng.integers(0, FRAME_HW - h))
    x1, y1 = x0 + w, y0 + h
    img[y0:y1, x0:x1] += rng.uniform(0.5, 0.8)   # the object is brighter than background
    img = np.clip(img, 0.0, 1.0)
    return img, np.array([x0, y0, x1, y1], dtype=np.int64)


def _synthesize(rng, n):
    frames = np.zeros((n, FRAME_HW, FRAME_HW), dtype=np.float32)
    boxes = np.zeros((n, 4), dtype=np.int64)
    for i in range(n):
        frames[i], boxes[i] = _make_sample(rng)
    return frames, boxes


def prepare() -> None:
    """Idempotent. Generate the three splits once. In a real run this decodes videos into
    frames and loads human annotations for the labeled seed + anchor."""
    if os.path.exists(LABELED_NPZ) and os.path.exists(UNLABELED_NPZ) and os.path.exists(ANCHOR_NPZ):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    lf, lb = _synthesize(rng, N_LABELED)
    uf, _ub = _synthesize(rng, N_UNLABELED)      # GT deliberately discarded — the loop must earn labels
    af, ab = _synthesize(rng, N_ANCHOR)
    np.savez_compressed(LABELED_NPZ, frames=lf, boxes=lb)
    np.savez_compressed(UNLABELED_NPZ, frames=uf)
    np.savez_compressed(ANCHOR_NPZ, frames=af, boxes=ab)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_labeled():
    """Labeled seed: frames + GT boxes. The critic learns 'true patterns' from these."""
    prepare()
    d = np.load(LABELED_NPZ)
    return d["frames"], d["boxes"]


def load_unlabeled():
    """Unlabeled pool: frames only. No ground truth is available to the loop, by design."""
    prepare()
    return np.load(UNLABELED_NPZ)["frames"]


def _load_anchor():
    prepare()
    d = np.load(ANCHOR_NPZ)
    return d["frames"], d["boxes"]


# ---------------------------------------------------------------------------
# The comparator (immutable) — IoU accuracy on the anchor. The only keep/revert signal.
# ---------------------------------------------------------------------------

def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(model) -> float:
    """Fraction of anchor frames where the model's box is 'close enough' (IoU >= threshold).

    Lives outside the mutable half so the loop cannot game its own judge.
    `model` need only expose predict(frame_2d) -> [x0, y0, x1, y1].
    """
    frames, gt = _load_anchor()
    hits = 0
    for f, g in zip(frames, gt):
        if iou(model.predict(f), g) >= IOU_THRESHOLD:
            hits += 1
    return hits / len(frames)
