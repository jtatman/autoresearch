"""
Direction 4 — Adversarial Self-Labeling :: MUTABLE HALF (the attempt).

Blow this away and rewrite it each iteration to improve HOW the loop labels and trains.
It may use ONLY the immutable contract in prepare.py. The keep/revert signal is anchor
accuracy from prepare.evaluate() — nothing else.

One round:
  generate K label variants  ->  NEGATIVE-ONLY discriminator veto  ->  consensus filter
    ->  augment survivors (counter attrition)  ->  fine-tune  ->  eval on anchor  ->  keep/revert

Design notes that came out of the philosophy:
  * The critic never asserts where the box SHOULD be; it only rejects where it clearly
    shouldn't. It can only shrink the acceptable set — never assert a specific wrong box.
  * Rejection is severe (~coin-flip survival). Survivors are AUGMENTED so the trainable set
    doesn't drain to nothing — approximation compensating for its own attrition.
  * Trust is a survival rate, not truth. CONF_THRESHOLD is the "good enough" line; raise it
    for purity at the cost of yield, lower it for yield at the cost of drift.
"""

from __future__ import annotations

import json
import os
import copy
import numpy as np
import torch
import torch.nn as nn

from prepare import (
    load_labeled, load_unlabeled, evaluate, iou,
    FRAME_HW, TARGET_ACCURACY, MAX_ITERATIONS, DRIFT_PATIENCE, HERE,
)

STATE_PATH = os.path.join(HERE, "loop_state.json")
CKPT_PATH = os.path.join(HERE, "best_model.pt")

DEVICE = "cpu"                 # tiny synthetic net; real scale would move to the GTX 1070 (fp32/fp16)
torch.manual_seed(0)
np.random.seed(0)

# --- tunable knobs (part of the mutable method) --------------------------------
K_VARIANTS = 5                # label proposals per frame
CONF_THRESHOLD = 0.55         # discriminator veto line (survive-better-than-coin-flip)
AGREE_IOU = 0.55              # survivors must agree this much to form a consensus label
AUG_PER_SAMPLE = 2            # augmented copies per accepted pseudo-label (attrition offset)
FINETUNE_EPOCHS = 8


# ---------------------------------------------------------------------------
# Model: a tiny CNN box regressor. (The LSTM half lands when frames become video;
# synthetic frames are i.i.d., so temporal recurrence is a no-op here and omitted.)
# ---------------------------------------------------------------------------

class BoxCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
            nn.Flatten(), nn.Linear(16 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, 4),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))   # normalized [x0, y0, x1, y1] in [0, 1]

    @torch.no_grad()
    def predict(self, frame_2d):
        x = torch.from_numpy(np.asarray(frame_2d, dtype=np.float32))[None, None].to(DEVICE)
        n = self(x)[0].cpu().numpy()
        b = (n * FRAME_HW).astype(np.int64)
        x0, y0 = min(b[0], b[2]), min(b[1], b[3])
        x1, y1 = max(b[0], b[2]) + 1, max(b[1], b[3]) + 1
        return np.array([x0, y0, x1, y1], dtype=np.int64)


# ---------------------------------------------------------------------------
# Negative-only discriminator: learns "true patterns" from the labeled seed, then only
# ever REJECTS candidates that violate them. Never proposes a box.
# ---------------------------------------------------------------------------

class Critic:
    def __init__(self, frames, boxes):
        wh = boxes[:, 2:] - boxes[:, :2]
        areas = wh[:, 0] * wh[:, 1]
        aspect = wh[:, 0] / np.maximum(wh[:, 1], 1)
        self.log_area_mu, self.log_area_sd = np.log(areas).mean(), np.log(areas).std() + 1e-6
        self.aspect_mu, self.aspect_sd = aspect.mean(), aspect.std() + 1e-6
        # typical interior-vs-exterior brightness margin of a true object
        self.contrast_ref = np.mean([self._contrast(f, b) for f, b in zip(frames, boxes)])

    @staticmethod
    def _contrast(frame, box):
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(FRAME_HW, x1), min(FRAME_HW, y1)
        if x1 <= x0 or y1 <= y0:
            return -1.0
        inside = frame[y0:y1, x0:x1].mean()
        mask = np.ones_like(frame, dtype=bool)
        mask[y0:y1, x0:x1] = False
        outside = frame[mask].mean() if mask.any() else 0.0
        return float(inside - outside)

    def score(self, frame, box):
        """Plausibility in [0, 1]. LOW = reject. Product of soft memberships => any single
        gross violation collapses the score (rejection dominates)."""
        wh = np.array([box[2] - box[0], box[3] - box[1]], dtype=np.float64)
        if wh[0] <= 0 or wh[1] <= 0:
            return 0.0
        area, aspect = wh[0] * wh[1], wh[0] / max(wh[1], 1)
        s_area = np.exp(-0.5 * ((np.log(area) - self.log_area_mu) / self.log_area_sd) ** 2)
        s_aspect = np.exp(-0.5 * ((aspect - self.aspect_mu) / self.aspect_sd) ** 2)
        contrast = self._contrast(frame, box)
        s_contrast = 1.0 / (1.0 + np.exp(-12.0 * (contrast - 0.5 * self.contrast_ref)))
        return float(s_area * s_aspect * s_contrast)


# ---------------------------------------------------------------------------
# Generator + consensus: propose K jittered variants, keep those the critic doesn't veto,
# accept only if survivors AGREE (consensus). Disagreement or empty => attrition.
# ---------------------------------------------------------------------------

def _jitter(box, rng):
    off = rng.integers(-3, 4, size=4)
    b = box + off
    b[:2] = np.clip(b[:2], 0, FRAME_HW - 2)
    b[2:] = np.clip(b[2:], b[:2] + 1, FRAME_HW)
    return b.astype(np.int64)


def propose_and_filter(model, frame, critic, rng):
    """Return (consensus_box or None, n_survived_veto)."""
    seed = model.predict(frame)
    variants = [seed] + [_jitter(seed, rng) for _ in range(K_VARIANTS - 1)]
    survivors = [b for b in variants if critic.score(frame, b) >= CONF_THRESHOLD]
    if len(survivors) < 2:
        return None, len(survivors)
    # agreement: mean pairwise IoU of survivors
    ious = [iou(survivors[i], survivors[j]) for i in range(len(survivors)) for j in range(i + 1, len(survivors))]
    if np.mean(ious) < AGREE_IOU:
        return None, len(survivors)
    consensus = np.stack(survivors).mean(0).astype(np.int64)
    return consensus, len(survivors)


# ---------------------------------------------------------------------------
# Augmentation: bolster accepted pseudo-labels to offset rejection attrition.
# ---------------------------------------------------------------------------

def augment(frame, box, rng):
    out = [(frame, box)]
    for _ in range(AUG_PER_SAMPLE):
        f = np.clip(frame + rng.normal(0, 0.03, frame.shape).astype(np.float32), 0, 1)
        if rng.random() < 0.5:  # horizontal flip (box x-coords mirror)
            f = f[:, ::-1].copy()
            b = np.array([FRAME_HW - box[2], box[1], FRAME_HW - box[0], box[3]], dtype=np.int64)
        else:
            b = box.copy()
        out.append((f, b))
    return out


# ---------------------------------------------------------------------------
# Fine-tune on labeled seed + accepted (augmented) pseudo-labels.
# ---------------------------------------------------------------------------

def fine_tune(model, frames, boxes):
    x = torch.from_numpy(np.asarray(frames, dtype=np.float32))[:, None].to(DEVICE)
    y = torch.from_numpy(np.asarray(boxes, dtype=np.float32) / FRAME_HW).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(FINETUNE_EPOCHS):
        opt.zero_grad()
        loss = nn.functional.smooth_l1_loss(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    return float(loss.item())


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"iteration": 0, "best_accuracy": 0.0, "drift_streak": 0}


def save_state(s):
    with open(STATE_PATH, "w") as f:
        json.dump(s, f, indent=2)


def run():
    lab_frames, lab_boxes = load_labeled()
    unlabeled = load_unlabeled()
    critic = Critic(lab_frames, lab_boxes)

    model = BoxCNN().to(DEVICE)
    fine_tune(model, lab_frames, lab_boxes)     # warm start on the labeled seed only
    best_state = copy.deepcopy(model.state_dict())

    state = load_state()
    rng = np.random.default_rng(123)

    while True:
        stop = None
        if state["best_accuracy"] >= TARGET_ACCURACY:
            stop = f"reached target {TARGET_ACCURACY}"
        elif state["iteration"] >= MAX_ITERATIONS:
            stop = "iteration cap"
        elif state["drift_streak"] >= DRIFT_PATIENCE:
            stop = "anchor drift (self-deception guard tripped)"
        if stop:
            print(f"EXIT: {stop}")
            break

        # --- self-label the unlabeled pool with the current model ---
        proposed = survived = accepted = 0
        pf, pb = [], []
        for frame in unlabeled:
            proposed += 1
            box, n_surv = propose_and_filter(model, frame, critic, rng)
            survived += 1 if n_surv >= 2 else 0
            if box is not None:
                accepted += 1
                for af, ab in augment(frame, box, rng):
                    pf.append(af); pb.append(ab)

        # --- train on labeled seed + augmented pseudo-labels ---
        train_f = list(lab_frames) + pf
        train_b = list(lab_boxes) + pb
        loss = fine_tune(model, train_f, train_b)
        acc = evaluate(model)                    # immutable judge — the only vote that counts

        kept = acc > state["best_accuracy"]
        if kept:
            state.update(best_accuracy=acc, drift_streak=0)
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, CKPT_PATH)
        else:
            state["drift_streak"] += 1
            model.load_state_dict(best_state)    # revert to the last kept model

        print(
            f"it={state['iteration']:02d} | proposed={proposed} veto_survived={survived} "
            f"accepted={accepted} (train_N={len(train_f)}) | loss={loss:.4f} "
            f"anchor_acc={acc:.3f} best={state['best_accuracy']:.3f} | "
            f"{'KEEP' if kept else 'REVERT'} drift={state['drift_streak']}"
        )

        state["iteration"] += 1
        save_state(state)


if __name__ == "__main__":
    run()
