"""
Direction 4 — Adversarial Self-Labeling :: MUTABLE HALF (the attempt).

B-generation: the discriminator is now a LEARNED, CO-EVOLVING adversary, not a fixed
statistical critic. Each round it is trained to tell real (labeled-seed) box placements
from corrupted ones, and — crucially — from the generator's own boundary proposals mined
last round. A fixed critic gets gamed; this one keeps getting retrained on the generator's
hardest cases, so it stays sharp as the generator improves. It remains NEGATIVE-ONLY at
inference: it only vetoes, it never proposes a box.

One round:
  train D (real vs corrupted vs generator hard-negatives)
    -> G proposes K variants -> D veto -> consensus -> augment survivors
    -> fine-tune G -> eval on immutable anchor -> keep/revert
    -> stash G's boundary proposals as next round's hard negatives (co-evolution)

The immutable anchor + drift guard remain the only truth signal and the only backstop
against the two GAN failure modes (D too strong -> starvation; D too weak -> slop).
"""

from __future__ import annotations

import json
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from prepare import (
    load_labeled, load_unlabeled, evaluate, iou,
    FRAME_HW, TARGET_ACCURACY, MAX_ITERATIONS, DRIFT_PATIENCE, HERE,
)

STATE_PATH = os.path.join(HERE, "loop_state.json")
CKPT_PATH = os.path.join(HERE, "best_model.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

# --- tunable knobs (mutable method) --------------------------------------------
# Proposals come from MC-DROPOUT: K stochastic forward passes expose the model's own
# uncertainty. Where it's confidently correct the boxes cluster; where it's guessing they
# scatter. Agreement across samples is therefore a real reliability signal — the property
# that timid jitter and photometric TTA both lacked (they produced false agreement).
K_MC = 8                      # stochastic forward passes per frame
DROPOUT = 0.3
CONF_THRESHOLD = 0.4          # learned-D veto line (negative-only: reject clear-wrong)
AGREE_IOU = 0.55
AUG_PER_SAMPLE = 2
G_EPOCHS = 8
D_EPOCHS = 3                  # keep D from overpowering G (GAN balance)
D_WEIGHT_DECAY = 1e-4
POS_LABEL = 0.9               # label smoothing -> less overconfident D
NEG_IOU_MAX = 0.3            # only CLEARLY-wrong boxes are negatives; the middle is left ambiguous
HARD_NEG_CAP = 150
FALSE_ACCEPT_PER_ROUND = 40  # bounded co-evolution intake (avoid cold-start poisoning)


# ---------------------------------------------------------------------------
# Generator: tiny CNN box regressor (LSTM lands when frames become video)
# ---------------------------------------------------------------------------

class BoxNet(nn.Module):
    """Lever 2: a FROZEN pretrained ResNet-18 feature extractor + a small trainable
    regression head. Grayscale frames are repeated to 3ch and ImageNet-normalized so the
    pretrained features apply. The backbone is never trained (tiny dataset); only the head
    learns. BatchNorm is kept in eval mode always — we toggle ONLY the head dropout for MC
    sampling, never model.train() (which would corrupt BN stats on single images)."""

    def __init__(self):
        super().__init__()
        bb = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(bb.children())[:-1])   # -> [B, 512, 1, 1]
        for p in self.features.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(128, 4),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def set_mc(self, on):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(on)                                        # dropout only; BN stays eval

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        with torch.no_grad():                                      # frozen backbone
            f = self.features(x)
        return torch.sigmoid(self.head(f))                         # head keeps grad

    def _to_box(self, n):
        b = (n * FRAME_HW)
        x0, y0 = min(b[0], b[2]), min(b[1], b[3])
        x1, y1 = max(b[0], b[2]) + 1, max(b[1], b[3]) + 1
        b = np.array([x0, y0, x1, y1], dtype=np.int64)
        b[:2] = np.clip(b[:2], 0, FRAME_HW - 2)
        b[2:] = np.clip(b[2:], b[:2] + 1, FRAME_HW)
        return b

    @torch.no_grad()
    def predict(self, frame_2d):                                   # deterministic (dropout off)
        self.eval()
        x = torch.from_numpy(np.asarray(frame_2d, dtype=np.float32))[None, None].to(DEVICE)
        return self._to_box(self(x)[0].cpu().numpy())

    @torch.no_grad()
    def predict_mc(self, frame_2d, k):                             # k dropout samples in one batch
        self.eval(); self.set_mc(True)
        x = torch.from_numpy(np.asarray(frame_2d, dtype=np.float32))[None, None].to(DEVICE)
        outs = self(x.repeat(k, 1, 1, 1)).cpu().numpy()            # [k, 4], each row a different mask
        self.set_mc(False)
        return [self._to_box(o) for o in outs]


# ---------------------------------------------------------------------------
# Learned discriminator: (frame, box-mask) -> plausibility. Real=1, corrupt=0.
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveMaxPool2d((8, 8)),                       # size-agnostic: 32px or 64px frames
            nn.Flatten(), nn.Linear(16 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def _box_mask(box):
    m = np.zeros((FRAME_HW, FRAME_HW), dtype=np.float32)
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(FRAME_HW, x1), min(FRAME_HW, y1)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 1.0
    return m


def _d_batch(pairs):
    """pairs: list of (frame, box) -> tensor [N, 2, H, W]."""
    arr = np.stack([np.stack([f, _box_mask(b)]) for f, b in pairs]).astype(np.float32)
    return torch.from_numpy(arr).to(DEVICE)


def _clearly_wrong(box, rng):
    """A corruption that is CLEARLY wrong (IoU < NEG_IOU_MAX with the real box). The
    ambiguous middle band is deliberately never labelled — D only learns to reject slop."""
    for _ in range(8):
        if rng.random() < 0.5:                                # random box
            w, h = rng.integers(3, 16), rng.integers(3, 16)
            x0, y0 = rng.integers(0, FRAME_HW - w), rng.integers(0, FRAME_HW - h)
            cand = np.array([x0, y0, x0 + w, y0 + h], dtype=np.int64)
        else:                                                 # heavy jitter
            off = rng.integers(-10, 11, size=4)
            cand = box + off
            cand[:2] = np.clip(cand[:2], 0, FRAME_HW - 2)
            cand[2:] = np.clip(cand[2:], cand[:2] + 1, FRAME_HW)
            cand = cand.astype(np.int64)
        if iou(cand, box) < NEG_IOU_MAX:
            return cand
    return cand


def train_discriminator(D, lab_frames, lab_boxes, hard_negs, rng):
    pos = list(zip(lab_frames, lab_boxes))
    neg = [(f, _clearly_wrong(b, rng)) for f, b in pos]       # clearly-wrong synthetic negatives
    neg += list(hard_negs)                                    # D's own false-accepts from last round
    if len(neg) > len(pos):                                   # class balance -> D isn't reject-biased
        idx = rng.choice(len(neg), size=len(pos), replace=False)
        neg = [neg[i] for i in idx]
    X = _d_batch(pos + neg)
    y = torch.cat([torch.full((len(pos),), POS_LABEL), torch.zeros(len(neg))]).to(DEVICE)
    opt = torch.optim.Adam(D.parameters(), lr=1e-3, weight_decay=D_WEIGHT_DECAY)
    D.train()
    for _ in range(D_EPOCHS):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy(D(X), y)
        loss.backward()
        opt.step()
    D.eval()
    with torch.no_grad():
        acc = ((D(X) > 0.5).float() == (y > 0.5).float()).float().mean().item()
    return acc


@torch.no_grad()
def d_score(D, frame, box):
    return float(D(_d_batch([(frame, box)]))[0].item())


# ---------------------------------------------------------------------------
# Generator proposals + consensus, judged by the learned D
# ---------------------------------------------------------------------------

def propose_and_filter(model, D, frame, rng):
    """MC-dropout proposals: K stochastic samples expose model uncertainty. D vetoes
    implausible ones; the survivors must then AGREE (low scatter) to be accepted. High
    scatter => the model is guessing => reject (real attrition correlated with error).
    Survivors that were let through but DISAGREE are D's false-accepts -> hard negatives."""
    samples = model.predict_mc(frame, K_MC)
    survivors = [b for b in samples if d_score(D, frame, b) >= CONF_THRESHOLD]
    if len(survivors) < 2:
        return None, survivors, len(survivors)
    ious = [iou(survivors[i], survivors[j]) for i in range(len(survivors)) for j in range(i + 1, len(survivors))]
    if np.mean(ious) < AGREE_IOU:                             # uncertain / let-through but unreliable
        return None, survivors, len(survivors)
    return np.stack(survivors).mean(0).astype(np.int64), survivors, len(survivors)


def augment(frame, box, rng):
    out = [(frame, box)]
    for _ in range(AUG_PER_SAMPLE):
        f = np.clip(frame + rng.normal(0, 0.03, frame.shape).astype(np.float32), 0, 1)
        if rng.random() < 0.5:
            f = f[:, ::-1].copy()
            b = np.array([FRAME_HW - box[2], box[1], FRAME_HW - box[0], box[3]], dtype=np.int64)
        else:
            b = box.copy()
        out.append((f, b))
    return out


def fine_tune(model, frames, boxes):
    x = torch.from_numpy(np.asarray(frames, dtype=np.float32))[:, None].to(DEVICE)
    y = torch.from_numpy(np.asarray(boxes, dtype=np.float32) / FRAME_HW).to(DEVICE)
    opt = torch.optim.Adam(model.head.parameters(), lr=1e-3)   # backbone frozen -> head only
    model.eval(); model.set_mc(True)                           # dropout on for training; BN stays eval
    for _ in range(G_EPOCHS):
        opt.zero_grad()
        loss = nn.functional.smooth_l1_loss(model(x), y)
        loss.backward()
        opt.step()
    model.set_mc(False)
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

    G = BoxNet().to(DEVICE)
    D = Discriminator().to(DEVICE)
    fine_tune(G, lab_frames, lab_boxes)
    best_state = copy.deepcopy(G.state_dict())

    state = load_state()
    rng = np.random.default_rng(123)
    hard_negs = []

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

        # --- co-evolve the adversary on real vs corrupt vs generator boundary cases ---
        d_acc = train_discriminator(D, lab_frames, lab_boxes, hard_negs, rng)

        # --- self-label the pool, judged by the learned D ---
        proposed = survived = accepted = 0
        pf, pb, false_accepts = [], [], []
        for frame in unlabeled:
            proposed += 1
            box, survivors, n_surv = propose_and_filter(G, D, frame, rng)
            survived += 1 if n_surv >= 2 else 0
            if box is not None:
                accepted += 1
                for af, ab in augment(frame, box, rng):
                    pf.append(af); pb.append(ab)
            elif n_surv >= 2:                                  # D let them through but they disagree
                false_accepts += [(frame, b) for b in survivors]
        if len(false_accepts) > FALSE_ACCEPT_PER_ROUND:        # bounded intake, no cold-start flood
            idx = rng.choice(len(false_accepts), size=FALSE_ACCEPT_PER_ROUND, replace=False)
            false_accepts = [false_accepts[i] for i in idx]
        hard_negs = (hard_negs + false_accepts)[-HARD_NEG_CAP:]  # co-evolution: sharpen D on its mistakes

        train_f = list(lab_frames) + pf
        train_b = list(lab_boxes) + pb
        loss = fine_tune(G, train_f, train_b)
        acc = evaluate(G)                                      # immutable judge

        kept = acc > state["best_accuracy"]
        if kept:
            state.update(best_accuracy=acc, drift_streak=0)
            best_state = copy.deepcopy(G.state_dict())
            torch.save(best_state, CKPT_PATH)
        else:
            state["drift_streak"] += 1
            G.load_state_dict(best_state)

        print(
            f"it={state['iteration']:02d} | d_acc={d_acc:.2f} hard_negs={len(hard_negs)} | "
            f"proposed={proposed} veto_survived={survived} accepted={accepted} "
            f"(train_N={len(train_f)}) | g_loss={loss:.4f} "
            f"anchor_acc={acc:.3f} best={state['best_accuracy']:.3f} | "
            f"{'KEEP' if kept else 'REVERT'} drift={state['drift_streak']}"
        )

        state["iteration"] += 1
        save_state(state)


if __name__ == "__main__":
    run()
