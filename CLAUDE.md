# loopme — Agentic Research Experiments

## What This Branch Is

A fresh exploration of autonomous research-loop patterns, forked from Karpathy's
`autoresearch` (nanochat) setup. The name is intentional: **the loop is the thing.**
Not the data, not the domain — the loop. The domain is swappable; the mechanism of
*try → measure → keep-or-revert → repeat* is what we're studying.

The original project is an LLM-pretraining loop (an agent edits `train.py`, trains for
a fixed 5-minute budget, and keeps the change only if `val_bpb` drops). The upstream
originals (`train.py`, `prepare.py`, `program.md`, `README.md`) are **archived on the
`master` branch** — the reference specimen, the box we differentiate *from*. They are
deliberately *not* carried on this branch so future-you doesn't trip on them; `git show
master:train.py` (etc.) to consult them. This file is the source of truth for the branch.

---

## The Prime Directive (the three laws of the box)

This framework is one **box** with a fixed contract. Many loop *strategies* (ralph loop,
mutation loop, N-parallel loop, …) may run on top of it — they're different ways of
searching, not different projects. Any strategy is legal iff it obeys all three laws:

- **a) What a loop CAN do** — freely regenerate `train.py` (blow it away and rewrite it
  each iteration to pursue the objective).
- **b) What a loop CANNOT do** — modify `prepare.py`. It holds the immutable constants and
  the measurement. Violating it destroys comparability between iterations; there's no
  longer a valid keep/revert.
- **c) What a loop SHOULD do** — exit on defined conditions. **Never run forever
  unsupervised.** A loop without an exit is a bug, not a feature.

Every loop, in every domain, reduces to one operation:
**generate a variant → measure it → compare to the incumbent → keep or revert.**
A domain is "loopable" only to the degree its comparator is *cheap and trustworthy*. Code
(`val_bpb`) is the ideal case (cheap + objective); that's exactly why it's off the table.
The hard/interesting domains (e.g. philosophy) are the ones where you must first *manufacture*
a trustworthy comparator (rubric, LLM-judge, novelty×coherence). No comparator = no loop,
only a generator.

---

## The Loop (mechanism)

Three phases: **planning → preparation → monitoring.**

1. **Planning** — pick a direction and its objective + exit conditions.
2. **Preparation** — build `prepare.py` (immutable once locked) and `iterate.py` (mutable).
   Careful separation of mutable vs. immutable state must be decided *ahead of time*.
3. **Monitoring** — run the loop until exit conditions are met or it needs killing.

Each direction lives in its own folder under `directions/` with this contract:
`prepare.py` (immutable — the box + the comparator) and `iterate.py` (mutable — the
attempt, blown away and rewritten each iteration). The comparator that decides keep/revert
must live in the immutable half, external to the mutation — *the judge cannot be a thing
the defendant is allowed to rewrite.*

Guiding principles:
- Simplest architecture that gets the job done — via code improvement *or* model training.
- Value **consistent improvement over perfection.** Each iteration must beat the last.
- Once info is gathered for a step, `prepare.py` locks; `iterate.py` stays mutable.
- `loop_state.json` (per direction) tracks iteration info across runs.

**On runtime / the 5-minute rule:** a 5-minute per-run cap is the *default* exit
condition, not the objective. It exists in Direction 0 only to make `val_bpb` comparable.
In other directions, time is just one available exit among several (profit confirmed, all
sites functional, paper drafted, N iterations, accuracy target). Keep a wall-clock cap as
a safety exit everywhere; treat it as the *goal* only where time is the resource being
compared (e.g. the CNN-LSTM).

### The original loop contract (Direction 0 — LLM pretraining)
- Objective: lowest `val_bpb` (validation bits/byte — vocab-size independent).
- Mutable: `train.py` (architecture, optimizer, hyperparameters, batch size, model size).
- Immutable: `prepare.py` (data, tokenizer, dataloader, `evaluate_bpb`, time budget, seq len).
- Per-run: edit → commit → `uv run train.py > run.log 2>&1` → `grep "^val_bpb:" run.log`.
- Keep if `val_bpb` improved (advance branch); else `git reset`. Log to `results.tsv`
  (tab-separated, untracked): `commit  val_bpb  memory_gb  status  description`.
- Crash = empty grep; `tail -n 50 run.log`, fix if trivial, else log `crash` and move on.
- Kill any run exceeding 10 minutes. Run autonomously; don't stop to ask.

---

## Candidate Directions ("What Autoresearch Could Mean Next")

The interesting question: what can we build with this framework and iterate toward a
moving objective? These are examples — there may be a better one.

**Direction 1 — Prediction-market research tool.** A code loop that iteratively builds a
tool to research a prediction market: find percentage arbitrage on short-term closes,
research probability to build a statistical spread, or find the most likely money-making
opportunity from known/researched info.
*Constraints:* runs on a demo account; continues until a confirmed profit; tabulate
results and log the winning methodology; exit on success or 10 loss-making failures.

**Direction 2 — Website by mutation for mastersentiment.com.** The loop mutates itself
into four prompts that build four websites in parallel, each a working product with
something useful for that domain plus a sensible analysis/ML component.
*Constraints:* exit when all four are functional (not perfect); human-in-the-loop picks
the winner; the winner seeds the next round of four mutated improvements.

**Direction 3 — The dialectical engine** → `directions/dir3-dialectic/` *(scaffolded, chosen).*
The **position is the artifact**, not a paper. Hegel as keep/revert: thesis → antithesis
(appended forever to an immutable ledger) → synthesis, kept only if it *sublates* — resolves
more of the ledger than its predecessor and survives the newest antithesis. The comparator
is a **falsifier**, not a verifier: it only ever rejects, never asserts truth/novelty; trust
is corroboration by survived challenge. *Constraints:* exit on dialectical closure, on
discovered irreducibility (itself a result), or an iteration cap. See its README.

**Direction 4 — Adversarial self-labeling (CNN-LSTM)** → `directions/dir4-labeling/` *(scaffolded, chosen).*
The loop **automates the labeler** and generalizes onto a new *unlabeled* source, then
fine-tunes a CNN-LSTM on labels it produced. Real research question: *how small can the
trusted anchor be before the loop lies to itself?* Guarded by a GAN-shaped critic — a
generator proposes 3×–5× label variants; a **negative-only discriminator** vetoes (knows
where the box *shouldn't* be, never asserts where it *should*); survivors that also agree
become pseudo-labels. Keep/revert is measured **only** on a small immutable human-labeled
anchor the loop never sees during labeling. *Constraints:* ≥89% anchor accuracy, or +40%
gain, or drift/iteration cap. See its README.

*Directions 1 (prediction market) and 2 (website mutation) remain on the shelf, not yet
scaffolded — code-only loops are deliberately excluded (too ideal).*

---

## Layout

```
CLAUDE.md                      source of truth (this file)
README.md                      framework overview
pyproject.toml / uv.lock       shared deps (torch pin — uv pip install only)
directions/
  dir3-dialectic/              prepare.py (immutable box) · iterate.py (mutable attempt) · README.md · ledger.jsonl
  dir4-labeling/               prepare.py (immutable box) · iterate.py (mutable attempt) · README.md · data/anchor …
```

Per direction: `prepare.py` = immutable box + comparator; `iterate.py` = the mutable
attempt (rewrite each loop); `loop_state.json` = runtime bookkeeping. Run a direction with
`uv run iterate.py` from inside its folder.

Upstream nanochat originals (`train.py`, `prepare.py`, `program.md`, `README.md`) live on
`master` — consult via `git show master:<file>`.

---

## Current State (as of 2026-07-08)

- Framework restructured into `directions/`. Dir 3 and Dir 4 are **scaffolded** — contracts
  and control flow are in place; the domain logic is `NotImplementedError`/`TODO` stubs.
  Neither loop has run yet.
- Dir 4 needs data wired into `prepare.py` (labeled source, unlabeled source, the anchor)
  and vision deps via `uv pip install`. Dir 3 needs source canons + the falsifier/LLM impl.
- ⚠️ **Hardware note for any GPU direction (Dir 4):** the archived nanochat `train.py`
  assumes Flash-Attention-3 + bf16, which will **not** run on the local **GTX 1070 (Pascal,
  sm_61)**. Dir 4 is a fresh CNN-LSTM (not that code), so build it Pascal-friendly from the
  start: `F.scaled_dot_product_attention`/plain conv-LSTM, fp16 not bf16, 8 GB VRAM budget.

---

## Constraints

- `uv pip install` only — never `uv add` (breaks the torch pin).
- GPU: GTX 1070, 8 GB VRAM, CUDA 12.6, torch 2.7.1+cu126, Pascal (sm_61). No bf16, no FA2/FA3.
- The comparator (`evaluate()` / the falsifier + ledger) lives in the immutable `prepare.py`
  and is never edited by the loop — that's rule (b) and the guard against self-deception.

---

## GitHub

`gh` CLI authenticated as `jtatman`. Remote: `https://github.com/jtatman/autoresearch`.
`.env` holds secrets (GitHub tokens, ANTHROPIC_API_KEY if needed). It is gitignored **and
untracked** as of the secrets fix. ⚠️ It was previously *committed* on the `choiceloop`
branch (from `d74ea2a`), so the tokens still live in that branch's history — never push
`choiceloop` without either purging `.env` from history or rotating those tokens.
