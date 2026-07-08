# CONTEXT.md

## The Short Answer to "What Am I Doing?"

We're coming up with a novel loop project. This involves three phases: planning, preparation of prepare.py and train.py and monitoring of the loop when exit conditions are reached or it needs to be killed artifically. 

---


### The Great Beyond


### Key Architectural Decisions 
- architectural decisions depend on the path chosen
- architecture should be the simplest possible to get job done, whether through code improvement or model training
- once information is gathered and ascertained for each step of process, prepare.py is locked - but train.py remains mutable for achieving goal
- this requires careful seperation of mutable and immutable information throughout the lifecycle, ahead of time

## What "Autoresearch" Could Mean Next

The interesting question is what can we create with this framework and iterate through to better an objective, not valuing perfection, but consistent improvement.

The other interesting approach is how can this be expanded or used in novel ways? These following directions illustrate examples - perhaps there is another way.  

### Direction 1: code loop that iteratively builds a functional tool that researches a prediction market
- finds percentage arbitage on short term closes or 
- finds opportunities and researches probability to create a statistical spread or
- finds the most likely money making opportunity given known or researched information
- constraints:
  - the loop is executed on a demo account 
  - the loop continues until a confirmed profit is made
  - the results are tabulated and the successful methodology logged
  - the loop exits on success or 10 failures resulting in losses

### Direction 2: builds a website from mutation for mastersentiment.com
- the loop itself mutates to produces four different prompts that build four websites in parallel
- we produce a working product
- we include something of interest that can be used at that existing domain address
- we include analysis of some type or machine learning analysis that makes sense for that domain
- constraints:
  - exits when they are all functional, not perfect
  - winner is chosen human in the loop
  - winner becomes the candidate for the next full iteration of mutated improvement, four more examples from the chosen version

### Direction 3: a philosophical engine
- combining known methodology from two existing and accepted/proposed papers and creates a new philosophical position
- articulated in terms of the previous philosophies, but shows genuine philosophical value
- combines philosophy from two or more cannons and introduces a new philosophical system 
- develops the idea into a distinct and important combinatorial or meta-governing idea
- constraints:
  - exits at draft of the paper, or completion of the paper

### Direction 4: a simple CNN-LSTM model
- bootstraps labeling from known dataset of close or vertical traffic using the minimum processing required (i.e. [unidatapro](https://huggingface.co/datasets/UniDataPro/united-states-license-plate-dataset)  or [tudat](https://github.com/pavana27/TU-DAT)  )
- downloads 3 videos for an independent test set
- either:
  - trains a cnn-lstm model on the static dataset, and morphs a cnn backbone into an lstm model to track video
  - finds another video dataset that is sufficiently labeled and trains from the full cnn-lstm
- confirms labeling is consistent
- trains for confirmed 55% accuracy or greater
- evaluates to confirm accuracy
- adjusts labeling to improve accuracy
- constraints:
  - improves at least 5% better accuracy each iteration
  - exits at 89% overall accuracy or 40% accuracy improvement or 20 iterations of model training and re-labeling for increased accuracy

---

## Current State of Each File

| File | Branch | Status |
|------|--------|--------|
| `prepare.py` | master | original
| `train.py` | master | original

---

## Constraints That Carry Forward

- `uv pip install` only - never `uv add`
- GTX 1070 (8 GB VRAM, Pascal)
- CUDA 12.6 max
- torch 2.7.1+cu126
