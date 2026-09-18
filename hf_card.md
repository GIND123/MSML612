---
license: apache-2.0
tags:
  - masked-diffusion
  - discrete-diffusion
  - constraint-satisfaction
  - sudoku
  - recurrent-transformer
  - from-scratch
---

# Recurrent Masked Diffusion for Constraint Satisfaction

**Constraint propagation is an iterative algorithm; masked diffusion models are
trained as fixed-depth denoisers. Making the denoiser recurrent — one
weight-shared block applied many times, supervised at every application —
reaches 99.70% on hard Sudoku with 212k parameters, where a 37.9M-parameter
feed-forward denoiser reaches 88.90%.**

Trained from scratch. No pretrained weights, no pretrained tokenizer.

Code: https://github.com/GIND123/MSML612

## Benchmark

1,000 held-out minimal Sudoku puzzles, 20–30 clues (mean 24.4). Uniqueness
verified after **every** cell removal; train/test solutions disjoint even up to
digit relabelling. Metric: board accuracy, all 81 cells correct.

## Results

| method | params | 1 pass | best |
|---|---|---|---|
| constraint propagation (no search) | – | – | 3.00% |
| greedy MRV | – | – | 5.90% |
| autoregressive prefix-LM | 2.4M | – | 0.30% |
| direct recurrent classifier* | 212k | 10.5% | 31.00% |
| recurrent, no input injection | 212k | – | 94.10% |
| feed-forward diffusion, MDLM | 37.9M | 4.40% | 88.15% |
| feed-forward diffusion, schedule-matched | 37.9M | 40.10% | 88.90% |
| 32 distinct layers, no weight sharing | 6.34M | 70.20% | 94.50% |
| recurrent diffusion, MDLM | 212k | 87.35% | 96.55% |
| recurrent, no deep supervision | 212k | 93.35% | 97.75% |
| **recurrent diffusion, full method** | **212k** | **97.35%** | **99.70%** |
| full backtracking search | – | – | 100% (423 nodes) |

\* Our reimplementation of the published recurrent classifier reaches 31%
against their reported 99.5%. **That is a failed reproduction on our side, not
evidence against their method**, and it is not used as a comparison.

## Ablations

Built as a ladder, each step changing exactly one thing:

| step | board | gain |
|---|---|---|
| feed-forward, 12 layers, 37.9M | 88.90% | – |
| feed-forward, **32 layers**, 6.34M | 88.95% | **+0.05** |
| + weight sharing (injection **off**), 212k | 94.10% | **+5.15** |
| + input injection | **99.70%** | **+5.60** |

**Depth alone is worth nothing** — 12→32 layers gains 0.05 points. Sharing one
block across 32 applications gains 5.15 at 30× fewer parameters; re-supplying the
input each application gains a further 5.60. The weight-sharing comparison holds
injection **off on both sides**, so the effects are separated rather than
confounded.

Removing either from the full method: deep supervision **−1.95**, schedule
matching **−3.15**.

## Recurrence extrapolates past its training depth

Trained at R = 32, evaluated elsewhere without retraining:

| R | 1 | 4 | 8 | 16 | **32** | **64** | 128 |
|---|---|---|---|---|---|---|---|
| board | 0.10% | 48.40% | 82.00% | 92.40% | **98.30%** | **99.40%** | 99.10% / 39.70% |

2× extrapolation is reliable; 4× collapses on one seed.

## The instructive failure

Both non-diffusion baselines reach **lower training loss** than our method and
solve almost nothing — AR at 0.0486 loss gives 0.30% boards, direct all-cell
supervision at 0.0722 gives 55%. AR's errors compound unrecoverably; all-cell
supervision is dominated by copying the given clues. Low loss, no capability.

## Honest limits

- Published comparison is across **different puzzle distributions**: our 20–30
  clue range sits inside the Recurrent Transformer's 17–34 but excludes the
  hardest 17-clue instances, so 99.70% at 212k reads as *competitive with* their
  99.5% at 211k, not better.
- **Input injection is not separated from weight sharing** in the 32-layer
  comparison; the isolating run is in progress.
- The SATNet split is not used: constraint propagation alone solves 100% of it.

## Negative results, reported

- **Any-budget conditioning fails.** Sampling the decoding budget during training
  is worse than the baseline at every budget on graphs (−4.2 at K=8, −31.7 at
  K=15) and destroys 20-digit addition outright (0.0% vs 99.9%), because it
  dilutes `t = 1` training 4.5×.
- **The entropy budget is mixed** — competitive on graphs, loses to fixed-`K` on
  text8.
- **Half the QM9 gap was our own representation.** Giving the baseline the atom
  count that DiGress samples first lifts it from 77.5% to 89.1%; the gap narrows
  from 22.2 to 10.7 points.
- **16-digit addition is unsettled** — a compute-matched baseline reaches 100% on
  one seed of two.
- **Our Sudoku prediction was refuted by our own framework.** We registered that
  entropy-budgeted decoding would beat fixed-`K` on Sudoku by the largest margin
  in the project. It loses on every arm by 0.35–1.40 points. The prediction
  contradicted the bound: Sudoku has a unique answer, so `TC = 0` and there is
  nothing for a mode-(ii) fix to repair.

## Audit

`src/audit.py`: 46 assertions across 11 sections, 0 failures. It caught two live
bugs that would each have produced confidently wrong numbers — a sampler emitting
the absorbing state so decoding silently stalled, and an aromatic-bond encoding
that made round-trip validity on *real* molecules 92.65% instead of 100%.
