---
license: apache-2.0
tags:
  - masked-diffusion
  - discrete-diffusion
  - parallel-decoding
  - constraint-satisfaction
  - text8
  - qm9
  - sudoku
  - from-scratch
---

# Two Failure Modes of Parallel Decoding in Masked Diffusion

Everything here is trained from scratch: no pretrained weights, no pretrained
tokenizer, and the autoregressive model used to score generated text is itself
trained from scratch on the same corpus.

Code: https://github.com/GIND123/MSML612

---

## The claim

Parallel decoding in masked diffusion loses accuracy for two reasons that need
**opposite** fixes, and the literature treats them as one.

| | failure mode | fixable by |
|---|---|---|
| **(i)** | the fully-masked state is never trained — a *K*-pass decode starts at `t = 1`, and continuous `t ~ U(0,1)` assigns that state probability **zero** | **training only** |
| **(ii)** | committed positions carry real mutual information — parallel decoding samples the product of marginals when the truth is the joint | **inference only** |

Prior work applies inference-time fixes to both, which is why those methods
flatline on arithmetic, where (ii) is identically zero.

## The bound

For a data distribution `P` uniform on a family `F`, let `V*` be the one-pass
validity achievable from the **true** marginals. Then

```
V*  ≥  2^(−TC),     TC = D_KL( P ‖ ∏ᵢ pᵢ ),   equality iff ∏ᵢ pᵢ is constant on F
```

**Every bit of total correlation among jointly committed variables at most halves
the one-pass success probability.**

Verified by exhaustive enumeration over graph families, with the enumerator itself
checked against Cayley's formula (spanning trees at n = 5, 6, 7 → 125, 1296,
16807):

| family | TC (bits) | V* | 2^(−TC) | ratio |
|---|---|---|---|---|
| bipartite (n=7) | 0.026 | 98.2816% | 98.1817% | 1.001 |
| tree (n=6) | 3.435 | **9.2488%** | **9.2488%** | **1.00000** |
| matching (n=6) | 6.922 | **0.8246%** | **0.8246%** | **1.00000** |
| 2-regular (n=7) | 10.423 | **0.0728%** | **0.0728%** | **1.00000** |

The inequality holds in every case; equality holds to five decimals on exactly
the fixed-edge-count families, and fails precisely where edge counts vary — which
is what the Jensen step requires.

## Finding 1 — train the schedule you intend to decode at

| task | baseline | ours |
|---|---|---|
| 20-digit addition, 1 pass | **0.00%** (also 0.00% at **2× compute**, both seeds) | **99.9%** (3/3 seeds) |
| text8, NFE 16 | 116.88 gen-ppl | **92.74** (−20.7%) |
| text8, validation BPC | 1.7292 | **1.7010** |

The text8 gain appears in generative quality **and** likelihood simultaneously,
so it is not bought by trading one for the other.

A 48-run ablation isolates the mechanism: the `t = 1` term alone removes **49%**
of the parallelism penalty, against 17% for reshaping the rest of the schedule.

## Finding 2 — one-pass decoding needs absolute position information

At `t = 1` every input token is `[MASK]`, so the sequence is constant. With a
relative encoding the attention score depends only on `(i − j)`, and identical
value vectors give identical outputs at every position. **Position-dependent
marginals are inexpressible at any training budget.**

| condition | learned (APE) | parameter-free (sinusoidal) |
|---|---|---|
| marginals **vary** | **+88.037 pts** | **+88.074 pts** |
| marginals **constant** (control) | **−0.024 pts** | **+0.000 pts** |

A ~3,700× dissociation. The parameter-free variant matching the learned one rules
out capacity; the control rules out "absolute is simply better". Predicted 5/5
before measurement, and the treated arm lands at **100.3% of `V*`** — at the
bound, not merely higher.

## Benchmark: SATNet Sudoku

9x9 Sudoku, SATNet's conventional 9,000/1,000 split, **board accuracy** = all 81
cells correct. Published number: **98.3%** (Wang et al., ICML 2019).

Un-augmented arm, which is the only like-for-like comparison:

| denoising passes | board accuracy |
|---|---|
| **1** | **87.85%** |
| **2** | **98.65%** |
| 8 | 99.70% |
| **32** | **100.00%** |

Two passes exceed the published number; 32 passes solve **all 1,000** test
puzzles exactly. One pass — the entire grid committed at once — solves 87.85%.

Leakage audited first: 0/1000 test puzzles in training, 0/1000 test solutions in
training, and 0/1000 test solutions matching a training solution even up to digit
relabelling. All 10,000 solutions are distinct.

**Scoped honestly:** this comes from the *plain MDLM baseline* with fixed-`K`
decoding. Neither finding above contributes — the standard formulation already
saturates this benchmark. It is a result for masked diffusion on SATNet Sudoku,
not evidence for our method.

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
