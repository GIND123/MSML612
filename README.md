# Two Failure Modes of Parallel Decoding in Masked Diffusion

**Parallel decoding fails for two different reasons that need opposite fixes.
Separating them is the method.**

Trained from scratch. No pretrained weights, no pretrained tokenizer, no
external teacher — including the evaluator used to score generated text.

---

## The claim

Masked diffusion's entire value proposition is **parallel decoding**: commit many
tokens per forward pass instead of one. Quality collapses as parallelism rises,
and roughly a dozen 2026 papers attack this at inference time with better
unmasking schedules — confidence ranking, entropy gating, dependency-aware
ordering.

Those methods treat one symptom as one disease. There are two:

| | failure mode | why it happens | fixable by |
|---|---|---|---|
| **(i)** | the fully-masked state is never trained | a *K*-pass decode evaluates the model at ratios `{1, (K−1)/K, …, 1/K}` and **starts from t = 1**. Continuous `t ~ U(0,1)` assigns that state probability **zero** | **training only** |
| **(ii)** | committed positions carry real mutual information | parallel decoding samples the **product** of marginals; the truth is the **joint** | **inference only** |

For (ii) the discarded dependence is the total correlation of the committed set
*S*:

```
TC(S) = Σᵢ H(xᵢ | c) − H(S | c)  ≤  Σᵢ H(xᵢ | c)
```

so **the summed conditional entropy of what you commit upper-bounds the error
you incur by committing it**. If every committed conditional is a point mass the
bound is zero and independent sampling reproduces the joint exactly. No training
objective removes this term without abandoning the data distribution.

Conversely, no inference-time reordering repairs a marginal that was never fit.

**Prior work applies inference-time fixes to both.** That is why it flatlines on
arithmetic, where (ii) is identically zero and all the loss is (i). The method
here fixes (i) at training time and composes with the correct (ii) fix at
inference — and the two are measured separately.

## The method

**Fix (i) — tell the model the budget it will be decoded at.**

A *K*-pass decode evaluates the model only at mask ratios `{1, (K−1)/K, …, 1/K}`.
Standard MDLM draws `t ~ U(0,1)`, which is *uniform over the same range* — so the
issue is not that uniform training under-weights heavy masking. It is that
continuous `U(0,1)` places **probability zero on t = 1**, the fully-masked state
every decode begins from. Drawing `t` from the discrete set instead puts an
**atom of mass 1/K** exactly there.

Matching a *single* `K` works, and it is what the addition results below use
(`K = 1`, i.e. always `t = 1`). But it is a **commitment to one decoding
budget**, and we measured the price: a model trained for one pass scores 99.8%
at one pass and **85.7% at twenty**. That is the baseline's own disease with the
roles reversed — intermediate partially-decoded states are off-distribution for
a model fitted only to `t = 1`. A method shaped like that can win one operating
point but cannot dominate a quality-vs-compute curve.

**Any-budget masked diffusion** removes the commitment. Sample the budget per
example, draw `t` from that budget's schedule, and pass the budget to the
denoiser as a log₂ embedding:

```
K ~ {1, 2, 4, …, 256}
t ~ {1, (K−1)/K, …, 1/K}
loss = CE( model(x_t, budget=K), x₀ )
```

It costs **one embedding lookup and no extra forward pass**, so it is cheaper
than the fixed-`K` method it replaces.

### It does not work

Sampling the budget during training was my proposed fix for fixed-`K`'s
commitment to one budget. Measured, it fails.

**20-digit addition**, against fixed-`K` (`K = 1`):

| training | 1 pass | 2 | 4 | 8 | 16 | 21 (sequential) |
|---|---|---|---|---|---|---|
| MDLM baseline | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | – |
| **fixed-K (K=1)** | **99.9** | 99.3 | 95.6 | 87.5 | 85.8 | – |
| any-budget | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

It does not flatten the crossover; it destroys the capability, scoring zero at
every budget including a fully sequential decode. The mechanism is arithmetic: a
one-pass decode starts at `t = 1`, and the training mass landing there is
`1.000` for fixed-`K` against `(1/9) Σ_K 1/K = 0.222` for uniform any-budget — a
**4.5× dilution** of the only signal that matters, on a task sitting exactly at
the learnability edge.

**Graphs**, mean over four families and two seeds, any-budget minus baseline:

| K = 1 | K = 2 | K = 4 | K = 8 | K = 15 |
|---|---|---|---|---|
| +0.3 | −2.7 | −2.9 | **−4.2** | **−31.7** |

Worse at every budget above one, and catastrophically worse at high budgets
(tree −81.6, 2-regular −43.7). Spreading training across nine budgets leaves
each one undertrained relative to a specialist, and the conditioning embedding
does not recover the difference — an unconditioned control sampling the same
budgets does no better.

**The negative result is the finding.** "One model for every decoding budget" is
not free, and at this scale it does not pay for itself at any budget. What
survives is the narrower claim the addition results already supported: *train
the schedule you intend to decode at.* Matching a single known budget works
extremely well (99.9% against a baseline's 0.0%); trying to serve all of them at
once works worse than either.

**Fix (ii) — entropy-budgeted commitment.** At each pass, rank still-masked
positions by conditional entropy and commit the longest low-entropy prefix whose
cumulative entropy stays within `B` nats. By the bound above, `B` *directly caps
the dependence discarded at that pass*, so it is an interpretable dial rather
than a tuned heuristic: `B → 0` recovers sequential decoding, `B → ∞` recovers
one-shot parallel decoding, and the curve between them is the quality/compute
trade-off. The number of passes becomes data-dependent, which is the point.

Cost of (i): one embedding lookup, no extra forward pass. Cost of (ii): none.

## What each regime isolates

| Regime | Example | mode (i) | mode (ii) | one pass? |
|---|---|---|---|---|
| **Non-unique answer** | groups must agree, value free | absent | **maximal** | never — information-theoretic |
| **Unique, shallow deps** | morphological inflection | small | small | mostly already works |
| **Unique, deep deps** | *n*-digit addition | **dominant** | zero | fails → **the method fixes it** |
| **Mixed** | natural text | present | present | **where the two must compose** |
| **Exactly measurable** | structured graphs | present | **dominant** | **where the bound is verifiable** |

---

## 1. Addition — mode (i) in isolation

Addition is the clean case: the answer is fully determined by the operands, so
every conditional is a point mass, mode (ii) is identically zero, and **all** of
the loss is mode (i). Digit *i* needs the carry out of digit *i−1*, so the
dependency is deep.

**Exact match at ONE forward pass.** Depth scaled as log₂(digits); lr 1e-4 with
long warmup, which matters — see §5.

| digits | MDLM baseline | matched schedule (ours) | seeds >90% (base → ours) |
|---|---|---|---|
| 8 | 100.0 | 100.0 | 4/4 → 4/4 (saturated) |
| **12** | **76.7** | **100.0** | 3/4 → **4/4** |
| **16** | **25.0** | **75.0** | 1/4 → **3/4** |
| **20** | **0.0** | **99.9** | 0/3 → **3/3** |
| 24 | 0.0 | 33.6 | 0/3 → 1/3 |

The 12- and 16-digit rows are true parallelism gaps: the baseline demonstrably
learned the task and simply cannot commit it in one pass.

**The 20- and 24-digit rows are a different and stronger claim, and are labelled
as such.** The standard pass grid stops at 16, which for a 21-slot answer is not
a sequential decode, so `src/seq_check.py` re-decodes every run at one token per
pass. That settles it:

| digits | mode | @1 pass | fully sequential | reading |
|---|---|---|---|---|
| 20 | MDLM | 0.0 | **0.0** | **never learned the task** |
| 20 | ours | **99.8** | 85.7 | learned it |
| 24 | MDLM | 0.0 | **0.0** | **never learned the task** |
| 24 | ours | 33.7 | 33.3 | learned it |

The baseline is at zero **at every budget, on every seed**. Beyond 16 digits it
does not fail to *parallelise* — it fails to *learn*. Uniform-`t` training
touches the fully-masked state with probability zero, and a 20-long carry chain
never gets enough gradient where it matters. So at these lengths fix (i) is a
**sample-efficiency** fix, not a parallelism fix, and the 99.8-vs-0.0 gap must
not be quoted as the latter.

24 digits remains at the edge: one seed of three reaches 100%, the others near
zero. Reported as the variance it is.

### The method inverts the trade rather than closing it

At 20 digits our own model scores **99.8% at one pass but 85.7% at twenty**, and
one seed drops from 99.5% to 57.0%. More passes make it *worse*.

This is the mechanism running in reverse, and it is the strongest available
evidence for the account. A model trained with mass at `t = 1` is fitted to the
fully-masked state; walking it through intermediate partially-decoded states
visits ratios it never trained on — exactly the baseline's disease, with the
roles swapped. A method that simply dominated at every budget would be weaker
evidence, because it would be consistent with "we just trained better".

It also means the honest framing is not "we close the parallelism gap" but **"the
schedule you train determines the budget you can decode at"**, in both
directions.

### No inference-time method closes the gap

Mode (ii) is zero here, so the prediction is that **every** inference-time
strategy lands on the same number and extra compute buys nothing.

**12-digit addition**, the *baseline* model decoded every way:

| strategy | 1 pass | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| confidence | 76.7 | 76.4 | 76.4 | 76.3 | 76.2 |
| entropy | 76.7 | 76.4 | 76.4 | 76.3 | 76.2 |
| entropy-gated | 76.7 | 76.4 | 76.4 | 76.2 | 76.2 |
| margin | 76.7 | 76.4 | 76.4 | 76.3 | 76.2 |
| random | 76.7 | 77.2 | 76.9 | 76.8 | 76.4 |
| **ours (training-time)** | **100.0** | 100.0 | 100.0 | 99.9 | 99.9 |

**16-digit:** every inference-time strategy 25.0 at one pass and 25.0 at *any*
budget; ours 75.0.

Two things stand out. Every strategy family lands on the same number, and
**sixteen times the inference compute buys nothing** — at 12 digits it is
marginally worse. The random-order row is the cleanest statement of it: choosing
positions at random does as well as every principled ordering heuristic, because
ordering is not what is broken.

This is the control that makes the text8 result in §3 meaningful: the same
entropy-gated rule that does nothing here is genuinely useful there, because
there mode (ii) is nonzero.

## 2. Inflection — both modes small

Morphological inflection (CoNLL-SIGMORPHON 2017) has unique answers, but forms
are ~10 characters with shallow local dependencies. Six typologically diverse
languages, 2 seeds each:

| language | baseline @1 | ours @1 | baseline gap 1→32 | **ours gap** | gap removed |
|---|---|---|---|---|---|
| Georgian | 94.6 | 94.1 | −0.0 | −0.0 | n/a (saturated) |
| Russian | 71.6 | 72.3 | +2.2 | **−0.3** | 114% |
| German | 76.0 | 78.4 | +3.2 | **+0.2** | 94% |
| Turkish | 78.7 | 82.3 | +3.5 | **+0.8** | 77% |
| Finnish | 51.4 | 51.9 | +7.7 | **+4.9** | 36% |
| Navajo | 28.0 | 26.9 | +8.0 | **+4.5** | 44% |
| **mean** | **66.7** | **67.7** | **+4.1** | **+1.7** | **59%** |

The honest reading: **the method removes 59% of the parallelism penalty at no
cost to one-pass accuracy (66.7 → 67.7), but it does not raise the ceiling** —
at 32 passes it is 1.4 points *below* the baseline (69.4 vs 70.8). Concentrating
the schedule at high mask ratios necessarily spends less on the low-ratio states
that sequential decoding uses.

The dose-response is the interesting part and it goes the right way: **Georgian
has no gap and the method changes nothing; the languages with the largest gaps
get the largest absolute reductions.** A method that "helped" on Georgian would
be evidence it was doing something other than what is claimed.

Reporting Georgian and English matters: it shows where the method has nothing to
add, which is what makes the gains elsewhere credible.

## 3. text8 — where the two fixes must compose *(running)*

Addition isolates mode (i); the free-choice probes isolate mode (ii). Natural
text contains both **in the same sequence**: closing a word, a suffix or a
function word is nearly deterministic given context, while the choice of the
next content word carries real entropy. So text is where the claim that the two
fixes *compose* is actually tested, as a 2×2:

| | fixed-*K* commit | entropy-budgeted commit |
|---|---|---|
| uniform-`t` training | (a) | (b) |
| matched training | (c) | (d) |

Prediction: (c) < (a) and (d) < (b) on generative perplexity at equal compute,
and the two gains are largely independent.

**A second prediction, registered before the runs finish.** If fix (i) works by
placing training mass on the ratios a decode actually visits, then matched-`K`
training should show a **crossover**, not a uniform win: better than uniform-`t`
near NFE ≈ K, and *worse* far from it. Training at K = 8 puts no mass at the
ratios a 256-pass decode visits, so uniform-`t` should win at high NFE. A method
that simply dominated everywhere would be evidence the mechanism is something
other than the one claimed.

By the same logic the `t = 1`-heavy variant should be the worst of the three on
text at moderate NFE, because at `t = 1` an unconditional model can only learn
character frequencies — there is no context to condition on. That variant is
included to be refuted, not to win.

Standard setup: text8, the conventional 90M/5M/5M character split, 27-symbol
vocabulary, 256-character windows. Metrics:

- **validation bits-per-character** (the MDLM bound) — the standard likelihood
  number, and the check that sample quality is not bought by wrecking the model;
- **generative perplexity vs. number of function evaluations**, scored by a
  held-out autoregressive character LM trained from scratch on text8 itself
  (a word-piece model trained on web text would judge character-level artifacts
  through a tokenizer that never sees them);
- **entropy and distinct-4-gram rate**, reported beside it, because generative
  perplexity alone is trivially gamed by degenerate repetition.

Against the published **confidence-threshold** rule, not a top-*k* stand-in.

## 3b. Graphs — where the bound stops being an assumption *(partial)*

The weakness of everything above is that the central quantity is never measured.
On addition `TC = 0` by construction; on text `H(S|c)` is not computable. So the
entropy budget rests on a bound the experiments never check.

Structured graphs close that. A graph on *n* nodes is a binary sequence of
`n(n−1)/2` edge indicators — the same machinery, unchanged — but for *n* ≤ 7 the
**entire family enumerates**, making every quantity exact: true marginals,
`H(S) = log₂|F|`, `TC`, and the one-pass ceiling

```
V* = Σ_{g valid} ∏ᵢ pᵢ(gᵢ)      (what perfect marginals achieve in one pass)
```

**Proposition.** For `P` uniform on a family `F`, `V* ≥ 2^(−TC)`, with equality
iff `∏ᵢ pᵢ` is constant on `F`. *Every bit of total correlation among jointly
committed variables at most halves the one-pass success probability.* Proof and
verification in [`THEORY.md`](THEORY.md) §5.1 — the inequality holds across all
nine family/size combinations, and equality holds **to five decimal places** on
exactly the fixed-edge-count families, failing precisely where edge counts vary,
as the Jensen step requires.

### Both failure modes, separated by measurement

This is what no earlier task could do. Perfect matchings on 6 nodes,
`TC = 6.92 bits`, `V* = 0.825%`:

Four families, two seeds each, one-pass validity against the exact ceiling:

| family | TC (bits) | V* | achieved @1 pass (95% CI) | reading |
|---|---|---|---|---|
| 2-regular | 8.43 | 0.289% | 0.317% [0.217, 0.465] | **at the ceiling** |
| matching | 6.92 | 0.825% | 0.769% [0.602, 0.983] | **at the ceiling** |
| tree | 3.43 | 9.249% | 9.204% [8.597, 9.849] | **at the ceiling** |
| **bipartite** | **0.11** | **93.632%** | **5.884% [5.395, 6.414]** | **16× below** |

This is the decomposition working as a measurement instrument, and it is sharper
than predicted.

**Where dependence is high, models land exactly on the information-theoretic
ceiling.** Three families, ceilings spanning 32× (0.289% to 9.249%), and in every
case the interval contains `V*`. The residual failure there is provably mode (ii)
and no training objective can recover it.

**Where dependence is nearly absent, the failure is entirely trainable — and
nothing we tried trains it.** Bipartite has `TC = 0.11` bits, so one-pass
decoding should be almost free at 93.6%, yet both the baseline and the
budget-conditioned model reach only 5.9%. That is **87.7 points of purely mode-(i)
headroom left on the table**, and it is the clearest open problem this framework
produces: an exactly quantified gap that is known to be trainable and that no
method here closes.

Coverage confirms none of this is degeneracy — 100% of the matching and
2-regular families are recovered at K = 8.

One caution the exactness makes visible: `tree`/any-budget scores 10.06%
[9.43, 10.73], statistically **above** `V* = 9.25%`. `V*` bounds models whose
marginals match the data's, not all models — exceeding it means the marginals
have drifted, trading distributional fidelity for validity. See
[`THEORY.md`](THEORY.md) §5.1.

Coverage confirms this is not degeneracy: 100% of the 15-member family is
recovered at K = 8 and under the entropy budget. (Reporting "uniqueness" as
unique/valid would have shown 0.4% and read as collapse — with `|F| = 15`,
coverage is the meaningful metric.)

The entropy budget also dominates fixed-K under both training modes, reaching
100% validity at 11 passes where fixed-K needs 15.

**Status: 3 of 16 runs.** The remaining families span `TC` from 0.11 to 8.44
bits, which is where the dose-response claim — larger `TC`, lower ceiling, more
of the loss unfixable — is actually tested. One non-monotonicity in the baseline
(83.6% at K=4 falling to 76.7% at K=8) is single-seed and not yet trustworthy.

### Phase 2: QM9

The synthetic families give an exact ceiling on an artificial task. QM9 gives a
real benchmark whose metric is unambiguous — RDKit sanitisation enforces valency
at every atom at once, a genuine joint constraint, with no evaluator model in
the loop. A molecule is 9 node slots plus 36 edge slots = 45 tokens, so training
from scratch is cheap and the comparison isolates schedule and decoding rule
rather than scale. The axis is the one the graph-diffusion literature competes
on: **validity retained as denoising steps fall.**

The encoding is verified lossless first — storing aromatic bonds loses per-atom
aromaticity flags and put round-trip validity on *real* molecules at 92.65%,
which would have measured generated samples against a target the representation
could not reach. Kekulised and charge-filtered, round-trip is 100.00% while
keeping 99.1% of the dataset.

## 4. The limit that cannot be trained away

Where a group of positions must agree but *which* value they take is free, the
likelihood-optimal marginal at each position is uniform over the valid values.
Two independent draws agree with probability 1/V. This is mode (ii) at its
maximum, and **no decoding schedule and no training objective can fix it**.

Tested rather than assumed: **66 runs** sweeping an entropy penalty designed to
break the symmetry, across weights spanning two orders of magnitude. Weak
settings changed nothing; strong settings destroyed the model, failing even at
fully sequential decoding.

This boundary is part of the contribution. A method claiming to fix this regime
would be claiming to beat an information-theoretic bound.

## 5. Why it should work — and the falsifiable prediction

See [`THEORY.md`](THEORY.md). Addition's carry is a **prefix scan**, computable
in **O(log n)** depth by carry-lookahead, and constant-depth log-precision
transformers are confined to uniform **TC⁰**, which contains addition. One-pass
decoding is therefore **expressible** at depth ~log n — the obstacle is
optimisation, not expressivity. Sequential decoding is best understood as
**borrowing depth from the decoding loop**: a *K*-pass decode has effective
depth ≈ L·K.

The account predicts minimum depth growing with **log n**, not n — cleanly
distinguishable from a ripple-carry account (linear) or an impossibility account
(no depth suffices). Measured, over {2,3,4,6,8,12} layers × 2 seeds:

| digits | 2 layers | 3 layers | minimum depth | ratio vs 4-digit |
|---|---|---|---|---|
| 4 | 100 | 100 | 2 | 1.00× |
| 8 | 99 | 100 | 2 | 1.00× |
| 12 | 48 | 100 | 3 | 1.50× |
| **16** | **0** | **100** | **3** | **1.50×** |

The discriminating quantity is the ratio, not the absolute count, since a wide
layer can fold several scan levels into one. At 16 digits:

| account | predicted ratio | consistent with 1.50×? |
|---|---|---|
| **prefix scan (ours)** | 2.00× | **yes** — measured growth is if anything slower |
| ripple carry | 4.00× | **no** |
| not expressible at any depth | — | **no**, 3 layers suffice |

**Ripple-carry is excluded.** Quadrupling the operand length from 4 to 16 digits
costs **one extra layer**, not four times the depth.

The 16-digit transition is also a clean threshold rather than a gradual ramp —
**0% at two layers, 100% at three** — which is the signature of a computation
that is depth-limited rather than capacity-limited or data-limited. The minimum
depth is now determined at every length tested.

## 6. A confound that nearly produced the wrong paper

Earlier runs showed the method "failing" at 10+ digits. It was a **learning-rate
artifact**:

| layers | lr | warmup | mean @1 pass | seeds >90% |
|---|---|---|---|---|
| 10 | 3e-4 | 500 | 47.3 | 1/4 |
| 10 | **1e-4** | 500 | **100.0** | **4/4** |
| 10 | 3e-4 | **4000** | **100.0** | **4/4** |

Deeper models need a lower learning rate and longer warmup. Without this check
we would have reported that the method does not generalise, and a depth sweep
run at the broken setting would have faked a clean "depth doesn't help" curve.

## 7. Audit

`src/audit.py` gates every claim and reports **0 failures**. Eight sections:
arithmetic correctness; fixed-width answer regions; the prompt never masked; the
metric ungameable by constant output; probe ground truth valid while degenerate
output scores zero; inflection train/dev disjointness; ≥2 seeds behind every
reported group; **decoder contracts** (no `[MASK]` survives, decoding always
progresses, the entropy budget is monotone in `B`, context is never
overwritten); and **text8 split integrity**.

It earns its place. The decoder section caught a live bug in this work:
`torch.multinomial` was sampling over the *full* vocabulary including the
absorbing state, so positions were "committed" as `[MASK]`, stayed masked, and
decoding silently stalled — 15 of 32 positions unfilled at `K=1`, and the
entropy budget non-monotone at `[35, 35, 37, 17, 3]` passes. The fix is the SUBS
parameterisation: the denoiser places zero probability on `[MASK]`. Afterwards
the budget is exactly monotone, `[32, 32, 32, 16, 1]`.

Two earlier projects in this line produced confident wrong numbers — an
answer-length leak and the learning-rate artifact above — neither visible in the
accuracy tables.

## 8. Layout

```
src/
  data.py          addition + dependency-controlled probes
  sigmorphon.py    CoNLL-SIGMORPHON 2017 inflection loader
  text8.py         text8 loader, standard 90M/5M/5M split
  model.py         bidirectional transformer denoiser
  diffusion.py     objectives (fix (i)) + decoders (fix (ii))
  train_add.py     addition trainer
  train_sig.py     inflection trainer
  train_text8.py   text8 trainer + quality-vs-compute evaluation
  train_eval_lm.py from-scratch AR evaluator for generative perplexity
  train.py         probe trainer (the unfixable-limit experiments)
  seq_check.py     did the baseline fail to parallelise, or fail to learn?
  beat_baselines.py  head-to-head vs inference-time methods
  audit.py         the gate on every number above
  collect.py       tables and figures
slurm/             grids, plus unattended finalize and watchdog
```

```bash
python src/train_add.py   --mode full --digits 12 --lr 1e-4 --warmup 4000 --out runs/demo
python src/train_sig.py   --lang turkish --mode full --out runs/tur
python src/train_text8.py --mode full --evallm runs/evallm --out runs/t8
python src/audit.py
```

## 9. Honest limitations

- Addition and inflection are probes, not applications; the claim is about the
  model class, and text8 (§3) is the test of whether it survives real data.
- On inflection the method **lowers** the 32-pass ceiling by 1.4 points while
  removing 59% of the one-pass penalty. That trade is inherent to
  reallocating the schedule and is reported, not hidden.
- The 20- and 24-digit results are sample-efficiency claims, **not** parallelism
  claims: `seq_check.py` confirms the baseline scores 0.0% at full sequential
  depth too. Quoting them as parallelism gaps would misrepresent them.
- Our own method loses accuracy when decoded at budgets far from the one it was
  trained for (99.8% at one pass, 85.7% at twenty). This is predicted by the
  account but it is a real limitation: the training `K` is a commitment to a
  decoding budget, not a free win.
- The any-budget fix for that commitment **fails outright on addition** (0.0% at
  every budget, against 99.9% for fixed-`K`), because spreading the schedule
  dilutes `t = 1` training 4.5× on a task that sits at the learnability edge. It
  helps on graphs and is untested on text. Uniform-`K` is not a safe default.
- The answer region reveals whether the sum carries out (its width is *d* or
  *d*+1). Both arms receive this equally, so comparisons are unaffected, but it
  is one bit of leakage.
- The depth-scaling prediction in §5 is stated and currently being measured.
- The non-unique limit is **not** solved here, and we argue it cannot be.
