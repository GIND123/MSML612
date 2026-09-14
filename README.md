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

One network then serves every budget instead of a family of networks each good
at a single point. It costs **one embedding lookup and no extra forward pass**,
which makes it cheaper than the fixed-`K` method it replaces — that one needed
two passes per step.

The ablation `anybudget_nc` samples `K` identically but withholds the
conditioning, which separates *mixing* budgets from *knowing* the budget. Both
arms are running; the prediction is that mixing alone improves the sequential
end while giving back the one-pass end, because one set of weights is then
serving contradictory targets, and that conditioning is what recovers both.

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

| digits | minimum depth for >90% at one pass | ratio vs 4-digit |
|---|---|---|
| 4 | 2 | 1.0× |
| 8 | 2 | 1.0× |
| **12** | **3** | **1.5×** |

The discriminating quantity is the ratio, not the absolute count (a wide layer
folds several scan levels into one). At 12 digits a ripple-carry account predicts
**3.0×** and the prefix-scan account predicts **1.79×**; the measurement is
**1.5×**. Ripple-carry is excluded. The 16-digit row is the clincher: ~4 layers
supports the scan account, ~8 supports linear growth.

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
- The answer region reveals whether the sum carries out (its width is *d* or
  *d*+1). Both arms receive this equally, so comparisons are unaffected, but it
  is one bit of leakage.
- The depth-scaling prediction in §5 is stated and currently being measured.
- The non-unique limit is **not** solved here, and we argue it cannot be.
