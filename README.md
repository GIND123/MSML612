# One-Pass Masked Diffusion

**When can a masked diffusion language model commit an entire answer in a single
forward pass — and when is that provably impossible?**

Trained from scratch. No pretrained weights, no pretrained tokenizer, no external
teacher.

---

## The question

Masked diffusion's whole value proposition is **parallel decoding**: commit many
tokens per pass instead of one. In practice quality collapses as parallelism
rises, and roughly a dozen 2026 papers attack this at inference time with better
unmasking schedules.

Training small models from scratch, where the dependency structure is under our
control, shows those methods are treating **three different situations as one**.
The organising variable is not "how hard is the task" but **how deep are the
dependencies among the tokens being committed together**.

| Regime | Example | One pass? | Fixable by training? |
|---|---|---|---|
| **Non-unique answer** | groups must agree, value free | never | **No — information-theoretic** |
| **Unique, shallow deps** | morphological inflection | **already works** | no method needed |
| **Unique, deep deps** | *n*-digit addition (carry chain) | fails | **Yes — this is the method** |

## 1. The limit that cannot be trained away

Where a group of positions must agree but *which* value they take is free, the
likelihood-optimal marginal at each position is uniform over the valid values.
Two independent draws — which is exactly what parallel decoding takes — agree
with probability 1/V. **No decoding schedule and no training objective can fix
this**, because succeeding would mean abandoning the data distribution.

We tested it rather than assuming it: **66 runs** sweeping an entropy penalty
designed to break the symmetry, across weights spanning two orders of magnitude.
Weak settings changed nothing; strong settings destroyed the model, failing even
at fully sequential decoding. Measured, not asserted.

## 2. The regime where no method is needed

Morphological inflection (SIGMORPHON 2017, 52 languages) has unique answers, but
forms are ~10 characters with shallow local dependencies.

| language | 1 pass | 32 passes | gap |
|---|---|---|---|
| English | **95.3** | 95.2 | none |
| German | 77.2 | 78.3 | +1.1 |
| Turkish | 76.4 | 81.8 | +5.4 |

The **plain baseline already decodes in one pass**, at accuracy competitive with
published transformer baselines (~95% on English). Reporting this matters: a
method evaluated only here would look like it does nothing, and a method that
claimed gains here would be suspect.

## 3. The regime the method targets

Addition is the clean case of *unique but deep*: the answer is fully determined
by the operands, yet digit *i* needs the carry out of digit *i−1*.

**Exact match at ONE forward pass** (4 seeds, bootstrap 95% intervals):

| digits | MDLM baseline | full method | seeds >90% |
|---|---|---|---|
| 8 | 100.0 | 100.0 | 4/4 → 4/4 |
| **12** | **76.7** | **100.0** | 3/4 → **4/4** |
| **16** | **25.0** | *(running)* | 1/4 → — |

The gap opens exactly where the carry chain outgrows what one forward pass can
compute, and the method closes it — **sixteen inference passes collapse to one,
with accuracy going up rather than down.**

## 4. The method

```
L = L_MDLM(t ~ schedule matched to K)  +  λ · L_one-pass(t = 1)
```

**Schedule matching.** Standard training draws the masking ratio `t ~ U(0,1)`,
spending most capacity on nearly-complete states. A *K*-pass decode only visits
ratios `{1, (K−1)/K, …, 1/K}`. At high parallelism the model is evaluated in a
regime it barely trained on.

**One-pass supervision.** Supervise the prediction made from the *fully masked*
answer directly against ground truth, forcing the whole chain into one forward
pass rather than letting it lean on partially-decoded scratch space.

**Parallelism curriculum** (`--mode prog`) anneals K from 16 → 1 so the model
first solves the task with borrowed depth, then internalises the computation.

Cost: one extra forward pass per training step. It removes fifteen at inference.

## 5. Why it should work — and the falsifiable prediction

See [`THEORY.md`](THEORY.md). In brief: addition's carry is a **prefix scan**,
computable in **O(log n)** depth by carry-lookahead, and constant-depth
log-precision transformers are confined to uniform **TC⁰**, which contains
addition. So one-pass decoding is **expressible** at depth ~log n — the obstacle
is optimisation, not expressivity.

Sequential decoding is best understood as **borrowing depth from the decoding
loop**: a *K*-pass decode has effective depth ≈ L·K.

The account predicts minimum depth growing with **log n**, not n — cleanly
distinguishable from a ripple-carry account (linear) or an impossibility account
(no depth suffices).

## 6. A confound that nearly produced the wrong paper

Earlier runs showed the method "failing" at 10+ digits. It was a **learning-rate
artifact**:

| layers | lr | warmup | mean @1 pass | seeds >90% |
|---|---|---|---|---|
| 10 | 3e-4 | 500 | 47.3 | 1/4 |
| 10 | **1e-4** | 500 | **100.0** | **4/4** |
| 10 | 3e-4 | **4000** | **100.0** | **4/4** |

Deeper models need a lower learning rate and longer warmup. Without this check we
would have reported that the method does not generalise, and a depth sweep run at
the broken setting would have faked a clean "depth doesn't help" curve.

## 7. Layout

```
src/
  data.py         addition + dependency-controlled probes
  sigmorphon.py   CoNLL-SIGMORPHON 2017 inflection loader
  model.py        bidirectional transformer denoiser
  diffusion.py    objectives (MDLM, schedule-matched, one-pass, curriculum) + decoders
  train_add.py    addition trainer, accuracy vs passes
  train_sig.py    inflection trainer
  train.py        probe trainer (the unfixable-limit experiments)
  collect.py      tables and figures
slurm/            grids, plus unattended finalize and watchdog
```

```bash
python src/train_add.py --mode full --digits 12 --lr 1e-4 --warmup 4000 --out runs/demo
python src/train_sig.py --lang turkish --mode full --out runs/tur
```

## 8. Honest limitations

- Addition and inflection are probes, not applications; the claim is about the
  model class.
- Natural-language modelling (text8 bits-per-character) is not yet demonstrated.
- The depth-scaling prediction in §5 is stated but not yet measured — the sweep
  was cancelled on discovering it would have run at the broken learning rate.
- The non-unique limit is **not** solved here, and we argue it cannot be. That
  boundary is part of the contribution rather than a gap in it.
