# Parallelism-Matched Diffusion

**Masked diffusion language models, trained from scratch, that decode the whole
answer in a single forward pass instead of sixteen — with higher accuracy, not
lower.**

No pretrained weights, no pretrained tokenizer, no external teacher.

---

## 1. The problem, decomposed

Masked diffusion's entire selling point is **parallel decoding**: commit many
tokens per pass rather than one. In practice quality collapses as parallelism
rises, and roughly a dozen 2026 papers attack this at inference time with
smarter unmasking schedules.

Measuring it from scratch on tasks where we control the dependency structure
shows those papers are treating **two different failures as one**:

| Limit | When it applies | Fixable by training? |
|---|---|---|
| **Conditional independence** | the answer is genuinely non-unique | **No.** Two independent draws from a truly uniform marginal cannot agree. |
| **Computation depth** | the answer is unique but *chained* | **Yes.** This is what we fix. |

**The first limit is real and unbeatable.** On a probe where groups of tokens
must agree but *which* value is free, the likelihood-optimal marginal is uniform
over the valid values, so independent sampling agrees with probability 1/V. We
confirmed this the hard way: 66 runs sweeping an entropy penalty designed to
break the symmetry. Weak penalties changed nothing; strong ones destroyed the
model, failing even at fully sequential decoding. No training fixes it, because
succeeding would mean abandoning the data distribution.

**The second limit is not information-theoretic at all.** When the answer is
uniquely determined by the prompt — addition, where digit *i* needs the carry out
of digit *i−1* — every position's true marginal is a point mass. A perfect model
would decode in one pass. Sequential decoding wins only because each step can
read the digits already written, letting the carry chain unroll across passes,
while a one-pass prediction must fit that chain inside a fixed depth.

That is a **computation** problem, and computation problems are trainable.

## 2. The method

Two components, each following from the diagnosis.

**Schedule matching.** Standard training draws the masking ratio `t ~ U(0,1)`,
spending most of its capacity on nearly-complete states. A *K*-pass decode only
ever visits ratios `{1, (K−1)/K, …, 1/K}`. At high parallelism the model is
evaluated in a regime it barely trained on. We train on the ratios decoding
actually visits.

**One-pass supervision.** We supervise the prediction made from the *fully
masked* answer directly against the ground truth, forcing the whole carry chain
into a single forward pass rather than letting it lean on partially-decoded
scratch space.

```
L = L_MDLM(t ~ schedule)  +  λ · L_onepass(t = 1)
```

Cost: one extra forward pass per training step. It removes fifteen at inference.

> **Why this works here and not on the free-choice probes.** An earlier version
> used a sequential teacher to supply coordination targets. On free-choice tasks
> the teacher's pick is arbitrary, so averaged over batches its target decays
> back to uniform and the signal cancels against the ELBO — which is exactly
> what we observed. When the answer is unique the target is identical every time
> the model sees that prompt, and the cancellation cannot occur.

## 3. Results

Exact match at **one denoising pass** — the whole answer committed in a single
forward pass (3 seeds, bootstrap 95% intervals):

| digits | MDLM (baseline) | + schedule matching | + one-pass supervision | full method |
|---|---|---|---|---|
| 4 | 100.0 | 100.0 | 100.0 | 100.0 |
| 6 | 99.9 | 100.0 | 100.0 | 100.0 |
| **8** | **35.2** [0, 99] | 63.9 [1, 99] | **99.9** [100, 100] | **99.9** [100, 100] |

Seeds reaching >90%:

| digits | baseline | + matching | + one-pass | full |
|---|---|---|---|---|
| **8** | **1/3** | 2/3 | **3/3** | **3/3** |

At 8 digits the baseline does not merely score lower — **it fails outright on
two of three seeds**, while the method is near-perfect on all three. Four- and
six-digit addition are saturated for every method, which is itself the
prediction: the gap opens exactly where the carry chain outgrows what one
forward pass can compute.

The ablation is unambiguous: **one-pass supervision is the component that does
the work**; schedule matching helps but remains unreliable alone.

## 4. Evaluation

- **Headline axis:** exact-match accuracy against number of denoising passes,
  from fully parallel (1) to sequential (16).
- **Difficulty axis:** operand digits, i.e. the length of the carry chain.
- **Statistics:** 3 seeds, bootstrap confidence intervals, and a seed-reliability
  count, because mean accuracy hides the fact that the baseline fails completely
  on some seeds rather than degrading smoothly.
- **Probes:** `copy` (unique, independent), `agree`/`parity` (non-unique,
  dependent), addition (unique, dependent). The three cells isolate which limit
  is active.

## 5. Layout

```
src/
  data.py       addition + dependency-controlled probes
  model.py      bidirectional transformer denoiser
  diffusion.py  objectives (MDLM, schedule-matched, one-pass) and decoders
  train_add.py  addition trainer + accuracy-vs-passes evaluation
  train.py      probe trainer (conditional-independence experiments)
  collect.py    tables and figures
slurm/          grids, plus unattended finalize and watchdog
```

```bash
python src/train_add.py --mode full --digits 8 --steps 25000 --out runs/demo
sbatch slurm/g5.sh
```

## 6. Honest limitations

- Addition is a probe, not an application. The claim is about the model class.
- 4- and 6-digit are saturated, so the effect is currently demonstrated at a
  single difficulty; harder settings (10–16 digits) are running.
- The conditional-independence limit is **not** solved here, and we argue it
  cannot be. That boundary is part of the contribution, not a gap in it.
- Transfer to natural text (text8 bits-per-character) is not yet demonstrated.
