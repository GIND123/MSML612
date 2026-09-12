# Coordinated Masked Diffusion

**A training objective that makes parallel decoding work on jointly-dependent
tokens, in masked diffusion language models trained from scratch.**

No pretrained weights, no pretrained tokenizer, no external teacher.

---

## The problem

Masked diffusion's whole value proposition is **parallel decoding** — commit many
tokens per pass instead of one. But training teaches each masked position its
**marginal** `p(x_i | observed)`, while parallel decoding samples the **product of
marginals**. The truth is the joint, and the gap is the mutual information among
the tokens revealed together.

Measured directly, with dependency as a dial:

| Probe | Structure | 1 pass (max parallel) | 128 passes (sequential) |
|---|---|---|---|
| `copy` | answer positions independent given context | **100%** | 100% |
| `agree` | positions jointly dependent, value free | **0%** | **100%** |
| `parity` | jointly dependent | **0%** | **100%** |

Independent positions parallelise perfectly at any level. Dependent positions
work **only** one token at a time.

**Why no decoding schedule can fix this.** On `agree`, a group must agree but
*which* value is free, so each position's likelihood-optimal marginal is uniform
over the valid values. Sampling those independently is guaranteed inconsistent.
The marginals themselves carry no coordination, so inference-time methods can
only avoid co-committing dependent positions — that is, surrender the very
parallelism they were bought to provide.

## The method

```
Loss = ELBO  +  α · L_selfcond  +  β · L_coord
```

**`L_coord` (the core).** Take a random subset S of masked positions.

- **Teacher:** decode S *sequentially* with no gradient, each position
  conditioned on those already chosen — a coherent sample from the joint.
- **Student:** the marginals S receives from **one parallel forward pass**.
- **Loss:** cross-entropy of the teacher's joint sample under the student's
  parallel marginals.

One-shot parallel sampling is trained to reproduce what careful sequential
decoding would have produced. The model is its own teacher, so this needs no
external model and works in pretraining from scratch. It deliberately trades
likelihood for parallel-decodability — a trade the ELBO alone can never make.

**`L_selfcond`.** Commit a fraction of masked positions from the model's own
parallel samples and train on the rest, plus repair the commits. The corruption
rate anneals from zero: a random model samples noise, and corrupting from step 0
prevents learning entirely.

## Evaluation

The headline axis is **accuracy as a function of denoising passes**, from fully
parallel (1) to fully sequential (128).

- **Probes** — `copy`, `agree`, `parity`, with coupling as a dial, so the method
  is falsifiable: it must **not** help on `copy`, where positions are already
  independent.
- **text8** — bits-per-character, the standard from-scratch benchmark for
  discrete diffusion (D3PM, SEDD, MDLM all report it).
- **Decoding baselines** — confidence, entropy-gated, random order, all
  reimplemented at matched scale.

## Layout

```
src/
  data.py       text8 loader + dependency-controlled probes
  model.py      bidirectional transformer denoiser
  diffusion.py  MDLM / self-conditioning / coordination objectives + decoders
  train.py      trainer and the quality-vs-parallelism evaluation
slurm/
  grid2.sh      method vs ablations vs baseline
  finalize.sh   unattended collect, back up and publish
  watchdog.sh   self-rearming safety net
```

## Reproducing

```bash
python src/train.py --task agree --objective cmd --coupling 4 \
    --seq_len 128 --steps 20000 --out runs/demo
sbatch slurm/grid2.sh
```
