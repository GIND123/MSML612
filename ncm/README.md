# NCM baseline

Scores the published Next-Concept Modeling Sudoku checkpoints on our benchmark,
so the comparison against the state of the art is a measurement rather than a
citation. Everything here runs on CPU and needs no cluster time.

**These are somebody else's trained weights, evaluated as a baseline.** No
weight from here is used by, initialised from, or distilled into any model we
train. Our own models remain randomly initialised and trained only on our data.

## What NCM is

Kumbong, Khare et al., *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine
Text Generation*, 2026 (arXiv:2610.14162). A multiscale residual VQ-VAE encodes
a sequence into a hierarchy of discrete codes, and a transformer generates the
levels from coarse to fine, predicting every code within a level in parallel.
For Sudoku that is 12 levels — `[1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]` —
so a board is produced in 12 function evaluations.

Their reported Sudoku accuracy is 99.5 / 96.5 / 86.1 percent at 40 / 35 / 30
clues, with a 25M-parameter generator.

## Provenance of the weights

The ONNX exports come from `mrfakename/ncm-webgpu-weights` on Hugging Face, a
third-party conversion of the official `hazyresearch/ncm-checkpoints` release.
Because they are a conversion, `verify_export.py` checks that our encoding
rebuilds the shipped fixture prefixes token for token and that the graph
reproduces the fixture outputs exactly. Run it before quoting any number:

```bash
python ncm/verify_export.py sudoku-hard     # prints VERIFIED or MISMATCH
```

If it ever prints MISMATCH, fall back to the official PyTorch checkpoints and
the upstream `evaluate.py` instead of using these.

## The distribution-shift problem, and how we handle it

Their difficulty levels are fixed clue counts (40, 35, 30). Our benchmark is
made of *minimal* puzzles and sits at 21-28 clues, mean 24.3. Running their
30-clue checkpoint on our set is therefore out of distribution for it, and a
drop there says as much about the clue count as about the architecture.

`make_clue_sets.py` removes that confound by generating test sets at their clue
counts, using the same solution sampler and the same per-removal uniqueness
check as `src/gen_hard_sudoku.py`. Reporting both models across 40, 35, 30 and
our own 21-28 gives a difficulty curve instead of one contested number.

A smoke test on 24 puzzles per point already shows the shape of it: the
`sudoku-hard` checkpoint solved 24/24 at 30 clues and 9/24 at 24 clues. Those
samples are far too small to report, and are only an indication that the full
sweep is worth running.

## Usage

```bash
# 1. confirm the export reproduces its reference outputs
python ncm/verify_export.py sudoku-hard

# 2. score it on our benchmark (expects test_puz.npy / test_sol.npy)
python ncm/run_ncm.py sudoku-hard --data data/sudoku_hard --out ncm/results/ncm_hard_ours.json

# 3. build matched-difficulty sets and score those too
python ncm/make_clue_sets.py --clues 30 --n 1000 --out data/sudoku_c30 \
    --train_sol data/sudoku_hard/train_sol.npy
python ncm/run_ncm.py sudoku-hard --data data/sudoku_c30 --out ncm/results/ncm_hard_c30.json

# 4. tabulate
python ncm/compare.py ncm/results/*.json
```

Throughput is about 1.4 puzzles per second on CPU, so a 1,000-puzzle run takes
roughly twelve minutes per checkpoint. The graphs are deterministic
(`sample: False`), so a run is reproducible and there is no seed variance to
average over.

Requirements beyond the repository's own: `onnx`, `onnxruntime`,
`huggingface_hub`. The checkpoints are about 152 MB each and are cached in
`ncm/.models/`, which is ignored by git along with `ncm/results/`.

## Files

| File | Purpose |
|---|---|
| `ncm_encoding.py` | board to 91-token prefix, decoding, and the three scores we report |
| `verify_export.py` | fixture check against the published export; also opens the session |
| `run_ncm.py` | scores one checkpoint over one test set, writes JSON |
| `make_clue_sets.py` | test sets at a fixed clue count, uniqueness verified per removal |
| `compare.py` | tabulates finished runs |

## Reporting notes

- Quote parameters, NFE and wall-clock together. Their 12 NFE at 25M parameters
  is not comparable to our pass count at 212k without stating all three.
- Board accuracy, our metric, requires all 81 cells to match the unique
  solution. Valid Puzzle Rate, theirs, only requires a legal board. `run_ncm.py`
  reports both, plus whether the given clues survived.
- State the out-of-distribution caveat wherever the 21-28 clue number appears.
