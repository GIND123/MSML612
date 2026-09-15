"""The Sudoku evaluation that actually discriminates.

The SATNet split is saturated: plain constraint propagation, with no search at
all, solves 1000/1000 of its test puzzles. Any method that scores 100% there has
demonstrated nothing, ours included, and the published 98.3% is below what a
forty-line classical solver achieves.

Harder instances are built by stripping clues from the SAME solutions while
verifying after every removal that the solution stays unique, so accuracy remains
well defined. Constraint propagation alone solves 27.5% of the 30-clue set, 4.2%
of the 26-clue set and 3.3% of the 22-clue set - these discriminate.

Two things are being measured at once, and both are worth having:

  * CAPABILITY. Can the diffusion model solve puzzles a search-free classical
    solver cannot? That is a real question with a real answer.
  * GENERALISATION. The models were trained on 31-42 clue puzzles, so 30, 26 and
    22 clues are out of distribution. This asks whether they learned Sudoku or
    learned a clue density.

The decoding comparison is repeated here because this is where it should finally
matter: with fewer clues the model's marginals are no longer near-perfect, so
commit order has something to do.
"""
import argparse, glob, json, os, sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Transformer
import diffusion as dfn
from sudoku import CELLS, SudokuTokenizer, is_valid_solution

p = argparse.ArgumentParser()
p.add_argument("--runs", default="runs")
p.add_argument("--figs", default="figures")
p.add_argument("--eval_bs", type=int, default=120)
p.add_argument("--out", default="figures/sudoku_hard.json")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = SudokuTokenizer()

sets = {}
for t in (30, 26, 22):
    pf = os.path.join(a.figs, f"hard_{t}_puz.npy")
    sf = os.path.join(a.figs, f"hard_{t}_sol.npy")
    if os.path.exists(pf):
        sets[t] = (np.load(pf), np.load(sf))
if not sets:
    sys.exit("no hard puzzle sets found - run sudoku_validate.py first")
for t, (P, S) in sets.items():
    print(f"  {t}-clue set: {len(P)} puzzles, mean {(P>0).sum(1).mean():.1f} clues",
          flush=True)


def load(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "result.json")))["args"]
    m = Transformer(len(tok), cfg["d"], cfg["layers"], cfg["heads"], cfg["pe"],
                    causal=False, max_len=CELLS + 8).to(dev)
    m.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=dev))
    m.eval()
    return m, cfg


@torch.no_grad()
def solve(model, P, S, fn):
    """Board accuracy on a hard set: every one of the 81 cells correct."""
    ok = []
    for i in range(0, len(P), a.eval_bs):
        pz = torch.from_numpy(P[i:i + a.eval_bs]).to(dev)
        gold = torch.from_numpy(S[i:i + a.eval_bs] - 1).to(dev)
        blank = pz == 0
        x = torch.where(blank, torch.full_like(gold, tok.mask), gold)
        y, _ = fn(x, blank)
        pred = (y.cpu().numpy() + 1)
        ok.extend((pred == S[i:i + a.eval_bs]).all(axis=1).tolist())
    return float(np.mean(ok))


res = defaultdict(dict)
runs = sorted(glob.glob(os.path.join(a.runs, "sud1-*")))
print(f"\nevaluating {len(runs)} trained models on the hard sets", flush=True)

for d in runs:
    if not os.path.exists(os.path.join(d, "model.pt")):
        continue
    m, cfg = load(d)
    arm = f"{cfg['mode']}-{'aug' if cfg.get('augment') else 'plain'}"
    for t, (P, S) in sets.items():
        for tag, fn in [
                ("fixedK_1", lambda x, b: dfn.fixed_k_decode(m, x, tok, dev, 1, fillable=b)),
                ("fixedK_8", lambda x, b: dfn.fixed_k_decode(m, x, tok, dev, 8, fillable=b)),
                ("fixedK_45", lambda x, b: dfn.fixed_k_decode(m, x, tok, dev, 45, fillable=b)),
                ("entbudget_0.01", lambda x, b: dfn.entropy_budget_decode(
                    m, x, tok, dev, 0.01, max_steps=CELLS + 2, fillable=b)),
                ("entbudget_0.2", lambda x, b: dfn.entropy_budget_decode(
                    m, x, tok, dev, 0.2, max_steps=CELLS + 2, fillable=b))]:
            res[(arm, t)].setdefault(tag, []).append(solve(m, P, S, fn))
    del m
    torch.cuda.empty_cache()
    print(f"  done {os.path.basename(d)}", flush=True)

CP = {30: 0.275, 26: 0.042, 22: 0.033}   # measured in sudoku_validate.py

print()
print("=" * 86)
print("HARD SUDOKU — board accuracy, against a search-free classical solver")
print("=" * 86)
print("Models were trained on 31-42 clues, so every column here is out of distribution.\n")
arms = sorted({k[0] for k in res})
for t in sorted(sets, reverse=True):
    print(f"--- {t}-clue puzzles   (constraint propagation alone: {CP[t]*100:.1f}%)")
    print(f"    {'arm':<16}" + "".join(f"{g:>16}" for g in
                                       ("fixedK_1", "fixedK_8", "fixedK_45",
                                        "entbud_0.01", "entbud_0.2")))
    for arm in arms:
        r = res.get((arm, t))
        if not r:
            continue
        cells = []
        for g in ("fixedK_1", "fixedK_8", "fixedK_45", "entbudget_0.01", "entbudget_0.2"):
            v = r.get(g)
            cells.append(f"{np.mean(v)*100:>15.1f}%" if v else f"{'—':>16}")
        print(f"    {arm:<16}" + "".join(cells))
    print()

json.dump({f"{k[0]}|{k[1]}": {g: float(np.mean(v)) for g, v in d.items()}
           for k, d in res.items()},
          open(a.out, "w"), indent=2)
print(f"wrote {a.out}")
print("\nReading: a number above the classical rate is a capability the search-free")
print("solver does not have. A number below it means the model is worse than the")
print("first thing anyone would write, and should be reported that way.")
