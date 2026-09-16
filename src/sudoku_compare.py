"""The full comparison: our diffusion model against classical solvers and the
published neural results, on a benchmark whose difficulty is stated rather than
assumed.

Published Sudoku numbers are reported on different puzzle distributions, and
quoting them side by side without saying so is exactly the error that cost this
project its SATNet claim. So:

  * classical baselines are MEASURED on our test set, not cited;
  * our model is reported STRATIFIED BY CLUE COUNT, so a reader can compare at
    whatever difficulty a published number used;
  * published numbers appear only in a clearly separated block, with their
    distribution named.

Three classical baselines, each measuring something different:

  propagation   naked + hidden singles to fixpoint, NO search. The "free" solver.
  backtracking  full search. Solves everything, so the informative quantity is
                its COST - how many search nodes it needs - which is the honest
                yardstick for "how hard is this puzzle really".
  greedy-MRV    one pass of most-constrained-variable assignment without
                backtracking: what a single confident sweep achieves.

For the diffusion model the quantity of interest is board accuracy at a small
fixed number of forward passes, which is the comparison that makes the
generative framing worth anything.
"""
import argparse, glob, json, os, sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RecurrentDenoiser, Transformer
import diffusion as dfn
from sudoku import CELLS, GRID, SudokuTokenizer
from gen_hard_sudoku import candidates, count_solutions, PEERS

# published results, with the distribution each was measured on
PUBLISHED = [
    ("SATNet (Wang et al. ICML 2019)", "SATNet easy split, ~36 clues", 0.983),
    ("SATNet, evaluated on RRN hard", "RRN hard, 17 clues", 0.061),
    ("RRN (Palm et al. NeurIPS 2018)", "RRN hard, 17 clues", 0.967),
    ("Recurrent Transformer (Yang et al. 2023)", "RRN hard, 17 clues", 0.995),
    ("Recurrent Transformer", "SATNet easy split", 1.000),
]
PARAMS = {"Recurrent Transformer (Yang et al. 2023)": "211k",
          "RRN (Palm et al. NeurIPS 2018)": "201k",
          "SATNet (Wang et al. ICML 2019)": "618k"}


def units():
    us = []
    for r in range(GRID):
        us.append([r * GRID + c for c in range(GRID)])
    for c in range(GRID):
        us.append([r * GRID + c for r in range(GRID)])
    for br in range(0, GRID, 3):
        for bc in range(0, GRID, 3):
            us.append([(br + dr) * GRID + bc + dc
                       for dr in range(3) for dc in range(3)])
    return us


UNITS = units()


def propagate(grid):
    g = [int(v) for v in grid]
    while True:
        cand = candidates(g)
        if cand is None:
            return None
        changed = False
        for i in range(CELLS):
            if g[i] == 0 and bin(cand[i]).count("1") == 1:
                g[i] = cand[i].bit_length()
                changed = True
        if not changed:
            for unit in UNITS:
                for d in range(9):
                    bit = 1 << d
                    spots = [i for i in unit if g[i] == 0 and cand[i] & bit]
                    if len(spots) == 1:
                        g[spots[0]] = d + 1
                        changed = True
        if not changed:
            return g


def greedy_mrv(grid):
    """Most-constrained-variable, one sweep, no backtracking."""
    g = [int(v) for v in grid]
    for _ in range(CELLS):
        cand = candidates(g)
        if cand is None:
            return None
        best, bi = 10, -1
        for i in range(CELLS):
            if g[i] == 0:
                n = bin(cand[i]).count("1")
                if n < best:
                    best, bi = n, i
        if bi < 0:
            return g
        g[bi] = cand[bi].bit_length() if best == 1 else \
            [d + 1 for d in range(9) if cand[bi] & (1 << d)][0]
    return g


def backtrack_cost(grid):
    """Solve by full search; return (solution, number of search nodes)."""
    g = [int(v) for v in grid]
    nodes = [0]

    def rec():
        nodes[0] += 1
        cand = candidates(g)
        if cand is None:
            return False
        best, bi = 10, -1
        for i in range(CELLS):
            if g[i] == 0:
                n = bin(cand[i]).count("1")
                if n < best:
                    best, bi = n, i
                    if n == 1:
                        break
        if bi < 0:
            return True
        for d in range(9):
            if cand[bi] & (1 << d):
                g[bi] = d + 1
                if rec():
                    return True
                g[bi] = 0
        return False

    ok = rec()
    return (np.array(g) if ok else None), nodes[0]


p = argparse.ArgumentParser()
p.add_argument("--hard", default="data/sudoku_hard")
p.add_argument("--runs", default="runs")
p.add_argument("--pattern", default="sudh-*")
p.add_argument("--eval_bs", type=int, default=200)
p.add_argument("--out", default="figures/sudoku_compare.json")
a = p.parse_args()

P = np.load(os.path.join(a.hard, "test_puz.npy"))
S = np.load(os.path.join(a.hard, "test_sol.npy"))
clues = (P > 0).sum(1)
print(f"benchmark: {len(P)} puzzles, clues {clues.min()}-{clues.max()} "
      f"(mean {clues.mean():.1f})\n", flush=True)

res = {"n": int(len(P)), "clues": {"min": int(clues.min()), "max": int(clues.max()),
                                   "mean": float(clues.mean())}}

print("=" * 88)
print("CLASSICAL BASELINES, measured on THIS test set")
print("=" * 88)
prop_ok = np.array([(lambda r: r is not None and all(v > 0 for v in r)
                     and np.array_equal(r, S[i]))(propagate(P[i]))
                    for i in range(len(P))])
mrv_ok = np.array([(lambda r: r is not None and all(v > 0 for v in r)
                    and np.array_equal(r, S[i]))(greedy_mrv(P[i]))
                   for i in range(len(P))])
bt = [backtrack_cost(P[i]) for i in range(len(P))]
bt_ok = np.array([g is not None and np.array_equal(g, S[i]) for i, (g, _) in enumerate(bt)])
nodes = np.array([n for _, n in bt])
res["classical"] = {"propagation": float(prop_ok.mean()),
                    "greedy_mrv": float(mrv_ok.mean()),
                    "backtracking": float(bt_ok.mean()),
                    "backtrack_nodes_mean": float(nodes.mean()),
                    "backtrack_nodes_median": float(np.median(nodes))}
print(f"  propagation only (no search) : {prop_ok.mean()*100:6.2f}%")
print(f"  greedy MRV, one sweep        : {mrv_ok.mean()*100:6.2f}%")
print(f"  full backtracking search     : {bt_ok.mean()*100:6.2f}%  "
      f"(mean {nodes.mean():.0f} search nodes, median {np.median(nodes):.0f})")
print(f"\n  >>> propagation leaves {(1-prop_ok.mean())*100:.1f}% unsolved, so this")
print(f"  >>> benchmark discriminates. Search solves all of it, at a cost.")

print()
print("=" * 88)
print("OUR FROM-SCRATCH MASKED DIFFUSION MODEL")
print("=" * 88)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = SudokuTokenizer()
runs = defaultdict(list)
for d in sorted(glob.glob(os.path.join(a.runs, a.pattern))):
    if os.path.exists(os.path.join(d, "result.json")):
        r = json.load(open(os.path.join(d, "result.json")))
        tag = ("recurrent-" if r["args"].get("recurrent") else "feedforward-") \
              + r["args"]["mode"]
        runs[tag].append((d, r))

if not runs:
    print("  (no trained models yet)")
else:
    Pt = torch.from_numpy(P)
    St = torch.from_numpy(S - 1)
    blank = Pt == 0

    def load(d):
        cfg = json.load(open(os.path.join(d, "result.json")))["args"]
        if cfg.get("recurrent"):
            m = RecurrentDenoiser(len(tok), cfg["d"], cfg["layers"], cfg["heads"],
                                  cfg["pe"], max_len=CELLS + 8,
                                  recurrences=cfg.get("R_infer") or cfg["R"]).to(dev)
        else:
            m = Transformer(len(tok), cfg["d"], cfg["layers"], cfg["heads"],
                            cfg["pe"], causal=False, max_len=CELLS + 8).to(dev)
        m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev))
        m.eval()
        return m

    @torch.no_grad()
    def run(m, steps):
        out = []
        for i in range(0, len(Pt), a.eval_bs):
            gold = St[i:i + a.eval_bs].to(dev)
            bl = blank[i:i + a.eval_bs].to(dev)
            x = torch.where(bl, torch.full_like(gold, tok.mask), gold)
            y, _ = dfn.fixed_k_decode(m, x, tok, dev, steps, fillable=bl)
            out.append(y.cpu().numpy() + 1)
        return np.concatenate(out)

    res["diffusion"] = {}
    for mode, lst in sorted(runs.items()):
        npar = json.load(open(os.path.join(lst[0][0], "result.json")))["args"]
        print(f"\n  {mode}  ({len(lst)} seeds, d={npar['d']} layers={npar['layers']}"
              + (f" R={npar.get('R')}→{npar.get('R_infer') or npar.get('R')}"
                 if npar.get('recurrent') else "") + ")")
        print(f"    {'passes':>8}{'board':>10}{'vs propagation':>17}")
        per_pass = {}
        for K in (1, 4, 8, 16, 32, 61):
            accs, preds = [], None
            for d, _ in lst:
                m = load(d)
                pr = run(m, K)
                accs.append(float((pr == S).all(axis=1).mean()))
                preds = pr if preds is None else preds
                del m
                torch.cuda.empty_cache()
            mean = float(np.mean(accs))
            per_pass[K] = {"board": mean, "per_seed": accs}
            if K == 61:
                per_pass[K]["by_clue"] = {
                    int(c): float((preds[clues == c] == S[clues == c]).all(axis=1).mean())
                    for c in sorted(set(clues.tolist()))
                    if (clues == c).sum() >= 20}
            print(f"    {K:>8}{mean*100:>9.2f}%{mean/max(prop_ok.mean(),1e-9):>15.1f}x")
        res["diffusion"][mode] = per_pass

    best_mode = max(res["diffusion"], key=lambda m: res["diffusion"][m][61]["board"])
    bc = res["diffusion"][best_mode][61].get("by_clue", {})
    if bc:
        print(f"\n  stratified by clue count (mode={best_mode}, 61 passes):")
        print(f"    {'clues':>7}{'n':>6}{'board':>10}")
        for c in sorted(bc):
            print(f"    {c:>7}{int((clues==c).sum()):>6}{bc[c]*100:>9.2f}%")

print()
print("=" * 88)
print("PUBLISHED NEURAL RESULTS — different puzzle distributions, NOT directly comparable")
print("=" * 88)
for name, dist, acc in PUBLISHED:
    pm = PARAMS.get(name, "")
    print(f"  {acc*100:6.1f}%   {name:<42} [{dist}]" + (f"  {pm} params" if pm else ""))
print(f"\n  Our benchmark is {clues.min()}-{clues.max()} clues (mean {clues.mean():.1f}).")
print("  The 17-clue RRN split is strictly harder than ours; the SATNet split,")
print("  at ~36 clues, is strictly easier and is fully solved by propagation alone.")
print("  The stratified table above is what permits a matched-difficulty comparison.")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
