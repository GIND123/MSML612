"""Does exact verifier feedback fix the absorbing-commit failure?

Diagnosis from the trained models: 94.9% of cells are correct but only 87.9% of
boards. Failed boards therefore carry roughly three wrong cells out of
fifty-seven - they are near misses, not collapses. Masked diffusion decoding is
absorbing, so those three cells can never be revised and each one destroys a
whole board.

Sudoku constraints are exactly checkable, so the repair signal is not a
heuristic: we know precisely which cells conflict. Remask exactly those (widened
to their row, column and box, since re-deciding one cell in an unchanged context
reproduces the same value), re-decode, repeat.

Every forward pass is counted, including the repair rounds, so the comparison
against fixed-K budgets is honest rather than flattering: a method that needs
sixty passes to repair is reported at sixty.

The control that matters is RANDOM remasking of the same number of cells. If
random repair helps as much as verifier-guided repair, then the gain is just
extra compute and the verifier is doing nothing.
"""
import argparse, glob, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Transformer
import diffusion as dfn
from sudoku import (CELLS, SudokuTokenizer, expand_to_units, violation_mask)

p = argparse.ArgumentParser()
p.add_argument("--hard", default="data/sudoku_hard")
p.add_argument("--runs", default="runs")
p.add_argument("--pattern", default="sudh-*")
p.add_argument("--eval_bs", type=int, default=250)
p.add_argument("--out", default="figures/sudoku_remask.json")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = SudokuTokenizer()
P = np.load(os.path.join(a.hard, "test_puz.npy"))
S = np.load(os.path.join(a.hard, "test_sol.npy"))
clues = (P > 0).sum(1)
print(f"benchmark: {len(P)} puzzles, clues {clues.min()}-{clues.max()} "
      f"(mean {clues.mean():.1f})", flush=True)

Pt, St = torch.from_numpy(P), torch.from_numpy(S - 1)
blank = Pt == 0


def load(d):
    cfg = json.load(open(os.path.join(d, "result.json")))["args"]
    m = Transformer(len(tok), cfg["d"], cfg["layers"], cfg["heads"], cfg["pe"],
                    causal=False, max_len=CELLS + 8).to(dev)
    m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev))
    m.eval()
    return m, cfg


def random_remask_like(bad_counts):
    """Control: remask the same NUMBER of cells, chosen at random."""
    def fn(x):
        B, L = x.shape
        out = torch.zeros_like(x, dtype=torch.bool)
        for b in range(B):
            k = int(bad_counts[b])
            if k:
                idx = torch.randperm(L, device=x.device)[:k]
                out[b, idx] = True
        return out
    return fn


@torch.no_grad()
def evaluate(m, fn):
    preds, nfes = [], []
    for i in range(0, len(Pt), a.eval_bs):
        gold = St[i:i + a.eval_bs].to(dev)
        bl = blank[i:i + a.eval_bs].to(dev)
        x = torch.where(bl, torch.full_like(gold, tok.mask), gold)
        y, nfe = fn(x, bl)
        preds.append(y.cpu().numpy() + 1)
        nfes.append(nfe)
    pred = np.concatenate(preds)
    board = float((pred == S).all(axis=1).mean())
    cell = float((pred[P == 0] == S[P == 0]).mean())
    return board, cell, float(np.mean(nfes)), pred


res = {}
dirs = sorted(glob.glob(os.path.join(a.runs, a.pattern)))
dirs = [d for d in dirs if os.path.exists(os.path.join(d, "model.pt"))]
print(f"evaluating {len(dirs)} models\n", flush=True)

for d in dirs:
    m, cfg = load(d)
    name = os.path.basename(d)
    print(f"--- {name}", flush=True)
    print(f"    {'decoder':<34}{'NFE':>8}{'BOARD':>10}{'cell':>9}", flush=True)
    row = {}

    for K in (16, 61):
        b, c, n, _ = evaluate(m, lambda x, bl, K=K: dfn.fixed_k_decode(
            m, x, tok, dev, K, fillable=bl))
        row[f"fixedK_{K}"] = {"board": b, "cell": c, "nfe": n}
        print(f"    {'fixed-K K=' + str(K):<34}{n:>8.1f}{b*100:>9.2f}%{c*100:>8.2f}%",
              flush=True)

    b, c, n, _ = evaluate(m, lambda x, bl: dfn.entropy_budget_decode(
        m, x, tok, dev, 0.01, max_steps=CELLS + 2, fillable=bl))
    row["entbudget"] = {"board": b, "cell": c, "nfe": n}
    print(f"    {'entropy-budget B=0.01':<34}{n:>8.1f}{b*100:>9.2f}%{c*100:>8.2f}%",
          flush=True)

    # Repair must start from the STRONGEST available decode, not a weak one.
    # An earlier version used base_steps=16, which decodes to ~83% before repair,
    # so the repaired result never beat plain 61-pass decoding at ~88% - the
    # method was handicapped by its starting point, not by the idea.
    for base in (32, 61):
        for rounds in (4, 8):
            b, c, n, _ = evaluate(
                m, lambda x, bl, r=rounds, bs=base: dfn.verifier_remask_decode(
                    m, x, tok, dev, violation_mask, base_steps=bs, rounds=r,
                    max_revisits=3, fillable=bl, expand_peers=expand_to_units))
            row[f"verifier_b{base}_r{rounds}"] = {"board": b, "cell": c, "nfe": n}
            print(f"    {'VERIFIER base=' + str(base) + ' rounds=' + str(rounds):<34}"
                  f"{n:>8.1f}{b*100:>9.2f}%{c*100:>8.2f}%", flush=True)

    # narrow repair: remask ONLY the conflicting cells, not their whole units
    b, c, n, _ = evaluate(m, lambda x, bl: dfn.verifier_remask_decode(
        m, x, tok, dev, violation_mask, base_steps=61, rounds=8,
        max_revisits=3, fillable=bl, expand_peers=None))
    row["verifier_narrow"] = {"board": b, "cell": c, "nfe": n}
    print(f"    {'VERIFIER base=61 narrow (no units)':<34}{n:>8.1f}{b*100:>9.2f}%"
          f"{c*100:>8.2f}%", flush=True)

    # control: same budget, but remask at random instead of where it is wrong
    b, c, n, _ = evaluate(m, lambda x, bl: dfn.verifier_remask_decode(
        m, x, tok, dev,
        lambda z: (torch.rand_like(z, dtype=torch.float) < 0.12),
        base_steps=61, rounds=8, max_revisits=3, fillable=bl,
        expand_peers=None))
    row["random_remask_control"] = {"board": b, "cell": c, "nfe": n}
    print(f"    {'CONTROL: random remask, 8 rounds':<34}{n:>8.1f}{b*100:>9.2f}%"
          f"{c*100:>8.2f}%", flush=True)

    res[name] = row
    del m
    torch.cuda.empty_cache()
    print(flush=True)

if res:
    print("=" * 76)
    print("SUMMARY (mean over models)")
    print("=" * 76)
    keys = list(next(iter(res.values())).keys())
    print(f"{'decoder':<34}{'NFE':>8}{'BOARD':>10}")
    base = None
    for k in keys:
        bs = [r[k]["board"] for r in res.values()]
        ns = [r[k]["nfe"] for r in res.values()]
        if k == "fixedK_61":
            base = float(np.mean(bs))
        print(f"{k:<34}{np.mean(ns):>8.1f}{np.mean(bs)*100:>9.2f}%")
    if base is not None:
        best_v = max((np.mean([r[k]["board"] for r in res.values()])
                      for k in keys if k.startswith("verifier_")), default=0)
        ctrl = np.mean([r["random_remask_control"]["board"] for r in res.values()])
        print(f"\n  best verifier-guided : {best_v*100:.2f}%")
        print(f"  fixed-K at 61 passes : {base*100:.2f}%   "
              f"(verifier gains {(best_v-base)*100:+.2f} pts)")
        print(f"  random-remask control: {ctrl*100:.2f}%   "
              f"({(ctrl-base)*100:+.2f} pts)")
        print("\n  If the control gains as much, the verifier is doing nothing and")
        print("  the improvement is only the extra compute.")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
