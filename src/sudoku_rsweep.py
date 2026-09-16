"""Does deep supervision make the recurrence EXTRAPOLATE past its training depth?

The claim behind supervising every recurrence is that it forces each application
of the shared block to be a valid one-step refinement. If that holds, the map is
iterable beyond where it was fit, and running more recurrences at inference than
at training should keep helping - or at least not collapse.

This is free: no retraining, only re-evaluating trained checkpoints at different
recurrence counts. A model whose accuracy peaks exactly at its training depth and
degrades past it has learned one entangled R-step function rather than a
repeatable refinement step.
"""
import argparse, glob, json, os, sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RecurrentDenoiser
import diffusion as dfn
from sudoku import CELLS, SudokuTokenizer

p = argparse.ArgumentParser()
p.add_argument("--hard", default="data/sudoku_hard")
p.add_argument("--runs", default="runs")
p.add_argument("--pattern", default="rec1-*")
p.add_argument("--passes", type=int, default=16)
p.add_argument("--eval_bs", type=int, default=250)
p.add_argument("--out", default="figures/sudoku_rsweep.json")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = SudokuTokenizer()
P = np.load(os.path.join(a.hard, "test_puz.npy"))
S = np.load(os.path.join(a.hard, "test_sol.npy"))
Pt, St = torch.from_numpy(P), torch.from_numpy(S - 1)
blank = Pt == 0

res = defaultdict(dict)
for d in sorted(glob.glob(os.path.join(a.runs, a.pattern))):
    rf = os.path.join(d, "result.json")
    if not (os.path.exists(rf) and os.path.exists(os.path.join(d, "model.pt"))):
        continue
    cfg = json.load(open(rf))["args"]
    if not cfg.get("recurrent"):
        continue
    m = RecurrentDenoiser(len(tok), cfg["d"], cfg["layers"], cfg["heads"], cfg["pe"],
                          max_len=CELLS + 8, recurrences=cfg["R"]).to(dev)
    m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev))
    m.eval()
    name = os.path.basename(d)
    trained_R = cfg["R"]
    print(f"\n--- {name}  (trained with R={trained_R})", flush=True)
    print(f"    {'R at inference':>16}{'board':>10}", flush=True)
    for R in (1, 2, 4, 8, 16, 32, 64, 128):
        m.recurrences = R
        preds = []
        with torch.no_grad():
            for i in range(0, len(Pt), a.eval_bs):
                gold = St[i:i + a.eval_bs].to(dev)
                bl = blank[i:i + a.eval_bs].to(dev)
                x = torch.where(bl, torch.full_like(gold, tok.mask), gold)
                y, _ = dfn.fixed_k_decode(m, x, tok, dev, a.passes, fillable=bl)
                preds.append(y.cpu().numpy() + 1)
        acc = float((np.concatenate(preds) == S).all(axis=1).mean())
        res[name][R] = acc
        flag = "  <- trained here" if R == trained_R else ""
        print(f"    {R:>16}{acc*100:>9.2f}%{flag}", flush=True)
    del m
    torch.cuda.empty_cache()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
