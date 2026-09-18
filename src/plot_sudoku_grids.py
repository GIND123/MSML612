"""Sudoku figures that show the model actually solving, plus difficulty curves.

Everything is drawn from real runs and real test puzzles - the grids below are
genuine model output on held-out instances, not illustrations.
"""
import glob, json, os, sys
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RecurrentDenoiser, Transformer
import diffusion as dfn
from sudoku import CELLS, GRID, SudokuTokenizer

RUNS = os.environ.get("RUNS", "runs")
OUT = os.environ.get("OUT", "figures")
HARD = os.environ.get("HARD", "data/sudoku_hard")
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = SudokuTokenizer()

P = np.load(os.path.join(HARD, "test_puz.npy"))
S = np.load(os.path.join(HARD, "test_sol.npy"))
clues = (P > 0).sum(1)


def draw_grid(ax, grid, given=None, wrong=None, title="", fs=9):
    """Render one 9x9 board. Given clues bold, wrong cells shaded red."""
    g = np.asarray(grid).reshape(GRID, GRID)
    gv = np.asarray(given).reshape(GRID, GRID) if given is not None else None
    wr = np.asarray(wrong).reshape(GRID, GRID) if wrong is not None else None
    ax.set_xlim(0, 9); ax.set_ylim(0, 9); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for i in range(GRID):
        for j in range(GRID):
            if wr is not None and wr[i, j]:
                ax.add_patch(Rectangle((j, i), 1, 1, color="#f5b7b1", zorder=0))
            elif gv is not None and gv[i, j]:
                ax.add_patch(Rectangle((j, i), 1, 1, color="#eaeded", zorder=0))
            v = int(g[i, j])
            if v > 0:
                bold = gv is not None and gv[i, j]
                ax.text(j + .5, i + .55, str(v), ha="center", va="center",
                        fontsize=fs, fontweight="bold" if bold else "normal",
                        color="#000" if bold else ("#922b21" if (wr is not None and wr[i, j])
                                                   else "#1a5276"), zorder=2)
    for k in range(10):
        lw = 1.9 if k % 3 == 0 else 0.5
        ax.plot([k, k], [0, 9], color="k", lw=lw)
        ax.plot([0, 9], [k, k], color="k", lw=lw)
    ax.set_title(title, fontsize=8.5)


def load_best():
    """The strongest trained model available."""
    best, bacc = None, -1
    for d in sorted(glob.glob(f"{RUNS}/rec1-*")) + sorted(glob.glob(f"{RUNS}/inj-rec*")):
        rf, mf = os.path.join(d, "result.json"), os.path.join(d, "model.pt")
        if not (os.path.exists(rf) and os.path.exists(mf)):
            continue
        r = json.load(open(rf))
        acc = max(v["board"] for v in r["decode"].values())
        if acc > bacc:
            bacc, best = acc, (d, r["args"])
    if best is None:
        return None, None
    d, cfg = best
    m = RecurrentDenoiser(len(tok), cfg["d"], cfg["layers"], cfg["heads"], cfg["pe"],
                          max_len=CELLS + 8,
                          recurrences=cfg.get("R_infer") or cfg["R"]).to(dev)
    m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev))
    m.eval()
    return m, cfg


model, cfg = load_best()

# ---- Figure 5: the model solving, pass by pass ------------------------------
if model is not None:
    hardest = np.argsort(clues)[:3]                 # fewest clues = hardest
    budgets = [1, 4, 16, 61]
    fig, axes = plt.subplots(len(hardest), len(budgets) + 2,
                             figsize=(2.0 * (len(budgets) + 2), 2.05 * len(hardest)))
    for r, idx in enumerate(hardest):
        puz, sol = P[idx], S[idx]
        given = puz > 0
        draw_grid(axes[r][0], puz, given,
                  title=f"puzzle  ({given.sum()} clues)" if r == 0 else f"({given.sum()} clues)")
        for c, K in enumerate(budgets):
            with torch.no_grad():
                gold = torch.from_numpy(sol - 1)[None].to(dev)
                bl = torch.from_numpy(~given)[None].to(dev)
                x = torch.where(bl, torch.full_like(gold, tok.mask), gold)
                y, _ = dfn.fixed_k_decode(model, x, tok, dev, K, fillable=bl)
            pred = y[0].cpu().numpy() + 1
            wrong = (pred != sol) & (~given)
            ok = not wrong.any()
            draw_grid(axes[r][c + 1], pred, given, wrong,
                      title=(f"{K} pass{'es' if K > 1 else ''}" if r == 0 else "")
                            + ("  ✓" if ok else f"  {wrong.sum()} wrong"))
        draw_grid(axes[r][-1], sol, given, title="solution" if r == 0 else "")
    fig.suptitle("Recurrent masked diffusion solving held-out hard Sudoku "
                 "(grey = given clue, red = wrong cell)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/sud_fig5_grids.png", dpi=165)
    print("wrote sud_fig5_grids.png")

# ---- Figure 6: how the board fills across recurrences -----------------------
if model is not None:
    idx = int(np.argsort(clues)[0])
    puz, sol = P[idx], S[idx]
    given = puz > 0
    Rs = [1, 2, 4, 8, 16, 32, 64]
    fig, axes = plt.subplots(1, len(Rs), figsize=(2.0 * len(Rs), 2.35))
    for c, R in enumerate(Rs):
        model.recurrences = R
        with torch.no_grad():
            gold = torch.from_numpy(sol - 1)[None].to(dev)
            bl = torch.from_numpy(~given)[None].to(dev)
            x = torch.where(bl, torch.full_like(gold, tok.mask), gold)
            y, _ = dfn.fixed_k_decode(model, x, tok, dev, 16, fillable=bl)
        pred = y[0].cpu().numpy() + 1
        wrong = (pred != sol) & (~given)
        draw_grid(axes[c], pred, given, wrong,
                  title=f"R={R}   {(~wrong[~given]).mean()*100:.0f}% cells", fs=7.5)
    model.recurrences = cfg.get("R_infer") or cfg["R"]
    fig.suptitle("The same block applied more times: iterative refinement, "
                 "trained at R=32", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/sud_fig6_recurrence.png", dpi=165)
    print("wrote sud_fig6_recurrence.png")

# ---- Figure 7: accuracy against puzzle difficulty ---------------------------
if model is not None:
    model.recurrences = cfg.get("R_infer") or cfg["R"]
    preds = []
    with torch.no_grad():
        for i in range(0, len(P), 200):
            gold = torch.from_numpy(S[i:i + 200] - 1).to(dev)
            bl = torch.from_numpy(P[i:i + 200] == 0).to(dev)
            x = torch.where(bl, torch.full_like(gold, tok.mask), gold)
            y, _ = dfn.fixed_k_decode(model, x, tok, dev, 61, fillable=bl)
            preds.append(y.cpu().numpy() + 1)
    pred = np.concatenate(preds)
    solved = (pred == S).all(axis=1)
    ks = sorted({int(c) for c in clues if (clues == c).sum() >= 15})
    acc = [solved[clues == k].mean() * 100 for k in ks]
    nn = [(clues == k).sum() for k in ks]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(ks, acc, "o-", color="#1a5276", lw=2, ms=7, label="recurrent diffusion (ours)")
    ax.axhline(3.0, ls=":", color="#c0392b", lw=1.4)
    ax.text(ks[0], 5.5, "constraint propagation, whole set: 3.0%", fontsize=7.5,
            color="#c0392b")
    for k, a, n in zip(ks, acc, nn):
        ax.annotate(f"n={n}", (k, a), fontsize=6.5, xytext=(0, -13),
                    textcoords="offset points", ha="center", color="#555")
    ax.set_xlabel("clues given (fewer = harder)")
    ax.set_ylabel("board accuracy (%)")
    ax.set_title("Accuracy against puzzle difficulty, held-out test set")
    ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_ylim(-3, 105)
    ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(f"{OUT}/sud_fig7_difficulty.png", dpi=170)
    print("wrote sud_fig7_difficulty.png")
    json.dump({"by_clue": {int(k): float(a) for k, a in zip(ks, acc)},
               "n_by_clue": {int(k): int(n) for k, n in zip(ks, nn)}},
              open(os.path.join(OUT, "sud_by_clue.json"), "w"), indent=2)

# ---- Figure 8: benchmark difficulty, ours vs the saturated standard split ---
fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.bar(["SATNet split\n(~36 clues)", "ours\n(20–30 clues)"], [100.0, 3.0],
       color=["#aab7b8", "#c0392b"], width=.55)
ax.set_ylabel("solved by constraint propagation alone (%)")
ax.set_title("Why we generated a new benchmark:\nthe standard split is saturated")
for i, v in enumerate([100.0, 3.0]):
    ax.text(i, v + 2.5, f"{v:.1f}%", ha="center", fontweight="bold")
ax.set_ylim(0, 112); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/sud_fig8_saturation.png", dpi=170)
print("wrote sud_fig8_saturation.png")
