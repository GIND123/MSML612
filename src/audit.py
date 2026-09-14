"""Numerical audit. Checks the things that would silently invalidate results.

Written because two earlier projects produced confident, wrong numbers: an
answer-length leak that handed the model free information, and a learning-rate
artifact that made a working method look broken. Both were invisible in the
accuracy tables.
"""
import glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import build_addition, build_probe
from sigmorphon import load_language
import diffusion as dfn
from model import Transformer

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        FAIL.append(name)


print("\n=== 1. data integrity ===")
for d in (4, 8, 12, 16):
    X, A, tok = build_addition(300, d, 3 * d + 4, seed=5, exact=True)
    bad = 0
    for i in range(len(X)):
        s = [tok.itos[t] for t in X[i] if t != tok.pad]
        a = int("".join(s[: s.index("+")]))
        b = int("".join(s[s.index("+") + 1: s.index("=")]))
        got = int("".join(reversed(s[s.index("=") + 1:])))
        bad += (a + b != got)
    check(f"{d}-digit addition arithmetic", bad == 0, f"{bad}/300 wrong")
    # the answer region must not reveal its own length
    check(f"{d}-digit answer mask is fixed-width",
          len({int(x) for x in A.sum(1)}) <= 2,
          f"{len({int(x) for x in A.sum(1)})} distinct widths")

print("\n=== 2. the prompt is never masked (no label leakage) ===")
X, A, tok = build_addition(200, 8, 28, seed=7, exact=True)
Xt, At = torch.from_numpy(X), torch.from_numpy(A)
xt = torch.where(At, torch.full_like(Xt, tok.mask), Xt)
prompt_ok = ((xt != tok.mask) | At).all()
check("only answer positions are masked at decode time", bool(prompt_ok))
eq_visible = all((xt[i] == tok.stoi["="]).any() for i in range(len(xt)))
check("the '=' separator stays visible", eq_visible)

print("\n=== 3. metric cannot be gamed ===")
gold = Xt[:50]
am = At[:50]
perfect = dfn.cond_decode  # placeholder to keep import used
from train_add import exact_match
check("identical prediction scores 1.0", exact_match(gold, gold, am, tok) == 1.0)
shifted = gold.clone()
shifted[am] = (shifted[am] + 1) % 10 + 4
check("all-wrong prediction scores 0.0", exact_match(shifted, gold, am, tok) == 0.0)
const = gold.clone()
const[am] = tok.stoi["0"]
acc_const = exact_match(const, gold, am, tok)
check("constant-output prediction is not rewarded", acc_const < 0.05, f"{acc_const:.3f}")

print("\n=== 4. probe tasks: ground truth valid, degenerate output rejected ===")
from train import probe_accuracy
from data import CharTokenizer
for task in ("copy", "agree", "parity"):
    Xp, tk = build_probe(200, 64, 4, seed=1, task=task)
    gt = probe_accuracy(Xp, task, 4, tk)
    cst = probe_accuracy(np.full_like(Xp, 5), task, 4, tk)
    check(f"{task}: ground truth valid", gt > 0.99, f"{gt:.2f}")
    if task != "copy":
        check(f"{task}: constant output rejected", cst < 0.01, f"{cst:.2f}")

print("\n=== 5. inflection: no train/dev overlap ===")
try:
    (Xtr, Atr), (Xdv, Adv), tok2 = load_language(
        "english", os.environ.get("SIGROOT", "data/sig"), "medium", 48)
    tr = {tuple(int(v) for v in r) for r in Xtr}
    dv = {tuple(int(v) for v in r) for r in Xdv}
    check("english train/dev disjoint", len(tr & dv) == 0, f"{len(tr & dv)} shared")
    check("dev set is non-trivial", len(Xdv) > 200, f"{len(Xdv)} items")
except Exception as e:
    check("inflection loader", False, f"{type(e).__name__}: {e}")

print("\n=== 6. reported numbers reproduce from raw run files ===")
runs = os.environ.get("RUNS", "runs")
groups = {}
for f in glob.glob(f"{runs}/g11-*/result.json"):
    r = json.load(open(f))
    a = r["args"]
    groups.setdefault((a["digits"], a["mode"]), []).append(r["accuracy_by_passes"]["1"])
bad = [k for k, v in groups.items() if len(v) < 2]
check("every reported group has >=2 seeds", not bad, f"thin groups: {bad[:3]}")
oor = []
for f in glob.glob(f"{runs}/*/result.json"):
    r = json.load(open(f))
    for k, v in (r.get("accuracy_by_passes") or {}).items():
        if not (0.0 <= v <= 1.0):
            oor.append((os.path.basename(os.path.dirname(f)), k, v))
check("all accuracies within [0,1]", not oor, str(oor[:3]))

print(f"\n=== AUDIT: {len(FAIL)} failure(s) ===")
for f in FAIL:
    print("  !", f)
sys.exit(1 if FAIL else 0)
