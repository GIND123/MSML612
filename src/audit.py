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

print("\n=== 7. decoders: adaptive commitment rules behave as specified ===")
# These three rules produce the headline text8 numbers, so their contract is
# checked directly rather than trusted: every masked position must end up
# filled, decoding must always make progress (no stall), and the entropy budget
# must actually behave monotonically in the budget - otherwise the
# quality/compute curve it traces would be meaningless.
try:
    from text8 import Text8Tokenizer
    t8 = Text8Tokenizer()
    torch.manual_seed(0)
    L, B = 32, 8
    dm = Transformer(len(t8), 64, 2, 2, "rope", causal=False, max_len=L + 8)
    dm.eval()
    blank = torch.full((B, L), t8.mask, dtype=torch.long)

    for name, fn in [
            ("fixed-K K=1", lambda b: dfn.fixed_k_decode(dm, b, t8, "cpu", 1)),
            ("fixed-K K=8", lambda b: dfn.fixed_k_decode(dm, b, t8, "cpu", 8)),
            ("confidence tau=0.99",
             lambda b: dfn.confidence_threshold_decode(dm, b, t8, "cpu", 0.99)),
            ("entropy-budget B=0.1",
             lambda b: dfn.entropy_budget_decode(dm, b, t8, "cpu", 0.1)),
            ("entropy-budget B=inf",
             lambda b: dfn.entropy_budget_decode(dm, b, t8, "cpu", 1e9))]:
        x, nfe = fn(blank)
        check(f"{name}: no [MASK] survives", not bool((x == t8.mask).any()),
              f"{int((x == t8.mask).sum())} left, nfe={nfe}")
        check(f"{name}: emits only real symbols", bool((x < len(t8) - 1).all()))

    _, n1 = dfn.fixed_k_decode(dm, blank, t8, "cpu", 1)
    check("fixed-K K=1 uses exactly one forward pass", n1 == 1, f"nfe={n1}")
    _, nb = dfn.entropy_budget_decode(dm, blank, t8, "cpu", 1e9)
    check("unbounded entropy budget commits in one pass", nb == 1, f"nfe={nb}")

    # a tighter budget must never be cheaper than a looser one
    nfes = [dfn.entropy_budget_decode(dm, blank, t8, "cpu", b)[1]
            for b in (0.01, 0.5, 2.0, 8.0, 1e9)]
    check("entropy budget is monotone: tighter budget costs >= passes",
          all(nfes[i] >= nfes[i + 1] for i in range(len(nfes) - 1)), str(nfes))

    # partial contexts must be left alone: only `fillable` positions may change
    part = blank.clone()
    part[:, :L // 2] = 3
    fill = torch.zeros_like(part, dtype=torch.bool)
    fill[:, L // 2:] = True
    y, _ = dfn.entropy_budget_decode(dm, part, t8, "cpu", 0.5, fillable=fill)
    check("decoding never overwrites the given context",
          bool((y[:, :L // 2] == 3).all()))
except Exception as e:
    check("decoder contract", False, f"{type(e).__name__}: {e}")

print("\n=== 8. text8 splits are disjoint and standard ===")
try:
    from text8 import load_text8, N_TRAIN, N_VALID, N_TEST
    root = os.environ.get("T8ROOT", "data/text8")
    tr8, va8, te8, tk8 = load_text8(root, 256)
    check("standard 90M/5M/5M split", (N_TRAIN, N_VALID, N_TEST) ==
          (90_000_000, 5_000_000, 5_000_000))
    check("vocabulary is 27 symbols + [MASK]", len(tk8) == 28, f"{len(tk8)}")
    check("no window straddles a split boundary",
          tr8.size + va8.size + te8.size <= 100_000_000,
          f"{tr8.size + va8.size + te8.size} chars covered")
    seen = {bytes(r) for r in tr8[::997]}
    dup = sum(bytes(r) in seen for r in te8[::997])
    check("no test window appears in the training sample", dup == 0, f"{dup} shared")
except FileNotFoundError:
    print("  [skip] text8 not downloaded here")
except Exception as e:
    check("text8 loader", False, f"{type(e).__name__}: {e}")

print("\n=== 9. graph families: enumeration matches known combinatorics ===")
# These families have closed-form counts, so the enumerator can be checked
# against mathematics rather than against itself. If enumeration is wrong then
# every marginal, every entropy and V* are wrong, and the central proposition
# would be "verified" against a fiction.
try:
    from graphs import enumerate_family, exact_stats, sample, is_valid, n_edges

    # Cayley: labelled spanning trees on n nodes = n^(n-2)
    for n in (5, 6, 7):
        got = len(enumerate_family("tree", n))
        check(f"spanning trees on {n} nodes = {n}^{{{n}-2}}", got == n ** (n - 2),
              f"{got} vs {n ** (n - 2)}")
    # perfect matchings on 2m nodes = (2m-1)!!
    for n, want in ((4, 3), (6, 15)):
        got = len(enumerate_family("matching", n))
        check(f"perfect matchings on {n} nodes = {want}", got == want, f"{got}")
    # 2-regular on 6 labelled vertices: one 6-cycle (5!/2=60) + two 3-cycles (10)
    got = len(enumerate_family("2regular", 6))
    check("2-regular graphs on 6 nodes = 70", got == 70, f"{got}")

    # the proposition itself, on every family
    bad = []
    for n in (6, 7):
        for fam in ("matching", "2regular", "tree", "trianglefree", "bipartite"):
            try:
                st = exact_stats(fam, n)
            except ValueError:
                continue
            V, tc = st["one_pass_ceiling"], st["total_correlation_bits"]
            if V < 2 ** (-tc) - 1e-9:
                bad.append((fam, n, V, 2 ** (-tc)))
            if tc < -1e-9:
                bad.append((fam, n, "negative TC", tc))
    check("V* >= 2^-TC on every family (the proposition)", not bad, str(bad[:2]))

    # sampled graphs must actually belong to the family they were drawn from
    for fam in ("matching", "tree", "bipartite"):
        X = sample(fam, 6, 200, seed=3)
        check(f"{fam}: sampled graphs are all in the family",
              bool(is_valid(X, fam, 6).all()))
        check(f"{fam}: sample width = n_edges", X.shape[1] == n_edges(6))
except Exception as e:
    check("graph families", False, f"{type(e).__name__}: {e}")

print("\n=== 10. molecules: the encoding is lossless ===")
# Storing aromatic bond types loses per-ATOM aromaticity, so real molecules
# failed to rebuild and generated-sample validity would have been scored against
# a target the representation could not reach. Round-trip on REAL data is the
# check that catches it.
try:
    from qm9 import MolTokenizer, SEQ_LEN, MAX_ATOMS, load_qm9, validity
    mt = MolTokenizer()
    am = mt.allowed_mask()
    check("node slots forbid bond symbols",
          not am[:MAX_ATOMS][:, mt.bond_ids].any())
    check("edge slots forbid atom symbols",
          not am[MAX_ATOMS:][:, mt.atom_ids].any())
    check("no position may emit [MASK]", not am[:, mt.mask].any())

    root = os.environ.get("QM9ROOT", "data/qm9")
    Xm, mtok, smis = load_qm9(root, limit=1500)
    v, _ = validity(Xm, mtok)
    check("round-trip validity on REAL molecules is ~100%", v > 0.999,
          f"{v*100:.2f}%")
    check("sequence width = 9 atoms + 36 bonds", Xm.shape[1] == SEQ_LEN,
          f"{Xm.shape[1]} vs {SEQ_LEN}")
except FileNotFoundError:
    print("  [skip] qm9.csv not downloaded here")
except Exception as e:
    check("molecule encoding", False, f"{type(e).__name__}: {e}")

print("\n=== 11. budget conditioning ===")
try:
    from data import build_addition as _ba
    _, _, tk = _ba(2, 4, 16, seed=0, exact=True)
    m0 = Transformer(len(tk), 32, 2, 2, "rope", causal=False, max_len=24)
    m1 = Transformer(len(tk), 32, 2, 2, "rope", causal=False, max_len=24,
                     budget_bins=8)
    check("budget_bins=0 adds no parameters",
          m0.n_params() == m1.n_params() - 8 * 32,
          f"{m0.n_params()} vs {m1.n_params()}")
    # a K=1 decode only ever sees the fully-masked state, so t must be exactly 1
    t1 = dfn._t_for_budget(torch.ones(256))
    check("K=1 always draws t=1", bool((t1 == 1.0).all()), f"max dev {float((t1-1).abs().max()):.2e}")
    t4 = dfn._t_for_budget(torch.full((4096,), 4.0))
    vals = sorted({round(float(v), 4) for v in t4})
    check("K=4 draws only {0.25,0.5,0.75,1.0}",
          vals == [0.25, 0.5, 0.75, 1.0], str(vals))
    check("t is never zero", bool((dfn._t_for_budget(torch.full((4096,), 64.0)) > 0).all()))
    b = Transformer.budget_bin(torch.tensor([1., 2., 4., 8., 1024.]), 8, "cpu")
    check("budget bins are log2 and clamped", b.tolist() == [0, 1, 2, 3, 7],
          str(b.tolist()))
except Exception as e:
    check("budget conditioning", False, f"{type(e).__name__}: {e}")

print(f"\n=== AUDIT: {len(FAIL)} failure(s) ===")
for f in FAIL:
    print("  !", f)
sys.exit(1 if FAIL else 0)
