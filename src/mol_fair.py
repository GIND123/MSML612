"""Is the QM9 result real, or an artifact of our own representation?

We encode a molecule as 9 fixed node slots plus 36 fixed edge slots. That forces
the model to learn slot-specific facts - slot 8 is never occupied, slot 7 is
occupied 80% of the time - which is precisely the position-dependence a relative
encoding cannot represent at the fully-masked state. Our RoPE baseline caps below
80% validity, and we attributed that to the encoding.

But DiGress and the graph-diffusion line do NOT generate this way. They sample
the number of atoms n from the training distribution FIRST, then generate an n-node
graph with a permutation-equivariant model. "Slot 8 is empty" never arises for
them. If that is the whole of our baseline's disadvantage, then our comparison is
against a strawman of our own construction and the QM9 claim does not stand.

This script removes exactly that disadvantage, without retraining. For each
sample it draws n from the empirical training distribution, fixes node slots
n..8 and every edge touching them to "none" as OBSERVED context, and decodes only
the remaining positions. That is the DiGress setup expressed in our
representation, and it is strictly generous to the baseline.

  RoPE recovers  -> the finding was representation-specific and the QM9 claim
                    must be withdrawn
  RoPE still fails -> the disadvantage is not the padding slots, and the finding
                    survives a fair baseline

Either way the answer goes in the paper.
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Transformer
import diffusion as dfn
from qm9 import EDGES, MAX_ATOMS, SEQ_LEN, MolTokenizer, load_qm9, validity

p = argparse.ArgumentParser()
p.add_argument("--runs", default="runs")
p.add_argument("--root", default="data/qm9")
p.add_argument("--n_gen", type=int, default=2048)
p.add_argument("--gen_bs", type=int, default=512)
p.add_argument("--out", default="figures/mol_fair.json")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
X, tok, train_smiles = load_qm9(a.root, 40000)
ALLOWED = torch.from_numpy(tok.allowed_mask()).to(dev)

# empirical distribution over heavy-atom counts, which is what DiGress samples
counts = np.array([(row[:MAX_ATOMS] != tok.none).sum() for row in X])
sizes, freqs = np.unique(counts, return_counts=True)
probs = freqs / freqs.sum()
print(f"atom-count distribution from training data: "
      f"{dict(zip(sizes.tolist(), np.round(probs, 3).tolist()))}", flush=True)


def scaffold(bs, rng):
    """A canvas with n sampled first: slots beyond n are OBSERVED as 'none'.

    This is the advantage DiGress has and our flat encoding withholds: the model
    never has to discover which slots exist, because the unused ones are given.
    """
    x = torch.full((bs, SEQ_LEN), tok.mask, dtype=torch.long)
    fill = torch.ones((bs, SEQ_LEN), dtype=torch.bool)
    ns = rng.choice(sizes, size=bs, p=probs)
    for b, n in enumerate(ns):
        for j in range(int(n), MAX_ATOMS):
            x[b, j] = tok.none
            fill[b, j] = False
        for k, (i, j) in enumerate(EDGES):
            if i >= n or j >= n:
                x[b, MAX_ATOMS + k] = tok.none
                fill[b, MAX_ATOMS + k] = False
    return x, fill


def load(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "result.json")))["args"]
    m = Transformer(len(tok), cfg["d"], cfg["layers"], cfg["heads"], cfg["pe"],
                    causal=False, max_len=SEQ_LEN + 8).to(dev)
    m.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=dev))
    m.eval()
    return m, cfg


@torch.no_grad()
def run(model, steps, seed):
    rng = np.random.default_rng(seed)
    outs = []
    for _ in range(max(1, a.n_gen // a.gen_bs)):
        x, fill = scaffold(a.gen_bs, rng)
        y, _ = dfn.fixed_k_decode(model, x.to(dev), tok, dev, steps,
                                  fillable=fill.to(dev), allowed=ALLOWED)
        outs.append(y.cpu())
    y = torch.cat(outs).numpy()
    v, smis = validity(y, tok)
    uniq = len(set(smis)) / max(1, len(smis))
    return v, uniq


res = {}
print(f"\n{'arm':<28}{'steps':>7}{'validity':>11}{'unique':>9}", flush=True)
for d in sorted(glob.glob(os.path.join(a.runs, "mol1-*"))):
    if not os.path.exists(os.path.join(d, "model.pt")):
        continue
    m, cfg = load(d)
    name = os.path.basename(d)
    key = f"{cfg['mode']}-{cfg['pe']}"
    for steps in (1, 4, 8, 16, 32, 45):
        v, u = run(m, steps, cfg["seed"])
        res.setdefault(key, {}).setdefault(str(steps), []).append(v)
        print(f"{name:<28}{steps:>7}{v*100:>10.2f}%{u*100:>8.1f}%", flush=True)
    del m
    torch.cuda.empty_cache()

print(f"\n{'='*70}")
print("FAIR BASELINE: atom count given, unused slots observed (the DiGress setup)")
print(f"{'='*70}")
print(f"{'arm':<22}" + "".join(f"{s:>9}" for s in (1, 4, 8, 16, 32, 45)))
for key in sorted(res):
    print(f"{key:<22}" + "".join(f"{np.mean(res[key][str(s)])*100:>8.1f}%"
                                 for s in (1, 4, 8, 16, 32, 45)))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
print("\nRead: if the relative-encoding arms now track the absolute ones, the QM9")
print("claim was an artifact of the flat fixed-slot encoding and must be withdrawn.")
