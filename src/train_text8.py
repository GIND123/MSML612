"""Masked diffusion on text8, and the experiment that tests the unified claim.

Addition isolates failure mode (i): every conditional is a point mass, so all of
the loss from parallel decoding comes from marginals that were never trained at
high mask ratios, and no inference-time rule recovers any of it. The free-choice
probes isolate failure mode (ii): the conditionals are maximum-entropy, nothing
is trainable, and committing less is the only remedy.

Natural text contains both in the same sequence. Finishing a word, a suffix or a
function word is nearly deterministic; choosing the next content word is not. So
text is where the two fixes have to compose, and this script measures whether
they do, as a 2x2:

                        fixed-K commit     entropy-budgeted commit
    uniform-t training      (a)                   (b)
    matched training        (c)                   (d)

The prediction is (d) < (b), (c) < (a) on generative perplexity at equal compute,
and that the two gains are largely independent - one fixes what the other cannot
touch.

Metrics, all standard for this benchmark:
  * validation bits-per-character (the MDLM bound) - likelihood, comparable to
    published text8 numbers, and the check that the schedule change does not
    buy sample quality by wrecking the model.
  * generative perplexity under a held-out autoregressive evaluator, against
    the number of function evaluations actually used.
  * entropy and distinct-n-gram rate of the samples, reported alongside, because
    generative perplexity alone is trivially gamed by degenerate repetition.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from model import Transformer
from text8 import load_text8, batches
import diffusion as dfn


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mdlm", "matched", "full"], default="mdlm")
    p.add_argument("--K", type=int, default=8, help="parallelism the schedule is matched to")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--root", default="data/text8")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_gen", type=int, default=256, help="samples per decoding setting")
    p.add_argument("--gen_bs", type=int, default=64)
    p.add_argument("--evallm", default="runs/evallm")
    p.add_argument("--out", default="runs/t8dev")
    return p.parse_args()


a = get_args()
torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)

tr, va, te, tok = load_text8(a.root, a.seq_len)
print(f"text8: train {tr.shape} valid {va.shape} test {te.shape} vocab {len(tok)}",
      flush=True)

model = Transformer(len(tok), a.d, a.layers, a.heads, "rope",
                    causal=False, max_len=a.seq_len + 8).to(dev)
print(f"denoiser params: {sum(q.numel() for q in model.parameters())/1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min((s + 1) / a.warmup, max(0.0, (a.steps - s) / max(1, a.steps - a.warmup))))

it = batches(tr, a.bs, seed=a.seed)
t0 = time.time()
for step in range(a.steps):
    x = torch.from_numpy(next(it)).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        loss = dfn.text_loss(model, x, tok, a.mode, a.K, a.lam)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 2000 == 0:
        print(f"step {step:6d}  loss {loss.item():.4f}  {time.time()-t0:.0f}s", flush=True)

model.eval()
res = {"args": vars(a)}

# ---- likelihood: the standard text8 number ---------------------------------
vb = [torch.from_numpy(va[i:i + a.bs]) for i in range(0, min(64 * a.bs, len(va)), a.bs)]
res["valid_bpc"] = dfn.nll_bpc(model, vb, tok, dev, mc=8)
print(f"\nvalidation BPC (MDLM bound): {res['valid_bpc']:.4f}", flush=True)

# ---- the evaluator ---------------------------------------------------------
ev = None
ev_path = os.path.join(a.evallm, "model.pt")
if os.path.exists(ev_path):
    ec = json.load(open(os.path.join(a.evallm, "result.json")))["args"]
    ev = Transformer(len(tok), ec["d"], ec["layers"], ec["heads"], "rope",
                     causal=True, max_len=ec["seq_len"] + 8).to(dev)
    ev.load_state_dict(torch.load(ev_path, map_location=dev))
    ev.eval()
    print(f"loaded evaluator from {a.evallm}", flush=True)
else:
    print(f"WARNING: no evaluator at {ev_path}; generative perplexity skipped", flush=True)


@torch.no_grad()
def gen_ppl(x):
    """Per-character perplexity the held-out autoregressive model assigns."""
    tot, cnt = 0.0, 0
    for i in range(0, len(x), a.gen_bs):
        b = x[i:i + a.gen_bs].to(dev)
        ce = F.cross_entropy(ev(b)[:, :-1].reshape(-1, len(tok)),
                             b[:, 1:].reshape(-1), reduction="sum")
        tot += float(ce); cnt += b[:, 1:].numel()
    return math.exp(tot / cnt)


def sample_stats(x):
    """Entropy and distinct-n-gram rate: the degeneracy guard.

    A model that emits the same low-perplexity string every time wins on
    generative perplexity while being useless, so the unigram entropy and the
    fraction of distinct 4-grams are reported next to it. Reference values from
    real text8 windows are computed below for comparison.
    """
    f = np.bincount(x.reshape(-1).numpy(), minlength=len(tok)).astype(float)
    f = f[f > 0] / f.sum()
    ent = float(-(f * np.log(f)).sum())
    grams = set()
    tot = 0
    for row in x.numpy():
        for i in range(len(row) - 3):
            grams.add(tuple(row[i:i + 4])); tot += 1
    return ent, len(grams) / max(tot, 1)


ref = torch.from_numpy(te[:a.n_gen])
res["reference"] = {"gen_ppl": gen_ppl(ref) if ev else None,
                    "entropy": sample_stats(ref)[0],
                    "distinct4": sample_stats(ref)[1]}
print(f"real text8 reference: ppl {res['reference']['gen_ppl']}, "
      f"entropy {res['reference']['entropy']:.3f}, "
      f"distinct-4 {res['reference']['distinct4']:.3f}", flush=True)


def blank():
    return torch.full((a.gen_bs, a.seq_len), tok.mask, dtype=torch.long)


def evaluate(tag, fn):
    """Run one decoding rule over n_gen samples and record quality vs compute."""
    outs, nfes = [], []
    for _ in range(max(1, a.n_gen // a.gen_bs)):
        x, nfe = fn(blank())
        outs.append(x.cpu()); nfes.append(nfe)
    x = torch.cat(outs)
    ent, d4 = sample_stats(x)
    row = {"nfe": float(np.mean(nfes)),
           "gen_ppl": gen_ppl(x) if ev else None,
           "entropy": ent, "distinct4": d4,
           "sample": tok.decode(x[0][:120].tolist())}
    print(f"  {tag:<26} nfe {row['nfe']:6.1f}  ppl {row['gen_ppl']:9.2f}  "
          f"H {ent:.3f}  d4 {d4:.3f}", flush=True)
    return row


print("\n=== decoding: quality against compute ===", flush=True)
res["decode"] = {}

print("fixed-K (non-adaptive reference):", flush=True)
for K in (1, 2, 4, 8, 16, 32, 64, 128, 256):
    res["decode"][f"fixedK_{K}"] = evaluate(
        f"fixed-K K={K}",
        lambda b, K=K: dfn.fixed_k_decode(model, b, tok, dev, K))

print("confidence-threshold (published inference-time method):", flush=True)
for tau in (0.5, 0.7, 0.9, 0.95, 0.99):
    res["decode"][f"conf_{tau}"] = evaluate(
        f"confidence tau={tau}",
        lambda b, tau=tau: dfn.confidence_threshold_decode(model, b, tok, dev, tau))

print("entropy-budget (ours, fix (ii)):", flush=True)
for B in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    res["decode"][f"entbudget_{B}"] = evaluate(
        f"entropy-budget B={B}",
        lambda b, B=B: dfn.entropy_budget_decode(model, b, tok, dev, B))

torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=2)
print(f"\nsaved to {a.out}", flush=True)
