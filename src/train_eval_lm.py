"""Autoregressive evaluator for generative perplexity on text8.

Generative perplexity - the likelihood an independent language model assigns to
our samples - is the standard quality metric for diffusion language models,
because the diffusion model's own ELBO says nothing about the quality of what it
actually generates at a low number of function evaluations.

The evaluator is trained here from scratch on the same corpus rather than using
a pretrained GPT-2, for two reasons. The project's constraint is that everything
is trained from scratch, and a character-level evaluator trained on text8 itself
is the correct judge for text8 samples - a word-piece model trained on web text
would score character-level artifacts through a tokenizer that never sees them.

The evaluator is trained ONCE and shared by every diffusion run, so all systems
are judged by exactly the same yardstick.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from model import Transformer
from text8 import load_text8, batches

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/text8")
p.add_argument("--seq_len", type=int, default=256)
p.add_argument("--d", type=int, default=512)
p.add_argument("--layers", type=int, default=8)
p.add_argument("--heads", type=int, default=8)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--steps", type=int, default=60000)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--warmup", type=int, default=2000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="runs/evallm")
a = p.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)

tr, va, te, tok = load_text8(a.root, a.seq_len)
print(f"text8: train {tr.shape}, valid {va.shape}, test {te.shape}, vocab {len(tok)}",
      flush=True)

model = Transformer(len(tok), a.d, a.layers, a.heads, "rope",
                    causal=True, max_len=a.seq_len + 8).to(dev)
print(f"evaluator params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min((s + 1) / a.warmup, max(0.0, (a.steps - s) / max(1, a.steps - a.warmup))))

it = batches(tr, a.bs, seed=a.seed)
t0 = time.time()
for step in range(a.steps):
    x = torch.from_numpy(next(it)).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        logits = model(x)
        # next-character prediction: position i predicts position i+1
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, len(tok)),
                               x[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 2000 == 0:
        print(f"step {step:6d}  loss {loss.item():.4f}  "
              f"bpc {loss.item()/math.log(2):.3f}  {time.time()-t0:.0f}s", flush=True)


@torch.no_grad()
def test_bpc(arr, n=200):
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, min(n * a.bs, len(arr)), a.bs):
        x = torch.from_numpy(arr[i:i + a.bs]).to(dev)
        if len(x) < 2:
            continue
        logits = model(x)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, len(tok)),
                             x[:, 1:].reshape(-1), reduction="sum")
        tot += float(ce); cnt += x[:, 1:].numel()
    model.train()
    return tot / cnt / math.log(2)


bpc = test_bpc(te)
print(f"\nevaluator test BPC: {bpc:.4f}", flush=True)
torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
json.dump({"args": vars(a), "test_bpc": bpc, "vocab": len(tok)},
          open(os.path.join(a.out, "result.json"), "w"), indent=2)
print(f"saved to {a.out}", flush=True)
