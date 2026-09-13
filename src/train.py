"""Train a masked diffusion LM from scratch and measure quality vs parallelism.

The headline axis is accuracy (or bits-per-character) as a function of the
number of denoising passes. Fewer passes means more tokens committed per pass,
which is the regime where the conditional-independence assumption bites and
where every acceleration method must eventually be judged.
"""
import argparse, json, math, os, time
import numpy as np
import torch

from data import load_text8, build_probe, CharTokenizer
from model import Transformer
import diffusion as dfn


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", choices=["mdlm", "selfcorrupt", "coord", "cmd"],
                   default="mdlm",
                   help="mdlm=ELBO baseline; selfcorrupt=+self-conditioning; "
                        "coord=+coordination; cmd=full method (all three)")
    p.add_argument("--alpha", type=float, default=0.5, help="self-conditioning weight")
    p.add_argument("--beta", type=float, default=1.0, help="coordination weight")
    p.add_argument("--gamma", type=float, default=0.0,
                   help="marginal-entropy penalty: the explicit price paid in "
                        "likelihood to buy parallel decodability")
    p.add_argument("--dep_weighted", type=int, default=1,
                   help="1 = sharpen only where positions are measurably dependent; "
                        "0 = sharpen everywhere (ablation)")
    p.add_argument("--elbo_w", type=float, default=1.0,
                   help="down-weight the ELBO so coordination is not cancelled by it")
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--coord_every", type=int, default=4,
                   help="apply the coordination term every N steps; the teacher\n                         costs `group` sequential passes, so this amortises it")
    p.add_argument("--k_frac", type=float, default=0.25,
                   help="fraction of masked positions committed from the model's "
                        "own parallel samples during self-corrupted training")
    p.add_argument("--correct_weight", type=float, default=1.0)
    p.add_argument("--ramp_frac", type=float, default=0.3,
                   help="fraction of training over which self-corruption ramps "
                        "from 0 to --k_frac; corrupting from step 0 trains on a "
                        "random model's noise and prevents learning entirely")
    p.add_argument("--task", choices=["text8", "copy", "agree", "parity"], default="agree")
    p.add_argument("--coupling", type=int, default=4,
                   help="probe tasks: how many positions are bound together")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_train", type=int, default=200000)
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--pe", default="rope")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_root", default="data")
    p.add_argument("--eval_bs", type=int, default=100)
    p.add_argument("--out", default="runs/dev")
    return p.parse_args()


# The whole point: sweep the number of passes from fully sequential (L) down to
# maximally parallel (1) and watch where each method falls over.
PASS_GRID = [1, 2, 4, 8, 16, 32, 64, 128]


def probe_accuracy(x, task, coupling, tok):
    """Fraction of sequences satisfying the task constraint exactly."""
    ok = 0
    for row in x.tolist():
        if task == "copy":
            h = len(row) // 2
            ok += row[:h] == row[h:h * 2]
        elif task == "agree":
            groups = [row[s:s + coupling] for s in range(0, len(row), coupling)
                      if len(row[s:s + coupling]) == coupling]
            good = all(len(set(g)) == 1 for g in groups) and \
                   all(groups[j][0] != groups[j + 1][0] for j in range(len(groups) - 1))
            ok += good
        elif task == "parity":
            groups = [row[s:s + coupling] for s in range(0, len(row), coupling)
                      if len(row[s:s + coupling]) >= 2]
            good = True
            for g in groups:
                bits = [v - 2 for v in g]
                if any(b not in (0, 1) for b in bits) or sum(bits[:-1]) % 2 != bits[-1]:
                    good = False
                    break
            if good:
                good = all(groups[j] != groups[j + 1] for j in range(len(groups) - 1))
            ok += good
    return ok / len(x)


@torch.no_grad()
def evaluate(model, args, tok, device, eval_data):
    model.eval()
    out = {}
    for steps in PASS_GRID:
        if steps > args.seq_len:
            continue
        accs = []
        for i in range(0, args.n_eval, args.eval_bs):
            b = min(args.eval_bs, args.n_eval - i)
            x_init = None
            if args.task == "copy":
                # Conditional: the first half is GIVEN, so there is one correct
                # completion. Unconditionally a constant sequence copies itself,
                # which any degenerate model satisfies.
                from data import build_probe as _bp
                src, _ = _bp(b, args.seq_len, args.coupling, seed=9000 + i, task="copy")
                x_init = torch.full((b, args.seq_len), tok.mask, device=device, dtype=torch.long)
                h = args.seq_len // 2
                x_init[:, :h] = torch.from_numpy(src[:, :h]).to(device)
            x = dfn.decode(model, (b, args.seq_len), tok, steps, device, x_init=x_init)
            if args.task == "text8":
                accs.append(0.0)      # quality for text8 is reported via bpc
            else:
                accs.append(probe_accuracy(x.cpu(), args.task, args.coupling, tok))
        out[str(steps)] = float(np.mean(accs))
    res = {"accuracy_by_passes": out}
    if args.task == "text8" and eval_data is not None:
        res["bpc"] = dfn.nll_bpc(model, eval_data[:8], tok, device)
    model.train()
    return res


def main():
    a = get_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    if a.task == "text8":
        tr, va, te, tok = load_text8(a.data_root, a.seq_len)
        train = torch.from_numpy(tr)
        eval_data = [torch.from_numpy(va[i:i + a.eval_bs]) for i in range(0, 800, a.eval_bs)]
    else:
        X, tok = build_probe(a.n_train, a.seq_len, a.coupling, seed=a.seed, task=a.task)
        train = torch.from_numpy(X)
        eval_data = None

    model = Transformer(len(tok), a.d, a.layers, a.heads, a.pe,
                        causal=False, max_len=a.seq_len + 8).to(dev)
    print(f"[{a.objective}/{a.task}] params={model.n_params()/1e6:.2f}M "
          f"vocab={len(tok)} seq={a.seq_len} device={dev}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                       0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" \
        else dict(device_type="cpu", enabled=False)

    t0 = time.time()
    for step in range(a.steps):
        idx = torch.randint(0, len(train), (a.bs,))
        x0 = train[idx].to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(**amp):
            ramp = min(1.0, step / max(1, a.ramp_frac * a.steps))
            uc = (step % a.coord_every == 0)
            if a.objective == "cmd":
                loss = dfn.cmd_loss(model, x0, tok, a.alpha, a.beta, a.k_frac,
                                    ramp, a.group, use_coord=uc,
                                    gamma=a.gamma, elbo_w=a.elbo_w,
                                    dep_weighted=bool(a.dep_weighted))
            elif a.objective == "coord":
                loss = dfn.cmd_loss(model, x0, tok, 0.0, a.beta, a.k_frac,
                                    ramp, a.group, use_coord=uc,
                                    gamma=a.gamma, elbo_w=a.elbo_w,
                                    dep_weighted=bool(a.dep_weighted))
            elif a.objective == "selfcorrupt":
                loss = dfn.cmd_loss(model, x0, tok, a.alpha, 0.0, a.k_frac, ramp)
            else:
                loss = dfn.mdlm_loss(model, x0, tok)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0:
            print(f"step {step:6d} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    res = evaluate(model, a, tok, dev, eval_data)
    n_par = model.n_params()
    res.update({"args": vars(a), "minutes": (time.time() - t0) / 60,
                "compute": {"params": n_par,
                            "train_tokens": a.steps * a.bs * a.seq_len,
                            "train_flops_approx": 6 * n_par * a.steps * a.bs * a.seq_len,
                            # self-corruption costs one extra no-grad pass
                            "fwd_per_step": 2 if a.objective == "selfcorrupt" else 1}})
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=2)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    print("FINAL", json.dumps(res["accuracy_by_passes"]), flush=True)
    if "bpc" in res:
        print(f"BPC {res['bpc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
