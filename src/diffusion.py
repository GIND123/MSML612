"""Masked diffusion: training objectives and decoding strategies.

The central problem this file exists to study:

  MDLM training teaches each masked position its MARGINAL p(x_i | observed).
  Parallel decoding reveals k positions at once by sampling the PRODUCT of
  those marginals. The truth is the joint, and the gap is exactly the mutual
  information among the revealed positions. Quality therefore falls as the
  parallelism rises - which is the only reason to use diffusion in the first
  place.

Every published fix reorders or gates commits at INFERENCE. The model has still
never seen a partially-decoded state containing its own correlated mistakes.
`self_corrupted_loss` changes that at training time.
"""
import torch
import torch.nn.functional as F


# --------------------------------------------------------------- training ---
def mdlm_loss(model, x0, tok, t_min=None, fixed_t=None):
    """Standard MDLM objective: mask each token independently with prob t,
    predict the originals, weight by 1/t."""
    B, L = x0.shape
    dev = x0.device
    t_min = t_min if t_min is not None else 1.0 / L
    t = (torch.full((B,), fixed_t, device=dev) if fixed_t
         else t_min + (1.0 - t_min) * torch.rand(B, device=dev))
    m = torch.rand(B, L, device=dev) < t[:, None]
    m[torch.arange(B, device=dev), torch.randint(0, L, (B,), device=dev)] = True

    xt = torch.where(m, torch.full_like(x0, tok.mask), x0)
    logits = model(xt)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x0.reshape(-1),
                         reduction="none").view(B, L)
    return ((1.0 / t) * (ce * m).sum(1) / L).mean()


def self_corrupted_loss(model, x0, tok, k_frac=0.25, t_min=None,
                        correct_weight=1.0, ramp=1.0):
    """OUR METHOD. Train on states that parallel decoding actually produces.

    1. mask as usual and take a NO-GRAD pass to get the current marginals
    2. reveal a random subset of the masked positions *in parallel*, sampling
       each independently - i.e. commit exactly the kind of jointly-inconsistent
       tokens that parallel decoding commits
    3. train on the remaining masked positions given that self-generated
       context, so the model learns p(x_i | observed, its own parallel samples)
    4. additionally train it to REPAIR the committed tokens, which turns a
       wrong commit from an absorbing error into a recoverable one

    The model is therefore optimised for the distribution it meets at inference
    rather than for ground-truth context it will never see.

    `ramp` in [0,1] scales the corruption rate. It must start near zero: a
    randomly-initialised model samples noise, so corrupting from step 0 trains
    on pure garbage context and the model never learns the task at all (it
    scored 0% even on copy, which the baseline solves trivially). This is the
    same reason scheduled sampling anneals its mixing probability.
    """
    B, L = x0.shape
    dev = x0.device
    t_min = t_min if t_min is not None else 1.0 / L
    t = t_min + (1.0 - t_min) * torch.rand(B, device=dev)
    m = torch.rand(B, L, device=dev) < t[:, None]
    m[torch.arange(B, device=dev), torch.randint(0, L, (B,), device=dev)] = True
    xt = torch.where(m, torch.full_like(x0, tok.mask), x0)

    with torch.no_grad():
        probs = model(xt).softmax(-1)
        samp = torch.distributions.Categorical(probs=probs).sample()

    # commit a random subset of masked positions from the model's own samples
    commit = m & (torch.rand(B, L, device=dev) < k_frac * ramp)
    x_self = torch.where(commit, samp, xt)
    still = m & ~commit

    logits = model(x_self)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x0.reshape(-1),
                         reduction="none").view(B, L)

    # Normalise by L exactly as mdlm_loss does, so the two objectives are on the
    # same scale and any difference in results is the method, not the loss size.
    main = ((1.0 / t) * (ce * still).sum(1) / L).mean()
    rep = ((1.0 / t) * (ce * commit).sum(1) / L).mean()
    return main + correct_weight * rep


# --------------------------------------------------------------- decoding ---
@torch.no_grad()
def decode(model, shape, tok, steps, device, strategy="confidence",
           temperature=0.0, entropy_tau=None, x_init=None, keep_trace=False):
    """Reveal the canvas over `steps` passes. Fewer passes = more tokens
    committed per pass = more parallelism = more conditional-independence error.

    strategies
      confidence : reveal the most confident masked positions (the standard)
      random     : reveal a random subset (ablation: is confidence doing work?)
      entropy    : reveal only positions whose entropy is below `entropy_tau`,
                   an inference-time baseline for dependency-aware revealing
    """
    B, L = shape
    x = torch.full((B, L), tok.mask, device=device, dtype=torch.long)
    if x_init is not None:
        x = x_init.clone()
    trace = []

    for s in range(steps, 0, -1):
        masked = x == tok.mask
        if not masked.any():
            break
        logits = model(x)
        probs = logits.softmax(-1)
        if temperature > 0:
            pred = torch.distributions.Categorical(
                logits=logits / temperature).sample()
            conf = probs.gather(-1, pred[..., None]).squeeze(-1)
        else:
            conf, pred = probs.max(-1)

        if strategy == "random":
            score = torch.rand_like(conf)
        elif strategy == "entropy":
            ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1)
            score = -ent
        else:
            score = conf
        score = score.masked_fill(~masked, -1e9)

        target_left = int(L * (s - 1) / steps)
        for b in range(B):
            nm = int(masked[b].sum())
            k = max(0, nm - target_left)
            if strategy == "entropy" and entropy_tau is not None and k > 0:
                ent_b = -(probs[b] * probs[b].clamp_min(1e-9).log()).sum(-1)
                allowed = int(((ent_b < entropy_tau) & masked[b]).sum())
                k = max(1, min(k, allowed)) if allowed else 1
            if k:
                idx = score[b].topk(k).indices
                x[b, idx] = pred[b, idx]
        if keep_trace:
            trace.append(x.clone())
    still = x == tok.mask
    if still.any():
        x[still] = pred[still]
    return (x, trace) if keep_trace else x


@torch.no_grad()
def nll_bpc(model, batches, tok, device, mc=8):
    """Monte-Carlo estimate of the MDLM bound, reported as bits per character -
    the standard text8 metric."""
    tot, n = 0.0, 0
    for x0 in batches:
        x0 = x0.to(device)
        B, L = x0.shape
        for _ in range(mc):
            t = (1.0 / L) + (1 - 1.0 / L) * torch.rand(B, device=device)
            m = torch.rand(B, L, device=device) < t[:, None]
            xt = torch.where(m, torch.full_like(x0, tok.mask), x0)
            ce = F.cross_entropy(model(xt).reshape(-1, len(tok)),
                                 x0.reshape(-1), reduction="none").view(B, L)
            tot += float(((1.0 / t) * (ce * m).sum(1) / L).sum())
            n += B
    return tot / max(n, 1) / torch.log(torch.tensor(2.0)).item()


def coordination_loss(model, x0, tok, group=4, t_min=None):
    """CORE OF THE METHOD: make parallel sampling reproduce sequential sampling.

    The measured failure: on jointly-constrained-but-individually-free positions
    (our `agree` probe), each position's likelihood-optimal marginal is uniform
    over the valid values, so sampling them independently is guaranteed to be
    inconsistent - 0% accuracy at every parallelism, 100% only when decoded one
    at a time. No decoding schedule can fix that, because the marginals
    themselves carry no coordination.

    So we change the marginals. For a random subset S:

      teacher  : decode S SEQUENTIALLY with no gradient, each position
                 conditioned on the ones already chosen. Because each step sees
                 the previous choices, the result is a coherent joint sample.
      student  : the marginals S gets from ONE parallel forward pass.
      loss     : cross-entropy of the teacher's joint sample under the student's
                 parallel marginals.

    The model is its own teacher, so this needs no external model and works in
    pretraining from scratch. It deliberately trades likelihood (marginals move
    off the true conditional, toward a coordinated mode) for parallel
    decodability - a trade the standard ELBO can never make.

    COST: the teacher costs `group` sequential forward passes, so `group` is the
    knob that keeps training affordable. An early version coordinated over 32
    positions and made every step 32x more expensive, which simply never
    finished. Combined with applying the term every `coord_every` steps, the
    amortised overhead is roughly 1 + group/coord_every.
    """
    B, L = x0.shape
    dev = x0.device
    t_min = t_min if t_min is not None else 1.0 / L
    t = t_min + (1.0 - t_min) * torch.rand(B, device=dev)
    m = torch.rand(B, L, device=dev) < t[:, None]
    m[torch.arange(B, device=dev), torch.randint(0, L, (B,), device=dev)] = True
    xt = torch.where(m, torch.full_like(x0, tok.mask), x0)

    # choose, per example, a subset S of masked positions to coordinate over
    score = torch.rand(B, L, device=dev).masked_fill(~m, -1.0)
    k = min(group, L)
    S = score.topk(k, dim=1).indices                      # (B, k)

    # ---- teacher: reveal S one position at a time, conditioning as we go ----
    with torch.no_grad():
        x_seq = xt.clone()
        for j in range(k):
            idx = S[:, j]
            logits = model(x_seq)
            pick = logits[torch.arange(B, device=dev), idx].argmax(-1)
            x_seq[torch.arange(B, device=dev), idx] = pick
        target = x_seq.gather(1, S)                       # the joint sample

    # ---- student: one parallel pass over the same masked state --------------
    logits = model(xt)
    sel = logits.gather(1, S[..., None].expand(-1, -1, logits.size(-1)))
    return F.cross_entropy(sel.reshape(-1, sel.size(-1)), target.reshape(-1))


def cmd_loss(model, x0, tok, alpha=0.5, beta=1.0, k_frac=0.25, ramp=1.0,
             group=4, use_coord=True):
    """Coordinated Masked Diffusion: ELBO + self-conditioning + coordination.

    The ELBO term is always present. The two additions are auxiliaries, not
    replacements: an earlier version used self-conditioning INSTEAD of the ELBO
    and scored 35% where the plain baseline scored 100%.
    """
    total = mdlm_loss(model, x0, tok)
    if alpha > 0 and ramp > 0:
        total = total + alpha * self_corrupted_loss(
            model, x0, tok, k_frac=k_frac, correct_weight=1.0, ramp=ramp)
    if beta > 0 and ramp > 0 and use_coord:
        total = total + beta * ramp * coordination_loss(model, x0, tok, group=group)
    return total
