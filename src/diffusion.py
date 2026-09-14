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
import math
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


def entropy_penalty(model, x0, tok, t_min=None):
    """Break the symmetry that makes free groups undecodable in parallel.

    Where a group must agree but the value is free, the likelihood-optimal
    marginal is uniform over the valid values, and independent sampling from
    uniform marginals can never coordinate. Penalising marginal entropy pushes
    each position toward a single mode; because the model is shared across the
    group, the same mode is chosen at each member and parallel sampling becomes
    self-consistent. This is the term that is *supposed* to cost likelihood -
    it is the explicit price of parallel decodability.
    """
    B, L = x0.shape
    dev = x0.device
    t_min = t_min if t_min is not None else 1.0 / L
    t = t_min + (1.0 - t_min) * torch.rand(B, device=dev)
    m = torch.rand(B, L, device=dev) < t[:, None]
    xt = torch.where(m, torch.full_like(x0, tok.mask), x0)
    p = model(xt).softmax(-1)
    ent = -(p * p.clamp_min(1e-9).log()).sum(-1)
    return (ent * m).sum() / m.sum().clamp_min(1)


def cmd_loss(model, x0, tok, alpha=0.5, beta=1.0, k_frac=0.25, ramp=1.0,
             group=4, use_coord=True, gamma=0.0, elbo_w=1.0,
             dep_weighted=True):
    """Coordinated Masked Diffusion: ELBO + self-conditioning + coordination.

    The ELBO term is always present. The two additions are auxiliaries, not
    replacements: an earlier version used self-conditioning INSTEAD of the ELBO
    and scored 35% where the plain baseline scored 100%.
    """
    total = elbo_w * mdlm_loss(model, x0, tok)
    if gamma > 0 and ramp > 0:
        total = total + gamma * ramp * (
            dependence_weighted_entropy(model, x0, tok) if dep_weighted
            else entropy_penalty(model, x0, tok))
    if alpha > 0 and ramp > 0:
        total = total + alpha * self_corrupted_loss(
            model, x0, tok, k_frac=k_frac, correct_weight=1.0, ramp=ramp)
    if beta > 0 and ramp > 0 and use_coord:
        total = total + beta * ramp * coordination_loss(model, x0, tok, group=group)
    return total


def dependence_weighted_entropy(model, x0, tok, t_min=None):
    """THE METHOD. Sharpen the marginals exactly where positions are dependent.

    Why entropy is the right quantity. Two independent draws from a marginal p
    agree with probability sum_v p(v)^2 = exp(-H2(p)), the Renyi-2 entropy. So
    minimising H2 directly maximises the probability that positions sampled
    independently - which is precisely what parallel decoding does - land on the
    same choice. For a group that must agree but whose value is free, this is
    the only way to make parallel sampling self-consistent: the likelihood
    optimum (uniform over valid values) is guaranteed to disagree.

    Why it must be WEIGHTED. Sharpening everywhere would destroy likelihood
    wherever high entropy is correct, which is most of natural text, and would
    also break `copy`, where positions are already independent and the baseline
    is perfect. So the penalty is scaled by a measured dependence:

        reveal one masked position, and see how far every other position's
        belief moves. A position whose distribution shifts a lot depends on the
        revealed one; a position that does not move is conditionally
        independent and is left alone.

    d_i = KL(p_i-after-reveal || p_i-before), normalised per example. The
    penalty is d_i * H2(p_i), so the likelihood cost is paid only where it buys
    coordination.
    """
    B, L = x0.shape
    dev = x0.device
    t_min = t_min if t_min is not None else 1.0 / L
    t = t_min + (1.0 - t_min) * torch.rand(B, device=dev)
    m = torch.rand(B, L, device=dev) < t[:, None]
    m[torch.arange(B, device=dev), torch.randint(0, L, (B,), device=dev)] = True
    xt = torch.where(m, torch.full_like(x0, tok.mask), x0)

    with torch.no_grad():
        p0 = model(xt).softmax(-1)
        samp = torch.distributions.Categorical(probs=p0).sample()
        # reveal one masked position per example and measure the ripple
        pick = torch.rand(B, L, device=dev).masked_fill(~m, -1.0).argmax(1)
        x1 = xt.clone()
        ar = torch.arange(B, device=dev)
        x1[ar, pick] = samp[ar, pick]
        p1 = model(x1).softmax(-1)
        d = (p1 * (p1.clamp_min(1e-9).log() - p0.clamp_min(1e-9).log())).sum(-1)
        d = (d * m).clamp_min(0)
        d = d / d.amax(1, keepdim=True).clamp_min(1e-6)      # per-example scale

    p = model(xt).softmax(-1)
    h2 = -(p.pow(2).sum(-1).clamp_min(1e-9)).log()           # Renyi-2 entropy
    return (d * h2 * m).sum() / m.sum().clamp_min(1)


# ===========================================================================
# Conditional objectives for UNIQUE-answer tasks (prompt visible, answer
# masked). Here the target is deterministic, so unlike the free-choice probes
# there is a correct answer to aim at and coordination is learnable.
# ===========================================================================

def _cond_mask(x0, amask, tok, t):
    """Mask a fraction t of the ANSWER positions only; the prompt stays visible."""
    dev = x0.device
    r = torch.rand_like(amask, dtype=torch.float)
    m = (r < t[:, None]) & amask
    # never leave an example with nothing to predict
    empty = ~m.any(1)
    if empty.any():
        first = amask.float().argmax(1)
        m[empty, first[empty]] = True
    return m


def cond_mdlm_loss(model, x0, amask, tok, t_dist="uniform", K=None):
    """Baseline, plus the schedule-matched variant.

    t_dist="uniform" is standard MDLM: every masking ratio is trained equally.
    t_dist="matched" instead draws t from the ratios a K-pass decode actually
    visits, namely {1, (K-1)/K, ..., 1/K}. Standard training spends most of its
    capacity on nearly-complete states that high-parallelism decoding never
    sees, which is a plain train/inference mismatch.
    """
    B, L = x0.shape
    dev = x0.device
    if t_dist == "matched" and K:
        idx = torch.randint(1, K + 1, (B,), device=dev).float()
        t = idx / K
    else:
        t = torch.rand(B, device=dev).clamp_min(1.0 / L)

    m = _cond_mask(x0, amask, tok, t)
    xt = torch.where(m, torch.full_like(x0, tok.mask), x0)
    logits = model(xt)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x0.reshape(-1),
                         reduction="none").view(B, L)
    return ((ce * m).sum(1) / m.sum(1).clamp_min(1)).mean()


def sequential_distill_loss(model, x0, amask, tok, chunk=8):
    """THE METHOD: distil the sequential computation into a single pass.

    Sequential decoding is strong because each step can read the digits already
    written, letting the carry chain unroll across passes. A one-pass parallel
    prediction has to fit that whole chain inside a fixed depth, which is why
    accuracy collapses as passes are reduced.

    So we train the ONE-PASS prediction, from the fully masked answer, to match
    the answer the model produces when it is allowed to decode sequentially.
    Because the answer is unique, the teacher's target is the same every time it
    sees the same prompt - which is exactly what was missing on the free-choice
    probes, where the teacher's arbitrary pick averaged back to uniform and the
    signal cancelled.

    Ground truth is available here, so the teacher is only a convenience: the
    loss below supervises the fully-masked one-pass prediction directly against
    the true answer, which is the strongest possible version of the same idea.
    """
    B, L = x0.shape
    xt = torch.where(amask, torch.full_like(x0, tok.mask), x0)   # whole answer hidden
    logits = model(xt)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x0.reshape(-1),
                         reduction="none").view(B, L)
    return ((ce * amask).sum(1) / amask.sum(1).clamp_min(1)).mean()


def pmd_loss(model, x0, amask, tok, mode="mdlm", K=8, lam=1.0):
    """Parallelism-Matched Diffusion.

      mdlm     standard uniform-t training (baseline)
      matched  train only at the masking ratios a K-pass decode visits
      distill  uniform training + one-pass supervision (the method)
      full     matched schedule + one-pass supervision
    """
    if mode == "mdlm":
        return cond_mdlm_loss(model, x0, amask, tok, "uniform")
    if mode == "matched":
        return cond_mdlm_loss(model, x0, amask, tok, "matched", K)
    base = cond_mdlm_loss(model, x0, amask, tok,
                          "matched" if mode == "full" else "uniform", K)
    return base + lam * sequential_distill_loss(model, x0, amask, tok)


@torch.no_grad()
def cond_decode(model, x0, amask, tok, steps, device):
    """Reveal the answer over `steps` passes, prompt fixed throughout."""
    x = torch.where(amask, torch.full_like(x0, tok.mask), x0).to(device)
    am = amask.to(device)
    B, L = x.shape
    n_ans = int(am[0].sum())
    pred = None
    for s in range(steps, 0, -1):
        masked = (x == tok.mask) & am
        if not masked.any():
            break
        logits = model(x)
        conf, pred = logits.softmax(-1).max(-1)
        conf = conf.masked_fill(~masked, -1e9)
        left = int(n_ans * (s - 1) / steps)
        for b in range(B):
            k = max(0, int(masked[b].sum()) - left)
            if k:
                idx = conf[b].topk(k).indices
                x[b, idx] = pred[b, idx]
    rem = (x == tok.mask) & am
    if rem.any() and pred is not None:
        x[rem] = pred[rem]
    return x


def pmd_progressive(model, x0, amask, tok, progress, K_max=16, lam_max=1.0,
                    anneal="log"):
    """PARALLELISM CURRICULUM - the fix for the instability at long chains.

    Supervising the fully-masked one-pass prediction from step 0 is a very hard
    objective for a long carry chain: at 8 digits every seed found it, at 12 most
    collapsed to zero. The target is right but the optimisation is not reachable
    from random initialisation in one jump.

    So we anneal along the parallelism axis instead. Early training is decoded
    over many passes, where each pass reads the digits already written and the
    chain unrolls cheaply. K then halves toward 1, each stage a small increment
    from the last, while the one-pass term is ramped in rather than imposed.
    This is progressive distillation's schedule applied to the number of
    PARALLEL commits rather than the number of noise levels, and unlike prior
    step-distillation work it runs inside pretraining from scratch.

    Addition's carry is a prefix scan, so a log-depth solution exists (carry
    lookahead) - the curriculum is about making it discoverable, not about
    adding capacity.

    progress in [0,1]; returns (loss, K_now, lam_now) so the schedule is logged.
    """
    if anneal == "log":
        # 16 -> 8 -> 4 -> 2 -> 1, equal time per halving
        K_now = max(1, int(K_max / (2 ** int(progress * (math.log2(K_max) + 1e-9)))))
    else:
        K_now = max(1, int(round(K_max * (1 - progress) + 1 * progress)))
    lam_now = lam_max * min(1.0, max(0.0, (progress - 0.25) / 0.5))

    loss = cond_mdlm_loss(model, x0, amask, tok, "matched", K_now)
    if lam_now > 0:
        loss = loss + lam_now * sequential_distill_loss(model, x0, amask, tok)
    return loss, K_now, lam_now
