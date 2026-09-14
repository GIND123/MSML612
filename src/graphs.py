"""Structured graph families where the information-theoretic quantities are EXACT.

The bound this project rests on,

    TC(S) = sum_i H(x_i | c) - H(S | c)  <=  sum_i H(x_i | c),

has so far been asserted rather than measured. On addition TC is zero by
construction, and on natural text H(S | c) is not computable, so the entropy
budget is motivated by a quantity we never verify. That is the soft spot.

Graphs close it. A graph on n nodes is a binary sequence of n(n-1)/2 edge
indicators, so the same masked-diffusion machinery applies unchanged - but for
small n the ENTIRE family can be enumerated, which makes every quantity exact:

  * the true edge marginals p(e_i = 1),
  * the true joint entropy H(S) for any subset of edges,
  * therefore the true total correlation TC(S),
  * and - the decisive one - the BEST POSSIBLE one-pass validity,

        V* = sum over valid graphs g of  prod_i p_i(g_i),

    which is the probability that independent draws from the true marginals
    happen to land on a valid graph. No model, however well trained, can beat
    V* at one pass, because V* already assumes perfect marginals.

That last quantity turns a qualitative claim into an airtight one. If a trained
model reaches V*, the remaining gap to 100% is *provably* information-theoretic
rather than an optimisation failure - which is exactly what mode (ii) asserts
and what no previous experiment here could demonstrate.

Enumeration is 2^(n(n-1)/2) graphs, so n <= 7 is the practical range: n=6 gives
32,768 candidates and n=7 gives 2,097,152.
"""
import math
from functools import lru_cache

import numpy as np


def edge_list(n):
    """Upper-triangular edge slots in a fixed canonical order."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def n_edges(n):
    return n * (n - 1) // 2


def mask_to_adj(mask, n):
    a = np.zeros((n, n), dtype=np.int8)
    for k, (i, j) in enumerate(edge_list(n)):
        if (mask >> k) & 1:
            a[i, j] = a[j, i] = 1
    return a


def degrees(mask, n):
    d = [0] * n
    for k, (i, j) in enumerate(edge_list(n)):
        if (mask >> k) & 1:
            d[i] += 1
            d[j] += 1
    return d


def _components(mask, n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cnt = 0
    for k, (i, j) in enumerate(edge_list(n)):
        if (mask >> k) & 1:
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
            cnt += 1
    return len({find(x) for x in range(n)}), cnt


def has_triangle(mask, n):
    adj = mask_to_adj(mask, n)
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j]:
                continue
            for k in range(j + 1, n):
                if adj[i, k] and adj[j, k]:
                    return True
    return False


# ---------------------------------------------------------------- families ---
# Each predicate defines a set of graphs; the distribution is uniform over it,
# which is the maximum-entropy choice and keeps H(S) = log2(|family|) exact.

def _is_perfect_matching(mask, n):
    return all(d == 1 for d in degrees(mask, n))


def _is_two_regular(mask, n):
    return all(d == 2 for d in degrees(mask, n))


def _is_tree(mask, n):
    comps, m = _components(mask, n)
    return comps == 1 and m == n - 1


def _is_triangle_free_dense(mask, n):
    # enough edges that the constraint actually bites, and no triangle
    return bin(mask).count("1") >= n and not has_triangle(mask, n)


def _is_bipartite_fixed(mask, n):
    half = n // 2
    for k, (i, j) in enumerate(edge_list(n)):
        if (mask >> k) & 1 and ((i < half) == (j < half)):
            return False
    return bin(mask).count("1") >= half


FAMILIES = {
    "matching": _is_perfect_matching,      # every degree exactly 1
    "2regular": _is_two_regular,           # every degree exactly 2
    "tree": _is_tree,                      # connected and acyclic
    "trianglefree": _is_triangle_free_dense,
    "bipartite": _is_bipartite_fixed,
}


@lru_cache(maxsize=None)
def enumerate_family(name, n):
    """Every graph in the family, as a tuple of bit masks. Exhaustive."""
    pred = FAMILIES[name]
    E = n_edges(n)
    if E > 22:
        raise ValueError(f"n={n} needs 2^{E} enumeration; use n <= 7")
    return tuple(m for m in range(1 << E) if pred(m, n))


def exact_stats(name, n):
    """Every information-theoretic quantity, computed exactly by enumeration.

    Returns a dict with the true marginals, the summed marginal entropy (the
    BOUND), the true joint entropy, the true total correlation, and V* - the
    highest one-pass validity any model can achieve.
    """
    fam = enumerate_family(name, n)
    M = len(fam)
    if M == 0:
        raise ValueError(f"family {name} is empty at n={n}")
    E = n_edges(n)

    bits = np.zeros((M, E), dtype=np.float64)
    for r, m in enumerate(fam):
        for k in range(E):
            bits[r, k] = (m >> k) & 1

    p = bits.mean(0)                                  # true edge marginals
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(p > 0, p * np.log2(p), 0.0) +
              np.where(p < 1, (1 - p) * np.log2(1 - p), 0.0))
    sum_h = float(h.sum())                            # the BOUND
    h_joint = math.log2(M)                            # uniform over the family
    tc = sum_h - h_joint                              # the TRUE dependence

    # V*: probability that independent draws from the true marginals land in
    # the family. This is the ceiling on one-pass validity for ANY model.
    logp1 = np.where(p > 0, np.log(np.clip(p, 1e-300, 1)), -np.inf)
    logp0 = np.where(p < 1, np.log(np.clip(1 - p, 1e-300, 1)), -np.inf)
    v = 0.0
    for r in range(M):
        lp = float(np.where(bits[r] > 0, logp1, logp0).sum())
        v += math.exp(lp)

    return {"family": name, "n": n, "n_edges": E, "size": M,
            "marginals": p.tolist(),
            "sum_marginal_entropy_bits": sum_h,
            "joint_entropy_bits": h_joint,
            "total_correlation_bits": tc,
            "one_pass_ceiling": v,
            "bound_holds": tc <= sum_h + 1e-9}


def sample(name, n, k, seed=0):
    """Draw k graphs uniformly from the family, as (k, E) arrays of 0/1."""
    fam = enumerate_family(name, n)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(fam), size=k)
    E = n_edges(n)
    out = np.zeros((k, E), dtype=np.int64)
    for r, t in enumerate(idx):
        m = fam[t]
        for b in range(E):
            out[r, b] = (m >> b) & 1
    return out


def is_valid(rows, name, n):
    """Validity of generated edge sequences, checked against the family."""
    pred = FAMILIES[name]
    ok = np.zeros(len(rows), dtype=bool)
    for r, row in enumerate(rows):
        m = 0
        for b, v in enumerate(row):
            if v == 1:
                m |= (1 << b)
        ok[r] = pred(m, n)
    return ok


class EdgeTokenizer:
    """Three symbols: absent, present, and the absorbing state."""

    def __init__(self):
        self.itos = ["0", "1", "[MASK]"]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.mask = 2
        self.pad = 2

    def __len__(self):
        return 3


if __name__ == "__main__":
    print(f"{'family':<14}{'n':>3}{'size':>8}{'sum H_i':>10}{'H(S)':>8}"
          f"{'TC':>8}{'V* (1-pass ceiling)':>22}")
    for n in (6, 7):
        for name in FAMILIES:
            try:
                s = exact_stats(name, n)
            except ValueError:
                continue
            print(f"{name:<14}{n:>3}{s['size']:>8}"
                  f"{s['sum_marginal_entropy_bits']:>10.3f}"
                  f"{s['joint_entropy_bits']:>8.3f}"
                  f"{s['total_correlation_bits']:>8.3f}"
                  f"{s['one_pass_ceiling']*100:>21.4f}%")
