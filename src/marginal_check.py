"""Are the failing family's marginals position-DEPENDENT, and the succeeding ones not?

At t = 1 every input token is [MASK], so the input sequence is constant. With a
purely RELATIVE position encoding the attention score between positions i and j
depends only on (i - j); when every value vector is identical the layer output is
v * sum(attn) = v, the same at every position. Boundary effects aside, a
RoPE-only model therefore cannot express position-DEPENDENT marginals at the
fully-masked state. This is an expressivity limit, not an optimisation one, so
more training cannot fix it.

That predicts exactly which families succeed: those whose true marginals are
constant across edge slots.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from graphs import exact_stats, FAMILIES

print(f"{'family':<14}{'n':>3}{'V*':>10}{'marginal spread':>18}{'position-dependent?':>22}")
for n in (6,):
    for name in FAMILIES:
        try:
            s = exact_stats(name, n)
        except ValueError:
            continue
        p = np.array(s["marginals"])
        spread = float(p.max() - p.min())
        dep = "YES - RoPE cannot" if spread > 1e-6 else "no  - RoPE suffices"
        print(f"{name:<14}{n:>3}{s['one_pass_ceiling']*100:>9.2f}%{spread:>18.4f}{dep:>22}")
        print(f"{'':>17}min={p.min():.3f} max={p.max():.3f} distinct={len(set(np.round(p,4)))}")
