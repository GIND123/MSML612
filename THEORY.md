# Why parallel decoding fails, in two separate ways

Parallel decoding in masked diffusion loses accuracy for two reasons that are
usually conflated. They have different mathematics and **opposite** remedies, and
keeping them apart is what this project is about.

- **Mode (i), a computational/optimisation problem.** The marginals the decoder
  reads were never fit at the mask ratios a high-parallelism decode visits. This
  is fixable **only at training time**, and §§1–4 below are its analysis.
- **Mode (ii), an information-theoretic problem.** Committing several positions
  at once samples the product of marginals when the truth is the joint. This is
  fixable **only at inference**, by committing less, and §5 is its analysis.

Section 6 states what each predicts, which is how they are told apart
empirically.

---

The claim behind mode (i) is that a masked diffusion model can commit an entire
chained answer in a **single forward pass**. That sounds like it should be
impossible: digit *i* of a sum needs the carry out of digit *i−1*, which needs the
carry before it, and so on. The reason it is *not* impossible is a known result in
circuit complexity, and it gives the method a falsifiable prediction rather than
just a benchmark number.

---

## 1. The task is inside the class a fixed-depth transformer can express

**Integer addition is in uniform TC⁰.** The carry chain looks serial but is a
*prefix scan*: writing `g_i = a_i·b_i` (generate) and `p_i = a_i ⊕ b_i`
(propagate), the carry into position *i* is

```
c_i  =  OR over j<i of ( g_j  AND  p_{j+1..i-1} )
```

which is an associative scan and therefore computable by a **carry-lookahead**
circuit of depth **O(log n)**, not O(n). This is textbook, and it is why hardware
adders are not ripple-carry.

**Constant-depth, log-precision transformers are confined to uniform TC⁰**
(Merrill & Sabharwal, *The Parallelism Tradeoff*, TACL). Addition sits inside that
class. So a transformer of depth ~log n **can express** one-pass *n*-digit
addition. Expressibility is not the obstacle.

## 2. Sequential decoding buys depth, which is why it looks better

Chain-of-thought lets transformers solve problems that are inherently serial for a
fixed-depth network, because each emitted token adds a round of computation
(Li et al., *Chain of Thought Empowers Transformers to Solve Inherently Serial
Problems*).

Masked diffusion decoding is the same trick. A *K*-pass decode has **effective
depth ≈ L·K**: each pass re-reads the tokens committed so far, so the carry chain
unrolls across passes instead of within the network. At K = 1 the model gets only
its L layers.

This reframes the usual story. Sequential decoding is not "more careful" — it is
**borrowing depth from the decoding loop**. That is why our earlier measurements
found extra passes bought nothing once the model was already deep enough, and
everything once it was not.

## 3. So the failure is optimisation, not expressivity

Putting those together:

| | Expressible at K=1? | Learned by standard training? |
|---|---|---|
| depth ≥ ~log n | **yes** (TC⁰, carry-lookahead) | **no** |
| depth < log n | no | no |

Standard MDLM samples the masking ratio `t ~ U(0,1)`, so the fully-masked state
that one-pass decoding actually starts from is a vanishing slice of training. The
log-depth solution is representable and never searched for.

Our measurements match exactly this: at 8 digits, one-pass supervision took the
baseline from 35.2% (1 of 3 seeds) to 99.9% (3 of 3). At 12 digits the same
objective became unreachable from random initialisation — some seeds found it,
most collapsed — which is what an optimisation barrier looks like, not a capacity
ceiling.

Hence the **parallelism curriculum**: anneal K from 16 → 1 so the model first
solves the task with borrowed depth, then is progressively forced to internalise
the scan into its layers. It is progressive distillation applied to the number of
parallel commits rather than to noise levels, run inside pretraining rather than
on a trained checkpoint.

## 4. The prediction that makes this falsifiable

If the account is right, the **minimum depth for successful one-pass decoding
grows like log n, not like n**.

The falsifiable content is the **growth rate, not the absolute layer counts**.
A depth-*d* transformer of sufficient width can fold several levels of the scan
into one layer, so measured minima may sit *below* log₂(n) — and in our sweep
they do: 8-digit addition is solved at 2 layers, not the 3 that a
one-level-per-layer reading would predict. That is consistent with a log-depth
(or better) circuit and is **not** evidence against the account. What would
refute it is linear growth.

The three accounts separate cleanly on the ratio between the smallest and
largest length tested:

| account | depth(16 digits) ÷ depth(4 digits) |
|---|---|
| **prefix scan (ours)** | **~2×**, since log₂16 / log₂4 = 2 |
| ripple carry | ~4×, linear in n |
| not expressible at any depth | no depth succeeds |

So the sweep over {2,3,4,6,8,12} layers × {4,8,12,16} digits discriminates: if 16
digits needs ≲4 layers the scan account survives; if it needs ≳8 the ripple-carry
account does. This is a prediction about the *method*, not a benchmark score.

## 5. Mode (ii): the part no training can fix

Everything above concerns whether a marginal was *learned*. Mode (ii) concerns
what happens when several correct marginals are committed **simultaneously**.

Parallel decoding samples each committed position independently, i.e. from the
product ∏ᵢ p(xᵢ | c). The truth is the joint p(S | c). The gap between them is the
**total correlation** (multi-information) of the committed set:

```
TC(S)  =  D_KL( p(S|c)  ||  ∏ᵢ p(xᵢ|c) )  =  Σᵢ H(xᵢ|c) − H(S|c)   ≥  0
```

That first equality is an identity, not an approximation: the KL divergence from
a joint to the product of its own marginals *is* the total correlation. Parallel
decoding draws from the right-hand product while the data follow the left-hand
joint, so TC(S) is exactly the quantity a parallel commit discards.

Since entropies of discrete variables are non-negative, H(S | c) ≥ 0 and so

```
TC(S)  ≤  Σᵢ H(xᵢ | c).
```

**The summed conditional entropy of what you commit upper-bounds the dependence
you throw away by committing it.** Two consequences follow directly:

1. **If every committed conditional is a point mass, the bound is zero.** Then
   independent sampling reproduces the joint *exactly*, and there is no mode (ii)
   at all. Addition is this case — which is why no inference-time reordering
   helps there, and why the entire gap must be mode (i).
2. **If the conditionals carry entropy, no training objective removes the term.**
   Fitting the marginals better does not reduce TC(S); the dependence is a
   property of the data distribution. The only lever is to commit a smaller or
   lower-entropy set. Driving the marginals to point masses *would* remove it,
   but that means abandoning the data distribution — which is exactly what our
   66-run entropy-penalty sweep found: weak settings changed nothing, strong
   settings destroyed the model.

This bound is what makes the entropy budget principled rather than heuristic.
Committing while Σᵢ H(xᵢ | c) ≤ B caps the discarded dependence at B nats per
pass, so B is a dial on a quantity with units, not a tuned threshold.

### 5.1 The dependence sets an exact ceiling on one-pass decoding

The bound above says *how much* is discarded. A sharper statement says what it
costs. Let `P` be the data distribution on a set of structured objects and
`Q = ∏ᵢ pᵢ` the product of its own marginals — which is exactly what a one-pass
parallel decode samples from. Define the **one-pass ceiling**

```
V*  =  Q(valid)  =  Σ_{g valid}  ∏ᵢ pᵢ(gᵢ),
```

the probability that independent draws from the *true* marginals land on a valid
object. No model can exceed `V*` at one pass, since `V*` already assumes perfect
marginals.

**Proposition.** If `P` is uniform over a family `F` of size `M`, then

```
V*  ≥  2^(−TC),        TC = D_KL(P ‖ Q),
```

with equality iff `Q` is constant on `F`.

*Proof.* `TC = −log₂M − (1/M) Σ_{g∈F} log₂Q(g)`. By Jensen,
`(1/M) Σ log₂Q(g) ≤ log₂((1/M) Σ Q(g)) = log₂(V*/M)`. Substituting gives
`−TC ≤ log₂V*`. Equality in Jensen holds iff `Q(g)` is constant on `F`. ∎

**Every bit of total correlation among jointly committed variables at most
halves the one-pass success probability.**

This is verified exactly rather than estimated. Enumerating graph families on
6 and 7 nodes gives closed-form `pᵢ`, `H(S) = log₂M`, `TC`, and `V*`:

| family | n | \|F\| | TC (bits) | V* | 2^(−TC) | ratio |
|---|---|---|---|---|---|---|
| bipartite | 7 | 4017 | 0.026 | 98.2816% | 98.1817% | 1.001 |
| bipartite | 6 | 466 | 0.112 | 93.6324% | 92.5073% | 1.012 |
| tree | 6 | 1296 | 3.435 | **9.2488%** | **9.2488%** | **1.00000** |
| trianglefree | 6 | 2335 | 3.571 | 8.5380% | 8.4147% | 1.015 |
| tree | 7 | 16807 | 4.089 | **5.8771%** | **5.8771%** | **1.00000** |
| matching | 6 | 15 | 6.922 | **0.8246%** | **0.8246%** | **1.00000** |
| 2regular | 6 | 70 | 8.435 | **0.2889%** | **0.2889%** | **1.00000** |
| 2regular | 7 | 465 | 10.423 | **0.0728%** | **0.0728%** | **1.00000** |

The inequality holds in every row. Equality holds to five decimal places on
exactly the families with a **fixed edge count** — matching, 2-regular, tree —
where symmetry makes `Q` constant on `F`, and fails precisely where edge counts
vary (bipartite, triangle-free), which is what the Jensen step predicts. The
proposition is therefore confirmed in both its equality and its strict-inequality
cases.

### 5.2 Why this makes the two modes separable by measurement

On addition `TC = 0`, so no gap can be attributed to mode (ii). On text `H(S|c)`
is not computable, so neither mode can be isolated numerically. On these graph
families both are exact at once, which turns the decomposition into an
experiment rather than an argument:

* a model that **reaches `V*`** at one pass has perfect marginals, so its
  residual failure is entirely mode (ii) and **provably not trainable**;
* a model that **falls short of `V*`** is short by exactly mode (i), the
  trainable part, and budget conditioning should recover it.

## 6. How the two modes are told apart

They make opposite predictions, which is what lets the experiments separate them:

| | mode (i) present | mode (ii) present |
|---|---|---|
| more inference passes help? | **no** — marginals are wrong at every budget | **yes** — smaller commits, less discarded TC |
| better commit *ordering* helps? | no | yes |
| more/better training helps? | **yes** | no |
| entropy of the conditionals | irrelevant | **it is the whole story** |

Addition is pure mode (i): unique answers, so TC = 0 by construction. The
free-choice probes are pure mode (ii): maximum-entropy conditionals, nothing to
learn. Natural text has both at once, which is why it is the test of whether the
two fixes compose rather than interfere.

## References

- Merrill & Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL.
- Merrill & Sabharwal. *A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers.*
- Li et al. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.*
- Sahoo et al. *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024.
- Salimans & Ho. *Progressive Distillation for Fast Sampling of Diffusion Models.*
- Watanabe. *Information theoretical analysis of multivariate correlation.* (total correlation)
- Cover & Thomas. *Elements of Information Theory*, ch. 2 (entropy bounds used in §5).
