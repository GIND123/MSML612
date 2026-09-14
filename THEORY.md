# Why one-pass parallel decoding should be possible — and why training misses it

The empirical claim in this project is that a masked diffusion model can commit an
entire chained answer in a **single forward pass**. That sounds like it should be
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
grows with log n, not with n**:

| digits n | predicted minimum depth |
|---|---|
| 4 | ~2 |
| 8 | ~3 |
| 16 | ~4 |
| 32 | ~5 |

A ripple-carry account predicts depth linear in n. A "transformers cannot do this"
account predicts no depth suffices. These are cleanly distinguishable by measuring
the minimum depth at which one-pass accuracy crosses a threshold, and it is a
prediction about the *method*, not merely a benchmark score.

The same framing says where the method must **fail**, which is equally important:
when the answer is not unique, the obstacle is information-theoretic rather than
computational, no depth or curriculum helps, and our 66-run entropy sweep confirmed
it.

## References

- Merrill & Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL.
- Merrill & Sabharwal. *A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers.*
- Li et al. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.*
- Sahoo et al. *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024.
- Salimans & Ho. *Progressive Distillation for Fast Sampling of Diffusion Models.*
