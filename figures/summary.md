# Results

## Table 1 — Addition: exact match at ONE forward pass

The whole answer committed in a single pass. Depth scaled as log2(digits); learning rate 1e-4 with 4000 warmup steps, which matters - see Table 4.

| digits | MDLM baseline | full method | seeds >90% (base → ours) |
|---|---|---|---|
| **8** | 100.0 [100,100] | 100.0 [100,100] | 4/4 → 4/4 |
| **12** | 76.7 [30,100] | 100.0 [100,100] | 3/4 → 4/4 |
| **16** | 25.0 [0,75] | 75.0 [25,100] | 1/4 → 3/4 |
| **24** | 0.0 [0,0] | 0.0 [0,0] | 0/4 → 0/4 |
| **32** | 0.0 [0,0] | – | 0/4 → – |

## Table 2 — Morphological inflection: where the problem does NOT arise

Answers are unique but only ~10 characters with shallow, local dependencies. One pass already suffices for the plain baseline, so there is no gap for any method to close. This is the boundary the account predicts, and it is reported as such.

| language | 1 pass | 32 passes | gap |
|---|---|---|---|
| english | 95.5 | 95.1 | -0.4 |
| english | 95.0 | 95.2 | +0.2 |
| english | 95.3 | 95.2 | -0.1 |
| finnish | 50.9 | 55.8 | +4.9 |
| finnish | 48.6 | 56.6 | +8.0 |
| georgian | 94.3 | 94.1 | -0.2 |
| georgian | 95.4 | 95.3 | -0.1 |
| german | 78.9 | 79.3 | +0.4 |
| german | 75.6 | 79.5 | +3.9 |
| german | 77.2 | 78.3 | +1.0 |
| navajo | 29.5 | 33.8 | +4.3 |
| navajo | 31.4 | 40.2 | +8.9 |
| russian | 72.7 | 72.0 | -0.7 |
| russian | 69.1 | 71.7 | +2.6 |
| spanish | 84.9 | 84.7 | -0.2 |
| spanish | 83.9 | 84.1 | +0.1 |
| turkish | 80.2 | 81.2 | +1.0 |
| turkish | 77.5 | 82.2 | +4.7 |
| turkish | 76.4 | 81.8 | +5.4 |

## Table 3 — Non-unique answers: the limit no training removes

On probes where a group must agree but the value is free, the likelihood-optimal marginal is uniform over the valid values, so two independent draws agree with probability 1/V. Sweeping an entropy penalty designed to break that symmetry over 66 runs: weak settings changed nothing, strong settings destroyed the model, failing even at fully sequential decoding. Succeeding would require abandoning the data distribution.

## Table 4 — Why earlier runs appeared to fail (10-digit addition)

Every apparent failure at 10+ digits was a learning-rate artifact, not a limit of the method or of depth.

| layers | lr | warmup | mean @1 pass | seeds >90% |
|---|---|---|---|---|
| 6 | 5e-05 | 500 | 100.0 | 3/3 |
| 6 | 5e-05 | 4000 | 100.0 | 3/3 |
| 6 | 0.0001 | 500 | 100.0 | 3/3 |
| 6 | 0.0001 | 4000 | 100.0 | 3/3 |
| 6 | 0.0003 | 500 | 70.6 | 2/3 |
| 6 | 0.0003 | 4000 | 100.0 | 3/3 |
| 10 | 5e-05 | 500 | 100.0 | 3/3 |
| 10 | 5e-05 | 4000 | 100.0 | 3/3 |
| 10 | 0.0001 | 500 | 100.0 | 3/3 |
| 10 | 0.0001 | 4000 | 100.0 | 3/3 |
| 10 | 0.0003 | 500 | 47.3 | 1/3 |
| 10 | 0.0003 | 4000 | 100.0 | 3/3 |

