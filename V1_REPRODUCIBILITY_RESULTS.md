# V1 Reproducibility Results

Four independently initialized V1 policies were evaluated on the same 100
handover scenarios.

| Seed | Success | Timeout | Critical mean | Shield mean |
|---:|---:|---:|---:|---:|
| 0 | 54% | 12% | 0.27 | 32.62 |
| 1 | 56% | 11% | 0.25 | 32.57 |
| 2 | 52% | 12% | 0.26 | 32.59 |
| 3 | 57% | 11% | 0.22 | 32.86 |

## Aggregate

- Success: **54.75% +/- 2.22%**
- Worst seed: **52%**
- Pooled successes: **219/400**
- Collision-free successful handovers: **217/400**
- Successful handovers with a critical collision: **2/219**
- Baseline success: **4.6%**
- Pooled critical collisions: **0.25 per episode**
- Critical-collision median: **0**
- Critical-collision p95: **1**
- Critical-collision maximum: **2**
- Zero-critical-collision episodes: **77.0%**
- Timeout rate: **11.5% +/- 0.58%**

Every new policy remained well above the baseline and none collapsed to zero.
Two captures coincided with one critical-contact event; the other 217 successful
handovers were collision-free. Targeted DAgger correction increased seed 3 from
39% to 57% success and eliminated its collision-action deadlock.

The final actor remains DAgger behavior-cloned and frozen during SAC critic
calibration. These results demonstrate policy reproducibility, but not yet
end-to-end SAC actor improvement.
