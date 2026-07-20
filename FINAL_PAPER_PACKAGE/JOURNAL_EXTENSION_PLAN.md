# Journal Extension Plan: RC-STAR for Safe Reactive Human-Robot Handover

## Working Title

**RC-STAR: Reactive Cooperative Safe Transfer with Active Shielding for Human-Robot Handover**

## Core Journal Claim

The conference paper shows that a fixed scripted-human baseline fails to produce robust handover behavior, while RC-STAR substantially improves task success and safety under the SaRA PFL shield. The journal extension should broaden this into a stronger scientific claim:

> Safe robot-to-human handover requires both formal action shielding and a learnable interaction event. Reactive human motion improves coordination only when paired with robust capture, release-confirmed success, and shield-aware policy learning.

## Main Contributions

1. **Reactive cooperative human model**
   - Delayed cooperative acceptance.
   - Bounded hand motion toward the transported object.
   - Avoids early target drift and unreachable target deadlock.

2. **Robust capture and release-confirmed success**
   - Capture triggered by object-surface proximity to the receiving palm.
   - Terminal success requires stable release and human-object possession.

3. **Shield-aware reinforcement learning**
   - SAC + SIR/RSI support.
   - SaRA PFL safety shield active during training and evaluation.
   - Optional failsafe-intervention penalty / governor mechanism.

4. **Root-cause analysis**
   - Baseline, reactive-only, original STAR, and final RC-STAR comparison.
   - Show why simple safety constraints and simple reactivity are insufficient.

## Current Final Results

| Metric | Baseline RHH-SIR | RC-STAR |
|---|---:|---:|
| Success rate | 4.6% | 72.6% |
| Shield activations / episode | 21.80 | 34.98 |
| Human collisions / episode | 0.50 | 0.00 |
| Critical collisions / episode | 6.038 | 0.228 |
| Episode reward | -810.40 | 21.06 |

## Cross-Seed 95% Confidence Intervals

Matched evaluation, 5 independent seeds x 100 evaluation episodes per
condition (500 episodes total per condition). Each seed's 100 episodes are
first reduced to one per-seed mean; the cross-seed mean +/- 95% CI is then
computed over the 5 per-seed means (t-distribution, df=4, t=2.776), so the
seed — not the raw episode — is the statistical unit.

| Metric | Baseline RHH-SIR | RC-STAR |
|---|---:|---:|
| Success rate | 0.046 +/- 0.121 | 0.726 +/- 0.071 |
| Shield interventions / episode | 21.80 +/- 27.25 | 34.98 +/- 2.92 |
| Human collisions / episode | 0.502 +/- 0.524 | 0.000 +/- 0.000 |
| Critical collisions / episode | 6.038 +/- 7.764 | 0.228 +/- 0.030 |
| Episode reward | -810.40 +/- 301.04 | 21.06 +/- 1.74 |

Full table: `csv_results/evaluation/cross_seed_ci_table.csv`.
Full writeup: `CROSS_SEED_CI_RESULTS.md`.

**Interpretation:** Baseline RHH-SIR has high seed-to-seed variance — its
per-seed means for success rate, shield interventions, and critical
collisions swing widely (e.g. one seed shows 0 shield interventions and 0
critical collisions, essentially failing immediately, while another seed
shows 58.25 interventions and 14.62 critical collisions). This variance
produces wide baseline CIs; for nonnegative metrics, the unbounded
t-based interval can even extend below zero. RC-STAR, in contrast, is
stable across seeds on every metric (e.g. success rate 0.65-0.78 across
all 5 seeds, critical collisions 0.19-0.25), with especially tight CIs for
shield interventions, critical collisions, and reward. This asymmetry is
itself a result worth stating explicitly: RC-STAR is not only better on
average, it is more reproducible across training seeds than the baseline.

## Recommended Journal Additions

### 1. More ablation structure

Add a clear ablation table:

| Variant | Reactive human | Robust capture | Release confirmation | Governor | Success | Critical collisions |
|---|---|---|---|---|---:|---:|
| Baseline RHH-SIR | No | No | Default | No | 4.6% | 6.038 |
| Reactive-only | Yes | No | Default | No | Low | High/variable |
| Original STAR | Yes | Partial | Partial | No | 0.0% | Lower |
| RC-STAR | Yes | Yes | Yes | Yes | 72.6% | 0.228 |

### 2. Learning-curve analysis

Include:

- success vs. training steps
- episode reward vs. training steps
- shield interventions vs. training steps
- per-seed variance

### 3. Failure-mode analysis

Classify failures:

- timeout
- early collision termination
- capture without release
- failed approach
- failed grasp

### 4. Safety-efficiency tradeoff

Discuss why RC-STAR has more shield interventions:

> RC-STAR enters the close handover zone more frequently because it actually completes the transfer, whereas baseline often times out far from successful transfer. Therefore, increased shield intervention is not necessarily worse; it indicates active safety mediation during close interaction.

### 5. Limitations

Be explicit:

- simulation-only study
- animated human model, not real human subject data
- single task/environment
- CPU-only training due to old PyTorch/CUDA compatibility
- SaRA shield guarantees are simulation-controller guarantees, not full system-level real-world guarantees

## Recommended Journal Structure

1. **Introduction**
   - Safe HRC handover problem.
   - Limitation of scripted human motion.
   - Need for learnable safe interaction events.

2. **Related Work**
   - Safe RL.
   - Shielded robot control.
   - Human-aware handover.
   - Reactive human modeling.

3. **Problem Formulation**
   - MDP / constrained MDP.
   - Robot-to-human handover.
   - Safety metrics and task metrics.

4. **Method: RC-STAR**
   - Reactive human acceptance.
   - Robust capture.
   - Release-confirmed success.
   - Predictive governor / shield integration.
   - Reward and imitation learning.

5. **Experimental Setup**
   - Human-Robot Gym.
   - Robot and task.
   - Baseline and variants.
   - Training details.
   - Evaluation protocol.

6. **Results**
   - Main table.
   - Learning curves.
   - Ablations.
   - Failure modes.
   - Safety tradeoffs.

7. **Discussion**
   - Why simple reactivity failed.
   - Why robust capture matters.
   - Why shield intervention should be interpreted carefully.

8. **Conclusion**
   - Summary and future work.

## Extra Experiments Worth Doing Before Journal Submission

Priority 1:

- Re-evaluate RC-STAR and baseline with the same number of episodes and matched seeds. **Done** — 5 seeds x 100 episodes per condition.
- Produce confidence intervals for success and collision metrics. **Done** — see "Cross-Seed 95% Confidence Intervals" above and `CROSS_SEED_CI_RESULTS.md`.

Priority 2:

- Run no-shield RC-STAR evaluation if safe in simulation only.
- Run shield-only / no-governor ablation.

Priority 3:

- Add one transferability test:
  - different human animation split, or
  - altered object position distribution, or
  - perturbed human offset.

## Journal Target Type

Good venue families:

- robotics and automation journals
- intelligent robots / HRI journals
- applied AI / safe RL special issues

The paper is strongest as a **simulation-based safe HRC method paper with root-cause analysis**, not as a human-subject study.
