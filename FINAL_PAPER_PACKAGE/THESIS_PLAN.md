# Thesis Plan: Safe Reinforcement Learning for Reactive Human-Robot Handover

## Working Thesis Title

**Safe Reinforcement Learning for Reactive Human-Robot Handover Using Shielded Control and Robust Capture**

## Thesis Central Question

> How can a robot learn to complete a human handover safely when the human is reactive rather than following a fixed pre-planned trajectory?

## Thesis Argument

The thesis should argue that safe handover is not solved by shielding alone. A robot also needs a learnable handover event and a human model that produces meaningful interaction timing. RC-STAR addresses this by combining reactive human acceptance, robust capture, release-confirmed success, and SaRA shielded control.

## Proposed Chapter Structure

### Chapter 1: Introduction

Contents:

- Human-robot collaboration motivation.
- Robot-to-human handover as a core HRC problem.
- Safety and coordination challenges.
- Why scripted human motion is limited.
- Research questions.
- Contributions.

Possible research questions:

1. Does reactive human motion change the safety and coordination behavior of learned handover policies?
2. Is safety shielding alone sufficient for successful handover?
3. What task-event design makes reactive handover learnable?

### Chapter 2: Background and Related Work

Sections:

- Reinforcement learning for robotics.
- Soft Actor-Critic.
- Imitation reward, SIR, and RSI.
- Human-robot handover.
- Safe RL and shielding.
- SaRA / PFL safety shield.
- Human motion modeling and reactive human behavior.

### Chapter 3: Baseline System and Problem Diagnosis

Goal:

Explain the starting point.

Sections:

- Human-Robot Gym.
- RobotHumanHandoverCart environment.
- Baseline RHH-SIR setup.
- Expert dataset collection.
- Baseline training and evaluation.
- Root-cause diagnosis:
  - timeout failures
  - sparse/fragile transfer event
  - shield alone does not ensure completion

Key baseline result:

| Metric | Baseline |
|---|---:|
| Success rate | 4.6% |
| Shield activations / episode | 21.80 |
| Human collisions / episode | 0.50 |
| Critical collisions / episode | 6.038 |

### Chapter 4: Reactive Human Modeling

Goal:

Present the development path from simple reactive motion to the final cooperative model.

Sections:

- Initial reactive-speed idea.
- Deadlock problem with too-small nominal motion.
- Final nonzero cooperative human acceptance.
- Bounded reactive target shift.
- Why delayed cooperation avoids early target drift.

Important lesson:

> A human model that is reactive but too passive creates a coordination deadlock: the human waits for robot progress, while the robot needs the human target to become reachable.

### Chapter 5: RC-STAR Method

Sections:

- System architecture.
- Reactive cooperative human acceptance.
- Robust capture mechanism.
- Strict release-confirmed success.
- Predictive human-aware governor.
- SaRA PFL shield integration.
- Reward structure.
- Training pipeline.

Include architecture figure:

- `FINAL_PAPER_PACKAGE/figures/paper_figures/fig5_architecture.png`

### Chapter 6: Experiments

Sections:

- Hardware/software setup.
- Simulation environment.
- Training configuration.
- Dataset collection.
- Evaluation protocol.
  - 5 independent seeds x 100 evaluation episodes per condition (matched
    across baseline and RC-STAR).
  - Confidence intervals computed at the seed level: each seed's 100
    episodes reduce to one per-seed mean, and the reported 95% CI is over
    the 5 per-seed means (t-distribution, df=4), not over raw episodes.
- Metrics:
  - success
  - safe success
  - timeout
  - shield interventions
  - human collisions
  - critical collisions
  - episode reward

### Chapter 7: Results and Analysis

Main table:

| Metric | Baseline RHH-SIR | RC-STAR |
|---|---:|---:|
| Success rate | 4.6% | 72.6% |
| Shield activations / episode | 21.80 | 34.98 |
| Human collisions / episode | 0.50 | 0.00 |
| Critical collisions / episode | 6.038 | 0.228 |
| Episode reward | -810.40 | 21.06 |

Cross-seed 95% confidence intervals (5 independent seeds x 100 evaluation
episodes per condition; each seed reduced to one per-seed mean, then
mean +/- 95% CI computed across the 5 per-seed means, t-distribution df=4):

| Metric | Baseline RHH-SIR | RC-STAR |
|---|---:|---:|
| Success rate | 0.046 +/- 0.121 | 0.726 +/- 0.071 |
| Shield interventions / episode | 21.80 +/- 27.25 | 34.98 +/- 2.92 |
| Human collisions / episode | 0.502 +/- 0.524 | 0.000 +/- 0.000 |
| Critical collisions / episode | 6.038 +/- 7.764 | 0.228 +/- 0.030 |
| Episode reward | -810.40 +/- 301.04 | 21.06 +/- 1.74 |

Full table: `csv_results/evaluation/cross_seed_ci_table.csv`.
Full writeup: `CROSS_SEED_CI_RESULTS.md`.

Baseline RHH-SIR shows high seed-to-seed variance — its per-seed success
rate, shield-intervention, and critical-collision means swing widely
across the 5 seeds (one seed effectively fails immediately with 0 shield
interventions and 0 critical collisions, another reaches 58.25
interventions and 14.62 critical collisions), so its CIs are wide; for
nonnegative metrics, the unbounded t-based interval can even extend below
zero. RC-STAR is stable across all 5 seeds on every metric, with especially
tight CIs for shield interventions, critical collisions, and reward. This
should be presented as a result in its own right: RC-STAR is not only
better on average, it is more reproducible across training seeds.

Discussion:

- RC-STAR improves success dramatically.
- RC-STAR eliminates human collisions in the matched evaluation.
- RC-STAR nearly eliminates critical collisions.
- Shield activations increase because successful handover requires close interaction.
- The shield becomes an active mediator rather than a sign of failure.

### Chapter 8: Limitations and Future Work

Limitations:

- Simulation only.
- No real human-subject validation.
- Single task/environment.
- Animated human model is simplified.
- Results depend on capture/release definitions.

Future work:

- Real robot validation.
- More human behavior models.
- Multiple object types.
- Multi-human or dynamic workspace scenarios.
- Formal analysis of shield-intervention tradeoffs.

### Chapter 9: Conclusion

Key thesis conclusion:

> Safe handover requires both safety constraints and a learnable coordination event. RC-STAR shows that reactive human modeling can improve handover success and safety when paired with robust capture, release-confirmed success, and active shielded control.

## Appendices

Recommended appendices:

- Appendix A: Full configuration files.
- Appendix B: Dataset and evaluation CSV paths.
- Appendix C: Additional learning curves.
- Appendix D: Failure examples.
- Appendix E: Reproducibility commands.

## Files to Use

Clean package:

- `FINAL_PAPER_PACKAGE/`

Figures:

- `FINAL_PAPER_PACKAGE/figures/paper_figures/`

Results:

- `FINAL_PAPER_PACKAGE/csv_results/evaluation/`

Reproducibility:

- `FINAL_PAPER_PACKAGE/REPRODUCIBILITY.md`

## Suggested Thesis Writing Order

1. Chapter 3: Baseline and diagnosis.
2. Chapter 5: RC-STAR method.
3. Chapter 7: Results.
4. Chapter 4: Reactive human model.
5. Chapter 2: Related work.
6. Chapter 1: Introduction.
7. Chapter 8: Limitations.
8. Chapter 9: Conclusion.

This order is easier than writing from Chapter 1 forward because the technical core is already known.
