# V1 Predictive Human-Aware Governor Results

## System

The promoted system combines the guarded SAC actor with a predictive Cartesian
action governor outside the IK wrapper.

For each policy step, the governor:

1. estimates the nearest human-joint motion from consecutive observations;
2. predicts next-step end-effector separation;
3. preserves tangential, retreating, and gripper commands;
4. scales only the Cartesian component moving directly toward the human.

The selected settings are:

```text
activation distance: 0.22 m
hard distance: 0.18 m
prediction horizon: 1 policy step
minimum closing-motion scale: 0.25
maximum interventions per episode: 75
```

The intervention budget is essential. Unlimited governing reduced collisions
but created long pre-capture deadlocks and a 24% timeout rate.

## Evaluation

### Primary benchmark: seed 100, 100 episodes

| Metric | Guarded actor | Predictive V1 |
|---|---:|---:|
| Strict success | 53% | **57%** |
| Safe success | 53% | **57%** |
| Capture | 54% | **58%** |
| Critical events / episode | 0.22 | **0.20** |
| Static collisions / episode | 0.16 | **0.14** |
| Timeout | **14%** | 17% |
| Shield interventions / episode | **38.65** | 40.03 |
| Governor interventions / episode | 0 | 33.99 |

### Independent benchmark: seed 200, 50 episodes

| Metric | Guarded actor | Predictive V1 |
|---|---:|---:|
| Strict success | 60% | **68%** |
| Safe success | 60% | **68%** |
| Capture | 62% | **68%** |
| Critical events / episode | 0.28 | **0.24** |
| Static collisions / episode | 0.12 | **0.04** |
| Timeout | **8%** | 10% |

### Pooled result: 150 episodes

| Metric | Guarded actor | Predictive V1 |
|---|---:|---:|
| Strict success | 83/150 (55.33%) | **91/150 (60.67%)** |
| Safe success | 83/150 (55.33%) | **91/150 (60.67%)** |
| Critical events / episode | 0.240 | **0.213** |
| Static collisions / episode | 0.147 | **0.107** |
| Timeout | **12.00%** | 14.67% |

All `91/91` successful predictive handovers were collision-free. The release
supervisor remained disabled.

## Decision

Predictive V1 is promoted because completion improved on both evaluation seeds
while pooled critical and static collision rates also decreased. The timeout
increase is the remaining cost and the next optimization target.

## Paths

```text
Model:
/home/human-robot-gym/models/icra_2024/
RHH_SIR_reactive_v1_predictive_final/run_0/model_final.zip

Training config:
/home/human-robot-gym/human_robot_gym/training/config_icra_2024/
environment_evaluation/training/RHH-SIR-reactive-v1-predictive.yaml

Evaluation config:
/home/human-robot-gym/human_robot_gym/training/config_icra_2024/
environment_evaluation/evaluation/RHH-SIR-reactive-v1-predictive.yaml
```
