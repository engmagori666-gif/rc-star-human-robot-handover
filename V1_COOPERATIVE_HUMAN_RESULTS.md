# V1 Delayed Cooperative Human Results

## System

This promoted V1 system combines the guarded SAC actor, the predictive
Cartesian human-aware governor, and delayed cooperative human reach.

The cooperative reach activates only after the robot has held a grasp in
`REACH_OUT` for 100 policy steps without handover. The human then advances
smoothly in the table plane toward the grasped object:

```text
target distance: 0.10 m
maximum human root offset: 0.10 m
offset step: 0.002 m per policy step
delay: 100 grasped REACH_OUT policy steps
```

This targets the validated timeout mode: the robot holds the object at the edge
of its reachable workspace while the animated hand remains 17-24 cm away.

## Evaluation

| Metric | Predictive V1 | Cooperative V1 |
|---|---:|---:|
| Seed-100 strict success | 57/100 | **66/100** |
| Seed-100 safe success | 57/100 | **66/100** |
| Seed-100 timeout | 17% | **8%** |
| Seed-100 critical mean | 0.20 | 0.20 |
| Seed-100 static mean | 0.14 | 0.14 |
| Seed-200 strict success | 34/50 | **38/50** |
| Seed-200 safe success | 34/50 | **38/50** |
| Seed-200 timeout | 10% | **2%** |
| Seed-200 critical mean | 0.24 | 0.24 |
| Seed-200 static mean | 0.04 | 0.04 |

Pooled over 150 episodes:

| Metric | Predictive V1 | Cooperative V1 |
|---|---:|---:|
| Strict success | 91/150 (60.67%) | **104/150 (69.33%)** |
| Safe success | 91/150 (60.67%) | **104/150 (69.33%)** |
| Critical events / episode | 0.213 | 0.213 |
| Static collisions / episode | 0.107 | 0.107 |
| Timeout | 14.67% | **6.00%** |

All successful cooperative handovers were safe successes.

## Decision

Cooperative V1 is promoted. It improves completion on both evaluation seeds and
cuts timeout failures by more than half without increasing pooled critical or
static collision rates.

## Paths

```text
Model:
/home/human-robot-gym/models/icra_2024/
RHH_SIR_reactive_v1_cooperative_final/run_0/model_final.zip

Training config:
/home/human-robot-gym/human_robot_gym/training/config_icra_2024/
environment_evaluation/training/RHH-SIR-reactive-v1-cooperative.yaml

Evaluation config:
/home/human-robot-gym/human_robot_gym/training/config_icra_2024/
environment_evaluation/evaluation/RHH-SIR-reactive-v1-cooperative.yaml
```
