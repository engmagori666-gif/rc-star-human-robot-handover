# V1 Final Robust-Capture Results

## System

Final V1 combines:

- guarded SAC actor
- predictive Cartesian human-aware governor
- delayed cooperative human reach
- strict robot release after human capture
- robust geometry-aware capture threshold

The only promoted behavioral change after cooperative V1 is:

```text
v1_capture_surface_dist: 0.06 m -> 0.12 m
```

## Root Cause Addressed

The remaining cooperative-V1 failures were not mainly release failures or
timeouts. Collision tracing showed that most remaining failed collision
episodes entered `REACH_OUT` with the robot already gripping the hammer, but
the handover weld had not activated yet. The old `0.06 m` capture threshold was
too brittle for a moving palm and elongated hammer, so some near-handover
states terminated as collision failures before they could become handovers.

An object-aware governor pilot was rejected because it was too conservative:
success dropped to 8/20 and timeout increased to 35%.

## Evaluation

| Metric | Cooperative V1 | Final V1 |
|---|---:|---:|
| Seed-100 strict success | 66/100 | **76/100** |
| Seed-100 safe success | 66/100 | **76/100** |
| Seed-100 timeout | 8% | **0%** |
| Seed-100 critical mean | 0.20 | **0.19** |
| Seed-100 static mean | 0.14 | **0.10** |
| Seed-200 strict success | 38/50 | **39/50** |
| Seed-200 safe success | 38/50 | **39/50** |
| Seed-200 timeout | 2% | **0%** |
| Seed-200 critical mean | 0.24 | 0.24 |
| Seed-200 static mean | 0.04 | 0.04 |

Pooled over 150 episodes:

| Metric | Cooperative V1 | Final V1 |
|---|---:|---:|
| Strict success | 104/150 (69.33%) | **115/150 (76.67%)** |
| Safe success | 104/150 (69.33%) | **115/150 (76.67%)** |
| Critical events / episode | 0.213 | **0.207** |
| Static collisions / episode | 0.107 | **0.080** |
| Timeout | 6.00% | **0.00%** |

All 115 successful final-V1 handovers were safe successes.

## Decision

Final robust-capture V1 is promoted. It improves success on both evaluation
seeds, eliminates timeout failures in the gated runs, and does not increase
critical or static collision rates.

## Paths

```text
Model:
/home/human-robot-gym/models/icra_2024/
RHH_SIR_reactive_v1_final/run_0/model_final.zip

Training config:
/home/human-robot-gym/human_robot_gym/training/config_icra_2024/
environment_evaluation/training/RHH-SIR-reactive-v1-final.yaml

Evaluation config:
/home/human-robot-gym/human_robot_gym/training/config_icra_2024/
environment_evaluation/evaluation/RHH-SIR-reactive-v1-final.yaml

Evaluation CSVs:
/home/human-robot-gym/csv/evaluation/reactive_v1_capture_012_100ep/
/home/human-robot-gym/csv/evaluation/reactive_v1_capture_012_seed200_50ep/
```
