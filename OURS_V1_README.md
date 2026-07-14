# Ours V1

**Reactive Human Modeling for Safe Human-Robot Collaboration Using
Reinforcement Learning**

V1 combines:

- a grip-gated reactive human
- geometry-aware physical handover capture
- DAgger behavior cloning
- SAC critic calibration
- SIR and RSI
- PFL safety shielding
- static/self-collision action filtering

## Capture-Only Reproducibility

Four independently initialized earlier capture-only policies were evaluated on
the same 100 scenarios:

| Seed | Success | Safe success | Critical mean |
|---:|---:|---:|---:|
| 0 | 54% | 53% | 0.27 |
| 1 | 56% | 56% | 0.25 |
| 2 | 52% | 52% | 0.26 |
| 3 | 57% | 56% | 0.22 |

Aggregate:

- success: **54.75% +/- 2.22%**
- safe success: **217/400**
- baseline success: **4.6%**
- critical collisions: **0.25 per episode**
- critical-collision p95: **1**
- critical-collision maximum: **2**
- timeout: **11.5% +/- 0.58%**

These runs support behavioral reproducibility but use the earlier capture event,
not the final release-confirmed task definition.

## Strict Handover Result

The promoted strict actor must release the object after capture and maintain
stable human ownership for five policy steps.

- strict success: **51/100**
- capture: **54/100**
- capture-to-release completion: **94.4%**
- safe strict success: **51/100**
- critical collisions: **0.29 per episode**
- timeout: **12%**
- release-supervisor use: **0 steps**

This result is actor-only. The optional release supervisor remained disabled.

## Guarded SAC Update

A proximity- and gripper-regularized SAC update was blended with the validated
strict actor. The promoted guarded actor achieved:

- strict success: **53/100**
- safe strict success: **53/100**
- critical events: **0.22 per episode**
- independent seed-200 result: **30/50**, equal to the strict fallback
- pooled result: **83/150 (55.33%)**

The original 51% strict checkpoint remains available as an immutable fallback.

## Predictive Governor Update

The guarded actor is now paired with a bounded predictive human-aware Cartesian
action governor:

- strict success: **57/100**
- independent seed-200 success: **34/50**
- pooled success: **91/150 (60.67%)**
- pooled critical events: **0.213 per episode**
- pooled static collisions: **0.107 per episode**
- all **91/91** successes collision-free

The governor scales only closing motion toward the nearest predicted human
joint. It preserves tangential motion and gripper commands and stops
intervening after 75 policy steps to prevent safety deadlocks.

## Delayed Cooperative Human Update

This V1 system keeps the predictive governor and adds delayed cooperative
human reach. The human only advances after a stable grasp has
remained in `REACH_OUT` for 100 policy steps without handover.

- strict success: **66/100**
- independent seed-200 success: **38/50**
- pooled success: **104/150 (69.33%)**
- pooled critical events: **0.213 per episode**
- pooled static collisions: **0.107 per episode**
- pooled timeout: **6.00%**
- all **104/104** successes collision-free

## Final Robust-Capture V1

The current promoted V1 keeps delayed cooperative reach and increases the
geometry-aware palm-to-object capture threshold from `0.06 m` to `0.12 m`.
Collision tracing showed that many remaining failures happened in `REACH_OUT`
while the robot already held the hammer, before the handover weld activated.

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

- strict success: **115/150 (76.67%)**
- safe success: **115/150 (76.67%)**
- critical events: **0.207 per episode**
- static collisions: **0.080 per episode**
- timeout: **0%**
- all **115/115** successful handovers were safe successes

## Reward

```text
r_env =
    -0.01
    + progress_reward
    + 2.0 * grasp_transition
    - 4.0 * object_drop
    + 20.0 * success
    - 1.0 * new_illegal_collision

r_total =
    0.05 * imitation_reward
    + 0.95 * r_env
    - 0.002 * shield_interventions
```

Progress is clipped to `0.05 m` per policy step:

```text
before grasp: 2.0 * delta(eef-to-object distance)
after grasp:  4.0 * delta(object-to-human-hand distance)
```

## Handover Condition

The human accepts the hammer when:

```text
phase == REACH_OUT
and robot still grips the object
and palm-to-nearest-object-surface gap <= 0.12 m
```

The normal MuJoCo human-object weld is activated. In strict V1, success then
requires the robot grasp to be absent and gripper aperture to remain at least
`0.02` for five consecutive policy steps.

## Final Training Recipe

- BC dataset: `reactive-human-handover-v1-bc-dagger-v3-deadlock`
- Dataset size: 406 episodes, 56,336 transitions
- BC epochs: 150
- Replay prefill: 5,000 deterministic actor steps
- Critic warm-up: 2,000 updates
- SAC calibration: 5,001 steps
- Actor: frozen during SAC calibration
- Learning rate: `1e-5`
- Entropy coefficient: `0.001`
- Batch size: 256
- Gamma: `0.995`
- Network: `[256, 256]`
- RSI: linear `0.20 -> 0.05` over the first 80%

The actor is DAgger-cloned and SAC-calibrated; it is not yet an end-to-end
SAC-improved actor.

Strict final actor:

- BC dataset: `reactive-human-handover-v1-bc-dagger-v4-strict`
- added strict DAgger episodes: 60
- BC epochs: 120
- final validation gripper MAE: `0.00844`

## Paths

Training config:

```text
/home/human-robot-gym/human_robot_gym/training/config_icra_2024/
environment_evaluation/training/RHH-SIR-reactive-v1.yaml
/home/human-robot-gym/human_robot_gym/training/config_icra_2024/
environment_evaluation/training/RHH-SIR-reactive-v1-strict.yaml
/home/human-robot-gym/human_robot_gym/training/config_icra_2024/
environment_evaluation/training/RHH-SIR-reactive-v1-final.yaml
/home/human-robot-gym/human_robot_gym/training/config_icra_2024/
environment_evaluation/evaluation/RHH-SIR-reactive-v1-final.yaml
```

Final models:

```text
/home/human-robot-gym/models/icra_2024/RHH_SIR_reactive_v1_final/
/home/human-robot-gym/models/icra_2024/RHH_SIR_reactive_v1_repair/run_3/
/home/human-robot-gym/models/icra_2024/
RHH_SIR_reactive_v1_strict_final/run_0/
/home/human-robot-gym/models/icra_2024/
RHH_SIR_reactive_v1_guarded_final/run_0/
/home/human-robot-gym/models/icra_2024/
RHH_SIR_reactive_v1_predictive_final/run_0/
/home/human-robot-gym/models/icra_2024/
RHH_SIR_reactive_v1_cooperative_final/run_0/
/home/human-robot-gym/models/icra_2024/
RHH_SIR_reactive_v1_final/run_0/
```

Final evaluation:

```text
/home/human-robot-gym/csv/evaluation/reactive_v1_reproducibility/
/home/human-robot-gym/csv/evaluation/reactive_v1_strict_final_100ep/
/home/human-robot-gym/csv/evaluation/reactive_v1_blend_alpha050_100ep/
/home/human-robot-gym/csv/evaluation/reactive_v1_governor_a022_b75_100ep/
/home/human-robot-gym/csv/evaluation/reactive_v1_coop_human_delayed_100ep/
/home/human-robot-gym/csv/evaluation/reactive_v1_coop_human_delayed_seed200_50ep/
/home/human-robot-gym/csv/evaluation/reactive_v1_capture_012_100ep/
/home/human-robot-gym/csv/evaluation/reactive_v1_capture_012_seed200_50ep/
```

Reports:

```text
/home/human-robot-gym/OURS_V1_FINAL_ROOT_CAUSE_REPORT.md
/home/human-robot-gym/V1_STRICT_FINAL_RESULTS.md
/home/human-robot-gym/V1_PRECAPTURE_IMPROVEMENT_REPORT.md
/home/human-robot-gym/V1_GUARDED_SAC_RESULTS.md
/home/human-robot-gym/V1_PREDICTIVE_GOVERNOR_RESULTS.md
/home/human-robot-gym/V1_COOPERATIVE_HUMAN_RESULTS.md
/home/human-robot-gym/V1_FINAL_CAPTURE_RESULTS.md
```
