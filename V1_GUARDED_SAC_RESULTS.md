# V1 Guarded SAC Results

## Method

The validated strict actor was fine-tuned with a frozen reference-policy
constraint:

```text
actor_loss =
    sac_actor_loss
    + lambda * proximity_weight
      * weighted_smooth_l1(actor_action, reference_action)
```

The constraint used:

- 5,000 deterministic replay-prefill steps
- 500 critic warm-up updates
- 500 online SAC actor updates selected for promotion
- actor learning rate: `1e-6`
- critic learning rate: `1e-5`
- base anchor weight: `50`
- gripper anchor factor: `10`
- human-proximity multiplier: `4`
- human-distance threshold: `0.45 m`

The promoted actor is a 50/50 parameter interpolation between the validated
strict actor and the 500-step guarded SAC actor. Interpolation was selected
because the full update recovered some failures but also lost many baseline
successes.

## Strict Evaluation

### Primary benchmark: 100 episodes, evaluation seed 100

| Metric | Previous strict V1 | Guarded V1 |
|---|---:|---:|
| Strict success | 51% | **53%** |
| Safe success | 51% | **53%** |
| Capture | 54% | 54% |
| Critical events / episode | 0.29 | **0.22** |
| Static collisions / episode | 0.18 | **0.16** |
| Timeout | **12%** | 14% |
| Shield interventions / episode | 37.34 | 38.65 |

All `53/53` guarded-V1 successes had zero critical collisions.

### Independent gate: 50 episodes, evaluation seed 200

| Metric | Previous strict V1 | Guarded V1 |
|---|---:|---:|
| Strict success | 60% | 60% |
| Safe success | 60% | 60% |
| Critical events / episode | **0.22** | 0.28 |
| Static collisions / episode | 0.14 | **0.12** |
| Timeout | 8% | 8% |

### Pooled 150 episodes

| Metric | Previous strict V1 | Guarded V1 |
|---|---:|---:|
| Strict success | 81/150 (54.00%) | **83/150 (55.33%)** |
| Safe success | 81/150 (54.00%) | **83/150 (55.33%)** |
| Critical events / episode | 0.267 | **0.240** |
| Static collisions / episode | 0.167 | **0.147** |
| Timeout | **10.67%** | 12.00% |

## Decision

The guarded actor passed the configured promotion rule: higher primary and
pooled success with lower pooled critical-event rate. The original strict model
is retained as an immutable fallback.

The gain is modest and should not be described as statistically conclusive.
The remaining failure rate is still dominated by pre-capture collision and
timeout behavior.

## Paths

```text
Promoted guarded model:
/home/human-robot-gym/models/icra_2024/
RHH_SIR_reactive_v1_guarded_final/run_0/model_final.zip

Original strict fallback:
/home/human-robot-gym/models/icra_2024/
RHH_SIR_reactive_v1_strict_final/run_0/model_final.zip

Primary evaluation:
/home/human-robot-gym/csv/evaluation/
reactive_v1_blend_alpha050_100ep/

Independent evaluation:
/home/human-robot-gym/csv/evaluation/
reactive_v1_bc_sac_risk500_seed200_50ep/
```
