# V1 Pre-Capture Improvement Cycle

## Failure Classification

The 49 failures in the strict 100-episode final evaluation were:

| Failure mode | Episodes |
|---|---:|
| Critical human collision | 24 |
| Static collision | 13 |
| Timeout | 12 |
| Capture without release | 0 |

Release is not the remaining bottleneck. Successful episodes averaged 126.7
steps and had zero critical collisions. Failed episodes averaged 302.4 steps.

## Experiments

### Phase-aware expert

The phase-aware expert was connected to the live environment phase and changed
to hold during APPROACH. A paired test produced the same trajectories as the
generic expert because the human was normally already in REACH_OUT when grasp
occurred. This did not raise the teacher ceiling.

### Grasp-gated human approach

The human was held at its initial pose until robot grasp. This changed the target
distribution too strongly:

- current actor: 0/20 success
- repeated collision-prevention resampling and timeout behavior

The gate was rejected and disabled.

### V5 success-only DAgger

Collected 80 complete, collision-free, release-confirmed episodes from 157
attempts. Three independent BC candidates were trained for 150 epochs.

Online gates showed severe conservative collapse:

- candidate seed 0: 0/20
- candidate seed 1: 0/20
- candidate seed 2: 0/20

Success-only filtering removed recovery and boundary-state coverage. All three
candidates were rejected.

### Proximity-yielding human

The human paused only when close to an unprepared robot:

- paired current model success: 40% -> 40%
- critical events mean: 0.50 -> 0.45

The safety change was too small and did not improve completion, so it was not
promoted.

## Decision

The validated strict V1 model remains the final model:

```text
/home/human-robot-gym/models/icra_2024/
RHH_SIR_reactive_v1_strict_final/run_0/model_final.zip
```

Its validated 100-episode result remains:

- strict success: 51%
- safe success: 51%
- capture: 54%
- timeout: 12%
- critical events mean: 0.29

The next credible route is not another BC-only dataset. It requires either
closed-loop RL fine-tuning with collision-risk state features or a predictive
human-aware shield that reasons about the human moving into the robot.
