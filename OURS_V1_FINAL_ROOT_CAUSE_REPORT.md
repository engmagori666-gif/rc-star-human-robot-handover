# Final Root-Cause Analysis: Safe Reactive Robot-to-Human Handover V1

## Executive Diagnosis

The historical policies failed primarily because the learning objective and task
completion event were badly conditioned for exploration:

1. Before grasping, the sparse environment reward was `-1` per step, while the
   final task reward was only `+1`.
2. SIR preserved 75% of that negative environment reward, so even perfect
   imitation could still produce negative pre-grasp reward.
3. Success depended on a narrow moving-hand contact event and, in the baseline,
   the completion of a retreat animation after the physical transfer.
4. The reactive target moved as a function of the robot's own progress, creating
   a coupled, non-stationary control problem.
5. SAC used a high learning rate, high initial entropy, burst updates, and a
   small network for a long-horizon manipulation task.
6. The collision-prevention wrapper replaced unsafe actions with exact zero
   actions. Failed V1 episodes averaged about 300 replacements, creating a
   recurrent "stop and time out" state.
7. A core `IntFlag` test incorrectly classified the NULL collision flag as part
   of the illegal collision mask. This blocked correct use of
   `done_at_collision` and could apply parent collision logic on clean steps.

The PFL shield itself was not the main success bottleneck. In final V1,
successful episodes had more shield interventions than failed episodes, proving
that intervention and task completion are compatible.

## Strict Release-Confirmed V1 Update

The original V1 success event ended at physical capture by the human weld. A
strict completion variant now requires all of the following:

```text
human-object weld active
and robot no longer gripping the object
and gripper aperture >= 0.02
for 5 consecutive policy steps
```

Capture is an intermediate `+5` milestone. The terminal `+20` reward is issued
only after release confirmation. Intended post-capture release is excluded from
the object-drop penalty.

The strict actor was trained with 60 additional DAgger episodes using
`release_when_delivered=true`. The release supervisor was disabled during the
final evaluation.

| Strict V1 metric | 100-episode result |
|---|---:|
| Capture | 54% |
| Confirmed release success | **51%** |
| Capture-to-release completion | **94.4%** |
| Safe success | **51%** |
| Critical collisions mean | **0.29** |
| Timeout | **12%** |
| Shield interventions mean | **37.34** |
| Release-supervisor steps | **0** |

All `51/51` strict successes had zero critical collisions. The 95% bootstrap
interval recorded by the evaluator for success was `40%` to `60%`. Relative to
the reproducible baseline success of `4.6%`, strict V1 is approximately `11.1x`
higher while enforcing a stronger task definition.

## Reproducible Evaluation Results

The supplied raw CSV directories contain:

| Method | Episodes | Success | Timeout | Critical collisions mean | Critical p95 | Shield mean |
|---|---:|---:|---:|---:|---:|---:|
| Baseline RHH-SIR | 500 | 4.60% | 95.40% | 6.038 | 30.05 | 21.80 |
| Reactive-only V2 | 800 | 0.25% | 99.75% | 14.979 | 55.10 | 11.42 |
| STAR V3 | 500 | 0.00% | 100.00% | 1.360 | 0.00 | 2.94 |
| Earlier capture-only V1 | 100 | **65.00%** | **11.00%** | **0.220** | **1.00** | 32.67 |

Additional safety results for the earlier capture-only checkpoint:

- Critical-collision median: `0`
- Critical-collision maximum: `3`
- Zero-critical-collision episodes: `83%`
- Capture-only successes with a critical collision: `0/65`
- Mean episode length: `195.95`, versus `976.84` for baseline
- Mean return: `+14.50`, versus `-810.40` for baseline

The originally stated V2 and V3 rates of approximately 4.1% and 3.9% are not in
the specified raw CSV directories. Those files contain 2/800 and 0/500
successes. The report uses the reproducible raw values.

### Capture-only training-seed reproducibility

Four independently initialized capture-only policies were evaluated on the same
100 environment episodes. These results are supporting evidence, not the final
strict release-confirmed metric:

| Training seed | Success | Timeout | Critical mean | Critical p95 |
|---:|---:|---:|---:|---:|
| 0 | 54% | 12% | 0.27 | 1 |
| 1 | 56% | 11% | 0.25 | 1 |
| 2 | 52% | 12% | 0.26 | 1 |
| 3 | 57% | 11% | 0.22 | 1 |

Across training seeds:

- Success: **54.75% +/- 2.22%**
- Worst seed: **52%**
- Pooled result: **219/400 successful handovers**
- Collision-free successful handovers: **217/400 episodes**
- Successful handovers with any critical collision: **2/219**
- Critical collisions: **0.25 +/- 0.022 per episode**
- Critical-collision p95: **1**
- Critical-collision maximum: **2**
- Zero-critical-collision episodes: **77.0%**
- Timeout rate: **11.5% +/- 0.58%**

Two capture episodes recorded one critical contact on the same simulation step
as the transfer. The other 217 successful handovers had zero critical
collisions. Reporting distinguishes `task_success` from
`safe_task_success = task_success and n_collisions_critical == 0`.

All four policies substantially exceed the 4.6% baseline. Targeted DAgger
corrections eliminated the former seed-3 deadlock: its success increased from
39% to 57%, timeout fell from 40% to 11%, and mean collision-action
replacements fell from 302.9 to 0.

## 1. Reward Analysis

### Historical baseline, Reactive V2, and STAR V3

With `reward_shaping: false`, `collision_reward: 0`,
`object_gripped_reward: -0.25`, and `task_reward: 1`, the environment reward was:

```text
r_env(t) =
    +1.00  if task_phase == COMPLETE
     0.00  if task_phase == RETREAT
    -0.25  if the robot grips the object
    -1.00  otherwise
```

The SIR wrapper computes:

```text
r_SIR(t) = alpha * r_imitation(t) + (1 - alpha) * r_env(t)
```

For the historical configs, `alpha = 0.25`:

```text
r_SIR(t) = 0.25 * r_imitation(t) + 0.75 * r_env(t)
```

Before grasping:

- With zero imitation reward: `r_SIR = -0.75`.
- Even with perfect imitation reward of 1: `r_SIR = -0.50`.
- A successful terminal environment reward contributes only `+0.75`.

STAR V3 then subtracts:

```text
0.01 * delta_failsafe_interventions
```

Therefore, the dominant failure signal was the per-step task reward, not the
shield penalty. The policy could accumulate hundreds of negative reward units
before receiving a terminal reward smaller than one.

Relevant code:

- `robot_human_handover_cartesian_env.py:516-554`
- `state_based_expert_imitation_reward_wrapper.py:210-233`
- `RHH-SIR-reactive-v3.yaml:73-115`

### V1 environment reward

Define:

- `d_eo(t)`: end-effector to object distance
- `d_oh(t)`: object to current human-hand target distance
- `C_t`: one if a new illegal collision event occurred
- `G_t`: one on the transition from not-gripped to gripped
- `D_t`: one on an object drop before success
- `S_t`: one on handover success

The V1 progress term is:

```text
P_t =
    2 * clip(d_eo(t-1) - d_eo(t), -0.05, 0.05)  before grasp
    4 * clip(d_oh(t-1) - d_oh(t), -0.05, 0.05)  after grasp
```

The environment reward is:

```text
r_env(t) =
    -0.01
    + P_t
    + 2 * G_t
    - 4 * D_t
    + 20 * S_t
    - 1 * C_t
```

On the regional/contact capture step, reward is explicitly replaced by:

```text
r_env(capture) = 20 - 0.01 - 1 * C_t
```

The complete wrapped reward is:

```text
r_total(t) =
    0.05 * r_imitation(t)
    + 0.95 * r_env(t)
    - 0.002 * delta_failsafe_interventions(t)
```

The penalties no longer dominate normal successful trajectories:

- A successful transfer gives approximately `+18.99` after SIR weighting.
- The mean shield cost on a successful final episode is only
  `38.29 * 0.002 = 0.0766`.
- Immediate termination on illegal collision prevents long collision-penalty
  tails.

Relevant code:

- `ours_v1_handover_env.py:181-279`
- `failsafe_penalty_wrapper.py:27-34`
- `RHH-SIR-reactive-v1.yaml:79-119`

### Collision-mask defect

The original code used:

```python
collision_type in (
    COLLISION_TYPE.STATIC
    | COLLISION_TYPE.ROBOT
    | COLLISION_TYPE.HUMAN_CRIT
)
```

For `IntFlag`, the zero/NULL flag can be treated as contained in a combined mask.
That caused collision-free states to be classified as illegal in
`done_at_collision` and parent collision-reward paths.

It is now:

```python
illegal_types = (
    COLLISION_TYPE.STATIC
    | COLLISION_TYPE.ROBOT
    | COLLISION_TYPE.HUMAN_CRIT
)
return bool(collision_type & illegal_types)
```

Relevant code: `human_env.py:860-882`.

## 2. Success Condition Analysis

### Historical baseline

Baseline success was:

```text
task_phase == COMPLETE
```

The required sequence was:

1. Human animation enters `REACH_OUT`.
2. Robot grasps the object.
3. The object collision geometry contacts the correct palm geometry.
4. The human-object weld is activated.
5. Task enters `RETREAT`.
6. The retreat animation reaches its final frame.
7. Task enters `COMPLETE`.

`goal_dist: 0.06` was not the baseline success threshold. Success depended on
mesh contact and animation state. This was excessively restrictive because the
robot was penalized until completing a long sequence, and final success was
delayed by an animation the robot did not control.

### V1 success

V1 accepts either exact palm contact or a geometry-aware capture:

```text
task_phase == REACH_OUT
and robot is still gripping the object
and palm-to-nearest-hammer-surface gap <= 0.06 m
```

On capture:

1. The nearest contact point is computed on the oriented hammer box/cylinder.
2. `_human_pickup_object()` activates the normal human-object equality weld.
3. Phase changes to `RETREAT`.
4. `goal_reached` and `n_goal_reached` are set.
5. `done_at_success: true` ends the episode.

This credits the physical transfer rather than the later retreat animation.

This was the limitation of the earlier capture-only checkpoint: success did not
require the robot gripper to open after the human weld activated. The final
strict variant resolves it by requiring:

```text
human weld active
and robot gripper open
and object no longer robot-gripped
for N consecutive steps
```

Relevant code:

- `ours_v1_handover_env.py:66-114`
- `ours_v1_handover_env.py:116-179`
- `robot_human_handover_cartesian_env.py:731-757`

## 3. Reactive Human Analysis

The reactive human enters `REACH_OUT` after the first animation keyframe.
During that phase:

```text
rho(t) = 0                                             if not gripping
rho(t) = clip(1 - d_oh(t) / 0.5, 0, 1)                if gripping
human_speed(t) = 0.001 + 2.0 * rho(t)
```

Consequences:

- Before grasping, the arm is nearly frozen but still moves at 0.001
  frame/control step.
- After grasping, the human moves faster as the object approaches.
- The target is the live mocap palm position and changes during every simulation
  update.
- Robot motion changes human speed, while human motion changes the robot target.

This is a coupled non-stationary system from the policy's perspective. It was
too difficult to solve from sparse reward alone. However, it is learnable when
the policy is trained on exact-environment corrective trajectories. DAgger was
essential: clean offline BC had low validation error but failed online because
small action errors moved the policy outside the demonstration distribution.

Relevant code:

- `reactive_handover_env.py:245-314`
- `robot_human_handover_cartesian_env.py:453-455`
- `RHH-SIR-reactive-v1.yaml:75-77`

## 4. Safety Shield Analysis

The final safety stack contains:

- SaRA PFL reachable-set shield
- Low-level control sample time: `0.004 s`
- PFL safety-rated speed: `0.05`
- Maximum stopping time: `0.2 s`
- Human-contact critical classification threshold: `safe_vel = 0.001`
- Static/self-collision action checking
- Immediate episode termination on illegal collision

The PFL intervention condition is not a single distance threshold. The shield
checks planned robot reachable occupancy against predicted human occupancy. Its
Schunk trajectory limits are in
`trajectory_parameters_schunk.yaml`.

### Was the shield overly conservative?

No, not as the primary cause:

- Final successful episodes: 38.29 PFL interventions on average.
- Final failed episodes: 22.23 interventions on average.
- Successful episodes therefore tolerate more intervention than failed ones.
- The average successful shield reward cost is only 0.0766.

The more damaging mechanism was the collision-prevention action replacement.
Originally `replace_type: 0` replaced a predicted static/self-collision action
with exact zero. In the earlier 100-episode V1 evaluation:

- Successful episodes: `0.015` action replacements on average.
- Failed episodes: `299.7` replacements on average.

That was a clear deadlock signature. V1 now uses deterministic scaled fallback:

```text
try 0.75 * action
then 0.50 * action
then 0.25 * action
otherwise zero
```

This preserves direction and progress while remaining within the same collision
checker. `done_at_collision: true`, enabled after fixing the bitmask defect,
prevents unsafe retry loops.

Relevant code:

- `collision_prevention_wrapper.py`
- `human_env.py:490-523`
- `failsafe_controller.py:175-191, 300-381`
- `trajectory_parameters_schunk.yaml`

## 5. RSI Curriculum Analysis

### STAR V3

STAR V3 starts at `rsi_prob = 0.9` over 3,000,001 steps:

| Training fraction | Approximate steps | RSI probability |
|---|---:|---:|
| 0.00-0.33 | 0-990,000 | 0.9 |
| 0.33-0.66 | 990,001-1,980,000 | 0.5 |
| 0.66-0.90 | 1,980,001-2,700,000 | 0.1 |
| 0.90-1.00 | 2,700,001-3,000,001 | 0.0 |

RSI assistance therefore becomes weak after 66% and disappears for the final
10%. Because success had not emerged before that point, the late training
distribution was dominated by full-horizon failures.

### V1

The completed V1 run used:

```text
initial RSI probability = 0.20
final RSI probability   = 0.05
schedule                = linear
decay end               = 80% of training
actual training steps   = 5,001
actual decay duration   = about 4,001 steps
```

The maintained V1 config is set for 500,001 steps, which would decay over about
400,001 steps. RSI does not disappear because the floor is 0.05.

For V1, RSI is no longer the main source of success. The actor was initialized
from exact-environment DAgger data, and RSI primarily supports critic
calibration.

Relevant code:

- `rsi_curriculum_callback.py:5-40`
- `RHH-SIR-reactive-v3.yaml:99, 167, 219`
- completed `run_0/config.yaml:247-251`

## 6. SAC Hyperparameter Analysis

| Parameter | Historical V3 | Completed V1 |
|---|---:|---:|
| Learning rate | `5e-4` | `1e-5` |
| Batch size | 128 | 256 |
| Replay buffer | 1,000,000 | 1,000,000 |
| Gamma | 0.99 | 0.995 |
| Entropy coefficient | `auto_0.2` | `0.001` |
| Train frequency | every 100 steps | every step |
| Gradient steps | `-1` (100-update bursts) | 1 |
| Network | `[64,64,64]` | `[256,256]` |
| Configured V3 steps | 3,000,001 | - |
| Completed V1 steps | - | 5,001 |

Common manipulation SAC configurations are closer to `3e-4`, batch 256,
one update per environment step, gamma 0.99-0.995, automatic entropy, and
`[256,256]` networks.

Historical V3 was aggressive: `5e-4`, forced initial entropy 0.2, and burst
updates against mostly failed trajectories. V1 deliberately uses much smaller
updates and entropy to preserve a working manipulation policy.

Important limitation: the final V1 actor is frozen during the completed SAC
run. SAC calibrated the critics and replay distribution, but it did not improve
the actor online. The actor is the DAgger-cloned policy. It is therefore accurate
to describe this result as:

```text
DAgger behavior-cloned actor + SAC critic calibration + RSI/SIR + shielding
```

It should not yet be claimed as unconstrained end-to-end SAC policy learning.
A future version should use a decaying BC regularizer on the actor rather than
fully freezing it.

## 7. Training Diagnostics

Historical evaluation is a strong collapse signal:

- Baseline: 95.4% timeouts.
- Reactive V2: 99.75% timeouts.
- STAR V3: 100% timeouts.
- STAR reduced shield activity to 2.94 per episode but still never succeeded.

This is conservative policy collapse: the policy reduces motion and
intervention without learning the handover.

V1 training diagnostics:

- BC validation MSE: `0.7201 -> 0.00635`.
- BC validation MAE: `0.7819 -> 0.02658`.
- Warm-start replay prefill: `28/30 = 93.33%` success.
- Critic loss: `4.3487 -> 1.7730` over 2,000 warm-up updates.
- Pure offline BC initially failed online despite low validation error.
- First DAgger policy reached 30% online success.
- Second DAgger policy reached 75% in a 20-episode pilot.
- Naive SAC actor updates erased the working policy, so the validated actor was
  frozen for the final run.
- Earlier capture-only checkpoint: `65/100` success.
- Four paired capture-only training-seed evaluations: `54%`, `56%`, `52%`, and
  `57%`.
- Capture-only training-seed mean: `54.75% +/- 2.22%`.
- Final strict actor-only evaluation: `51/100` release-confirmed successes.
- Final corrective dataset: 406 episodes and 56,336 transitions.

Reward has moved from a timeout-dominated negative regime to positive completed
returns. Mean final V1 return is `+14.50`.

## 8. Ranked Top Five Causes and Exact Fixes

### 1. Sparse reward made task progress economically irrational

Evidence: pre-grasp reward remained `-0.50` even under perfect imitation, while
terminal environment reward was only `+1`.

Impact: conservative behavior, poor long-horizon credit assignment, and timeout
collapse.

Implemented fix: `ours_v1_handover_env.py` adds clipped approach/transport
progress, grasp bonus, drop penalty, step cost, and `+20` success.

### 2. Success was delayed and geometrically brittle

Evidence: baseline required exact palm contact and final retreat animation.

Impact: almost no terminal examples for SAC, even when the robot approached the
human correctly.

Implemented fix: accept the object during `REACH_OUT` when the robot still grips
it and palm-to-object-surface gap is at most 0.06 m; activate the physical weld
and terminate at transfer.

### 3. Reactive dynamics created demonstration/policy distribution shift

Evidence: the target moves with the human, and human speed depends on robot
transport progress. Offline BC had low validation error but zero useful online
robustness before corrective data.

Impact: small errors compound until the policy misses grasp or handover.

Implemented fix: two DAgger correction rounds in the exact reactive, shielded
environment; 40,350 transitions in the final BC dataset.

### 4. SAC updates were too destructive for the learned manipulation behavior

Evidence: historical `5e-4`, entropy 0.2, 100-step update bursts, and small
network; naive V1 SAC updates collapsed a successful cloned actor.

Impact: useful grasp/transport behavior was overwritten by critics trained on
mostly failed replay.

Implemented fix: 5,000-step successful replay prefill, 2,000 critic-only warmup,
`1e-5` learning rate, entropy 0.001, `[256,256]`, and frozen actor.

Next exact change: implement a decaying actor BC penalty and unfreeze only after
critic validation:

```text
actor_loss = sac_actor_loss
             + lambda_bc(t) * MSE(actor_action, replay_expert_action)
```

### 5. Safety action handling created deadlocks, and collision logic had a mask bug

Evidence: failed episodes averaged 299.7 zero replacements; NULL collision was
misclassified by `IntFlag` membership.

Impact: the robot stopped moving and timed out, while immediate safety abort
could not be enabled correctly.

Implemented fixes:

- Replace unsafe actions by deterministic magnitude reduction instead of
  immediate zero.
- Use bitwise intersection for illegal collision classification.
- Enable `done_at_collision: true`.

For the earlier capture-only checkpoint, success remained 65%, critical
collisions fell to mean 0.22, p95 1, maximum 3, and all 65 capture events had
zero critical collisions. Under the final strict task definition, the
actor-only policy completed 51/100 release-confirmed handovers, and all 51 were
collision-free.

## Final Assessment

The original near-zero success was not one isolated hyperparameter problem. It
was the combination of a hostile reward scale, brittle success event, moving
target distribution shift, destructive off-policy updates, and zero-action
safety deadlocks.

V1 now demonstrates that the task is feasible:

- reactive human: active and coupled to robot progress
- PFL safety shield: active
- collision action filtering: active
- illegal-collision abort: active
- physical human-object weld: active
- final strict release-confirmed success: 51%
- final strict safe success: 51%
- capture-to-release completion after capture: 94.4%
- final strict critical collisions: 0.29 per episode
- release-supervisor steps in final evaluation: 0

The earlier capture-only behavior was reproduced across four training seeds,
and the stronger final strict actor does not collapse to zero. The remaining
scientific work is to replace the frozen actor with BC-regularized SAC
fine-tuning while preserving the validated safety and success gates.

## Post-Validation Improvement Cycle

The 49 failures in the final strict 100-episode evaluation were classified as:

| Failure mode | Episodes |
|---|---:|
| Critical human collision | 24 |
| Static collision | 13 |
| Timeout | 12 |
| Capture without confirmed release | 0 |

Release is therefore not the current bottleneck. The dominant remaining problem
is reaching and transporting safely around a reactive human.

Four targeted changes were tested after the strict result:

1. A phase-aware expert produced the same paired trajectories as the generic
   expert and did not raise the teacher ceiling.
2. Holding the human still until grasp changed the state distribution too
   strongly and reduced the current actor to 0/20 success.
3. Three success-only DAgger/BC candidates collapsed to 0/20 online under the
   original strict environment and were rejected.
4. A proximity-yielding human reduced critical events from 0.50 to 0.45 in a
   paired 20-episode check, but success stayed at 40%.

At that stage no candidate passed the promotion gate, so the validated 51%
strict checkpoint remained the final model. The next credible improvement was
closed-loop RL with a decaying BC constraint or a predictive human-aware
shield; another success-only BC dataset was unlikely to solve the remaining
failures.

## Guarded SAC Follow-Up

A later guarded SAC cycle implemented the recommended closed-loop update. The
actor loss was constrained to a frozen copy of the validated actor, with
stronger anchoring on the gripper dimension and when the robot was close to the
human. Unconstrained or weakly constrained updates still degraded after 1,000 to
3,000 steps.

The selected policy used 500 guarded SAC updates followed by a 50/50 actor
parameter interpolation with the validated strict actor.

| Evaluation | Strict fallback | Guarded V1 |
|---|---:|---:|
| Seed 100, 100 episodes | 51% | **53%** |
| Seed 200, 50 episodes | 60% | 60% |
| Pooled success | 81/150 (54.00%) | **83/150 (55.33%)** |
| Pooled critical events / episode | 0.267 | **0.240** |

The guarded actor is promoted to
`RHH_SIR_reactive_v1_guarded_final/run_0/model_final.zip`. The original strict
checkpoint remains unchanged as a fallback. The improvement is real under the
promotion rule but small enough that it should not be presented as statistically
conclusive.

## Predictive Human-Aware Governor Follow-Up

The guarded actor's remaining 47 primary-benchmark failures were:

| Failure mode | Episodes |
|---|---:|
| Critical human collision | 18 |
| Static or self collision | 15 |
| Timeout | 14 |

Only one failed episode reached capture, confirming that the remaining problem
was pre-capture motion rather than release.

An optional Cartesian governor was added outside the IK wrapper. It predicts
next-step separation using the nearest human-joint displacement and scales only
the action component moving directly toward the human. Tangential, retreating,
and gripper commands are unchanged.

Unlimited intervention improved safety but raised timeout to 24%. Successful
episodes averaged 54 governor interventions, while governor-induced deadlocks
were typically governed 875 to 944 times. Capping interventions at 75 removed
that deadlock mode.

| Evaluation | Guarded V1 | Predictive V1 |
|---|---:|---:|
| Seed 100, 100 episodes | 53% | **57%** |
| Seed 200, 50 episodes | 60% | **68%** |
| Pooled success | 83/150 (55.33%) | **91/150 (60.67%)** |
| Pooled critical events / episode | 0.240 | **0.213** |
| Pooled static collisions / episode | 0.147 | **0.107** |
| Pooled timeout | **12.00%** | 14.67% |

Predictive V1 is promoted because it improves completion on both seeds and
reduces pooled collision rates. The remaining engineering target is recovering
the additional 2.67 percentage points of timeout without removing the early
collision protection.
