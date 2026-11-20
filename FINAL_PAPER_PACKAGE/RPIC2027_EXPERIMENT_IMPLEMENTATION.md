# RPIC 2027 Zero-Shot Robustness Experiment

## Scope

This implementation adds an evaluation-only RC-STAR environment and runner. It does not alter the trained policy, policy observations, actions, rewards, capture logic, release-confirmed success, predictive governor, PFL shield, or original baseline branch.

## Conditions

All conditions use `ReactiveHumanHandoverCartV1RPIC`, which subclasses the final RC-STAR environment.

- `normal`: unchanged RC-STAR human animation.
- `delayed_response`: holds the current `REACH_OUT` animation progress for 8 policy steps (0.8 s at 10 Hz).
- `reach_reversal`: decreases `REACH_OUT` progress by 0.5 frame per low-level animation update for 6 policy steps (0.6 s at 10 Hz), clamped to the start of `REACH_OUT`.

The disturbance becomes eligible only during `REACH_OUT`, after two consecutive policy steps with a valid robot grasp, and before RC-STAR object capture. It stops immediately if phase, grasp, or pre-capture eligibility is lost. A private episode RNG deterministically samples a trigger delay from 2-5 eligible policy steps. Episode seed is `disturbance_seed + 1009 * episode_index`.

The private disturbance RNG does not consume NumPy's global environment RNG. Each condition starts from the same evaluation seed and is run in a fresh environment. The runner hashes the reset observation and full MuJoCo `qpos` and `qvel` arrays, then requires identical fingerprints for each paired episode across all conditions.

## Deadlock definition

While the robot holds the object during pre-capture `REACH_OUT`, a stall counter increases whenever object-to-target distance improves by no more than 1 mm in a policy step. It resets on greater progress. An episode is marked with the **pre-capture progress-stall diagnostic** when the counter reaches 50 consecutive policy steps. The legacy `rpic_deadlock` column is retained as an alias. This diagnostic does not terminate or otherwise change the episode.

## Shield-path validation

The evaluation subclass instruments the existing `_set_human_measurement` call without changing its forwarding behavior. It records:

- calls made while a disturbance is active;
- changing measurements during the disturbance;
- any mismatch between the forwarded and current simulation measurement;
- non-increasing measurement timestamps.

The focused test verifies that an active disturbance measurement is forwarded unchanged through the inherited method that calls each PFL robot controller. Pilot CSVs record successful PFL forwards, stale-measurement mismatches, and timestamp regressions. The runner rejects a triggered event with no PFL forward or any stale/timestamp violation.

## Provenance

`RPIC2027_PROTOCOL.yaml` records file sizes and SHA-256 hashes for:

- `model_final.zip`;
- the checkpoint's saved training configuration;
- observation-normalization statistics;
- dataset episode statistics;
- the final evaluation configuration.

The runner verifies every artifact before constructing an environment. It also checks checkpoint/environment observation and action shapes before evaluation. If the checkpoint has the one-element SIR time feature used by the repository's original evaluator, the runner reproduces that established evaluation behavior using the recorded dataset mean episode length. No physical observation channel is added or removed.

The saved training configuration reflects the original training task. The final evaluation configuration contains the later RC-STAR evaluation-time task settings, predictive governor, and release-confirmed protocol. Both are retained and hashed rather than treating them as identical.

## Outputs

The runner writes:

- one episode-level CSV per condition;
- `episodes.csv` containing all conditions and paired episode indices;
- `summary.csv` with descriptive pilot totals/rates;
- `manifest.json` containing protocol, spaces, seed, and verified artifact hashes.

The summary is descriptive only. Formal analysis should use the episode-level CSV and a prespecified confidence-interval method.

## Commands

Verify artifacts only:

```bash
python -m human_robot_gym.training.evaluate_rpic2027_robustness --verify-only
```

Run focused tests:

```bash
python -m unittest discover -v -s tests -p 'test_rpic_human_motion_disturbance.py'
```

Run the permitted 10-episode-per-condition pilot:

```bash
python -m human_robot_gym.training.evaluate_rpic2027_robustness \
  --episodes 10 \
  --output-dir csv/evaluation/rpic2027_robustness/pilot_seed_2701
```

If the installed PyTorch build does not support the host GPU, hide CUDA and run
the same evaluation on CPU. This changes only the inference device:

```bash
CUDA_VISIBLE_DEVICES= python -m human_robot_gym.training.evaluate_rpic2027_robustness \
  --episodes 10 \
  --output-dir csv/evaluation/rpic2027_robustness/pilot_seed_2701_cpu
```

Run the final paired 100-episode-per-condition evaluation later:

```bash
python -m human_robot_gym.training.evaluate_rpic2027_robustness \
  --episodes 100 \
  --allow-full \
  --output-dir csv/evaluation/rpic2027_robustness/final_seed_2701
```

No training command is involved.
