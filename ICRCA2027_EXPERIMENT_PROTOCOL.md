# ICRCA 2027 — Isolated evaluation protocol (draft)

## Scope and preservation
This is an **evaluation-only** plan. Do not train a new policy, alter the original RC-STAR V1 environment, edit existing YAML files, or overwrite existing outputs. Work only on `icrca2027-velocity-adaptive-governor`. The new governor is not a certified safety mechanism. The existing PFL shield remains enabled.

## Existing implementation
- Baseline comparator: `human_robot_gym/wrappers/human_aware_action_governor.py`.
- Experimental comparator: `human_robot_gym/wrappers/velocity_adaptive_action_governor.py`.
- Opt-in integration: `human_robot_gym/training/icrca2027/governor_adapter.py`.
- Original evaluation configuration: `human_robot_gym/training/config_icra_2024/environment_evaluation/evaluation/RHH-SIR-reactive-v1-final.yaml`.
- Original V1 environment: `ReactiveHumanHandoverCartV1`, 10 Hz, 1000-step horizon, PFL shield, release-confirmed success.

## Comparators
1. Existing fixed-threshold predictive governor (0.22 m activation, 0.18 m hard distance, one-step horizon).
2. Existing fixed-threshold governor with a **separately tuned** prediction horizon; tune only on development seeds.
3. Proposed velocity-adaptive threshold, with unchanged prediction horizon and intervention budget.

Optional ablation: linear velocity-adaptive threshold. Keep all other factors matched.

## Evaluation isolation
1. Locate the *actual* frozen final model checkpoint and observation-normalization dataset. Do not infer they are present.
2. Verify the original evaluation loads and produces a valid observation and action shape; do not run before explicit approval.
3. Create a **fresh** environment per comparator, with the original wrapper ordering. Use the opt-in adapter only on the adaptive condition.
4. Keep PFL, V1 success definition, normalization, policy, action bounds, reward, 75-intervention budget, and episode horizon identical.
5. Pair initial seeds across conditions. Preserve independent development and final test seeds.
6. Write only to a newly created unique results directory; fail if the directory already exists.
7. Never use an evaluation helper that deletes or recreates an existing statistics directory.
8. Log configuration hashes, code commit, checkpoint hash, random seeds, episode metrics, and the adaptive threshold trajectory.

## Validation gates — not yet executed
- Static checks: formula units, bounds, budget exhaustion, gripper preservation, tangential motion preservation, and post-capture bypass.
- Integration checks: wrapper ordering, no double governor, observation compatibility, no change to model weights, and successful checkpoint load.
- Pilot: 10 paired episodes per governor, followed by frozen-parameter final evaluation.
- Only after pilot validation, run 100 paired episodes per governor for normal motion and 100 per governor for each pre-registered challenge scenario.

## Risks and limitations
The existing governor already incorporates closing displacement. Therefore, a higher adaptive threshold is not automatically novel or better than tuning the existing prediction horizon. The proposed robot speed is based on **commanded** Cartesian displacement, not measured executed velocity; deceleration is assumed, not identified. The adaptive threshold may cause excessive interventions, budget exhaustion, or timeouts. Report all of these alongside success and collision counts. No safety guarantee follows from this heuristic.

## Approval gate
No training, simulator evaluations, or changes to existing files are authorized by this protocol. Obtain explicit approval for experimental execution and verify artifact paths first.
