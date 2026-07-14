#!/bin/bash
# probe_v2.sh — V2 smoke test before full 3M × 8-seed experiment.
#
# Trains both conditions for 100k steps (1 seed), evaluates 4 checkpoints
# (25k / 50k / 75k / 100k steps, 50 episodes each), and runs a grip-guard
# sanity check on the random-init policy (model_0, 20 episodes).
#
# Usage (from project root):
#   bash human_robot_gym/training/probe_v2.sh
#
# Pass criteria before launching full training:
#   [1] No crash in training steps — both envs initialize and step without error
#   [2] success_mean at step 25k ≈ 0 for scripted (random policy learns slowly)
#   [3] success_mean at step 100k > 0 for at least one condition (learning signal exists)
#   [4] grip_guard check: success_mean at step 0 ≈ 0 (no spurious RETREAT triggers)
#   [5] n_collisions_critical_mean < 5 per episode on average
#
# Output files:
#   Training logs  — runs/icra_2024/RHH_{scripted,reactive}_v7/run_0/
#   Checkpoints    — models/icra_2024/RHH_{scripted,reactive}_v7/run_0/model_{25,50,75,100}_000.zip
#   Eval results   — csv/evaluation/probe_v2/{scripted,reactive}/stats.csv
#   Grip guard     — csv/evaluation/probe_v2/{scripted,reactive}_step0/stats.csv
#
# Runtime estimate: ~30–60 min per condition on 8 CPU cores (CUDA disabled).

set -euo pipefail

# RTX 5070 Ti driver incompatibility — force CPU training
export CUDA_VISIBLE_DEVICES=""

# ─── Probe parameters ─────────────────────────────────────────────────────────
SEED=0
N_STEPS=100000          # Training steps per condition
SAVE_FREQ=25000         # Checkpoint every 25k steps → 4 checkpoints
N_ENVS=8                # Parallel environments (matches full training)
N_EVAL_EPISODES=50      # Episodes per checkpoint evaluation
LOG_INTERVAL=1000       # Tensorboard logging frequency (steps)

# Derived paths (must match wandb_run.{project,group,name} in the v7 configs)
PROJECT=icra_2024
SCRIPTED_GROUP=RHH_scripted_v7
REACTIVE_GROUP=RHH_reactive_v7
SCRIPTED_RUN_ID="${PROJECT}/${SCRIPTED_GROUP}/run_${SEED}"
REACTIVE_RUN_ID="${PROJECT}/${REACTIVE_GROUP}/run_${SEED}"

print_step() { printf '\n\033[0;32m[probe_v2] %s\033[0m\n' "$1"; }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Train ScriptedHandoverCartV2
# ─────────────────────────────────────────────────────────────────────────────
print_step "1/6  Training ScriptedHandoverCartV2 — ${N_STEPS} steps, seed ${SEED}"
(
    set -x
    python human_robot_gym/training/train_SB3.py --multirun \
        -cp config_icra_2024/environment_evaluation/training \
        -cn RHH-scripted-v7 \
        hydra/launcher=ray \
        run.type=tensorboard \
        run.n_steps=${N_STEPS} \
        "run.seed=${SEED}" \
        run.n_envs=${N_ENVS} \
        run.save_freq=${SAVE_FREQ} \
        "run.log_interval=[${LOG_INTERVAL},'step']" \
        environment.verbose=False
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Train ReactiveHumanHandoverCart
# ─────────────────────────────────────────────────────────────────────────────
print_step "2/6  Training ReactiveHumanHandoverCart — ${N_STEPS} steps, seed ${SEED}"
(
    set -x
    python human_robot_gym/training/train_SB3.py --multirun \
        -cp config_icra_2024/environment_evaluation/training \
        -cn RHH-reactive-v7 \
        hydra/launcher=ray \
        run.type=tensorboard \
        run.n_steps=${N_STEPS} \
        "run.seed=${SEED}" \
        run.n_envs=${N_ENVS} \
        run.save_freq=${SAVE_FREQ} \
        "run.log_interval=[${LOG_INTERVAL},'step']" \
        environment.verbose=False
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Evaluate ScriptedHandoverCartV2 — 4 checkpoints
# ─────────────────────────────────────────────────────────────────────────────
print_step "3/6  Evaluating ScriptedHandoverCartV2 — 4 checkpoints × ${N_EVAL_EPISODES} episodes"
(
    set -x
    python human_robot_gym/training/evaluate_models_to_csv_SB3.py \
        -cp config_icra_2024/environment_evaluation/evaluation \
        -cn RHH-scripted-v7 \
        "run.id=[${SCRIPTED_RUN_ID}]" \
        group_name=probe_v2/scripted \
        max_parallel_runs=4 \
        run.load_step=all \
        run.n_test_episodes=${N_EVAL_EPISODES} \
        run.n_steps=${N_STEPS} \
        run.save_freq=${SAVE_FREQ} \
        environment.verbose=False
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Evaluate ReactiveHumanHandoverCart — 4 checkpoints
# ─────────────────────────────────────────────────────────────────────────────
print_step "4/6  Evaluating ReactiveHumanHandoverCart — 4 checkpoints × ${N_EVAL_EPISODES} episodes"
(
    set -x
    python human_robot_gym/training/evaluate_models_to_csv_SB3.py \
        -cp config_icra_2024/environment_evaluation/evaluation \
        -cn RHH-reactive-v7 \
        "run.id=[${REACTIVE_RUN_ID}]" \
        group_name=probe_v2/reactive \
        max_parallel_runs=4 \
        run.load_step=all \
        run.n_test_episodes=${N_EVAL_EPISODES} \
        run.n_steps=${N_STEPS} \
        run.save_freq=${SAVE_FREQ} \
        environment.verbose=False
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Grip-guard sanity check — random-init policy (model_0)
#   A random policy should never trigger a legitimate RETREAT.
#   Any success here means spurious RETREAT transitions survived the grip guard.
# ─────────────────────────────────────────────────────────────────────────────
print_step "5/6  Grip-guard check — ScriptedHandoverCartV2 at step 0 (random policy, 20 episodes)"
(
    set -x
    python human_robot_gym/training/evaluate_models_to_csv_SB3.py \
        -cp config_icra_2024/environment_evaluation/evaluation \
        -cn RHH-scripted-v7 \
        "run.id=[${SCRIPTED_RUN_ID}]" \
        group_name=probe_v2/scripted_step0 \
        max_parallel_runs=1 \
        run.load_step=0 \
        run.n_test_episodes=20 \
        environment.verbose=False
)

print_step "6/6  Grip-guard check — ReactiveHumanHandoverCart at step 0 (random policy, 20 episodes)"
(
    set -x
    python human_robot_gym/training/evaluate_models_to_csv_SB3.py \
        -cp config_icra_2024/environment_evaluation/evaluation \
        -cn RHH-reactive-v7 \
        "run.id=[${REACTIVE_RUN_ID}]" \
        group_name=probe_v2/reactive_step0 \
        max_parallel_runs=1 \
        run.load_step=0 \
        run.n_test_episodes=20 \
        environment.verbose=False
)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print_step "PROBE COMPLETE — check results below"

echo ""
echo "  Checkpoints evaluated at: 25 000 / 50 000 / 75 000 / 100 000 steps"
echo ""
echo "  Training logs:"
echo "    runs/${PROJECT}/${SCRIPTED_GROUP}/run_${SEED}/"
echo "    runs/${PROJECT}/${REACTIVE_GROUP}/run_${SEED}/"
echo ""
echo "  Eval stats CSV (columns: step, success_mean, n_collisions_*_mean, failsafe_interventions_mean):"
echo "    csv/evaluation/probe_v2/scripted/stats.csv"
echo "    csv/evaluation/probe_v2/reactive/stats.csv"
echo ""
echo "  Grip-guard check CSV (success_mean should be 0.0):"
echo "    csv/evaluation/probe_v2/scripted_step0/stats.csv"
echo "    csv/evaluation/probe_v2/reactive_step0/stats.csv"
echo ""
echo "  Pass criteria for launching full 3M × 8-seed training:"
echo "    [PASS] No Python/MuJoCo exception in steps 1 or 2"
echo "    [PASS] success_mean at step=0 == 0.0 in grip-guard CSVs"
echo "    [PASS] success_mean at step=100000 > 0.0 in at least one condition"
echo "    [PASS] n_collisions_critical_mean < 5 per episode"
echo ""
echo "  If all pass, launch full training with:"
echo "    bash human_robot_gym/training/run_v2_full.sh"
