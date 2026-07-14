#!/bin/bash
# run_v2_full.sh — Full V2 experiment: 3M steps × 8 seeds × 2 conditions.
#
# DO NOT run this until probe_v2.sh has passed all 5 criteria.
#
# Usage (from project root):
#   bash human_robot_gym/training/run_v2_full.sh
#
# Output:
#   Models   — models/icra_2024/RHH_{scripted,reactive}_v7/run_{0..7}/
#   TB logs  — runs/icra_2024/RHH_{scripted,reactive}_v7/run_{0..7}/
#   Eval CSV — csv/evaluation/full_v2/{scripted,reactive}/stats.csv
#
# Evaluation config uses 500 episodes (same as baseline RHH-SIR evaluation).

set -euo pipefail

export CUDA_VISIBLE_DEVICES=""

N_STEPS=3000001
SAVE_FREQ=200000
N_ENVS=8
N_EVAL_EPISODES=500
LOG_INTERVAL=8000
SEEDS="0,1,2,3,4,5,6,7"

PROJECT=icra_2024
SCRIPTED_GROUP=RHH_scripted_v7
REACTIVE_GROUP=RHH_reactive_v7

print_step() { printf '\n\033[0;32m[run_v2_full] %s\033[0m\n' "$1"; }

build_run_ids() {
    local group=$1
    local ids=""
    for s in 0 1 2 3 4 5 6 7; do
        ids="${ids:+${ids},}${PROJECT}/${group}/run_${s}"
    done
    echo "${ids}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Train ScriptedHandoverCartV2 (8 seeds in parallel via Ray)
# ─────────────────────────────────────────────────────────────────────────────
print_step "1/4  Training ScriptedHandoverCartV2 — ${N_STEPS} steps, seeds ${SEEDS}"
(
    set -x
    python human_robot_gym/training/train_SB3.py --multirun \
        -cp config_icra_2024/environment_evaluation/training \
        -cn RHH-scripted-v7 \
        hydra/launcher=ray \
        run.type=tensorboard \
        run.n_steps=${N_STEPS} \
        "run.seed=${SEEDS}" \
        run.n_envs=${N_ENVS} \
        run.save_freq=${SAVE_FREQ} \
        "run.log_interval=[${LOG_INTERVAL},'step']" \
        environment.verbose=False
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Train ReactiveHumanHandoverCart (8 seeds in parallel via Ray)
# ─────────────────────────────────────────────────────────────────────────────
print_step "2/4  Training ReactiveHumanHandoverCart — ${N_STEPS} steps, seeds ${SEEDS}"
(
    set -x
    python human_robot_gym/training/train_SB3.py --multirun \
        -cp config_icra_2024/environment_evaluation/training \
        -cn RHH-reactive-v7 \
        hydra/launcher=ray \
        run.type=tensorboard \
        run.n_steps=${N_STEPS} \
        "run.seed=${SEEDS}" \
        run.n_envs=${N_ENVS} \
        run.save_freq=${SAVE_FREQ} \
        "run.log_interval=[${LOG_INTERVAL},'step']" \
        environment.verbose=False
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluate ScriptedHandoverCartV2 — all checkpoints, all seeds
# ─────────────────────────────────────────────────────────────────────────────
print_step "3/4  Evaluating ScriptedHandoverCartV2 — all checkpoints, 8 seeds, ${N_EVAL_EPISODES} episodes each"
SCRIPTED_RUN_IDS="[$(build_run_ids ${SCRIPTED_GROUP})]"
(
    set -x
    python human_robot_gym/training/evaluate_models_to_csv_SB3.py \
        -cp config_icra_2024/environment_evaluation/evaluation \
        -cn RHH-scripted-v7 \
        "run.id=${SCRIPTED_RUN_IDS}" \
        group_name=full_v2/scripted \
        max_parallel_runs=50 \
        run.load_step=all \
        run.n_test_episodes=${N_EVAL_EPISODES} \
        run.n_steps=${N_STEPS} \
        run.save_freq=${SAVE_FREQ} \
        environment.verbose=False
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluate ReactiveHumanHandoverCart — all checkpoints, all seeds
# ─────────────────────────────────────────────────────────────────────────────
print_step "4/4  Evaluating ReactiveHumanHandoverCart — all checkpoints, 8 seeds, ${N_EVAL_EPISODES} episodes each"
REACTIVE_RUN_IDS="[$(build_run_ids ${REACTIVE_GROUP})]"
(
    set -x
    python human_robot_gym/training/evaluate_models_to_csv_SB3.py \
        -cp config_icra_2024/environment_evaluation/evaluation \
        -cn RHH-reactive-v7 \
        "run.id=${REACTIVE_RUN_IDS}" \
        group_name=full_v2/reactive \
        max_parallel_runs=50 \
        run.load_step=all \
        run.n_test_episodes=${N_EVAL_EPISODES} \
        run.n_steps=${N_STEPS} \
        run.save_freq=${SAVE_FREQ} \
        environment.verbose=False
)


print_step "FULL TRAINING COMPLETE"
echo ""
echo "  Final eval stats:"
echo "    csv/evaluation/full_v2/scripted/stats.csv"
echo "    csv/evaluation/full_v2/reactive/stats.csv"
echo ""
echo "  Columns: step, success_mean/std/025/975, n_collisions_*_mean, failsafe_interventions_mean"
