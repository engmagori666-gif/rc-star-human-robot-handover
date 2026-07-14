"""Paper table evaluation.
Patches Ray's Docker cgroup bug, then runs the standard evaluator sequentially.
Run from /home/human-robot-gym inside the container.
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MUJOCO_GL"] = "osmesa"

# Patch Ray BEFORE any import triggers ray.init
import ray._private.utils as _ray_utils
_orig_gsm = _ray_utils.get_system_memory
def _patched_gsm():
    try:
        return _orig_gsm()
    except (ValueError, TypeError):
        import psutil
        return psutil.virtual_memory().total
_ray_utils.get_system_memory = _patched_gsm

import numpy as np
import robosuite  # noqa: F401
import human_robot_gym.robots  # noqa: F401
import human_robot_gym.environments  # noqa: F401

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from human_robot_gym.training.evaluate_models_to_csv_SB3 import evaluate_to_stats_df

BASELINE_RUNS = [f"icra_2024/RHH_SIR/run_{s}" for s in range(5)]
REACTIVE_RUNS = [f"icra_2024/RHH_SIR_reactive/run_{s}" for s in [1, 2, 3, 4]]

METRICS = {
    "n_goal_reached_mean":        "Success rate",
    "failsafe_interventions_mean": "Shield activations / ep",
    "n_collisions_human_mean":    "Human collisions / ep",
    "n_collisions_critical_mean": "Critical collisions / ep",
    "ep_len_mean":                "Episode length",
}


def build_config(config_name, run_ids, dataset_name):
    GlobalHydra.instance().clear()
    config_dir = os.path.abspath(
        "human_robot_gym/training/config_icra_2024/environment_evaluation/evaluation"
    )
    with initialize_config_dir(config_dir=config_dir, job_name="eval"):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"run.id=[{','.join(run_ids)}]",
                "run.load_step=final",
                "run.n_test_episodes=20",
                "run.n_steps=3000001",
                "run.save_freq=200000",
                "environment.verbose=False",
                f"wrappers.dataset_obs_norm.dataset_name={dataset_name}",
                "group_name=paper_eval",
            ],
        )
    return cfg


def run_evaluation(config_name, run_ids, dataset_name, label):
    print(f"\n{'='*60}\n{label}\nSeeds: {run_ids}\n{'='*60}")
    cfg = build_config(config_name, run_ids, dataset_name)
    # Run evaluations one at a time to avoid pipe issues (max_parallel_runs=1)
    df = evaluate_to_stats_df(cfg, max_parallel_runs=1)
    print(df.to_string())
    return df


if __name__ == "__main__":
    os.chdir("/home/human-robot-gym")

    df_base  = run_evaluation("RHH",          BASELINE_RUNS, "robot-human-handover",
                               "BASELINE — pre-planned human (RHH-SIR, 5 seeds)")
    df_react = run_evaluation("RHH-reactive",  REACTIVE_RUNS, "reactive-human-handover",
                               "REACTIVE — your model (RHH-SIR-reactive, 4 seeds)")

    print("\n" + "="*60)
    print("PAPER COMPARISON TABLE  (mean ± std across seeds)")
    print("="*60)
    for label, df in [("Baseline (pre-planned)", df_base),
                      ("Reactive (your model)",  df_react)]:
        if df is None:
            continue
        print(f"\n{label}:")
        for col, display in METRICS.items():
            std_col = col.replace("_mean", "_std")
            if col in df.columns:
                m = df[col].values[0]
                s = df[std_col].values[0] if std_col in df.columns else float("nan")
                print(f"  {display:35s}: {m:.3f} ± {s:.3f}")
