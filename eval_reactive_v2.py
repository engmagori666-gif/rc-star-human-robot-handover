"""Evaluate reactive-v2 final models (speed_min=0.05)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MUJOCO_GL"] = "osmesa"

import ray._private.utils as _u
_o = _u.get_system_memory
def _p():
    try: return _o()
    except: import psutil; return psutil.virtual_memory().total
_u.get_system_memory = _p

import robosuite  # noqa: F401
import human_robot_gym.robots  # noqa: F401
import human_robot_gym.environments  # noqa: F401

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from human_robot_gym.training.evaluate_models_to_csv_SB3 import evaluate_to_stats_df

RUNS = ["icra_2024/RHH_SIR_reactive_v2/run_" + str(s) for s in range(5)]

def build_config():
    GlobalHydra.instance().clear()
    config_dir = os.path.abspath(
        "human_robot_gym/training/config_icra_2024/environment_evaluation/evaluation"
    )
    runs_str = ",".join(RUNS)
    with initialize_config_dir(config_dir=config_dir, job_name="eval"):
        cfg = compose(
            config_name="RHH-reactive",
            overrides=[
                "run.id=[" + runs_str + "]",
                "run.load_step=final",
                "run.n_test_episodes=20",
                "run.n_steps=3000001",
                "run.save_freq=200000",
                "environment.verbose=False",
                "+environment.speed_min=0.05",
                "wrappers.dataset_obs_norm.dataset_name=reactive-human-handover-v2",
                "group_name=reactive_v2",
            ],
        )
    return cfg

if __name__ == "__main__":
    os.chdir("/home/human-robot-gym")
    cfg = build_config()
    print("Evaluating: " + str(RUNS))
    df = evaluate_to_stats_df(cfg, max_parallel_runs=1)
    print("")
    print("RESULTS:")
    print(df.to_string())
