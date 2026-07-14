"""Trace deterministic predictive-V1 episodes and summarize approach stalls."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


ROOT = Path("/home/human-robot-gym")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import human_robot_gym.environments  # noqa: E402,F401
import human_robot_gym.robots  # noqa: E402,F401
from human_robot_gym.utils.training_utils import (  # noqa: E402
    create_wrapped_env_from_config,
)
from human_robot_gym.utils.training_utils_SB3 import load_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=100)
    parser.add_argument(
        "--model",
        default=(
            "icra_2024/"
            "RHH_SIR_reactive_v1_predictive_final/run_0"
        ),
    )
    parser.add_argument(
        "--output",
        default="/home/human-robot-gym/csv/evaluation/"
        "predictive_v1_timeout_trace",
    )
    return parser.parse_args()


def min_human_distance(base) -> float:
    eef_pos = np.asarray(
        base.sim.data.site_xpos[base.robots[0].eef_site_id],
        dtype=float,
    )
    human = np.asarray(base.human_measurement, dtype=float)
    return float(np.linalg.norm(human - eef_pos, axis=1).min())


def main():
    args = parse_args()
    os.chdir(ROOT)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    GlobalHydra.instance().clear()
    config_dir = (
        ROOT
        / "human_robot_gym/training/config_icra_2024/"
        "environment_evaluation/evaluation"
    )
    with initialize_config_dir(
        config_dir=str(config_dir),
        job_name="trace_predictive_timeouts",
        version_base=None,
    ):
        config = compose(
            config_name="RHH-SIR-reactive-v1-predictive",
            overrides=[
                f"run.eval_seed={args.eval_seed}",
                "run.verbose=False",
            ],
        )

    env = create_wrapped_env_from_config(
        config=config,
        evaluation_mode=True,
    )
    model = load_model(
        config=config,
        env=None,
        run_id=args.model,
        load_step="final",
    )
    model.set_env(env)
    np.random.seed(args.eval_seed)
    base = env.unwrapped
    rows = []
    episode_summaries = []

    try:
        for episode in range(args.episodes):
            obs = env.reset()
            done = False
            step = 0
            info = {}
            episode_rows = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=float)
                cache = getattr(base, "_obs_cache", {})
                eef_object = float(
                    np.linalg.norm(
                        np.asarray(
                            cache.get("vec_eef_to_object", np.zeros(3))
                        )
                    )
                )
                eef_target = float(
                    np.linalg.norm(
                        np.asarray(
                            cache.get("vec_eef_to_target", np.zeros(3))
                        )
                    )
                )
                row = {
                    "episode": episode,
                    "step": step,
                    "phase": getattr(base.task_phase, "name", "UNKNOWN"),
                    "eef_object_distance": eef_object,
                    "eef_target_distance": eef_target,
                    "eef_human_distance": min_human_distance(base),
                    "object_gripped": int(
                        base._is_robot_gripping_object()
                    ),
                    "action_x": float(action[0]),
                    "action_y": float(action[1]),
                    "action_z": float(action[2]),
                    "action_norm": float(np.linalg.norm(action[:3])),
                    "gripper_action": float(action[-1]),
                }
                obs, reward, done, info = env.step(action)
                row.update(
                    {
                        "reward": float(reward),
                        "shield_interventions": int(
                            info.get("failsafe_interventions", 0)
                        ),
                        "governor_interventions": int(
                            info.get(
                                "human_governor_interventions",
                                0,
                            )
                        ),
                        "governor_scale": float(
                            info.get("human_governor_last_scale", 1.0)
                        ),
                        "critical": int(
                            info.get("n_collisions_critical", 0)
                        ),
                        "static": int(
                            info.get("n_collisions_static", 0)
                        ),
                    }
                )
                episode_rows.append(row)
                step += 1

            frame = pd.DataFrame(episode_rows)
            success = int(info.get("n_goal_reached", 0) > 0)
            timeout = int(info.get("timeout", 0))
            summary = {
                "episode": episode,
                "success": success,
                "timeout": timeout,
                "steps": step,
                "captured": int(info.get("v1_object_captured", 0)),
                "min_eef_object": float(
                    frame["eef_object_distance"].min()
                ),
                "final_eef_object": float(
                    frame["eef_object_distance"].iloc[-1]
                ),
                "min_eef_human": float(
                    frame["eef_human_distance"].min()
                ),
                "mean_action_norm_last_200": float(
                    frame["action_norm"].tail(200).mean()
                ),
                "mean_object_progress_last_200": float(
                    frame["eef_object_distance"].tail(200).iloc[0]
                    - frame["eef_object_distance"].tail(200).iloc[-1]
                ),
                "mean_gripper_last_200": float(
                    frame["gripper_action"].tail(200).mean()
                ),
                "shield": int(
                    info.get("failsafe_interventions", 0)
                ),
                "governor": int(
                    info.get("human_governor_interventions", 0)
                ),
                "budget_exhausted": int(
                    info.get(
                        "human_governor_budget_exhausted",
                        0,
                    )
                ),
                "critical": int(
                    info.get("n_collisions_critical", 0)
                ),
                "static": int(info.get("n_collisions_static", 0)),
            }
            episode_summaries.append(summary)
            if timeout:
                rows.extend(episode_rows)
                frame.to_csv(
                    output / f"episode_{episode:03d}.csv",
                    index=False,
                )
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        env.close()

    pd.DataFrame(episode_summaries).to_csv(
        output / "episode_summary.csv",
        index=False,
    )
    if rows:
        pd.DataFrame(rows).to_csv(
            output / "timeout_steps.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
