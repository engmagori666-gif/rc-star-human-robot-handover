"""Trace deterministic cooperative-V1 collision episodes."""
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
        default="icra_2024/RHH_SIR_reactive_v1_predictive_final/run_0",
    )
    parser.add_argument(
        "--output",
        default=(
            "/home/human-robot-gym/csv/evaluation/"
            "reactive_v1_coop_collision_trace"
        ),
    )
    parser.add_argument("--capture-surface-dist", type=float, default=0.06)
    parser.add_argument("--release-supervisor", action="store_true")
    return parser.parse_args()


def min_distance(point: np.ndarray, cloud: np.ndarray) -> float:
    if len(cloud) == 0:
        return float("inf")
    return float(np.linalg.norm(cloud - point, axis=1).min())


def base_positions(base):
    eef_pos = np.asarray(
        base.sim.data.site_xpos[base.robots[0].eef_site_id],
        dtype=float,
    )
    object_pos = np.asarray(
        base.sim.data.body_xpos[base.manipulation_object_body_id],
        dtype=float,
    )
    human = np.asarray(base.human_measurement, dtype=float)
    target = np.asarray(base.target_pos, dtype=float)
    return eef_pos, object_pos, human, target


def contact_pairs(base):
    pairs = []
    for contact in base.sim.data.contact[:base.sim.data.ncon]:
        name1 = base.sim.model.geom_id2name(contact.geom1)
        name2 = base.sim.model.geom_id2name(contact.geom2)
        pairs.append(f"{name1}|{name2}")
    return ";".join(pairs)


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
        job_name="trace_cooperative_collisions",
        version_base=None,
    ):
        config = compose(
            config_name="RHH-SIR-reactive-v1-strict",
            overrides=[
                f"run.eval_seed={args.eval_seed}",
                "run.verbose=False",
                "+wrappers.human_aware_action_governor.activation_distance=0.22",
                "+wrappers.human_aware_action_governor.hard_distance=0.18",
                "+wrappers.human_aware_action_governor.prediction_horizon=1.0",
                "+wrappers.human_aware_action_governor.minimum_scale=0.25",
                "+wrappers.human_aware_action_governor.max_interventions=75",
                "+environment.v1_cooperative_human_reach=true",
                "+environment.v1_cooperative_reach_target_dist=0.10",
                "+environment.v1_cooperative_reach_max_offset=0.10",
                "+environment.v1_cooperative_reach_step=0.002",
                "+environment.v1_cooperative_reach_delay_steps=100",
                f"environment.v1_capture_surface_dist={args.capture_surface_dist}",
            ],
        )
        if args.release_supervisor:
            config.environment.v1_release_supervisor = True

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
    episode_summaries = []
    collision_rows = []

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
                eef_pos, object_pos, human, target = base_positions(base)
                capture_gap = np.inf
                try:
                    _, capture_gap = base._get_capture_contact()
                except Exception:
                    pass
                row = {
                    "episode": episode,
                    "step": step,
                    "phase_before": getattr(
                        base.task_phase,
                        "name",
                        "UNKNOWN",
                    ),
                    "object_gripped_before": int(
                        base._is_robot_gripping_object()
                    ),
                    "object_captured_before": int(
                        getattr(base, "_v1_object_captured", False)
                    ),
                    "eef_object_distance": float(
                        np.linalg.norm(eef_pos - object_pos)
                    ),
                    "eef_human_distance": min_distance(eef_pos, human),
                    "object_human_distance": min_distance(object_pos, human),
                    "object_target_distance": float(
                        np.linalg.norm(object_pos - target)
                    ),
                    "capture_surface_gap": float(capture_gap),
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
                        "phase_after": getattr(
                            base.task_phase,
                            "name",
                            "UNKNOWN",
                        ),
                        "collision": bool(info.get("collision", False)),
                        "collision_type": int(
                            info.get("collision_type", 0)
                        ),
                        "critical": int(
                            info.get("n_collisions_critical", 0)
                        ),
                        "static": int(
                            info.get("n_collisions_static", 0)
                        ),
                        "shield": int(
                            info.get("failsafe_interventions", 0)
                        ),
                        "governor": int(
                            info.get(
                                "human_governor_interventions",
                                0,
                            )
                        ),
                        "governor_scale": float(
                            info.get("human_governor_last_scale", 1.0)
                        ),
                        "governor_min_predicted_distance": float(
                            info.get(
                                "human_governor_min_predicted_distance",
                                np.inf,
                            )
                        ),
                        "cooperative_reach_steps": int(
                            info.get("v1_cooperative_reach_steps", 0)
                        ),
                        "human_yield_steps": int(
                            info.get("v1_human_yield_steps", 0)
                        ),
                        "contact_pairs": contact_pairs(base)
                        if info.get("collision", False)
                        else "",
                    }
                )
                episode_rows.append(row)
                step += 1

            frame = pd.DataFrame(episode_rows)
            success = int(info.get("n_goal_reached", 0) > 0)
            collision = int(bool(info.get("collision", False)))
            summary = {
                "episode": episode,
                "success": success,
                "collision": collision,
                "timeout": int(info.get("timeout", 0)),
                "steps": step,
                "captured": int(info.get("v1_object_captured", 0)),
                "collision_type": int(info.get("collision_type", 0)),
                "critical": int(info.get("n_collisions_critical", 0)),
                "static": int(info.get("n_collisions_static", 0)),
                "min_eef_human": float(
                    frame["eef_human_distance"].min()
                ),
                "min_object_human": float(
                    frame["object_human_distance"].min()
                ),
                "final_phase": getattr(base.task_phase, "name", "UNKNOWN"),
                "object_gripped_last": int(
                    frame["object_gripped_before"].iloc[-1]
                ),
                "shield": int(info.get("failsafe_interventions", 0)),
                "governor": int(
                    info.get("human_governor_interventions", 0)
                ),
                "budget_exhausted": int(
                    info.get("human_governor_budget_exhausted", 0)
                ),
            }
            episode_summaries.append(summary)
            if collision and not success:
                collision_rows.extend(episode_rows)
                frame.to_csv(
                    output / f"collision_episode_{episode:03d}.csv",
                    index=False,
                )
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        env.close()

    pd.DataFrame(episode_summaries).to_csv(
        output / "episode_summary.csv",
        index=False,
    )
    if collision_rows:
        pd.DataFrame(collision_rows).to_csv(
            output / "collision_steps.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
