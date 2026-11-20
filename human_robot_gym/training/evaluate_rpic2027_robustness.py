"""Run the paired, zero-shot RPIC 2027 human-motion robustness evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import robosuite  # noqa: F401
import human_robot_gym.environments  # noqa: F401

from human_robot_gym.utils.training_utils import create_wrapped_env_from_config
from human_robot_gym.utils.training_utils_SB3 import load_model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / (
    "human_robot_gym/training/config_icra_2024/environment_evaluation/"
    "evaluation/RPIC2027_PROTOCOL.yaml"
)

METRICS = (
    "n_goal_reached",
    "n_collisions_human",
    "n_collisions_critical",
    "timeout",
    "failsafe_interventions",
    "human_governor_interventions",
    "human_governor_budget_exhausted",
    "v1_capture_count",
    "v1_release_confirmed",
    "rpic_deadlock",
    "rpic_max_stall_steps",
    "rpic_disturbance_triggered",
    "rpic_disturbance_completed",
    "rpic_trigger_delay_steps",
    "rpic_disturbed_policy_steps",
    "rpic_disturbed_low_level_updates",
    "rpic_shield_measurement_calls_during_disturbance",
    "rpic_shield_measurement_changes_during_disturbance",
    "rpic_shield_measurement_stale_mismatches",
    "rpic_shield_measurement_time_regressions",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_provenance(protocol) -> dict:
    report = {}
    for name, artifact in protocol.provenance.items():
        path = ROOT / str(artifact.path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(artifact.bytes) or digest != str(artifact.sha256):
            raise RuntimeError(
                f"Provenance mismatch for {name}: {path}\n"
                f"expected bytes={artifact.bytes}, sha256={artifact.sha256}\n"
                f"actual   bytes={size}, sha256={digest}"
            )
        report[name] = {"path": str(path), "bytes": size, "sha256": digest}
    return report


def make_config(protocol, condition: str, episodes: int, evaluation_seed: int):
    config = OmegaConf.load(ROOT / str(protocol.base_evaluation_config))
    config.environment.env_id = str(protocol.environment_id)
    config.environment.rpic_condition = condition
    config.environment.rpic_disturbance_seed = int(protocol.disturbance.seed)
    config.environment.rpic_stable_grasp_steps = int(
        protocol.disturbance.stable_grasp_steps
    )
    config.environment.rpic_trigger_min_steps = int(
        protocol.disturbance.trigger_min_steps
    )
    config.environment.rpic_trigger_max_steps = int(
        protocol.disturbance.trigger_max_steps
    )
    config.environment.rpic_delay_duration_steps = int(
        protocol.disturbance.delay_duration_steps
    )
    config.environment.rpic_reversal_duration_steps = int(
        protocol.disturbance.reversal_duration_steps
    )
    config.environment.rpic_reversal_speed = float(
        protocol.disturbance.reversal_speed_frames_per_low_level_update
    )
    config.environment.rpic_deadlock_window_steps = int(
        protocol.deadlock.window_steps
    )
    config.environment.rpic_deadlock_progress_epsilon = float(
        protocol.deadlock.progress_epsilon_m
    )
    config.run.id = str(protocol.model_run_id)
    config.run.load_step = str(protocol.load_step)
    config.run.eval_seed = int(evaluation_seed)
    config.run.n_test_episodes = int(episodes)
    return config


def evaluate_condition(protocol, condition: str, episodes: int, seed: int):
    config = make_config(protocol, condition, episodes, seed)
    # Environment construction pre-samples animation IDs. Seed before creating
    # each condition so those IDs and all other initial conditions are paired.
    np.random.seed(seed)
    env = create_wrapped_env_from_config(config=config, evaluation_mode=True)
    model = load_model(
        config=config,
        env=None,
        run_id=str(protocol.model_run_id),
        load_step=str(protocol.load_step),
    )
    environment_observation_shape = env.observation_space.shape
    append_time_feature = model.observation_space.shape != environment_observation_shape
    mean_episode_length = None
    if append_time_feature:
        model_shape = model.observation_space.shape
        valid_sir_shape = (
            len(model_shape) == len(environment_observation_shape) == 1
            and model_shape[0] == environment_observation_shape[0] + 1
        )
        if not valid_sir_shape:
            raise RuntimeError(
                "Unsupported checkpoint/environment observation mismatch: "
                f"{model_shape} != {environment_observation_shape}"
            )
        stats_path = ROOT / (
            "datasets/reactive-human-handover-v1-bc-success/stats.csv"
        )
        mean_episode_length = float(pd.read_csv(stats_path).ep_len_mean.iloc[0])
        env.observation_space = model.observation_space
    if model.action_space.shape != env.action_space.shape:
        raise RuntimeError(
            "Checkpoint and evaluation action shapes differ: "
            f"{model.action_space.shape} != {env.action_space.shape}"
        )
    model.set_env(env)

    # Model loading restores its training RNG. Resetting here preserves paired
    # environment initial conditions across independently run conditions.
    np.random.seed(seed)
    rows = []
    for episode_index in range(episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            policy_obs = obs
            if append_time_feature:
                time_value = min(episode_length / mean_episode_length, 1.0)
                policy_obs = np.append(obs, time_value)
            action, _ = model.predict(policy_obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_return += float(reward)
            episode_length += 1

        row = {
            "condition": condition,
            "evaluation_seed": seed,
            "paired_episode_index": episode_index,
            "episode_return": episode_return,
            "episode_length": episode_length,
            "success": int(info["n_goal_reached"] > 0),
        }
        for metric in METRICS:
            row[metric] = info[metric]
        rows.append(row)
    env.close()
    return pd.DataFrame(rows), {
        "environment_observation_shape": list(environment_observation_shape),
        "observation_shape": list(model.observation_space.shape),
        "action_shape": list(model.action_space.shape),
        "existing_sir_time_feature_appended": append_time_feature,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "csv/evaluation/rpic2027_robustness",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--allow-full",
        action="store_true",
        help="Required when requesting more than 10 episodes per condition.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.episodes > 10 and not args.allow_full:
        raise ValueError("Use --allow-full for more than 10 episodes per condition")

    protocol = OmegaConf.load(args.protocol)
    provenance = verify_provenance(protocol)
    if args.verify_only:
        print(json.dumps(provenance, indent=2))
        return

    conditions = args.conditions or list(protocol.conditions)
    unknown = sorted(set(conditions) - set(protocol.conditions))
    if unknown:
        raise ValueError(f"Conditions are not in the protocol: {unknown}")
    seed = int(args.seed if args.seed is not None else protocol.evaluation_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    spaces = {}
    for condition in conditions:
        frame, condition_spaces = evaluate_condition(
            protocol, condition, args.episodes, seed
        )
        frame.to_csv(args.output_dir / f"{condition}.csv", index=False)
        frames.append(frame)
        spaces[condition] = condition_spaces

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output_dir / "episodes.csv", index=False)
    summary = combined.groupby("condition", sort=False).agg(
        episodes=("success", "size"),
        success_rate=("success", "mean"),
        human_collisions=("n_collisions_human", "sum"),
        critical_collisions=("n_collisions_critical", "sum"),
        timeout_rate=("timeout", "mean"),
        deadlock_rate=("rpic_deadlock", "mean"),
        shield_interventions=("failsafe_interventions", "sum"),
        governor_interventions=("human_governor_interventions", "sum"),
        governor_budget_exhaustion_rate=(
            "human_governor_budget_exhausted",
            "mean",
        ),
        disturbance_trigger_rate=("rpic_disturbance_triggered", "mean"),
        stale_measurement_mismatches=(
            "rpic_shield_measurement_stale_mismatches",
            "sum",
        ),
    )
    summary.to_csv(args.output_dir / "summary.csv")
    manifest = {
        "protocol": OmegaConf.to_container(protocol, resolve=True),
        "episodes_per_condition": args.episodes,
        "evaluation_seed": seed,
        "conditions": conditions,
        "spaces": spaces,
        "provenance": provenance,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string())
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()

