"""Trace V1 rewards and collision events under the built-in expert."""
import os
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


ROOT = Path("/home/human-robot-gym")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import human_robot_gym.environments  # noqa: E402,F401
import human_robot_gym.robots  # noqa: E402,F401
from human_robot_gym.utils.training_utils import (  # noqa: E402
    create_expert,
    create_wrapped_env_from_config,
)
from human_robot_gym.wrappers.expert_obs_wrapper import (  # noqa: E402
    ExpertObsWrapper,
)


def main() -> None:
    os.chdir(ROOT)
    GlobalHydra.instance().clear()
    config_dir = (
        ROOT
        / "human_robot_gym/training/config_icra_2024/"
        "environment_evaluation/evaluation"
    )
    with initialize_config_dir(
        config_dir=str(config_dir),
        job_name="trace_v1_rewards",
        version_base=None,
    ):
        config = compose(
            config_name="RHH-SIR-reactive-v1",
            overrides=[
                "run.eval_seed=123",
                "run.verbose=False",
            ],
        )

    env = create_wrapped_env_from_config(config=config, evaluation_mode=True)
    expert = create_expert(config=config, env=env)
    expert_obs = ExpertObsWrapper.get_from_wrapped_env(env=env)
    base = env.unwrapped
    try:
        env.reset()
        done = False
        step = 0
        rewards = []
        previous_illegal = 0
        new_collision_rewards = []
        info = {}
        while not done:
            action = np.asarray(
                expert(expert_obs.current_expert_observation)
            )
            _, reward, done, info = env.step(action)
            rewards.append(float(reward))
            illegal = (
                int(info["n_collisions_static"])
                + int(info["n_collisions_robot"])
                + int(info["n_collisions_critical"])
            )
            if illegal > previous_illegal:
                new_collision_rewards.append(
                    {
                        "step": step,
                        "reward": float(reward),
                        "illegal_count": illegal,
                        "collision_type": int(info["collision_type"]),
                    }
                )
            if step < 10:
                print(
                    {
                        "step": step,
                        "reward": round(float(reward), 4),
                        "collision_type": int(info["collision_type"]),
                        "illegal_count": illegal,
                        "phase": base.task_phase.name,
                    }
                )
            previous_illegal = illegal
            step += 1
        print(
            {
                "success": int(info["n_goal_reached"] > 0),
                "steps": step,
                "return": float(np.sum(rewards)),
                "mean_reward": float(np.mean(rewards)),
                "minimum_reward": float(np.min(rewards)),
                "new_collision_rewards": new_collision_rewards,
            }
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
