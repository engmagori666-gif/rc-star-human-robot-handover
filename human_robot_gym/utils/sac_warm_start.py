"""Conservative warm-start utilities for behavior-cloned SAC policies."""
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update


def prefill_sac_replay_buffer(
    model,
    env,
    steps: int,
) -> Dict[str, float]:
    """Fill replay with deterministic trajectories from the cloned actor."""
    if steps <= 0:
        return {"steps": 0}

    obs = env.reset()
    successes = 0
    episodes = 0
    rewards = []

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        buffer_action = model.policy.scale_action(action)
        next_obs, reward, done, infos = env.step(action)

        buffer_next_obs = np.array(next_obs).copy()
        for index, terminal in enumerate(done):
            if terminal and "terminal_observation" in infos[index]:
                buffer_next_obs[index] = infos[index][
                    "terminal_observation"
                ]

        model.replay_buffer.add(
            obs=obs,
            next_obs=buffer_next_obs,
            action=buffer_action,
            reward=reward,
            done=done,
            infos=infos,
        )
        rewards.extend(np.asarray(reward, dtype=float).tolist())
        for index, terminal in enumerate(done):
            if terminal:
                episodes += 1
                successes += int(
                    infos[index].get("n_goal_reached", 0) > 0
                )
        obs = next_obs

    metrics = {
        "steps": int(steps),
        "episodes": int(episodes),
        "successes": int(successes),
        "success_rate": (
            float(successes / episodes) if episodes else 0.0
        ),
        "mean_reward": float(np.mean(rewards)),
        "buffer_size": int(model.replay_buffer.size()),
    }
    print(f"[SAC warm start] replay prefill: {json.dumps(metrics)}")
    return metrics


def pretrain_sac_critics(
    model,
    gradient_steps: int,
    batch_size: int,
) -> Dict[str, float]:
    """Ground SAC critics on the fixed deterministic cloned actor."""
    if gradient_steps <= 0:
        return {"gradient_steps": 0}
    if model.replay_buffer.size() < batch_size:
        raise RuntimeError(
            "Replay buffer is too small for critic warmup: "
            f"{model.replay_buffer.size()} < {batch_size}"
        )

    model.policy.set_training_mode(True)
    actor_parameters = list(model.actor.parameters())
    actor_requires_grad = [
        parameter.requires_grad for parameter in actor_parameters
    ]
    for parameter in actor_parameters:
        parameter.requires_grad_(False)

    critic_losses = []
    try:
        for gradient_step in range(gradient_steps):
            replay_data = model.replay_buffer.sample(
                batch_size,
                env=model._vec_normalize_env,
            )
            with th.no_grad():
                next_actions = model.actor(
                    replay_data.next_observations,
                    deterministic=True,
                )
                next_q_values = th.cat(
                    model.critic_target(
                        replay_data.next_observations,
                        next_actions,
                    ),
                    dim=1,
                )
                next_q_values, _ = th.min(
                    next_q_values,
                    dim=1,
                    keepdim=True,
                )
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones)
                    * model.gamma
                    * next_q_values
                )

            current_q_values = model.critic(
                replay_data.observations,
                replay_data.actions,
            )
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            model.critic.optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(
                model.critic.parameters(),
                max_norm=10.0,
            )
            model.critic.optimizer.step()
            critic_losses.append(float(critic_loss.detach().cpu()))

            if gradient_step % model.target_update_interval == 0:
                polyak_update(
                    model.critic.parameters(),
                    model.critic_target.parameters(),
                    model.tau,
                )
            if (
                gradient_step == 0
                or (gradient_step + 1) % 1000 == 0
                or gradient_step + 1 == gradient_steps
            ):
                recent = critic_losses[-1000:]
                print(
                    f"[SAC warm start] critic step "
                    f"{gradient_step + 1}/{gradient_steps}, "
                    f"loss={np.mean(recent):.6f}"
                )
    finally:
        for parameter, requires_grad in zip(
            actor_parameters,
            actor_requires_grad,
        ):
            parameter.requires_grad_(requires_grad)

    metrics = {
        "gradient_steps": int(gradient_steps),
        "initial_critic_loss": float(critic_losses[0]),
        "final_critic_loss": float(
            np.mean(critic_losses[-min(1000, len(critic_losses)):])
        ),
    }
    return metrics


def configure_sac_actor_finetune(
    model,
    actor_learning_rate: float,
    freeze_actor: bool = False,
) -> None:
    """Use a small actor learning rate after critic grounding."""
    model.actor.optimizer = th.optim.Adam(
        model.actor.parameters(),
        lr=actor_learning_rate,
    )
    print(
        "[SAC warm start] actor fine-tune learning rate: "
        f"{actor_learning_rate}"
    )
    if freeze_actor:
        for parameter in model.actor.parameters():
            parameter.requires_grad_(False)
        print(
            "[SAC warm start] actor frozen; SAC will calibrate critics "
            "without erasing the validated handover policy."
        )


def write_warm_start_metrics(metrics: Dict, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2))
