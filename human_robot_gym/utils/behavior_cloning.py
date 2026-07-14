"""Behavior cloning utilities for SAC handover policies."""
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F


PROJECT_ROOT = Path("/home/human-robot-gym")


def load_bc_dataset(
    dataset_name: str,
    validation_fraction: float = 0.1,
    seed: int = 0,
    append_time: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and normalize demonstration observations and actions.

    Dataset observations are normalized with the same statistics used by
    ``DatasetObsNormWrapper``. The optional SIR time feature is appended
    afterward when it is part of the policy observation.
    """
    dataset_dir = PROJECT_ROOT / "datasets" / dataset_name
    stats_path = dataset_dir / "observations.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing observation stats: {stats_path}")

    stats = pd.read_csv(stats_path)
    mean = stats["mean"].to_numpy(dtype=np.float32)
    std = stats["std"].to_numpy(dtype=np.float32)
    std[std == 0] = 1

    episodes = sorted(dataset_dir.glob("ep_*"))
    if not episodes:
        raise RuntimeError(f"No episodes in BC dataset: {dataset_dir}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    validation_count = max(1, int(round(len(episodes) * validation_fraction)))
    validation_indices = set(order[:validation_count].tolist())

    train_obs = []
    train_actions = []
    validation_obs = []
    validation_actions = []

    for episode_index, episode_dir in enumerate(episodes):
        with np.load(episode_dir / "state.npz", allow_pickle=True) as data:
            observations = np.asarray(data["observations"][:-1], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)

        if len(observations) != len(actions):
            raise RuntimeError(
                f"Observation/action mismatch in {episode_dir}: "
                f"{len(observations)} != {len(actions)}"
            )
        if observations.shape[1] != mean.shape[0]:
            raise RuntimeError(
                f"Observation stats mismatch: {observations.shape[1]} != {mean.shape[0]}"
            )

        observations = (observations - mean) / std
        if append_time:
            transition_count = max(len(actions), 1)
            time_feature = (
                np.arange(len(actions), dtype=np.float32) / transition_count
            )[:, None]
            observations = np.concatenate([observations, time_feature], axis=1)

        if episode_index in validation_indices:
            validation_obs.append(observations)
            validation_actions.append(actions)
        else:
            train_obs.append(observations)
            train_actions.append(actions)

    return (
        np.concatenate(train_obs),
        np.concatenate(train_actions),
        np.concatenate(validation_obs),
        np.concatenate(validation_actions),
    )


def _evaluate_actor(
    model,
    observations: np.ndarray,
    scaled_actions: np.ndarray,
    batch_size: int,
) -> Dict[str, float]:
    actor = model.actor
    device = model.device
    squared_errors = []
    absolute_errors = []

    actor.set_training_mode(False)
    with th.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = th.as_tensor(
                observations[start:start + batch_size],
                device=device,
            )
            target = th.as_tensor(
                scaled_actions[start:start + batch_size],
                device=device,
            )
            prediction = actor(obs, deterministic=True)
            error = prediction - target
            squared_errors.append(th.square(error).cpu().numpy())
            absolute_errors.append(th.abs(error).cpu().numpy())

    squared_error = np.concatenate(squared_errors)
    absolute_error = np.concatenate(absolute_errors)
    return {
        "mse": float(np.mean(squared_error)),
        "mae": float(np.mean(absolute_error)),
        "motion_mae": float(np.mean(absolute_error[:, :3])),
        "gripper_mae": float(np.mean(absolute_error[:, 3:])),
    }


def pretrain_sac_actor(
    model,
    dataset_name: str,
    epochs: int = 50,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    validation_fraction: float = 0.1,
    log_std_target: float = -3.0,
    log_std_weight: float = 0.01,
    seed: int = 0,
    append_time: bool = False,
    metrics_path: str = None,
) -> Dict[str, float]:
    """Pretrain the SAC actor on successful final-environment demonstrations."""
    if not hasattr(model, "actor"):
        raise TypeError("Behavior cloning currently supports SAC-style actors only")

    train_obs, train_actions, validation_obs, validation_actions = load_bc_dataset(
        dataset_name=dataset_name,
        validation_fraction=validation_fraction,
        seed=seed,
        append_time=append_time,
    )
    if train_obs.shape[1:] != model.observation_space.shape:
        raise RuntimeError(
            f"BC/model observation mismatch: {train_obs.shape[1:]} "
            f"!= {model.observation_space.shape}"
        )

    train_scaled_actions = model.policy.scale_action(train_actions)
    validation_scaled_actions = model.policy.scale_action(validation_actions)
    if np.max(np.abs(train_scaled_actions)) > 1.0001:
        raise RuntimeError("Scaled BC actions exceed SAC action bounds")

    actor = model.actor
    device = model.device
    optimizer = th.optim.Adam(actor.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)

    initial_metrics = _evaluate_actor(
        model,
        validation_obs,
        validation_scaled_actions,
        batch_size,
    )
    print(
        f"[BC] transitions train={len(train_obs)} validation={len(validation_obs)} "
        f"initial_val_mse={initial_metrics['mse']:.6f}"
    )

    for epoch in range(epochs):
        actor.set_training_mode(True)
        permutation = rng.permutation(len(train_obs))
        epoch_losses = []

        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start + batch_size]
            obs = th.as_tensor(train_obs[indices], device=device)
            target = th.as_tensor(train_scaled_actions[indices], device=device)

            prediction = actor(obs, deterministic=True)
            _, log_std, _ = actor.get_action_dist_params(obs)
            action_loss = F.smooth_l1_loss(prediction, target)
            std_loss = F.mse_loss(
                log_std,
                th.full_like(log_std, log_std_target),
            )
            loss = action_loss + log_std_weight * std_loss

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            validation_metrics = _evaluate_actor(
                model,
                validation_obs,
                validation_scaled_actions,
                batch_size,
            )
            print(
                f"[BC] epoch={epoch + 1}/{epochs} "
                f"loss={np.mean(epoch_losses):.6f} "
                f"val_mse={validation_metrics['mse']:.6f} "
                f"motion_mae={validation_metrics['motion_mae']:.6f} "
                f"gripper_mae={validation_metrics['gripper_mae']:.6f}"
            )

    # Continue SAC with its configured actor learning rate.
    model.actor.optimizer = th.optim.Adam(
        model.actor.parameters(),
        lr=model.lr_schedule(1),
    )

    final_metrics = _evaluate_actor(
        model,
        validation_obs,
        validation_scaled_actions,
        batch_size,
    )
    metrics = {
        "dataset_name": dataset_name,
        "train_transitions": int(len(train_obs)),
        "validation_transitions": int(len(validation_obs)),
        "epochs": int(epochs),
        "append_time": bool(append_time),
        "initial_validation": initial_metrics,
        "final_validation": final_metrics,
    }
    if metrics_path is not None:
        output = Path(metrics_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2))
    return metrics
