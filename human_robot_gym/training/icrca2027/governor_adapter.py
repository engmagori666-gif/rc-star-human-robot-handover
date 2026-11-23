"""Opt-in, evaluation-only governor replacement without editing legacy code.

Use AFTER create_wrapped_env_from_config creates a *fresh* environment.
Do not call on a live environment, after reset, or on a shared training env.

This replaces the fixed governor at the same location in the wrapper chain,
preserving the ordering of normalization, recovery and other wrappers.
It never mutates files, model weights, or configuration objects.
"""
from human_robot_gym.wrappers.human_aware_action_governor import (
    HumanAwareActionGovernor,
)
from human_robot_gym.wrappers.velocity_adaptive_action_governor import (
    VelocityAdaptiveActionGovernor,
)


def install_velocity_adaptive_governor(
    env,
    *,
    policy_dt=0.1,
    extra_response_time=0.1,
    effective_deceleration=2.0,
    max_activation_distance=0.45,
):
    """Replace exactly one existing fixed governor in a fresh wrapper chain.

    Raises before modifying the chain if no governor or multiple governors
    are found. Caller owns and must close the fresh environment on failure.
    """
    links = []
    cursor = env
    visited = set()
    while True:
        if id(cursor) in visited:
            raise RuntimeError("Cyclic environment wrapper chain")
        visited.add(id(cursor))
        if isinstance(cursor, HumanAwareActionGovernor):
            links.append(cursor)
        if not hasattr(cursor, "env"):
            break
        cursor = cursor.env
    if len(links) != 1:
        raise ValueError(
            "Expected exactly one existing governor; found %d" % len(links)
        )
    original = links[0]
    if isinstance(original, VelocityAdaptiveActionGovernor):
        raise ValueError("Environment already uses the adaptive governor")

    names = (
        "activation_distance",
        "hard_distance",
        "prediction_horizon",
        "minimum_scale",
        "max_interventions",
        "include_held_object",
        "object_clearance_margin",
        "include_robot_links",
        "robot_link_clearance_margin",
    )
    kwargs = {name: getattr(original, "_" + name) for name in names}
    kwargs["robot_link_names"] = ",".join(original._robot_link_names)
    replacement = VelocityAdaptiveActionGovernor(
        original.env,
        policy_dt=policy_dt,
        extra_response_time=extra_response_time,
        effective_deceleration=effective_deceleration,
        max_activation_distance=max_activation_distance,
        **kwargs,
    )
    if env is original:
        return replacement
    cursor = env
    while cursor.env is not original:
        cursor = cursor.env
    cursor.env = replacement
    return env
