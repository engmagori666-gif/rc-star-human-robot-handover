"""This package defines all environments and necessary gym wrappers."""
from robosuite.environments.base import REGISTERED_ENVS, MujocoEnv  # noqa: F401

from human_robot_gym.environments.manipulation.human_env import HumanEnv  # noqa: F401
from human_robot_gym.environments.manipulation.reach_human_env import ReachHuman  # noqa: F401
from human_robot_gym.environments.manipulation.reach_human_cartesian_env import ReachHumanCart  # noqa: F401
from human_robot_gym.environments.manipulation.pick_place_human_cartesian_env import PickPlaceHumanCart  # noqa: F401
from human_robot_gym.environments.manipulation.human_object_inspection_cartesian_env import (  # noqa: F401
    HumanObjectInspectionCart
)
from human_robot_gym.environments.manipulation.robot_human_handover_cartesian_env import (  # noqa: F401
    RobotHumanHandoverCart
)
from human_robot_gym.environments.manipulation.human_robot_handover_cartesian_env import (  # noqa: F401
    HumanRobotHandoverCart
)
from human_robot_gym.environments.manipulation.collaborative_lifting_cartesian_env import (  # noqa: F401
    CollaborativeLiftingCart
)
from human_robot_gym.environments.manipulation.collaborative_stacking_cartesian_env import (  # noqa: F401
    CollaborativeStackingCart
)
from human_robot_gym.environments.manipulation.collaborative_hammering_cartesian_env import (  # noqa: F401
    CollaborativeHammeringCart
)
from human_robot_gym.environments.manipulation.reactive_handover_env import (  # noqa: F401
    ReactiveHumanHandoverCart
)
from human_robot_gym.environments.manipulation.scripted_handover_env import (  # noqa: F401
    ScriptedHandoverCartV2
)
from human_robot_gym.environments.manipulation.transparent_handover_env import (  # noqa: F401
    TransparentHandoverCart
)

from human_robot_gym.environments.manipulation.ours_v1_handover_env import (  # noqa: F401
    ReactiveHumanHandoverCartV1
)

ALL_ENVIRONMENTS = REGISTERED_ENVS.keys()
