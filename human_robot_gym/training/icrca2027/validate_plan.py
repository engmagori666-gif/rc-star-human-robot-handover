"""Read-only static validator for the ICRCA experiment plan.

No MuJoCo, policy, datasets, or experiment outputs are opened or changed.
Run manually if desired: python -m human_robot_gym.training.icrca2027.validate_plan
"""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[3]
FILES = (
    "human_robot_gym/wrappers/velocity_adaptive_action_governor.py",
    "human_robot_gym/training/icrca2027/governor_adapter.py",
    "tests/test_velocity_adaptive_action_governor.py",
)
FORBIDDEN = ("model.learn(", "shutil.rmtree(", "os.remove(", "unlink(")


def validate():
    for relative in FILES:
        source = (ROOT / relative).read_text(encoding="utf-8")
        ast.parse(source, filename=relative)
        for forbidden in FORBIDDEN:
            if forbidden in source:
                raise ValueError("Potential destructive/training call in " + relative)
    plan = (ROOT / "human_robot_gym/training/icrca2027/evaluation_plan.yaml").read_text(
        encoding="utf-8"
    )
    if "run_training: false" not in plan or "overwrite_existing: false" not in plan:
        raise ValueError("Evaluation plan is missing required preservation flags")
    print("ICRCA plan static checks passed (not a simulator test)")


if __name__ == "__main__":
    validate()
