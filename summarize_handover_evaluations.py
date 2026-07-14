"""Summarize the historical handover evaluations and Ours V1."""
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/home/human-robot-gym/csv/evaluation")
GROUPS = {
    "Baseline": ROOT / "paper_eval_100ep/raw/icra_2024/RHH_SIR",
    "Reactive-only V2": (
        ROOT / "reactive_v2_100ep/raw/icra_2024/RHH_SIR_reactive_v2"
    ),
    "STAR V3": (
        ROOT / "reactive_v3_100ep/raw/icra_2024/RHH_SIR_reactive_v3"
    ),
    "Ours V1": ROOT / "reactive_v1_safe_final_100ep/raw",
}


def summarize(name: str, directory: Path) -> dict:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {directory}")
    frames = []
    per_run_success = []
    for file in files:
        frame = pd.read_csv(file)
        frames.append(frame)
        per_run_success.append(float(frame["success"].mean()))
    data = pd.concat(frames, ignore_index=True)
    critical = data["n_collisions_critical"]
    shield = data["failsafe_interventions"]
    timeout = (
        data["timeout"]
        if "timeout" in data
        else (data["ep_len"] >= 1000).astype(float)
    )
    success = data["success"] == 1
    failed = ~success
    return {
        "method": name,
        "runs_or_shards": len(files),
        "episodes": int(len(data)),
        "successes": int(data["success"].sum()),
        "success_rate": float(data["success"].mean()),
        "per_run_success_mean": float(pd.Series(per_run_success).mean()),
        "per_run_success_std": float(pd.Series(per_run_success).std(ddof=0)),
        "timeout_rate": float(timeout.mean()),
        "mean_return": float(data["ep_rew"].mean()),
        "critical_mean": float(critical.mean()),
        "critical_median": float(critical.median()),
        "critical_p90": float(critical.quantile(0.90)),
        "critical_p95": float(critical.quantile(0.95)),
        "critical_max": float(critical.max()),
        "zero_critical_rate": float((critical == 0).mean()),
        "critical_mean_success": (
            float(critical[success].mean()) if success.any() else None
        ),
        "critical_mean_failure": (
            float(critical[failed].mean()) if failed.any() else None
        ),
        "shield_mean": float(shield.mean()),
        "shield_median": float(shield.median()),
        "shield_p95": float(shield.quantile(0.95)),
        "shield_mean_success": (
            float(shield[success].mean()) if success.any() else None
        ),
        "shield_mean_failure": (
            float(shield[failed].mean()) if failed.any() else None
        ),
        "mean_episode_length": float(data["ep_len"].mean()),
        "files": [str(file) for file in files],
    }


def main() -> None:
    rows = [summarize(name, path) for name, path in GROUPS.items()]
    output = ROOT / "reactive_v1_safe_final_100ep"
    (output / "comparison_summary.json").write_text(
        json.dumps(rows, indent=2)
    )
    table = pd.DataFrame(rows).drop(columns=["files"])
    table.to_csv(output / "comparison_summary.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
