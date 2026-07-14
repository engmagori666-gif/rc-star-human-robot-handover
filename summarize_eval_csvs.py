"""Print compact handover metrics for one or more evaluation CSVs."""
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    for csv_path in args.csv:
        path = Path(csv_path)
        frame = pd.read_csv(path)
        if args.limit is not None:
            frame = frame.iloc[:args.limit].copy()
        success = frame["success"].astype(float)
        critical = frame["n_collisions_critical"].astype(float)
        summary = {
            "label": path.stem,
            "episodes": len(frame),
            "successes": int(success.sum()),
            "success_rate": float(success.mean()),
            "capture_rate": float(
                frame["v1_object_captured"].astype(float).mean()
            ),
            "safe_success_rate": float(
                ((success > 0) & (critical == 0)).mean()
            ),
            "critical_episode_rate": float((critical > 0).mean()),
            "critical_mean": float(critical.mean()),
            "static_mean": float(
                frame["n_collisions_static"].astype(float).mean()
            ),
            "timeout_rate": float(
                frame["timeout"].astype(float).mean()
            ),
            "shield_mean": float(
                frame["failsafe_interventions"].astype(float).mean()
            ),
            "episode_length_mean": float(
                frame["ep_len"].astype(float).mean()
            ),
        }
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
