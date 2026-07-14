import csv
import glob
import os
import statistics as st


BASE = "/home/multi-i9/human-robot-gym/csv/evaluation"
PATHS = {
    "baseline_100ep": BASE + "/paper_eval_100ep/raw/icra_2024/RHH_SIR/run_*_final.csv",
    "reactive_matched_500ep": BASE + "/rc_star_matched_500ep/raw/*.csv",
}
COLS = [
    "success",
    "failsafe_interventions",
    "n_collisions_human",
    "n_collisions_critical",
    "ep_rew",
]
LABELS = {
    "success": "Success rate",
    "failsafe_interventions": "Shield activations / ep",
    "n_collisions_human": "Human collisions / ep",
    "n_collisions_critical": "Critical collisions / ep",
    "ep_rew": "Episode reward",
}


def mean(values):
    return sum(values) / len(values)


def load_seed_mean(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "file": os.path.basename(path),
        "n": len(rows),
        **{col: mean([float(row[col]) for row in rows]) for col in COLS},
    }


summary = {}
for name, pattern in PATHS.items():
    files = sorted(glob.glob(pattern))
    print(f"\n== {name}: {len(files)} files ==")
    seed_rows = [load_seed_mean(path) for path in files]
    for row in seed_rows:
        print(
            f"{row['file']}: n={row['n']} "
            f"success={row['success']:.3f} "
            f"failsafe={row['failsafe_interventions']:.2f} "
            f"human={row['n_collisions_human']:.2f} "
            f"critical={row['n_collisions_critical']:.2f} "
            f"rew={row['ep_rew']:.1f}"
        )

    summary[name] = {}
    print("MEAN +/- STD")
    for col in COLS:
        vals = [row[col] for row in seed_rows]
        mu = mean(vals)
        sigma = st.pstdev(vals) if len(vals) > 1 else 0.0
        summary[name][col] = (mu, sigma)
        print(f"{LABELS[col]}: {mu:.4f} +/- {sigma:.4f}")

print("\nFINAL TABLE")
print("Metric | Baseline | Reactive matched")
for col in COLS:
    b = summary["baseline_100ep"][col]
    r = summary["reactive_matched_500ep"][col]
    print(f"{LABELS[col]} | {b[0]:.3f} +/- {b[1]:.3f} | {r[0]:.3f} +/- {r[1]:.3f}")
