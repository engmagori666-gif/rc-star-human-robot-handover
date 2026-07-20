"""Cross-seed 95% CI aggregation for the RC-STAR vs. baseline RHH-SIR comparison.

Reads the existing per-episode evaluation CSVs (5 seeds x 100 episodes for
each condition), computes one metric mean per seed, then aggregates the 5
per-seed means into a cross-seed mean +/- 95% CI (t-distribution, df=4).

This is read-only analysis over existing evaluation output. It does not run
any environment, training, or evaluation code.

Outputs:
    FINAL_PAPER_PACKAGE/csv_results/evaluation/cross_seed_ci_table.csv
    FINAL_PAPER_PACKAGE/CROSS_SEED_CI_RESULTS.md
"""
import csv
import glob
import math
import os
import statistics as st

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_ROOT = os.path.join(PACKAGE_ROOT, "csv_results", "evaluation")

BASELINE_FILES = sorted(
    glob.glob(
        os.path.join(
            EVAL_ROOT, "paper_eval_100ep", "raw", "icra_2024", "RHH_SIR", "run_*_final.csv"
        )
    )
)
RC_STAR_SEED_DIRS = sorted(
    glob.glob(os.path.join(EVAL_ROOT, "rc_star_matched_seed*_100ep"))
)

CONDITIONS = {
    "baseline_rhh_sir": {
        "label": "Baseline RHH-SIR",
        "files": BASELINE_FILES,
    },
    "rc_star": {
        "label": "RC-STAR",
        "files": [
            glob.glob(os.path.join(seed_dir, "raw", "*.csv"))[0]
            for seed_dir in RC_STAR_SEED_DIRS
        ],
    },
}

METRICS = [
    ("success", "Success rate"),
    ("failsafe_interventions", "Shield interventions / episode"),
    ("n_collisions_human", "Human collisions / episode"),
    ("n_collisions_critical", "Critical collisions / episode"),
    ("ep_rew", "Episode reward"),
]

# Two-tailed 95% critical t-value by degrees of freedom (n_seeds - 1).
T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}


def t_critical(df: int) -> float:
    if df in T_TABLE_95:
        return T_TABLE_95[df]
    if df > 10:
        return 1.96  # normal approximation for large df
    raise ValueError(f"No t-critical value tabulated for df={df}")


def load_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def per_seed_mean(path: str, column: str) -> float:
    rows = load_rows(path)
    return sum(float(row[column]) for row in rows) / len(rows)


def aggregate(seed_means):
    n = len(seed_means)
    mean = sum(seed_means) / n
    std = st.stdev(seed_means) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    margin = t_critical(n - 1) * sem if n > 1 else 0.0
    return {
        "n_seeds": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_margin": margin,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "seed_means": seed_means,
    }


def main():
    for key, cond in CONDITIONS.items():
        if len(cond["files"]) != 5:
            raise RuntimeError(
                f"Expected 5 seed files for {key!r}, found "
                f"{len(cond['files'])}: {cond['files']}"
            )

    results = []
    for column, label in METRICS:
        for key, cond in CONDITIONS.items():
            seed_means = [per_seed_mean(path, column) for path in cond["files"]]
            n_episodes_per_seed = [len(load_rows(path)) for path in cond["files"]]
            agg = aggregate(seed_means)
            results.append(
                {
                    "metric": column,
                    "metric_label": label,
                    "condition": key,
                    "condition_label": cond["label"],
                    "n_seeds": agg["n_seeds"],
                    "n_episodes_per_seed": n_episodes_per_seed[0],
                    "total_episodes": sum(n_episodes_per_seed),
                    "mean": agg["mean"],
                    "std": agg["std"],
                    "sem": agg["sem"],
                    "ci95_margin": agg["ci95_margin"],
                    "ci95_low": agg["ci95_low"],
                    "ci95_high": agg["ci95_high"],
                    "seed_means": ";".join(f"{v:.6f}" for v in agg["seed_means"]),
                    "seed_files": ";".join(
                        os.path.relpath(p, PACKAGE_ROOT) for p in cond["files"]
                    ),
                }
            )

    csv_path = os.path.join(EVAL_ROOT, "cross_seed_ci_table.csv")
    fieldnames = [
        "metric", "metric_label", "condition", "condition_label",
        "n_seeds", "n_episodes_per_seed", "total_episodes",
        "mean", "std", "sem", "ci95_margin", "ci95_low", "ci95_high",
        "seed_means", "seed_files",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Wrote {csv_path}")

    md_path = os.path.join(PACKAGE_ROOT, "CROSS_SEED_CI_RESULTS.md")
    write_markdown(md_path, results)
    print(f"Wrote {md_path}")


def write_markdown(md_path, results):
    by_metric = {}
    for row in results:
        by_metric.setdefault(row["metric"], {})[row["condition"]] = row

    lines = []
    lines.append("# Cross-Seed 95% Confidence Interval Results")
    lines.append("")
    lines.append(
        "Cross-seed aggregation of the matched evaluation: 5 seeds x 100 "
        "episodes for baseline RHH-SIR and 5 seeds x 100 episodes for "
        "RC-STAR. Each seed's 100 episodes are first reduced to one "
        "per-seed mean; the 5 per-seed means are then aggregated into a "
        "cross-seed mean +/- 95% CI (t-distribution, df=4, t=2.776)."
    )
    lines.append("")
    lines.append(
        "This is read-only analysis over existing evaluation CSVs. No "
        "environment, training, or evaluation logic was changed to "
        "produce this table."
    )
    lines.append("")
    lines.append("## Source data")
    lines.append("")
    lines.append(
        "- Baseline RHH-SIR: "
        "`csv_results/evaluation/paper_eval_100ep/raw/icra_2024/RHH_SIR/run_{0..4}_final.csv`"
    )
    lines.append(
        "- RC-STAR: "
        "`csv_results/evaluation/rc_star_matched_seed{100,200,300,400,500}_100ep/raw/*.csv`"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Metric | Baseline RHH-SIR (mean +/- 95% CI) | RC-STAR (mean +/- 95% CI) |"
    )
    lines.append("|---|---:|---:|")
    for column, label in METRICS:
        base = by_metric[column]["baseline_rhh_sir"]
        star = by_metric[column]["rc_star"]
        lines.append(
            f"| {label} "
            f"| {base['mean']:.4f} +/- {base['ci95_margin']:.4f} "
            f"| {star['mean']:.4f} +/- {star['ci95_margin']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-seed means")
    lines.append("")
    for column, label in METRICS:
        lines.append(f"### {label}")
        lines.append("")
        base = by_metric[column]["baseline_rhh_sir"]
        star = by_metric[column]["rc_star"]
        lines.append(
            "- Baseline RHH-SIR per-seed means: "
            + ", ".join(f"{v:.4f}" for v in [float(x) for x in base["seed_means"].split(";")])
        )
        lines.append(
            "- RC-STAR per-seed means: "
            + ", ".join(f"{v:.4f}" for v in [float(x) for x in star["seed_means"].split(";")])
        )
        lines.append(
            f"- Baseline: n_seeds={base['n_seeds']}, std={base['std']:.4f}, "
            f"sem={base['sem']:.4f}, 95% CI=[{base['ci95_low']:.4f}, {base['ci95_high']:.4f}]"
        )
        lines.append(
            f"- RC-STAR: n_seeds={star['n_seeds']}, std={star['std']:.4f}, "
            f"sem={star['sem']:.4f}, 95% CI=[{star['ci95_low']:.4f}, {star['ci95_high']:.4f}]"
        )
        lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "For each condition and metric, each seed's 100-episode raw CSV is "
        "reduced to one scalar (the per-episode mean over that seed's 100 "
        "episodes). The cross-seed mean, sample standard deviation "
        "(n-1 denominator), and standard error of the mean (SEM = std / "
        "sqrt(5)) are computed over the 5 resulting per-seed values. The "
        "95% CI half-width is `t_(0.975, df=4) * SEM` with `t = 2.776`."
    )
    lines.append("")
    lines.append(
        "Generated by `FINAL_PAPER_PACKAGE/source_notes/tools_compute_cross_seed_ci.py`."
    )
    lines.append("")

    with open(md_path, "w") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    main()
