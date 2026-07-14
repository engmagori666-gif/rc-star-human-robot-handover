# RC-STAR Reproducibility Notes

This note records the final experiment state used for the RC-STAR conference
paper results.

## Project

- Repository/workspace: `/home/multi-i9/human-robot-gym`
- Clean paper package: `/home/multi-i9/human-robot-gym/FINAL_PAPER_PACKAGE`
- Main framework: Human-Robot Gym on MuJoCo 2.1
- Python environment: Conda `hrgym`, Python 3.8
- RL library: Stable-Baselines3 1.5.0
- Safety controller: SaRA PFL shield

## Final Paper Files

- Revised DOCX:
  `FINAL_PAPER_PACKAGE/paper/Emanuel Magori-Conference Paper - RC-STAR Updated - Codex Revised.docx`
- PDF:
  `FINAL_PAPER_PACKAGE/paper/STAR_paper.pdf`
- Figures:
  `FINAL_PAPER_PACKAGE/figures/paper_figures/`

## Key Source Files

- Reactive/RC-STAR handover environment:
  `human_robot_gym/environments/manipulation/reactive_handover_env.py`
- ICRA baseline config:
  `human_robot_gym/training/config_icra_2024/environment_evaluation/training/RHH-SIR.yaml`
- Reactive config:
  `human_robot_gym/training/config_icra_2024/environment_evaluation/training/RHH-SIR-reactive.yaml`
- Evaluation configs:
  `human_robot_gym/training/config_icra_2024/environment_evaluation/evaluation/RHH.yaml`
  `human_robot_gym/training/config_icra_2024/environment_evaluation/evaluation/RHH-reactive.yaml`

## Final Datasets and Evaluation CSVs

Baseline evaluation:

- Raw CSVs:
  `csv/evaluation/paper_eval_100ep/raw/icra_2024/RHH_SIR/run_*_final.csv`
- Evaluation count:
  5 seeds x 100 episodes = 500 episodes

RC-STAR matched evaluation:

- Combined raw CSV:
  `csv/evaluation/rc_star_matched_500ep/raw/combined_final.csv`
- Summary statistics:
  `csv/evaluation/rc_star_matched_500ep/stats.csv`
- Evaluation count:
  500 episodes

## Final Result Table

| Metric | Baseline RHH-SIR | RC-STAR |
|---|---:|---:|
| Success rate | 4.6% | 72.6% |
| Shield activations / episode | 21.80 | 34.98 |
| Human collisions / episode | 0.50 | 0.00 |
| Critical collisions / episode | 6.038 | 0.228 |
| Episode reward | -810.40 | 21.06 |

## Interpretation

The baseline frequently timed out and rarely completed release-confirmed
handover. RC-STAR substantially improved handover completion while reducing
human and critical collisions. The higher shield-intervention count in RC-STAR
is expected: the robot actually enters the close handover interaction zone,
whereas baseline failures often remain far from successful transfer.

## Notes

- The first simple reactive variant with very small nominal motion
  (`speed_min = 0.003`) produced poor learned-policy success because it created
  a coordination deadlock.
- The final promoted method uses a nonzero nominal cooperative reach plus
  robust capture and release-confirmed success.
- The clean submission/share folder is `FINAL_PAPER_PACKAGE`; the full project
  workspace should be kept for reproducibility.
