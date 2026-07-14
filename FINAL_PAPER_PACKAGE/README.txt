RC-STAR Final Paper Package
===========================

This folder is a clean package for the conference-paper materials.
The original research workspace is intentionally left untouched.

Folders
-------
paper/
  Final/revised manuscript files.

figures/
  Figure images used in the paper.

csv_results/
  Evaluation CSVs and summary statistics used for the reported paper table.
  Key outputs:
    - evaluation/paper_eval_100ep/raw/icra_2024/RHH_SIR/
      Baseline RHH-SIR evaluation CSVs.
    - evaluation/rc_star_matched_500ep/raw/combined_final.csv
      Final RC-STAR matched 500-episode evaluation CSV.
    - evaluation/rc_star_matched_500ep/stats.csv
      Final RC-STAR summary statistics.

source_notes/
  Small helper/stat scripts and selected reproducibility notes, if present.

Important Results
-----------------
Baseline RHH-SIR:
  Success rate: 4.6%
  Shield activations / episode: 21.8
  Human collisions / episode: 0.50
  Critical collisions / episode: 6.04

RC-STAR:
  Success rate: 72.6%
  Shield activations / episode: 34.98
  Human collisions / episode: 0.00
  Critical collisions / episode: 0.228

Recommended Use
---------------
Use this folder for submission, sharing with your advisor, and paper revision.
Keep the full human-robot-gym workspace for reproducibility and future experiments.
