#!/usr/bin/env bash
set -euo pipefail

cd /home/human-robot-gym

PYTHON=/opt/conda/envs/hrgym/bin/python

echo "START $(date -Is)"

for seed in 100 200 300 400 500; do
  group="rc_star_matched_seed${seed}_100ep"
  echo "=== ${group} === $(date -Is)"
  "${PYTHON}" evaluate_v1_sequential.py \
    --group "${group}" \
    --config-name RHH-SIR-reactive-v1-final \
    --episodes 100 \
    --eval-seed "${seed}" \
    --runs icra_2024/RHH_SIR_reactive_v1_final/run_0
done

"${PYTHON}" combine_v1_eval_shards.py \
  --output-group rc_star_matched_500ep \
  --input-groups \
    rc_star_matched_seed100_100ep \
    rc_star_matched_seed200_100ep \
    rc_star_matched_seed300_100ep \
    rc_star_matched_seed400_100ep \
    rc_star_matched_seed500_100ep

echo "DONE $(date -Is)"
