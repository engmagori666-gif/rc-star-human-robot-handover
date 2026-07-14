# V1 Strict Final Results

Strict success requires capture, robot release, and five consecutive policy
steps of stable human ownership.

## Final 100-Episode Evaluation

| Metric | Result |
|---|---:|
| Success | 51/100 |
| Capture | 54/100 |
| Confirmed release | 51/100 |
| Capture-to-release completion | 94.4% |
| Safe success | 51/100 |
| Critical-collision episodes | 24/100 |
| Critical events per episode | 0.29 |
| Timeout | 12/100 |
| Shield interventions per episode | 37.34 |
| Release-supervisor steps | 0 |

The actor completed every successful episode without a critical collision.
Success is approximately 11.1 times the reproducible 4.6% baseline.

## Artifacts

```text
Model:
/home/human-robot-gym/models/icra_2024/
RHH_SIR_reactive_v1_strict_final/run_0/model_final.zip

Training config:
/home/human-robot-gym/human_robot_gym/training/config_icra_2024/
environment_evaluation/training/RHH-SIR-reactive-v1-strict.yaml

Evaluation config:
/home/human-robot-gym/human_robot_gym/training/config_icra_2024/
environment_evaluation/evaluation/RHH-SIR-reactive-v1-strict.yaml

Dataset:
/home/human-robot-gym/datasets/
reactive-human-handover-v1-bc-dagger-v4-strict

Evaluation:
/home/human-robot-gym/csv/evaluation/
reactive_v1_strict_final_100ep/
```
