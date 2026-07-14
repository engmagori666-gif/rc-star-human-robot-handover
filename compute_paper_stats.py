import pandas as pd, os, numpy as np

def load_runs(path):
    dfs = []
    for f in sorted(os.listdir(path)):
        if f.endswith('.csv'):
            dfs.append(pd.read_csv(os.path.join(path, f)))
    return pd.concat(dfs, ignore_index=True)

base = load_runs('/home/human-robot-gym/csv/evaluation/paper_eval_100ep/raw/icra_2024/RHH_SIR')
v2   = load_runs('/home/human-robot-gym/csv/evaluation/reactive_v2_100ep/raw/icra_2024/RHH_SIR_reactive_v2')

for name, df in [('Baseline', base), ('Reactive V2', v2)]:
    n = len(df)
    print(f'{name} (n={n} episodes):')
    print(f'  success_rate      = {df.success.mean()*100:.1f}% +/- {df.success.std()*100:.1f}')
    print(f'  ep_rew            = {df.ep_rew.mean():.1f} +/- {df.ep_rew.std():.1f}')
    print(f'  failsafe/ep       = {df.failsafe_interventions.mean():.1f} +/- {df.failsafe_interventions.std():.1f}')
    print(f'  collisions/ep     = {df.n_collisions.mean():.3f} +/- {df.n_collisions.std():.3f}')
    print(f'  ep_len            = {df.ep_len.mean():.1f} +/- {df.ep_len.std():.1f}')
    print()
