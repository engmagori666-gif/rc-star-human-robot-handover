import pandas as pd, os

EVAL = '/home/human-robot-gym/csv/evaluation'
base_path = f'{EVAL}/paper_eval_100ep/raw/icra_2024/RHH_SIR'
star_path = f'{EVAL}/reactive_v3_100ep/raw/icra_2024/RHH_SIR_reactive_v3'

def load(path):
    dfs = [pd.read_csv(os.path.join(path, f)) for f in sorted(os.listdir(path)) if f.endswith('.csv')]
    return pd.concat(dfs, ignore_index=True)

def per_seed(path):
    rows = []
    for f in sorted(os.listdir(path)):
        if f.endswith('.csv'):
            d = pd.read_csv(os.path.join(path, f))
            rows.append({
                'sr': d.success.mean(), 'fs': d.failsafe_interventions.mean(),
                'co': d.n_collisions.mean(), 'hc': d.n_collisions_human.mean(),
                'cc': d.n_collisions_critical.mean()
            })
    return pd.DataFrame(rows)

base = load(base_path); star = load(star_path)
ps_base = per_seed(base_path); ps_star = per_seed(star_path)

for name, df, ps in [('Baseline', base, ps_base), ('STAR(V3)', star, ps_star)]:
    print(f'--- {name} ---')
    print(f'success: {df.success.mean():.3f} +/- {ps.sr.std():.3f}')
    print(f'shield/ep: {df.failsafe_interventions.mean():.2f} +/- {ps.fs.std():.2f}')
    print(f'n_collisions/ep: {df.n_collisions.mean():.2f} +/- {ps.co.std():.2f}')
    print(f'human_coll/ep: {df.n_collisions_human.mean():.3f} +/- {ps.hc.std():.3f}')
    print(f'critical_coll/ep: {df.n_collisions_critical.mean():.2f} +/- {ps.cc.std():.2f}')
    print()

print('per-seed STAR(V3):')
print(ps_star)
print('per-seed Baseline:')
print(ps_base)
