"""Generate all paper figures for STAR paper."""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

OUT_DIR = '/home/human-robot-gym/paper_figures'
os.makedirs(OUT_DIR, exist_ok=True)

BASE = '/home/human-robot-gym'
EVAL = f'{BASE}/csv/evaluation'
RUNS = f'{BASE}/runs/icra_2024'

COLORS = {'Baseline': '#4C72B0', 'STAR': '#DD8452'}
FONT = {'family': 'DejaVu Sans', 'size': 11}
matplotlib.rc('font', **FONT)

METHOD_LABEL = 'STAR'
PAPER_TITLE  = 'Reactive Human Modeling for Safe HRC using RL'

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_tb(run_dir, key):
    files = sorted(glob.glob(run_dir + '/**/*.tfevents.*', recursive=True),
                   key=os.path.getmtime, reverse=True)
    if not files: return None, None
    ea = event_accumulator.EventAccumulator(files[0]); ea.Reload()
    if key not in ea.scalars.Keys(): return None, None
    items = ea.scalars.Items(key)
    return np.array([x.step for x in items]), np.array([x.value for x in items])

def load_eval(path):
    dfs = [pd.read_csv(os.path.join(path, f))
           for f in sorted(os.listdir(path)) if f.endswith('.csv')]
    return pd.concat(dfs, ignore_index=True)

def smooth(y, w=15):
    return np.convolve(y, np.ones(w)/w, mode='same')

def per_seed_stats(path):
    rows = []
    for f in sorted(os.listdir(path)):
        if f.endswith('.csv'):
            d = pd.read_csv(os.path.join(path, f))
            rows.append({'sr': d.success.mean()*100,
                         'fs': d.failsafe_interventions.mean(),
                         'co': d.n_collisions.mean()})
    return pd.DataFrame(rows)

# ── Load eval data ─────────────────────────────────────────────────────────────
base_path = f'{EVAL}/paper_eval_100ep/raw/icra_2024/RHH_SIR'
star_path  = f'{EVAL}/reactive_v3_100ep/raw/icra_2024/RHH_SIR_reactive_v3'
base_df    = load_eval(base_path)
star_df    = load_eval(star_path)
ps_base    = per_seed_stats(base_path)
ps_star    = per_seed_stats(star_path)

# ── Figure 1: Training curves ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle('Training Curves: Baseline vs STAR', fontsize=12, fontweight='bold', y=1.01)

tb_configs = {
    'Baseline':    ('RHH_SIR',              COLORS['Baseline']),
    METHOD_LABEL:  ('RHH_SIR_reactive_v3',  COLORS['STAR']),
}
for metric, ax, ylabel, title in [
    ('rollout/ep_rew_mean',             axes[0], 'Mean Episode Reward',       'Episode Reward'),
    ('rollout/failsafe_interventions',  axes[1], 'Failsafe Interventions/ep', 'Failsafe Interventions'),
]:
    for name, (group, color) in tb_configs.items():
        group_dir = os.path.join(RUNS, group)
        if not os.path.exists(group_dir): continue
        all_steps, all_vals = [], []
        for run in sorted(os.listdir(group_dir)):
            steps, vals = load_tb(os.path.join(group_dir, run), metric)
            if steps is not None:
                all_steps.append(steps); all_vals.append(vals)
        if not all_steps: continue
        min_len = min(len(s) for s in all_steps)
        arr   = np.array([v[:min_len] for v in all_vals])
        steps = all_steps[0][:min_len]
        mean  = smooth(arr.mean(axis=0))
        lo    = smooth(arr.min(axis=0))
        hi    = smooth(arr.max(axis=0))
        ax.plot(steps, mean, color=color, linewidth=2, label=name)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel('Training Steps'); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_: f'{x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig1_training_curves.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 1 saved.')

# ── Figure 2: Evaluation bar chart (Baseline vs STAR) ─────────────────────────
metrics_eval = [
    ('sr', 'Success Rate (%)'),
    ('fs', 'Failsafe / Episode'),
    ('co', 'Collisions / Episode'),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle(f'Evaluation Results — Baseline vs {METHOD_LABEL}', fontsize=12, fontweight='bold')

x = np.array([0, 1])
labels = ['Baseline\n(ICRA 2024)', f'{METHOD_LABEL}']
bar_colors = [COLORS['Baseline'], COLORS['STAR']]
w = 0.5

for ax, (key, ylabel) in zip(axes, metrics_eval):
    bm = ps_base[key].mean()
    sm = ps_star[key].mean()
    vals = [bm, sm]
    bars = ax.bar(x, vals, w, color=bar_colors, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig2_eval_bars.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 2 saved.')

# ── Figure 3: Safety improvement ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
categories = ['Failsafe\nInterventions/ep', 'Collisions/ep']
base_vals  = [base_df.failsafe_interventions.mean(), base_df.n_collisions.mean()]
star_vals  = [star_df.failsafe_interventions.mean(), star_df.n_collisions.mean()]

xc = np.arange(len(categories))
w  = 0.35
ax.bar(xc - w/2, base_vals, w, label='Baseline (ICRA 2024)', color=COLORS['Baseline'], edgecolor='white')
ax.bar(xc + w/2, star_vals, w, label=METHOD_LABEL,           color=COLORS['STAR'],     edgecolor='white')

y_max = max(base_vals) * 1.1
ax.set_ylim(0, y_max)

ax.set_xticks(xc); ax.set_xticklabels(categories, fontsize=11)
ax.set_ylabel('Mean count per episode', fontsize=11)
ax.set_title('Safety Improvement: STAR vs Baseline (ICRA 2024)', fontweight='bold', pad=14)
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_safety_improvement.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 3 saved.')

# ── Figure 4: Per-seed scatter ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f'Per-Seed Variance: Baseline vs {METHOD_LABEL}', fontsize=12, fontweight='bold')

for ax, metric, ylabel in [
    (axes[0], 'fs', 'Failsafe Interventions / ep'),
    (axes[1], 'co', 'Collisions / ep'),
]:
    ax.scatter(np.zeros(len(ps_base)), ps_base[metric], color=COLORS['Baseline'], s=80, zorder=3, label='Baseline')
    ax.scatter(np.ones(len(ps_star)),  ps_star[metric],  color=COLORS['STAR'],    s=80, zorder=3, label=METHOD_LABEL)
    ax.errorbar([0], [ps_base[metric].mean()], yerr=[ps_base[metric].std()],
                fmt='D', color=COLORS['Baseline'], markersize=10, capsize=8, lw=2)
    ax.errorbar([1], [ps_star[metric].mean()],  yerr=[ps_star[metric].std()],
                fmt='D', color=COLORS['STAR'],    markersize=10, capsize=8, lw=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Baseline', METHOD_LABEL])
    ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=0.3); ax.legend()

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_per_seed.png', dpi=150, bbox_inches='tight')
plt.close(); print('Fig 4 saved.')

print(f'\nAll figures saved to {OUT_DIR}')
