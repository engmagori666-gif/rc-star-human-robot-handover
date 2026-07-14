"""Generate architecture diagram: RL agent + safety shield + reactive human model (STAR)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = '/home/human-robot-gym/paper_figures/fig5_architecture.png'

fig, ax = plt.subplots(figsize=(6.4, 7.4))
ax.set_xlim(-2.6, 10)
ax.set_ylim(-2.0, 11.2)
ax.axis('off')

def box(cx, cy, w, h, text, fc='#EFEFEF', ec='black', fontsize=9.5,
        fontweight='normal', lw=1.0, textcolor='black'):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle='round,pad=0.02,rounding_size=0.06',
                           facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, color=textcolor, zorder=4)

def arrow(p1, p2, ls='-', lw=1.2, color='black'):
    ax.annotate('', xy=p2, xytext=p1,
                 arrowprops=dict(arrowstyle='-|>', lw=lw, color=color,
                                  linestyle=ls, shrinkA=0, shrinkB=0),
                 zorder=2)

# ── Legend ──────────────────────────────────────────────────────────────────
arrow((5.6, 11.0), (6.4, 11.0))
ax.text(6.55, 11.0, 'High frequency (e.g. 250 Hz)', va='center', fontsize=8.5)
arrow((5.6, 10.55), (6.4, 10.55), ls=(0, (4, 3)))
ax.text(6.55, 10.55, 'Low frequency (e.g. 5 Hz)', va='center', fontsize=8.5)

# ── Safety shield (yellow shield shape) ────────────────────────────────────
shield = Polygon([(2.9, 9.9), (9.7, 9.9), (9.7, 3.0), (6.3, 0.9), (2.9, 3.0)],
                  closed=True, facecolor='#F2C40C', edgecolor='black',
                  linewidth=1.2, zorder=1)
ax.add_patch(shield)
ax.text(3.1, 9.45, 'Safety shield', fontsize=12, fontweight='bold', color='#222222')

# Inside-shield boxes
box(6.3, 8.7, 3.2, 0.6, 'Long-term planner')
box(7.6, 7.4, 2.8, 0.6, 'Failsafe planner')
box(7.6, 6.1, 2.0, 0.6, 'Time delay')

# Trajectory labels
ax.text(4.4, 5.0, 'Intended\ntrajectory', ha='right', fontsize=9, color='#444444')
ax.text(8.3, 5.0, 'Failsafe\ntrajectory', ha='left', fontsize=9, color='#444444')

# Verify box
box(6.3, 3.5, 3.6, 0.9, 'Verify', fontsize=10, fontweight='bold')
ax.text(5.0, 3.5, '✓', fontsize=13, color='green', ha='center', va='center', zorder=4)
ax.text(7.6, 3.5, '✗', fontsize=13, color='red', ha='center', va='center', zorder=4)

# Internal arrows
arrow((6.3, 8.4), (7.6, 7.7))           # long-term -> failsafe planner
arrow((7.6, 7.1), (7.6, 6.4))           # failsafe planner -> time delay
arrow((4.6, 8.7), (4.6, 3.95), color='black')  # long-term -> verify (intended)
ax.plot([6.3, 4.6], [8.7, 8.7], color='black', lw=1.2, zorder=2)
arrow((8.1, 5.8), (8.1, 3.95), color='black')  # time delay -> verify (failsafe)
ax.plot([7.6, 8.1], [5.8, 5.8], color='black', lw=1.2, zorder=2)

# ── Agent and Environment ──────────────────────────────────────────────────
box(0.9, 4.8, 1.6, 0.7, 'Agent', fontsize=10, fontweight='bold')
box(5.5, -0.9, 2.2, 0.7, 'Environment', fontsize=10, fontweight='bold')

# State / reward / goal (dashed, env -> agent)
arrow((5.0, -0.9), (0.9, 4.3), ls=(0, (4, 3)))
ax.text(0.3, 1.95, 'State $s_i$', fontsize=8.5, color='#444444')
ax.text(0.3, 1.6, 'Reward $r_i$', fontsize=8.5, color='#444444')
ax.text(0.3, 1.25, 'Episode goal $g$', fontsize=8.5, color='#444444')
ax.text(0.3, 0.75, r'$-\lambda$ if shield' '\nintervenes', fontsize=7.8,
        color='#C0504D', fontweight='bold')

# RSI curriculum annotation
box(-1.3, 4.8, 2.2, 1.4,
    'RSI curriculum:\n' r'$\rho$: 0.9 $\to$ 0.0',
    fc='#FCE4D6', ec='#C0504D', lw=1.6, fontsize=8.5, fontweight='bold')
arrow((-0.2, 4.8), (0.1, 4.8), color='#C0504D', lw=1.4)

# Agent -> long-term planner (solid, high freq)
arrow((1.7, 5.1), (4.55, 8.55))

# Intermediate goal feedback (dashed, shield -> agent, over the top)
ax.plot([2.9, 2.9], [9.9, 10.6], color='black', lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.plot([0.9, 2.9], [10.6, 10.6], color='black', lw=1.0, ls=(0, (4, 3)), zorder=2)
arrow((0.9, 10.6), (0.9, 5.15), ls=(0, (4, 3)))
ax.text(1.9, 10.75, r'Intermediate goal $\bar{g}_{i+1}$', fontsize=8.5, ha='center')

# Safe action -> environment
arrow((6.3, 3.05), (6.0, -0.55))
ax.text(4.5, 1.1, 'Safe action', fontsize=9, color='#444444')

# ── Reactive human model (new, STAR) ───────────────────────────────────────
box(8.9, -0.9, 2.3, 0.9, 'Reactive\nhuman model',
    fc='#FCE4D6', ec='#C0504D', lw=1.6, fontsize=9, fontweight='bold')
arrow((7.75, -0.7), (6.65, -0.7), color='#C0504D', lw=1.4)
arrow((6.65, -1.1), (7.75, -1.1), color='#C0504D', lw=1.4)
ax.text(7.2, -1.7, 'human state / reaction', fontsize=7.8, color='#C0504D', ha='center')

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Architecture figure saved to {OUT}')
