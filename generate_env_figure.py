"""Generate environment figure in grid style matching human-robot-gym paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

OUT   = '/home/human-robot-gym/paper_figures/fig0_environment.png'
IMG1  = '/home/human-robot-gym/hrgym_demo.png'
IMG2  = '/home/human-robot-gym/hrgym_demo_failsafe.png'

img1 = mpimg.imread(IMG1)
img2 = mpimg.imread(IMG2)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
fig.patch.set_facecolor('white')

for ax, img, label in zip(axes, [img1, img2],
                           ['Robot-to-human\nhandover', 'Failsafe\nintervention']):
    # Black border panel like the reference style
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)
    ax.set_xlabel(label, fontsize=13, fontweight='bold',
                  fontfamily='monospace', labelpad=8, color='black')

plt.suptitle('ReactiveHumanHandoverCart simulation environment',
             fontsize=11, y=1.01, color='#333333')

fig.text(0.5, -0.04,
         'Fig. 1.   ReactiveHumanHandoverCart presents the robot-to-human handover task\n'
         '(left: normal operation; right: safety shield intervention active).',
         ha='center', fontsize=9.5, fontfamily='monospace', color='#222222')

plt.tight_layout(pad=0.8)
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Environment figure saved to {OUT}')
