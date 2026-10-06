"""Plot the frozen Holmes confidence reliability bins from compact diagnostics."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'jev_full_holmes_diagnostics.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True,
                         layout='constrained')
for ax, (name, label, color) in zip(axes, [
        ('platt', 'Platt', '#2471a3'),
        ('confidence_lora', 'Confidence LoRA', '#c0392b')]):
    result = data['holmes_reliability'][name]
    bins = [b for b in result['bins'] if b['count'] >= 20]
    ax.plot([0, 1], [0, 1], '--', color='0.5', linewidth=1,
            label='Perfect calibration')
    ax.plot([b['mean_q'] for b in bins], [b['accuracy'] for b in bins],
            color=color, linewidth=1.5, alpha=0.7)
    ax.scatter([b['mean_q'] for b in bins], [b['accuracy'] for b in bins],
               s=[max(18, b['count'] * 0.45) for b in bins],
               color=color, edgecolors='white', linewidths=0.7, zorder=3)
    for b in bins:
        ax.annotate(str(b['count']), (b['mean_q'], b['accuracy']),
                    xytext=(4, 4), textcoords='offset points', fontsize=7)
    ax.set_title(f"{label}  |  ECE = {result['ece_10_equal_width']:.3f}")
    ax.set_xlabel('Mean predicted confidence')
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.18)
axes[0].set_ylabel('Observed accuracy')
axes[0].legend(loc='upper left', fontsize=8)
fig.suptitle('Holmes reliability, 1,837 questions (bins with n >= 20; labels = n)',
             fontsize=11)
fig.savefig(ROOT / 'jev_full_holmes_reliability.png', dpi=180)
