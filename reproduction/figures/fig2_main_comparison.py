"""
Generate Figure 2: Main Comparison (Boxplot)

This script generates the main comparison figure showing C-Index boxplots
for baseline methods vs DeepSurv vs PiDeeL.
"""
import sys
from pathlib import Path

# Add repository root to path
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import config as repo_config
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

concordence = {}

# Load baseline results
baseline_tasks = ["baseline/coxph", "baseline/cwgb", "baseline/rf"]
for task in baseline_tasks:
    concordence[task] = []
    log_path = REPO_ROOT / "logs" / task / "c_indices.txt"
    with open(log_path, "r") as f:
        for lines in f:
            concordence[task].append(float(lines.strip()))

# Load DeepHit results (from test_logs/test29)
tasks = ["4layer/no_pathway_dh"]
for task in tasks:
    concordence[task] = []
    taske = task[:-3]
    # Note: test29 structure might differ slightly, checking path
    log_path = REPO_ROOT / "test_logs" / "test29" / taske / "c_indices.txt"
    with open(log_path, "r") as f:
        for lines in f:
            concordence[task].append(float(lines.strip()))

# Load PC-Hazard results (from test_logs/test30)
tasks = ["4layer/no_pathway_pc"]
for task in tasks:
    concordence[task] = []
    taske = task[:-3]
    log_path = REPO_ROOT / "test_logs" / "test30" / taske / "c_indices.txt"
    with open(log_path, "r") as f:
        for lines in f:
            concordence[task].append(float(lines.strip()))

# Load DeepSurv results (no_pathway)
tasks = ["4layer/no_pathway"]
for task in tasks:
    concordence[task] = []
    log_path = REPO_ROOT / "logs" / task / "c_indices.txt"
    with open(log_path, "r") as f:
        for lines in f:
            concordence[task].append(float(lines.strip()))

# Load PiDeeL results (pathway)
tasks = ["4layer/pathway"]
for task in tasks:
    concordence[task] = []
    log_path = REPO_ROOT / "logs" / task / "c_indices.txt"
    with open(log_path, "r") as f:
        for lines in f:
            concordence[task].append(float(lines.strip()))

# Color palette
color_palette_dct = {
    "baseline/coxph": '#808080',
    "baseline/rf": '#808080',
    "baseline/cwgb": '#808080',
    "baseline/ipc": '#808080',
    "2layer/no_pathway": '#008080',
    "3layer/no_pathway": '#008080',
    "4layer/no_pathway": '#008080',
    "2layer/pathway": '#2f4f4f',
    "3layer/pathway": '#2f4f4f',
    "4layer/pathway": '#2f4f4f',
    "2layer/no_pathway_dh": '#008080',
    "3layer/no_pathway_dh": '#008080',
    "4layer/no_pathway_dh": '#008080',
    "2layer/pathway_dh": '#2f4f4f',
    "3layer/pathway_dh": '#2f4f4f',
    "4layer/pathway_dh": '#2f4f4f',
    "2layer/no_pathway_pc": '#008080',
    "3layer/no_pathway_pc": '#008080',
    "4layer/no_pathway_pc": '#008080',
    "2layer/pathway_pc": '#2f4f4f',
    "3layer/pathway_pc": '#2f4f4f',
    "4layer/pathway_pc": '#2f4f4f'
}

# Tasks to plot
tasks = ["baseline/coxph", "baseline/cwgb", "baseline/rf", 
         "4layer/no_pathway_dh", "4layer/no_pathway_pc", 
         "4layer/no_pathway", "4layer/pathway"]

color_palette_dct = {name: color_palette_dct[name] for name in tasks}

# Create figure
fig, ax = plt.subplots()
bp = ax.boxplot(concordence.values(), patch_artist=True, showfliers=True)
ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
for idx, model in enumerate(tasks):
    bp['boxes'][idx].set(color=color_palette_dct[model])
    bp['boxes'][idx].set(facecolor=color_palette_dct[model])

# Labels
display_names = ["Cox-PH", "CWGB", "RSF", "DeepHit", "PC-Hazard", "DeepSurv", "PiDeeL"]
ax.set_xticklabels(display_names, fontsize=12, rotation=90)
ax.set_ylabel("C-Index", fontsize=12)
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
ax.set_ylim(0.5, 0.8)

fontP = FontProperties()
fontP.set_size('x-small')

# Draw separator lines
ax.axvline(x=3.5, color='black', linestyle='--', linewidth=1)
ax.axvline(x=6.5, color='black', linestyle='--', linewidth=1)

plt.tight_layout()

# Save figure
output_dir = REPO_ROOT / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "fig2_main_comparison.png")
fig.savefig(output_dir / "fig2_main_comparison.pdf")

print(f"Figure saved to {output_dir / 'fig2_main_comparison.png'}")
print(f"Figure saved to {output_dir / 'fig2_main_comparison.pdf'}")
