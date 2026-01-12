import sys
from pathlib import Path

# Setup paths
FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np
import pdb
tasks = ["2layer/no_pathway", "2layer/pathway", "3layer/no_pathway", "3layer/pathway","4layer/no_pathway", "4layer/pathway"]

concordence50 = {}
for task in tasks:
    concordence50[task] = []

tasks = ["2layer/no_pathway","2layer/pathway", "3layer/no_pathway","3layer/pathway","4layer/no_pathway" ,"4layer/pathway"]
for task in tasks:
    with open (REPO_ROOT / f"test_logs/test28/{task}/c_indices_50.txt", "r") as f:
        cs = f.readlines()
        cs = [float(c.strip()) for c in cs]
        cs = np.array(cs)
        cs = cs.reshape(-1, 15)
        cs = np.mean(cs, axis=0)
        concordence50[task] = cs.tolist()

concordence100 = {}
for task in tasks:
    concordence100[task] = []

tasks = ["2layer/no_pathway","2layer/pathway", "3layer/no_pathway","3layer/pathway","4layer/no_pathway" ,"4layer/pathway"]
for task in tasks:
    with open (REPO_ROOT / f"test_logs/test28/{task}/c_indices_100.txt", "r") as f:
        cs = f.readlines()
        cs = [float(c.strip()) for c in cs]
        cs = np.array(cs)
        cs = cs.reshape(-1, 15)
        cs = np.mean(cs, axis=0)
        concordence100[task] = cs.tolist()
concordence200 = {}
for task in tasks:
    concordence200[task] = []

tasks = ["2layer/no_pathway","2layer/pathway", "3layer/no_pathway","3layer/pathway","4layer/no_pathway" ,"4layer/pathway"]
for task in tasks:
    with open (REPO_ROOT / f"test_logs/test28/{task}/c_indices_200.txt", "r") as f:
        cs = f.readlines()
        cs = [float(c.strip()) for c in cs]
        cs = np.array(cs)
        cs = cs.reshape(-1, 15)
        cs = np.mean(cs, axis=0)
        concordence200[task] = cs.tolist()

color_palette_dct = {
"2layer/no_pathway": '#008080',
"2layer/random": '#008080',
"2layer/random2": '#008080',
"2layer/pathway": '#2f4f4f',
"3layer/no_pathway": '#008080',
"3layer/random": '#008080',
"3layer/random2": '#008080',
"3layer/pathway": '#2f4f4f',
"4layer/no_pathway": '#008080',
"4layer/random": '#008080',
"4layer/random2": '#008080',
"4layer/pathway": '#2f4f4f'}
tasks = ["2layer/no_pathway", "2layer/pathway","3layer/no_pathway", "3layer/pathway","4layer/no_pathway", "4layer/pathway"]

fig, (ax3, ax2, ax1) = plt.subplots(1, 3,figsize=(15,5))

bp1 = ax1.boxplot(concordence50.values(), patch_artist=True, showfliers=True)
color_palette_dct = {name:color_palette_dct[name] for name in tasks}
ax1.set_xticks([1,2,3,4,5,6])
for idx, model in enumerate(tasks):
    bp1['boxes'][idx].set(color=color_palette_dct[model])
    bp1['boxes'][idx].set(facecolor=color_palette_dct[model])

tasks = ["DeepSurv", "PiDeeL","DeepSurv", "PiDeeL","DeepSurv","PiDeeL"]

ax1.set_xticklabels(tasks,fontsize=12, rotation=90)
ax1.set_ylabel("C-Index",fontsize=12)
ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
ax1.set_ylim(0.4, 0.8)

fontP = FontProperties()
fontP.set_size('x-small')
#write 2 layer, 3 layer, 4 layer in the plot 
ax1.text(0.175, 0.9, '2-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12)
ax1.text(0.5, 0.9, '3-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12)
ax1.text(0.825, 0.9, '4-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12)




#draw a line to separate the layers
ax1.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
ax1.axvline(x=4.5, color='black', linestyle='--', linewidth=1)

ax1.set_title("Dataset size undersampled to 50",fontsize = 14)




tasks = ["2layer/no_pathway", "2layer/pathway","3layer/no_pathway", "3layer/pathway","4layer/no_pathway", "4layer/pathway"]


bp2 = ax2.boxplot(concordence100.values(), patch_artist=True, showfliers=True)
color_palette_dct = {name:color_palette_dct[name] for name in tasks}
ax2.set_xticks([1,2,3,4,5,6])
for idx, model in enumerate(tasks):
    bp2['boxes'][idx].set(color=color_palette_dct[model])
    bp2['boxes'][idx].set(facecolor=color_palette_dct[model])

tasks = ["DeepSurv", "PiDeeL","DeepSurv", "PiDeeL","DeepSurv","PiDeeL"]

ax2.set_xticklabels(tasks,fontsize=12, rotation=90)
ax2.set_ylabel("C-Index",fontsize=12)
ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
ax2.set_ylim(0.4, 0.8)

fontP = FontProperties()
fontP.set_size('x-small')
#write 2 layer, 3 layer, 4 layer in the plot
ax2.text(0.175, 0.9, '2-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=12)
ax2.text(0.5, 0.9, '3-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=12)
ax2.text(0.825, 0.9, '4-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=12)




#draw a line to separate the layers
ax2.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
ax2.axvline(x=4.5, color='black', linestyle='--', linewidth=1)
ax2.set_title("Dataset size undersampled to 100",fontsize = 14)



tasks = ["2layer/no_pathway", "2layer/pathway","3layer/no_pathway", "3layer/pathway","4layer/no_pathway", "4layer/pathway"]


bp3 = ax3.boxplot(concordence200.values(), patch_artist=True, showfliers=True)
color_palette_dct = {name:color_palette_dct[name] for name in tasks}
ax3.set_xticks([1,2,3,4,5,6])
for idx, model in enumerate(tasks):
    bp3['boxes'][idx].set(color=color_palette_dct[model])
    bp3['boxes'][idx].set(facecolor=color_palette_dct[model])

tasks = ["DeepSurv", "PiDeeL","DeepSurv", "PiDeeL","DeepSurv","PiDeeL"]

ax3.set_xticklabels(tasks,fontsize=12, rotation=90)
ax3.set_ylabel("C-Index",fontsize=12)
ax3.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
ax3.set_ylim(0.4, 0.8)

fontP = FontProperties()
fontP.set_size('x-small')
#write 2 layer, 3 layer, 4 layer in the plot
ax3.text(0.175, 0.9, '2-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize=12)
ax3.text(0.5, 0.9, '3-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize=12)
ax3.text(0.825, 0.9, '4-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize=12)




#draw a line to separate the layers
ax3.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
ax3.axvline(x=4.5, color='black', linestyle='--', linewidth=1)
ax3.set_title("Dataset size undersampled to 200",fontsize = 14)

plt.tight_layout()


plt.savefig("suppfig7_undersampled.png")
plt.savefig("suppfig7_undersampled.pdf")
plt.clf()