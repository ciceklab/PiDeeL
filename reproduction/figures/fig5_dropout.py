import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np




tasks2 = ['2layer/pathway','2layer/no_pathway0.5', '2layer/no_pathway0.6', '2layer/no_pathway0.7', '2layer/no_pathway0.8', '2layer/no_pathway0.91' ,'3layer/pathway','3layer/no_pathway0.5', '3layer/no_pathway0.6', '3layer/no_pathway0.7', '3layer/no_pathway0.8', '3layer/no_pathway0.91','4layer/pathway', '4layer/no_pathway0.5', '4layer/no_pathway0.6', '4layer/no_pathway0.7', '4layer/no_pathway0.8', '4layer/no_pathway0.91']

concordence = {}
for task in tasks2:
    concordence[task] = []

tasks = ["2layer/pathway", "3layer/pathway","4layer/pathway"]
for task in tasks:
    with open (f"../logs/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordence[task].append(float(lines.strip()))

drop_out_probs = [0.5,0.6,0.7,0.8,0.91]

tasks = ["2layer/no_pathway", "3layer/no_pathway","4layer/no_pathway"]
for task in tasks:
    for p in drop_out_probs:
        with open (f"../test_logs/test13/{task}/{p}_c_indices.txt", "r") as f:
            for lines in f:
                concordence[str(task)+str(p)].append(float(lines.strip()))




fig, ax = plt.subplots()





color_palette_dct = {
"2layer/no_pathway0.5": '#008080',
"2layer/no_pathway0.6": '#008080',
"2layer/no_pathway0.7": '#008080',
"2layer/no_pathway0.8": '#008080',
"2layer/no_pathway0.91": '#008080',
"2layer/pathway": '#2f4f4f',
"3layer/no_pathway0.5": '#008080',
"3layer/no_pathway0.6": '#008080',
"3layer/no_pathway0.7": '#008080',
"3layer/no_pathway0.8": '#008080',
"3layer/no_pathway0.91": '#008080',
"3layer/pathway": '#2f4f4f',
"4layer/no_pathway0.5": '#008080',
"4layer/no_pathway0.6": '#008080',
"4layer/no_pathway0.7": '#008080',
"4layer/no_pathway0.8": '#008080',
"4layer/no_pathway0.91": '#008080',
"4layer/pathway": '#2f4f4f'}

color_palette_dct = {name:color_palette_dct[name] for name in tasks2}


ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_ylabel("C-Index",fontsize=12)
bp = ax.boxplot(concordence.values(), patch_artist=True, showfliers=True)
tasks2 = ['PiDeeL','p=0.5', 'p=0.6', 'p=0.7', 'p=0.8', 'p=0.9' ,'PiDeeL','p=0.5', 'p=0.6', 'p=0.7', 'p=0.8', 'p=0.9','PiDeeL', 'p=0.5', 'p=0.6', 'p=0.7', 'p=0.8', 'p=0.9']


ax.set_xticklabels(tasks2,fontsize=10, rotation=90)
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
ax.set_ylim(0.3, 0.8)
tasks2 = ['2layer/pathway','2layer/no_pathway0.5', '2layer/no_pathway0.6', '2layer/no_pathway0.7', '2layer/no_pathway0.8', '2layer/no_pathway0.91' ,'3layer/pathway','3layer/no_pathway0.5', '3layer/no_pathway0.6', '3layer/no_pathway0.7', '3layer/no_pathway0.8', '3layer/no_pathway0.91','4layer/pathway', '4layer/no_pathway0.5', '4layer/no_pathway0.6', '4layer/no_pathway0.7', '4layer/no_pathway0.8', '4layer/no_pathway0.91']

for idx, model in enumerate(tasks2):
    bp['boxes'][idx].set(color=color_palette_dct[model])
    bp['boxes'][idx].set(facecolor=color_palette_dct[model])

fontP = FontProperties()
fontP.set_size('x-small')

#draw a line to separate the layers
plt.axvline(x=6.5, color='black', linestyle='--', linewidth=1)
plt.axvline(x=12.5, color='black', linestyle='--', linewidth=1)

ax.text(0.175, 1.1, '2-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
ax.text(0.5, 1.1, '3-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
ax.text(0.825, 1.1, '4-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)

ax.text(0.2, -0.26, 'DeepSurv', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
ax.text(0.55, -0.26, 'DeepSurv', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
ax.text(0.875, -0.26, 'DeepSurv', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)

plt.axvline(x=1.5, ymin = -0.2, ymax = -0.1, color='black', linestyle='-', linewidth=1,clip_on=False)
plt.axvline(x=6.5, ymin = -0.2, ymax = -0.1, color='black', linestyle='-', linewidth=1,clip_on=False)
#draw a horizontal line at y = -0.2 between x = 1.5 and x = 6.5 
plt.axhline(y=0.2, xmin = 0.055, xmax = 0.332, color='black', linestyle='-', linewidth=1,clip_on=False)





plt.axvline(x=7.5, ymin = -0.2, ymax = -0.1, color='black', linestyle='-', linewidth=1,clip_on=False)
plt.axvline(x=12.5, ymin = -0.2, ymax = -0.1, color='black', linestyle='-', linewidth=1,clip_on=False)
plt.axhline(y=0.2, xmin = 0.39, xmax = 0.665, color='black', linestyle='-', linewidth=1,clip_on=False)

plt.axvline(x=13.5, ymin = -0.2, ymax = -0.1, color='black', linestyle='-', linewidth=1,clip_on=False)
plt.axvline(x=18.5, ymin = -0.2, ymax = -0.1, color='black', linestyle='-', linewidth=1,clip_on=False)
plt.axhline(y=0.2, xmin = 0.722, xmax = 1, color='black', linestyle='-', linewidth=1,clip_on=False)

plt.tight_layout()

plt.savefig("fig5_dropout.png")

plt.savefig("fig5_dropout.pdf")
