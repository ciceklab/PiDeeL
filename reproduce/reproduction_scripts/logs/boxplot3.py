import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../../../")
sys.path.insert(1, "../../../../")
import os
from hyper_config import *
print(ultimate_path)

tasks = ["2layer/no_pathway", "2layer/pathway","3layer/no_pathway","3layer/pathway","4layer/no_pathway","4layer/pathway"]

concordence = {}
for task in tasks:
    concordence[task] = []

tasks = ["2layer/pathway", "3layer/pathway","4layer/pathway"]
for task in tasks:
    with open (ultimate_path + f"/reproduce/reproduction_scripts/logs/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordence[task].append(float(lines.strip()))
tasks = ["2layer/no_pathway", "3layer/no_pathway","4layer/no_pathway"]
for task in tasks:
    with open (ultimate_path +f"/reproduce/reproduction_scripts/tests/logs/test2/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordence[task].append(float(lines.strip()))


tasks = ["2layer/no_pathway", "2layer/pathway","3layer/no_pathway", "3layer/pathway","4layer/no_pathway", "4layer/pathway"]

concordence2 = {}
for task in tasks:
    concordence2[task] = []

tasks = ["2layer/pathway", "3layer/pathway","4layer/pathway"]
for task in tasks:
    with open (ultimate_path + f"/reproduce/reproduction_scripts/logs/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordence2[task].append(float(lines.strip()))
tasks = ["2layer/no_pathway", "3layer/no_pathway","4layer/no_pathway"]
for task in tasks:
    with open (ultimate_path +f"/reproduce/reproduction_scripts/tests/logs/test1/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordence2[task].append(float(lines.strip()))

tasks = ["2layer/no_pathway", "2layer/pathway","3layer/no_pathway", "3layer/pathway","4layer/no_pathway", "4layer/pathway"]


color_palette_dct = {
"2layer/no_pathway": '#008080',
"2layer/pathway": '#2f4f4f',
"3layer/no_pathway": '#008080',
"3layer/pathway": '#2f4f4f',
"4layer/no_pathway": '#008080',
"4layer/pathway": '#2f4f4f'}

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))
bp1 = ax1.boxplot(concordence2.values(), patch_artist=True, showfliers=True)
color_palette_dct = {name:color_palette_dct[name] for name in tasks}


#axes.get_xaxis().tick_bottom()
#axes.get_yaxis().tick_left()
tasks = ["2-layer DeepSurv", "2-layer PiDeeL","3-layer DeepSurv", "3-layer PiDeeL","4-layer DeepSurv", "4-layer PiDeeL"]

ax1.set_xticks([1,2,3,4,5,6])
ax1.set_xticklabels(tasks,fontsize=8,rotation=45)


ax1.set_ylabel("C-Index",fontsize=12)



ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
ax1.set_ylim(0.4, 0.8)
tasks = ["2layer/no_pathway", "2layer/pathway","3layer/no_pathway", "3layer/pathway","4layer/no_pathway", "4layer/pathway"]

for idx, model in enumerate(tasks):
    bp1['boxes'][idx].set(color=color_palette_dct[model])
    bp1['boxes'][idx].set(facecolor=color_palette_dct[model])

fontP = FontProperties()
fontP.set_size('x-small')

bp2 = ax2.boxplot(concordence.values(), patch_artist=True, showfliers=True)
color_palette_dct = {name:color_palette_dct[name] for name in tasks}


tasks = ["Randomly connected \n 2-layer PiDeeL", "2-layer PiDeeL","Randomly connected \n 2-layer PiDeeL", "3-layer PiDeeL","Randomly connected \n 2-layer PiDeeL", "4-layer PiDeeL"]

ax2.set_xticks([1,2,3,4,5,6])
ax2.set_xticklabels(tasks,fontsize=8,rotation=45)


ax2.set_ylabel("C-Index",fontsize=12)



ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
ax2.set_ylim(0.4, 0.8)
tasks = ["2layer/no_pathway", "2layer/pathway","3layer/no_pathway", "3layer/pathway","4layer/no_pathway", "4layer/pathway"]

for idx, model in enumerate(tasks):
    bp2['boxes'][idx].set(color=color_palette_dct[model])
    bp2['boxes'][idx].set(facecolor=color_palette_dct[model])

fontP = FontProperties()
fontP.set_size('x-small')
#write a b on the plot
ax1.text(-0.1, 1.05, 'a', transform=ax1.transAxes, size=20, weight='bold')
ax2.text(-0.1, 1.05, 'b', transform=ax2.transAxes, size=20, weight='bold')



#figure.tight_layout()
plt.tight_layout()


plt.savefig(ultimate_path + "/reproduce/reproduction_scripts/figures/Fig3.pdf")
plt.clf()
