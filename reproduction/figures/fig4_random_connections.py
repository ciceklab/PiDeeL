import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np

tasks = ["2layer/no_pathway", "2layer/random", "2layer/random2", "2layer/pathway","3layer/no_pathway","3layer/random", "3layer/random2", "3layer/pathway","4layer/no_pathway","4layer/random", "4layer/random2", "4layer/pathway"]

concordence = {}
for task in tasks:
    concordence[task] = []

tasks = ["2layer/pathway", "3layer/pathway","4layer/pathway"]
for task in tasks:
    with open (f"../logs/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordence[task].append(float(lines.strip()))
tasks = ["2layer/no_pathway", "3layer/no_pathway","4layer/no_pathway"]
for task in tasks:
    with open (f"../test_logs/test1/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordence[task].append(float(lines.strip()))


tasks = ["2layer/random", "3layer/random","4layer/random"]
for task in tasks:
    tako = task.split("/")[0]
    with open (f"../test_logs/test2/{tako}/no_pathway/c_indices.txt", "r") as f:
        cs = f.readlines()
        cs = [float(c.strip()) for c in cs]
        cs = np.array(cs)
        cs = cs.reshape(-1, 15)
        cs = np.mean(cs, axis=0)
        concordence[task] = cs.tolist()



tasks = ["2layer/random2", "3layer/random2","4layer/random2"]
for task in tasks:
    tako = task.split("/")[0]
    with open (f"../test_logs/test14/{tako}/no_pathway/c_indices.txt", "r") as f:
        cs = f.readlines()
        cs = [float(c.strip()) for c in cs]
        cs = np.array(cs)
        cs = cs.reshape(-1, 15)
        cs = np.mean(cs, axis=0)
        concordence[task] = cs.tolist()


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
tasks = ["2layer/no_pathway", "2layer/random", "2layer/random2", "2layer/pathway","3layer/no_pathway","3layer/random", "3layer/random2", "3layer/pathway","4layer/no_pathway","4layer/random", "4layer/random2", "4layer/pathway"]


fig, ax = plt.subplots()
bp = ax.boxplot(concordence.values(), patch_artist=True, showfliers=True)
color_palette_dct = {name:color_palette_dct[name] for name in tasks}
ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
for idx, model in enumerate(tasks):
    bp['boxes'][idx].set(color=color_palette_dct[model])
    bp['boxes'][idx].set(facecolor=color_palette_dct[model])

tasks = ["DeepSurv", "Randomly\nconnected", "Randomly\nshuffled", "PiDeeL","DeepSurv", "Randomly\nconnected", "Randomly\nshuffled", "PiDeeL","DeepSurv", "Randomly\nconnected", "Randomly\nshuffled", "PiDeeL"]

ax.set_xticklabels(tasks,fontsize=12, rotation=90)
ax.set_ylabel("C-Index",fontsize=12)
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
ax.set_ylim(0.55, 0.8)

fontP = FontProperties()
fontP.set_size('x-small')
#write 2 layer, 3 layer, 4 layer in the plot
ax.text(0.175, 1.1, '2-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
ax.text(0.5, 1.1, '3-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
ax.text(0.825, 1.1, '4-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)




#draw a line to separate the layers
ax.axvline(x=4.5, color='black', linestyle='--', linewidth=1)
ax.axvline(x=8.5, color='black', linestyle='--', linewidth=1)


plt.tight_layout()


plt.savefig("fig4_random_connections.png")
plt.savefig("fig4_random_connections.tiff", dpi =300)