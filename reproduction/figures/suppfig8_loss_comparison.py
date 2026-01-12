import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np

tasks = ["DeepHit", "PC-Hazard","DeepSurv"]

concordence = {}
for task in tasks:
    concordence[task] = []


tasks = ["4layer/pathway"]
for task in tasks:
    with open (f"../test_logs/test29/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordence["DeepHit"].append(float(lines.strip()))
tasks = ["4layer/pathway"]
for task in tasks:
    with open (f"../test_logs/test30/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordence["PC-Hazard"].append(float(lines.strip()))

tasks = ["4layer/pathway"]
for task in tasks:
    with open (f"../logs/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordence["DeepSurv"].append(float(lines.strip()))
color_palette_dct = {
"DeepHit": '#008080',
"PC-Hazard": '#008080',
"DeepSurv": '#2f4f4f'}
tasks = ["DeepHit", "PC-Hazard","DeepSurv"]
print("\n")
for task in tasks:
    print(f"ablation6 = {task}, median = {np.median(concordence[task])} , mean = {np.mean(concordence[task])}, std = {np.std(concordence[task])}")



fig, ax = plt.subplots()
bp = ax.boxplot(concordence.values(), patch_artist=True, showfliers=True)
color_palette_dct = {name:color_palette_dct[name] for name in tasks}
ax.set_xticks([1,2,3])
for idx, model in enumerate(tasks):
    bp['boxes'][idx].set(color=color_palette_dct[model])
    bp['boxes'][idx].set(facecolor=color_palette_dct[model])

tasks = ["PiDeeL\n(DeepHit loss)", "PiDeeL\n(PC-Hazard loss)","PiDeeL\n(DeepSurv loss)"]

ax.set_xticklabels(tasks,fontsize=12)
ax.set_ylabel("C-Index",fontsize=12)
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
ax.set_ylim(0.5, 0.8)

fontP = FontProperties()
fontP.set_size('x-small')
#write 2 layer, 3 layer, 4 layer in the plot
#ax.text(0.175, 1.1, '2-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
#ax.text(0.5, 1.1, '3-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
#ax.text(0.825, 1.1, '4-layer\nmodels', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)




#draw a line to separate the layers
#ax.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
#ax.axvline(x=4.5, color='black', linestyle='--', linewidth=1)


plt.tight_layout()


fig.savefig("suppfig8_loss_comparison.png")
fig.savefig("suppfig8_loss_comparison.pdf")
