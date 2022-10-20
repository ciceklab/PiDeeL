import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

sys.path.insert(1, "../../")
sys.path.insert(1, "../../../")
sys.path.insert(1, "../../../../")
import os
from hyper_config import *
print(ultimate_path)

tasks = ["baseline/coxph","baseline/rf","2layer/no_pathway", "2layer/pathway","3layer/no_pathway","3layer/pathway","4layer/no_pathway","4layer/pathway"]
color_palette_dct = {
"baseline/coxph": '#808080',
"baseline/rf": '#808080',
"2layer/no_pathway": '#008080',
"3layer/no_pathway": '#008080',
"4layer/no_pathway": '#008080',
"2layer/pathway": '#2f4f4f',
"3layer/pathway": '#2f4f4f',
"4layer/pathway": '#2f4f4f',}
color_palette_dct = {name:color_palette_dct[name] for name in tasks}


concordance = {}
for task in tasks:
    concordance[task] = []
    with open (ultimate_path + f"/reproduce/reproduction_scripts/logs/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordance[task].append(float(lines.strip()))
tasks = ["2layer/no_pathway", "2layer/pathway","3layer/no_pathway","3layer/pathway","4layer/no_pathway","4layer/pathway"]

aupr = {}
for task in tasks:
    aupr[task] = []
    with open (ultimate_path +f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/aupr.txt", "r") as f:
        for lines in f:
            aupr[task].append(float(lines.strip()))

auroc = {}
for task in tasks:
    auroc[task] = []
    with open (ultimate_path +f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/auroc.txt", "r") as f:
        for lines in f:
            auroc[task].append(float(lines.strip()))

fig, (ax3, ax1, ax2) = plt.subplots(1, 3,figsize=(16,5))
#increase ax3 width
box = ax3.get_position()
ax3.set_position([box.x0, box.y0, box.width * (4.0/3), box.height])
#slide ax1 and ax2 to the right
box = ax1.get_position()
ax1.set_position([box.x0 + box.width * 0.05, box.y0, box.width, box.height])
box = ax2.get_position()
ax2.set_position([box.x0 + box.width * 0.12, box.y0, box.width, box.height])
#slide ax3 to the left
box = ax3.get_position()
ax3.set_position([box.x0 - box.width * 0.25, box.y0, box.width, box.height])

bp3 = ax3.boxplot(concordance.values(), patch_artist=True, showfliers=True)
tasks = ["baseline/coxph","baseline/rf", "2layer/no_pathway","3layer/no_pathway","4layer/no_pathway","2layer/pathway","3layer/pathway","4layer/pathway"]
for idx, model in enumerate(tasks):
    bp3['boxes'][idx].set(color=color_palette_dct[model])
    bp3['boxes'][idx].set(facecolor=color_palette_dct[model])

fontP = FontProperties()
fontP.set_size('x-small')
color_palette_dct = {name:color_palette_dct[name] for name in tasks}
ax3.set_ylabel("C-Index",fontsize=14)
ax3.set_xticks([1,2,3,4,5,6,7,8])
ax3.set_ylim(0.5, 0.8)

tasks = ["Cox-PH","RSF", "2-layer","3-layer","4-layer","2-layer","3-layer","4-layer"]
ax3.set_yticks([0.5, 0.6,0.7,0.8],fontsize=12)

ax3.set_xticklabels(tasks,fontsize=12,rotation=45)

bp1 = ax1.boxplot(auroc.values(), patch_artist=True, showfliers=True)
tasks = ["2layer/no_pathway", "3layer/no_pathway", "4layer/no_pathway", "2layer/pathway","3layer/pathway","4layer/pathway"]
color_palette_dct = {
"2layer/no_pathway": '#008080',
"3layer/no_pathway": '#008080',
"4layer/no_pathway": '#008080',
"2layer/pathway": '#2f4f4f',
"3layer/pathway": '#2f4f4f',
"4layer/pathway": '#2f4f4f',}
color_palette_dct = {name:color_palette_dct[name] for name in tasks}

ax1.set_ylabel("AUC-ROC",fontsize=14)
tasks = ["2-layer", "3-layer","4-layer","2-layer","3-layer","4-layer"]

ax1.set_xticklabels(tasks,fontsize=12,rotation=45)




ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1],fontsize=12)
ax1.set_ylim(0.75, 1.0)
tasks = ["2layer/no_pathway", "3layer/no_pathway", "4layer/no_pathway", "2layer/pathway","3layer/pathway","4layer/pathway"]

for idx, model in enumerate(tasks):
    bp1['boxes'][idx].set(color=color_palette_dct[model])
    bp1['boxes'][idx].set(facecolor=color_palette_dct[model])

fontP = FontProperties()
fontP.set_size('x-small')



ax2.boxplot(aupr.values(), patch_artist=True, showfliers=True)
bp2 = ax2.boxplot(aupr.values(), patch_artist=True, showfliers=True)
color_palette_dct = {name:color_palette_dct[name] for name in tasks}
ax2.set_xticks([1,2,3,4,5,6])
ax2.set_ylabel("AUC-PR",fontsize=14)
tasks = ["2-layer", "3-layer","4-layer","2-layer","3-layer","4-layer"]

ax2.set_xticklabels(tasks,fontsize=12,rotation=45)

ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1],fontsize=12)
ax2.set_ylim(0.75, 1.0)
tasks = ["2layer/no_pathway", "3layer/no_pathway", "4layer/no_pathway", "2layer/pathway","3layer/pathway","4layer/pathway"]

for idx, model in enumerate(tasks):
    bp2['boxes'][idx].set(color=color_palette_dct[model])
    bp2['boxes'][idx].set(facecolor=color_palette_dct[model])

fontP = FontProperties()
fontP.set_size('x-small')

ax3.axvspan(0.5, 2.5, facecolor='gray', alpha=0.2)
ax3.text(1.5,0.82,'Baseline',ha="center",fontsize=14)
ax3.axvspan(2.5, 5.5, facecolor='green', alpha=0.2)
ax3.text(4,0.82,'DeepSurv',ha="center",fontsize=14)
ax3.axvspan(5.5, 8.5, facecolor='darkblue', alpha=0.2)
ax3.text(7,0.82,'PiDeeL',ha="center",fontsize=14)

ax1.axvspan(0.5, 3.5, facecolor='green', alpha=0.2)
ax1.text(2, 1.016,'Fully-connected',ha="center",fontsize=14)
ax1.axvspan(3.5, 6.5, facecolor='darkblue', alpha=0.2)
ax1.text(5 ,1.016,'PiDeeL', ha="center",fontsize=14)

ax2.axvspan(0.5, 3.5, facecolor='green', alpha=0.2)
ax2.text(2, 1.016,'Fully-connected',ha="center",fontsize=14)
ax2.axvspan(3.5, 6.5, facecolor='darkblue', alpha=0.2)
ax2.text(5 ,1.016,'PiDeeL', ha="center",fontsize=14)


#write a) b) c) on the plot
ax1.text(-0.1, 1.1, 'b', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
ax3.text(-0.1, 1.1, 'a', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

#draw black line between ax1 and ax2 to divide the plot
ax3.plot([1.1,1.1], [-1, 2], transform=ax3.transAxes, color='k', clip_on=False)




plt.tight_layout()
box = ax3.get_position()
ax3.set_position([box.x0 - box.width * 0.25, box.y0, box.width, box.height])
box = ax1.get_position()
ax1.set_position([box.x0 - box.width * 0.1, box.y0, box.width, box.height])
plt.savefig(ultimate_path + "/reproduce/reproduction_scripts/figures/Fig1.pdf")
plt.clf()