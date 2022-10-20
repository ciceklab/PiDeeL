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
color_palette_dct = {
"2layer/pathway_singletask": '#2f4f4f',
"3layer/pathway_singletask": '#2f4f4f',
"4layer/pathway_singletask": '#2f4f4f',
"2layer/pathway_multitask": '#8b0000',
"3layer/pathway_multitask": '#8b0000',
"4layer/pathway_multitask": '#8b0000'}
tasks = ["2layer/pathway_singletask", "3layer/pathway_singletask","4layer/pathway_singletask", "2layer/pathway_multitask","3layer/pathway_multitask","4layer/pathway_multitask"]

concordence = {}
for task in tasks:
    concordence[task] = []



tasks = ["2layer/pathway", "3layer/pathway", "4layer/pathway"]
for task in tasks:
    with open (ultimate_path + f"/reproduce/reproduction_scripts/logs/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordence[f"{task}_singletask"].append(float(lines.strip()))
tasks = ["2layer/pathway", "3layer/pathway", "4layer/pathway"]

for task in tasks:
    with open (ultimate_path +f"/reproduce/reproduction_scripts/tests/logs/test7/{task}/c_index.txt", "r") as f:
        for lines in f:
            concordence[f"{task}_multitask"].append(float(lines.strip()))


tasks = ["2layer/pathway_singletask", "3layer/pathway_singletask","4layer/pathway_singletask", "2layer/pathway_multitask","3layer/pathway_multitask","4layer/pathway_multitask"]

auroc = {}
for task in tasks:
    auroc[task] = []



tasks = ["2layer/pathway", "3layer/pathway", "4layer/pathway"]
for task in tasks:
    with open (ultimate_path +f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/auroc.txt", "r") as f:
        for lines in f:
            auroc[f"{task}_singletask"].append(float(lines.strip()))
tasks = ["2layer/pathway", "3layer/pathway", "4layer/pathway"]

for task in tasks:
    with open (ultimate_path +f"/reproduce/reproduction_scripts/tests/logs/test7/{task}/auroc.txt", "r") as f:
        for lines in f:
            auroc[f"{task}_multitask"].append(float(lines.strip()))


tasks = ["2layer/pathway_singletask", "3layer/pathway_singletask","4layer/pathway_singletask", "2layer/pathway_multitask","3layer/pathway_multitask","4layer/pathway_multitask"]

aupr = {}
for task in tasks:
    aupr[task] = []



tasks = ["2layer/pathway", "3layer/pathway", "4layer/pathway"]
for task in tasks:
    with open (ultimate_path +f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/aupr.txt", "r") as f:
        for lines in f:
            aupr[f"{task}_singletask"].append(float(lines.strip()))
#tasks = ["rf/no_pathway", "1layer/no_pathway", "2layer/no_pathway", "2layer/pathway","3layer/no_pathway","3layer/pathway","3layer/pathway_process","3layer/pathway_process2","4layer/no_pathway","4layer/pathway","4layer/pathway_process","4layer/pathway_process2","4layer/pathway_process_type"]
tasks = ["2layer/pathway", "3layer/pathway", "4layer/pathway"]

for task in tasks:
    with open (ultimate_path +f"/reproduce/reproduction_scripts/tests/logs/test7/{task}/aupr.txt", "r") as f:
        for lines in f:
            aupr[f"{task}_multitask"].append(float(lines.strip()))

tasks = ["2layer/pathway_singletask", "3layer/pathway_singletask","4layer/pathway_singletask", "2layer/pathway_multitask","3layer/pathway_multitask","4layer/pathway_multitask"]




fig, (ax3, ax1, ax2) = plt.subplots(1, 3,figsize=(16,5))
"""
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

"""
box = ax1.get_position()
ax1.set_position([box.x0 + box.width * 0.05, box.y0, box.width, box.height])
box = ax2.get_position()
ax2.set_position([box.x0 + box.width * 0.05, box.y0, box.width, box.height])




bp3 = ax3.boxplot(concordence.values(), patch_artist=True, showfliers=True)
tasks = ["2layer/pathway_singletask", "3layer/pathway_singletask","4layer/pathway_singletask", "2layer/pathway_multitask","3layer/pathway_multitask","4layer/pathway_multitask"]
for idx, model in enumerate(tasks):
    bp3['boxes'][idx].set(color=color_palette_dct[model])
    bp3['boxes'][idx].set(facecolor=color_palette_dct[model])

fontP = FontProperties()
fontP.set_size('x-small')
color_palette_dct = {name:color_palette_dct[name] for name in tasks}
ax3.set_ylabel("C-Index",fontsize=14)
ax3.set_xticks([1,2,3,4,5,6])
ax3.set_ylim(0.5, 0.8)
ax3.set_yticks([0.5, 0.6,0.7,0.8],fontsize=12)

tasks = ["2-layer","3-layer","4-layer","2-layer","3-layer","4-layer"]

ax3.set_xticklabels(tasks,fontsize=12,rotation=45)




bp1 = ax1.boxplot(auroc.values(), patch_artist=True, showfliers=True)
tasks = ["2layer/pathway_singletask", "3layer/pathway_singletask","4layer/pathway_singletask", "2layer/pathway_multitask","3layer/pathway_multitask","4layer/pathway_multitask"]

color_palette_dct = {name:color_palette_dct[name] for name in tasks}
#ax1.get_xaxis().tick_bottom()
#ax1.get_yaxis().tick_left()

ax1.set_ylabel("AUC-ROC",fontsize=14)
tasks = ["2-layer","3-layer","4-layer","2-layer","3-layer","4-layer"]

ax1.set_xticklabels(tasks,fontsize=12,rotation=45)




ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1],fontsize=12)
ax1.set_ylim(0.75, 1.0)
tasks = ["2layer/pathway_singletask", "3layer/pathway_singletask","4layer/pathway_singletask", "2layer/pathway_multitask","3layer/pathway_multitask","4layer/pathway_multitask"]

for idx, model in enumerate(tasks):
    bp1['boxes'][idx].set(color=color_palette_dct[model])
    bp1['boxes'][idx].set(facecolor=color_palette_dct[model])

fontP = FontProperties()
fontP.set_size('x-small')



ax2.boxplot(aupr.values(), patch_artist=True, showfliers=True)
bp2 = ax2.boxplot(aupr.values(), patch_artist=True, showfliers=True)
color_palette_dct = {name:color_palette_dct[name] for name in tasks}
#ax2.get_xaxis().tick_bottom()
ax2.set_xticks([1,2,3,4,5,6])
#ax2.get_yaxis().tick_left()
ax2.set_ylabel("AUC-PR",fontsize=14)
tasks = ["2-layer","3-layer","4-layer","2-layer","3-layer","4-layer"]

ax2.set_xticklabels(tasks,fontsize=12,rotation=45)

ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1],fontsize=12)
ax2.set_ylim(0.75, 1.0)
tasks = ["2layer/pathway_singletask", "3layer/pathway_singletask","4layer/pathway_singletask", "2layer/pathway_multitask","3layer/pathway_multitask","4layer/pathway_multitask"]

for idx, model in enumerate(tasks):
    bp2['boxes'][idx].set(color=color_palette_dct[model])
    bp2['boxes'][idx].set(facecolor=color_palette_dct[model])

fontP = FontProperties()
fontP.set_size('x-small')
"""
plt.tight_layout()
box = ax3.get_position()
ax3.set_position([box.x0 - box.width * 0.2, box.y0, box.width, box.height])
box = ax1.get_position()
ax1.set_position([box.x0 - box.width * 0.1, box.y0, box.width, box.height])
"""

ax3.axvspan(0.5, 3.5, facecolor='darkblue', alpha=0.2)
ax3.text(2, 0.82,'Single task',ha="center",fontsize=14)
ax3.axvspan(3.5, 6.5, facecolor='red', alpha=0.2)
ax3.text(5 ,0.82,'Multitask', ha="center",fontsize=14)

ax1.axvspan(0.5, 3.5, facecolor='darkblue', alpha=0.2)
ax1.text(2, 1.016,'Single task',ha="center",fontsize=14)
ax1.axvspan(3.5, 6.5, facecolor='red', alpha=0.2)
ax1.text(5 ,1.016,'Multitask', ha="center",fontsize=14)

ax2.axvspan(0.5, 3.5, facecolor='darkblue', alpha=0.2)
ax2.text(2, 1.016,'Single task',ha="center",fontsize=14)
ax2.axvspan(3.5, 6.5, facecolor='red', alpha=0.2)
ax2.text(5 ,1.016,'Multitask', ha="center",fontsize=14)


#write a) b) c) on the plot
ax1.text(-0.1, 1.1, 'b', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
#ax2.text(-0.1, 1.1, 'c', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
ax3.text(-0.1, 1.1, 'a', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

#draw black line between ax1 and ax2 to divide the plot
ax3.plot([1.05,1.05], [-1, 2], transform=ax3.transAxes, color='k', clip_on=False)
#ax3.plot([2.0,2.0], [-1, 2], transform=ax3.transAxes, color='k', clip_on=False)

#ax2.plot([0, 8], [1.005, 1.005], transform=ax2.transAxes, color='k', clip_on=False)








box = ax1.get_position()
ax1.set_position([box.x0 - box.width * 0.1, box.y0, box.width, box.height])
plt.tight_layout()
plt.savefig(ultimate_path + "/reproduce/reproduction_scripts/figures/Fig2.pdf")
plt.clf()