

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties



tasks = ["baseline/coxph","baseline/rf", "2layer/no_pathway", "3layer/no_pathway", "4layer/no_pathway", "2layer/pathway","3layer/pathway","4layer/pathway"]

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



tasks = ["2layer/no_pathway", "3layer/no_pathway", "4layer/no_pathway", "2layer/pathway","3layer/pathway","4layer/pathway"]

aupr = {}
aupr["rf"] = []
with open (f"../test_logs/test22/aupr.txt", "r") as f:
    for lines in f:
        aupr["rf"].append(float(lines.strip()))
for task in tasks:
    aupr[task] = []
    with open (f"../test_logs/test5/{task}/aupr.txt", "r") as f:
        for lines in f:
            aupr[task].append(float(lines.strip()))

auroc = {}
auroc["rf"] = []
with open (f"../test_logs/test22/auroc.txt", "r") as f:
    for lines in f:
        auroc["rf"].append(float(lines.strip()))
for task in tasks:
    auroc[task] = []
    with open (f"../test_logs/test5/{task}/auroc.txt", "r") as f:
        for lines in f:
            auroc[task].append(float(lines.strip()))


fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,5))

bp1 = ax1.boxplot(auroc.values(), patch_artist=True, showfliers=True)
tasks = ["baseline/rf","2layer/no_pathway", "3layer/no_pathway", "4layer/no_pathway", "2layer/pathway","3layer/pathway","4layer/pathway"]

color_palette_dct = {name:color_palette_dct[name] for name in tasks}
#ax1.get_xaxis().tick_bottom()
#ax1.get_yaxis().tick_left()





ax1.set_ylabel("AUC-ROC",fontsize=14)
tasks = ["RF","2-layer", "3-layer","4-layer","2-layer","3-layer","4-layer"]

ax1.set_xticklabels(tasks,fontsize=14,rotation=45)




ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
ax1.set_ylim(0.75, 1.0)
tasks = ["baseline/rf","2layer/no_pathway", "3layer/no_pathway", "4layer/no_pathway", "2layer/pathway","3layer/pathway","4layer/pathway"]

for idx, model in enumerate(tasks):
    bp1['boxes'][idx].set(color=color_palette_dct[model])
    bp1['boxes'][idx].set(facecolor=color_palette_dct[model])

fontP = FontProperties()
fontP.set_size('x-small')



ax2.boxplot(aupr.values(), patch_artist=True, showfliers=True)
bp2 = ax2.boxplot(aupr.values(), patch_artist=True, showfliers=True)
color_palette_dct = {name:color_palette_dct[name] for name in tasks}
#ax2.get_xaxis().tick_bottom()
ax2.set_xticks([1,2,3,4,5,6,7])
#ax2.get_yaxis().tick_left()
ax2.set_ylabel("AUC-PR",fontsize=14)
tasks = ["RF", "2-layer", "3-layer","4-layer","2-layer","3-layer","4-layer"]

ax2.set_xticklabels(tasks,fontsize=14,rotation=45)

ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
ax2.set_ylim(0.75, 1.0)
tasks = ["baseline/rf","2layer/no_pathway", "3layer/no_pathway", "4layer/no_pathway", "2layer/pathway","3layer/pathway","4layer/pathway"]

for idx, model in enumerate(tasks):
    bp2['boxes'][idx].set(color=color_palette_dct[model])
    bp2['boxes'][idx].set(facecolor=color_palette_dct[model])

fontP = FontProperties()
fontP.set_size('x-small')


#ax1.axvspan(0.5, 1.5, facecolor='gray', alpha=0.2)
ax1.text(0.92,1.016,'Baseline',ha="center",fontsize=14)
#ax1.axvspan(1.5, 4.5, facecolor='green', alpha=0.2)
ax1.text(3.2, 1.016,'Fully-connected',ha="center",fontsize=14)
#ax1.axvspan(4.5, 7.5, facecolor='darkblue', alpha=0.2)
ax1.text(6 ,1.016,'PiDeeL', ha="center",fontsize=14)

#ax2.axvspan(0.5, 1.5, facecolor='gray', alpha=0.2)
ax2.text(0.92,1.016,'Baseline',ha="center",fontsize=14)
#ax2.axvspan(1.5, 4.5, facecolor='green', alpha=0.2)
ax2.text(3.2, 1.016,'Fully-connected',ha="center",fontsize=14)
#ax2.axvspan(4.5, 7.5, facecolor='darkblue', alpha=0.2)
ax2.text(6 ,1.016,'PiDeeL', ha="center",fontsize=14)

plt.tight_layout()

plt.savefig("suppfig2_auroc_aupr.png")
plt.savefig("suppfig2_auroc_aupr.pdf")
