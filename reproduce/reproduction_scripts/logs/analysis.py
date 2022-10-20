import numpy as np
import sys

sys.path.insert(1, "../../")
sys.path.insert(1, "../../../")
sys.path.insert(1, "../../../../")
from hyper_config import *

print(ultimate_path)

tasksc = ["baseline/rf", "baseline/coxph", "2layer/no_pathway", "2layer/pathway", "3layer/no_pathway", "3layer/pathway",
          "4layer/no_pathway", "4layer/pathway"]
tasks = ["2layer/no_pathway", "2layer/pathway", "3layer/no_pathway", "3layer/pathway", "4layer/no_pathway",
         "4layer/pathway"]

concordance = {}
for task in tasksc:
    concordance[task] = []
    with open(ultimate_path + f"/reproduce/reproduction_scripts/logs/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordance[task].append(float(lines.strip()))

auroc = {}
for task in tasks:
    auroc[task] = []
    with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/auroc.txt", "r") as f:
        for lines in f:
            auroc[task].append(float(lines.strip()))
aupr = {}
for task in tasks:
    aupr[task] = []
    with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/aupr.txt", "r") as f:
        for lines in f:
            aupr[task].append(float(lines.strip()))

print("Concordance")
for task in tasksc:
    print(
        f"{task}, median = {np.median(concordance[task])} , mean = {np.mean(concordance[task])}, std = {np.std(concordance[task])}")

print("AUROC")
for task in tasks:
    print(f"{task}, median = {np.median(auroc[task])} , mean = {np.mean(auroc[task])}, std = {np.std(auroc[task])}")
print("AUPR")
for task in tasks:
    print(f"{task}, median = {np.median(aupr[task])} , mean = {np.mean(aupr[task])}, std = {np.std(aupr[task])}")

print("\n")
tasks = ["2layer/pathway", "3layer/pathway", "4layer/pathway"]

concordence_multitask = {}
for task in tasks:
    concordence_multitask[task] = []
    with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test7/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordence_multitask[task].append(float(lines.strip()))

auroc_multitask = {}
for task in tasks:
    auroc_multitask[task] = []
    with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test7/{task}/auroc.txt", "r") as f:
        for lines in f:
            auroc_multitask[task].append(float(lines.strip()))

aupr_multitask = {}
for task in tasks:
    aupr_multitask[task] = []
    with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test7/{task}/aupr.txt", "r") as f:
        for lines in f:
            aupr_multitask[task].append(float(lines.strip()))

print("Concordence")
for task in tasks:
    print(
        f"singletask = {task}, median = {np.median(concordance[task])} , mean = {np.mean(concordance[task])}, std = {np.std(concordance[task])}")
    print(
        f"multitask = {task}, median = {np.median(concordence_multitask[task])} , mean = {np.mean(concordence_multitask[task])}, std = {np.std(concordence_multitask[task])}")
print("AUROC")
for task in tasks:
    print(
        f"singletask = {task}, median = {np.median(auroc[task])} , mean = {np.mean(auroc[task])}, std = {np.std(auroc[task])}")
    print(
        f"multitask = {task}, median = {np.median(auroc_multitask[task])} , mean = {np.mean(auroc_multitask[task])}, std = {np.std(auroc_multitask[task])}")

print("AUPR")
for task in tasks:
    print(
        f"singletask = {task}, median = {np.median(aupr[task])} , mean = {np.mean(aupr[task])}, std = {np.std(aupr[task])}")
    print(
        f"multitask = {task}, median = {np.median(aupr_multitask[task])} , mean = {np.mean(aupr_multitask[task])}, std = {np.std(aupr_multitask[task])}")

tasks = ["2layer/no_pathway", "3layer/no_pathway", "4layer/no_pathway"]

concordance_ablation1 = {}
for task in tasks:
    concordance_ablation1[task] = []
    with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test1/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordance_ablation1[task].append(float(lines.strip()))

concordance_ablation2 = {}
for task in tasks:
    concordance_ablation2[task] = []
    with open(f"/reproduce/reproduction_scripts/tests/logs/test2/{task}/c_indices.txt", "r") as f:
        for lines in f:
            concordance_ablation2[task].append(float(lines.strip()))

print("Concordance")
for task in tasks:
    print(
        f"ablation1 = {task}, median = {np.median(concordance_ablation1[task])} , mean = {np.mean(concordance_ablation1[task])}, std = {np.std(concordance_ablation1[task])}")

print("\n")

for task in tasks:
    print(
        f"ablation2 = {task}, median = {np.median(concordance_ablation2[task])} , mean = {np.mean(concordance_ablation2[task])}, std = {np.std(concordance_ablation2[task])}")
