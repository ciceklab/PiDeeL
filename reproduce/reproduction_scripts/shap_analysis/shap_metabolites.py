import sys
import pickle

sys.path.insert(1, "../../")
sys.path.insert(1, "../../../")
sys.path.insert(1, "../../../../")
sys.path.insert(1, "../../../../../")
sys.path.insert(1, "../../../../../../")

from hyper_config import *

with open("../../../pred_quant.pickle", "rb") as f:
    pred_quant = pickle.load(f)
with open("../../../grade.pickle", "rb") as f:
    grade = pickle.load(f)
with open("../../../events.pickle", "rb") as f:
    events = pickle.load(f)
with open("../../../survival.pickle", "rb") as f:
    survival = pickle.load(f)
with open("../../../pathway_info.pickle", "rb") as f:
    pathway_info = pickle.load(f)



from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from model import Net, initialize_weights, Net2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import shap
import pandas as pd
task = "4layer/pathway"

all_shap = []
X_tests = []
import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, pathway_info):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = pathway_info.shape[1]
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.pathway_info = pathway_info.float()

    def forward(self, x):
        pathway = torch.matmul(x, (self.pathway_info * self.fc1.weight.t())) + self.fc1.bias
        pathway = F.relu(pathway)
        hid = F.relu(self.fc2(pathway))
        out = F.relu((self.fc3(hid)))
        out = ((self.fc4(out)))
        return out


cols_stand = ['2-hydroxyglutarate', '3-hydroxybutyrate', 'Acetate', \
              'Alanine', 'Allocystathionine', 'Arginine', 'Ascorbate', \
              'Aspartate', 'Betaine', 'Choline', 'Creatine', \
              'Ethanolamine', 'GABA', 'Glutamate', 'Glutamine', 'Glutahionine (GSH)', \
              'Glycerophosphocholine', 'Glycine', 'hypo-Taurine', \
              'Isoleucine', 'Lactate', 'Leucine', 'Lysine', 'Methionine', \
              'myo-Inositol', 'NAL', 'NAA', 'o-Acetylcholine', 'Ornithine', \
              'Phosphocholine', 'Phosphocreatine', 'Proline', \
              'scyllo-Inositol', 'Serine', 'Taurine', 'Threonine', \
              'Valine']
device = torch.device(f"cuda:0")
SEEDS = [6, 35, 81]
c_indices = []
for SEED in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    result = list(kf.split(events))
    for i in range(len(result)):
        X_train = pred_quant[result[i][0]]
        X_test = pred_quant[result[i][1]]
        labels = np.hstack((events, survival))
        y_train = labels[result[i][0]]
        y_test = labels[result[i][1]]
        X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_cv = scaler.transform(X_cv)
        X_test = scaler.transform(X_test)
        pathway_info = torch.as_tensor(pathway_info).to(device)

        model = Net(X_train.shape[1], 2, pathway_info).to(device)
        pretrained_dict = torch.load(
            ultimate_path + f"/reproduce/reproduction_scripts/tests/models/test7/{task}/{SEED}_{i}net.pth")
        X_train = torch.tensor(X_train).float().to(device)
        y_train = torch.tensor(y_train).float().to(device)
        X_test = torch.tensor(X_test).float().to(device)

        explainer = shap.DeepExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)
        X_test = X_test.cpu().detach().numpy()
        X_tests.append(X_test)
        all_shap.append(shap_values)

ba = np.reshape(all_shap[0][0], (77, 37))
ab = np.reshape(X_tests[0], (77, 37))
for i in range(1, 15):
    ba = np.append(ba, all_shap[i][0], axis=0)
    ab = np.append(ab, X_tests[i], axis=0)
df = pd.DataFrame(ab)


def plot_all_shap_spectrum(shap_values, metabolites):
    metabolites = metabolites / np.amax(metabolites, axis=0, keepdims=True)

    cols_stand = ['2-hydroxyglutarate', '3-hydroxybutyrate', 'Acetate', \
                  'Alanine', 'Allocystathionine', 'Arginine', 'Ascorbate', \
                  'Aspartate', 'Betaine', 'Choline', 'Creatine', \
                  'Ethanolamine', 'GABA', 'Glutamate', 'Glutamine', 'Glutahionine (GSH)', \
                  'Glycerophosphocholine', 'Glycine', 'hypo-Taurine', \
                  'Isoleucine', 'Lactate', 'Leucine', 'Lysine', 'Methionine', \
                  'myo-Inositol', 'NAL', 'NAA', 'o-Acetylcholine', 'Ornithine', \
                  'Phosphocholine', 'Phosphocreatine', 'Proline', \
                  'scyllo-Inositol', 'Serine', 'Taurine', 'Threonine', \
                  'Valine']
    xs = []
    ys = []
    vals = []
    sizes = []
    for i in range(shap_values.shape[1]):
        for j in range(shap_values.shape[0]):
            xs.append((i + 1))
            ys.append(shap_values[j, i])
            vals.append(metabolites[j, i])
            sizes.append(1)

    # scatter plot
    fig = plt.figure()
    res = plt.scatter(xs, ys, c=vals, s=sizes, marker='o', cmap="cool", alpha=0.3)
    # colorbar at right of the figure
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Signal Amplitude', rotation=90)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.ax.text(0.5, -0.01, 'Low', transform=cbar.ax.transAxes,
                 va='top', ha='center')
    cbar.ax.text(0.5, 1.01, 'High', transform=cbar.ax.transAxes,
                 va='bottom', ha='center')

    # .set_xticklabels(cols_stand,rotation = (45), fontsize = 10, va='top', ha='center')

    locs = [i for i in range(1, 38)]
    plt.xticks(locs, cols_stand, rotation=(90), fontsize=8, va='top', ha='center')

    plt.xlabel("Metabolites")
    plt.ylabel("SHAP Value")
    plt.tight_layout()
    return fig


figure = plot_all_shap_spectrum(ba, ab)
plt.savefig(ultimate_path + "/reproduce/reproduction_scripts/figures/Fig4.pdf")
