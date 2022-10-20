import sys
import pickle

sys.path.insert(1, "../../")
sys.path.insert(1, "../../../")
sys.path.insert(1, "../../../../")
sys.path.insert(1, "../../../../../")
sys.path.insert(1, "../../../../../../")

from hyper_config import *

with open("../scripts/pred_quant.pickle", "rb") as f:
    pred_quant = pickle.load(f)
with open("../scripts/grade.pickle", "rb") as f:
    grade = pickle.load(f)
with open("../scripts/events.pickle", "rb") as f:
    events = pickle.load(f)
with open("../scripts/survival.pickle", "rb") as f:
    survival = pickle.load(f)
with open("../scripts/pathway_info.pickle", "rb") as f:
    pathway_info = pickle.load(f)




from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import shap
import pdb
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
        return pathway


def initialize_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Net2(nn.Module):
    def __init__(self, input_dim, output_dim, pathway_info):
        super(Net2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = pathway_info.shape[1]
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.pathway_info = pathway_info.float()

    def forward(self, x):
        hid = F.relu(self.fc2(x))
        out = F.relu((self.fc3(hid)))
        out = ((self.fc4(out)))
        return out


def initialize_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
all_shap = []
pathway_tests = []

row_stand = ['Butanoate metabolism', 'C5-Branched dibasic acid metabolism', 'Metabolic pathways',
             'cAMP signaling pathway', 'Glycolysis / Gluconeogenesis', 'Taurine and hypotaurine metabolism',
             'Phosphonate and phosphinate metabolism', 'Glycosaminoglycan biosynthesis - heparan sulfate / heparin',
             'Pyruvate metabolism', 'Glyoxylate and dicarboxylate metabolism', 'Methane metabolism',
             'Carbon fixation pathways in prokaryotes', 'Zeatin biosynthesis', 'Sulfur metabolism',
             'Biosynthesis of secondary metabolites', 'Microbial metabolism in diverse environments',
             'Carbon metabolism', 'Degradation of aromatic compounds', 'Cholinergic synapse', 'Alcoholic liver disease',
             'Carbohydrate digestion and absorption', 'Protein digestion and absorption',
             'Alanine| aspartate and glutamate metabolism', 'Cysteine and methionine metabolism',
             'Selenocompound metabolism', 'D-Amino acid metabolism', 'Carbon fixation in photosynthetic organisms',
             'Aminoacyl-tRNA biosynthesis', 'Biosynthesis of various other secondary metabolites',
             'Biosynthesis of plant secondary metabolites', 'Biosynthesis of amino acids', 'Vancomycin resistance',
             'ABC transporters', 'Sulfur relay system', 'Mineral absorption', 'Central carbon metabolism in cancer',
             'Glycine| serine and threonine metabolism', 'Biosynthesis of cofactors', 'Arginine biosynthesis',
             'Monobactam biosynthesis', 'Arginine and proline metabolism', 'Clavulanic acid biosynthesis',
             'Biosynthesis of various antibiotics',
             'Biosynthesis of alkaloids derived from ornithine| lysine and nicotinic acid', 'mTOR signaling pathway',
             'Amyotrophic lateral sclerosis', 'Pathways of neurodegeneration - multiple diseases', 'Chagas disease',
             'Amoebiasis', 'Ascorbate and aldarate metabolism', 'Glutathione metabolism',
             'Phosphotransferase system (PTS)', 'HIF-1 signaling pathway', 'Vitamin digestion and absorption',
             'Lysine biosynthesis', 'Histidine metabolism', 'beta-Alanine metabolism', 'Cyanoamino acid metabolism',
             'Nicotinate and nicotinamide metabolism', 'Pantothenate and CoA biosynthesis',
             'Biosynthesis of various plant secondary metabolites', 'Biosynthesis of plant hormones',
             '2-Oxocarboxylic acid metabolism', 'Two-component system', 'Bacterial chemotaxis',
             'Neuroactive ligand-receptor interaction', 'Teichoic acid biosynthesis', 'Glycerophospholipid metabolism',
             'Bile secretion', 'Choline metabolism in cancer', 'Retrograde endocannabinoid signaling', 'Quorum sensing',
             'Synaptic vesicle cycle', 'GABAergic synapse', 'Taste transduction', 'Estrogen signaling pathway',
             'GnRH secretion', 'Morphine addiction', 'Nicotine addiction', 'Carbapenem biosynthesis',
             'Neomycin| kanamycin and gentamicin biosynthesis', 'Porphyrin metabolism', 'Nitrogen metabolism',
             'FoxO signaling pathway', 'Phospholipase D signaling pathway', 'Ferroptosis', 'Gap junction',
             'Circadian entrainment', 'Long-term potentiation', 'Glutamatergic synapse', 'Long-term depression',
             'Proximal tubule bicarbonate reclamation', 'Huntington disease', 'Spinocerebellar ataxia',
             'Cocaine addiction', 'Amphetamine addiction', 'Alcoholism', 'Purine metabolism', 'Pyrimidine metabolism',
             'Vitamin B6 metabolism', 'Nucleotide metabolism', 'Thyroid hormone synthesis',
             'Chemical carcinogenesis - reactive oxygen species', 'Diabetic cardiomyopathy', 'Ether lipid metabolism',
             'Primary bile acid biosynthesis', 'Lysine degradation', 'Thiamine metabolism',
             'Biofilm formation - Escherichia coli', 'Valine| leucine and isoleucine degradation',
             'Valine| leucine and isoleucine biosynthesis', 'Tropane| piperidine and pyridine alkaloid biosynthesis',
             'Glucosinolate biosynthesis', 'Shigellosis', 'Fructose and mannose metabolism', 'Propanoate metabolism',
             'Styrene degradation', 'Glucagon signaling pathway',
             'Biosynthesis of alkaloids derived from histidine and purine', 'Biotin metabolism',
             'Antifolate resistance', 'Galactose metabolism', 'Streptomycin biosynthesis',
             'Inositol phosphate metabolism', 'Biosynthesis of nucleotide sugars',
             'Phosphatidylinositol signaling system', 'Regulation of actin cytoskeleton', 'Insulin secretion',
             'Salivary secretion', 'Gastric acid secretion', 'Pancreatic secretion', 'Prodigiosin biosynthesis',
             'Novobiocin biosynthesis', 'Staurosporine biosynthesis', 'Sphingolipid metabolism',
             'Sphingolipid signaling pathway', 'Penicillin and cephalosporin biosynthesis',
             'Biosynthesis of alkaloids derived from shikimate pathway']

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
        model2 = Net2(X_train.shape[1], 2, pathway_info).to(device)
        pretrained_dict = torch.load(
            ultimate_path + f"/reproduce/reproduction_scripts/tests/models/test7/{task}/{SEED}_{i}net.pth")
        # keys = ['fc1.weight', 'fc1.bias']
        # dict_for1 = {x:pretrained_dict[x] for x in keys}
        # keys = ['fc2.weight', 'fc2.bias']
        # dict_for2 = {x:pretrained_dict[x] for x in keys}
        X_train = torch.tensor(X_train).float().to(device)
        y_train = torch.tensor(y_train).float().to(device)
        X_test = torch.tensor(X_test).float().to(device)

        model.load_state_dict(pretrained_dict)
        model2.load_state_dict(pretrained_dict)

        pathway = model.forward(X_train)
        pathway_test = model.forward(X_test)
        explainer = shap.DeepExplainer(model2, pathway)
        shap_values = explainer.shap_values(pathway_test)

        pathway_test = pathway_test.cpu().detach().numpy()
        pathway_tests.append(pathway_test)
        all_shap.append(shap_values)

ba = np.reshape(all_shap[0][0], (77, 138))
ab = np.reshape(pathway_tests[0], (77, 138))
for i in range(1, 15):
    ba = np.append(ba, all_shap[i][0], axis=0)
    ab = np.append(ab, pathway_tests[i], axis=0)
df = pd.DataFrame(ab)


mean_shap = np.mean(ba, axis=0)
# indices of the biggest 30 mean shap values
indices = np.argpartition(mean_shap, -20)[-20:]
for i in indices:
    print(row_stand[i], mean_shap[i])

shap.summary_plot(ba, df, feature_names=row_stand, max_display=10)
_, h = plt.gcf().get_size_inches()
plt.gcf().set_size_inches(h * 4, h)
plt.tight_layout()
plt.savefig(ultimate_path + "/reproduce/reproduction_scripts/figures/shap_path.pdf")
