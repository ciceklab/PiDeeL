import sys
from pathlib import Path

# Setup paths FIRST before other imports
FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import torchtuples as tt
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pathway_integrate import pathway_info
from sklearn import preprocessing 
import pickle
from joblib import dump, load
from pycox.models import CoxPH, DeepHitSingle, PCHazard
import pdb
from pycox.evaluation import EvalSurv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, RandomSurvivalForest

from load_targeted_data import df

#argument parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--layer", help="layer number", type=int)
args = parser.parse_args()
layer = args.layer
if layer is None:
    print("No layer specified, defaulting to layer 4")
    layer = 4


device = torch.device("cuda:0")



if layer == 2:
    class Net(nn.Module):
        def __init__(self, input_dim, output_dim, pathway_info):
            super(Net, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_dim = pathway_info.shape[1]
            self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
            self.pathway_info = pathway_info.float()

        def forward(self, x):
            pathway = torch.matmul(x, (self.pathway_info * self.fc1.weight.t())) + self.fc1.bias
            pathway = F.relu(pathway)
            out = (self.fc2(pathway))
            return out
        
        
    def initialize_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)

elif layer == 3:
    class Net(nn.Module):
        def __init__(self, input_dim, output_dim, pathway_info):
            super(Net, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_dim = pathway_info.shape[1]
            self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, 64)
            self.fc3 = nn.Linear(64, output_dim)
            self.pathway_info = pathway_info.float()

        def forward(self, x):
            pathway = torch.matmul(x, (self.pathway_info * self.fc1.weight.t())) + self.fc1.bias
            pathway = F.relu(pathway)
            hid = F.relu(self.fc2(pathway))
            out = ((self.fc3(hid)))
            return out
        
        
    def initialize_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)

elif layer == 4:
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
        
        
    def initialize_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
else:
    print("layer number must be 2, 3, or 4")
    sys.exit()


class Net_DS(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net_DS, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        hid = F.relu(self.fc1(x))
        hid = F.relu(self.fc2(hid))
        hid = F.relu(self.fc3(hid))
        out = (self.fc4(hid))
        return out
    
    
class Net_DH(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net_DH, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        hid = F.relu(self.fc1(x))
        hid = F.relu(self.fc2(hid))
        hid = F.relu(self.fc3(hid))
        out = (self.fc4(hid))
        return out

class Net_PC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net_PC, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        hid = F.relu(self.fc1(x))
        hid = F.relu(self.fc2(hid))
        hid = F.relu(self.fc3(hid))
        out = (self.fc4(hid))
        return out

with open(REPO_ROOT / 'pideel_data/targeted/predicted_quant.pickle', 'rb') as handle:
    predicted_quant2 = pickle.load(handle)
with open(REPO_ROOT / 'pideel_data/targeted/censorship.pickle', 'rb') as handle:
    censorship = pickle.load(handle)
with open(REPO_ROOT / 'pideel_data/targeted/survival.pickle', 'rb') as handle:
    survival = pickle.load(handle)
with open(REPO_ROOT / 'pideel_data/targeted/tumor_grade.pickle', 'rb') as handle:
    tumor_grade = pickle.load(handle)
with open(REPO_ROOT / 'pideel_data/targeted/type.pickle', 'rb') as handle:
    types = pickle.load(handle)
events =  [True if i == 1 else False for i in censorship]
events = np.reshape(events,(384,1))

censor= [False if i == 1 else True for i in censorship]
censor = np.array(censor)
survival= np.reshape(survival,(384,1))
censor= np.reshape(censor,(384,1))
#scaler=load('std_scaler.bin')
scaler = StandardScaler()
predicted_quant = scaler.fit_transform(predicted_quant2)
features = torch.tensor(predicted_quant).float().to(device)
labels = np.hstack((events,survival))
labels = torch.tensor(labels).float().to(device)
pathway_info = torch.as_tensor(pathway_info).to(device)


df3 = df.drop(columns=["duration","event"],axis=1)
X_rf = df3
y_rf = []
for i in range(len(df["event"])):
    a = (bool(df["event"][i]), float(df["duration"][i]))
    y_rf.append(a)
y_rf = np.array(y_rf, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])


with open(REPO_ROOT / 'test_scripts/test31/quant_test.pickle', 'rb') as handle:
    quant_test = pickle.load(handle)

#pdb.set_trace()
quant_test = (np.mean(predicted_quant2) / np.mean(quant_test) ) * quant_test 
quant_test= scaler.transform(quant_test)
quant_test2 = quant_test
quant_test = torch.as_tensor(quant_test).float().to(device)

model_ds = Net_DS(37,1).to(device)
model_ds.load_state_dict(torch.load(REPO_ROOT / "models/4layer/no_pathway/35_3net.pth"))
model_ds1 = CoxPH(model_ds, tt.optim.Adam)
_2 = model_ds1.compute_baseline_hazards(input=features,target=(labels[:,1],labels[:,0]))
surv_ds = model_ds1.predict_surv_df(quant_test)

model_dh = Net_DH(37,10).to(device)
model_dh.load_state_dict(torch.load(REPO_ROOT / "test_scripts/test29/4layer/no_pathway/35_3net.pth"))
model_dh1 = DeepHitSingle(model_dh, tt.optim.Adam)
#_3 = model_dh1.compute_baseline_hazards(input=features,target=(labels[:,1],labels[:,0]))
surv_dh = model_dh1.predict_surv_df(quant_test)

model_pc = Net_PC(37,10).to(device)
model_pc.load_state_dict(torch.load(REPO_ROOT / "test_scripts/test30/4layer/no_pathway/35_3net.pth"))
model_pc1 = PCHazard(model_pc, tt.optim.Adam)
#_4 = model_pc1.compute_baseline_hazards(input=features,target=(labels[:,1],labels[:,0]))
surv_pc = model_pc1.predict_surv_df(quant_test)



model = Net(37,1,pathway_info).to(device)
model.load_state_dict(torch.load(REPO_ROOT / f"test_scripts/test31/PiDeeL_{layer}layer.pth"))
model1 = CoxPH(model, tt.optim.Adam)
_ = model1.compute_baseline_hazards(input=features,target=(labels[:,1],labels[:,0]))
surv = model1.predict_surv_df(quant_test)

cox = CoxPHSurvivalAnalysis()
cox.fit(X_rf, y_rf)

rsf = RandomSurvivalForest()
rsf.fit(X_rf, y_rf)


cwgb = ComponentwiseGradientBoostingSurvivalAnalysis()
cwgb.fit(X_rf, y_rf)
"""
labels_test_sim_event = [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
labels_test_sim_event = np.array(labels_test_sim_event)
"""
pideel_indices = []
cox_indices = []
rsf_indices = []
cwgb_indices = []
deepsurv_indices = []
deephit_indices = []
pc_indices = []
test_types = ["AST", "OAST", "OAST","OAST","OAST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","OAST"]
test_types = np.array(test_types)
SEEDS = [81]
for SEED in SEEDS:
    np.random.seed(SEED)
    for j in range(10):
        labels_test_sim_durations = []
        labels_test_sim_event = []
        labels_test_grades = [3,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,2]
        for k, item in enumerate(labels_test_grades):
            #while True:
            #sim_dur = np.random.normal(loc =np.mean(survival[(types == test_types[k]) & (tumor_grade == str(item))]),scale = np.std(survival[(types == test_types[k]) & (tumor_grade == str(item))]))
            sim_dur = np.random.uniform(low =np.min(survival[(types == test_types[k]) & (tumor_grade == str(item))]),high = np.max(survival[(types == test_types[k]) & (tumor_grade == str(item))]))
            #if sim_dur > np.min(survival[(types == test_types[k]) & (tumor_grade == str(item))]) and sim_dur <= np.max(survival[(types == test_types[k]) & (tumor_grade == str(item))]):
            #    break
            labels_test_sim_durations.append(sim_dur)
            labels_test_sim_event.append(np.random.binomial(n=1,p=1-np.sum([(censorship == 0) & (tumor_grade == str(item)) & (types == test_types[k]) ]) / np.sum([(tumor_grade==str(item)) & (types == test_types[k])])))
        labels_test_sim_durations= np.array(labels_test_sim_durations)
        labels_test_sim_event = np.array(labels_test_sim_event)
        #print(labels_test_sim_durations)
        #print(labels_test_sim_event)
        #pdb.set_trace()
        #print(model(quant_test))
        #print(surv)
        ev = EvalSurv(surv_dh,  labels_test_sim_durations, labels_test_sim_event, censor_surv='km')
        deephit_indices.append(ev.concordance_td())
        ev = EvalSurv(surv_pc,  labels_test_sim_durations, labels_test_sim_event, censor_surv='km')
        pc_indices.append(ev.concordance_td())
        ev = EvalSurv(surv_ds,  labels_test_sim_durations, labels_test_sim_event, censor_surv='km')
        deepsurv_indices.append(ev.concordance_td())
        ev = EvalSurv(surv,  labels_test_sim_durations, labels_test_sim_event, censor_surv='km')
        pideel_indices.append(ev.concordance_td())
        y_test = np.stack((labels_test_sim_event,labels_test_sim_durations),axis = 1)
        test_labels = np.array([(True,y) if x==1 else (False,y)\
            for x,y in zip(y_test[:,0], y_test[:,1])],\
            dtype=[('event', np.bool_), ('surv', np.int32)])
        cox_indices.append(cox.score(quant_test2, test_labels))
        rsf_indices.append(rsf.score(quant_test2, test_labels))
        cwgb_indices.append(cwgb.score(quant_test2, test_labels))
        
concordence= {"Cox-PH" : cox_indices, "CWGB" : cwgb_indices, "RSF": rsf_indices, "DeepHit": deephit_indices, "PC-Hazard": pc_indices,"DeepSurv": deepsurv_indices, "PiDeeL": pideel_indices}
print(concordence)
"""
surv_grade1 = np.random.normal(loc =np.mean(survival[tumor_grade == "1"]),scale = np.std(survival[tumor_grade == "1"]))
surv_grade2 = np.random.normal(loc =np.mean(survival[tumor_grade == "2"]),scale = np.std(survival[tumor_grade == "2"]))
surv_grade3 = np.random.normal(loc =np.mean(survival[tumor_grade == "3"]),scale = np.std(survival[tumor_grade == "3"]))
surv_grade4 = np.random.normal(loc =np.mean(survival[tumor_grade == "4"]),scale = np.std(survival[tumor_grade == "4"]))

event_grade1 = np.random.binomial(n=1,p=1-np.sum([(censorship == 0) & (tumor_grade == "1")]) / np.sum([tumor_grade=="1"]))
event_grade2 = np.random.binomial(n=1,p=1-np.sum([(censorship == 0) & (tumor_grade == "2")]) / np.sum([tumor_grade=="2"]))
event_grade3 = np.random.binomial(n=1,p=1-np.sum([(censorship == 0) & (tumor_grade == "3")]) / np.sum([tumor_grade=="3"]))
event_grade4 = np.random.binomial(n=1,p=1-np.sum([(censorship == 0) & (tumor_grade == "4")]) / np.sum([tumor_grade=="4"]))
"""



#print(indices)
#print(np.median(indices))
#print(np.min(indices))
#print(np.max(indices))
concordence= {"DeepHit": deephit_indices, "PC-Hazard": pc_indices,"DeepSurv": deepsurv_indices, "PiDeeL": pideel_indices}
tasks = ["DeepHit", "PC-Hazard","DeepSurv", "PiDeeL"]

for task in tasks:
    print(f"ablation6 = {task}, median = {np.median(concordence[task])} , mean = {np.mean(concordence[task])}, std = {np.std(concordence[task])}")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
color_palette_dct = {
"DeepHit": '#008080',
"PC-Hazard": '#008080',
"DeepSurv": '#008080',
"PiDeeL": '#2f4f4f',}
fig, ax = plt.subplots()

bp = ax.boxplot(concordence.values(), patch_artist=True, showfliers=True)

color_palette_dct = {name:color_palette_dct[name] for name in concordence.keys()}
for idx, model in enumerate(concordence.keys()):
    bp['boxes'][idx].set(color=color_palette_dct[model])
    bp['boxes'][idx].set(facecolor=color_palette_dct[model])
ax.set_ylabel("C-Index",fontsize=12)
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
#ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(concordence.keys(),fontsize=12, rotation=90)


ax.set_ylim(0.2, 0.7)
fontP = FontProperties()
fontP.set_size('x-small')
plt.tight_layout()


fig.savefig("fig6_external_validation.png")
fig.savefig("fig6_external_validation.pdf")