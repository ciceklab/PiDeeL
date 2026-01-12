import torchtuples as tt
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pathway_integrate import pathway_info
from model import Net
from sklearn import preprocessing 
import pickle
import torch.nn as nn
import torch
from joblib import dump, load
import sys
from pycox.models import CoxPH
import torchtuples as tt
import pdb
from pycox.evaluation import EvalSurv

#argument parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--layer", help="layer number", type=int)
parser.add_argument("--dev", help="device", type=str)
args = parser.parse_args()
layer = args.layer
dev = args.dev


if dev == "gpu":
    device = torch.device("cuda:0")
elif dev == "cpu":
    device = torch.device("cpu")
else:
    print("device must be gpu or cpu")
    sys.exit()


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

with open('../../pideel_data/targeted/predicted_quant.pickle', 'rb') as handle:
    predicted_quant2 = pickle.load(handle)
with open('../../pideel_data/targeted/censorship.pickle', 'rb') as handle:
    censorship = pickle.load(handle)
with open('../../pideel_data/targeted/survival.pickle', 'rb') as handle:
    survival = pickle.load(handle)
with open('../../pideel_data/targeted/tumor_grade.pickle', 'rb') as handle:
    tumor_grade = pickle.load(handle)
with open('../../pideel_data/targeted/type.pickle', 'rb') as handle:
    types = pickle.load(handle)
events =  [True if i == 1 else False for i in censorship]
events = np.reshape(events,(384,1))

censor= [False if i == 1 else True for i in censorship]
censor = np.array(censor)
survival= np.reshape(survival,(384,1))
censor= np.reshape(censor,(384,1))
scaler=load('std_scaler.bin')
#scaler = StandardScaler()
predicted_quant = scaler.transform(predicted_quant2)
features = torch.tensor(predicted_quant).float().to(device)
labels = np.hstack((events,survival))
labels = torch.tensor(labels).float().to(device)
pathway_info = torch.as_tensor(pathway_info).to(device)
model = Net(37,1,pathway_info).to(device)


model.load_state_dict(torch.load(f"PiDeeL_{layer}layer.pth"))


with open('quant_test.pickle', 'rb') as handle:
    quant_test = pickle.load(handle)

#pdb.set_trace()
#quant_test = (np.mean(predicted_quant2) / np.mean(quant_test) ) * quant_test 

quant_test= scaler.transform(quant_test)
quant_test = torch.as_tensor(quant_test).float().to(device)
#print(quant_test)
#print(model(quant_test))


model1 = CoxPH(model, tt.optim.Adam)
_ = model1.compute_baseline_hazards(input=features,target=(labels[:,1],labels[:,0]))








surv = model1.predict_surv_df(quant_test)

"""
labels_test_sim_durations = [100, 2000, 3000, 2000, 1500, 150, 130, 110, 115, 125, 113, 135, 341, 121, 451, 451, 651, 121, 651, 781, 681, 2000, 2600, 5000, 4000, 3000]
labels_test_sim_durations= np.array(labels_test_sim_durations)
labels_test_sim_event = [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
labels_test_sim_event = np.array(labels_test_sim_event)
"""
indices = []
test_types = ["AST", "OAST", "OAST","OAST","OAST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","OAST"]
test_types = np.array(test_types)
SEEDS = [6, 35, 81]
for SEED in SEEDS:
    np.random.seed(SEED)
    for j in range(10):
        labels_test_sim_durations = []
        labels_test_sim_event = []
        labels_test_grades = [3,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,2]
        for k, item in enumerate(labels_test_grades):
            while True:
                sim_dur = np.random.normal(loc =np.mean(survival[(types == test_types[k]) & (tumor_grade == str(item))]),scale = np.std(survival[(types == test_types[k]) & (tumor_grade == str(item))]))
                if sim_dur > np.min(survival[(types == test_types[k]) & (tumor_grade == str(item))]) and sim_dur <= np.max(survival[(types == test_types[k]) & (tumor_grade == str(item))]):
                    break
            labels_test_sim_durations.append(sim_dur)
            labels_test_sim_event.append(np.random.binomial(n=1,p=1-np.sum([(censorship == 0) & (tumor_grade == str(item)) & (types == test_types[k]) ]) / np.sum([(tumor_grade==str(item)) & (types == test_types[k])])))
        labels_test_sim_durations= np.array(labels_test_sim_durations)
        labels_test_sim_event = np.array(labels_test_sim_event)

        #print(labels_test_sim_durations)
        #print(labels_test_sim_event)
        #pdb.set_trace()
        #print(model(quant_test))
        #print(surv)

        ev = EvalSurv(surv,  labels_test_sim_durations, labels_test_sim_event, censor_surv='km')
        if ev.concordance_td() >= 0.5:
            indices.append(ev.concordance_td())

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
print(np.median(indices))
print(np.min(indices))
print(np.max(indices))

import matplotlib.pyplot as plt
concordence= { "Simulation": indices}
fig, ax = plt.subplots()
bp = ax.boxplot(concordence.values(), patch_artist=True, showfliers=True)
ax.set_ylabel("C-Index",fontsize=12)
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
ax.set_xticks([1,2])
ax.set_xticklabels(["Simulated\n independent data"],fontsize=12)

ax.set_ylim(0.5, 0.8)
plt.tight_layout()


fig.savefig("a2.png")
"""

indices = []
test_types = ["AST", "OAST", "OAST","OAST","OAST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","OAST"]
test_types = np.array(test_types)
SEEDS = [6, 35, 81]
for SEED in SEEDS:
    np.random.seed(SEED)
    for j in range(10):
        labels_test_sim_durations = []
        labels_test_sim_event = []
        labels_test_grades = [3,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,2]
        for k, item in enumerate(labels_test_grades):
            while True:
                sim_dur = np.random.normal(loc =np.mean(survival[(types == test_types[k]) & (tumor_grade == str(item))]),scale = np.std(survival[(types == test_types[k]) & (tumor_grade == str(item))]))
                if sim_dur > np.min(survival[(types == test_types[k]) & (tumor_grade == str(item))]) and sim_dur <= np.max(survival[(types == test_types[k]) & (tumor_grade == str(item))]):
                    break
            labels_test_sim_durations.append(sim_dur)
            labels_test_sim_event.append(np.random.binomial(n=1,p=1-np.sum([(censorship == 0) & (tumor_grade == str(item)) & (types == test_types[k]) ]) / np.sum([(tumor_grade==str(item)) & (types == test_types[k])])))
        labels_test_sim_durations= np.array(labels_test_sim_durations)
        labels_test_sim_event = np.array(labels_test_sim_event)

        #print(labels_test_sim_durations)
        #print(labels_test_sim_event)
        #pdb.set_trace()
        #print(model(quant_test))
        #print(surv)

        ev = EvalSurv(surv,  labels_test_sim_durations, labels_test_sim_event, censor_surv='km')
        if ev.concordance_td() >= 0.5:
            indices.append(ev.concordance_td())"""