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


with open('sample_quant.pickle', 'rb') as handle:
    sample_quant = pickle.load(handle)

scaler=load('std_scaler.bin')
sample_quant= scaler.transform(sample_quant)
sample_quant = torch.as_tensor(sample_quant).float().to(device)
pathway_info = torch.as_tensor(pathway_info).to(device)
model = Net(37,1,pathway_info).to(device)
if dev == "cpu":
    model.load_state_dict(torch.load(f"PiDeeL_{layer}layer.pth",map_location=torch.device('cpu') ))
else:
    model.load_state_dict(torch.load(f"PiDeeL_{layer}layer.pth"))
risk_scores = model(sample_quant)
print(risk_scores)
