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

class Net2(nn.Module):
    def __init__(self, input_dim, output_dim, pathway_info):
        super(Net2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = pathway_info.shape[1]
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
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