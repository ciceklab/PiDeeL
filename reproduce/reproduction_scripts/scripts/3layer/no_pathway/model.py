import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        hid = F.relu(self.fc1(x))
        hid = F.relu(self.fc2(hid))
        out = (self.fc3(hid))
        return out
    
    
    
def initialize_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)