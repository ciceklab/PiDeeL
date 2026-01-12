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
#import torchtuples as tt
import pdb
from pycox.evaluation import EvalSurv



with open('../../pideel_data/targeted/grade.pickle', 'rb') as handle:
    grade = pickle.load(handle)

with open('../../pideel_data/targeted/censorship.pickle', 'rb') as handle:
    censorship = pickle.load(handle)

with open('../../pideel_data/targeted/survival.pickle', 'rb') as handle:
    survival = pickle.load(handle)
with open('../../pideel_data/targeted/type.pickle', 'rb') as handle:
    types = pickle.load(handle)

with open('../../pideel_data/targeted/tumor_grade.pickle', 'rb') as handle:
    tumor_grade = pickle.load(handle)
pdb.set_trace()
print(tumor_grade)
test_types = ["AST", "DNET", "DNET","DNET","DNET","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST","AST"]

surv_grade1 = np.random.normal(loc =np.mean(survival[tumor_grade == "1"]),scale = np.std(survival[tumor_grade == "1"]))
surv_grade2 = np.random.normal(loc =np.mean(survival[tumor_grade == "2"]),scale = np.std(survival[tumor_grade == "2"]))
surv_grade3 = np.random.normal(loc =np.mean(survival[tumor_grade == "3"]),scale = np.std(survival[tumor_grade == "3"]))
surv_grade4 = np.random.normal(loc =np.mean(survival[tumor_grade == "4"]),scale = np.std(survival[tumor_grade == "4"]))

event_grade1 = np.random.binomial(n=1,p=1-np.sum([(censorship == 0) & (tumor_grade == "1")]) / np.sum([tumor_grade=="1"]))
event_grade2 = np.random.binomial(n=1,p=1-np.sum([(censorship == 0) & (tumor_grade == "2")]) / np.sum([tumor_grade=="2"]))
event_grade3 = np.random.binomial(n=1,p=1-np.sum([(censorship == 0) & (tumor_grade == "3")]) / np.sum([tumor_grade=="3"]))
event_grade4 = np.random.binomial(n=1,p=1-np.sum([(censorship == 0) & (tumor_grade == "4")]) / np.sum([tumor_grade=="4"]))