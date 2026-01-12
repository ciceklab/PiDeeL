import sys
from pathlib import Path
TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent.parent

import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

import pdb

with open(REPO_ROOT / 'pideel_data/targeted/predicted_quant.pickle', 'rb') as handle:
    predicted_quant = pickle.load(handle)
with open(REPO_ROOT / 'pideel_data/targeted/age.pickle', 'rb') as handle:
    ages = pickle.load(handle)
with open(REPO_ROOT / 'pideel_data/targeted/grade.pickle', 'rb') as handle:
    grade = pickle.load(handle)
with open(REPO_ROOT / 'pideel_data/targeted/censorship.pickle', 'rb') as handle:
    censorship = pickle.load(handle)
with open(REPO_ROOT / 'pideel_data/targeted/survival.pickle', 'rb') as handle:
    survival = pickle.load(handle)
with open(REPO_ROOT / 'pideel_data/targeted/age_no_imputation.pickle', 'rb') as handle:
    age_no_imputation = pickle.load(handle)
#events = [False if i == 1 else True for i in censorship]
events =  [True if i == 1 else False for i in censorship]
events = np.reshape(events,(384,1))

d = {'duration': pd.to_numeric(survival), 'event': events}

censor= [False if i == 1 else True for i in censorship]
censor = np.array(censor)
survival= np.reshape(survival,(384,1))
censor= np.reshape(censor,(384,1))

list_of_targeted_metabolites =['2-hydroxyglutarate', '3-hydroxybutyrate', 'Acetate',\
       'Alanine', 'Allocystathionine', 'Arginine', 'Ascorbate',\
       'Aspartate', 'Betaine', 'Choline', 'Creatine',\
       'Ethanolamine', 'GABA', 'Glutamate', 'Glutamine', 'Glutahionine (GSH)',\
       'Glycerophosphocholine', 'Glycine', 'hypo-Taurine',\
       'Isoleucine', 'Lactate', 'Leucine', 'Lysine', 'Methionine',\
       'myo-Inositol', 'NAL', 'NAA', 'o-Acetylcholine', 'Ornithine',\
       'Phosphocholine', 'Phosphocreatine', 'Proline',\
       'scyllo-Inositol', 'Serine', 'Taurine', 'Threonine',\
       'Valine']

age_no_imputation = [np.nan if i == "remove" else i for i in age_no_imputation]

for i, metabolite in enumerate(list_of_targeted_metabolites):
    d[metabolite] = pd.to_numeric(predicted_quant[:,i])
df = pd.DataFrame.from_dict(d, orient='index')
df = df.transpose()
for column in df:
    df[column] = pd.to_numeric(df[column],errors = 'coerce')



pathway_info=[]
with open(REPO_ROOT / 'scripts/result.csv') as f:
    for index, lines in enumerate(f):
        if index == 0:
            continue

        else:
            tokens = lines.split(",")
            pathway_info.append(tokens[2:])

pathway_info = np.reshape(pathway_info,(37,len(pathway_info[0])))
pathway_info = pathway_info.astype(np.int64)
process_info=[]
with open(REPO_ROOT / 'scripts/result2.csv') as f:
    for index, lines in enumerate(f):
        if index == 0:
            continue
        else:
            tokens = (lines.split(","))
            process_info.append(tokens[2:])

process_info = np.reshape(process_info,(len(process_info),len(process_info[0])))
process_info = process_info.astype(np.int64)
type_info=[]
with open(REPO_ROOT / 'scripts/result3.csv') as f:
    for index, lines in enumerate(f):
        if index == 0:
            continue
        else:
            tokens = (lines.strip().split(","))
            type_info.append(tokens[-1:-6:-1])
            

type_info = np.reshape(type_info,(len(type_info),len(type_info[0])))
type_info = type_info.astype(np.int64)


process_info2=[]
with open(REPO_ROOT / 'scripts/result4.csv') as f:
    for index, lines in enumerate(f):
        if index == 0:
            continue
        else:
            tokens = (lines.split(","))
            process_info2.append(tokens[2:])

process_info2 = np.reshape(process_info2,(len(process_info2),len(process_info2[0])))
process_info2 = process_info2.astype(np.int64)

