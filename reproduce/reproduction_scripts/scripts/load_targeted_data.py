import pdb
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
import sys
import pynmr
import pynmr.model.parser.topSpin as T
import pynmr.model.processor as P
import pynmr.model.operations as O
from torch import nn
device = torch.device("cpu")
sys.path.insert(1, "../")
sys.path.insert(1, "../../")
from hyper_config import *

# file_path print
# print(ultimate_path)
excel_file_path = os.path.join(ultimate_path, "reproduce/data/Dataset_Labels.xlsx")
# excel_path print
print(excel_file_path, " xlxs loaded")
# excel read
df = pd.read_excel(excel_file_path)
# excel read print
# print(df)

# fid name from excel to numpy array
fid_sample_name = df["FID sample filename"].to_numpy()
# print(fid_sample_name)
# duration from excel to numpy array
survival = df["Duration (between the surgery and last control or decease)"].to_numpy()
# event from excel to numpy array
events = df["Event (0 = survived 1 = deceased)"].to_numpy()
# pathological grade from excel to numpy array
grade = df["Pathological Class"].to_numpy()
grade = [0 if i == "Benign Glioma" else 1 for i in grade]


def preprocess_spectrum(path):
    # print(path, " loaded")
    #return np.absolute(data.allFid[3][0])

    data = T.TopSpin(path)
    Processor = P.Processor([O.LeftShift(70),
                         O.LineBroadening(10),
                         O.FourierTransform(),
                         O.Phase0D(190)])
    Processor.runStack(data)

    #pdb.set_trace()
    return np.absolute(data.allFid[2][0])




spectra = []
for fid_sample in fid_sample_name:
    path = os.path.join(ultimate_path, "reproduce/data/FID-samples/{}/4/".format(fid_sample))
    if not os.path.exists(path):
        print("File not found: {}".format(path))
        continue

    spectra.append(preprocess_spectrum(path).astype(float))
print(len(spectra), "FID samples loaded")

# predict quantification per fold
folder2dataset = {
    "2-hg": "2-hydroxyglutarate",
    "3-hb": "Hydroxybutyrate",
    "acetate": "Acetate",
    "alanine": "Alanine",
    "allocystathionine": "Allocystathionine",
    "arginine": "Arginine",
    "ascorbate": "Ascorbate",
    "aspartate": "Aspartate",
    "betaine": "Betaine",
    "choline": "Choline",
    "creatine": "Creatine",
    "ethanolamine": "Ethanolamine",
    "gaba": "GABA",
    "glutamate": "Glutamate",
    "glutamine": "Glutamine",
    "glutathionine": "GSH",
    "glycerophosphocholine": "Glycerophosphocholine",
    "glycine": "Glycine",
    "hypotaurine": "Hypotaurine",
    "isoleucine": "Isoleucine",
    "lactate": "Lactate",
    "leucine": "Leucine",
    "lysine": "Lysine",
    "methionine": "Methionine",
    "myoinositol": "Myoinositol",
    "NAA": "NAA",
    "NAL": "NAL",
    "o-acetylcholine": "O-acetylcholine",
    "ornithine": "Ornithine",
    "phosphocholine": "Phosphocholine",
    "phosphocreatine": "Phosphocreatine",
    "proline": "Proline",
    "scylloinositol": "Scylloinositol",
    "serine": "Serine",
    "taurine": "Taurine",
    "threonine": "Threonine",
    "valine": "Valine"
}
dataset2folder = {value: key for key, value in folder2dataset.items()}


# Single Metabolite Quantification model
class Single_Metabolite_Model(nn.Module):
    def __init__(self):
        super(Single_Metabolite_Model, self).__init__()
        self.all_mutual = nn.Linear(1401, 192)
        self.m1 = nn.Linear(192, 1)

    def forward(self, x):
        inp = F.relu(self.all_mutual(x))
        m1 = F.relu(self.m1(inp)).squeeze()
        return m1


# Multiple Metabolite Quantification Wrapper model
model_load_base_path = os.path.join(ultimate_path, "reproduce/reproduction_scripts/quantification_models/")


class QuantificationWrapper(nn.Module):
    def __init__(self, quantifiers):
        super(QuantificationWrapper, self).__init__()
        self.quantifiers = quantifiers

    def forward(self, x):
        q0 = self.quantifiers[0](x)
        q1 = self.quantifiers[1](x)
        q2 = self.quantifiers[2](x)
        q3 = self.quantifiers[3](x)
        q4 = self.quantifiers[4](x)
        q5 = self.quantifiers[5](x)
        q6 = self.quantifiers[6](x)
        q7 = self.quantifiers[7](x)
        q8 = self.quantifiers[8](x)
        q9 = self.quantifiers[9](x)
        q10 = self.quantifiers[10](x)
        q11 = self.quantifiers[11](x)
        q12 = self.quantifiers[12](x)
        q13 = self.quantifiers[13](x)
        q14 = self.quantifiers[14](x)
        q15 = self.quantifiers[15](x)
        q16 = self.quantifiers[16](x)
        q17 = self.quantifiers[17](x)
        q18 = self.quantifiers[18](x)
        q19 = self.quantifiers[19](x)
        q20 = self.quantifiers[20](x)
        q21 = self.quantifiers[21](x)
        q22 = self.quantifiers[22](x)
        q23 = self.quantifiers[23](x)
        q24 = self.quantifiers[24](x)
        q25 = self.quantifiers[25](x)
        q26 = self.quantifiers[26](x)
        q27 = self.quantifiers[27](x)
        q28 = self.quantifiers[28](x)
        q29 = self.quantifiers[29](x)
        q30 = self.quantifiers[30](x)
        q31 = self.quantifiers[31](x)
        q32 = self.quantifiers[32](x)
        q33 = self.quantifiers[33](x)
        q34 = self.quantifiers[34](x)
        q35 = self.quantifiers[35](x)
        q36 = self.quantifiers[36](x)

        return torch.stack(
            [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20, q21, q22,
             q23, q24, q25, q26, q27, q28, q29, q30, q31, q32, q33, q34, q35, q36]).T


pred_quant = np.zeros((384, 37))
quantifiers = []
for name in dataset2folder.keys():
    state_dct = torch.load(os.path.join(model_load_base_path, f"{dataset2folder[name]}/test_fold_0.pth"),
                           map_location=torch.device('cpu'))
    quantifiers.append(Single_Metabolite_Model())
    quantifiers[-1].load_state_dict(state_dct)
    quantifiers[-1].eval()
model = QuantificationWrapper(quantifiers).to(device)
# load fold data and quantify
sample_ids_num = list(range(0, len(spectra)))


def ppm_to_idx(ppm):
    exact_idx = (ppm + 2) * 16314 / 14
    upper_idx = np.floor((ppm + 2.01) * 16314 / 14)
    lower_idx = np.ceil((ppm + 1.99) * 16314 / 14)
    return int(lower_idx), int(upper_idx)


def spectrum2ppm(spectra, step_size=0.01):
    LOWER_PPM_BOUND, STEP, UPPER_PPM_BOUND = -2.0, step_size, 12.0
    ppm_spectra = np.zeros((spectra.shape[0], int((UPPER_PPM_BOUND - LOWER_PPM_BOUND + STEP) / STEP)))
    for ppm in np.arange(LOWER_PPM_BOUND / STEP, (UPPER_PPM_BOUND + STEP) / STEP, 1):
        ppm *= STEP
        lower_idx, upper_idx = ppm_to_idx(ppm)
        if lower_idx < 0:
            lower_idx = 0
        if upper_idx > 16313:
            upper_idx = 16313
        idx_range = range(lower_idx, upper_idx + 1)
        ppm_spectra[:, int((ppm - LOWER_PPM_BOUND) / STEP)] = np.sum(spectra[:, idx_range], axis=1)
    return ppm_spectra


spectra = np.reshape(spectra, (len(spectra), 16314))
spectra = spectrum2ppm(spectra, step_size=0.01)
input = torch.from_numpy(spectra[sample_ids_num, :]).float().to(device)
result = model(input).detach().cpu().numpy()
pred_quant[sample_ids_num, :] = result
print("Metabolite levels quantified")



d = {'duration': pd.to_numeric(survival), 'event': events}




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

for i, metabolite in enumerate(list_of_targeted_metabolites):
    d[metabolite] = pd.to_numeric(pred_quant[:,i])
df = pd.DataFrame.from_dict(d, orient='index')
df = df.transpose()
for column in df:
    df[column] = pd.to_numeric(df[column],errors = 'coerce')

pathway_info=[]
with open(ultimate_path + "/run/result.csv") as f:
    for index, lines in enumerate(f):
        if index == 0:
            continue

        else:
            tokens = lines.split(",")
            pathway_info.append(tokens[2:])

pathway_info = np.reshape(pathway_info,(37,len(pathway_info[0])))
pathway_info = pathway_info.astype(np.int64)
print("Pathway information matrix loaded")
survival= np.reshape(survival,(384,1))
events= np.reshape(events,(384,1))
grade = np.reshape(grade,(384,1))
with open('pred_quant.pickle', 'wb') as handle:
    pickle.dump(pred_quant, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('grade.pickle', 'wb') as handle:
    pickle.dump(grade, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('events.pickle', 'wb') as handle:
    pickle.dump(events, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('survival.pickle', 'wb') as handle:
    pickle.dump(survival, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('pathway_info.pickle', 'wb') as handle:
    pickle.dump(pathway_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(ultimate_path + '/reproduce/reproduction_scripts/tests/scripts/pred_quant.pickle', 'wb') as handle:
    pickle.dump(pred_quant, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(ultimate_path + '/reproduce/reproduction_scripts/tests/scripts/grade.pickle', 'wb') as handle:
    pickle.dump(grade, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(ultimate_path + '/reproduce/reproduction_scripts/tests/scripts/events.pickle', 'wb') as handle:
    pickle.dump(events, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(ultimate_path + '/reproduce/reproduction_scripts/tests/scripts/survival.pickle', 'wb') as handle:
    pickle.dump(survival, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(ultimate_path + '/reproduce/reproduction_scripts/tests/scripts/pathway_info.pickle', 'wb') as handle:
    pickle.dump(pathway_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Data saved")
