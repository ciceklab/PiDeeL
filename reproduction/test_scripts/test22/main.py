import sys
sys.path.insert(1, "../../")
from sklearn_pandas import DataFrameMapper
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import torchtuples as tt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from load_targeted_data import predicted_quant,ages,grade,pathway_info
from config import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from model_utils import summary, ClassificationDataset, EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from utils import generator,index_dct, measure_performance
metric_names = ["auroc", "aupr", "precision", "recall", "f1", "acc"]

metrics = {}
for name in metric_names:
    metrics[name] = []
    
n_estimators = [50, 150, 300, 400]
max_depth = [10, 15, 25, 30]
min_samples_split = [5, 10, 15]
min_samples_leaf = [2, 10, 20] 
criterion = ["gini", "entropy"]
parameter_space = dict(n_estimators = n_estimators, max_depth = max_depth,  
    min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, criterion=criterion)

import pdb
cols_stand =['2-hydroxyglutarate', '3-hydroxybutyrate', 'Acetate',\
       'Alanine', 'Allocystathionine', 'Arginine', 'Ascorbate',\
       'Aspartate', 'Betaine', 'Choline', 'Creatine',\
       'Ethanolamine', 'GABA', 'Glutamate', 'Glutamine', 'Glutahionine (GSH)',\
       'Glycerophosphocholine', 'Glycine', 'hypo-Taurine',\
       'Isoleucine', 'Lactate', 'Leucine', 'Lysine', 'Methionine',\
       'myo-Inositol', 'NAL', 'NAA', 'o-Acetylcholine', 'Ornithine',\
       'Phosphocholine', 'Phosphocreatine', 'Proline',\
       'scyllo-Inositol', 'Serine', 'Taurine', 'Threonine',\
       'Valine']
for SEED in SEEDS:
    kf = KFold(n_splits=5, shuffle= True,random_state=SEED)
    result = list(kf.split(grade))
    for i in range(len(result)):
        X_train = predicted_quant[result[i][0]]
        X_test = predicted_quant[result[i][1]]
        y_train = grade[result[i][0]]
        y_test = grade[result[i][1]]
        X_train, X_cv , y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_cv = scaler.transform(X_cv)
        X_test =  scaler.transform(X_test)



        rf = RandomForestClassifier(class_weight="balanced", verbose=False, random_state=SEED)
        gs = GridSearchCV(rf, parameter_space, verbose=0, refit=False, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_cv, np.ravel(y_cv))

        criterion = gs.best_params_["criterion"]
        max_depth = gs.best_params_["max_depth"]
        min_samples_leaf = gs.best_params_["min_samples_leaf"]
        min_samples_split = gs.best_params_["min_samples_split"]
        n_estimators = gs.best_params_["n_estimators"]
        model = RandomForestClassifier(class_weight="balanced", verbose=False, random_state=SEED, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)
        model.fit(X_train, np.ravel(y_train))





        # calculate
        test_pred_probas = model.predict_proba(X_test)[:,1]
        test_preds =  model.predict(X_test)


        cm, auroc, aupr, prec, rec, f1, acc = measure_performance(test_preds, test_pred_probas, y_test)
        print(f"Confusion Matrix: {cm}")
        print(f"AUC-ROC: {auroc}")
        print(f"AUC-PR: {aupr}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1-score: {f1}")
        print(f"Accuracy: {acc}")

        metrics["auroc"].append(auroc)
        metrics["aupr"].append(aupr)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)
        metrics["acc"].append(acc)

with open((f"../../test_logs/test22/auroc.txt"), "w") as f:
    for auroc in metrics["auroc"]:
        f.write("%f\n" % auroc)
with open((f"../../test_logs/test22/aupr.txt"), "w") as f:
    for aupr in metrics["aupr"]:
        f.write("%f\n" % aupr)
with open((f"../../test_logs/test22/precision.txt"), "w") as f:
    for precision in metrics["precision"]:
        f.write("%f\n" % precision)
with open((f"../../test_logs/test22/recall.txt"), "w") as f:
    for recall in metrics["recall"]:
        f.write("%f\n" % recall)
with open((f"../../test_logs/test22/f1.txt"), "w") as f:
    for f1 in metrics["f1"]:
        f.write("%f\n" % f1)
with open((f"../../test_logs/test22/acc.txt"), "w") as f:
    for accuracy in metrics["acc"]:
        f.write("%f\n" % accuracy)