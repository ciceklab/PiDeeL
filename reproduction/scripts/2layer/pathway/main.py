import sys
from pathlib import Path
import importlib.util

# Get script directory paths
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

# First add local paths for local config and modules
sys.path.insert(0, str(SCRIPT_DIR))  # For local config.py and model.py
sys.path.insert(1, str(SCRIPT_DIR.parent.parent))  # scripts/ for load_targeted_data.py

# Import local config (hyperparameters)
from config import learningrate, epochnum, task, SEEDS, patience, l2_lambda
from model import Net, initialize_weights

# Load repo-level config using importlib (avoids sys.path conflicts)
config_path = REPO_ROOT / "config.py"
spec = importlib.util.spec_from_file_location("repo_config", config_path)
repo_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repo_config)

from sklearn_pandas import DataFrameMapper
from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss
from pycox.evaluation import EvalSurv
import torchtuples as tt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from load_targeted_data import predicted_quant, events, survival, pathway_info
import pdb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from model_utils import summary, ClassificationDataset, EarlyStopping

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
device = repo_config.DEVICE

c_indices = []
for SEED in SEEDS:
    kf = KFold(n_splits=5, shuffle= True,random_state=SEED)
    result = list(kf.split(events))
    for i in range(len(result)):
        X_train = predicted_quant[result[i][0]]
        X_test = predicted_quant[result[i][1]]
        labels = np.hstack((events,survival))
        y_train = labels[result[i][0]]
        y_test = labels[result[i][1]]
        X_train, X_cv , y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_cv = scaler.transform(X_cv)
        X_test =  scaler.transform(X_test)
        pathway_info = torch.as_tensor(pathway_info).to(device)
        model = Net(X_train.shape[1],1,pathway_info).to(device)
        model.apply(initialize_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learningrate,weight_decay=1e-5)
        early_stopper = EarlyStopping(patience=patience)
        criterion = CoxPHLoss()
        train_losses = []
        vald_losses = []
        for epoch in range(1, epochnum+1):

            # train on minibatches
            train_loss = 0
            features = torch.tensor(X_train).float().to(device)
            labels = torch.tensor(y_train).float().to(device)
            # reset gradients
            optimizer.zero_grad()
            # forward pass
            pred_probas = model(features)
            # calculate loss
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

            loss = criterion(pred_probas, labels[:,1], labels[:,0]) + l2_norm * l2_lambda
            train_loss += loss.item()
            # backward pass
            loss.backward()
            # update model
            optimizer.step()
            train_losses.append(train_loss)

            # validation
            vald_loss = 0
            with torch.no_grad():
                features_cv = torch.tensor(X_cv).float().to(device)
                labels_cv = torch.tensor(y_cv).float().to(device)
                # forward pass
                pred_probas = model(features_cv)
                # calculate loss 
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

                loss = criterion(pred_probas, labels_cv[:,1], labels_cv[:,0])+ l2_norm * l2_lambda
                vald_loss += loss.item()
            vald_losses.append(vald_loss)

            # report losses
            print(f"Epoch {epoch}, Train loss: {train_losses[-1]}, Vald loss: {vald_losses[-1]}")
            
            # early stopping
            early_stopper(vald_losses[-1],model)
            if early_stopper.early_stop == True:
                print(f"Early stopping at epoch {epoch}")
                break
        model.load_state_dict(torch.load('checkpoint.pt'))
        epoch_count = range(1, len(train_losses) + 1)
        plt.plot(epoch_count, train_losses, 'r--')
        plt.plot(epoch_count, vald_losses, 'b-')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(repo_config.get_plot_path(task, f"{SEED}_{i}.png"))
        plt.clf()
        features_test = torch.tensor(X_test).float().to(device)
        labels_test = torch.tensor(y_test).float().to(device)
        risk_scores = model(features_test)
        event_tests = [True if i == 1 else False for i in y_test[:,0]]

        # Alternative C-index calculation method using concordance_index_ipcw - kept for reference
        """
        labels_numpy = labels.cpu().numpy()
        labels_numpy_event = [True if i == 1 else False for i in labels_numpy[:,0]]
        labels_numpy_time = labels_numpy[:,1]
        labels_numpy = np.array([labels_numpy_event,labels_numpy_time]).T

        labels_test_numpy = labels_test.cpu().numpy()
        labels_test_numpy_event = [True if i == 1 else False for i in labels_test_numpy[:,0]]
        labels_test_numpy_time = labels_test_numpy[:,1]
        labels_test_numpy = np.array([labels_test_numpy_event,labels_test_numpy_time]).T
        #structured array for c-index calculation
        labels_structured = np.zeros(labels_numpy.shape[0], dtype=[('event', bool), ('time', float)])
        labels_structured['event'] = labels_numpy[:,0]
        labels_structured['time'] = labels_numpy[:,1]
        labels_test_structured = np.zeros(labels_test_numpy.shape[0], dtype=[('event', bool), ('time', float)])
        labels_test_structured['event'] = labels_test_numpy[:,0]
        labels_test_structured['time'] = labels_test_numpy[:,1]
        cindex = concordance_index_ipcw(labels_structured, labels_test_structured, risk_scores.cpu().detach().numpy()[:,0])
        #cindex = concordance_index_censored(event_tests, y_test[:,1], risk_scores.cpu().detach().numpy()[:,0])
        c_indices.append(cindex[0])
        """

        
        model1 = CoxPH(model, tt.optim.Adam)
        total_features = torch.cat((features,features_cv),dim=0)
        total_labels = torch.cat((labels,labels_cv),dim=0)

        _ = model1.compute_baseline_hazards(input=features,target=(labels[:,1],labels[:,0]))
        surv = model1.predict_surv_df(features_test)
        ev = EvalSurv(surv,  y_test[:,1], y_test[:,0], censor_surv='km')
        c_indices.append(ev.concordance_td())
        torch.save(model.state_dict(), repo_config.get_model_path(task, f"{SEED}_{i}net.pth"))

with open(repo_config.get_log_path(task, "c_indices.txt"), "w") as f:
    for c_index in c_indices:
        f.write("%f\n" % c_index)



