import sys
import pickle

sys.path.insert(1, "../../")
sys.path.insert(1, "../../../")
sys.path.insert(1, "../../../../")
sys.path.insert(1, "../../../../../")
sys.path.insert(1, "../../../../../../")

import os
from hyper_config import *

with open("../../../pred_quant.pickle", "rb") as f:
    pred_quant = pickle.load(f)
with open("../../../grade.pickle", "rb") as f:
    grade = pickle.load(f)
with open("../../../events.pickle", "rb") as f:
    events = pickle.load(f)
with open("../../../survival.pickle", "rb") as f:
    survival = pickle.load(f)
with open("../../../pathway_info.pickle", "rb") as f:
    pathway_info = pickle.load(f)

from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss
from pycox.evaluation import EvalSurv
import torchtuples as tt
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from model import Net, initialize_weights, Net2
from config import learningrate, epochnum, task, SEEDS, patience, l2_lambda
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from model_utils import summary, ClassificationDataset, EarlyStopping
from utils import generator, index_dct, measure_performance

metric_names = ["auroc", "aupr", "precision", "recall", "f1", "acc"]
metrics = {}
for name in metric_names:
    metrics[name] = []
cols_stand = ['2-hydroxyglutarate', '3-hydroxybutyrate', 'Acetate', \
              'Alanine', 'Allocystathionine', 'Arginine', 'Ascorbate', \
              'Aspartate', 'Betaine', 'Choline', 'Creatine', \
              'Ethanolamine', 'GABA', 'Glutamate', 'Glutamine', 'Glutahionine (GSH)', \
              'Glycerophosphocholine', 'Glycine', 'hypo-Taurine', \
              'Isoleucine', 'Lactate', 'Leucine', 'Lysine', 'Methionine', \
              'myo-Inositol', 'NAL', 'NAA', 'o-Acetylcholine', 'Ornithine', \
              'Phosphocholine', 'Phosphocreatine', 'Proline', \
              'scyllo-Inositol', 'Serine', 'Taurine', 'Threonine', \
              'Valine']
device = torch.device(f"cuda:0")

c_indices = []
for SEED in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    result = list(kf.split(events))
    for i in range(len(result)):
        print(f"seed : {SEED}, fold : {i}")
        X_train = pred_quant[result[i][0]]
        X_test = pred_quant[result[i][1]]
        labels = np.hstack((events, survival, grade))
        y_train = labels[result[i][0]]
        y_test = labels[result[i][1]]
        X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_cv = scaler.transform(X_cv)
        X_test = scaler.transform(X_test)
        pathway_info = torch.as_tensor(pathway_info).to(device)
        model = Net(X_train.shape[1], 2, pathway_info).to(device)
        model.apply(initialize_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=1e-5)
        early_stopper = EarlyStopping(patience=10)
        criterion = CoxPHLoss()

        pos_count = np.sum(labels[:, 2])
        neg_count = labels[:, 2].shape[0] - pos_count
        class_weights = torch.FloatTensor([pos_count / labels[:, 2].shape[0], \
                                           neg_count / labels[:, 2].shape[0]]).to(device)
        # loss_func = nn.CrossEntropyLoss(weight=class_weights)
        criterion2 = nn.BCEWithLogitsLoss(pos_weight=class_weights[0])

        train_losses = []
        vald_losses = []
        for epoch in range(1, epochnum + 1):

            # train on minibatches
            train_loss = 0
            features = torch.tensor(X_train).float().to(device)
            labels = torch.tensor(y_train[:, 0:2]).float().to(device)
            labels2 = torch.tensor(y_train[:, 2]).float().to(device)
            # reset gradients
            optimizer.zero_grad()
            # forward pass
            pred_probas = model(features)
            # calculate loss
            loss1 = criterion(pred_probas[:, 0], labels[:, 1], labels[:, 0])
            loss2 = criterion2(torch.sigmoid(pred_probas[:, 1]), labels2)
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # print(f"loss1: {loss1/10}, loss2: {loss2}, l2_norm: {l2_norm* l2_lambda}")
            loss = loss1 / 10 + loss2 + (l2_norm * l2_lambda)
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
                labels_cv = torch.tensor(y_cv[:, 0:2]).float().to(device)
                labels2_cv = torch.tensor(y_cv[:, 2]).float().to(device)
                # forward pass
                pred_probas = model(features_cv)
                # calculate loss
                loss1 = criterion(pred_probas[:, 0], labels_cv[:, 1], labels_cv[:, 0])
                loss2 = criterion2(torch.sigmoid(pred_probas[:, 1]), labels2_cv)
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

                loss = loss1 / 10 + loss2 + (l2_norm * l2_lambda)
                vald_loss += loss.item()
            vald_losses.append(vald_loss)

            # report losses
            #print(f"Epoch {epoch}, Train loss: {train_losses[-1]}, Vald loss: {vald_losses[-1]}")

            # early stopping
            early_stopper(vald_losses[-1], model)
            if early_stopper.early_stop == True:
                #print(f"Early stopping at epoch {epoch}")
                break
        model.load_state_dict(torch.load('checkpoint.pt'))
        """
        epoch_count = range(1, len(train_losses) + 1)
        plt.plot(epoch_count, train_losses, 'r--')
        plt.plot(epoch_count, vald_losses, 'b-')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f"/home/gunkaynar/targeted_survival/tests/plots/test7/{task}/{SEED}_{i}.png")
        plt.clf()
        """
        with torch.no_grad():

            features_test = torch.tensor(X_test).float().to(device)
            labels_test = torch.tensor(y_test[:, 0:2]).float().to(device)
            labels2_test = torch.tensor(y_test[:, 2]).float()
            # forward pass
            pred_probas = model(features_test)
        pretrained_dict = model.state_dict()

        net = Net2(X_train.shape[1], 1, pathway_info).to(device)
        model1 = CoxPH(net, tt.optim.Adam)

        pretrained_dict["fc2.weight"] = torch.reshape(pretrained_dict["fc2.weight"][0], (1, 138))
        pretrained_dict["fc2.bias"] = torch.reshape(pretrained_dict["fc2.bias"][0], (1,))
        model1.net.load_state_dict(pretrained_dict)

        _ = model1.compute_baseline_hazards(input=features, target=(labels[:, 1], labels[:, 0]))
        surv = model1.predict_surv_df(features_test)
        ev = EvalSurv(surv, y_test[:, 1], y_test[:, 0], censor_surv='km')
        c_indices.append(ev.concordance_td())

        test_pred_probas = torch.sigmoid(pred_probas)
        test_pred_probas = test_pred_probas.cpu().numpy()
        test_preds = [1 if i >= 0.5 else 0 for i in test_pred_probas[:, 1]]
        cm, auroc, aupr, prec, rec, f1, acc = measure_performance(test_preds, test_pred_probas[:, 1], y_test[:, 2])
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
        """

        #model.save_net(f"/home/gunkaynar/targeted_survival/models/{task}/{SEED}_{i}.pth")
        #torch.save(net.state_dict(), f"/home/gunkaynar/targeted_survival/models/{task}/{SEED}_{i}net.pth")
        """
        torch.save(model.state_dict(),
                   ultimate_path + f"/reproduce/reproduction_scripts/tests/models/test7/{task}/{SEED}_{i}net.pth")

with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/c_index.txt", "w") as f:
    for c_index in c_indices:
        f.write("%f\n" % c_index)
with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/auroc.txt", "w") as f:
    for auroc in metrics["auroc"]:
        f.write("%f\n" % auroc)
with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/aupr.txt", "w") as f:
    for aupr in metrics["aupr"]:
        f.write("%f\n" % aupr)
with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/precision.txt", "w") as f:
    for precision in metrics["precision"]:
        f.write("%f\n" % precision)
with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/recall.txt", "w") as f:
    for recall in metrics["recall"]:
        f.write("%f\n" % recall)
with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/f1.txt", "w") as f:
    for f1 in metrics["f1"]:
        f.write("%f\n" % f1)
with open(ultimate_path + f"/reproduce/reproduction_scripts/tests/logs/test5/{task}/acc.txt", "w") as f:
    for accuracy in metrics["acc"]:
        f.write("%f\n" % accuracy)