from sksurv.ensemble import RandomSurvivalForest
import numpy as np
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

import pickle
sys.path.insert(1, "../../")
sys.path.insert(1, "../../../")
sys.path.insert(1, "../../../../")
import os
from hyper_config import *
with open("../../pred_quant.pickle", "rb") as f:
    pred_quant = pickle.load(f)
with open("../../events.pickle", "rb") as f:
    events = pickle.load(f)
with open("../../survival.pickle", "rb") as f:
    survival = pickle.load(f)

task = f"rf"
c_indices = []

n_estimators = [20, 50, 100, 150, 300, 400]
max_depth = [None, 10, 15, 25, 30]
min_samples_split = [2, 5, 10, 15]
min_samples_leaf = [2, 5, 10, 20]
max_features = ["auto", "log2", None]

hyperparameter_space = dict(n_estimators=n_estimators, max_depth=max_depth, \
                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, \
                            max_features=max_features)
SEEDS = [6, 35, 81]
for SEED in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    result = list(kf.split(events))
    for i in range(len(result)):
        print(f"seed : {SEED}, fold : {i}")
        X_train = pred_quant[result[i][0]]
        X_test = pred_quant[result[i][1]]
        labels = np.hstack((events, survival))
        y_train = labels[result[i][0]]
        y_test = labels[result[i][1]]
        X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_cv = scaler.transform(X_cv)
        X_test = scaler.transform(X_test)
        train_labels = np.array([(True, y) if x == 1 else (False, y) \
                                 for x, y in zip(y_train[:, 0], y_train[:, 1])], \
                                dtype=[('event', np.bool_), ('surv', np.int32)])
        vald_labels = np.array([(True, y) if x == 1 else (False, y) \
                                for x, y in zip(y_cv[:, 0], y_cv[:, 1])], \
                               dtype=[('event', np.bool_), ('surv', np.int32)])
        test_labels = np.array([(True, y) if x == 1 else (False, y) \
                                for x, y in zip(y_test[:, 0], y_test[:, 1])], \
                               dtype=[('event', np.bool_), ('surv', np.int32)])

        grid_search_features = np.concatenate((X_train, X_cv), axis=0)
        grid_search_survival = np.concatenate((y_train[:, 1], y_cv[:, 1]), axis=0)
        grid_search_censoring = np.concatenate((y_train[:, 0], y_cv[:, 0]), axis=0)
        grid_search_labels = np.array([(True, y) if x == 1 else (False, y) \
                                       for x, y in zip(grid_search_censoring, grid_search_survival)], \
                                      dtype=[('event', np.bool_), ('surv', np.int32)])
        last_train_data_idx = X_train.shape[0]
        train_vald_split_indices = [(list(range(last_train_data_idx)), \
                                     list(range(last_train_data_idx, grid_search_features.shape[0])))]

        model = RandomSurvivalForest(verbose=0, random_state=SEED)
        gs = GridSearchCV(model, hyperparameter_space, cv=train_vald_split_indices, verbose=0, refit=True, n_jobs=-1)
        gs.fit(grid_search_features, grid_search_labels)
        model = gs.best_estimator_
        c_indices.append(model.score(X_test, test_labels))

log_path = os.path.join(ultimate_path, "reproduce/reproduction_scripts/logs/")
with open(log_path + f"baseline/{task}/c_indices.txt", "w") as f:
    for c_index in c_indices:
        f.write("%f\n" % c_index)
