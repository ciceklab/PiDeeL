import random
import os
import sys
import pdb
import numpy as np 
import pandas as pd
from collections import Counter, defaultdict
from pprint import pprint

from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc)


"""
Measure classification performance using confusion matrix,
auroc, aupr, precision, recall, f1 score and accuracy
"""
def measure_performance(pred, pred_prob, label):
    cm = confusion_matrix(label, pred)
    #auroc = roc_auc_score(label, pred_prob[:,1])
    auroc = roc_auc_score(label, pred_prob)

    #p,r,t = precision_recall_curve(label, pred_prob[:,1])
    p,r,t = precision_recall_curve(label, pred_prob)

    aupr = auc(r,p)
    prec = precision_score(label, pred, average="binary")
    rec = recall_score(label, pred, average="binary")
    f1 = f1_score(label, pred, average="binary")
    acc = accuracy_score(label, pred)
    return cm, auroc, aupr, prec, rec, f1, acc

"""
A Utility function to extract provided indices from the lists provided in dictionary
"""
def index_dct(dct, indices):
    temp = {}
    for key in dct.keys():
        try:
            temp[key] = dct[key][indices]
        except:
            pdb.set_trace()
    return temp


"""
Stratified Grouped K Fold related functions
"""
# calculate distribution of class labels in a given dataset
def get_distribution(labels):
    label_distribution = Counter(labels)
    sum_labels = sum(label_distribution.values())
    return [f'{label_distribution[i] / sum_labels:.2%}' for i in range(np.max(labels) + 1)]

# stratified and grouped k fold cross validation
def stratified_group_k_fold(y, groups, k, seed=None):
    label_count = np.max(y) + 1
    label_counts_per_group = defaultdict(lambda: np.zeros(label_count))
    label_distribution = Counter()
    for label, group in zip(y, groups):
        label_counts_per_group[group][label] += 1
        label_distribution[label] += 1
    
    label_counts_per_fold = defaultdict(lambda: np.zeros(label_count))
    groups_per_fold = defaultdict(set)

    def eval_label_counts_per_fold(label_counts, fold):
        label_counts_per_fold[fold] += label_counts
        std_per_label = []
        for label in range(label_count):
            label_std = np.std([label_counts_per_fold[i][label] / label_distribution[label] for i in range(k)])
            std_per_label.append(label_std)
        label_counts_per_fold[fold] -= label_counts
        return np.mean(std_per_label)
    
    groups_and_label_counts = list(label_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_label_counts)

    for group, label_counts in sorted(groups_and_label_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_label_counts_per_fold(label_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        label_counts_per_fold[best_fold] += label_counts
        groups_per_fold[best_fold].add(group)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, group in enumerate(groups) if group in train_groups]
        test_indices = [i for i, group in enumerate(groups) if group in test_groups]

        yield train_indices, test_indices

# split dataset to folds and return each fold's indices
def split_data_to_kfold(dataset, grouping, strat, K=5, seed=1):
    grouping_var = dataset[grouping]
    stratification_var = dataset[strat]
    folds = []
    for _, test_idx in stratified_group_k_fold(stratification_var, grouping_var, K, seed):
        folds.append(test_idx)
    return folds


"""
Stratified Grouped K fold cross validation training, validation and test dataset generator function
"""
def generator(dataset, grouping, strat, K, seed):
    # create folds
    folds = split_data_to_kfold(dataset, grouping, strat, K, seed)
    # yield training, validation and test datasets
    for i in range(len(folds)):
        # extract test dataset
        test_indices = folds[i]
        test_dataset = index_dct(dataset, test_indices)
        # extract validation dataset
        vald_indices = folds[(i+1) % K]
        vald_dataset = index_dct(dataset, vald_indices)
        # extract training dataset
        temp = list(range(K))
        temp.remove(i)
        temp.remove((i+1) % K)
        train_indices = []
        for idx in temp:
            train_indices += folds[idx]
        train_dataset = index_dct(dataset, train_indices)
        # yield the datasets
        yield(train_dataset, vald_dataset, test_dataset)


"""
HRMAS NMR Spectrum preprocessing related functions
"""
# convert from  a given ppm value (i.e. from -2 to 12 ppm) to spectrum scale (i.e. from 0 to 16313)
def ppm_to_idx(ppm):
    exact_idx = (ppm + 2) * 16314 / 14
    upper_idx = np.floor((ppm + 2.01) * 16314 / 14)
    lower_idx = np.ceil((ppm + 1.99) * 16314 / 14)
    return int(lower_idx), int(upper_idx)

# conversion between HRMAS NMR spectrum from [0, 16313] scale to [-2 ppm, 12 ppm] scale.
def spectrum2ppm(spectra, step_size=0.01):
    LOWER_PPM_BOUND, STEP, UPPER_PPM_BOUND = -2.0, step_size, 12.0 
    ppm_spectra = np.zeros((spectra.shape[0], int((UPPER_PPM_BOUND - LOWER_PPM_BOUND + STEP) / STEP)))
    for ppm in np.arange(LOWER_PPM_BOUND/STEP, (UPPER_PPM_BOUND+STEP)/STEP, 1):
        ppm *= STEP
        lower_idx, upper_idx = ppm_to_idx(ppm)
        if lower_idx < 0:
            lower_idx = 0
        if upper_idx > 16313:
            upper_idx = 16313
        idx_range = range(lower_idx, upper_idx+1)
        ppm_spectra[:, int((ppm - LOWER_PPM_BOUND) / STEP)] = np.sum(spectra[:, idx_range], axis=1)
    return ppm_spectra